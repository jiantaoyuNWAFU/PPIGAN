import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import amino_acids
from dataset import MyDataset
from Discriminator import Dis
from Generator import Gen
from torch_utils import select_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, train_ratio: float = 0.8, seed: Optional[int] = None):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    print(f"[Info] dataset split -> train: {train_size}, test: {test_size}")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )


def calculate_protein_degree(tsv_path: str):
    degree_dict = defaultdict(int)

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            p1, p2, label = parts
            if label == "1":
                degree_dict[p1] += 1
                degree_dict[p2] += 1

    return degree_dict


def calculate_protein_degree_from_dataset(dataset):
    degree_dict = defaultdict(int)

    if (
        isinstance(dataset, torch.utils.data.Subset)
        and hasattr(dataset.dataset, "y_train")
        and hasattr(dataset.dataset, "m_h")
    ):
        base = dataset.dataset

        for idx in dataset.indices:
            y = base.y_train[idx]
            y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

            if y_arr.shape[-1] >= 2 and int(y_arr[1]) == 1:
                pid1, pid2 = base.m_h[idx][0], base.m_h[idx][1]
                degree_dict[pid1] += 1
                degree_dict[pid2] += 1

        return degree_dict

    for sample in dataset:
        if len(sample) < 5:
            continue

        y = sample[2]
        pid1 = sample[3]
        pid2 = sample[4]
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        if y_arr.shape[-1] >= 2 and int(y_arr[1]) == 1:
            degree_dict[pid1] += 1
            degree_dict[pid2] += 1

    return degree_dict


def load_dataset_auto(data_path: str, seq_path: str):
    if isinstance(data_path, str) and data_path.endswith(".pth") and os.path.exists(data_path):
        print(f"[Info] loading dataset from pth: {data_path}")
        return torch.load(data_path)

    print(f"[Info] building dataset from raw files: {data_path}")
    return MyDataset(data_path, seq_path)


def prepare_output_dirs(save_dir: str):
    dirs = {
        "root": save_dir,
        "checkpoints": os.path.join(save_dir, "checkpoints"),
        "datasets": os.path.join(save_dir, "datasets"),
        "fake_samples": os.path.join(save_dir, "fake_samples"),
        "fasta_list": os.path.join(save_dir, "fasta_list"),
        "logs": os.path.join(save_dir, "logs"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def fake_distribution_to_embedding(fake_inputs: torch.Tensor, D: nn.Module) -> torch.Tensor:
    if fake_inputs.shape[-1] == D.embedding_layer.embedding_dim:
        return fake_inputs

    return torch.matmul(fake_inputs, D.embedding_layer.weight)


def compute_fake_real_stats(fake_inputs: torch.Tensor, left_protein: torch.Tensor, D: nn.Module):
    with torch.no_grad():
        real_embed = D.embedding_layer(left_protein.long())
        fake_embed = fake_distribution_to_embedding(fake_inputs, D)

        fake_flat = fake_embed.reshape(fake_embed.size(0), -1)
        real_flat = real_embed.reshape(real_embed.size(0), -1)

        cosine = torch.nn.functional.cosine_similarity(fake_flat, real_flat, dim=1)
        l2 = torch.norm(fake_flat - real_flat, p=2, dim=1)

        return {
            "cosine_mean": cosine.mean().item(),
            "cosine_std": cosine.std(unbiased=False).item() if cosine.numel() > 1 else 0.0,
            "l2_mean": l2.mean().item(),
            "l2_std": l2.std(unbiased=False).item() if l2.numel() > 1 else 0.0,
        }


def build_id_to_token():
    if not hasattr(amino_acids, "amino_acid"):
        raise RuntimeError("amino_acids.py does not contain amino_acid")

    aa_dict = amino_acids.amino_acid
    if not isinstance(aa_dict, dict):
        raise RuntimeError("amino_acids.amino_acid is not a dict")

    id_to_token = {v: str(k) for k, v in aa_dict.items() if isinstance(v, int)}
    if not id_to_token:
        raise RuntimeError("failed to build id_to_token from amino_acids.amino_acid")

    return id_to_token


def fake_tensor_to_ids_by_nearest_embedding(fake_inputs: torch.Tensor, D: nn.Module):
    with torch.no_grad():
        if not hasattr(D, "embedding_layer"):
            raise RuntimeError("Discriminator has no embedding_layer")

        embed_table = D.embedding_layer.weight.detach().cpu()[1:]
        x = fake_inputs.detach().cpu()

        if x.shape[-1] != D.embedding_layer.embedding_dim:
            emb_weight = D.embedding_layer.weight.detach().cpu()
            x = torch.matmul(x, emb_weight)

        batch_size, seq_len, emb_dim = x.shape
        x_flat = x.reshape(-1, emb_dim)

        dist = torch.cdist(x_flat, embed_table, p=2)
        ids = torch.argmin(dist, dim=1) + 1

    return ids.view(batch_size, seq_len)


def get_real_aa_freq_from_dataset(train_dataset, num_tokens: int = 21):
    counts = torch.zeros(num_tokens, dtype=torch.float)

    for sample in train_dataset:
        x1, x2 = sample[0], sample[1]

        for x in (x1, x2):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)

            ids = x.view(-1).long()
            ids = ids[(ids > 0) & (ids < num_tokens)]

            if ids.numel() > 0:
                counts += torch.bincount(ids, minlength=num_tokens).float()

    return counts[1:] / counts[1:].sum().clamp_min(1.0)


def get_fake_aa_freq(fake_inputs: torch.Tensor, D: nn.Module, num_tokens: int = 21):
    fake_ids = fake_tensor_to_ids_by_nearest_embedding(fake_inputs, D)
    fake_ids = fake_ids.view(-1).long()
    fake_ids = fake_ids[(fake_ids > 0) & (fake_ids < num_tokens)]

    counts = torch.bincount(fake_ids, minlength=num_tokens).float()
    freq = counts[1:] / counts[1:].sum().clamp_min(1.0)

    return freq.to(fake_inputs.device)


def ids_to_seq(ids, id_to_token, remove_zero: bool = True) -> str:
    seq = []

    for idx in ids:
        idx = int(idx)

        if remove_zero and idx == 0:
            continue

        token = id_to_token.get(idx, "X")

        if remove_zero and token == "0":
            continue

        seq.append(token)

    return "".join(seq)


def save_fake_samples(
    output_dirs: dict,
    epoch: int,
    step: int,
    chosen_protein: torch.Tensor,
    left_protein: torch.Tensor,
    fake_inputs: torch.Tensor,
    max_save: int = 8,
) -> None:
    n = min(max_save, fake_inputs.size(0))
    save_obj = {
        "epoch": epoch,
        "step": step,
        "chosen_protein": chosen_protein[:n].detach().cpu(),
        "left_protein": left_protein[:n].detach().cpu(),
        "fake_inputs": fake_inputs[:n].detach().cpu(),
    }

    torch.save(
        save_obj,
        os.path.join(output_dirs["fake_samples"], f"fake_epoch_{epoch}_step_{step}.pt"),
    )


def save_fake_fasta(
    output_dirs: dict,
    epoch: int,
    step: int,
    fake_inputs: torch.Tensor,
    D: nn.Module,
    max_save: int = 8,
) -> None:
    id_to_token = build_id_to_token()
    n = min(max_save, fake_inputs.size(0))
    fake_ids = fake_tensor_to_ids_by_nearest_embedding(fake_inputs[:n], D)

    uniq, cnt = torch.unique(fake_ids, return_counts=True)
    print("[Debug] unique decoded ids:", list(zip(uniq.tolist(), cnt.tolist())))

    fasta_path = os.path.join(output_dirs["fasta_list"], f"fake_epoch_{epoch}_step_{step}.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i in range(fake_ids.size(0)):
            seq = ids_to_seq(fake_ids[i].tolist(), id_to_token, remove_zero=True)
            if len(seq) == 0:
                seq = "X"

            f.write(f">fake_epoch_{epoch}_step_{step}_sample{i}\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j:j + 60] + "\n")

    print(f"[Saved FASTA] {fasta_path}")


def append_fake_stats(output_dirs: dict, epoch: int, step: int, stats: dict) -> None:
    log_path = os.path.join(output_dirs["logs"], "fake_similarity_log.txt")
    with open(log_path, "a+", encoding="utf-8") as f:
        f.write(
            f"Epoch {epoch}, Step {step}, "
            f"cosine_mean={stats['cosine_mean']:.6f}, "
            f"cosine_std={stats['cosine_std']:.6f}, "
            f"l2_mean={stats['l2_mean']:.6f}, "
            f"l2_std={stats['l2_std']:.6f}\n"
        )


def append_epoch_fake_stats(output_dirs: dict, epoch: int, epoch_stats: dict) -> None:
    if len(epoch_stats["cosine_mean"]) == 0:
        return

    cosine_mean = float(np.mean(epoch_stats["cosine_mean"]))
    cosine_std = float(np.mean(epoch_stats["cosine_std"]))
    l2_mean = float(np.mean(epoch_stats["l2_mean"]))
    l2_std = float(np.mean(epoch_stats["l2_std"]))

    log_path = os.path.join(output_dirs["logs"], "fake_similarity_epoch_log.txt")
    with open(log_path, "a+", encoding="utf-8") as f:
        f.write(
            f"Epoch {epoch}, "
            f"cosine_mean={cosine_mean:.6f}, "
            f"cosine_std={cosine_std:.6f}, "
            f"l2_mean={l2_mean:.6f}, "
            f"l2_std={l2_std:.6f}\n"
        )

    print(
        f"[Epoch FakeStats] Epoch {epoch}, "
        f"cosine_mean={cosine_mean:.4f}, "
        f"l2_mean={l2_mean:.4f}"
    )


def build_loaders(args, output_dirs: dict):
    if args.train_dataset == "" and args.test_dataset == "":
        dataset = MyDataset(args.interaction_data, args.sequence_data)
        train_dataset, test_dataset = split_dataset(dataset, seed=args.seed)

        torch.save(train_dataset, os.path.join(output_dirs["datasets"], "train_dataset.pth"))
        torch.save(test_dataset, os.path.join(output_dirs["datasets"], "test_dataset.pth"))
        print("[Info] train/test split saved.")
    else:
        train_dataset = load_dataset_auto(args.train_dataset, args.sequence_data)
        test_dataset = load_dataset_auto(args.test_dataset, args.sequence_data)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return train_dataset, train_loader, test_loader


def init_models(args):
    D = Dis(args)
    G = Gen(args)

    G.weight_init(mean=0.0, std=0.02)

    if args.d_pth == "":
        D.weight_init(mean=0.0, std=0.02)
    else:
        print(f"[Info] loading discriminator from: {args.d_pth}")
        state_dict = torch.load(args.d_pth, map_location=args.device)
        D.load_state_dict(state_dict)

    G.to(args.device)
    D.to(args.device)

    return D, G


def select_positive_pairs(x1, x2, y, pid1, pid2, protein_degrees, device):
    positive_mask = (y == torch.tensor([0, 1], device=device)).all(dim=1)

    x1_pos = x1[positive_mask]
    x2_pos = x2[positive_mask]
    pid1_pos = [pid1[idx] for idx, flag in enumerate(positive_mask.tolist()) if flag]
    pid2_pos = [pid2[idx] for idx, flag in enumerate(positive_mask.tolist()) if flag]

    if x1_pos.size(0) == 0:
        return None, None

    chosen_list = []
    left_list = []

    for idx in range(len(pid1_pos)):
        d1 = protein_degrees.get(pid1_pos[idx], 0)
        d2 = protein_degrees.get(pid2_pos[idx], 0)

        if d1 > d2:
            chosen_list.append(x1_pos[idx])
            left_list.append(x2_pos[idx])
        elif d2 > d1:
            chosen_list.append(x2_pos[idx])
            left_list.append(x1_pos[idx])
        elif np.random.rand() < 0.5:
            chosen_list.append(x1_pos[idx])
            left_list.append(x2_pos[idx])
        else:
            chosen_list.append(x2_pos[idx])
            left_list.append(x1_pos[idx])

    return torch.stack(chosen_list), torch.stack(left_list)


def train(args):
    output_dirs = prepare_output_dirs(args.save_dir)

    if args.seed is not None:
        seed_everything(args.seed)

    train_dataset, train_loader, test_loader = build_loaders(args, output_dirs)

    real_aa_freq = get_real_aa_freq_from_dataset(train_dataset).to(args.device)
    print("[Info] real amino acid frequency loaded.")
    print("[Info] real_aa_freq =", real_aa_freq.detach().cpu().numpy())

    D, G = init_models(args)
    print("[Info] D.embedding_layer is trainable.")

    criterion = nn.CrossEntropyLoss()
    criterion_gen = nn.CrossEntropyLoss()

    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, D.parameters()),
        lr=args.d_lr,
        betas=(0.9, 0.999),
        eps=1e-6,
    )
    optimizer_G = torch.optim.Adam(
        G.parameters(),
        lr=args.g_lr,
        betas=(0.9, 0.999),
        eps=1e-6,
    )

    best_acc = 0.0
    best_epoch = 0
    global_step = 0

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    try:
        protein_degrees = calculate_protein_degree_from_dataset(train_dataset)
        print(f"[Info] protein degree dict (train-only) loaded, size = {len(protein_degrees)}")
    except Exception as e:
        print(f"[Warn] failed to compute degree from train_dataset: {e}")
        protein_degrees = calculate_protein_degree(args.interaction_data)
        print(f"[Info] protein degree dict (from full file) loaded, size = {len(protein_degrees)}")

    for epoch in range(args.epoch):
        D.train()
        G.train()

        epoch_fake_stats = {
            "cosine_mean": [],
            "cosine_std": [],
            "l2_mean": [],
            "l2_std": [],
        }

        for i, (x1, x2, y, pid1, pid2) in enumerate(train_loader):
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)
            y_cls = torch.argmax(y, dim=1).long()

            optimizer_D.zero_grad()

            real_outputs = D(x1, x2, None, return_logits=True)
            real_loss = criterion(real_outputs, y_cls)

            if args.is_only_dis:
                real_loss.backward()
                optimizer_D.step()
                print(
                    f"Epoch [{epoch + 1}/{args.epoch}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"D_loss: {real_loss.item():.4f}"
                )
                continue

            chosen_protein, left_protein = select_positive_pairs(
                x1=x1,
                x2=x2,
                y=y,
                pid1=pid1,
                pid2=pid2,
                protein_degrees=protein_degrees,
                device=args.device,
            )

            if chosen_protein is None:
                real_loss.backward()
                optimizer_D.step()
                global_step += 1
                continue

            batch_pos_size = chosen_protein.size(0)
            fake_labels = torch.zeros(batch_pos_size, dtype=torch.long, device=args.device)
            real_labels = torch.ones(batch_pos_size, dtype=torch.long, device=args.device)

            z = args.noise_scale * torch.randn(
                (batch_pos_size, 1500, args.em_dim),
                device=args.device,
            )

            with torch.no_grad():
                fake_inputs = G(left_protein, z)

            fake_outputs = D(chosen_protein, fake_inputs, True, return_logits=True)
            fake_loss = criterion(fake_outputs, fake_labels)

            stats = compute_fake_real_stats(fake_inputs, left_protein, D)
            for key in epoch_fake_stats:
                epoch_fake_stats[key].append(stats[key])

            print(
                f"[FakeStats] Epoch {epoch + 1}, Step {i + 1}, "
                f"cosine_mean={stats['cosine_mean']:.4f}, "
                f"l2_mean={stats['l2_mean']:.4f}"
            )
            append_fake_stats(output_dirs, epoch + 1, i + 1, stats)

            if i == 0:
                save_fake_samples(
                    output_dirs=output_dirs,
                    epoch=epoch + 1,
                    step=i + 1,
                    chosen_protein=chosen_protein,
                    left_protein=left_protein,
                    fake_inputs=fake_inputs,
                    max_save=args.max_save_fake,
                )
                save_fake_fasta(
                    output_dirs=output_dirs,
                    epoch=epoch + 1,
                    step=i + 1,
                    fake_inputs=fake_inputs,
                    D=D,
                    max_save=args.max_save_fake,
                )

            d_loss = args.beta_real_loss * real_loss + args.beta_fake_loss * fake_loss
            d_loss.backward()
            optimizer_D.step()
            global_step += 1

            last_g_adv_loss = 0.0
            last_g_freq_loss = 0.0
            last_g_align_loss = 0.0
            last_g_loss = 0.0

            for _ in range(args.g_steps):
                optimizer_G.zero_grad()

                for param in D.parameters():
                    param.requires_grad = True
                for param in G.parameters():
                    param.requires_grad = True

                z_g = args.noise_scale * torch.randn(
                    (left_protein.size(0), 1500, args.em_dim),
                    device=args.device,
                )
                fake_inputs_g = G(left_protein, z_g)
                fake_outputs_g = D(chosen_protein, fake_inputs_g, True, return_logits=True)

                g_adv_loss = criterion_gen(fake_outputs_g, real_labels)
                fake_aa_freq = get_fake_aa_freq(fake_inputs_g, D)
                g_freq_loss = torch.mean((fake_aa_freq - real_aa_freq) ** 2)

                real_embed_g = D.embedding_layer(left_protein.long()).detach()
                emb_weight = D.embedding_layer.weight.detach()
                fake_embed_g = torch.matmul(fake_inputs_g, emb_weight)
                g_align_loss = torch.mean((fake_embed_g - real_embed_g) ** 2)

                g_loss = (
                    g_adv_loss
                    + args.lambda_freq * g_freq_loss
                    + args.lambda_align * g_align_loss
                )

                g_loss.backward()
                optimizer_G.step()

                for param in D.parameters():
                    param.requires_grad = True

                last_g_adv_loss = g_adv_loss.item()
                last_g_freq_loss = g_freq_loss.item()
                last_g_align_loss = g_align_loss.item()
                last_g_loss = g_loss.item()

            print(
                f"Epoch [{epoch + 1}/{args.epoch}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"G_adv: {last_g_adv_loss:.4f}, "
                f"G_freq: {last_g_freq_loss:.4f}, "
                f"G_align: {last_g_align_loss:.4f}, "
                f"G_loss: {last_g_loss:.4f}, "
                f"D_loss: {d_loss.item():.4f}"
            )

        append_epoch_fake_stats(output_dirs, epoch + 1, epoch_fake_stats)
        best_acc, best_epoch = evaluate_and_save(
            D=D,
            G=G,
            test_loader=test_loader,
            args=args,
            output_dirs=output_dirs,
            epoch=epoch,
            best_acc=best_acc,
            best_epoch=best_epoch,
        )


def evaluate_and_save(D, G, test_loader, args, output_dirs, epoch, best_acc, best_epoch):
    D.eval()

    with torch.no_grad():
        y_true = []
        y_pred = []

        for x1, x2, y, _, _ in test_loader:
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)

            outputs = D(x1, x2, None)
            outputs = outputs.cpu().numpy()[:, 1]
            y_true_batch = y.cpu().numpy()[:, 1]
            outputs = (outputs > 0.5).astype(int)

            y_true.extend(y_true_batch.tolist())
            y_pred.extend(outputs.tolist())

        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        if accuracy > best_acc:
            best_acc = accuracy
            best_epoch = epoch + 1
            save_checkpoint(D, os.path.join(output_dirs["checkpoints"], "D_best_acc.pth"))
            if not args.is_only_dis:
                save_checkpoint(G, os.path.join(output_dirs["checkpoints"], "G_best_acc.pth"))

            print(f"New best model saved! Best accuracy = {best_acc:.4f} at epoch {best_epoch}")

        if args.is_only_dis:
            save_checkpoint(D, os.path.join(output_dirs["checkpoints"], f"D_epoch_{epoch + 1}.pth"))
        else:
            save_checkpoint(
                D,
                os.path.join(output_dirs["checkpoints"], f"D_epoch_{epoch + 1}_{accuracy:.4f}.pth"),
            )
            save_checkpoint(
                G,
                os.path.join(output_dirs["checkpoints"], f"G_epoch_{epoch + 1}_{accuracy:.4f}.pth"),
            )
            print("Current epoch model saved!")

        print("混淆矩阵:")
        print(cm)
        print("准确率:", accuracy)
        print("精确率:", precision)
        print("特异性:", specificity)
        print("召回率:", recall)
        print("F1值:", f1)
        print("MCC:", mcc)
        print(f"Best accuracy so far: {best_acc:.4f}, Best epoch: {best_epoch}")
        print("===============================================")

        metric_log_path = os.path.join(output_dirs["logs"], f"log_{args.beta_real_loss}_{args.beta_fake_loss}.txt")
        with open(metric_log_path, "a+", encoding="utf-8") as f:
            f.write(
                f"Epoch [{epoch + 1}/{args.epoch}]\n"
                f"cm:{cm}\n"
                f"Accuracy: {accuracy}, Precision: {precision}, "
                f"Specificity: {specificity}, Recall: {recall}, F1: {f1}, MCC:{mcc}\n"
                f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}\n"
                "===============================================\n"
            )

    return best_acc, best_epoch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--interaction_data", default="./data/yeast core dataset from PIPR/protein.actions.tsv", type=str)
    parser.add_argument("--sequence_data", default="./data/yeast core dataset from PIPR/protein.dictionary.tsv", type=str)
    parser.add_argument("--train_dataset", default="", type=str)
    parser.add_argument("--test_dataset", default="", type=str)
    parser.add_argument("--d_pth", default="", type=str)
    parser.add_argument("--save_dir", default="./Result/PPIGAN_default", type=str)

    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--is_only_dis", action="store_true")

    parser.add_argument("--em_dim", default=15, type=int)
    parser.add_argument("--hidden_dim", default=25, type=int)
    parser.add_argument("--conv_num", default=10, type=int)
    parser.add_argument("--node_num", default=256, type=int)

    parser.add_argument("--sp_drop", default=0.005, type=float)
    parser.add_argument("--con_drop", default=0.05, type=float)
    parser.add_argument("--fn_drop_1", default=0.2, type=float)
    parser.add_argument("--fn_drop_2", default=0.1, type=float)

    parser.add_argument("--kernel_rate_1", default=0.16, type=float)
    parser.add_argument("--strides_rate_1", default=0.15, type=float)
    parser.add_argument("--kernel_rate_2", default=0.14, type=float)
    parser.add_argument("--strides_rate_2", default=0.25, type=float)

    parser.add_argument("--filter_num_1", default=150, type=int)
    parser.add_argument("--filter_num_2", default=175, type=int)

    parser.add_argument("--d_lr", default=1e-4, type=float)
    parser.add_argument("--g_lr", default=5e-5, type=float)
    parser.add_argument("--g_steps", default=1, type=int)

    parser.add_argument("--beta_real_loss", default=1.0, type=float)
    parser.add_argument("--beta_fake_loss", default=0.005, type=float)
    parser.add_argument("--lambda_freq", default=10.0, type=float)
    parser.add_argument("--lambda_align", default=0.1, type=float)
    parser.add_argument("--noise_scale", default=0.1, type=float)
    parser.add_argument("--max_save_fake", default=8, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cuda and torch.cuda.is_available():
        args.device = select_device("cuda:0")
    else:
        args.device = select_device("cpu")

    print("[Info] args =", args)
    train(args)
