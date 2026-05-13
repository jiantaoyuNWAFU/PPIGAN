# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

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
    torch.backends.cudnn.benchmark = True


def prepare_output_dirs(save_dir: str) -> Dict[str, str]:
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


def get_dataset_labels(dataset) -> np.ndarray:
    labels = []

    if hasattr(dataset, "y_train"):
        for y in dataset.y_train:
            y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            labels.append(int(np.argmax(y_arr)))
        return np.asarray(labels)

    for i in range(len(dataset)):
        sample = dataset[i]
        y = sample[2]
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        labels.append(int(np.argmax(y_arr)))

    return np.asarray(labels)


def build_5fold_loaders(args, output_dirs: Dict[str, str]):
    dataset = MyDataset(args.interaction_data, args.sequence_data)
    labels = get_dataset_labels(dataset)

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_loaders = []

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
        test_dataset = torch.utils.data.Subset(dataset, test_idx.tolist())

        torch.save(
            train_dataset,
            os.path.join(output_dirs["datasets"], f"fold_{fold_id}_train_dataset.pth"),
        )
        torch.save(
            test_dataset,
            os.path.join(output_dirs["datasets"], f"fold_{fold_id}_test_dataset.pth"),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )

        print(
            f"[Fold {fold_id}] train: {len(train_dataset)}, "
            f"test: {len(test_dataset)}, "
            f"train_pos: {labels[train_idx].sum()}, "
            f"test_pos: {labels[test_idx].sum()}"
        )

        fold_loaders.append((fold_id, train_dataset, train_loader, test_loader))

    return fold_loaders


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


def select_s1_s2_by_degree(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y_cls: torch.Tensor,
    pid1,
    pid2,
    protein_degrees,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    pos_mask = (y_cls == 1)

    x1_pos = x1[pos_mask]
    x2_pos = x2[pos_mask]

    pid1_pos = [pid1[idx] for idx, flag in enumerate(pos_mask.tolist()) if flag]
    pid2_pos = [pid2[idx] for idx, flag in enumerate(pos_mask.tolist()) if flag]

    if x1_pos.size(0) == 0:
        return None, None

    s1_high_list = []
    s2_real_list = []

    for idx in range(len(pid1_pos)):
        d1 = protein_degrees.get(pid1_pos[idx], 0)
        d2 = protein_degrees.get(pid2_pos[idx], 0)

        if d1 > d2:
            s1_high_list.append(x1_pos[idx])
            s2_real_list.append(x2_pos[idx])
        elif d2 > d1:
            s1_high_list.append(x2_pos[idx])
            s2_real_list.append(x1_pos[idx])
        else:
            if np.random.rand() < 0.5:
                s1_high_list.append(x1_pos[idx])
                s2_real_list.append(x2_pos[idx])
            else:
                s1_high_list.append(x2_pos[idx])
                s2_real_list.append(x1_pos[idx])

    return torch.stack(s1_high_list), torch.stack(s2_real_list)


def build_id_to_token():
    if not hasattr(amino_acids, "amino_acid"):
        raise RuntimeError("amino_acids.py does not contain amino_acid")

    aa_dict = amino_acids.amino_acid
    id_to_token = {}

    for k, v in aa_dict.items():
        if isinstance(v, int):
            id_to_token[v] = str(k)

    if len(id_to_token) == 0:
        raise RuntimeError("failed to build id_to_token")

    return id_to_token


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


def get_real_aa_freq_from_dictionary_tsv(tsv_path: str, device: torch.device):
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}

    counts = torch.zeros(20, dtype=torch.float)

    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            seq = parts[1].strip()
            for aa in seq:
                if aa in aa_to_idx:
                    counts[aa_to_idx[aa]] += 1

    freq = counts / counts.sum().clamp_min(1.0)
    return freq.to(device)


def logits_to_soft_embedding(fake_logits: torch.Tensor, embedding_layer: nn.Embedding):
    fake_probs = torch.softmax(fake_logits, dim=-1)
    embed_table = embedding_layer.weight
    fake_embed = torch.matmul(fake_probs, embed_table)
    return fake_embed, fake_probs


def compute_fake_aa_freq_from_probs(fake_probs: torch.Tensor, condition_protein: torch.Tensor):
    valid_mask = (condition_protein != 0).unsqueeze(-1).float()
    masked_probs = fake_probs * valid_mask

    aa_sum = masked_probs.sum(dim=(0, 1))
    fake_aa_freq = aa_sum[1:] / aa_sum[1:].sum().clamp_min(1.0)

    return fake_aa_freq


def compute_fake_real_stats(fake_inputs: torch.Tensor, real_target_protein: torch.Tensor, D: nn.Module):
    with torch.no_grad():
        real_embed = D.embedding_layer(real_target_protein.long())

        fake_flat = fake_inputs.reshape(fake_inputs.size(0), -1)
        real_flat = real_embed.reshape(real_embed.size(0), -1)

        cosine = torch.nn.functional.cosine_similarity(fake_flat, real_flat, dim=1)
        l2 = torch.norm(fake_flat - real_flat, p=2, dim=1)

        return {
            "cosine_mean": cosine.mean().item(),
            "cosine_std": cosine.std(unbiased=False).item() if cosine.numel() > 1 else 0.0,
            "l2_mean": l2.mean().item(),
            "l2_std": l2.std(unbiased=False).item() if l2.numel() > 1 else 0.0,
        }


def save_fake_samples(
    output_dirs: Dict[str, str],
    epoch: int,
    step: int,
    s1_high: torch.Tensor,
    s2_real: torch.Tensor,
    fake_inputs: torch.Tensor,
    max_save: int = 8,
) -> None:
    n = min(max_save, fake_inputs.size(0))

    save_obj = {
        "epoch": epoch,
        "step": step,
        "s1_high_condition": s1_high[:n].detach().cpu(),
        "s2_real": s2_real[:n].detach().cpu(),
        "s2_fake_inputs": fake_inputs[:n].detach().cpu(),
    }

    torch.save(
        save_obj,
        os.path.join(output_dirs["fake_samples"], f"fake_epoch_{epoch}_step_{step}.pt"),
    )


def save_fake_fasta_from_logits(
    output_dirs: Dict[str, str],
    epoch: int,
    step: int,
    fake_logits: torch.Tensor,
    condition_protein: torch.Tensor,
    max_save: int = 8,
) -> None:
    id_to_token = build_id_to_token()

    n = min(max_save, fake_logits.size(0))
    probs = torch.softmax(fake_logits[:n].detach().cpu(), dim=-1)

    probs[:, :, 0] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    fake_ids = torch.multinomial(
        probs.reshape(-1, probs.size(-1)),
        num_samples=1,
    ).reshape(probs.size(0), probs.size(1))

    valid_mask = (condition_protein[:n].detach().cpu() != 0)

    fasta_path = os.path.join(
        output_dirs["fasta_list"],
        f"fake_epoch_{epoch}_step_{step}.fasta",
    )

    with open(fasta_path, "w", encoding="utf-8") as f:
        for i in range(fake_ids.size(0)):
            ids = fake_ids[i][valid_mask[i]].tolist()
            seq = ids_to_seq(ids, id_to_token, remove_zero=True)

            if len(seq) == 0:
                seq = "X"

            f.write(f">fake_S2prime_epoch_{epoch}_step_{step}_sample{i}\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j:j + 60] + "\n")

    print(f"[Saved FASTA] {fasta_path}")


def append_fake_stats(output_dirs: Dict[str, str], epoch: int, step: int, stats: dict) -> None:
    log_path = os.path.join(output_dirs["logs"], "fake_similarity_log.txt")

    with open(log_path, "a+", encoding="utf-8") as f:
        f.write(
            f"Epoch {epoch}, Step {step}, "
            f"cosine_mean={stats['cosine_mean']:.6f}, "
            f"cosine_std={stats['cosine_std']:.6f}, "
            f"l2_mean={stats['l2_mean']:.6f}, "
            f"l2_std={stats['l2_std']:.6f}\n"
        )


def append_epoch_fake_stats(output_dirs: Dict[str, str], epoch: int, epoch_stats: dict) -> None:
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


def freeze_discriminator_embedding(D: nn.Module):
    frozen_embedding_param_ids = set()

    if hasattr(D, "embedding_layer"):
        for param in D.embedding_layer.parameters():
            param.requires_grad = False
            frozen_embedding_param_ids.add(id(param))
        print("[Info] D.embedding_layer is frozen.")
    else:
        print("[Warn] D has no embedding_layer, skip freezing.")

    return frozen_embedding_param_ids


def restore_discriminator_trainable_state(D: nn.Module, frozen_embedding_param_ids: set):
    for param in D.parameters():
        if id(param) not in frozen_embedding_param_ids:
            param.requires_grad = True

    if hasattr(D, "embedding_layer"):
        for param in D.embedding_layer.parameters():
            param.requires_grad = False


def evaluate_and_save(
    D: nn.Module,
    G: nn.Module,
    test_loader,
    args,
    output_dirs: Dict[str, str],
    epoch: int,
    best_acc: float,
    best_epoch: int,
):
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
            outputs = (outputs > args.threshold).astype(int)

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

        if args.save_each_epoch:
            if args.is_only_dis:
                save_checkpoint(
                    D,
                    os.path.join(output_dirs["checkpoints"], f"D_epoch_{epoch + 1}.pth"),
                )
            else:
                save_checkpoint(
                    D,
                    os.path.join(output_dirs["checkpoints"], f"D_epoch_{epoch + 1}_{accuracy:.4f}.pth"),
                )
                save_checkpoint(
                    G,
                    os.path.join(output_dirs["checkpoints"], f"G_epoch_{epoch + 1}_{accuracy:.4f}.pth"),
                )

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

        metric_log_path = os.path.join(
            output_dirs["logs"],
            f"log_{args.beta_real_loss}_{args.beta_fake_loss}.txt",
        )

        with open(metric_log_path, "a+", encoding="utf-8") as f:
            f.write(
                f"Epoch [{epoch + 1}/{args.epoch}]\n"
                f"cm:{cm}\n"
                f"Accuracy: {accuracy}, Precision: {precision}, "
                f"Specificity: {specificity}, Recall: {recall}, F1: {f1}, MCC:{mcc}\n"
                f"Best accuracy: {best_acc:.4f}, Best epoch: {best_epoch}\n"
                "===============================================\n"
            )

    return best_acc, best_epoch, {
        "accuracy": accuracy,
        "precision": precision,
        "specificity": specificity,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }


def train_one_fold(args, fold_id, train_dataset, train_loader, test_loader, fold_save_dir):
    print(f"\n========== Start Fold {fold_id}/{args.n_splits} ==========")

    output_dirs = prepare_output_dirs(fold_save_dir)

    if args.seed is not None:
        seed_everything(args.seed + fold_id)

    real_aa_freq = get_real_aa_freq_from_dictionary_tsv(args.sequence_data, args.device)
    print("[Info] real amino acid frequency loaded.")
    print("[Info] real_aa_freq =", real_aa_freq.detach().cpu().numpy())

    D, G = init_models(args)
    frozen_embedding_param_ids = freeze_discriminator_embedding(D)

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
    best_metrics = None

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    protein_degrees = calculate_protein_degree_from_dataset(train_dataset)
    print(f"[Info] protein degree dict fold-{fold_id} train-only size = {len(protein_degrees)}")

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
                    f"[Fold {fold_id}] Epoch [{epoch + 1}/{args.epoch}], "
                    f"Step [{i + 1}/{len(train_loader)}], "
                    f"D_loss: {real_loss.item():.4f}"
                )
                continue

            s1_high, s2_real = select_s1_s2_by_degree(
                x1=x1,
                x2=x2,
                y_cls=y_cls,
                pid1=pid1,
                pid2=pid2,
                protein_degrees=protein_degrees,
            )

            if s1_high is None:
                real_loss.backward()
                optimizer_D.step()
                continue

            batch_pos_size = s1_high.size(0)

            fake_labels = torch.zeros(batch_pos_size, dtype=torch.long, device=args.device)
            real_labels = torch.ones(batch_pos_size, dtype=torch.long, device=args.device)

            z = args.noise_scale * torch.randn(
                (batch_pos_size, 1500, args.em_dim),
                device=args.device,
            )

            with torch.no_grad():
                fake_logits = G(s2_real, z)
                s2_fake_embed, s2_fake_probs = logits_to_soft_embedding(
                    fake_logits,
                    D.embedding_layer,
                )

            fake_outputs = D(s1_high, s2_fake_embed, None, return_logits=True)
            fake_loss = criterion(fake_outputs, fake_labels)

            stats = compute_fake_real_stats(s2_fake_embed, s2_real, D)
            for key in epoch_fake_stats:
                epoch_fake_stats[key].append(stats[key])

            append_fake_stats(output_dirs, epoch + 1, i + 1, stats)

            if i == 0 and ((epoch + 1) % args.save_fake_every == 0 or epoch == 0):
                save_fake_samples(
                    output_dirs=output_dirs,
                    epoch=epoch + 1,
                    step=i + 1,
                    s1_high=s1_high,
                    s2_real=s2_real,
                    fake_inputs=s2_fake_embed,
                    max_save=args.max_save_fake,
                )

                save_fake_fasta_from_logits(
                    output_dirs=output_dirs,
                    epoch=epoch + 1,
                    step=i + 1,
                    fake_logits=fake_logits,
                    condition_protein=s2_real,
                    max_save=args.max_save_fake,
                )

            d_loss = args.beta_real_loss * real_loss + args.beta_fake_loss * fake_loss
            d_loss.backward()
            optimizer_D.step()

            last_g_adv_loss = 0.0
            last_g_freq_loss = 0.0
            last_g_loss = 0.0

            for _ in range(args.g_steps):
                optimizer_G.zero_grad()

                for param in D.parameters():
                    param.requires_grad = False

                for param in G.parameters():
                    param.requires_grad = True

                z_g = args.noise_scale * torch.randn(
                    (s2_real.size(0), 1500, args.em_dim),
                    device=args.device,
                )

                fake_logits_g = G(s2_real, z_g)
                s2_fake_embed_g, s2_fake_probs_g = logits_to_soft_embedding(
                    fake_logits_g,
                    D.embedding_layer,
                )

                fake_outputs_g = D(
                    s1_high,
                    s2_fake_embed_g,
                    None,
                    return_logits=True,
                )

                g_adv_loss = criterion_gen(fake_outputs_g, real_labels)

                fake_aa_freq = compute_fake_aa_freq_from_probs(
                    fake_probs=s2_fake_probs_g,
                    condition_protein=s2_real,
                )

                g_freq_loss = torch.mean((fake_aa_freq - real_aa_freq) ** 2)

                lambda_freq_now = (
                    args.lambda_freq
                    if (epoch + 1) > args.freq_warmup_epochs
                    else 0.0
                )

                g_entropy_loss = torch.tensor(0.0, device=args.device)

                if args.lambda_entropy > 0:
                    entropy = -torch.sum(
                        s2_fake_probs_g[:, :, 1:]
                        * torch.log(s2_fake_probs_g[:, :, 1:] + 1e-8),
                        dim=-1,
                    )
                    valid_mask = (s2_real != 0).float()
                    g_entropy_loss = -torch.sum(entropy * valid_mask) / valid_mask.sum().clamp_min(1.0)

                g_loss = (
                    g_adv_loss
                    + lambda_freq_now * g_freq_loss
                    + args.lambda_entropy * g_entropy_loss
                )

                g_loss.backward()
                optimizer_G.step()

                restore_discriminator_trainable_state(D, frozen_embedding_param_ids)

                last_g_adv_loss = g_adv_loss.item()
                last_g_freq_loss = g_freq_loss.item()
                last_g_loss = g_loss.item()

            print(
                f"[Fold {fold_id}] Epoch [{epoch + 1}/{args.epoch}], "
                f"Step [{i + 1}/{len(train_loader)}], "
                f"G_adv: {last_g_adv_loss:.4f}, "
                f"G_freq: {last_g_freq_loss:.6f}, "
                f"G_loss: {last_g_loss:.4f}, "
                f"D_loss: {d_loss.item():.4f}"
            )

        append_epoch_fake_stats(output_dirs, epoch + 1, epoch_fake_stats)

        best_acc, best_epoch, metrics = evaluate_and_save(
            D=D,
            G=G,
            test_loader=test_loader,
            args=args,
            output_dirs=output_dirs,
            epoch=epoch,
            best_acc=best_acc,
            best_epoch=best_epoch,
        )

        if best_epoch == epoch + 1:
            best_metrics = metrics

    print(f"========== Fold {fold_id} Finished. Best Acc = {best_acc:.4f}, Best Epoch = {best_epoch} ==========\n")

    if best_metrics is None:
        best_metrics = metrics

    best_metrics["best_acc"] = best_acc
    best_metrics["best_epoch"] = best_epoch
    best_metrics["fold"] = fold_id

    return best_metrics


def train_5fold(args):
    root_output_dirs = prepare_output_dirs(args.save_dir)

    if args.seed is not None:
        seed_everything(args.seed)

    fold_loaders = build_5fold_loaders(args, root_output_dirs)

    all_results = []

    for fold_id, train_dataset, train_loader, test_loader in fold_loaders:
        fold_save_dir = os.path.join(args.save_dir, f"fold_{fold_id}")

        result = train_one_fold(
            args=args,
            fold_id=fold_id,
            train_dataset=train_dataset,
            train_loader=train_loader,
            test_loader=test_loader,
            fold_save_dir=fold_save_dir,
        )

        all_results.append(result)

    summary_path = os.path.join(args.save_dir, "five_fold_summary.txt")

    metric_names = ["accuracy", "precision", "specificity", "recall", "f1", "mcc", "best_acc"]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== Five-fold Cross Validation Summary ==========\n")

        for r in all_results:
            line = (
                f"Fold {r['fold']}: "
                f"BestEpoch={r['best_epoch']}, "
                f"Acc={r['accuracy']:.6f}, "
                f"Precision={r['precision']:.6f}, "
                f"Specificity={r['specificity']:.6f}, "
                f"Recall={r['recall']:.6f}, "
                f"F1={r['f1']:.6f}, "
                f"MCC={r['mcc']:.6f}, "
                f"BestAcc={r['best_acc']:.6f}\n"
            )
            print(line.strip())
            f.write(line)

        f.write("\n========== Mean ± Std ==========\n")

        for name in metric_names:
            values = np.array([r[name] for r in all_results], dtype=float)
            line = f"{name}: {values.mean():.6f} ± {values.std(ddof=1):.6f}\n"
            print(line.strip())
            f.write(line)

    print(f"[Saved] five-fold summary -> {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--interaction_data", default="./data/yeast/protein.actions.tsv", type=str)
    parser.add_argument("--sequence_data", default="./data/yeast/protein.dictionary.tsv", type=str)
    parser.add_argument("--d_pth", default="", type=str)
    parser.add_argument("--save_dir", default="./Result/PPIGAN_5fold", type=str)

    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--epoch", default=300, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--is_only_dis", action="store_true")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--lambda_entropy", default=0.0, type=float)

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

    parser.add_argument("--beta_real_loss", default=1.0, type=float)
    parser.add_argument("--beta_fake_loss", default=0.05, type=float)
    parser.add_argument("--lambda_freq", default=5.0, type=float)
    parser.add_argument("--freq_warmup_epochs", default=5, type=int)
    parser.add_argument("--g_steps", default=2, type=int)
    parser.add_argument("--noise_scale", default=1.0, type=float)

    parser.add_argument("--max_save_fake", default=8, type=int)
    parser.add_argument("--save_fake_every", default=10, type=int)
    parser.add_argument("--save_each_epoch", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cuda and torch.cuda.is_available():
        args.device = select_device("cuda:0")
    else:
        args.device = select_device("cpu")

    print("[Info] args =", args)

    train_5fold(args)
