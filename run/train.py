# -*- coding: utf-8 -*-
import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
import torch.nn.functional as F

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
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_output_dirs(save_dir: str) -> Dict[str, str]:
    dirs = {
        "root": save_dir,
        "checkpoints": os.path.join(save_dir, "checkpoints"),
        "datasets": os.path.join(save_dir, "datasets"),
        "fake_samples": os.path.join(save_dir, "fake_samples"),
        "fasta_list": os.path.join(save_dir, "fasta_list"),
        "Log": os.path.join(save_dir, "Log"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_dataset_auto(data_path: str, seq_path: str):
    if data_path.endswith(".pth") and os.path.exists(data_path):
        print(f"[Info] loading dataset from pth: {data_path}")
        return torch.load(data_path)
    print(f"[Info] building dataset from raw files: {data_path}")
    return MyDataset(data_path, seq_path)


def split_train_val(dataset, val_ratio: float, seed: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val_ratio must be within (0, 1) when validation is enabled.")
    val_size = int(round(len(dataset) * val_ratio))
    val_size = max(1, min(val_size, len(dataset) - 1))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    print(f"[Info] internal split -> train: {train_size}, validation: {val_size}")
    return train_set, val_set


def build_loaders(args, output_dirs: Dict[str, str]):
    if args.train_dataset:
        train_dataset = load_dataset_auto(args.train_dataset, args.sequence_data)
    else:
        full_dataset = MyDataset(
            args.interaction_data, args.sequence_data
        )
        if args.val_ratio > 0.0 and not args.val_dataset:
            train_dataset, val_dataset = split_train_val(full_dataset, args.val_ratio, args.seed)
        else:
            train_dataset = full_dataset
            val_dataset = None

    if args.val_dataset:
        val_dataset = load_dataset_auto(args.val_dataset, args.sequence_data)
    elif "val_dataset" not in locals():
        val_dataset = None

    print(f"[Info] training samples: {len(train_dataset)}")
    print(f"[Info] validation samples: {len(val_dataset) if val_dataset is not None else 0}")

    torch.save(train_dataset, os.path.join(output_dirs["datasets"], "train_dataset.pth"))
    if val_dataset is not None:
        torch.save(val_dataset, os.path.join(output_dirs["datasets"], "validation_dataset.pth"))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.cuda,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )
    return train_dataset, train_loader, val_loader

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
        y, pid1, pid2 = sample[2], sample[3], sample[4]
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        if y_arr.shape[-1] >= 2 and int(y_arr[1]) == 1:
            degree_dict[pid1] += 1
            degree_dict[pid2] += 1
    return degree_dict


def select_condition_proteins(
    x1: torch.Tensor,
    x2: torch.Tensor,
    y_cls: torch.Tensor,
    pid1,
    pid2,
    protein_degrees,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

    pos_mask = y_cls == 1
    x1_pos = x1[pos_mask]
    x2_pos = x2[pos_mask]
    if x1_pos.size(0) == 0:
        return None, None

    mask_list = pos_mask.detach().cpu().tolist()
    pid1_pos = [pid1[idx] for idx, flag in enumerate(mask_list) if flag]
    pid2_pos = [pid2[idx] for idx, flag in enumerate(mask_list) if flag]

    condition_list = []
    partner_list = []
    for idx in range(len(pid1_pos)):
        d1 = protein_degrees.get(pid1_pos[idx], 0)
        d2 = protein_degrees.get(pid2_pos[idx], 0)
        if d1 > d2:
            condition_list.append(x1_pos[idx])
            partner_list.append(x2_pos[idx])
        elif d2 > d1:
            condition_list.append(x2_pos[idx])
            partner_list.append(x1_pos[idx])
        elif np.random.rand() < 0.5:
            condition_list.append(x1_pos[idx])
            partner_list.append(x2_pos[idx])
        else:
            condition_list.append(x2_pos[idx])
            partner_list.append(x1_pos[idx])

    return torch.stack(condition_list), torch.stack(partner_list)

def build_id_to_token():
    if not hasattr(amino_acids, "amino_acid") or not isinstance(amino_acids.amino_acid, dict):
        raise RuntimeError("amino_acids.py does not contain a valid amino_acid dictionary.")
    id_to_token = {v: str(k) for k, v in amino_acids.amino_acid.items() if isinstance(v, int)}
    if not id_to_token:
        raise RuntimeError("failed to build id_to_token from amino_acids.amino_acid")
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
            for aa in parts[1].strip():
                if aa in aa_to_idx:
                    counts[aa_to_idx[aa]] += 1
    return (counts / counts.sum().clamp_min(1.0)).to(device)


def logits_to_soft_embedding(fake_logits: torch.Tensor, embedding_layer: nn.Embedding):
    fake_probs = torch.softmax(fake_logits, dim=-1)
    fake_embed = torch.matmul(fake_probs, embedding_layer.weight)
    return fake_embed, fake_probs


def compute_fake_aa_freq_from_probs(fake_probs: torch.Tensor, condition_protein: torch.Tensor):
    valid_mask = (condition_protein != 0).unsqueeze(-1).float()
    masked_probs = fake_probs * valid_mask
    aa_sum = masked_probs.sum(dim=(0, 1))
    return aa_sum[1:] / aa_sum[1:].sum().clamp_min(1.0)


def save_fake_fasta_from_logits(
    output_dirs: Dict[str, str],
    epoch: int,
    fake_logits: torch.Tensor,
    condition_protein: torch.Tensor,
    max_save: int,
):
    id_to_token = build_id_to_token()
    n = min(max_save, fake_logits.size(0))
    probs = torch.softmax(fake_logits[:n].detach().cpu(), dim=-1)

    probs[:, :, 0] = 0.0
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    fake_ids = torch.multinomial(
        probs.reshape(-1, probs.size(-1)),
        num_samples=1
    ).reshape(probs.size(0), probs.size(1))

    valid_mask = condition_protein[:n].detach().cpu() != 0
    fasta_path = os.path.join(output_dirs["fasta_list"], f"fake_epoch_{epoch}.fasta")
    with open(fasta_path, "w", encoding="utf-8") as f:
        for i in range(n):
            seq = ids_to_seq(fake_ids[i][valid_mask[i]].tolist(), id_to_token, remove_zero=True)
            if not seq:
                seq = "X"
            f.write(f">fake_epoch_{epoch}_sample{i}\n")
            for j in range(0, len(seq), 60):
                f.write(seq[j:j + 60] + "\n")

def init_models(args):

    D = Dis(args).to(args.device)
    G = Gen(args).to(args.device)

    if args.d_pth:
        print(f"[Info] loading initial discriminator from: {args.d_pth}")
        state_dict = torch.load(args.d_pth, map_location=args.device)
        D.load_state_dict(state_dict)
    else:
        print("[Info] models use PyTorch default parameter initialisation.")

    return D, G


def configure_embedding_trainability(D: nn.Module, freeze_embedding: bool) -> None:
    if hasattr(D, "embedding_layer"):
        for param in D.embedding_layer.parameters():
            param.requires_grad = not freeze_embedding
        status = "frozen" if freeze_embedding else "trainable"
        print(f"[Info] D.embedding_layer is {status}.")


def set_discriminator_grad(D: nn.Module, enabled: bool, freeze_embedding: bool) -> None:
    for param in D.parameters():
        param.requires_grad = enabled
    if enabled and freeze_embedding and hasattr(D, "embedding_layer"):
        for param in D.embedding_layer.parameters():
            param.requires_grad = False

def evaluate(D: nn.Module, data_loader, args) -> Dict[str, float]:
    D.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x1, x2, y, _, _ in data_loader:
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            outputs = D(x1, x2, None).detach().cpu().numpy()[:, 1]
            labels = y.detach().cpu().numpy()[:, 1]
            y_true.extend(labels.tolist())
            y_prob.extend(outputs.tolist())

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob > args.threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if np.unique(y_true).size == 2:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc_ap"] = average_precision_score(y_true, y_prob)
        metrics["auprc_trapz"] = auc(recall_curve, precision_curve)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc_ap"] = float("nan")
        metrics["auprc_trapz"] = float("nan")
    metrics["cm"] = cm
    return metrics


def append_validation_metrics(output_dirs: Dict[str, str], epoch: int, metrics: Dict[str, float]) -> None:
    path = os.path.join(output_dirs["Log"], "validation_metrics.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            f"Epoch: {epoch}\n"
            f"Confusion Matrix:\n{metrics['cm']}\n"
            f"Accuracy: {metrics['accuracy']}\n"
            f"Precision: {metrics['precision']}\n"
            f"Specificity: {metrics['specificity']}\n"
            f"Recall: {metrics['recall']}\n"
            f"F1: {metrics['f1']}\n"
            f"MCC: {metrics['mcc']}\n"
            f"AUROC: {metrics['auroc']}\n"
            f"AUPRC_AP: {metrics['auprc_ap']}\n"
            f"AUPRC_TRAPZ: {metrics['auprc_trapz']}\n"
            "===============================================\n"
        )

def train(args):
    output_dirs = prepare_output_dirs(args.save_dir)
    seed_everything(args.seed)
    train_dataset, train_loader, val_loader = build_loaders(args, output_dirs)

    real_aa_freq = None
    if args.lambda_freq > 0.0:
        real_aa_freq = get_real_aa_freq_from_dictionary_tsv(args.sequence_data, args.device)
        print("[Info] lambda_freq enabled; real amino-acid frequency loaded.")
    else:
        print("[Info] lambda_freq=0: amino-acid frequency regularisation is OFF for baseline reproduction.")

    D, G = init_models(args)
    configure_embedding_trainability(D, args.freeze_embedding)

    criterion = nn.CrossEntropyLoss()
    optimizer_D = torch.optim.Adam(
        filter(lambda p: p.requires_grad, D.parameters()),
        lr=args.d_lr,
        betas=(0.9, 0.999),
        eps=1e-6,
    )
    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=args.g_lr, betas=(0.9, 0.999), eps=1e-6
    )

    protein_degrees = calculate_protein_degree_from_dataset(train_dataset)
    print(f"[Info] protein-degree dictionary from training data only: {len(protein_degrees)} proteins")
    print(f"[Info] update ratio D:G = {args.d_steps}:{args.g_steps}")

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    best_val_ap = -float("inf")
    best_epoch = 0

    best_acc = 0.0
    best_acc_epoch = 0

    for epoch in range(1, args.epoch + 1):
        D.train()
        G.train()
        d_losses, g_losses = [], []
        last_fake_logits = None
        last_condition = None

        for step, (x1, x2, y, pid1, pid2) in enumerate(train_loader, start=1):
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)
            y_cls = torch.argmax(y, dim=1).long()

            condition_protein, native_partner = select_condition_proteins(
                x1, x2, y_cls, pid1, pid2, protein_degrees
            )

            for _ in range(args.d_steps):
                set_discriminator_grad(D, True, args.freeze_embedding)
                optimizer_D.zero_grad(set_to_none=True)

                real_outputs = D(x1, x2, None, return_logits=True)
                real_loss = criterion(real_outputs, y_cls)

                if args.is_only_dis or condition_protein is None:
                    d_loss = real_loss
                else:
                    batch_pos_size = condition_protein.size(0)
                    fake_labels = torch.zeros(
                        batch_pos_size,
                        dtype=torch.long,
                        device=args.device,
                    )

                    z = args.noise_scale * torch.randn(
                        (batch_pos_size, 1500, args.em_dim),
                        device=args.device,
                    )

                    with torch.no_grad():
                        fake_logits = G(condition_protein, z)
                        fake_probs = torch.softmax(fake_logits, dim=-1)

                    fake_outputs = D(
                        condition_protein,
                        fake_probs,
                        True,
                        return_logits=True,
                    )

                    fake_loss = criterion(fake_outputs, fake_labels)

                    d_loss = (
                        args.beta_real_loss * real_loss
                        + args.beta_fake_loss * fake_loss
                    )

                d_loss.backward()
                optimizer_D.step()

                d_losses.append(float(d_loss.item()))

            if not args.is_only_dis and condition_protein is not None:
                real_labels = torch.ones(
                    condition_protein.size(0),
                    dtype=torch.long,
                    device=args.device,
                )

                for _ in range(args.g_steps):
                    set_discriminator_grad(D, True, args.freeze_embedding)

                    optimizer_G.zero_grad(set_to_none=True)
                    optimizer_D.zero_grad(set_to_none=True)

                    z_g = args.noise_scale * torch.randn(
                        (condition_protein.size(0), 1500, args.em_dim),
                        device=args.device,
                    )
                    fake_logits_g = G(condition_protein, z_g)
                    fake_probs_g = torch.softmax(fake_logits_g, dim=-1)

                    fake_outputs_g = D(
                        condition_protein,
                        fake_probs_g,
                        True,
                        return_logits=True,
                    )

                    g_adv_loss = criterion(fake_outputs_g, real_labels)

                    if args.lambda_freq > 0.0:
                        fake_aa_freq = compute_fake_aa_freq_from_probs(
                            fake_probs_g,
                            condition_protein,
                        )

                        g_freq_loss = F.kl_div(
                            torch.log(fake_aa_freq + 1e-8),
                            real_aa_freq,
                            reduction="batchmean"
                        )

                        lambda_now = (
                            args.lambda_freq
                            if epoch > args.freq_warmup_epochs
                            else 0.0
                        )
                    else:
                        g_freq_loss = torch.zeros((), device=args.device)
                        lambda_now = 0.0

                    g_loss = g_adv_loss + lambda_now * g_freq_loss

                    if epoch == 1 and step == 1:
                        print(
                            "[GradCheck before backward] "
                            f"fake_logits_g={fake_logits_g.requires_grad}, "
                            f"fake_probs_g={fake_probs_g.requires_grad}, "
                            f"fake_outputs_g={fake_outputs_g.requires_grad}, "
                            f"g_loss={g_loss.requires_grad}"
                        )

                    if not g_loss.requires_grad:
                        raise RuntimeError(
                            "Generator loss has no gradient graph. "
                            "Please check the generated-sequence branch in Discriminator.py."
                        )

                    g_loss.backward()

                    if epoch == 1 and step == 1:
                        has_g_grad = any(
                            param.grad is not None
                            for param in G.parameters()
                            if param.requires_grad
                        )

                        print(
                            f"[GradCheck after backward] "
                            f"any_generator_grad={has_g_grad}"
                        )

                        if not has_g_grad:
                            raise RuntimeError(
                                "Gradient still cannot reach Generator. "
                                "Check whether D is called with is_gen=True."
                            )

                    optimizer_G.step()
                    optimizer_D.zero_grad(set_to_none=True)

                    g_losses.append(float(g_loss.item()))
                    last_fake_logits = fake_logits_g
                    last_condition = condition_protein

            if step % args.log_interval == 0 or step == len(train_loader):
                d_mean = float(np.mean(d_losses[-args.log_interval:])) if d_losses else float("nan")
                g_mean = float(np.mean(g_losses[-args.log_interval:])) if g_losses else float("nan")
                print(
                    f"Epoch [{epoch}/{args.epoch}] Step [{step}/{len(train_loader)}] "
                    f"D_loss={d_mean:.4f} G_loss={g_mean:.4f}"
                )

        if last_fake_logits is not None and last_condition is not None:
            save_fake_fasta_from_logits(
                output_dirs, epoch, last_fake_logits, last_condition, args.max_save_fake
            )

        if epoch % args.save_interval == 0 or epoch == args.epoch:
            save_checkpoint(D, os.path.join(output_dirs["checkpoints"], f"D_epoch_{epoch}.pth"))
            if not args.is_only_dis:
                save_checkpoint(G, os.path.join(output_dirs["checkpoints"], f"G_epoch_{epoch}.pth"))

                print("Current epoch model saved!")

        train_log = os.path.join(output_dirs["Log"], "train_loss.txt")
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch}: D_loss={np.mean(d_losses):.8f}, "
                f"G_loss={np.mean(g_losses) if g_losses else float('nan'):.8f}\n"
            )

        if val_loader is not None:
            metrics = evaluate(D, val_loader, args)
            append_validation_metrics(output_dirs, epoch, metrics)

            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                best_acc_epoch = epoch

                save_checkpoint(
                    D,
                    os.path.join(output_dirs["checkpoints"], "D_best_acc.pth"),
                )

                if not args.is_only_dis:
                    save_checkpoint(
                        G,
                        os.path.join(output_dirs["checkpoints"], "G_best_acc.pth"),
                    )

                print(
                    f"New best model saved! "
                    f"Best accuracy = {best_acc:.4f} at epoch {best_acc_epoch}"
                )

            if metrics["auprc_ap"] > best_val_ap:
                best_val_ap = metrics["auprc_ap"]
                best_epoch = epoch

                save_checkpoint(
                    D,
                    os.path.join(output_dirs["checkpoints"], "D_best_val_auprc.pth"),
                )

                if not args.is_only_dis:
                    save_checkpoint(
                        G,
                        os.path.join(output_dirs["checkpoints"], "G_best_val_auprc.pth"),
                    )

            print("混淆矩阵:")
            print(metrics["cm"])
            print("准确率:", metrics["accuracy"])
            print("精确率:", metrics["precision"])
            print("特异性:", metrics["specificity"])
            print("召回率:", metrics["recall"])
            print("F1值:", metrics["f1"])
            print("MCC:", metrics["mcc"])
            print("AUROC:", metrics["auroc"])
            print("AUPRC:", metrics["auprc_ap"])
            print(f"Best accuracy so far: {best_acc:.4f}, Best epoch: {best_acc_epoch}")
            print("===============================================")

            metric_log_path = os.path.join(output_dirs["Log"], "validation_old_style_metrics.txt")

            with open(metric_log_path, "a+", encoding="utf-8") as f:
                f.write(
                    f"Epoch [{epoch}/{args.epoch}]\n"
                    f"cm:{metrics['cm']}\n"
                    f"Accuracy: {metrics['accuracy']}, "
                    f"Precision: {metrics['precision']}, "
                    f"Specificity: {metrics['specificity']}, "
                    f"Recall: {metrics['recall']}, "
                    f"F1: {metrics['f1']}, "
                    f"MCC:{metrics['mcc']}, "
                    f"AUROC:{metrics['auroc']}, "
                    f"AUPRC:{metrics['auprc_ap']}\n"
                    f"Best accuracy: {best_acc:.4f}, "
                    f"Best epoch: {best_acc_epoch}\n"
                    "===============================================\n"
                )

    save_checkpoint(D, os.path.join(output_dirs["checkpoints"], "D_final.pth"))
    if not args.is_only_dis:
        save_checkpoint(G, os.path.join(output_dirs["checkpoints"], "G_final.pth"))
    print("[Finished] Training completed.")
    print(f"[Finished] Final discriminator: {os.path.join(output_dirs['checkpoints'], 'D_final.pth')}")
    if val_loader is not None:
        print(f"[Finished] Best validation AUPRC={best_val_ap:.6f} at epoch={best_epoch}")
    else:
        print("[Protocol] No validation/test data were used during training; evaluate D_final.pth once on the independent test set.")

def parse_args():
    parser = argparse.ArgumentParser(description="PPIGAN paper-aligned baseline training")

    parser.add_argument("--interaction_data", default="./data/Biogrid-human/protein.actions.tsv", type=str)
    parser.add_argument("--sequence_data", default="./data/Biogrid-human/protein.dictionary.tsv", type=str)
    parser.add_argument("--train_dataset", default="", type=str, help="Optional saved training dataset .pth")
    parser.add_argument("--val_dataset", default="", type=str, help="Optional saved validation dataset .pth; never pass virus-human here")
    parser.add_argument("--val_ratio", default=0.0, type=float, help="Internal BioGRID validation ratio; 0 trains on full BioGRID for fixed-epoch reproduction")
    parser.add_argument("--d_pth", default="", type=str, help="Optional initial discriminator checkpoint")
    parser.add_argument("--save_dir", default="./Result/Biogrid-human/PPIGAN_paper_aligned", type=str)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--detect_anomaly", action="store_true")
    parser.add_argument("--is_only_dis", action="store_true")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--d_steps", default=2, type=int, help="Paper-aligned default: D updates twice")
    parser.add_argument("--g_steps", default=1, type=int, help="Paper-aligned default: G updates once")

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
    parser.add_argument("--g_lr", default=1e-4, type=float)

    parser.add_argument("--beta_real_loss", default=1.0, type=float)
    parser.add_argument("--beta_fake_loss", default=0.05, type=float)
    parser.add_argument("--lambda_freq", default=0.0, type=float)
    parser.add_argument("--freq_warmup_epochs", default=5, type=int)
    parser.add_argument("--freeze_embedding", action="store_true")
    parser.add_argument("--noise_scale", default=1.0, type=float)

    parser.add_argument("--save_interval", default=1, type=int)
    parser.add_argument("--max_save_fake", default=128, type=int)
    parser.add_argument("--log_interval", default=20, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.cuda and torch.cuda.is_available():
        args.device = select_device("cuda:0")
    else:
        args.device = select_device("cpu")
    print("[Info] args =", args)
    train(args)
