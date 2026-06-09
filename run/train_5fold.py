# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
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


def get_dataset_labels(dataset) -> np.ndarray:
    labels = []

    if hasattr(dataset, "y_train"):
        for y in dataset.y_train:
            y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            labels.append(int(np.argmax(y_arr)))
        return np.asarray(labels, dtype=int)

    for idx in range(len(dataset)):
        sample = dataset[idx]
        y = sample[2]
        y_arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        labels.append(int(np.argmax(y_arr)))

    return np.asarray(labels, dtype=int)


def build_5fold_loaders(args, root_output_dirs: Dict[str, str]):
    dataset = load_dataset_auto(args.interaction_data, args.sequence_data)
    labels = get_dataset_labels(dataset)

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_items = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
        train_dataset = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_dataset = torch.utils.data.Subset(dataset, val_idx.tolist())

        torch.save(
            train_dataset,
            os.path.join(root_output_dirs["datasets"], f"fold_{fold_id}_train_dataset.pth"),
        )
        torch.save(
            val_dataset,
            os.path.join(root_output_dirs["datasets"], f"fold_{fold_id}_validation_dataset.pth"),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.cuda,
        )

        print(
            f"[Fold {fold_id}] train={len(train_dataset)}, validation={len(val_dataset)}, "
            f"train_pos={int(labels[train_idx].sum())}, val_pos={int(labels[val_idx].sum())}, "
            f"train_neg={len(train_idx) - int(labels[train_idx].sum())}, "
            f"val_neg={len(val_idx) - int(labels[val_idx].sum())}"
        )

        fold_items.append((fold_id, train_dataset, train_loader, val_loader))

    return fold_items


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


def append_fold_metrics(output_dirs: Dict[str, str], epoch: int, metrics: Dict[str, float]) -> None:
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


def train_one_fold(args, fold_id, train_dataset, train_loader, val_loader, fold_save_dir):
    print(f"\n========== Start Fold {fold_id}/{args.n_splits} ==========")

    output_dirs = prepare_output_dirs(fold_save_dir)
    seed_everything(args.seed + fold_id)

    real_aa_freq = None
    if args.lambda_freq > 0.0:
        real_aa_freq = get_real_aa_freq_from_dictionary_tsv(args.sequence_data, args.device)
        print("[Info] lambda_freq enabled; real amino-acid frequency loaded.")
        print("[Info] real_aa_freq =", real_aa_freq.detach().cpu().numpy())
    else:
        print("[Info] lambda_freq=0: amino-acid frequency regularisation is OFF.")

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
    print(f"[Info] fold-{fold_id} protein-degree dictionary from training data only: {len(protein_degrees)} proteins")
    print(f"[Info] update ratio D:G = {args.d_steps}:{args.g_steps}")

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    best_val_ap = -float("inf")
    best_acc = 0.0
    best_epoch_by_ap = 0
    best_epoch_by_acc = 0
    best_metrics_by_ap = None
    final_metrics = None

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
                    fake_labels = torch.zeros(batch_pos_size, dtype=torch.long, device=args.device)

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
                    d_loss = args.beta_real_loss * real_loss + args.beta_fake_loss * fake_loss

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
                            reduction="batchmean",
                        )
                        lambda_now = args.lambda_freq if epoch > args.freq_warmup_epochs else 0.0
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
                        print(f"[GradCheck after backward] any_generator_grad={has_g_grad}")
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
                    f"[Fold {fold_id}] Epoch [{epoch}/{args.epoch}] Step [{step}/{len(train_loader)}] "
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
            print(f"[Fold {fold_id}] Current epoch model saved.")

        train_log = os.path.join(output_dirs["Log"], "train_loss.txt")
        with open(train_log, "a", encoding="utf-8") as f:
            f.write(
                f"Epoch {epoch}: D_loss={np.mean(d_losses):.8f}, "
                f"G_loss={np.mean(g_losses) if g_losses else float('nan'):.8f}\n"
            )

        metrics = evaluate(D, val_loader, args)
        append_fold_metrics(output_dirs, epoch, metrics)
        final_metrics = metrics

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_epoch_by_acc = epoch
            save_checkpoint(D, os.path.join(output_dirs["checkpoints"], "D_best_acc.pth"))
            if not args.is_only_dis:
                save_checkpoint(G, os.path.join(output_dirs["checkpoints"], "G_best_acc.pth"))
            print(f"[Fold {fold_id}] New best ACC saved: {best_acc:.6f} at epoch {best_epoch_by_acc}")

        if metrics["auprc_ap"] > best_val_ap:
            best_val_ap = metrics["auprc_ap"]
            best_epoch_by_ap = epoch
            best_metrics_by_ap = metrics.copy()
            save_checkpoint(D, os.path.join(output_dirs["checkpoints"], "D_best_val_auprc.pth"))
            if not args.is_only_dis:
                save_checkpoint(G, os.path.join(output_dirs["checkpoints"], "G_best_val_auprc.pth"))
            print(f"[Fold {fold_id}] New best AUPRC saved: {best_val_ap:.6f} at epoch {best_epoch_by_ap}")

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
        print(
            f"[Fold {fold_id}] Best ACC={best_acc:.6f} at epoch={best_epoch_by_acc}; "
            f"Best AUPRC={best_val_ap:.6f} at epoch={best_epoch_by_ap}"
        )
        print("===============================================")

    save_checkpoint(D, os.path.join(output_dirs["checkpoints"], "D_final.pth"))
    if not args.is_only_dis:
        save_checkpoint(G, os.path.join(output_dirs["checkpoints"], "G_final.pth"))

    if best_metrics_by_ap is None:
        best_metrics_by_ap = final_metrics

    result = {
        "fold": fold_id,
        "best_epoch_by_auprc": best_epoch_by_ap,
        "best_epoch_by_acc": best_epoch_by_acc,
        "best_acc": best_acc,
        "best_auprc_ap": best_val_ap,
    }
    for key in [
        "accuracy",
        "precision",
        "specificity",
        "recall",
        "f1",
        "mcc",
        "auroc",
        "auprc_ap",
        "auprc_trapz",
    ]:
        result[key] = float(best_metrics_by_ap[key])

    print(
        f"========== Fold {fold_id} Finished. "
        f"BestAUPRC={best_val_ap:.6f}, BestACC={best_acc:.6f} ==========\n"
    )
    return result


def write_5fold_summary(args, all_results: List[Dict[str, float]]) -> None:
    summary_path = os.path.join(args.save_dir, "five_fold_summary.txt")
    metric_names = [
        "accuracy",
        "precision",
        "specificity",
        "recall",
        "f1",
        "mcc",
        "auroc",
        "auprc_ap",
        "auprc_trapz",
        "best_acc",
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== PPIGAN Five-fold Cross Validation Summary ==========\n")
        f.write(f"interaction_data: {args.interaction_data}\n")
        f.write(f"sequence_data: {args.sequence_data}\n")
        f.write(f"n_splits: {args.n_splits}\n")
        f.write(f"seed: {args.seed}\n")
        f.write("Selection rule: report metrics from D_best_val_auprc.pth within each fold.\n\n")

        for r in all_results:
            line = (
                f"Fold {r['fold']}: "
                f"BestEpochByAUPRC={r['best_epoch_by_auprc']}, "
                f"BestEpochByACC={r['best_epoch_by_acc']}, "
                f"Accuracy={r['accuracy']:.6f}, "
                f"Precision={r['precision']:.6f}, "
                f"Specificity={r['specificity']:.6f}, "
                f"Recall={r['recall']:.6f}, "
                f"F1={r['f1']:.6f}, "
                f"MCC={r['mcc']:.6f}, "
                f"AUROC={r['auroc']:.6f}, "
                f"AUPRC_AP={r['auprc_ap']:.6f}, "
                f"AUPRC_TRAPZ={r['auprc_trapz']:.6f}, "
                f"BestACC={r['best_acc']:.6f}\n"
            )
            print(line.strip())
            f.write(line)

        f.write("\n========== Mean ± Std ==========\n")
        for name in metric_names:
            values = np.asarray([r[name] for r in all_results], dtype=float)
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            line = f"{name}: {values.mean():.6f} ± {std:.6f}\n"
            print(line.strip())
            f.write(line)

    print(f"[Saved] five-fold summary -> {summary_path}")


def train_5fold(args):
    root_output_dirs = prepare_output_dirs(args.save_dir)
    seed_everything(args.seed)

    fold_items = build_5fold_loaders(args, root_output_dirs)

    all_results = []
    for fold_id, train_dataset, train_loader, val_loader in fold_items:
        fold_save_dir = os.path.join(args.save_dir, f"fold_{fold_id}")
        result = train_one_fold(
            args=args,
            fold_id=fold_id,
            train_dataset=train_dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            fold_save_dir=fold_save_dir,
        )
        all_results.append(result)

    write_5fold_summary(args, all_results)


def parse_args():
    parser = argparse.ArgumentParser(description="PPIGAN paper-aligned 5-fold cross-validation training")

    parser.add_argument("--interaction_data", default="./data/yeast/protein.actions.tsv", type=str)
    parser.add_argument("--sequence_data", default="./data/yeast/protein.dictionary.tsv", type=str)
    parser.add_argument("--d_pth", default="", type=str, help="Optional initial discriminator checkpoint")
    parser.add_argument("--save_dir", default="./Result/PPIGAN_5fold", type=str)

    parser.add_argument("--n_splits", default=5, type=int)
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
    train_5fold(args)
