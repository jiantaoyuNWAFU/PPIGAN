import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import MyDataset
from Discriminator import Dis
from torch_utils import select_device


def test(args):
    os.makedirs(args.save_dir, exist_ok=True)

    test_dataset = MyDataset(args.interaction_data, args.sequence_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = Dis(args).to(args.device)

    if args.d_pth == "":
        raise ValueError("Please provide --d_pth, e.g. ./Result/Biogrid-human/xxx/checkpoints/D_best_acc.pth")

    state_dict = torch.load(args.d_pth, map_location=args.device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in test_loader:
            x1, x2, y = batch[0], batch[1], batch[2]

            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)

            outputs = model(x1, x2, None)
            prob = outputs.cpu().numpy()[:, 1]
            label = y.cpu().numpy()[:, 1]

            pred = (prob > args.threshold).astype(int)

            y_true.extend(label.tolist())
            y_pred.extend(pred.tolist())
            y_prob.extend(prob.tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

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

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    print("Confusion Matrix:")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Specificity:", specificity)
    print("Recall:", recall)
    print("F1:", f1)
    print("MCC:", mcc)
    print("AUROC:", roc_auc)
    print("AUPRC:", auprc)

    metric_path = os.path.join(args.save_dir, "independent_test_metrics.txt")
    with open(metric_path, "w", encoding="utf-8") as f:
        f.write(f"d_pth: {args.d_pth}\n")
        f.write(f"interaction_data: {args.interaction_data}\n")
        f.write(f"sequence_data: {args.sequence_data}\n")
        f.write(f"threshold: {args.threshold}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Specificity: {specificity}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"MCC: {mcc}\n")
        f.write(f"AUROC: {roc_auc}\n")
        f.write(f"AUPRC: {auprc}\n")

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"PPIGAN (area = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "auroc_curve.png"), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(pr_recall, pr_precision, lw=2, label=f"PPIGAN (area = {auprc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "auprc_curve.png"), dpi=300)
    plt.close()

    print(f"Results saved to: {args.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--interaction_data",
        default="./data/virus-human/protein.actions.tsv",
        type=str
    )
    parser.add_argument(
        "--sequence_data",
        default="./data/virus-human/protein.dictionary.tsv",
        type=str
    )
    parser.add_argument("--d_pth", default="", type=str)
    parser.add_argument("--save_dir", default="./Result/independent_test", type=str)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("--em_dim", default=15, type=int)
    parser.add_argument("--conv_num", default=10, type=int)
    parser.add_argument("--sp_drop", default=0.005, type=float)
    parser.add_argument("--kernel_rate_1", default=0.16, type=float)
    parser.add_argument("--strides_rate_1", default=0.15, type=float)
    parser.add_argument("--kernel_rate_2", default=0.14, type=float)
    parser.add_argument("--strides_rate_2", default=0.25, type=float)
    parser.add_argument("--filter_num_1", default=150, type=int)
    parser.add_argument("--filter_num_2", default=175, type=int)
    parser.add_argument("--con_drop", default=0.05, type=float)
    parser.add_argument("--fn_drop_1", default=0.2, type=float)
    parser.add_argument("--fn_drop_2", default=0.1, type=float)
    parser.add_argument("--node_num", default=256, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cuda and torch.cuda.is_available():
        args.device = select_device("cuda:0")
    else:
        args.device = select_device("cpu")

    print("[Info] args =", args)
    test(args)
