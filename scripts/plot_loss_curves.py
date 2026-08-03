"""Plot train/val loss and val AUC across folds from a loss_curves.json file.

Usage:
    python scripts/plot_loss_curves.py --in outputs/akd_full.loss_curves.json \
        --out outputs/akd_full.loss_curves.png --title "dinov2 akd late-fusion"
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


FOLD_COLORS = ["#1f77b4", "#2ca02c", "#8c564b", "#7f7f7f", "#17becf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--title", default="5-fold training curves")
    args = ap.parse_args()

    data = json.loads(Path(args.in_path).read_text())
    fold_keys = sorted(data.keys(), key=int)

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(13, 5))

    for i, k in enumerate(fold_keys):
        h = data[k]
        color = FOLD_COLORS[i % len(FOLD_COLORS)]
        epochs = range(1, len(h["train_loss"]) + 1)
        ax_loss.plot(epochs, h["train_loss"], color=color, alpha=0.45, linewidth=1.5)
        ax_loss.plot(epochs, h["val_loss"], color=color, linestyle="--", linewidth=1.5)
        ax_auc.plot(epochs, h["val_auc"], color=color, linewidth=1.8, label=f"fold {k}")

    train_proxy = plt.Line2D([], [], color="gray", linewidth=1.5, label="train")
    val_proxy = plt.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="val")
    fold_proxies = [
        plt.Line2D([], [], color=FOLD_COLORS[i % len(FOLD_COLORS)], linewidth=1.5,
                   linestyle="--", label=f"fold {k}")
        for i, k in enumerate(fold_keys)
    ]
    ax_loss.legend(handles=[train_proxy, val_proxy, *fold_proxies],
                   loc="upper right", fontsize=8, ncol=2)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("BCE loss")
    ax_loss.set_title("train vs val loss (5 folds)")
    ax_loss.grid(alpha=0.3)

    ax_auc.legend(loc="lower right", fontsize=8)
    ax_auc.set_xlabel("epoch")
    ax_auc.set_ylabel("val pat-AUC mean")
    ax_auc.set_title("val AUC (5 folds)")
    ax_auc.grid(alpha=0.3)

    fig.suptitle(args.title, fontsize=14)
    fig.tight_layout()
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()