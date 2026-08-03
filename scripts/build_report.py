"""Build report.md from outputs/dinov2_biopsy/result.json.

For each of 4 outcomes, reports (pooled across 5 validation folds):
  - AUC, accuracy, sensitivity, specificity, PPV, NPV at threshold 0.5
  - confusion matrix
  - pos_patient(%)  = fraction of val patients with positive label
  - pos_image(%)    = fraction of val images with positive label

Usage:
  python scripts/build_report.py --result outputs/dinov2_biopsy/result.json \
      --cohort data/stage2_cohort_biopsy.parquet --out report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

OUTCOME_COLS = ["outcome1", "outcome2", "outcome3", "outcome4"]


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    if y_true.min() == y_true.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)   # recall
    spec = tn / max(1, tn + fp)
    ppv  = tp / max(1, tp + fp)   # precision
    npv  = tn / max(1, tn + fn)
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "sen": float(sens),
        "spe": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1":  float(f1_score(y_true, y_pred, zero_division=0)),
        "cm":  (int(tn), int(fp), int(fn), int(tp)),
        "n":   int(len(y_true)),
        "n_pos": int(y_true.sum()),
    }


def _fmt(v: float, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if not np.isnan(v) else "—"


def _fmt_pct(v: float, digits: int = 1) -> str:
    """Proportion displayed as percent (0.762 → 76.2), no % sign."""
    return f"{v * 100:.{digits}f}" if not np.isnan(v) else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patient-key", default="病理号",
                    help="Patient-id column in cohort parquet (default 病理号; use 病案号 for AKD)")
    ap.add_argument("--title", default="Late-fusion — per-outcome report",
                    help="Report title prefix")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    r = json.load(open(args.result))
    cohort = pd.read_parquet(args.cohort)
    pat_col = args.patient_key
    assert pat_col in cohort.columns, f"cohort missing patient column {pat_col}"
    thr = args.threshold

    sections = []
    summary_rows = []

    METRIC_KEYS = [
        ("AUC (%)", "auc"), ("Acc (%)", "acc"), ("Sen (%)", "sen"),
        ("Spe (%)", "spe"), ("PPV (%)", "ppv"), ("NPV (%)", "npv"),
        ("F1 (%)", "f1"),
    ]

    for col in OUTCOME_COLS:
        # ---- pool probs & labels across folds ----
        all_p, all_y, all_ids = [], [], []
        for fold_id, (val_pat_ids, probs, labels) in r["outcomes"][col]["fold_probs"].items():
            all_p.append(np.asarray(probs))
            all_y.append(np.asarray(labels).astype(int))
            all_ids.extend(val_pat_ids)
        y_true = np.concatenate(all_y)
        y_prob = np.concatenate(all_p)
        m = _metrics(y_true, y_prob, thr=thr)

        # ---- image-level pos rate (cohort restricted to val patients) ----
        val_set = set(all_ids)
        val_imgs = cohort[cohort[pat_col].astype(str).isin(val_set)]
        per_fold_img_pos, per_fold_pat_pos = [], []
        fold_items = sorted(r["outcomes"][col]["fold_probs"].items(), key=lambda kv: int(kv[0]))
        for fold_id, (val_pat_ids, probs, labels) in fold_items:
            f_set = set(val_pat_ids)
            lab = np.asarray(labels).astype(int)
            per_fold_pat_pos.append(float(lab.mean()) if len(lab) else float("nan"))
            f_imgs = cohort[cohort[pat_col].astype(str).isin(f_set)]
            pos = f_imgs[col].mean() if len(f_imgs) else float("nan")
            per_fold_img_pos.append(float(pos))
        pos_pat_mean = float(np.mean(per_fold_pat_pos))
        pos_img_mean = float(np.mean(per_fold_img_pos))
        pos_pat_std  = float(np.std(per_fold_pat_pos))
        pos_img_std  = float(np.std(per_fold_img_pos))

        # ---- per-fold metrics (for the per-outcome table) ----
        fold_metrics = [
            _metrics(
                np.asarray(labels).astype(int),
                np.asarray(probs),
                thr,
            )
            for _, (_, probs, labels) in fold_items
        ]
        metric_rows = []
        for label, key in METRIC_KEYS:
            vals_pct = [fm[key] * 100 for fm in fold_metrics]
            mean_pct = float(np.mean(vals_pct))
            std_pct = float(np.std(vals_pct))
            cells = " | ".join(f"{v:.1f}" for v in vals_pct)
            metric_rows.append(f"| {label} | {cells} | {mean_pct:.1f} ± {std_pct:.1f} |")
        metric_table = (
            "| metric | " + " | ".join(f"fold {k}" for k, _ in fold_items) + " | mean ± std |\n"
            + "|---" + "|---" * (len(fold_items) + 1) + "|\n"
            + "\n".join(metric_rows)
        )

        # mean ± std across folds, for the summary table
        mean_std = {}
        for _, key in METRIC_KEYS:
            vals_pct = [fm[key] * 100 for fm in fold_metrics]
            mean_std[key] = f"{float(np.mean(vals_pct)):.1f} ± {float(np.std(vals_pct)):.1f}"

        summary_rows.append({
            "outcome": col, **m,
            "pos_pat_mean": pos_pat_mean, "pos_pat_std": pos_pat_std,
            "pos_img_mean": pos_img_mean, "pos_img_std": pos_img_std,
            "mean_std": mean_std,
        })

        # ---- build markdown section ----
        tn, fp, fn, tp = m["cm"]
        sections.append(f"""## {col}

- **n val patients (pooled)**: {m['n']:,}  (positives: {m['n_pos']:,})
- **pos_patient(%)**: {pos_pat_mean*100:.2f} ± {pos_pat_std*100:.2f}  (per-fold mean ± std)
- **pos_image(%)**:   {pos_img_mean*100:.2f} ± {pos_img_std*100:.2f}  (per-fold mean ± std; n_val_images={len(val_imgs):,})

{metric_table}

**Confusion matrix** (threshold = {thr}):

|              | pred neg | pred pos |
|---|---|---|
| **true neg** | {tn} | {fp} |
| **true pos** | {fn} | {tp} |
""")

    # ---- summary table ----
    summary_md = [
        f"# {args.title}",
        "",
        f"Source: `{args.result}` · threshold = {thr} · pooled across 5 validation folds",
        "",
        "| outcome | AUC (%) | Acc (%) | Sen (%) | Spe (%) | PPV (%) | NPV (%) | F1 (%) | pos_pat(%) | pos_img(%) | n |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in summary_rows:
        ms = row["mean_std"]
        summary_md.append(
            f"| {row['outcome']} "
            f"| {ms['auc']} "
            f"| {ms['acc']} "
            f"| {ms['sen']} "
            f"| {ms['spe']} "
            f"| {ms['ppv']} "
            f"| {ms['npv']} "
            f"| {ms['f1']} "
            f"| {row['pos_pat_mean']*100:.2f} ± {row['pos_pat_std']*100:.2f} "
            f"| {row['pos_img_mean']*100:.2f} ± {row['pos_img_std']*100:.2f} "
            f"| {row['n']:,} |"
        )
    summary_md.append("")

    report_md = "\n".join(summary_md) + "\n\n" + "\n".join(sections) + "\n"
    Path(args.out).write_text(report_md)
    print(f"wrote {args.out}  ({len(report_md):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
