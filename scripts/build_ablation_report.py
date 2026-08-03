"""Build ablation report: full late-fusion vs clinical-only (and optional image-only).

For each result file, compute per-fold and mean metrics:
  - TP, FP, TN, FN, AUC, ACC, SEN, SPE, PPV, NPV
Plus a side-by-side comparison (2 or 3 arms).

Usage:
  # 2-arm (full vs clinical-only)
  python scripts/build_ablation_report.py \
      --full outputs/dinov2_biopsy/result.json \
      --clinical-only outputs/clinical_only.json \
      --out outputs/ablation_report.md

  # 3-arm (full vs clinical-only vs image-only)
  python scripts/build_ablation_report.py \
      --full outputs/dinov2_biopsy/result.json \
      --clinical-only outputs/clinical_only.json \
      --image-only outputs/biopsy_image_only.json \
      --out outputs/ablation_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

OUTCOME_COLS = ["outcome1", "outcome2", "outcome3", "outcome4"]


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    if y_true.min() == y_true.max():
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    return {
        "auc": auc, "acc": float(accuracy_score(y_true, y_pred)),
        "sen": float(sens), "spe": float(spec),
        "ppv": float(ppv), "npv": float(npv),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)), "n_pos": int(y_true.sum()),
    }


def _per_fold_metrics(result: dict, outcome: str, thr: float) -> list[dict]:
    fold_items = list(result["outcomes"][outcome]["fold_probs"].items())
    fold_items.sort(key=lambda kv: int(kv[0]))
    return [
        _metrics(
            np.asarray(labels).astype(int),
            np.asarray(probs),
            thr,
        )
        for _, (_, probs, labels) in fold_items
    ]


def _fmt(v: float, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}f}"


def _fmt_pct(v: float, digits: int = 1) -> str:
    """Proportion displayed as percent (0.762 → 76.2), no % sign."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v * 100:.{digits}f}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def _per_fold_table(folds: list[dict]) -> str:
    """Returns a markdown table with one row per fold and a mean row."""
    lines = [
        "| fold | TP | FP | TN | FN | AUC (%) | Acc (%) | Sen (%) | Spe (%) | PPV (%) | NPV (%) | n | n_pos |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(folds):
        lines.append(
            f"| {i} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} "
            f"| {_fmt_pct(m['auc'])} | {_fmt_pct(m['acc'])} | {_fmt_pct(m['sen'])} "
            f"| {_fmt_pct(m['spe'])} | {_fmt_pct(m['ppv'])} | {_fmt_pct(m['npv'])} "
            f"| {m['n']} | {m['n_pos']} |"
        )
    mean_tp = int(round(np.mean([m["tp"] for m in folds])))
    mean_fp = int(round(np.mean([m["fp"] for m in folds])))
    mean_tn = int(round(np.mean([m["tn"] for m in folds])))
    mean_fn = int(round(np.mean([m["fn"] for m in folds])))
    auc_mean, auc_std = _mean_std([m["auc"] for m in folds])
    acc_mean, acc_std = _mean_std([m["acc"] for m in folds])
    sen_mean, sen_std = _mean_std([m["sen"] for m in folds])
    spe_mean, spe_std = _mean_std([m["spe"] for m in folds])
    ppv_mean, ppv_std = _mean_std([m["ppv"] for m in folds])
    npv_mean, npv_std = _mean_std([m["npv"] for m in folds])
    lines.append(
        f"| **mean** | **{mean_tp}** | **{mean_fp}** | **{mean_tn}** | **{mean_fn}** "
        f"| **{_fmt_pct(auc_mean)} ± {_fmt_pct(auc_std)}** "
        f"| **{_fmt_pct(acc_mean)} ± {_fmt_pct(acc_std)}** "
        f"| **{_fmt_pct(sen_mean)} ± {_fmt_pct(sen_std)}** "
        f"| **{_fmt_pct(spe_mean)} ± {_fmt_pct(spe_std)}** "
        f"| **{_fmt_pct(ppv_mean)} ± {_fmt_pct(ppv_std)}** "
        f"| **{_fmt_pct(npv_mean)} ± {_fmt_pct(npv_std)}** "
        f"| — | — |"
    )
    return "\n".join(lines)


def _pooled_cm(folds: list[dict]) -> str:
    pooled_tp = sum(m["tp"] for m in folds)
    pooled_fp = sum(m["fp"] for m in folds)
    pooled_tn = sum(m["tn"] for m in folds)
    pooled_fn = sum(m["fn"] for m in folds)
    return (
        "|              | pred neg | pred pos |\n"
        "|---|---|---|\n"
        f"| **true neg** | {pooled_tn} | {pooled_fp} |\n"
        f"| **true pos** | {pooled_fn} | {pooled_tp} |"
    )


def _study_section(title: str, result: dict, thr: float) -> str:
    parts = [f"## {title}\n", f"Source: `{result['run']}` · threshold = {thr}\n"]
    for col in OUTCOME_COLS:
        folds = _per_fold_metrics(result, col, thr)
        parts.append(f"### {col}\n")
        parts.append(_per_fold_table(folds))
        parts.append("")
        parts.append("**Pooled confusion matrix** (sum of TP/FP/TN/FN across the 5 folds):\n")
        parts.append(_pooled_cm(folds))
        parts.append("")
    return "\n".join(parts)


def _comparison_section(full: dict, clinical: dict, thr: float) -> str:
    parts = [
        "## Comparison: full vs clinical-only (mean per-fold metrics)\n",
        "Each cell is `full (mean ± std) / clinical-only (mean ± std) / Δ (full − clinical)`.\n",
        "Positive Δ = full is better.  All metrics are reported in percent (Δ in pp).\n",
        "",
        "| outcome | AUC (%) | Acc (%) | Sen (%) | Spe (%) | PPV (%) | NPV (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for col in OUTCOME_COLS:
        full_folds = _per_fold_metrics(full, col, thr)
        clin_folds = _per_fold_metrics(clinical, col, thr)
        cells = []
        for key in ("auc", "acc", "sen", "spe", "ppv", "npv"):
            f_mean, f_std = _mean_std([m[key] for m in full_folds])
            c_mean, c_std = _mean_std([m[key] for m in clin_folds])
            delta = f_mean - c_mean
            cell = f"{_fmt_pct(f_mean)} ± {_fmt_pct(f_std)} / {_fmt_pct(c_mean)} ± {_fmt_pct(c_std)} / **{delta * 100:+.2f}**"
            cells.append(cell)
        parts.append(f"| {col} | " + " | ".join(cells) + " |")
    parts.append("")
    return "\n".join(parts)


def _auc_summary(full: dict, clinical: dict) -> str:
    parts = ["## AUC summary (mean over folds)\n", ""]
    parts.append("| outcome | full AUC (%) | clinical-only AUC (%) | Δ (pp) |")
    parts.append("|---|---|---|---|")
    for col in OUTCOME_COLS:
        f = float(np.nanmean(full["outcomes"][col]["fold_aucs"]))
        c = float(np.nanmean(clinical["outcomes"][col]["fold_aucs"]))
        parts.append(f"| {col} | {_fmt_pct(f)} | {_fmt_pct(c)} | **{(f-c)*100:+.2f}** |")
    f_overall = float(full["summary_auc_mean_overall"])
    c_overall = float(clinical["summary_auc_mean_overall"])
    parts.append(f"| **OVERALL** | **{_fmt_pct(f_overall)}** | **{_fmt_pct(c_overall)}** | **{(f_overall-c_overall)*100:+.2f}** |")
    parts.append("")
    return "\n".join(parts)


def _auc_summary_3arm(full: dict, clinical: dict, image: dict) -> str:
    parts = ["## AUC summary (mean over folds, 3-arm)\n", ""]
    parts.append("| outcome | full AUC (%) | clinical-only AUC (%) | image-only AUC (%) | Δ(f-c) (pp) | Δ(f-i) (pp) | Δ(c-i) (pp) |")
    parts.append("|---|---|---|---|---|---|---|")
    for col in OUTCOME_COLS:
        f = float(np.nanmean(full["outcomes"][col]["fold_aucs"]))
        c = float(np.nanmean(clinical["outcomes"][col]["fold_aucs"]))
        i = float(np.nanmean(image["outcomes"][col]["fold_aucs"]))
        parts.append(
            f"| {col} "
            f"| {_fmt_pct(f)} | {_fmt_pct(c)} | {_fmt_pct(i)} "
            f"| **{(f-c)*100:+.2f}** | **{(f-i)*100:+.2f}** | **{(c-i)*100:+.2f}** |"
        )
    f_o = float(full["summary_auc_mean_overall"])
    c_o = float(clinical["summary_auc_mean_overall"])
    i_o = float(image["summary_auc_mean_overall"])
    parts.append(
        f"| **OVERALL** "
        f"| **{_fmt_pct(f_o)}** | **{_fmt_pct(c_o)}** | **{_fmt_pct(i_o)}** "
        f"| **{(f_o-c_o)*100:+.2f}** | **{(f_o-i_o)*100:+.2f}** | **{(c_o-i_o)*100:+.2f}** |"
    )
    parts.append("")
    return "\n".join(parts)


def _comparison_section_3arm(full: dict, clinical: dict, image: dict, thr: float) -> str:
    """3-arm comparison: rows = (outcome, metric), columns = each arm + pairwise deltas."""
    metric_labels = [("auc", "AUC"), ("acc", "Acc"), ("sen", "Sen"),
                      ("spe", "Spe"), ("ppv", "PPV"), ("npv", "NPV")]
    parts = [
        "## Comparison: full vs clinical-only vs image-only (mean per-fold metrics)",
        "",
        "Each cell is `<arm> (mean ± std)`.  Δ values are in percentage points (pp).",
        "Positive Δ = first arm is better.",
        "",
        "| outcome | metric | full (mean ± std) | clinical-only (mean ± std) | image-only (mean ± std) | Δ(f-c) (pp) | Δ(f-i) (pp) | Δ(c-i) (pp) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for col in OUTCOME_COLS:
        full_folds = _per_fold_metrics(full, col, thr)
        clin_folds = _per_fold_metrics(clinical, col, thr)
        image_folds = _per_fold_metrics(image, col, thr)
        for key, label in metric_labels:
            f_mean, f_std = _mean_std([m[key] for m in full_folds])
            c_mean, c_std = _mean_std([m[key] for m in clin_folds])
            i_mean, i_std = _mean_std([m[key] for m in image_folds])
            df_c = f_mean - c_mean
            df_i = f_mean - i_mean
            dc_i = c_mean - i_mean
            parts.append(
                f"| {col} | {label} "
                f"| {_fmt_pct(f_mean)} ± {_fmt_pct(f_std)} "
                f"| {_fmt_pct(c_mean)} ± {_fmt_pct(c_std)} "
                f"| {_fmt_pct(i_mean)} ± {_fmt_pct(i_std)} "
                f"| **{df_c * 100:+.2f}** "
                f"| **{df_i * 100:+.2f}** "
                f"| **{dc_i * 100:+.2f}** |"
            )
    parts.append("")
    return "\n".join(parts)


def _dynamic_interpretation_2arm(full: dict, clinical: dict) -> str:
    """2-arm interpretation: derived from actual deltas + per-fold consistency."""
    rows = []
    for col in OUTCOME_COLS:
        f_folds = full["outcomes"][col]["fold_aucs"]
        c_folds = clinical["outcomes"][col]["fold_aucs"]
        f_mean = float(np.nanmean(f_folds))
        c_mean = float(np.nanmean(c_folds))
        delta = f_mean - c_mean
        pos_folds = sum(1 for ff, cc in zip(f_folds, c_folds) if ff > cc)
        rows.append({"col": col, "delta": delta, "f_mean": f_mean,
                     "c_mean": c_mean, "pos_folds": pos_folds})

    helps = [r for r in rows if r["delta"] >= 0.01]
    marginal = [r for r in rows if 0.005 <= r["delta"] < 0.01]
    neutral = [r for r in rows if -0.005 < r["delta"] < 0.005]
    hurts = [r for r in rows if r["delta"] <= -0.005]

    parts = []
    if helps:
        helps_str = ", ".join(
            f"**{r['col']}** (Δ {r['delta']*100:+.1f} pp, positive in {r['pos_folds']}/5 folds)"
            for r in helps
        )
        parts.append(f"Image features contribute meaningfully to {helps_str}.")
    if marginal:
        marginal_str = ", ".join(
            f"**{r['col']}** (Δ {r['delta']*100:+.1f} pp, positive in {r['pos_folds']}/5 folds)"
            for r in marginal
        )
        parts.append(f"For {marginal_str}, the deltas are smaller and per-fold std is comparable to the mean, so the effect is suggestive but not confirmed.")
    if neutral:
        neutral_str = ", ".join(
            f"**{r['col']}** (Δ {r['delta']*100:+.1f} pp)"
            for r in neutral
        )
        parts.append(f"For {neutral_str}, the delta is within fold-level noise — image features do not help.")
    if hurts:
        hurts_str = ", ".join(
            f"**{r['col']}** (Δ {r['delta']*100:+.1f} pp)"
            for r in hurts
        )
        parts.append(f"For {hurts_str}, image features hurt fusion.")
    if not parts:
        parts.append("Image features have minimal effect on all outcomes.")
    parts.append("")
    parts.append("**Caveat**: this is a single seed.  Per-fold std on the smaller deltas "
                "is wide enough that a multi-seed run is the right next step before drawing "
                "strong conclusions.")
    return "\n".join(parts)


def _dynamic_interpretation_3arm(full: dict, clinical: dict, image: dict) -> str:
    """3-arm interpretation: derived from actual deltas + clinical-vs-image gap."""
    rows = []
    for col in OUTCOME_COLS:
        f_mean = float(np.nanmean(full["outcomes"][col]["fold_aucs"]))
        c_mean = float(np.nanmean(clinical["outcomes"][col]["fold_aucs"]))
        i_mean = float(np.nanmean(image["outcomes"][col]["fold_aucs"]))
        rows.append({
            "col": col, "f": f_mean, "c": c_mean, "i": i_mean,
            "df_c": f_mean - c_mean, "df_i": f_mean - i_mean, "dc_i": c_mean - i_mean,
        })

    parts = ["### Per-outcome findings (3-arm)", ""]
    for r in rows:
        if r["df_c"] >= 0.01:
            tag = f"image adds **{r['df_c']*100:+.1f} pp** to clinical in fusion"
        elif r["df_c"] >= 0.005:
            tag = f"image adds **{r['df_c']*100:+.1f} pp** (marginal)"
        elif r["df_c"] <= -0.005:
            tag = f"image *hurts* fusion by **{r['df_c']*100:+.1f} pp**"
        else:
            tag = f"no fusion effect ({r['df_c']*100:+.1f} pp)"
        parts.append(
            f"- **{r['col']}**: {tag}.  clinical-only AUC {r['c']:.3f}, "
            f"image-only AUC {r['i']:.3f} (Δ {r['dc_i']*100:+.1f} pp)."
        )
    parts.append("")

    big_clin_wins = sum(1 for r in rows if r["dc_i"] > 0.05)
    best_fusion = max(rows, key=lambda r: r["df_c"])
    worst_image = min(rows, key=lambda r: r["i"])

    if big_clin_wins >= 3:
        parts.append(
            f"**Overall pattern**: clinical features carry most of the signal — clinical beats image "
            f"by >5 pp on {big_clin_wins}/4 outcomes. Image features add the most fusion value on "
            f"**{best_fusion['col']}** ({best_fusion['df_c']*100:+.1f} pp gain). Image-only is weakest "
            f"on **{worst_image['col']}** (AUC {worst_image['i']:.3f}, vs clinical {worst_image['c']:.3f})."
        )
    else:
        parts.append("**Overall pattern**: see per-outcome findings above.")
    parts.append("")
    parts.append("**Caveat**: this is a single seed.  Per-fold std on the smaller deltas "
                "is wide enough that a multi-seed run is the right next step before drawing "
                "strong conclusions.")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="path to full model result.json")
    ap.add_argument("--clinical-only", required=True, help="path to clinical-only result.json")
    ap.add_argument("--image-only", help="optional path to image-only result.json")
    ap.add_argument("--out", required=True, help="path to output report.md")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cohort", help="optional cohort name for the report title (e.g. 'Biopsy', 'AKD')")
    args = ap.parse_args()

    full = json.load(open(args.full))
    clin = json.load(open(args.clinical_only))
    image = json.load(open(args.image_only)) if args.image_only else None
    thr = args.threshold

    cohort_label = args.cohort or ""
    title = "# Late-fusion ablation report"
    if cohort_label:
        title += f" ({cohort_label})"

    sources_str = f"`{args.full}`, `{args.clinical_only}`"
    if image:
        sources_str += f", `{args.image_only}`"

    n_arms = 3 if image is not None else 2

    if n_arms == 3:
        intro = (
            "Comparison of the full DINOv2 late-fusion model against a clinical-only "
            "ablation (image slot held at zero) and an image-only ablation (clinical branch "
            "removed).  All runs use the same 5-fold patient-aware CV, same seed, same head "
            "architecture — the only difference is which branches are exercised.\n\n"
            "- **Full**: 768-d DINOv2 → 256 → ReLU → Dropout (patient-mean) ‖ clinical → 64 → ReLU → Dropout → concat → MLP(128) → 4\n"
            "- **Clinical-only**: same head; image slot fixed to zero.  Effective learnable model: `clinical d → 64 → 128 → 4`\n"
            "- **Image-only**: image branch only.  Effective learnable model: `image 768 → 256 → 128 → 4` (patient-mean between 256 and 128)\n"
        )
    else:
        intro = (
            "Comparison of the full DINOv2 late-fusion model against a clinical-only "
            "ablation where the 256-d image slot in the MLP head is held at zero. "
            "Both runs use the same 5-fold patient-aware CV, same seed, same head "
            "architecture — the only difference is whether the image branch is "
            "exercised or held at zero.\n\n"
            "- **Full**: 768-d DINOv2 → 256 → ReLU → Dropout (patient-mean over per-image "
            "embeddings) ‖ clinical d → 64 → ReLU → Dropout → concat(320) → MLP(128) → 4\n"
            "- **Clinical-only**: same head; image slot fixed to zero.  Effective learnable "
            "model is `clinical d → 64 → 128 → 4`\n"
        )

    sections = [
        title,
        "",
        intro,
        f"- 5-fold patient-aware CV, AdamW, BCE-with-logits + per-outcome pos_weight, patience 5 on val-pat-AUC-mean",
        f"- Threshold for ACC/SEN/SPE/PPV/NPV: {thr} (fixed across folds)",
        f"- Sources: {sources_str}",
        "",
    ]

    if n_arms == 3:
        sections.append(_auc_summary_3arm(full, clin, image))
    else:
        sections.append(_auc_summary(full, clin))

    sections.append(_study_section("Full late-fusion model (DINOv2 + clinical)", full, thr))
    sections.append(_study_section("Clinical-only ablation", clin, thr))
    if image is not None:
        sections.append(_study_section("Image-only ablation", image, thr))

    if n_arms == 3:
        sections.append(_comparison_section_3arm(full, clin, image, thr))
    else:
        sections.append(_comparison_section(full, clin, thr))

    sections.append("## Interpretation")
    sections.append("")
    if n_arms == 3:
        sections.append(_dynamic_interpretation_3arm(full, clin, image))
    else:
        sections.append(_dynamic_interpretation_2arm(full, clin))
    sections.append("")

    report = "\n".join(sections)
    Path(args.out).write_text(report)
    print(f"wrote {args.out}  ({len(report):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
