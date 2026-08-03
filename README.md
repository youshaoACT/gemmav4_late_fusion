# gemmav4 late fusion — Stage-2 image + clinical prediction

The model fuses two streams to predict **four binary clinical outcomes**:

- **Image stream** — 768-d per-image features pre-extracted from the
  SimMIM-pretrained Gemma 4 ViT (the "V1" encoder; see
  `scripts/ssl_core.py` and `scripts/cache_image_features.py`).
- **Clinical stream** — per-patient clinical variables
  (`con_val_1..con_val_44`, `age`, `sex_idx`).

Two cohorts are supported on the same architecture and same 5-fold
patient-aware CV:

| cohort | patient key | size |
|---|---|---|
| AKD | `病案号` | 1,324 patients · 6,172 images |
| Biopsy | `病理号` | 5,890 patients · 26,561 images |


## Layout

```
gemmav4_latefusion_2026-08-01/
├── scripts/   training, feature extraction, drift analysis, reporting
├── configs/   YAML configs (V1 SimMIM + Stage-2 late-fusion, AKD + biopsy)
├── data/      cohort parquets + 5-fold patient-aware splits
└── features/  pre-extracted 768-d per-image features (.npy + .parquet meta)
```

## Pipeline (end-to-end)

The full late-fusion result for either cohort is reproduced in three
steps.

### 1. Pre-extract 768-d per-image features

```bash
# AKD cohort (default)
python scripts/cache_image_features.py --encoder both

# Biopsy cohort
python scripts/cache_image_features.py --encoder both \
    --cohort data/stage2_cohort_biopsy.parquet \
    --folds  data/stage2_folds_biopsy.json \
    --features-prefix stage2_biopsy
```


### 2. Train the late-fusion head (5-fold patient-aware CV)

```bash
# AKD
python scripts/stage2_train.py \
    --config configs/stage2_m2_late_fusion.yaml \
    --out-json checkpoints/stage2_l2/eval.json

# Biopsy
python scripts/stage2_train.py \
    --config configs/stage2_biopsy_l2_late_fusion.yaml \
    --patient-key 病理号 \
    --out-json checkpoints/stage2_biopsy_l2/eval.json
```


### 3. Build the report

```bash
# Per-cohort report (AUC, Acc, Sen, Spe, PPV, NPV, F1, confusion matrix)
python scripts/build_report.py \
    --result checkpoints/stage2_l2/eval.json \
    --cohort data/stage2_cohort.parquet \
    --patient-key 病案号 \
    --out report.md

# Full vs clinical-only (and optional image-only) ablation comparison
python scripts/build_ablation_report.py \
    --full         <path/to/full.json> \
    --clinical-only <path/to/clinical_only.json> \
    --image-only   <path/to/image_only.json> \
    --out          ablation.md

# 5-fold loss-curve plot
python scripts/plot_loss_curves.py \
    --in  checkpoints/stage2_l2/loss_curves.json \
    --out loss_curves.png \
    --title "gemmav4 akd late-fusion (5 folds)"
```

## File reference

### `scripts/`

| File | Purpose |
|---|---|
| `ssl_core.py` | Shared Stage-1 utilities: `load_vision_tower` (loads the Gemma 4 E2B vision tower, disables QAT `Gemma4ClippableLinear` modules on ultrasound per Plan §2.3), `rename_position_ids` (renames the processor's `image_position_ids` → encoder's `pixel_position_ids`), `make_block_mask` (Plan §2.2 step 2 — tile-mask at `block_size` granularity over only valid patches), `SimMIM` module (mask_token + LayerNorm + zero-init linear decoder for the reconstruction target), and `reconstruction_loss` (L1/L2 over masked patches only). |
| `dataset.py` | `KidneyROIDataset` reads PNG paths from a list file, runs `Gemma4ImageProcessor`, returns `(pixel_values, pixel_position_ids, num_soft_tokens_per_image)` per image. `collate_padded` stacks a variable-N batch by padding `pixel_values` with 0 and `pixel_position_ids` with `(-1, -1)`. |
| `cache_image_features.py` | Runs both the teacher (frozen E2B) and the V1 student over a cohort, mean-pools valid patch tokens per image, writes `(N, 768)` float32 `.npy` plus a `_meta.parquet` sidecar (row_idx, image_name, patient_id, group_id, fold, full_path). Reads `max_soft_tokens` from `configs/v1_baseline.yaml` as the single source of truth. |
| `analyze_v1.py` | D1–D4 drift sanity checks vs the frozen teacher. D1 = base-vs-base (should be ~1.0). D2 = student-init vs teacher. D3 = base(masked) vs base(unmasked) — pure mask perturbation. D4 = **trained student vs teacher (unmasked)** — the number that matters for downstream OOD risk; emits a verdict band (>0.85 ship, 0.70–0.85 consider V2/V3, <0.70 distill). |
| `stage2_train.py` | **Main late-fusion trainer.** Builds per-fold patient-aware splits (`make_split`), preprocesses clinical (median-fill → winsorize 1/99 → standard scale, all stats from train fold only), trains `LateFusion` (image `768→256`, clinical `d_clin→64`, concat `320→MLP(128)→4`) with patient-mean pooling, BCE-with-logits + per-outcome `pos_weight`, patience on val-pat-AUC-mean. Writes `eval.json` (per-outcome fold AUCs, fold probs, summaries) and `loss_curves.json`. Patient key (`病案号` default; `病理号` for biopsy) is a CLI flag. |
| `stage2_train_image_only.py` | Image-only ablation of `stage2_train.py`: identical data, folds, pooling, optimizer, patience, pos_weight schedule, but the clinical branch is removed. Effective learnable model: `image 768 → 256 → MLP(128) → 4` with patient-mean pool between the 256 and 128 layers. Reuses `make_split` / `patient_logit_mean` from `stage2_train`. |
| `build_report.py` | Renders a per-outcome Markdown report from a `result.json`: pooled AUC/Acc/Sen/Spe/PPV/NPV/F1, per-fold metric table, confusion matrix, pos-rate stats. `--patient-key` selects which patient column the image-pos-rate breakdown uses. |
| `build_ablation_report.py` | 2-arm (full vs clinical-only) or 3-arm (+ image-only) ablation report: per-fold tables, pooled confusion matrices, side-by-side comparison with Δ in percentage points, and a dynamic interpretation section that tags each outcome as helps / marginal / neutral / hurts based on the actual Δ and per-fold consistency. |
| `plot_loss_curves.py` | Reads a `loss_curves.json` and renders a two-panel matplotlib figure (train vs val loss left, val pat-AUC-mean right), one color per fold. |

### `configs/`

| File | Purpose |
|---|---|
| `v1_baseline.yaml` | Stage-1 SimMIM training config: model path, `max_soft_tokens=70`, block mask (ratio 0.6, block 8×8), AdamW with LLRD, cosine schedule, 3 epochs, bf16, single 4090. Also the **single source of truth for `max_soft_tokens`** shared by `cache_image_features.py` and `ssl_core.py`. |
| `stage2_m2_late_fusion.yaml` | **Main L2 late-fusion config (AKD).** Image `768→256` (dropout 0.3), clinical `d_clin→64`, concat `320→MLP(128)→4`, 30 epochs, batch 64, lr 1e-3, wd 1e-4, seed 42, patience 5. |
| `stage2_m2_late_fusion_epoch100.yaml` | Same architecture as `m2` but with 100 epochs (longer-budget AKD run). |
| `stage2_l5_late_fusion_teacher.yaml` | **L5 ablation (AKD):** identical head + data + folds to `m2`, but image features come from the *frozen E2B teacher* (no SimMIM). This is the "SSL-in-fusion" probe. |
| `stage2_biopsy_l2_late_fusion.yaml` | L2 late-fusion on the biopsy cohort (`patient_key=病理号`). Same architecture as `m2`. |
| `stage2_biopsy_l5_late_fusion_teacher.yaml` | L5 biopsy: same as the L2 biopsy but on teacher features. |

### `data/`

| File | Purpose |
|---|---|
| `stage2_cohort.parquet` | AKD cohort: per-image rows with `病案号` (patient id), `pathoID_std` (pathology id used to build group ids), `image_name`, `full_path`, 4 `outcome*` labels, and `con_val_1..con_val_44` clinical features. |
| `stage2_cohort_biopsy.parquet` | Biopsy cohort: same schema but patient key is `病理号` (string) and the cohort is shipped with a pre-computed `group_id` column. |
| `stage2_folds.json` | 5-fold `StratifiedGroupKFold` split over AKD groups, versioned (`stage2-v1`). |
| `stage2_folds_biopsy.json` | Same, for the biopsy cohort. |

### `features/`

| File | Purpose |
|---|---|
| `teacher_stage2.npy` + `_meta.parquet` | Teacher (frozen E2B) 768-d mean-pooled features for AKD, row-aligned to `stage2_cohort.parquet`. |
| `v1_stage2.npy` + `_meta.parquet` | V1 (SimMIM) 768-d features for AKD. This is what the main late-fusion model uses. |
| `teacher_stage2_biopsy.npy` + `_meta.parquet` | Teacher features for biopsy. |
| `v1_stage2_biopsy.npy` + `_meta.parquet` | V1 features for biopsy. |

The `_meta.parquet` sidecar is the join key between the `.npy` row
order and the cohort parquet (and between cohorts and folds); the
training scripts assert `(meta["image_name"] == df["image_name"]).all()`
at load time.
