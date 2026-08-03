"""Image-only ablation of the late-fusion model.

Mirrors `scripts/stage2_train.py` exactly (same data, folds, patient-aware
pooling, BCE-with-logits, per-outcome pos_weight, AdamW, patience on
val-pat-AUC-mean, seed schedule) but the clinical branch is removed.
Only the image branch is exercised: per-image 768-d Gemmav4 V1 features
→ 256-d projection (ReLU, Dropout) → patient-mean pool → MLP(128) → 4
outcomes.

Effective learnable model: `image 768 → 256 → 128 → 4` (patient-mean
pool between the 256-d projection and the 128-d head layer).

Adapted from /home/tle/dinov2_latefusion_2026-08-01/scripts/stage2_train_image_only.py
with two changes:
  - Imports `make_split`, `patient_logit_mean` from gemmav4's stage2_train.
  - Cohort ↔ features alignment uses row-order assert (Gemmav4 features
    are already aligned by row to the cohort parquet) instead of
    dinov2's `(image_name, patho_id)` join.

Usage:
  python scripts/stage2_train_image_only.py \
      --config configs/stage2_m2_late_fusion.yaml \
      --out-json outputs/gemmav4_akd/image_only.json
  python scripts/stage2_train_image_only.py \
      --config configs/stage2_biopsy_l2_late_fusion.yaml \
      --patient-key 病理号 \
      --out-json outputs/gemmav4_biopsy/image_only.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

from stage2_train import (
    OUTCOME_COLS,
    DEFAULT_PATIENT_KEY,
    make_split,
    patient_logit_mean,
    _json_default,
)


# ===========================================================================
# Image-only model
# ===========================================================================
class ImageOnly(nn.Module):
    """Per-image projection → patient-mean pool → MLP head.

    Drops the clinical branch entirely.  Architecturally equivalent to
    `LateFusion` with the clinical projection removed and the head's
    input dim reduced from `image_proj + clin_proj` to `image_proj`.
    """
    def __init__(self, image_in: int = 768, image_proj: int = 256,
                 mlp_hidden: int = 128, dropout: float = 0.3, n_out: int = 4):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_in, image_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(image_proj, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_out),
        )

    def forward(self, x_img_flat: torch.Tensor, pat_idx: torch.Tensor,
                n_patients: int) -> torch.Tensor:
        e_img = self.image_proj(x_img_flat)
        e_img_pat = patient_logit_mean(e_img, pat_idx, n_patients)
        return self.head(e_img_pat)


# ===========================================================================
# Image-only data loader (no clinical features)
# ===========================================================================
class _PatientImageDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, rows_by_patient, x_img, labels):
        self.patient_ids = list(patient_ids)
        self.rows_by_patient = rows_by_patient
        self.x_img = x_img
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        rows = self.rows_by_patient[patient_id]
        return {
            "x_img_flat": torch.from_numpy(self.x_img[rows]),
            "y": self.labels[idx],
        }


def _collate_image_only(batch):
    n_p = len(batch)
    flat_imgs = [item["x_img_flat"] for item in batch]
    sizes = [images.shape[0] for images in flat_imgs]
    return {
        "n_patients": n_p,
        "x_img_flat": torch.cat(flat_imgs, dim=0),
        "pat_idx": torch.cat(
            [torch.full((size,), idx, dtype=torch.long)
             for idx, size in enumerate(sizes)]
        ),
        "y": torch.stack([item["y"] for item in batch], dim=0),
    }


def _make_image_only_loaders(split, x_img, batch_size):
    train_dataset = _PatientImageDataset(
        split.train_pat_ids, split.train_rows_by_pat, x_img, split.train_pat_y,
    )
    val_dataset = _PatientImageDataset(
        split.val_pat_ids, split.val_rows_by_pat, x_img, split.val_pat_y,
    )
    return (
        torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, collate_fn=_collate_image_only,
        ),
        torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=_collate_image_only,
        ),
    )


# ===========================================================================
# Per-fold training loop (image-only)
# ===========================================================================
def _train_per_fold_image_only(model, train_loader, val_loader, optim, cfg,
                               device, fold: int, pos_weight: torch.Tensor):
    best_auc = -1.0
    best_state = None
    bad = 0
    loss_hist, val_loss_hist, val_auc_hist = [], [], []
    for ep in range(cfg.train.epochs):
        model.train()
        ep_loss_sum, ep_batches = 0.0, 0
        for batch in train_loader:
            optim.zero_grad()
            x_img = batch["x_img_flat"].to(device)
            pat_idx = batch["pat_idx"].to(device)
            n_patients = batch["n_patients"]
            y = batch["y"].to(device)
            logits = model(x_img, pat_idx, n_patients)
            loss = sum(
                F.binary_cross_entropy_with_logits(
                    logits[:, j], y[:, j].float(), pos_weight=pos_weight[j]
                )
                for j in range(4)
            ) / 4
            loss.backward()
            optim.step()
            ep_loss_sum += float(loss.detach())
            ep_batches += 1
        loss_hist.append(ep_loss_sum / max(1, ep_batches))

        model.eval()
        with torch.no_grad():
            all_probs, all_y = [], []
            val_loss_sum, val_batches = 0.0, 0
            for batch in val_loader:
                x_img = batch["x_img_flat"].to(device)
                pat_idx = batch["pat_idx"].to(device)
                n_patients = batch["n_patients"]
                y = batch["y"].to(device)
                logits = model(x_img, pat_idx, n_patients)
                val_loss = sum(
                    F.binary_cross_entropy_with_logits(
                        logits[:, j], y[:, j].float(), pos_weight=pos_weight[j]
                    )
                    for j in range(4)
                ) / 4
                val_loss_sum += float(val_loss)
                val_batches += 1
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(batch["y"].numpy())
            probs = np.concatenate(all_probs, axis=0)
            labels = np.concatenate(all_y, axis=0)
            aucs = [
                roc_auc_score(labels[:, j], probs[:, j])
                for j in range(4)
                if labels[:, j].min() != labels[:, j].max()
            ]
            auc_mean = float(np.mean(aucs)) if aucs else float("nan")
        val_loss_hist.append(val_loss_sum / max(1, val_batches))
        val_auc_hist.append(auc_mean)
        if auc_mean > best_auc + 1e-5:
            best_auc = auc_mean
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1
            if bad >= cfg.train.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_auc, {
        "train_loss": loss_hist,
        "val_loss": val_loss_hist,
        "val_auc": val_auc_hist,
    }


# ===========================================================================
# Main
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--patient-key",
        default=DEFAULT_PATIENT_KEY,
        help=f"Patient-id column name (default: {DEFAULT_PATIENT_KEY})",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if cfg.model.kind not in ("late_fusion", "image_only"):
        raise SystemExit(
            f"only model.kind in (late_fusion, image_only) is supported, "
            f"got {cfg.model.kind}"
        )
    seed = int(cfg.train.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(cfg.data.cohort)
    if "group_id" not in df.columns:
        if "病案号" not in df.columns or "pathoID_std" not in df.columns:
            raise SystemExit("cohort has no group_id and no 病案号/pathoID_std to construct it from")
        group_per_patient = (
            df.groupby("病案号")["pathoID_std"]
            .agg(lambda values: "||".join(sorted({str(value) for value in values})))
        )
        df = df.merge(
            group_per_patient.rename("group_id").reset_index(),
            on="病案号", how="left",
        )
    else:
        df["group_id"] = df["group_id"].astype(str)
    folds = json.load(open(cfg.data.folds))
    print(f"[image_only] {cfg.model.run}: image-only ablation; "
          f"patient_key={args.patient_key}")

    x_img = np.load(cfg.model.features_npy).astype(np.float32)
    meta = pd.read_parquet(cfg.model.meta_parquet)
    print(f"[image_only] loaded features {x_img.shape} (dim {x_img.shape[1]})")
    assert (meta["image_name"].values == df["image_name"].values).all(), \
        "feature meta out of sync with cohort"
    print(f"[image_only] cohort/features row-aligned at {len(df)} rows")

    # make_split requires a non-empty clin_cols (it builds a per-patient
    # clinical matrix we never read).  Pick any numeric-ish column.
    placeholder_clin = [c for c in df.columns
                        if c.startswith("con_val_")][:1]
    if not placeholder_clin:
        placeholder_clin = [c for c in df.columns
                            if c not in {"group_id", "病理号", "病案号",
                                         "image_name", "Unnamed: 0",
                                         "pathoID_std"}
                            and not c.startswith("outcome")][:1]
    assert placeholder_clin, "no placeholder clinical column found"

    splits = [
        make_split(df, fold, placeholder_clin, patient_key=args.patient_key)
        for fold in folds["folds"]
    ]

    per_outcome = {column: {"fold_aucs": [], "fold_probs": {}} for column in OUTCOME_COLS}
    loss_curves = {}
    for split in splits:
        torch.manual_seed(seed + split.fold)
        pos_weight = []
        for outcome_idx in range(4):
            n_pos = max(1, int(split.train_pat_y[:, outcome_idx].sum()))
            n_neg = max(1, len(split.train_pat_y) - n_pos)
            pos_weight.append(n_neg / n_pos)
        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32)

        model = ImageOnly(
            image_in=int(x_img.shape[1]),
            image_proj=cfg.model.image_proj_dim,
            mlp_hidden=cfg.model.fusion_mlp_hidden,
            dropout=cfg.model.dropout,
            n_out=4,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
        )
        train_loader, val_loader = _make_image_only_loaders(
            split, x_img, cfg.train.batch_size,
        )
        model, best_auc, history = _train_per_fold_image_only(
            model, train_loader, val_loader, optimizer, cfg, device,
            split.fold, pos_weight_t,
        )
        loss_curves[str(split.fold)] = history
        print(f"  fold {split.fold}: best val-pat-AUC-mean = {best_auc:.4f}")

        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["x_img_flat"].to(device)
                pat_idx = batch["pat_idx"].to(device)
                logits = model(images, pat_idx, batch["n_patients"])
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(batch["y"].numpy())
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_y, axis=0)
        for outcome_idx, column in enumerate(OUTCOME_COLS):
            auc = (
                float(roc_auc_score(labels[:, outcome_idx], probs[:, outcome_idx]))
                if labels[:, outcome_idx].min() != labels[:, outcome_idx].max()
                else float("nan")
            )
            per_outcome[column]["fold_aucs"].append(auc)
            per_outcome[column]["fold_probs"][split.fold] = (
                split.val_pat_ids, probs[:, outcome_idx], labels[:, outcome_idx],
            )

    output_dir = Path(args.out_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "run": cfg.model.run + "_image_only",
        "kind": "image_only_ablation",
        "config": str(args.config),
        "features": str(cfg.model.features_npy),
        "n_splits": folds["n_splits"],
        "cohort_rows": int(len(df)),
        "n_patients": int(df[args.patient_key].nunique()),
        "outcomes": per_outcome,
        "summary_auc_mean_per_outcome": {
            column: float(np.nanmean(per_outcome[column]["fold_aucs"]))
            for column in OUTCOME_COLS
        },
        "summary_auc_mean_overall": float(
            np.nanmean([
                np.nanmean(per_outcome[column]["fold_aucs"])
                for column in OUTCOME_COLS
            ])
        ),
    }
    Path(args.out_json).write_text(json.dumps(output, indent=2, default=_json_default))
    curve_path = Path(args.out_json).with_suffix(".loss_curves.json")
    curve_path.write_text(json.dumps(loss_curves, indent=2))
    print(f"\n[image_only] {output['run']} headline (mean AUC over folds):")
    for column in OUTCOME_COLS:
        aucs = per_outcome[column]["fold_aucs"]
        print(f"  {column}: {np.nanmean(aucs):.4f}  "
              f"(folds={[f'{auc:.3f}' for auc in aucs]})")
    print(f"[image_only] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
