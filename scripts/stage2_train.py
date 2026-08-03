"""Train the L2 late-fusion model on cached V1 image features and clinical data."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score


OUTCOME_COLS = ["outcome1", "outcome2", "outcome3", "outcome4"]
# Cohort-agnostic patient key. AKD cohort uses 病案号 (int → str'd).
# Biopsy cohort uses 病理号 (string already). Override via --patient-key CLI.
DEFAULT_PATIENT_KEY = "病案号"


# ===========================================================================
# Patient aggregation & folds
# ===========================================================================
@dataclass
class CohortSplit:
    """Per-fold patient-aware view of the Stage-2 cohort."""
    fold: int
    # per-image row indices into the cohort DataFrame, grouped by patient
    train_rows_by_pat: dict[str, np.ndarray]
    val_rows_by_pat:   dict[str, np.ndarray]
    # patient-level clinical matrix & label matrix
    train_pat_clin: np.ndarray                    # (n_train_pat, d_clin) float32
    val_pat_clin:   np.ndarray
    train_pat_y:    np.ndarray                    # (n_train_pat, 4) int
    val_pat_y:      np.ndarray
    train_pat_ids:  list[str]
    val_pat_ids:    list[str]


def make_split(df: pd.DataFrame, fold_meta: dict, clin_cols: list[str],
               patient_key: str = DEFAULT_PATIENT_KEY) -> CohortSplit:
    """Build per-fold arrays. fold_meta is one entry of folds['folds'].
    All patient identifiers are strings (int 病案号 → str; 病理号 already str).
    Patient key is parameterized via `patient_key` so the biopsy cohort (which
    uses 病理号) can reuse this function unchanged."""
    train_groups = set(fold_meta["train_groups_str"])
    val_groups   = set(fold_meta["val_groups_str"])
    train_mask = df["group_id"].astype(str).apply(lambda g: g in train_groups).values
    val_mask   = df["group_id"].astype(str).apply(lambda g: g in val_groups).values

    # Patient IDs as strings (works for both int 病案号 and string 病理号)
    train_pats = sorted(str(p) for p in df.loc[train_mask, patient_key].unique().tolist())
    val_pats   = sorted(str(p) for p in df.loc[val_mask,   patient_key].unique().tolist())

    # patient → array of row indices (string-keyed lookup)
    train_rows_by_pat = {pid: np.where((df[patient_key].astype(str) == pid).values)[0]
                         for pid in train_pats}
    val_rows_by_pat   = {pid: np.where((df[patient_key].astype(str) == pid).values)[0]
                         for pid in val_pats}

    # patient-level clinical: take first row's clinical (it's identical within a patient)
    # Reindex with str-typed patient ids (works for both int and string keys).
    # Group by the string-cast key so the index matches the str-typed *_pats lists.
    key_str = df[patient_key].astype(str)
    train_pat_clin = (df.loc[train_mask].groupby(key_str[train_mask])[clin_cols].first()
                        .reindex(train_pats).values.astype(np.float32))
    val_pat_clin   = (df.loc[val_mask].groupby(key_str[val_mask])[clin_cols].first()
                        .reindex(val_pats).values.astype(np.float32))
    train_pat_y = (df.loc[train_mask].groupby(key_str[train_mask])[OUTCOME_COLS].first()
                     .reindex(train_pats).values.astype(np.float32)).astype(np.int64)
    val_pat_y   = (df.loc[val_mask].groupby(key_str[val_mask])[OUTCOME_COLS].first()
                     .reindex(val_pats).values.astype(np.float32)).astype(np.int64)

    return CohortSplit(
        fold=fold_meta["fold"],
        train_rows_by_pat=train_rows_by_pat, val_rows_by_pat=val_rows_by_pat,
        train_pat_clin=train_pat_clin, val_pat_clin=val_pat_clin,
        train_pat_y=train_pat_y, val_pat_y=val_pat_y,
        train_pat_ids=train_pats, val_pat_ids=val_pats,
    )


# ===========================================================================
# Clinical preprocessing (per-fold, no leakage)
# ===========================================================================
def prepare_clinical(clin_train: np.ndarray, clin_val: np.ndarray,
                     medians: np.ndarray | None = None,
                     scaler_mean: np.ndarray | None = None,
                     scaler_std:  np.ndarray | None = None):
    """median-fill missing, winsorize 1/99, StandardScaler. Returns
    (X_tr, X_va, medians, scaler_mean, scaler_std). If stats provided,
    reuses them (for inference)."""
    medians = medians if medians is not None else np.nanmedian(clin_train, axis=0)
    Xtr = np.where(np.isnan(clin_train), medians, clin_train)
    Xva = np.where(np.isnan(clin_val),   medians, clin_val)
    # winsorize 1/99 from train fold
    if scaler_mean is None or scaler_std is None:
        lo = np.percentile(Xtr, 1, axis=0); hi = np.percentile(Xtr, 99, axis=0)
        Xtr = np.clip(Xtr, lo, hi); Xva = np.clip(Xva, lo, hi)
        scaler_mean = Xtr.mean(0); scaler_std = Xtr.std(0); scaler_std[scaler_std == 0] = 1
    Xtr = (Xtr - scaler_mean) / scaler_std
    Xva = (Xva - scaler_mean) / scaler_std
    return Xtr.astype(np.float32), Xva.astype(np.float32), medians, scaler_mean, scaler_std


# ===========================================================================
# Late-fusion model
# ===========================================================================
class LateFusion(nn.Module):
    """Image per-image → 256 → ReLU → Dropout, patient-mean; clinical →
    64 → ReLU → Dropout; concat → MLP(320→128→4)."""
    def __init__(self, image_in=768, image_proj=256,
                 clin_in=None, clin_proj=64, mlp_hidden=128, dropout=0.3, n_out=4):
        super().__init__()
        self.image_proj = nn.Sequential(
            nn.Linear(image_in, image_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.clin_proj = nn.Sequential(
            nn.Linear(clin_in, clin_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(image_proj + clin_proj, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_out),
        )

# ===========================================================================
# Patient-level logit-mean helpers (Plan §S6 collate, fixed-batch variant)
# ===========================================================================
def patient_logit_mean(logits_flat: torch.Tensor, pat_idx: torch.Tensor,
                       n_patients: int) -> torch.Tensor:
    """scatter_mean over per-image logits, returning (B, n_out)."""
    out = torch.zeros(n_patients, logits_flat.shape[-1],
                      device=logits_flat.device, dtype=logits_flat.dtype)
    out.index_add_(0, pat_idx, logits_flat)
    counts = torch.zeros(n_patients, device=logits_flat.device, dtype=logits_flat.dtype)
    counts.index_add_(0, pat_idx, torch.ones_like(pat_idx, dtype=logits_flat.dtype))
    return out / counts.clamp(min=1).unsqueeze(-1)


# ===========================================================================
# Per-fold training loop
# ===========================================================================
def train_torch_per_fold(model, train_loader, val_loader, optim, cfg,
                         device, fold: int):
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
            x_clin = batch["x_clin"].to(device)
            y = batch["y"].to(device)
            e_img = model.image_proj(x_img)
            e_img_pat = patient_logit_mean(e_img, pat_idx, n_patients)
            logits = model.head(torch.cat([e_img_pat, model.clin_proj(x_clin)], dim=-1))
            pos_weight = batch["pos_weight"].to(device)
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
                x_clin = batch["x_clin"].to(device)
                y = batch["y"].to(device)
                e_img = model.image_proj(x_img)
                e_img_pat = patient_logit_mean(e_img, pat_idx, n_patients)
                logits = model.head(torch.cat([e_img_pat, model.clin_proj(x_clin)], dim=-1))
                pos_weight = batch["pos_weight"].to(device)
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
        help=f"Patient-id column name (default: {DEFAULT_PATIENT_KEY}; use 病理号 for biopsy)",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if cfg.model.kind != "late_fusion":
        raise SystemExit(f"only model.kind=late_fusion is supported, got {cfg.model.kind}")
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
            on="病案号",
            how="left",
        )
    else:
        df["group_id"] = df["group_id"].astype(str)
    folds = json.load(open(cfg.data.folds))
    clin_cols = [
        column
        for column in df.columns
        if (column.startswith("con_val_") or column in ("age", "sex_idx"))
        and column != "con_val"
    ]
    assert clin_cols, "no clinical-feature columns found"
    print(
        f"[stage2_train] {cfg.model.run}: late_fusion; "
        f"d_clin={len(clin_cols)}; patient_key={args.patient_key}"
    )

    splits = [
        make_split(df, fold, clin_cols, patient_key=args.patient_key)
        for fold in folds["folds"]
    ]
    x_img = np.load(cfg.model.features_npy).astype(np.float32)
    meta = pd.read_parquet(cfg.model.meta_parquet)
    assert (meta["image_name"].values == df["image_name"].values).all(),         "feature meta out of sync with cohort"
    print(f"[stage2_train] loaded V1 features {x_img.shape}")

    per_outcome = {
        column: {"fold_aucs": [], "fold_probs": {}}
        for column in OUTCOME_COLS
    }
    loss_curves = {}
    for split in splits:
        torch.manual_seed(seed + split.fold)
        x_train_clin, x_val_clin, _, _, _ = prepare_clinical(
            split.train_pat_clin, split.val_pat_clin
        )
        pos_weight = []
        for outcome_idx in range(4):
            n_pos = max(1, int(split.train_pat_y[:, outcome_idx].sum()))
            n_neg = max(1, len(split.train_pat_y) - n_pos)
            pos_weight.append(n_neg / n_pos)

        model = LateFusion(
            image_in=768,
            image_proj=cfg.model.image_proj_dim,
            clin_in=len(clin_cols),
            clin_proj=cfg.model.clinical_proj_dim,
            mlp_hidden=cfg.model.fusion_mlp_hidden,
            dropout=cfg.model.dropout,
            n_out=4,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
        )
        train_loader, val_loader = _torch_loaders(
            split, x_img, x_train_clin, x_val_clin, pos_weight, cfg.train.batch_size
        )
        model, best_auc, history = train_torch_per_fold(
            model, train_loader, val_loader, optimizer, cfg, device, split.fold
        )
        loss_curves[str(split.fold)] = history
        print(f"  fold {split.fold}: best val-pat-AUC-mean = {best_auc:.4f}")

        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["x_img_flat"].to(device)
                pat_idx = batch["pat_idx"].to(device)
                clinical = batch["x_clin"].to(device)
                image_embedding = patient_logit_mean(
                    model.image_proj(images), pat_idx, batch["n_patients"]
                )
                logits = model.head(
                    torch.cat([image_embedding, model.clin_proj(clinical)], dim=-1)
                )
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
                split.val_pat_ids,
                probs[:, outcome_idx],
                labels[:, outcome_idx],
            )

    output_dir = Path(args.out_json).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "run": cfg.model.run,
        "kind": cfg.model.kind,
        "config": str(args.config),
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
    (output_dir / "loss_curves.json").write_text(json.dumps(loss_curves, indent=2))
    print(f"[stage2_train] wrote {output_dir / 'loss_curves.json'}")
    print(f"\n[stage2_train] {cfg.model.run} headline (mean over folds, patient-pooled):")
    for column in OUTCOME_COLS:
        aucs = per_outcome[column]["fold_aucs"]
        print(f"  {column}: {np.nanmean(aucs):.4f}  (folds={[f'{auc:.3f}' for auc in aucs]})")
    print(f"[stage2_train] wrote {args.out_json}")
    return 0


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"not json-serializable: {type(obj)}")


# ===========================================================================
# Patient-aware torch DataLoader builders
# ===========================================================================
def _torch_loaders(split, x_img, x_train_clin, x_val_clin,
                   pos_weight, batch_size):
    class PatientDataset(torch.utils.data.Dataset):
        def __init__(self, patient_ids, rows_by_patient, clinical, labels):
            self.patient_ids = list(patient_ids)
            self.rows_by_patient = rows_by_patient
            self.clinical = torch.from_numpy(clinical)
            self.labels = torch.from_numpy(labels)

        def __len__(self):
            return len(self.patient_ids)

        def __getitem__(self, idx):
            patient_id = self.patient_ids[idx]
            rows = self.rows_by_patient[patient_id]
            return {
                "x_img_flat": torch.from_numpy(x_img[rows]),
                "x_clin": self.clinical[idx],
                "y": self.labels[idx],
                "pos_weight": torch.tensor(pos_weight, dtype=torch.float32),
            }

    train_dataset = PatientDataset(
        split.train_pat_ids,
        split.train_rows_by_pat,
        x_train_clin,
        split.train_pat_y,
    )
    val_dataset = PatientDataset(
        split.val_pat_ids,
        split.val_rows_by_pat,
        x_val_clin,
        split.val_pat_y,
    )
    return (
        torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_patients,
        ),
        torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate_patients,
        ),
    )


def _collate_patients(batch):
    """Flatten variable-size patient image sets and stack clinical inputs."""
    n_p = len(batch)
    out: dict = {"n_patients": n_p,
                 "y": torch.stack([b["y"] for b in batch], dim=0)}
    flat_imgs = [item["x_img_flat"] for item in batch]
    sizes = [images.shape[0] for images in flat_imgs]
    out["x_img_flat"] = torch.cat(flat_imgs, dim=0)
    out["pat_idx"] = torch.cat(
        [torch.full((size,), idx, dtype=torch.long) for idx, size in enumerate(sizes)]
    )
    out["x_clin"] = torch.stack([item["x_clin"] for item in batch], dim=0)
    # pos_weight: shape (4,) — shared across patients in a fold
    out["pos_weight"] = batch[0]["pos_weight"]
    return out


if __name__ == "__main__":
    sys.exit(main())
