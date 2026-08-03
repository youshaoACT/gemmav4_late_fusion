"""
Cache 768-d mean-pooled per-image features for the Stage-2 cohort.

Encodes with both the original teacher and the V1-frozen fine-tuned
encoder (sharing config knobs via configs/v1_baseline.yaml). Output:
  features/{teacher,v1}_stage2.npy        (N, 768) float32
  features/{teacher,v1}_stage2_meta.parquet  row metadata (image_name, 病案号, fold)

Per Plan §S3:
  - `max_soft_tokens` is read from configs/v1_baseline.yaml, the single
    source shared with downstream_eval.py — never hard-coded here.
  - Mean-pool over valid (non-pad) patches per image, identical to
    downstream_eval.encode_all.

Per §S3 timing: 32k imgs ~164s on 4090 → 18.8k imgs < 100s in theory.
Reality: the actual cohort here has 6,172 PNG rows (12,687 missing) so
encoding is well under a minute.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ssl_core import load_vision_tower, rename_position_ids
from analyze_v1 import _load_student_into      # V1 student-loader
from dataset import collate_padded             # processor-output collator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_CKPT = Path(os.environ.get(
    "GEMMAV4_V1_CKPT", PROJECT_ROOT / "checkpoints/v1_baseline/best.pt"
))
TEACHER_DIR = Path(os.environ.get("GEMMAV4_TEACHER_DIR", "/home/vipuser/models/gemma-4-E2B"))
V1_CONFIG = Path(os.environ.get(
    "GEMMAV4_V1_CONFIG", PROJECT_ROOT / "configs/v1_baseline.yaml"
))
DATA_DIR = Path(os.environ.get("GEMMAV4_DATA_DIR", PROJECT_ROOT / "data"))
COHORT_PARQUET_DEFAULT = DATA_DIR / "stage2_cohort.parquet"
FOLDS_JSON_DEFAULT = DATA_DIR / "stage2_folds.json"
FEATURES_DIR = Path(os.environ.get("GEMMAV4_FEATURES_DIR", PROJECT_ROOT / "features"))
FEATURES_PREFIX_DEFAULT = "stage2"


@torch.no_grad()
def encode_all(model_bundle, loader, device, dtype) -> np.ndarray:
    """Mean-pool patch features over valid patches. (N, 768) float32."""
    model_bundle.vit.eval()
    feats = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        batch = rename_position_ids(batch)
        pv, pid = batch["pixel_values"], batch["pixel_position_ids"]
        pad = (pid == -1).all(dim=-1)
        valid = ~pad
        pe = model_bundle.patch_embedder
        x = pe.input_proj((2 * (pv - 0.5)).to(pe.input_proj.weight.dtype))
        x = x + pe._position_embeddings(pid, pad)
        h = model_bundle.vit.encoder(
            inputs_embeds=x,
            attention_mask=valid,
            pixel_position_ids=pid,
        ).last_hidden_state                          # (B, N, 768)
        mask_f = valid.float().unsqueeze(-1)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


class _ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = list(paths), processor
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        out = self.processor(images=img, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in out.items()}


def build_loader(paths, processor, batch_size=8):
    return DataLoader(
        _ImagePathDataset(paths, processor),
        batch_size=batch_size, shuffle=False, num_workers=2,
        collate_fn=collate_padded,
    )


def cache_one(name: str, bundle, image_paths, device, dtype, batch_size=8):
    loader = build_loader(image_paths, bundle.processor, batch_size=batch_size)
    t0 = time.time()
    X = encode_all(bundle, loader, device, dtype)
    print(f"[cache] {name}: {X.shape} in {time.time() - t0:.1f}s")
    return X


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["teacher", "v1", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cohort", type=Path, default=COHORT_PARQUET_DEFAULT,
                   help="Cohort parquet path (default: AKD)")
    p.add_argument("--folds", type=Path, default=FOLDS_JSON_DEFAULT,
                   help="Folds JSON path (default: AKD)")
    p.add_argument("--features-prefix", default=FEATURES_PREFIX_DEFAULT,
                   help="Filename prefix for output npy/meta (default 'stage2'; "
                        "biopsy uses 'stage2_biopsy'). Outputs: "
                        "features/{teacher,v1}_{prefix}.npy + _meta.parquet")
    args = p.parse_args()
    cohort_path: Path = args.cohort
    folds_path: Path = args.folds
    features_prefix: str = args.features_prefix

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # ---- cohort + folds --------------------------------------------------
    df = pd.read_parquet(cohort_path)
    # If cohort has group_id (new biopsy builder writes one), use directly.
    # Otherwise build composite (AKD legacy).
    if "group_id" not in df.columns:
        grp_per_pat = (df.groupby("病案号")["pathoID_std"]
                         .agg(lambda s: "||".join(sorted({str(x) for x in s}))))
        df = df.merge(grp_per_pat.rename("group_id").reset_index(), on="病案号", how="left")
    else:
        df["group_id"] = df["group_id"].astype(str)

    folds = json.load(open(folds_path))
    fold_per_row = np.full(len(df), -1, dtype=int)
    for k, fold in enumerate(folds["folds"]):
        val_set = set(fold["val_groups_str"])
        for g_str in val_set:
            mask = (df["group_id"] == g_str).values
            fold_per_row[mask] = k
    df = df.assign(fold=fold_per_row)
    print(f"[cache] cohort: {cohort_path}  ({len(df):,} rows)")

    # max_soft_tokens single source from v1_baseline.yaml
    cfg = OmegaConf.load(V1_CONFIG)
    max_soft_tokens = int(cfg.data.max_soft_tokens)
    print(f"[cache] max_soft_tokens = {max_soft_tokens}  (from {V1_CONFIG})")

    paths = df["full_path"].tolist()
    image_names = df["image_name"].tolist()
    # patient-id column for the meta sidecar: prefer 病理号 (biopsy) else 病案号
    if "病理号" in df.columns:
        pat_ids = df["病理号"].astype(str).tolist()
    else:
        pat_ids = df["病案号"].astype(str).tolist()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame({
        "row_idx": np.arange(len(df)),
        "image_name": image_names,
        "patient_id": pat_ids,                # renamed from 病案号 to be agnostic
        "group_id": df["group_id"].astype(str),
        "fold": df["fold"].astype(int),
        "full_path": paths,
    })

    teacher_npy = FEATURES_DIR / f"teacher_{features_prefix}.npy"
    teacher_meta = FEATURES_DIR / f"teacher_{features_prefix}_meta.parquet"
    v1_npy = FEATURES_DIR / f"v1_{features_prefix}.npy"
    v1_meta = FEATURES_DIR / f"v1_{features_prefix}_meta.parquet"

    # ---- teacher (frozen E2B) -------------------------------------------
    if args.encoder in ("teacher", "both"):
        print(f"\n[cache] loading teacher: {TEACHER_DIR}")
        teacher = load_vision_tower(str(TEACHER_DIR), max_soft_tokens,
                                     dtype=dtype, device=device)
        Xt = cache_one("teacher", teacher, paths, device, dtype, args.batch_size)
        np.save(teacher_npy, Xt)
        meta.to_parquet(teacher_meta, index=False)
        print(f"[cache] wrote {teacher_npy}")
        del teacher
        torch.cuda.empty_cache()

    # ---- V1 frozen ------------------------------------------------------
    if args.encoder in ("v1", "both"):
        print(f"\n[cache] loading V1 from {V1_CKPT}")
        v1 = load_vision_tower(str(TEACHER_DIR), max_soft_tokens,
                                dtype=dtype, device=device)
        vit, mt = _load_student_into(v1, str(V1_CKPT), device, dtype)
        Xv = cache_one("v1", v1, paths, device, dtype, args.batch_size)
        np.save(v1_npy, Xv)
        meta.to_parquet(v1_meta, index=False)
        print(f"[cache] wrote {v1_npy}")

    print(f"\n[cache] done. files:")
    for f in sorted(FEATURES_DIR.glob(f"*_{features_prefix}.npy")):
        sz = f.stat().st_size / 1024 / 1024
        print(f"  {f}  {sz:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
