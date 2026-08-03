"""
Dataset wrapper for Stage-1 SSL.

Loads PNG paths from a list file, opens with PIL, runs Gemma4ImageProcessor,
returns a dict ready to feed the encoder. Collate stacks variable-N tensors
into a padded batch.

Padded entries have position_ids = (-1, -1) and pixel_values = 0.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image

class KidneyROIDataset(Dataset):
    def __init__(self, list_file: str | Path, image_root: str | None = None,
                 processor=None):
        self.paths: List[str] = []
        with open(list_file) as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                if image_root and not Path(p).is_absolute():
                    p = str(Path(image_root) / p)
                self.paths.append(p)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        img = Image.open(path)
        # convert to RGB so processor's do_convert_rgb is a no-op; saves CPU
        if img.mode != "RGB":
            img = img.convert("RGB")
        out = self.processor(images=img, return_tensors="pt")
        # Squeeze batch dim — processor returns (1, N, 768), (1, N, 2)
        return {k: v.squeeze(0) for k, v in out.items()}


def collate_padded(batch: List[dict], pad_value: float = 0.0) -> dict:
    """Pad variable-N tensors to max N in batch.

    For each key:
      - pixel_values:   (B, N_max, 768); pad with 0
      - pixel_position_ids / image_position_ids: (B, N_max, 2); pad with -1
      - num_soft_tokens_per_image (and other scalar-per-image tensors):
          stack directly (1,) -> (B,)
    """
    keys = batch[0].keys()
    N_max = max(b["pixel_values"].size(0) for b in batch)
    out = {}
    for k in keys:
        ref = batch[0][k]
        if ref.dim() == 2 and ref.size(-1) == 768:  # pixel_values
            t = torch.full((len(batch), N_max, 768), pad_value, dtype=ref.dtype)
            for i, b in enumerate(batch):
                n = b[k].size(0)
                t[i, :n] = b[k]
        elif ref.dim() == 2 and ref.size(-1) == 2:    # position ids
            t = torch.full((len(batch), N_max, 2), -1, dtype=ref.dtype)
            for i, b in enumerate(batch):
                n = b[k].size(0)
                t[i, :n] = b[k]
        elif ref.dim() <= 1:                          # scalar per image (e.g. (1,) or 0-d)
            t = torch.stack([b[k].reshape(-1)[0:1] for b in batch], dim=0).reshape(len(batch))
        else:
            # Fallback: assume already equal-sized
            t = torch.stack([b[k] for b in batch], dim=0)
        out[k] = t
    return out