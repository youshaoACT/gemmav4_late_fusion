"""
Shared utilities for Stage-1 SSL: model loading, processor setup,
mask generation, forward pass through the patched ViT.

This module is the implementation of Plan §2.2 (steps 1-4) and §2.3
(QAT clipped-linears). Importable from both smoke test and train script.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# QAT clipped-linears (Plan §2.3)
# ---------------------------------------------------------------------------
def disable_clipped_linears(vit: nn.Module) -> int:
    """Walk vit, set use_clipped_linears=False on every Gemma4ClippableLinear.

    Why: ckpt carries input_min/max calibrated on natural images. Forward-time
    clamp would clip ultrasound activations. Buffers stay registered; only the
    forward-time toggle is flipped. See Plan §2.3 + summary §7.3.
    """
    # Lazy import: the symbol lives in transformers, may not exist in older
    # versions.
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    except Exception:
        return 0
    n = 0
    for m in vit.modules():
        if isinstance(m, Gemma4ClippableLinear):
            m.use_clipped_linears = False
            n += 1
    return n

# ---------------------------------------------------------------------------
# Load the vision tower
# ---------------------------------------------------------------------------
@dataclass
class VisionBundle:
    """Holds vision_tower + processor + config knobs."""
    vit: nn.Module                     # Gemma4VisionModel (encoder + patch_embedder + pooler)
    patch_embedder: nn.Module          # Gemma4VisionPatchEmbedder
    encoder: nn.Module                 # Gemma4VisionEncoder
    processor: object                  # Gemma4ImageProcessor
    hidden_size: int
    patch_size: int
    max_soft_tokens: int
    pooling_kernel_size: int           # = 3 by default
    dtype: torch.dtype

def load_vision_tower(model_dir: str | Path, max_soft_tokens: int = 70,
                      dtype: torch.dtype = torch.bfloat16,
                      device: str = "cuda") -> VisionBundle:
    """Load Gemma 4 vision tower from a local snapshot.

    Uses AutoModel + AutoProcessor. Disables clipped-linears post-load.
    """
    from transformers import AutoModel, AutoProcessor
    model_dir = str(model_dir)

    # Load full multi-modal model (only the vision tower is used here), but
    # we never instantiate the LLM tensors — we strip them after load.
    full = AutoModel.from_pretrained(
        model_dir, dtype=dtype, low_cpu_mem_usage=True
    )
    # Extract vision tower.
    vit = full.vision_tower
    vit.eval()  # pooler etc. behave fine; we'll override train/eval at call site
    for p in vit.parameters():
        p.requires_grad_(True)  # caller will freeze as needed

    patch_embedder = vit.patch_embedder
    encoder = vit.encoder

    processor = AutoProcessor.from_pretrained(model_dir)
    # NaFlex processor: max_num_patches = max_soft_tokens * pooling_kernel_size^2
    # but the processor API exposes `max_num_patches` directly; pass through.
    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        if hasattr(ip, "max_num_patches"):
            ip.max_num_patches = max_soft_tokens * (ip.pooling_kernel_size ** 2
                                                   if hasattr(ip, "pooling_kernel_size")
                                                   else 9)
        if hasattr(ip, "max_soft_tokens"):
            ip.max_soft_tokens = max_soft_tokens

    # Plan §2.3: disable clipped linears on ultrasound
    n_disabled = disable_clipped_linears(vit)
    print(f"[load_vision_tower] disabled {n_disabled} Gemma4ClippableLinear modules")

    vit.to(device)

    # Pull a few structural knobs from config for downstream use.
    cfg = full.config.vision_config if hasattr(full.config, "vision_config") else full.config
    hidden_size = getattr(cfg, "hidden_size", 768)
    patch_size  = getattr(cfg, "patch_size", 16)
    pool_ks     = getattr(cfg, "pooling_kernel_size", 3)

    return VisionBundle(
        vit=vit, patch_embedder=patch_embedder, encoder=encoder,
        processor=processor, hidden_size=hidden_size, patch_size=patch_size,
        max_soft_tokens=max_soft_tokens, pooling_kernel_size=pool_ks, dtype=dtype,
    )

# ---------------------------------------------------------------------------
# Processor output rename (Plan §1.4 + summary §7.2)
# ---------------------------------------------------------------------------
def rename_position_ids(batch: dict) -> dict:
    """processor outputs `image_position_ids`; encoder wants `pixel_position_ids`.

    Gemma4Model.forward does this rename internally. SSL bypasses Gemma4Model
    and calls the encoder directly, so we must do it ourselves.
    """
    if "image_position_ids" in batch and "pixel_position_ids" not in batch:
        batch["pixel_position_ids"] = batch.pop("image_position_ids")
    return batch

# ---------------------------------------------------------------------------
# Block mask generation (Plan §2.2 step 2)
# ---------------------------------------------------------------------------
def make_block_mask(pixel_position_ids: torch.Tensor,
                    ratio: float = 0.6, block: int = 8,
                    rng: Optional[torch.Generator] = None) -> torch.Tensor:
    """Return bool mask (B, N) — True on masked positions.

    - Mask only valid (non-pad) patches: padding sentinel = (-1, -1)
    - Block granularity: tile the (H, W) grid with block x block squares; sample
      squares until ratio*|valid| patches are covered.
    - mask ⊂ ~padding (Plan §2.2 step 2 invariant)
    """
    B, N, _ = pixel_position_ids.shape
    device = pixel_position_ids.device
    valid = (pixel_position_ids != -1).all(dim=-1)              # (B, N)
    out = torch.zeros(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        idx = valid[b].nonzero(as_tuple=False).squeeze(-1)     # (n_valid,)
        if idx.numel() == 0:
            continue
        # Recover H, W from position_ids.
        pos = pixel_position_ids[b, idx]                        # (n_valid, 2)
        # positions are (x, y) per Plan §1.4; recover grid extent
        H = int(pos[:, 1].max().item()) + 1
        W = int(pos[:, 0].max().item()) + 1
        # Build grid mask of shape (H, W)
        grid_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
        # Sample blocks until ratio*|valid| reached
        target = max(1, int(round(ratio * idx.numel())))
        # number of full blocks (block x block) along each axis
        nby = max(1, H // block)
        nbx = max(1, W // block)
        blocks_total = nby * nbx
        if blocks_total == 0:
            continue
        # Random ordering of blocks
        order = torch.randperm(blocks_total, generator=rng, device=device)
        per_block = block * block
        covered = 0
        for k in order.tolist():
            by, bx = divmod(k, nbx)
            y0, x0 = by * block, bx * block
            y1 = min(y0 + block, H)
            x1 = min(x0 + block, W)
            grid_mask[y0:y1, x0:x1] = True
            covered += int((y1 - y0) * (x1 - x0))
            if covered >= target:
                break
        # Map grid_mask back to flat indices via positions
        # grid_mask[y, x] == True  =>  patch at position (x, y) is masked
        gm = grid_mask  # (H, W)
        # pos[:, 0] = x, pos[:, 1] = y
        sel = gm[pos[:, 1], pos[:, 0]]
        out[b, idx] = sel
    return out

# ---------------------------------------------------------------------------
# Forward pass (Plan §2.2 step 3)
# ---------------------------------------------------------------------------
class SimMIM(nn.Module):
    """Wraps the vision tower with mask_token + lightweight decoder.

    Implements Plan §2.2 step 3 (encoder forward with mask_token injection)
    and step 4 (linear decoder 768 -> 256). Loss is computed externally
    (so smoke test and training share this module).
    """
    def __init__(self, bundle: VisionBundle, decoder_type: str = "linear",
                 hidden_dim: Optional[int] = None, num_layers: int = 1):
        super().__init__()
        self.bundle = bundle
        pe = bundle.patch_embedder
        # mask_token: same dtype as input_proj.weight (Plan §2.2 step 3)
        w = pe.input_proj.weight
        self.mask_token = nn.Parameter(
            torch.zeros(bundle.hidden_size, dtype=w.dtype, device=w.device)
        )
        nn.init.normal_(self.mask_token, std=0.02)
        # Encoder hidden states have huge magnitude (~30 mean abs) — add a
        # LayerNorm so the decoder sees normalized features. Without this the
        # predictions are ~25 mean abs vs targets ~0.5.
        self.norm = nn.LayerNorm(bundle.hidden_size).to(w.dtype)
        # Decoder — zero-init weight & bias (SimMIM/MAE convention). This
        # makes initial predictions = 0 so L1 loss starts at mean(|target|)
        # rather than at a random-Linear-inflation value.
        out_dim = (bundle.patch_size ** 2)  # 256 for 16x16 single-channel
        if decoder_type == "linear":
            self.decoder = nn.Linear(bundle.hidden_size, out_dim, bias=True)
        elif decoder_type == "mlp":
            hd = hidden_dim or bundle.hidden_size
            self.decoder = nn.Sequential(
                nn.Linear(bundle.hidden_size, hd),
                nn.GELU(),
                nn.Linear(hd, out_dim),
            )
        else:
            raise ValueError(f"unknown decoder_type: {decoder_type}")
        # Cast decoder to encoder dtype
        self.decoder = self.decoder.to(w.dtype)
        # Zero-init the last linear layer
        if decoder_type == "linear":
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)
        else:  # mlp: zero-init the last layer only
            last_linear = self.decoder[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def encode(self, pixel_values: torch.Tensor,
               pixel_position_ids: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run patch_embedder + encoder with optional mask_token injection.

        Returns (B, N, 768) encoder hidden states.
        """
        pe = self.bundle.patch_embedder
        enc = self.bundle.encoder
        w = pe.input_proj.weight

        # Plan §2.2 step 3: explicit cast (matches source L617-618)
        x = pe.input_proj((2 * (pixel_values - 0.5)).to(w.dtype))
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        # Position embeddings via private API _position_embeddings (Plan §2.2 step 3)
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        x = x + pe._position_embeddings(pixel_position_ids, padding_positions)

        attn_mask = (~padding_positions)  # True = attend
        # transformers 5.13 accepts bool attention_mask (verify in smoke test)
        out = enc(
            inputs_embeds=x,
            attention_mask=attn_mask,
            pixel_position_ids=pixel_position_ids,
        )
        return out.last_hidden_state

    def forward(self, pixel_values: torch.Tensor,
                pixel_position_ids: torch.Tensor,
                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(pixel_values, pixel_position_ids, mask=mask)
        h = self.norm(h)
        pred = self.decoder(h)                              # (B, N, 256)
        return h, pred

# ---------------------------------------------------------------------------
# Loss (Plan §2.2 step 4-5)
# ---------------------------------------------------------------------------
def reconstruction_target(pixel_values: torch.Tensor) -> torch.Tensor:
    """Return target = 2*(pixel_values - 0.5) sliced to single channel.

    pixel_values is (B, N, 768) where 768 = 16*16*3, C fastest axis.
    Single-channel slice = pixel_values[..., ::3] -> (B, N, 256). See Plan §2.2.
    """
    return 2 * (pixel_values - 0.5)[..., ::3]

def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor, kind: str = "l1") -> torch.Tensor:
    """Mean over masked patches only. Pred/target (B, N, 256), mask (B, N)."""
    if mask.sum() == 0:
        return pred.sum() * 0.0  # no-op grad
    diff = (pred - target)
    if kind == "l1":
        per = diff.abs().mean(dim=-1)                       # (B, N)
    elif kind == "l2":
        per = (diff ** 2).mean(dim=-1)
    else:
        raise ValueError(kind)
    return per[mask].mean()