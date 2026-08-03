"""
D1-D4 drift sanity checks per gpt_review.md / claude_review.md.

  D1  base(input_original) vs base(input_original)         expect ≈ {1.0, 1.0}
  D2  student_init(input_original) vs base(input_original) expect ≈ {1.0, 1.0}
  D3  base(input_masked) vs base(input_original)           measures mask-only perturbation
  D4  trained_student(input_original) vs base(input_original)  THE number that matters

Usage:
  # Run all four checks (requires both best.pt and init.pt from training):
  python scripts/analyze_v1.py \\
      --ckpt checkpoints/v1_baseline/best.pt \\
      --init-ckpt checkpoints/v1_baseline/init.pt \\
      --n-images 200

  # D1 + D3 only (no ckpt needed):
  python scripts/analyze_v1.py --n-images 200
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from ssl_core import (load_vision_tower, rename_position_ids, make_block_mask,
                      SimMIM)
from dataset import KidneyROIDataset, collate_padded


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None,
                   help="trained student (best.pt / latest.pt) for D4")
    p.add_argument("--init-ckpt", default=None,
                   help="step-0 student (init.pt) for D2; not the same as teacher")
    p.add_argument("--model-dir", default="/home/vipuser/models/gemma-4-E2B")
    p.add_argument("--val-list", default="/home/vipuser/gemmav4_2026-07-09/data/val.txt")
    p.add_argument("--max-soft-tokens", type=int, default=70)
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--out-json", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def _forward(vit, pe, pixel_values, pixel_position_ids,
             mask_token=None, inject_mask=False, mask=None):
    """Run patch_embedder + encoder. Optionally inject mask_token.

    Returns (B, N, 768) hidden states for valid (non-pad) patches.
    """
    pad = (pixel_position_ids == -1).all(dim=-1)
    valid = ~pad
    x = pe.input_proj((2 * (pixel_values - 0.5)).to(pe.input_proj.weight.dtype))
    if inject_mask:
        assert mask is not None
        if mask_token is None:
            # D3: synthesize a mask_token equivalent to a freshly-init'd
            # student. Use a small random init in the encoder dtype.
            mt = torch.zeros(pe.input_proj.weight.shape[0],
                             dtype=pe.input_proj.weight.dtype,
                             device=pe.input_proj.weight.device)
            torch.nn.init.normal_(mt, std=0.02)
            mask_token = mt
        x = torch.where(mask.unsqueeze(-1), mask_token.to(x.dtype), x)
    x = x + pe._position_embeddings(pixel_position_ids, pad)
    h = vit.encoder(
        inputs_embeds=x, attention_mask=valid, pixel_position_ids=pixel_position_ids,
    ).last_hidden_state
    return h[valid].float()


@torch.no_grad()
def _drift(h_a: torch.Tensor, h_b: torch.Tensor) -> dict:
    cos = torch.nn.functional.cosine_similarity(h_a, h_b, dim=-1)
    return {
        "cos_sim": cos.mean().item(),
        "norm_a": h_a.norm(dim=-1).mean().item(),
        "norm_b": h_b.norm(dim=-1).mean().item(),
        "norm_ratio_a_over_b":
            h_a.norm(dim=-1).mean().item() / max(1e-12, h_b.norm(dim=-1).mean().item()),
        "n_patches_compared": h_a.size(0),
    }


@torch.no_grad()
def run_check(teacher_bundle, student_vit, student_mask_token,
              loader, device, mask_ratio, block, mode: str) -> dict:
    """mode is one of:
        D1_base_vs_base              (input_original vs input_original, same model)
        D2_student_init_vs_base      (input_original vs input_original)
        D3_base_masked_vs_base       (input_masked vs input_original, both teacher)
        D4_student_trained_vs_base   (input_original vs input_original)
    """
    teacher_bundle.vit.eval()
    student_vit.eval()

    cos_acc = 0.0
    norm_a_acc = 0.0
    norm_b_acc = 0.0
    n_total = 0

    pe_t = teacher_bundle.patch_embedder
    pe_s = student_vit.patch_embedder

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        batch = rename_position_ids(batch)
        pv = batch["pixel_values"]
        pid = batch["pixel_position_ids"]

        # D1: same model, same input, twice
        if mode == "D1_base_vs_base":
            h_a = _forward(teacher_bundle.vit, pe_t, pv, pid)
            h_b = _forward(teacher_bundle.vit, pe_t, pv, pid)

        # D2: student_init (init.pt) vs teacher, unmasked
        elif mode == "D2_student_init_vs_base":
            mask = make_block_mask(pid, ratio=mask_ratio, block=block)
            h_b = _forward(teacher_bundle.vit, pe_t, pv, pid)
            h_a = _forward(student_vit, pe_s, pv, pid,
                           mask_token=student_mask_token, inject_mask=False)

        # D3: teacher(masked) vs teacher(unmasked)  -- pure mask perturbation
        elif mode == "D3_base_masked_vs_base":
            mask = make_block_mask(pid, ratio=mask_ratio, block=block)
            h_a = _forward(teacher_bundle.vit, pe_t, pv, pid,
                           mask_token=None, inject_mask=True, mask=mask)
            h_b = _forward(teacher_bundle.vit, pe_t, pv, pid)
            # IMPORTANT: only compare unmasked positions to avoid trivial
            # masked-token disagreement dominating the metric.
            unmasked = ~mask[~((pid == -1).all(dim=-1))]
            h_a_unmasked = h_a[unmasked]
            h_b_unmasked = h_b[unmasked]

            # Aggregate both: unmasked-only (the cleaner number) and
            # all-positions (what summary.md likely measured, which mixes
            # mask_token itself with propagation effects).
            cos_unmasked = torch.nn.functional.cosine_similarity(
                h_a_unmasked, h_b_unmasked, dim=-1).mean().item()
            cos_all = torch.nn.functional.cosine_similarity(h_a, h_b, dim=-1).mean().item()
            return {
                "cos_sim": cos_unmasked,
                "cos_sim_all_positions": cos_all,
                "n_unmasked_patches": h_a_unmasked.size(0),
                "n_all_patches": h_a.size(0),
                "note": (
                    "cos_sim = unmasked positions only (the clean D3); "
                    "cos_sim_all_positions matches what summary.md likely measured"
                ),
            }

        # D4: trained student vs teacher, unmasked
        elif mode == "D4_student_trained_vs_base":
            mask = make_block_mask(pid, ratio=mask_ratio, block=block)
            h_b = _forward(teacher_bundle.vit, pe_t, pv, pid)
            h_a = _forward(student_vit, pe_s, pv, pid,
                           mask_token=student_mask_token, inject_mask=False)

        else:
            raise ValueError(mode)

        if h_a.size(0) == 0:
            continue
        cos = torch.nn.functional.cosine_similarity(h_a, h_b, dim=-1)
        cos_acc += cos.sum().item()
        norm_a_acc += h_a.norm(dim=-1).sum().item()
        norm_b_acc += h_b.norm(dim=-1).sum().item()
        n_total += h_a.size(0)

    if n_total == 0:
        return {"error": "no valid patches"}

    return {
        "cos_sim": cos_acc / n_total,
        "norm_ratio_a_over_b": (norm_a_acc / n_total) / max(1e-12, (norm_b_acc / n_total)),
    }


def _load_student_into(teacher, ckpt_path: str, device: str, dtype) -> tuple:
    """Load a student checkpoint into a copy of teacher. Return (vit, mask_token)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state"]
    # vit.* keys are at top level; mask_token key is 'mask_token'
    vit_sd = {k[len("vit."):]: v for k, v in sd.items() if k.startswith("vit.")}
    teacher.vit.load_state_dict(vit_sd, strict=False)
    if "mask_token" in sd:
        mt = torch.nn.Parameter(sd["mask_token"].to(dtype))
    else:
        mt = torch.nn.Parameter(
            torch.zeros(teacher.hidden_size, dtype=dtype, device=device))
    return teacher.vit, mt


def main() -> int:
    args = parse()
    dtype = torch.bfloat16
    teacher = load_vision_tower(args.model_dir, args.max_soft_tokens,
                                dtype=dtype, device=args.device)

    ds = KidneyROIDataset(args.val_list, processor=teacher.processor)
    if args.n_images < len(ds):
        ds.paths = ds.paths[:args.n_images]
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                        collate_fn=collate_padded)

    out: dict = {"n_images": len(ds), "ckpt": args.ckpt, "init_ckpt": args.init_ckpt}

    # D1: base vs base (same model twice)
    print("[analyze] D1: base(input_orig) vs base(input_orig)", flush=True)
    out["D1_base_vs_base"] = run_check(
        teacher, teacher.vit, None, loader, args.device,
        args.mask_ratio, args.block_size, mode="D1_base_vs_base",
    )
    print(f"  -> {out['D1_base_vs_base']}", flush=True)

    # D3: base(masked) vs base(unmasked) -- pure mask perturbation
    print("[analyze] D3: base(masked) vs base(unmasked)", flush=True)
    out["D3_base_masked_vs_base"] = run_check(
        teacher, teacher.vit, None, loader, args.device,
        args.mask_ratio, args.block_size, mode="D3_base_masked_vs_base",
    )
    print(f"  -> {out['D3_base_masked_vs_base']}", flush=True)

    # D2: student_init vs teacher (requires init.pt)
    if args.init_ckpt and Path(args.init_ckpt).exists():
        print(f"[analyze] D2: student_init({args.init_ckpt}) vs teacher", flush=True)
        # Make a deep copy of teacher to load init weights into
        import copy
        student_init = load_vision_tower(args.model_dir, args.max_soft_tokens,
                                         dtype=dtype, device=args.device)
        vit, mt = _load_student_into(student_init, args.init_ckpt,
                                     args.device, dtype)
        out["D2_student_init_vs_base"] = run_check(
            teacher, vit, mt, loader, args.device,
            args.mask_ratio, args.block_size, mode="D2_student_init_vs_base",
        )
        print(f"  -> {out['D2_student_init_vs_base']}", flush=True)
        del student_init
    else:
        out["D2_student_init_vs_base"] = {
            "skipped": "no --init-ckpt provided (or file missing)",
        }

    # D4: trained_student vs teacher (the one that matters)
    if args.ckpt and Path(args.ckpt).exists():
        print(f"[analyze] D4: trained_student({args.ckpt}) vs teacher", flush=True)
        import copy
        trained = load_vision_tower(args.model_dir, args.max_soft_tokens,
                                    dtype=dtype, device=args.device)
        vit, mt = _load_student_into(trained, args.ckpt, args.device, dtype)
        out["D4_student_trained_vs_base"] = run_check(
            teacher, vit, mt, loader, args.device,
            args.mask_ratio, args.block_size, mode="D4_student_trained_vs_base",
        )
        print(f"  -> {out['D4_student_trained_vs_base']}", flush=True)
        del trained
    else:
        out["D4_student_trained_vs_base"] = {
            "skipped": "no --ckpt provided (or file missing)",
        }

    # Verdict helper (informational; the user makes the final call)
    d4 = out.get("D4_student_trained_vs_base", {})
    if "cos_sim" in d4:
        c = d4["cos_sim"]
        if c > 0.85:
            verdict = "V1 likely fine to ship (drift acceptable)"
        elif c > 0.70:
            verdict = "moderate drift; consider V2 (conservative) or V3 (teacher distill)"
        else:
            verdict = "heavy drift; V3 teacher distillation strongly recommended"
        out["verdict_D4"] = verdict
        print(f"[verdict] D4 cos_sim={c:.3f} -> {verdict}", flush=True)

    out_path = args.out_json or Path("drift_analysis.json")
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"[analyze] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())