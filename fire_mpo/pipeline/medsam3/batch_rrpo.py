#!/usr/bin/env python3
"""
Batch MedSAM3 segmentation + noisy rejected images for RRPO JSON.

Reads JSON enriched with ``medsam3_prompt`` (from ``rrpo_enrich_medsam3_prompt.py``).
If ``medsam3_prompt`` is missing or empty after stripping, **MedSAM3 is not run** and
Gaussian noise is applied to **the entire image**. Otherwise runs MedSAM3, unions masks,
adds noise **inside the mask only**, and saves. Sets ``rejected_image_path`` on each record.

Reuses one loaded model. Caches predictions by (image_path, medsam3_prompt).

Requires: MedSAM3 install, GPU (or pass --device cpu), same env as medsam3_inference.py.

Does not overwrite the input JSON unless --output points to the same path as --input.
"""

from __future__ import annotations

import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from fire_mpo.pipeline.medsam3.inference import (
    DEFAULT_MEDSAM3_LORA_WEIGHTS,
    _medsam3_cwd,
    build_medsam3_inferencer,
)


def _union_mask_from_results(results: Dict[Any, Any]) -> Optional[np.ndarray]:
    """(H, W) bool mask, union of all detection masks for prompt index 0; or None."""
    parts = []
    for k, r in results.items():
        if k == "_image":
            continue
        m = r.get("masks")
        if m is None or r.get("num_detections", 0) == 0:
            continue
        # m: [N, H, W]
        parts.append(m)
    if not parts:
        return None
    stacked = np.concatenate(parts, axis=0)
    return stacked.any(axis=0)


def _apply_mask_noise(
    pil_rgb: Image.Image,
    mask_hw: np.ndarray,
    *,
    sigma: float,
    seed: Optional[int],
) -> Image.Image:
    """Add Gaussian noise (per-channel) where ``mask_hw`` is True."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(pil_rgb).astype(np.float32)
    h, w = mask_hw.shape
    if arr.shape[0] != h or arr.shape[1] != w:
        raise ValueError(f"Mask size {mask_hw.shape} != image {arr.shape[:2]}")
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = arr.copy()
    m = mask_hw[..., None] if arr.ndim == 3 else mask_hw
    out = np.where(m, out + noise, out)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _apply_global_noise(
    pil_rgb: Image.Image,
    *,
    sigma: float,
    seed: Optional[int],
) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(pil_rgb).astype(np.float32)
    noise = rng.normal(0.0, sigma, arr.shape).astype(np.float32)
    out = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _dataset_greedy_dir(name: str) -> Path:
    return REPO_ROOT / "preference_dataset" / name / "greedy"


def _stable_seed_from_fields(*fields: Any) -> int:
    """
    Produce a deterministic int seed from arbitrary (possibly non-int) fields.

    This avoids crashes when ids are strings/UUIDs while keeping runs reproducible.
    """
    # Use a stable byte representation across python runs
    msg = "|".join("" if f is None else str(f) for f in fields).encode("utf-8", errors="replace")
    # 64-bit seed space is plenty for numpy RNG
    return int.from_bytes(hashlib.blake2b(msg, digest_size=8).digest(), "little", signed=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch MedSAM3 + noisy rejected images for RRPO.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["slake", "vqa_rad", "iu_xray"],
        required=True,
        help="Default --input/--output/--rejected-dir under preference_dataset/<dataset>/ unless those are set explicitly",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "RRPO JSON with medsam3_prompt field. "
            "Default: preference_dataset/<dataset>/greedy/rrpo_with_medsam3_prompt.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSON path. "
            "Default: preference_dataset/<dataset>/greedy/rrpo_with_medsam3_rejected.json"
        ),
    )
    parser.add_argument(
        "--rejected-dir",
        type=str,
        default=None,
        help=(
            "Directory for rejected PNG files. "
            "Default: medsam3_rejected_images/<dataset>/"
        ),
    )
    parser.add_argument("--medsam3-root", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--weights", type=str, default="")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=45.0,
        help="Gaussian noise std (0-255 scale) inside segmented region",
    )
    parser.add_argument(
        "--empty-prompt-noise-sigma",
        type=float,
        default=None,
        help=(
            "Sigma for full-image noise when medsam3_prompt is empty "
            "(default: --fallback-noise-sigma; set e.g. to --noise-sigma to match mask strength)"
        ),
    )
    parser.add_argument(
        "--fallback-noise-sigma",
        type=float,
        default=12.0,
        help="If medsam3_prompt is set but MedSAM3 finds no mask: light noise on full image",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip row if rejected_image_path file already exists",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="If >0, process only first N rows")
    args = parser.parse_args()

    gdir = _dataset_greedy_dir(args.dataset)
    if args.input is None:
        args.input = str(gdir / "rrpo_with_medsam3_prompt.json")
    if args.output is None:
        args.output = str(gdir / "rrpo_with_medsam3_rejected.json")
    if args.rejected_dir is None:
        if args.dataset == "slake":
            args.rejected_dir = str(REPO_ROOT / "medsam3_rejected_images")
        else:
            args.rejected_dir = str(REPO_ROOT / "medsam3_rejected_images" / args.dataset)

    input_path = Path(args.input).expanduser().resolve()
    rejected_dir = Path(args.rejected_dir).expanduser().resolve()
    rejected_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    weights = args.weights.strip() or str(DEFAULT_MEDSAM3_LORA_WEIGHTS)
    cfg = Path(args.config.strip()) if args.config.strip() else None
    root, inferencer = build_medsam3_inferencer(
        medsam3_root=Path(args.medsam3_root) if args.medsam3_root.strip() else None,
        config=cfg,
        weights=Path(weights),
        resolution=args.resolution,
        threshold=args.threshold,
        nms_iou=args.nms_iou,
        device=args.device,
    )

    pred_cache: Dict[Tuple[str, str], Dict[Any, Any]] = {}
    n_done = 0
    limit = args.max_samples if args.max_samples and args.max_samples > 0 else len(data)
    sigma_empty_prompt = (
        args.empty_prompt_noise_sigma
        if args.empty_prompt_noise_sigma is not None
        else args.fallback_noise_sigma
    )

    for idx, item in enumerate(data[:limit]):
        raw_mp = item.get("medsam3_prompt")
        mp = (raw_mp if raw_mp is not None else "").strip()
        img_path = Path(item["image_path"]).expanduser().resolve()
        out_name = f"{item['id']}_{item['qid']}_noise.png"
        out_path = rejected_dir / out_name
        item["rejected_image_path"] = str(out_path)

        if args.skip_existing and out_path.is_file():
            continue

        pil = Image.open(img_path).convert("RGB")
        seed = _stable_seed_from_fields(item.get("id"), item.get("qid"))

        if not mp:
            # No text prompt: skip MedSAM3; corrupt the whole frame (every pixel).
            noisy = _apply_global_noise(pil, sigma=sigma_empty_prompt, seed=seed)
            noisy.save(out_path)
            item["rejected_medsam3_note"] = "empty_medsam3_prompt_full_image_noise"
            n_done += 1
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{limit}")
            continue

        cache_key = (str(img_path), mp)
        if cache_key not in pred_cache:
            with _medsam3_cwd(root):
                pred_cache[cache_key] = inferencer.predict(str(img_path), [mp])
        results = pred_cache[cache_key]
        union = _union_mask_from_results(results)

        if union is None:
            noisy = _apply_global_noise(pil, sigma=args.fallback_noise_sigma, seed=seed)
            noisy.save(out_path)
            item["rejected_medsam3_note"] = "no_detection_fallback_noise"
        else:
            noisy = _apply_mask_noise(
                pil, union, sigma=args.noise_sigma, seed=seed
            )
            noisy.save(out_path)
            item.pop("rejected_medsam3_note", None)

        n_done += 1
        # break
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{limit}")

    out_json = Path(args.output).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {n_done} rejected images under {rejected_dir}")
    print(f"Wrote JSON ({len(data)} rows) to {out_json}")


if __name__ == "__main__":
    main()
