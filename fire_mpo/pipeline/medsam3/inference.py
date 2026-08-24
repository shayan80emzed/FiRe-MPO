#!/usr/bin/env python3
"""
MedSAM3 text-guided segmentation for a single image.

This script loads the official inference code from a local clone of:
  https://github.com/Joey-S-Liu/MedSAM3

Setup (once):
  git clone https://github.com/Joey-S-Liu/MedSAM3.git
  cd MedSAM3 && pip install -e .
  hf auth login
  LoRA weights default to shared/model_weights/MedSAM3_v1/best_lora_weights.pt
  (override with --weights if needed).

Usage:
  # If you cloned MedSAM3 next to this script (med-align/MedSAM3), no env var needed.
  python medsam3_inference.py --image scan.png --prompt "liver lesion"

  # Otherwise:
  export MEDSAM3_ROOT=/path/to/MedSAM3
  python medsam3_inference.py --image scan.png --prompt "liver lesion"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

DEFAULT_MEDSAM3_LORA_WEIGHTS = Path(
    os.environ.get(
        "MEDSAM3_LORA_WEIGHTS",
        "/data/shayan/med-align/MedSAM3/best_lora_weights.pt",
    )
)

BPE_REL = Path("sam3/assets/bpe_simple_vocab_16e6.txt.gz")


@contextmanager
def _medsam3_cwd(medsam3_root: Path) -> Iterator[None]:
    """infer_sam uses cwd-relative paths (e.g. sam3/assets/bpe_...); run from repo root."""
    prev = os.getcwd()
    os.chdir(medsam3_root)
    try:
        yield
    finally:
        os.chdir(prev)


def resolve_medsam3_root(explicit: Optional[Path]) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("MEDSAM3_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    for guess in (script_dir / "MedSAM3", Path.cwd() / "MedSAM3"):
        g = guess.resolve()
        if (g / "infer_sam.py").is_file():
            return g
    raise FileNotFoundError(
        "Could not find MedSAM3 (no infer_sam.py). Either:\n"
        f"  - Clone into {script_dir / 'MedSAM3'}, or\n"
        "  - Set MEDSAM3_ROOT, or pass --medsam3-root.\n"
        "Clone: https://github.com/Joey-S-Liu/MedSAM3"
    )


def _assert_medsam3_assets(medsam3_root: Path) -> None:
    bpe = medsam3_root / BPE_REL
    if not bpe.is_file():
        raise FileNotFoundError(
            f"Missing {bpe}. MedSAM3 needs sam3/assets (BPE vocab from SAM3). "
            "Re-clone the repo or copy assets from facebookresearch/sam3."
        )


def load_infer_sam_module(medsam3_root: Path):
    root = medsam3_root.resolve()
    infer_path = root / "infer_sam.py"
    if not infer_path.is_file():
        raise FileNotFoundError(
            f"Expected infer_sam.py at {infer_path}. "
            "Clone https://github.com/Joey-S-Liu/MedSAM3 and install with pip install -e ."
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("medsam3_infer_sam", infer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {infer_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_medsam3_inferencer(
    *,
    medsam3_root: Optional[Path] = None,
    config: Optional[Union[str, Path]] = None,
    weights: Optional[Union[str, Path]] = None,
    resolution: int = 1008,
    threshold: float = 0.5,
    nms_iou: float = 0.5,
    device: str = "cuda",
) -> tuple[Path, Any]:
    """
    Construct a single ``SAM3LoRAInference`` for repeated ``predict`` calls (batch jobs).

    Returns ``(medsam3_root, inferencer)``. Call ``predict`` inside ``_medsam3_cwd(root)``.
    """
    root = resolve_medsam3_root(Path(medsam3_root) if medsam3_root else None)
    infer_mod = load_infer_sam_module(root)
    cfg_raw = Path(config) if config else root / "configs" / "full_lora_config.yaml"
    cfg_path = (cfg_raw if cfg_raw.is_absolute() else (root / cfg_raw)).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    _assert_medsam3_assets(root)
    lora = Path(weights) if weights is not None else DEFAULT_MEDSAM3_LORA_WEIGHTS
    cfg_abs = str(cfg_path)
    lora_abs = str(Path(lora).expanduser().resolve())
    with _medsam3_cwd(root):
        inferencer = infer_mod.SAM3LoRAInference(
            config_path=cfg_abs,
            weights_path=lora_abs,
            resolution=resolution,
            detection_threshold=threshold,
            nms_iou_threshold=nms_iou,
            device=device,
        )
    return root, inferencer


def predict(
    image: Union[str, Path],
    prompt: Union[str, Sequence[str]],
    *,
    medsam3_root: Optional[Path] = None,
    config: Optional[Union[str, Path]] = None,
    weights: Optional[Union[str, Path]] = None,
    resolution: int = 1008,
    threshold: float = 0.5,
    nms_iou: float = 0.5,
    device: str = "cuda",
) -> Dict[Any, Any]:
    """
    Run MedSAM3 on one image with one or more text prompts.

    If ``weights`` is None, uses ``DEFAULT_MEDSAM3_LORA_WEIGHTS``.

    Returns the same dict as MedSAM3's SAM3LoRAInference.predict (includes '_image').
    """
    if isinstance(prompt, str):
        prompts: List[str] = [prompt]
    else:
        prompts = list(prompt)

    root, inferencer = build_medsam3_inferencer(
        medsam3_root=medsam3_root,
        config=config,
        weights=weights,
        resolution=resolution,
        threshold=threshold,
        nms_iou=nms_iou,
        device=device,
    )
    image_abs = str(Path(image).expanduser().resolve())
    with _medsam3_cwd(root):
        return inferencer.predict(image_abs, prompts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MedSAM3: text-guided medical segmentation (Joey-S-Liu/MedSAM3)."
    )
    parser.add_argument("--image", type=str, required=True, help="Path input image")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        required=True,
        help='Text concept(s), e.g. "skin lesion" (multi-word counts as one prompt)',
    )
    parser.add_argument(
        "--medsam3-root",
        type=str,
        default="",
        help="Path to cloned MedSAM3 repo (default: MEDSAM3_ROOT, else ./MedSAM3 beside this script or cwd)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to full_lora_config.yaml (default: <medsam3-root>/configs/full_lora_config.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help=f"Path to LoRA checkpoint (default: {DEFAULT_MEDSAM3_LORA_WEIGHTS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./medsam3_output.png",
        help="Visualization PNG path (default: <image_stem>_medsam3.png next to input)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument(
        "--boundingbox",
        action="store_true",
        help="Draw bounding boxes on the visualization",
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Do not overlay segmentation masks",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving visualization (still runs inference and prints summary)",
    )
    args = parser.parse_args()

    root = resolve_medsam3_root(Path(args.medsam3_root) if args.medsam3_root.strip() else None)
    _assert_medsam3_assets(root)
    infer_mod = load_infer_sam_module(root)

    cfg_arg = Path(args.config.strip()) if args.config.strip() else root / "configs" / "full_lora_config.yaml"
    cfg_path = (cfg_arg if cfg_arg.is_absolute() else (root / cfg_arg)).resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    weights = args.weights.strip() or str(DEFAULT_MEDSAM3_LORA_WEIGHTS)

    image_path = Path(args.image).expanduser().resolve()
    cfg_abs = str(cfg_path)
    weights_abs = str(Path(weights).expanduser().resolve())
    if args.output.strip():
        out_abs = str(Path(args.output).expanduser().resolve())
    else:
        out_abs = str(image_path.with_name(f"{image_path.stem}_medsam3.png"))

    with _medsam3_cwd(root):
        inferencer = infer_mod.SAM3LoRAInference(
            config_path=cfg_abs,
            weights_path=weights_abs,
            resolution=args.resolution,
            detection_threshold=args.threshold,
            nms_iou_threshold=args.nms_iou,
        )
        results = inferencer.predict(str(image_path), args.prompt)
        if not args.no_save:
            inferencer.visualize(
                results,
                out_abs,
                show_boxes=args.boundingbox,
                show_masks=not args.no_masks,
            )

    print("\n" + "=" * 60)
    print("Summary:")
    for idx in sorted(k for k in results if k != "_image"):
        r = results[idx]
        print(f"  '{r['prompt']}': {r['num_detections']} detections")
        if r["num_detections"] > 0 and r.get("scores") is not None:
            print(f"    max confidence: {float(r['scores'].max()):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
