#!/usr/bin/env python3
"""
Lesion-targeted image corruption via MedSAM3 (paper visual preference v').

Wraps ``fire_mpo.pipeline.medsam3.batch_rrpo`` with ``--model`` path resolution under
preference_dataset/{model}/{dataset}/greedy/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_mpo.pipeline.paths import (  # noqa: E402
    preference_json,
    rejected_images_dir,
    resolve_optional,
)


def main():
    p = argparse.ArgumentParser(description="Corrupt lesions for FiRe-MPO visual prefs")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["slake", "vqa_rad", "iu_xray"])
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--rejected-dir", type=str, default=None)
    p.add_argument("--noise-std", type=float, default=45.0)
    p.add_argument("--fallback-noise-std", type=float, default=12.0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--medsam3-root", type=str, default="")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--weights", type=str, default="")
    args = p.parse_args()

    inp = resolve_optional(
        args.input,
        preference_json(args.model, args.dataset, "rrpo_with_medsam3_prompt.json"),
    )
    out = resolve_optional(
        args.output,
        preference_json(args.model, args.dataset, "rrpo_with_medsam3_rejected.json"),
    )
    rej = resolve_optional(args.rejected_dir, rejected_images_dir(args.dataset))

    # Delegate to existing batch script via argv rewrite
    argv = [
        "medsam3_batch_rrpo.py",
        "--dataset",
        args.dataset,
        "--input",
        str(inp),
        "--output",
        str(out),
        "--rejected-dir",
        str(rej),
        "--noise-std",
        str(args.noise_std),
        "--fallback-noise-std",
        str(args.fallback_noise_std),
        "--device",
        args.device,
        "--max-samples",
        str(args.max_samples),
    ]
    if args.medsam3_root:
        argv += ["--medsam3-root", args.medsam3_root]
    if args.config:
        argv += ["--config", args.config]
    if args.weights:
        argv += ["--weights", args.weights]

    print(f"[corrupt_lesions] {inp} → {out} (images: {rej})")
    sys.argv = argv
    from fire_mpo.pipeline.medsam3 import batch_rrpo as batch

    batch.main()


if __name__ == "__main__":
    main()
