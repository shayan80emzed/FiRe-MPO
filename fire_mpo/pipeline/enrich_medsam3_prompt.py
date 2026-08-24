#!/usr/bin/env python3
"""
Enrich preference JSON with MedSAM3 text prompts (GPT).

Wraps ``fire_mpo.pipeline.medsam3.enrich_prompt`` with ``--model``/``--dataset`` path
resolution. Extends support to ``iu_xray`` via a radiology-report prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_mpo.pipeline.paths import preference_json, resolve_optional  # noqa: E402

# Import enrich backend and extend dataset prompts
from fire_mpo.pipeline.medsam3 import enrich_prompt as enrich  # noqa: E402

IU_XRAY_PROMPT = """You extract a short MedSAM3 text prompt for a chest X-ray finding.

Given a clinical question and ground-truth answer/report snippet, output ONLY a short
anatomical or finding phrase suitable for medical segmentation (1-4 words).
No quotes, no punctuation beyond hyphens.

Question: {question}
Answer: {answer}
"""

enrich.MEDSAM3_USER_PROMPT_BY_DATASET = {
    **enrich.MEDSAM3_USER_PROMPT_BY_DATASET,
    "iu_xray": IU_XRAY_PROMPT,
}


def main():
    p = argparse.ArgumentParser(description="Enrich FiRe-MPO prefs with MedSAM3 prompts")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["slake", "vqa_rad", "iu_xray"])
    p.add_argument("--input", type=str, default=None, help="Default: .../greedy/rrpo.json")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: .../greedy/rrpo_with_medsam3_prompt.json",
    )
    p.add_argument("--splits", type=str, default="train", help="Comma-separated VQADataset splits")
    p.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-concurrent", type=int, default=16)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    inp = resolve_optional(args.input, preference_json(args.model, args.dataset, "rrpo.json"))
    out = resolve_optional(
        args.output,
        preference_json(args.model, args.dataset, "rrpo_with_medsam3_prompt.json"),
    )
    if out.exists() and not args.force:
        raise FileExistsError(f"Output exists: {out} (pass --force)")

    # Build Namespace compatible with enrich._run_async
    ns = argparse.Namespace(
        dataset=args.dataset,
        input=str(inp),
        output=str(out),
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        slake_splits=None,
        model=args.openai_model,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
        skip_existing=not args.force,
    )
    print(f"[enrich_medsam3_prompt] {inp} → {out}")
    data = asyncio.run(enrich._run_async(ns))
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(data)} records → {out}")


if __name__ == "__main__":
    main()
