#!/usr/bin/env python3
"""
Build D1 / D2 / D3 ablation preference JSONs from on-policy FiRe-MPO pairs.

  D1 — mask-inner phrases only (style-agnostic short spans)
  D2 — GT answer as chosen; rejected with masks stripped (GT style / off-policy)
  D3 — full sentences with masks stripped (sequence-level DPO)

Does not overwrite existing outputs unless --force.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_mpo.pipeline.paths import preference_json, resolve_optional  # noqa: E402
from utils.convert_rrpo_dpo import remove_mask_tags  # noqa: E402
from utils.create_D1_D2 import _load_answer_by_id_qid, build_d2_rows  # noqa: E402

_MASK_INNER = re.compile(r"<mask>(.*?)</mask>", re.DOTALL | re.IGNORECASE)


def _extract_mask_inners(text: str) -> str:
    parts = _MASK_INNER.findall(text or "")
    return " ".join(p.strip() for p in parts if p.strip())


def build_d1(records: list[dict]) -> list[dict]:
    rows = []
    for rec in records:
        chosen = _extract_mask_inners(rec.get("chosen", ""))
        rejected = _extract_mask_inners(rec.get("rejected", ""))
        if not chosen or not rejected:
            continue
        rows.append(
            {
                "id": rec["id"],
                "qid": rec["qid"],
                "prompt": rec["prompt"],
                "chosen": chosen,
                "rejected": rejected,
                "image_path": rec["image_path"],
            }
        )
    return rows


def build_d3(records: list[dict]) -> list[dict]:
    rows = []
    seen = set()
    for rec in records:
        key = (rec["id"], rec["qid"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "id": rec["id"],
                "qid": rec["qid"],
                "prompt": rec["prompt"],
                "chosen": remove_mask_tags(rec["chosen"]),
                "rejected": remove_mask_tags(rec["rejected"]),
                "image_path": rec["image_path"],
                "was_correct": rec.get("was_correct", False),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser(description="Build D1/D2/D3 ablation preference sets")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, choices=["slake", "vqa_rad", "iu_xray"])
    p.add_argument("--input", type=str, default=None, help="Default: .../rrpo.json")
    p.add_argument("--variants", type=str, default="d1,d2,d3", help="Comma list: d1,d2,d3")
    p.add_argument("--vqadataset-split", type=str, default="train", help="Split for D2 GT lookup")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    inp = resolve_optional(args.input, preference_json(args.model, args.dataset, "rrpo.json"))
    with open(inp, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input must be a JSON list")

    variants = {v.strip().lower() for v in args.variants.split(",") if v.strip()}
    out_dir = inp.parent

    if "d1" in variants:
        path = out_dir / "dpo_D1.json"
        if path.exists() and not args.force:
            raise FileExistsError(path)
        rows = build_d1(records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"D1: {len(rows)} → {path}")

    if "d2" in variants:
        path = out_dir / "dpo_D2.json"
        if path.exists() and not args.force:
            raise FileExistsError(path)
        answer_by = _load_answer_by_id_qid(args.dataset, args.vqadataset_split)
        rows = build_d2_rows(records, answer_by)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"D2: {len(rows)} → {path}")

    if "d3" in variants:
        path = out_dir / "dpo_D3.json"
        if path.exists() and not args.force:
            raise FileExistsError(path)
        rows = build_d3(records)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"D3: {len(rows)} → {path}")


if __name__ == "__main__":
    main()
