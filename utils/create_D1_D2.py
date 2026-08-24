#!/usr/bin/env python3
"""
Create the D2 preference JSON file from an RRPO-style JSON list.

Input records (e.g. preference_dataset/slake/greedy/rrpo.json) must include:
  id, qid, prompt, chosen, rejected, image_path

D2 rows are built by matching each JSON record against a ``VQADataset`` by
``(id, qid)``:
  - chosen   = the ground-truth ``answer`` from the dataset for that (id, qid).
  - rejected = the JSON record's ``rejected`` with all ``<mask>...</mask>``
               tags removed (inner text kept).
JSON rows with no matching (id, qid) in the dataset are skipped.

Output format matches preference_dataset/slake/greedy/dpo_D1.json:
  id, qid, prompt, chosen, rejected, image_path

Writes ``dpo_D2.json`` in the same directory as the input ``--json`` file.
If the output file already exists, the script exits with an error (nothing is written).
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.dataset.vqa_dataset import VQADataset

_FIRST_MASK = re.compile(r"<mask>(.*?)</mask>", re.DOTALL | re.IGNORECASE)


def _strip_mask_tags(text: str) -> str:
    """Remove <mask>...</mask> wrappers, keeping inner text."""
    if not text:
        return ""
    return _FIRST_MASK.sub(r"\1", text)


def _load_answer_by_id_qid(dataset_name: str, split: str) -> dict[tuple[str, str], str]:
    """Map (id, qid) -> ground-truth answer string from a ``VQADataset``."""
    dataset = VQADataset(dataset_name, split)
    out: dict[tuple[str, str], str] = {}
    for item in dataset:
        key = (str(item["id"]).strip(), str(item["qid"]).strip())
        answer = str(item["answer"]).strip()
        if answer:
            out[key] = answer
    return out


def build_d2_rows(
    records: list[dict],
    answer_by_id_qid: dict[tuple[str, str], str],
) -> list[dict]:
    required = ("id", "qid", "prompt", "chosen", "rejected", "image_path")
    rows: list[dict] = []
    for obj in records:
        missing = [k for k in required if k not in obj]
        if missing:
            raise ValueError(f"JSON record missing keys {missing}: {str(obj)[:200]!r}")

        key = (str(obj["id"]).strip(), str(obj["qid"]).strip())
        if key not in answer_by_id_qid:
            continue
        chosen = answer_by_id_qid[key]
        rejected = _strip_mask_tags(str(obj["rejected"])).strip()
        if not chosen or not rejected:
            continue

        rows.append({
            "id": int(obj["id"]),
            "qid": int(obj["qid"]),
            "prompt": str(obj["prompt"]).strip(),
            "chosen": chosen,
            "rejected": rejected,
            "image_path": str(obj["image_path"]).strip(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Create the D2 preference JSON from RRPO-style JSON."
    )
    parser.add_argument(
        "--json",
        type=Path,
        dest="json_path",
        help="Input JSON list (e.g. preference_dataset/slake/greedy/rrpo.json)",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        required=True,
        help="VQADataset name to match against for ground-truth answers (e.g. slake).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="VQADataset split to load (default: train).",
    )
    args = parser.parse_args()

    path = args.json_path
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of objects")

    answer_by_id_qid = _load_answer_by_id_qid(args.dataset_name, args.split)

    json_path = path.resolve()
    out_dir = json_path.parent
    d2_path = out_dir / "dpo_D2.json"
    if d2_path.exists():
        raise FileExistsError(
            f"Output file already exists next to the input JSON; remove or move it first: {d2_path}"
        )

    d2_rows = build_d2_rows(records, answer_by_id_qid)
    with open(d2_path, "w", encoding="utf-8") as f:
        json.dump(d2_rows, f, indent=2, ensure_ascii=False)
    print(f"D2: {len(d2_rows)} rows written to {d2_path}")


if __name__ == "__main__":
    main()
