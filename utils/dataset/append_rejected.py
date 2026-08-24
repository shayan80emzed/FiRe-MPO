#!/usr/bin/env python3
"""Merge rejected_image_path from a reference JSON into another by matching (id, qid)."""

import argparse
import json
from pathlib import Path
from typing import Any


def _key(sample: dict[str, Any]) -> tuple[str, int]:
    sid = str(sample["id"])
    qid = sample["qid"]
    qid_int = int(qid) if not isinstance(qid, int) else qid
    return (sid, qid_int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--without-rejected",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--with-rejected",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: <without-rejected stem>_with_medsam3_rejected.json next to without-rejected)",
    )
    args = parser.parse_args()

    without_path: Path = args.without_rejected
    with_path: Path = args.with_rejected
    out_path: Path = args.output or (
        without_path.parent / f"{without_path.stem}_with_medsam3_rejected.json"
    )
    if out_path.exists():
        raise FileExistsError(f"Output path {out_path} already exists.")
   

    with open(with_path, encoding="utf-8") as f:
        with_rejected: list[dict[str, Any]] = json.load(f)
    with open(without_path, encoding="utf-8") as f:
        without_rejected: list[dict[str, Any]] = json.load(f)

    rejected_path_by_key: dict[tuple[str, int], str] = {}
    for row in with_rejected:
        k = _key(row)
        if "rejected_image_path" not in row:
            raise KeyError(f"with-rejected row missing rejected_image_path: {k}")
        rejected_path_by_key[k] = row["rejected_image_path"]

    merged: list[dict[str, Any]] = []
    for row in without_rejected:
        k = _key(row)
        if k not in rejected_path_by_key:
            continue
        new_row = dict(row)
        new_row["rejected_image_path"] = rejected_path_by_key[k]
        merged.append(new_row)

    dropped = len(without_rejected) - len(merged)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(merged)} samples to {out_path}")
    print(f"Dropped {dropped} samples (no matching id+qid in with-rejected)")


if __name__ == "__main__":
    main()
