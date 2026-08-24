#!/usr/bin/env python3
"""Thin CLI wrappers for FiRe-MPO evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description="FiRe-MPO evaluation dispatcher")
    p.add_argument(
        "task",
        choices=["vqa", "green", "grounding"],
        help="vqa: GPT judge accuracy; green: IU-Xray Green Score; grounding: VGMED",
    )
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Args forwarded to the underlying script")
    args = p.parse_args()
    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.task == "vqa":
        cmd = [sys.executable, str(ROOT / "utils/correctness_evaluator.py"), *extra]
    elif args.task == "green":
        cmd = [sys.executable, str(ROOT / "utils/eval/green_score_eval.py"), *extra]
    else:
        cmd = [sys.executable, str(ROOT / "utils/eval/visual_grounding.py"), *extra]

    print("Running:", " ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
