#!/usr/bin/env python3
"""
Build on-policy text preference pairs (Fig. 4 / LVLM-style editing).

Dispatches to VQA (SLAKE, VQA-RAD) or report (IU-Xray) builders.
Writes under preference_dataset/{model}/{dataset}/greedy/rrpo.json by default.
Does not overwrite existing files unless --force.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fire_mpo.pipeline.paths import preference_json, resolve_optional  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Build FiRe-MPO on-policy text preference pairs")
    p.add_argument("--model", type=str, required=True, help="Model folder name under preference_dataset/")
    p.add_argument("--dataset", type=str, required=True, choices=["slake", "vqa_rad", "iu_xray"])
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="On-policy inference CSV (question/answer/output/image_path).",
    )
    p.add_argument("--output", type=str, default=None, help="Output JSON (default: preference_dataset/.../rrpo.json)")
    p.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = p.parse_args()

    out = resolve_optional(args.output, preference_json(args.model, args.dataset, "rrpo.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise FileExistsError(f"Output exists: {out} (pass --force to overwrite)")

    if args.dataset == "iu_xray":
        from utils.dataset.rrpo_chest import main as chest_main  # type: ignore

        print(f"[build_text_prefs] IU-Xray via rrpo_chest → {out}")
        # rrpo_chest uses module-level __main__; call its async pipeline if available
        import utils.dataset.rrpo_chest as chest

        if hasattr(chest, "run_from_csv"):
            chest.run_from_csv(args.csv, str(out))
        else:
            # Fallback: set paths and invoke __main__ logic via env
            os.environ["FIRE_MPO_CSV"] = args.csv
            os.environ["FIRE_MPO_OUT"] = str(out)
            print(
                "NOTE: utils.dataset.rrpo_chest has hardcoded __main__ paths. "
                f"Prefer editing that module or pass paths. Writing target: {out}"
            )
            print(f"CSV={args.csv}")
            # Minimal: re-export instruction for user
            raise SystemExit(
                "iu_xray builder still uses utils.dataset.rrpo_chest.__main__. "
                "Update csv/json paths there, or add run_from_csv(). "
                f"Intended output: {out}"
            )
    else:
        from utils.dataset.rrpo import main as vqa_main, get_rrpo_prompt_config  # noqa: F401
        import pandas as pd
        from utils.dataset import rrpo as rrpo_mod

        print(f"[build_text_prefs] VQA ({args.dataset}) → {out}")
        prompt_config = rrpo_mod.get_rrpo_prompt_config()
        df = pd.read_csv(args.csv)
        train_data = df.to_dict("records")
        result_df, preference_data = asyncio.run(
            rrpo_mod.main(args.csv, train_data, prompt_config)
        )

        for rec in preference_data:
            rec["chosen"] = rrpo_mod._normalize_chosen_rejected(rec["chosen"])
            rec["rejected"] = rrpo_mod._normalize_chosen_rejected(rec["rejected"])

        def _has_mask_tags(text: str) -> bool:
            return "<mask>" in text and "</mask>" in text

        n_before = len(preference_data)
        preference_data = [
            rec
            for rec in preference_data
            if _has_mask_tags(rec["chosen"]) or _has_mask_tags(rec["rejected"])
        ]
        if n_before - len(preference_data):
            print(f"Dropped {n_before - len(preference_data)} samples with no <mask> tags.")

        seen: set[tuple] = set()
        unique_data = []
        for rec in preference_data:
            key = (rec["id"], rec["chosen"], rec["rejected"])
            if key not in seen:
                seen.add(key)
                unique_data.append(rec)
        preference_data = unique_data

        import json

        with open(out, "w", encoding="utf-8") as f:
            json.dump(preference_data, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(preference_data)} pairs → {out}")
        # optional side CSV
        if result_df is not None:
            side = Path(args.csv).with_name(Path(args.csv).stem + "_with_opp.csv")
            result_df.to_csv(side, index=False)
            print(f"Wrote side CSV → {side}")


if __name__ == "__main__":
    main()
