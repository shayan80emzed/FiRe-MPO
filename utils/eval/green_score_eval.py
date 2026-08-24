"""
Run GREEN (Stanford AIMI) scoring on an inference CSV: references = ground-truth
answers, hypotheses = model outputs.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_green_on_path() -> None:
    green_root = _REPO_ROOT / "GREEN"
    if green_root.is_dir() and str(green_root) not in sys.path:
        sys.path.insert(0, str(green_root))


def _default_report_dir(csv_path: Path, base_out: Path) -> Path:
    stem = csv_path.stem
    return base_out / stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GREEN scoring on a CSV with reference answers and predicted outputs."
    )
    parser.add_argument(
        "--csv",
        "--csv_path",
        dest="csv_path",
        type=Path,
        required=True,
        help="Path to CSV (e.g. experiments/.../None_1_test.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_REPO_ROOT / "experiments" / "green_reports",
        help="Base directory; a subfolder named after the CSV stem holds this run.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="StanfordAIMI/GREEN-radllama2-7b",
        help="Hugging Face model id for GREEN.",
    )
    parser.add_argument(
        "--ref_column",
        type=str,
        default="answer",
        help="Column name for reference (ground-truth) text.",
    )
    parser.add_argument(
        "--hyp_column",
        type=str,
        default="output",
        help="Column name for hypothesis (model output) text.",
    )
    args = parser.parse_args()

    _ensure_green_on_path()
    import pandas as pd
    from green_score import GREEN

    csv_path = args.csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for col in (args.ref_column, args.hyp_column):
        if col not in df.columns:
            raise SystemExit(
                f"Missing column {col!r}. Available: {list(df.columns)}"
            )

    refs = df[args.ref_column].fillna("").astype(str).tolist()
    hyps = df[args.hyp_column].fillna("").astype(str).tolist()

    report_dir = _default_report_dir(csv_path, args.output_dir.resolve())
    report_dir.mkdir(parents=True, exist_ok=True)

    green_scorer = GREEN(args.model_name, output_dir=str(report_dir))
    mean, std, _green_scores, summary, result_df = green_scorer(refs, hyps)

    # Attach source row identifiers when present
    extra_cols = [c for c in ("id", "qid", "sequence_id", "augmentation_type") if c in df.columns]
    if extra_cols:
        meta = df[extra_cols].reset_index(drop=True)
        result_df = pd.concat([meta, result_df.reset_index(drop=True)], axis=1)

    out_csv = report_dir / "green_results.csv"
    result_df.to_csv(out_csv, index=False)

    summary_path = report_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary if summary is not None else "")

    def _json_float(x):
        if x is None:
            return None
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None
        return xf

    metrics = {
        "csv_path": str(csv_path),
        "model_name": args.model_name,
        "n_samples": len(refs),
        "mean_green": _json_float(mean),
        "std_green": _json_float(std),
        "report_dir": str(report_dir),
    }
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(summary)
    print(f"Mean GREEN: {mean}, std: {std}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {report_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
