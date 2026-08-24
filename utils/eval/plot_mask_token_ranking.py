"""
Render the mask-token reward-ranking metrics produced by
``utils/eval/mask_token_ranking.py`` as a horizontal grouped bar chart
(one row per run, in the style of the GOAT / Crescendo ASR figure).

Each input is a per-sample CSV written by ``mask_token_ranking.py``. If a
sidecar ``<csv_stem>.summary.json`` exists, the global-pool metric
``global_frac_masked_in_top_p_pct`` is read from it (it cannot be recovered
from the CSV alone); per-sample metrics ``any_in_top_p_pct`` and
``frac_masked_in_top_p_pct`` are averaged directly from the CSV over
rows with ``n_masked_tokens > 0``.

The metric families correspond to color families (e.g. red for ``any``,
blue for ``frac``), and percentiles correspond to shades within a family
(darker = stricter / smaller top-X% cutoff). Values are shown as percentages
(``×100``) so the chart reads like the reference figure.

Activate the project venv first (``source dpo_env/bin/activate`` from the repo root).

Example::

    # Overlay every *.csv under experiments/mask_token_ranking (default --csv_dir)
    python utils/eval/plot_mask_token_ranking.py --figure_path ./compare_mtr.png

    # Single CSV
    python utils/eval/plot_mask_token_ranking.py \\
        --csv_path ./experiments/mask_token_ranking/huatuo-fire-mpo-slake__rrpo.csv \\
        --figure_path ./mtr_fig.png

    # Show 3 percentiles and add the global-pool metric (needs summary JSONs)
    python utils/eval/plot_mask_token_ranking.py \\
        --percentiles 5 10 50 --metrics any frac global \\
        --figure_path ./compare_mtr.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

_DEFAULT_CSV_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "experiments", "mask_token_ranking",
    )
)

_LEGEND_NAME_MAP: Dict[str, str] = {
    "HuatuoGPT-Vision-7B-Qwen2.5VL": "base",
    "huatuo-dpo-slake": "base+DPO",
    "huatuo-fire-mpo-slake": "base+FiRe-MPO",
    "qwen3-4B-slake-mask-dpo": "Qwen3-4B+DPO",
    "qwen3-4B-fire-mpo-slake": "Qwen3-4B+FiRe-MPO",
}

# Color families per metric (dark -> light goes with stricter -> looser percentile).
_METRIC_CMAPS: Dict[str, str] = {
    "any": "Reds",
    "frac": "Blues",
    "global": "Greens",
}

_METRIC_LABELS: Dict[str, str] = {
    "any": "any masked tok in top {p}%",
    "frac": "frac masked toks in top {p}%",
    "global": "global frac masked in top {p}%",
}

_CSV_COL_PATTERNS: Dict[str, str] = {
    "any": "any_in_top_{p}_pct",
    "frac": "frac_masked_in_top_{p}_pct",
    "global": "global_frac_masked_in_top_{p}_pct",
}


def _parse_float_cell(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _detect_percentiles_in_csv(fieldnames: Sequence[str]) -> List[int]:
    """Pick percentiles ``p`` such that ``any_in_top_{p}_pct`` is in the CSV."""
    rx = re.compile(r"^any_in_top_(\d+)_pct$")
    out: List[int] = []
    for f in fieldnames:
        m = rx.match(f)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _load_csv_aggregates(csv_path: str) -> Tuple[Dict[str, float], int, List[int]]:
    """
    Aggregate per-sample CSV to dataset-level numbers (mean over rows with
    ``n_masked_tokens > 0``). Returns ``(values, n_samples_with_mask, percentiles)``
    where ``values`` maps CSV column names to mean values.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    percentiles = _detect_percentiles_in_csv(fields)
    if not percentiles:
        raise ValueError(
            f"CSV {csv_path} has no any_in_top_<p>_pct columns; "
            "did you generate it with utils/eval/mask_token_ranking.py?"
        )

    # Keep only samples that actually have masked tokens.
    kept = []
    for r in rows:
        n_masked = r.get("n_masked_tokens", "0").strip()
        try:
            if int(n_masked) > 0:
                kept.append(r)
        except ValueError:
            continue

    out: Dict[str, float] = {}
    for col in fields:
        if not (col.startswith("any_in_top_") or col.startswith("frac_masked_in_top_")):
            continue
        vals = [_parse_float_cell(r.get(col, "")) for r in kept]
        vals = [v for v in vals if not math.isnan(v)]
        out[col] = float(np.mean(vals)) if vals else float("nan")
    return out, len(kept), percentiles


def _load_sidecar_summary(csv_path: str) -> Dict[str, float]:
    """Return ``{column_name: value}`` from ``<csv_stem>.summary.json`` if present.

    The JSON keys are the same names used as CSV columns where applicable
    (``any_in_top_{p}_pct``, ``frac_masked_in_top_{p}_pct``,
    ``global_frac_masked_in_top_{p}_pct``).
    """
    stem, _ = os.path.splitext(csv_path)
    candidate = stem + ".summary.json"
    if not os.path.isfile(candidate):
        return {}
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        if k.startswith("any_in_top_") or k.startswith("frac_masked_in_top_") or k.startswith(
            "global_frac_masked_in_top_"
        ):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(fv) or math.isinf(fv):
                continue
            out[k] = fv
    return out


def _legend_label_for_path(csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    # CSVs from mask_token_ranking.py default to ``<run>__<dataset_stem>``;
    # the run identifier (before "__") is what we want for the row label.
    run = stem.split("__", 1)[0] if "__" in stem else stem
    return _LEGEND_NAME_MAP.get(run, run)


def _csv_paths_in_dir(csv_dir: str) -> List[str]:
    if not os.path.isdir(csv_dir):
        raise ValueError(f"Not a directory: {csv_dir}")
    names = sorted(
        n for n in os.listdir(csv_dir)
        if n.lower().endswith(".csv") and os.path.isfile(os.path.join(csv_dir, n))
    )
    if not names:
        raise ValueError(f"No .csv files found in {csv_dir}")
    return [os.path.join(csv_dir, n) for n in names]


def _shade_for_percentile(metric: str, percentile_rank: int, n_percentiles: int) -> Tuple[float, float, float, float]:
    """
    Pick a shade from the metric's colormap. Darker = stricter (smaller p%).
    ``percentile_rank`` is 0-based index into the sorted (ascending) percentile list.
    """
    cmap = plt.get_cmap(_METRIC_CMAPS.get(metric, "Greys"))
    if n_percentiles <= 1:
        t = 0.75
    else:
        # Reserve the very-dark end of the colormap for the strictest percentile.
        # Map rank 0 -> 0.85 (dark) and rank n-1 -> 0.35 (light).
        t = 0.85 - (percentile_rank / (n_percentiles - 1)) * 0.50
    return cmap(t)


def _value_for_bar(
    metric: str,
    percentile: int,
    csv_vals: Dict[str, float],
    json_vals: Dict[str, float],
) -> float:
    col = _CSV_COL_PATTERNS[metric].format(p=percentile)
    if metric == "global":
        v = json_vals.get(col, float("nan"))
    else:
        v = csv_vals.get(col, float("nan"))
        if math.isnan(v) and col in json_vals:
            v = json_vals[col]
    return v


def plot_from_csv_paths(
    csv_paths: Sequence[str],
    figure_path: str,
    *,
    metrics: Sequence[str],
    percentiles: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    show_counts: bool = True,
) -> None:
    """
    Render a horizontal grouped-bar plot. Each input CSV becomes one row;
    inside each row there is one bar per ``(metric, percentile)`` combination.
    """
    paths = [os.path.abspath(p) for p in csv_paths]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    for m in metrics:
        if m not in _METRIC_CMAPS:
            raise ValueError(f"Unknown metric {m!r}; choose from {sorted(_METRIC_CMAPS)}")

    aggregates: List[Tuple[Dict[str, float], Dict[str, float], int, List[int]]] = []
    auto_percentiles_union: List[int] = []
    for csv_path in paths:
        csv_vals, n_kept, pcts = _load_csv_aggregates(csv_path)
        json_vals = _load_sidecar_summary(csv_path)
        aggregates.append((csv_vals, json_vals, n_kept, pcts))
        for p in pcts:
            if p not in auto_percentiles_union:
                auto_percentiles_union.append(p)
    auto_percentiles_union = sorted(auto_percentiles_union)

    if percentiles is None or len(percentiles) == 0:
        # Default: take whatever the CSV(s) contain
        used_pcts = list(auto_percentiles_union)
    else:
        used_pcts = sorted(set(int(p) for p in percentiles))
    if not used_pcts:
        raise ValueError("No percentiles available to plot.")

    n_rows = len(paths)
    n_metrics = len(metrics)
    n_pcts = len(used_pcts)
    bars_per_row = n_metrics * n_pcts

    # Geometry: matches the grouped-bar look of the reference image. Scale the
    # group height so each bar is ~0.18 inch tall regardless of bars_per_row,
    # and add a small inter-group gap.
    bar_height_in = 0.20
    group_gap_in = 0.18
    row_in = bars_per_row * bar_height_in + group_gap_in
    # In data coords each row occupies [y-0.5, y+0.5], so the group fills 0.82
    # of that span (leaves 9% gap on each side between rows).
    group_height = 0.82
    bar_height = group_height / bars_per_row
    y_positions = np.arange(n_rows)
    base_offsets = (
        -group_height / 2 + bar_height / 2
        + np.arange(bars_per_row) * bar_height
    )

    # Reserve vertical space for the legend (above the axes) and the figure title.
    n_legend_cols = min(bars_per_row, 4)
    n_legend_rows = max(1, math.ceil(bars_per_row / n_legend_cols))
    fig_h = max(2.6, n_rows * row_in + 0.45 * n_legend_rows + 1.2)
    fig_w = 11.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")
    ax.set_facecolor("#eef0f4")

    # Bars are stacked top->bottom inside each row group; reverse so the
    # *first* metric in the user's --metrics list ends up on the *top* bar of
    # each row (matches the GOAT/Crescendo convention in the reference figure).
    flat_index = 0
    for m_idx, metric in enumerate(metrics):
        for p_idx, p in enumerate(used_pcts):
            color = _shade_for_percentile(metric, p_idx, n_pcts)
            label = _METRIC_LABELS[metric].format(p=p)
            values_pct: List[float] = []
            for csv_vals, json_vals, _n_kept, _pcts in aggregates:
                v = _value_for_bar(metric, p, csv_vals, json_vals)
                values_pct.append(float("nan") if math.isnan(v) else 100.0 * v)
            # Flip vertical order so flat_index 0 is the top-most bar of each
            # row (matches the legend order shown at the top of the figure).
            slot = bars_per_row - 1 - flat_index
            ys = y_positions + base_offsets[slot]
            bar_vals = [0.0 if math.isnan(v) else v for v in values_pct]
            bars = ax.barh(
                ys, bar_vals, height=bar_height * 0.95,
                color=color, edgecolor="white", linewidth=0.7,
                label=label, zorder=3,
            )
            # Value labels at bar tips, formatted as percentages.
            for b, v in zip(bars, values_pct):
                if math.isnan(v):
                    ax.text(
                        1.0, b.get_y() + b.get_height() / 2.0, "n/a",
                        ha="left", va="center", fontsize=8,
                        color="#444444", zorder=4,
                    )
                    continue
                txt = f"{v:.1f}%"
                # Inside the bar when wide enough, otherwise just to the right.
                inside_threshold = 18.0
                if v >= inside_threshold:
                    ax.text(
                        v - 1.2, b.get_y() + b.get_height() / 2.0, txt,
                        ha="right", va="center", fontsize=8.5,
                        color="white", fontweight="bold", zorder=5,
                    )
                else:
                    ax.text(
                        v + 0.7, b.get_y() + b.get_height() / 2.0, txt,
                        ha="left", va="center", fontsize=8.5,
                        color="#202020", zorder=5,
                    )
            flat_index += 1

    # Row labels
    row_labels = []
    for i, csv_path in enumerate(paths):
        base_label = _legend_label_for_path(csv_path)
        if show_counts:
            _csv_vals, _json_vals, n_kept, _pcts = aggregates[i]
            row_labels.append(f"{base_label}\n(n={n_kept})")
        else:
            row_labels.append(base_label)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(row_labels)
    ax.invert_yaxis()  # first input on top, like the reference figure

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels([f"{t}%" for t in [0, 20, 40, 60, 80, 100]])
    ax.set_xlabel("Per-sample mean / global share (%)")
    ax.xaxis.grid(True, color="white", linewidth=1.2, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors="#444444")

    if title is None:
        title = "Mask-token reward ranking (higher = masked tokens get higher PEFT-vs-base reward)"

    # Top legend in the order users see metrics × percentiles (top-most bar first).
    legend_handles: List[Patch] = []
    for m_idx, metric in enumerate(metrics):
        for p_idx, p in enumerate(used_pcts):
            color = _shade_for_percentile(metric, p_idx, n_pcts)
            legend_handles.append(
                Patch(facecolor=color, edgecolor="white",
                      label=_METRIC_LABELS[metric].format(p=p))
            )
    # Place the legend ABOVE the axes; constrained_layout will give it room.
    ax.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, 1.02), ncol=n_legend_cols,
        frameon=False, fontsize=9, handlelength=1.4, handleheight=1.0,
        columnspacing=1.4, borderaxespad=0.0,
    )
    fig.suptitle(title, fontsize=11)

    out_dir = os.path.dirname(figure_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(figure_path, dpi=170, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mask-token reward ranking metrics (from utils/eval/mask_token_ranking.py "
            "CSV/JSON output) as a horizontal grouped bar chart, one row per run."
        )
    )
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Single per-sample CSV; if set, --csv_dir is ignored.",
    )
    parser.add_argument(
        "--csv_dir", type=str, default=_DEFAULT_CSV_DIR,
        help=f"Directory of *.csv to overlay (default: {_DEFAULT_CSV_DIR}).",
    )
    parser.add_argument(
        "--figure_path", type=str, default="./mtr_compare.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["any", "frac"],
        choices=sorted(_METRIC_CMAPS.keys()),
        help=(
            "Metric families to plot. 'any' and 'frac' are read from the CSV; "
            "'global' requires the sidecar <csv_stem>.summary.json."
        ),
    )
    parser.add_argument(
        "--percentiles", nargs="*", type=int, default=None,
        help=(
            "Top-X percentiles to plot (e.g. 10 50). "
            "Default: every percentile present in the CSV(s)."
        ),
    )
    parser.add_argument(
        "--title", type=str, default=None,
        help="Figure title (default: a generic mask-token-ranking title).",
    )
    parser.add_argument(
        "--no_counts", action="store_true",
        help="Hide the (n=...) sample-count suffix on row labels.",
    )
    args = parser.parse_args()

    if args.csv_path:
        paths = [args.csv_path]
    else:
        paths = _csv_paths_in_dir(args.csv_dir)
    plot_from_csv_paths(
        paths, args.figure_path,
        metrics=args.metrics,
        percentiles=args.percentiles,
        title=args.title,
        show_counts=not args.no_counts,
    )
    print(f"Saved figure to {args.figure_path} ({len(paths)} CSV(s))")


if __name__ == "__main__":
    main()
