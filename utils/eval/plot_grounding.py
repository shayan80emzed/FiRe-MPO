"""
Plot **AR**, **KL**, and **JS** vs decoder layer from metrics CSV(s) produced by
``visual_grounding.py`` (``--csv_path``).

Pass a single CSV file, or a directory of ``*.csv`` files (each series is labeled
by the file stem and shown in all three panels).

Activate the project venv first (``source dpo_env/bin/activate`` from the repo root).

Example::

    python utils/eval/plot_grounding.py --csv_path ./visual_grounding_metrics.csv \\
        --figure_path ./visual_grounding_fig.png

    # Overlay every *.csv under experiments/visual_grounding (default --csv_dir)
    python utils/eval/plot_grounding.py --figure_path ./compare.png

    python utils/eval/plot_grounding.py --csv_dir ./runs/metrics/ --figure_path ./compare.png
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_DEFAULT_CSV_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "visual_grounding")
)

_ROW_LABEL_HUATUO = "HuatuoGPT-Vision 7B"
_ROW_LABEL_QWEN = "Qwen3-VL-4B-Instruct"


def _parse_float_cell(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("nan")
    return float(s)


def read_grounding_metrics_csv(
    csv_path: str,
) -> Tuple[List[Dict[str, str]], List[int], List[float], List[float], List[float]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    required = {"layer_index", "ar", "kl", "js"}
    if not required.issubset(set(rows[0].keys())):
        raise ValueError(f"CSV missing required columns {required}; got {list(rows[0].keys())}")
    rows_sorted = sorted(rows, key=lambda r: int(r["layer_index"]))
    xs = [int(r["layer_index"]) for r in rows_sorted]
    ars = [_parse_float_cell(r["ar"]) for r in rows_sorted]
    kls = [_parse_float_cell(r["kl"]) for r in rows_sorted]
    jss = [_parse_float_cell(r["js"]) for r in rows_sorted]
    return rows_sorted, xs, ars, kls, jss


def default_title_from_rows(rows: List[Dict[str, str]]) -> str:
    r0 = rows[0]
    mode = r0.get("mode", "")
    qp = r0.get("query_position", "")
    sid = (r0.get("sample_id") or "").strip()
    base = os.path.basename(r0.get("dataset_json", "") or "")
    tail = f"sample {sid}" if sid else "dataset mean"
    nlayer = len(rows)
    crow = (r0.get("contributing_rows_layer0") or "").strip()
    extra = f" | contributing_rows={crow}" if crow else ""
    peft = (r0.get("peft_model_path") or "").strip()
    adapter = " | PEFT" if peft else ""
    return f"{tail} | {mode} | {qp} | L={nlayer}{extra}{adapter} | {base}"


def _plot_series(
    ax: plt.Axes,
    x: List[int],
    y: List[float],
    *,
    panel_title: str,
    color: str,
    nan_message: str,
    label: Optional[str] = None,
) -> None:
    y_clean = [v if v == v else float("nan") for v in y]
    has_data = any(v == v for v in y_clean)
    if not has_data:
        if label is None:
            ax.text(0.5, 0.5, nan_message, transform=ax.transAxes, ha="center", va="center", fontsize=9)
    else:
        ax.plot(
            x,
            y_clean,
            color=color,
            linewidth=1.5,
            linestyle="-",
            alpha=0.95,
            label=label,
        )
    ax.set_xlabel("Layer")
    ax.set_title(panel_title)
    ax.grid(True, alpha=0.3)


def _legend_label_for_path(csv_path: str) -> str:
    """Map a metrics CSV to its variant label from the file stem.

    Works for both the HuatuoGPT and Qwen3-VL families (e.g. ``huatuo-dpo-slake``
    and ``qwen3-4B-slake-dpo`` both -> ``base+DPO``).
    """
    stem = os.path.splitext(os.path.basename(csv_path))[0].lower()
    if "dpo" in stem:
        return "base+DPO"
    if "fire" in stem or "mpo" in stem:
        return "base+FiRe-MPO"
    return "base"


def _row_label_for_paths(csv_paths: Sequence[str]) -> str:
    """Model-family label shown on the left of the figure.

    HuatuoGPT checkpoints carry ``qwen2.5vl`` in the base name, so check for
    ``huatuo`` first; anything else containing ``qwen`` is the Qwen3-VL family.
    """
    stems = [os.path.basename(p).lower() for p in csv_paths]
    if any("huatuo" in s for s in stems):
        return _ROW_LABEL_HUATUO
    if any("qwen" in s for s in stems):
        return _ROW_LABEL_QWEN
    return _ROW_LABEL_HUATUO


def _csv_paths_in_dir(csv_dir: str) -> List[str]:
    if not os.path.isdir(csv_dir):
        raise ValueError(f"Not a directory: {csv_dir}")
    names = sorted(
        n for n in os.listdir(csv_dir) if n.lower().endswith(".csv") and os.path.isfile(os.path.join(csv_dir, n))
    )
    if not names:
        raise ValueError(f"No .csv files found in {csv_dir}")
    return [os.path.join(csv_dir, n) for n in names]


def plot_from_csv_paths(
    csv_paths: Sequence[str],
    figure_path: str,
    title: Optional[str] = None,
) -> None:
    paths = [os.path.abspath(p) for p in csv_paths]
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    first_rows, _, _, _, _ = read_grounding_metrics_csv(paths[0])
    if title is not None:
        use_title = title
    elif len(paths) == 1:
        use_title = default_title_from_rows(first_rows)
    else:
        stems = ", ".join(_legend_label_for_path(p) for p in paths)
        use_title = f"Visual grounding vs layer ({len(paths)} CSVs): {stems}"

    cmap = plt.get_cmap("tab10")
    # Pin a color per variant so FiRe-MPO is always green and DPO always orange,
    # regardless of filename sort order; unknown labels fall back to position.
    variant_color = {
        "base": cmap(0),
        "base+DPO": cmap(1),
        "base+FiRe-MPO": cmap(2),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), sharex=True)
    ax_ar, ax_kl, ax_js = axes

    legend_handles: List[Line2D] = []
    for i, csv_path in enumerate(paths):
        rows, xs, ars, kls, jss = read_grounding_metrics_csv(csv_path)
        leg = _legend_label_for_path(csv_path)
        c = variant_color.get(leg, cmap(i % 10))
        if len(paths) > 1:
            legend_handles.append(Line2D([0], [0], color=c, linewidth=2, linestyle="-", label=leg))
        _plot_series(
            ax_ar,
            xs,
            ars,
            panel_title=r"Attention Ratio (AR) $\uparrow$",
            color=c,
            nan_message="AR undefined",
            label=None,
        )
        _plot_series(
            ax_kl,
            xs,
            kls,
            panel_title=r"KL Divergence $\downarrow$",
            color=c,
            nan_message="KL undefined",
            label=None,
        )
        _plot_series(
            ax_js,
            xs,
            jss,
            panel_title=r"JS Divergence $\downarrow$",
            color=c,
            nan_message="JS undefined",
            label=None,
        )

    fig.suptitle(use_title, fontsize=10, y=0.985)
    fig.supylabel(_row_label_for_paths(paths), fontsize=12, fontweight="bold", x=0.012)
    if legend_handles:
        legend_order = {"base": 0, "base+DPO": 1, "base+FiRe-MPO": 2}
        legend_handles.sort(
            key=lambda h: legend_order.get(h.get_label(), len(legend_order))
        )
        ncol = min(len(legend_handles), 6)
        fig.subplots_adjust(
            left=0.07, right=0.995, top=0.84, bottom=0.24, wspace=0.20
        )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=ncol,
            fontsize=12,
            frameon=True,
        )
    else:
        fig.subplots_adjust(
            left=0.07, right=0.995, top=0.84, bottom=0.14, wspace=0.20
        )
    out_dir = os.path.dirname(figure_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_from_csv(csv_path: str, figure_path: str, title: Optional[str] = None) -> None:
    plot_from_csv_paths([csv_path], figure_path, title=title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AR / KL / JS vs layer from visual_grounding.py CSV output (one file or a folder)"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Single metrics CSV; if set, --csv_dir is ignored",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default=_DEFAULT_CSV_DIR,
        help=f"Directory of *.csv to overlay (default: {_DEFAULT_CSV_DIR})",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default="./vg_all_fig.png",
        help="Output image path",
    )
    parser.add_argument("--title", type=str, default=None, help="Figure title (default: infer from CSV(s))")
    args = parser.parse_args()
    if args.csv_path:
        paths = [args.csv_path]
    else:
        paths = _csv_paths_in_dir(args.csv_dir)
    plot_from_csv_paths(paths, args.figure_path, title=args.title)
    print(f"Saved figure to {args.figure_path} ({len(paths)} CSV(s))")


if __name__ == "__main__":
    main()
