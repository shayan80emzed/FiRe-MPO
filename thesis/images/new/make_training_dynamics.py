#!/usr/bin/env python3
"""
Training-dynamics figure for the thesis (Section: Results / Training Dynamics).

Reads the real ``trainer_state.json`` log histories written by the HuggingFace
trainer during the SLAKE alignment runs and produces a four-panel figure:

  (a) reward accuracy on the masked spans   -- comparable across objectives
  (b) reward margin on the masked spans     -- comparable across objectives
  (c) FiRe-MPO loss decomposition           -- relative size of the three terms
  (d) gradient norm                         -- optimization stability

Panels (a), (b), (d) compare DPO against FiRe-MPO on both backbones; colour
encodes the method and line style encodes the backbone, so identity never rests
on colour alone.

Usage:
    python thesis/images/new/make_training_dynamics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "training_dynamics.pdf"

# Validated categorical slots 1-3 (light surface). See dataviz reference palette.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

RUNS = {
    ("HuatuoGPT-Vision-7B", "DPO"): "models/huatuo-dpo-slake/checkpoint-611",
    ("HuatuoGPT-Vision-7B", "FiRe-MPO"): "models/huatuo-fire-mpo-slake/checkpoint-611",
    ("Qwen3-VL-4B", "DPO"): "models/qwen3-4B-slake-dpo/checkpoint-613",
    ("Qwen3-VL-4B", "FiRe-MPO"): "models/qwen3-4B-fire-mpo-slake/checkpoint-610",
}

METHOD_COLOR = {"DPO": ORANGE, "FiRe-MPO": BLUE}
BACKBONE_STYLE = {"HuatuoGPT-Vision-7B": "-", "Qwen3-VL-4B": (0, (4, 2))}

# Loss coefficients from the default configuration (Table: hyperparameters).
GAMMA, ALPHA = 0.1, 0.01


def load(rel: str) -> list[dict]:
    path = REPO / rel / "trainer_state.json"
    with open(path) as fh:
        state = json.load(fh)
    return [e for e in state["log_history"] if "loss" in e]


def series(hist: list[dict], key: str):
    pts = [(e["step"], e[key]) for e in hist if key in e]
    return [p[0] for p in pts], [p[1] for p in pts]


def style_axis(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=6)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK_MUTED)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_MUTED)
    ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main() -> None:
    hists = {k: load(v) for k, v in RUNS.items()}

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    fig.patch.set_facecolor("white")

    # ---- (a) reward accuracy, (b) reward margin, (d) grad norm ----
    panels = [
        (axes[0][0], "rewards/accuracies", "Reward accuracy",
         "(a) Reward accuracy on masked spans"),
        (axes[0][1], "rewards/margins", "Reward margin",
         "(b) Reward margin on masked spans"),
        (axes[1][1], "grad_norm", "Gradient norm",
         "(d) Gradient norm"),
    ]
    for ax, key, ylabel, title in panels:
        for (backbone, method), hist in hists.items():
            x, y = series(hist, key)
            ax.plot(
                x, y,
                color=METHOD_COLOR[method],
                linestyle=BACKBONE_STYLE[backbone],
                linewidth=2.0, solid_capstyle="round",
            )
        style_axis(ax, "Training step", ylabel, title)
    axes[1][1].set_yscale("log")

    # ---- (c) FiRe-MPO loss decomposition, weighted as they enter the objective ----
    ax = axes[1][0]
    hist = hists[("Qwen3-VL-4B", "FiRe-MPO")]
    terms = [
        ("loss/rrpo_base", 1.0, BLUE, r"base ranking"),
        ("loss/rrpo_v3", GAMMA, ORANGE, r"$\gamma\cdot$visual"),
        ("mixed_tkl_loss", ALPHA, AQUA, r"$\alpha\cdot$KL"),
    ]
    for key, coef, color, label in terms:
        x, y = series(hist, key)
        y = [coef * v for v in y]
        ax.plot(x, y, color=color, linewidth=2.0, solid_capstyle="round")
        # Direct label at the right end (relief rule: aqua is below 3:1 on white).
        ax.annotate(
            label, xy=(x[-1], y[-1]), xytext=(5, 0), textcoords="offset points",
            fontsize=8, color=INK, va="center", ha="left",
        )
    style_axis(ax, "Training step", "Contribution to loss",
               "(c) FiRe-MPO loss decomposition (Qwen3-VL-4B)")
    # Headroom for the direct labels, without inventing tick marks past the data.
    last_step = series(hist, "loss/rrpo_base")[0][-1]
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600])
    ax.set_xlim(left=-15, right=last_step * 1.26)

    # ---- shared legend: colour = method, line style = backbone ----
    handles = [
        plt.Line2D([], [], color=METHOD_COLOR["DPO"], lw=2.0, label="DPO"),
        plt.Line2D([], [], color=METHOD_COLOR["FiRe-MPO"], lw=2.0, label="FiRe-MPO"),
        plt.Line2D([], [], color=INK_MUTED, lw=2.0,
                   linestyle=BACKBONE_STYLE["HuatuoGPT-Vision-7B"],
                   label="HuatuoGPT-Vision-7B"),
        plt.Line2D([], [], color=INK_MUTED, lw=2.0,
                   linestyle=BACKBONE_STYLE["Qwen3-VL-4B"], label="Qwen3-VL-4B"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
