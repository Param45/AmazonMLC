"""Small matplotlib helpers used by the notebook (static charts, consistent styling).

Colour roles: slot 1 blue = matches / the main series, slot 2 orange = non-matches / comparison,
slot 3 aqua = third series. Ink and grid colours stay neutral; every multi-series chart has a legend.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def use_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.labelsize": 9, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0, "figure.dpi": 100,
        "font.family": "sans-serif", "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def grouped_bars(table: pd.DataFrame, title: str, ylabel: str = "share of entities", ax=None):
    """Rows = categories on the x axis, columns = series (fixed colour order)."""
    ax = ax or plt.subplots(figsize=(7, 3.2))[1]
    n_rows, n_cols = table.shape
    width = 0.8 / max(n_cols, 1)
    x = np.arange(n_rows)
    for j, col in enumerate(table.columns):
        ax.bar(x + (j - (n_cols - 1) / 2) * width, table[col].values, width=width * 0.92,
               color=SERIES[j % len(SERIES)], label=str(col))
    ax.set_xticks(x, [str(i) for i in table.index])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if n_cols > 1:
        ax.legend(ncols=min(n_cols, 4), loc="upper right")
    return ax


def compare_hist(df: pd.DataFrame, features: Sequence[str], label_col: str = "label",
                 order: Sequence[str] = ("match", "random non-match"), title: Optional[str] = None, bins: int = 40):
    """Small multiples: one panel per feature, step histograms per label (density)."""
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.0), squeeze=False)
    for ax, feat in zip(axes[0], features):
        edges = np.linspace(0, 1, bins + 1)
        for j, lab in enumerate(order):
            vals = df.loc[df[label_col] == lab, feat].dropna().clip(0, 1)
            if len(vals):
                ax.hist(vals, bins=edges, density=True, histtype="step", linewidth=2, color=SERIES[j], label=lab)
        ax.set_title(feat)
        ax.set_xlabel("similarity")
        ax.set_yticks([])
    axes[0][0].legend(loc="upper center")
    if title:
        fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return fig


def importance_bars(importance: pd.Series, top: int = 20, title: str = "Feature importance (share of total gain)"):
    s = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(s) + 1))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.7)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(s.values):
        if y >= len(s) - 3:   # label only the top three bars
            ax.text(v, y, f" {v:.1%}", va="center", fontsize=8, color=INK_2)
    fig.tight_layout()
    return fig


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10, min_count: int = 20,
                title: str = "Calibration of out-of-fold probabilities"):
    """Reliability diagram on fixed-width bins (bins with fewer than `min_count` pairs are skipped)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    df = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    df = df[df["n"] >= min_count]
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="perfect calibration")
    ax.plot(df["p"], df["y"], marker="o", markersize=6, color=SERIES[0], label="model (10 equal-width bins)")
    for _, r in df.iterrows():
        ax.annotate(f"{int(r['n']):,}", (r["p"], r["y"]), textcoords="offset points", xytext=(4, -10),
                    fontsize=7, color=MUTED)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed match rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def threshold_curve(full: pd.DataFrame, best_threshold: Optional[float] = None,
                    title: str = "Macro F0.5 vs global threshold (out-of-fold)"):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    rows = full[full["family"] == "threshold"].copy()
    rows["t"] = rows["policy"].str.extract(r"p>=([0-9.]+)").astype(float)
    for j, (excl, sub) in enumerate(rows.groupby("exclusive")):
        sub = sub.sort_values("t")
        ax.plot(sub["t"], sub["macro_F0.5 (OOF)"], color=SERIES[j], label="exclusive targets" if excl else "independent pairs")
    if best_threshold is not None:
        ax.axvline(best_threshold, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("macro F0.5")
    ax.set_title(title)
    ax.legend(loc="lower center")
    fig.tight_layout()
    return fig


def max_prob_by_group(scored: pd.DataFrame, group_of_s1: dict, max_groups: int = 3,
                      title: str = "Share of S1 entities whose best candidate reaches probability p"):
    """Survival curve of each entity's best candidate probability, one line per group (e.g. country)."""
    best = scored.groupby("s1_id")["p"].max()
    groups = best.index.map(group_of_s1)
    grid = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    labels = pd.Series(groups).value_counts().index[:max_groups]
    for j, g in enumerate(labels):
        vals = np.sort(best.values[np.asarray(groups == g)])
        share = 1.0 - np.searchsorted(vals, grid, side="left") / max(len(vals), 1)
        ax.plot(grid, share, color=SERIES[j], label=f"{g} (n={len(vals):,})")
    ax.set_xlabel("probability p")
    ax.set_ylabel("share of entities with best p >= x")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig
