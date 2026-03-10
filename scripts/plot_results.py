import os
import numpy as np
import matplotlib.pyplot as plt
from pacmap_core import load_result


def scatter(ax, Y, colors, title=""):
    """Scatter a 2D embedding on an axis, auto-zoomed to fit the data."""
    ax.scatter(Y[:,0], Y[:,1], c=colors, s=0.3, cmap="Spectral", alpha=0.8, rasterized=True)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    # Zoom using 1st-99th percentile to ignore outliers, with 10% margin
    xlo, xhi = np.percentile(Y[:,0], [1, 99])
    ylo, yhi = np.percentile(Y[:,1], [1, 99])
    dx = max((xhi - xlo) * 0.10, 1e-3)
    dy = max((yhi - ylo) * 0.10, 1e-3)
    ax.set_xlim(xlo - dx, xhi + dx)
    ax.set_ylim(ylo - dy, yhi + dy)

def savefig(fig, name, d="figures"):
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {d}/{name}.png")

def _make_fig(n_rows, n_cols, cell=3.5):
    """Create a figure with uniform subplot sizes."""
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell*n_cols, cell*n_rows),
                             squeeze=False, constrained_layout=True)
    return fig, axes

def plot_scatter_grid(rows, colors, title):
    """
    Scatter grid. Each panel auto-zooms to its own data.
    rows = [(row_label, [(result_name, col_label), ...])]
    """
    n_rows = len(rows)
    n_cols = max(len(r[1]) for r in rows)
    fig, axes = _make_fig(n_rows, n_cols)
    for i, (row_label, cfgs) in enumerate(rows):
        for j, (name, label) in enumerate(cfgs):
            scatter(axes[i,j], load_result(name)["embedding"], colors, label)
        for j in range(len(cfgs), n_cols):
            axes[i,j].axis("off")
        axes[i,0].set_ylabel(row_label, fontsize=11, fontweight="bold",
                              rotation=0, labelpad=60, va="center")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig

def plot_bars(names, labels, title):
    """Bar chart of final triplet accuracy and neighbor preservation."""
    ta  = [load_result(n)["metrics"][-1]["triplet_acc"] for n in names]
    np_ = [load_result(n)["metrics"][-1]["neighbor_pres"] for n in names]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                    constrained_layout=True)

    b1 = ax1.bar(x, ta, 0.5, color="#2196F3", alpha=0.85)
    ax1.set_ylabel("Triplet Accuracy"); ax1.set_title("Global Structure", fontweight="bold")
    ax1.set_ylim(0.4, 1.0); ax1.axhline(0.5, color="gray", ls="--", alpha=0.5)
    for b, v in zip(b1, ta):
        ax1.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=8)

    b2 = ax2.bar(x, np_, 0.5, color="#FF9800", alpha=0.85)
    ax2.set_ylabel("Neighbor Preservation"); ax2.set_title("Local Structure", fontweight="bold")
    ax2.set_ylim(0, 0.75); ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    for b, v in zip(b2, np_):
        ax2.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=8)

    sep = next((i for i, n in enumerate(names) if "rand" in n), None)
    for ax in (ax1, ax2):
        if sep: ax.axvline(sep-0.5, color="red", ls=":", alpha=0.6)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig

def plot_lines(config_groups, title):
    """
    Line charts of metrics over iterations.
    config_groups = [(group_label, [(name, label, color), ...])]
    """
    n = len(config_groups)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5*n), squeeze=False,
                             constrained_layout=True)
    for row, (glabel, cfgs) in enumerate(config_groups):
        for name, label, color in cfgs:
            r = load_result(name)
            iters = [m["iter"] for m in r["metrics"]]
            axes[row,0].plot(iters, [m["triplet_acc"] for m in r["metrics"]],
                             label=label, color=color, lw=1.5)
            axes[row,1].plot(iters, [m["neighbor_pres"] for m in r["metrics"]],
                             label=label, color=color, lw=1.5)
        for c in range(2):
            axes[row,c].legend(fontsize=8); axes[row,c].grid(alpha=0.3)
            axes[row,c].axvline(100, color="gray", ls="--", alpha=0.3)
            axes[row,c].axvline(200, color="gray", ls="--", alpha=0.3)
        axes[row,0].set_title(f"{glabel} — Triplet Accuracy", fontweight="bold")
        axes[row,1].set_title(f"{glabel} — Neighbor Preservation", fontweight="bold")
    axes[-1,0].set_xlabel("Iteration"); axes[-1,1].set_xlabel("Iteration")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig

def plot_snapshots(config_list, colors, title):
    """Embedding evolution over iterations."""
    iters = [0, 10, 50, 100, 200, 450]
    nr, nc = len(config_list), len(iters)
    fig, axes = _make_fig(nr, nc)

    for row, (name, label) in enumerate(config_list):
        r = load_result(name)
        for col, t in enumerate(iters):
            scatter(axes[row,col], r["snapshots"][t], colors)
            if row == 0:
                axes[row,col].set_title(f"iter {t}", fontsize=9, fontweight="bold")
        axes[row,0].set_ylabel(label, fontsize=10, fontweight="bold",
                                rotation=0, labelpad=70, va="center")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig