from __future__ import annotations

import test
"""Reusable plotting helpers with a unified seaborn / matplotlib style."""
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.collections as mcoll
import numpy as np
from matplotlib.axes import Axes
from typing import Sequence
import pandas as pd
from aeon.visualisation import plot_critical_difference as cd

sns.set_theme(style="ticks", font_scale=1.1)

__all__ = ["metric_box", "compare_models", "metric_grid", 
           "plot_confidence", "plot_detection", "critical_difference", "window_bar"]

def metric_box(df: pd.DataFrame, metric: str, *, x="model",
               hue=None, order=None, title=None,
               ax: Axes | None = None, grid_ax=None):
    """Box‑plot of *metric* by model. Returns the Axes for further tweaking."""
    ax = ax or plt.gca()
    sns.boxplot(data=df, x=x, y=metric, order=order, hue=hue, ax=ax, width=.5, palette='tab10')
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    if grid_ax is not None:
        ax.grid(axis=grid_ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sns.despine()
    return ax

def compare_models(df_list: list[pd.DataFrame], metric: str, *, labels=None):
    """Create a 1×N grid of box‑plots to compare datasets or configs."""
    n = len(df_list)
    fig, axs = plt.subplots(1, n, figsize=(4*n, 4), sharey=True, layout="constrained")
    if n == 1:
        axs = [axs]
    for ax, d, lbl in zip(axs, df_list, labels or [None]*n):
        metric_box(d, metric, ax=ax, title=lbl)
    plt.show()

def metric_grid(
    df: pd.DataFrame,
    metrics: list[str],
    *,
    order: list[str] | None = None,
    n_cols: int = 2,
    fig_size: tuple[int, int] | None = None,
    suptitle: str | None = None,
    **kwargs
):
    """Display many metrics for one dataset in a tidy grid.

    Parameters
    ----------
    df : DataFrame
        Source metrics table (one row per model).
    metrics : list[str]
        List of column names to visualise.
    order : list[str], optional
        Fixed order of models on the x‑axis; defaults to alphabetical.
    n_cols : int, default=2
        Number of columns in the grid.
    fig_size : (w, h), optional
        Overrides automatic figure size.
    suptitle : str, optional
        Super‑title for the whole figure.
    """
    n = len(metrics)
    n_cols = max(1, n_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=fig_size or (4 * n_cols, 4 * n_rows),
        sharex=True,
        layout="constrained",
    )
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    for ax, metric in zip(axs, metrics):
        metric_box(df, metric, order=order, ax=ax, **kwargs)
        ax.set_title(metric)
    # Empty axes in last row if metrics % n_cols != 0
    for ax in axs[len(metrics):]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)
    plt.show()

def plot_confidence(
    ts,
    c,
    y,
    tp,
    fp,
    tn,
    fn,
    *,
    high_conf: Sequence[int] | None = None,
    ave_time: float = 0.0,
    model_name: str = "",
    title: str | None = None,
):
    """Plot raw signal coloured by confidence plus optional high‑conf spans."""
    x = np.arange(len(c))
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    cmap = plt.get_cmap("coolwarm")
    ax.plot(x, c, linestyle=":", label="confidence")

    # colourise raw signal by confidence
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(c[:-1]))
    segments = [[(x[i], ts[i]), (x[i + 1], ts[i + 1])] for i in range(len(ts) - 1)]
    ax.add_collection(mcoll.LineCollection(segments, colors=colors, linewidths=1.5))

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Acceleration (g)")
    ax.set_ylim(-0.1, max(ts) + 0.5)

    # colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.08, label="Confidence")

    # highlight detections
    if high_conf is not None:
        for h in high_conf:
            ax.axvspan(h, h + 4000, color="gray", alpha=0.3)
    if y != -1:
        ax.axvline(x=y, color="red", linestyle="--", label="Fall")
    ax.legend()

    header = title or ""
    stats  = f"TP:{tp} FP:{fp} TN:{tn} FN:{fn}  Time/sample:{ave_time:.2f} ms"
    ax.set_title(f"{header} {stats} {model_name}")
    plt.show()


def plot_detection(
    ts,
    y,
    c,
    cm,
    high_conf,
    ave_time,
    **kwargs,
):
    """Wrapper that centralises the 3 common detection plots.

    Parameters
    ----------
    ts : array‑like
        Raw signal.
    y : int
        Fall index (or ‑1).
    c : array‑like
        Confidence trajectory.
    cm : ndarray shape=(2,2)
        Confusion‑matrix count for this trace.
    high_conf : array‑like | None
        Detected high‑confidence points.
    ave_time : float
        Runtime per sample (ms).
    kwargs
        Recognised keys:
        * plot – plot always
        * plot_errors – plot when FP or FN present
        * plot_ave_conf – overlay average confidence
        Any extra kwargs are forwarded to `utils.plot_confidence`.
    """
    defaults = {"plot_ave_conf": False}
    cfg = {**defaults, **kwargs}

    tn, fp, fn, tp = cm.ravel()
    should_plot = cfg.get("plot", False)
    err_plot    = cfg.get("plot_errors", False) and (fp or fn)
    ave_plot    = cfg.get("plot_ave_conf", False) and (fp or fn)

    if should_plot or err_plot or ave_plot:
        plot_confidence(
            ts,
            c,
            y,
            tp,
            fp,
            tn,
            fn,
            high_conf=high_conf,
            ave_time=ave_time,
            **cfg,
        )

def critical_difference(
    df: pd.DataFrame,
    metric: str = "f1-score",
    pivot_column: str = "model",
    *,
    alpha: float = 0.05,
    title: str | None = None,
    save: str | None = None,
):
    """
    Draw a Demšar critical-difference diagram for one metric.

    Parameters
    ----------
    df      : CV table with columns  fold, model, <metric>
    metric  : which column to rank (higher = better)
    alpha   : significance level for Nemenyi test
    title   : figure title
    save    : path to save (PDF/PNG).  If None, no file is written.
    """
    pivot = (
        df.pivot_table(index="fold", columns=pivot_column, values=metric, aggfunc="median")
    )

    methods = pivot.columns.tolist()
    results = pivot.values

    cd(results, methods, alpha=alpha)
    plt.title(title or f"CD diagram – {metric}  (α={alpha})")
    if save:
        plt.savefig(save, bbox_inches="tight")
    plt.show()

def window_bar(
    df: pd.DataFrame,
    metric: str = "f1-score",
    x: str = "window_size",
    hue: str = "model",
    *,
    order: Sequence[int] | None = None,
    ci: str | float | None = "sd",
    palette: str | Sequence[str] = "tab10",
    ax: plt.Axes | None = None,
    title: str | None = None,
    legend_out: bool = True,
):
    """
    Grouped bar-plot: one group per window, coloured bars per model.

    Parameters
    ----------
    df      : DataFrame with *window_size*, *model*, <metric>, *fold*
    metric  : column to plot
    order   : explicit x-axis order of window sizes
    ci      : 'sd', 'se', numeric (e.g. 95) or None  (seaborn style)
    legend_out : place legend outside plot on the right
    """
    ax = ax or plt.gca()
    sns.barplot(
        data=df,
        x=x,
        y=metric,
        hue=hue,
        order=order,
        ci=ci,
        palette=palette,
        capsize=.12,
        ax=ax,
        errwidth=1,
    )
    ax.set_xlabel("Window size (s)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by window and model")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sns.despine()

    if legend_out:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax