"""
Visualization utilities for model evaluation and streaming confidence maps.
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union

import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay

# Optional dependency for Critical Difference diagrams
try:
    from aeon.visualisation import plot_critical_difference as cd

    _AEON_AVAILABLE = True
except ImportError:
    _AEON_AVAILABLE = False


__all__ = [
    "set_style",
    "metric_box",
    "metric_grid",
    "plot_confidence",
    "plot_detection",
    "critical_difference",
    "save_confusion_matrix",
    "plot_grouped_stacked",
    "window_bar",
    "plot_cm_comparison",
    "plot_threshold_curve",
]


def set_style(style: str = "ticks", font_scale: float = 1.1):
    """Set the seaborn plotting style."""
    sns.set_theme(style=style, font_scale=font_scale)


# Set default style on import
set_style()

def plot_cm_comparison(
    cm_untuned: np.ndarray,
    cm_tuned: np.ndarray,
    labels: List[str] = ["Fall", "ADL"],
    title_untuned: str = r"Untuned ($\tau$=0.50)",
    title_tuned: str = r"Tuned ($\tau$=opt)",
    save_path: Optional[str] = None,
    cm_order: List[str] = ["tp", "fn", "fp", "tn"],
):
    """
    Plot two confusion matrices side-by-side (Untuned vs Tuned).
    """
    fig, axs = plt.subplots(1, 2, figsize=(5, 2), dpi=400, sharey=True)

    if cm_order == ["tp", "fn", "fp", "tn"]:
        def reorder_cm(cm: np.ndarray) -> np.ndarray:
            return np.array([[cm[1, 1], cm[1, 0]], [cm[0, 1], cm[0, 0]]])
        cm_untuned = reorder_cm(cm_untuned)
        cm_tuned = reorder_cm(cm_tuned)
    
    # Plot Untuned
    ConfusionMatrixDisplay(cm_untuned, display_labels=labels).plot(
        ax=axs[0], colorbar=False, values_format="d"
    )
    axs[0].set_title(title_untuned, fontsize=10)
    
    # Plot Tuned
    ConfusionMatrixDisplay(cm_tuned, display_labels=labels).plot(
        ax=axs[1], colorbar=False, values_format="d"
    )
    axs[1].set_title(title_tuned, fontsize=10)
    
    # Shared Labels
    for ax in axs:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        
    fig.supxlabel("Predicted", y=-0.12, fontsize=11)
    fig.supylabel("Actual", x=0.001, fontsize=11)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_threshold_curve(
    model: Any, 
    save_path: Optional[str] = None
):
    """
    Plot the Cost/Gain vs. Threshold curve for a fitted CostClassifierCV.
    Adapted to match sklearn's TunedThresholdClassifierCV style.
    """
    # Check if we have the curve stored
    if not hasattr(model, "optimization_curve_"):
        print("Model missing 'optimization_curve_'. Ensure it is a fitted CostClassifierCV.")
        return

    taus, scores = model.optimization_curve_
    best_tau = model.threshold_
    
    # Find score at best tau
    best_score_idx = np.abs(taus - best_tau).argmin()
    best_score = scores[best_score_idx]

    plt.figure(figsize=(6, 5), dpi=400)

    # Plot Curve
    # Divide by 100 to match your scaling preference in the snippet, 
    # or keep raw if score is already scaled. Assuming raw gain here.
    # If you want "gain * 10^-2", we scale:
    scale_factor = 100.0
    
    plt.plot(
        taus,
        scores / scale_factor,
        color="tab:orange",
        lw=2
    )
    
    # Plot Optimal Point
    plt.plot(
        best_tau,
        best_score / scale_factor,
        "o",
        markersize=10,
        color="tab:orange",
        label="Optimal cut-off point",
    )
    
    plt.legend()
    plt.xlabel("Decision threshold")
    plt.ylabel(r"gain $\times 10^{-2}$")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def metric_box(
    df: pd.DataFrame,
    metric: str,
    *,
    x: str = "model",
    hue: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    grid_ax: Optional[str] = None,
    palette: str = "tab10",
    type: str = "box",
) -> Axes:
    """
    Draw a box-plot of a specific metric by model.
    """
    ax = ax or plt.gca()
    if type == "box":
        sns.boxplot(
            data=df,
            x=x,
            y=metric,
            order=order,
            hue=x,
            ax=ax,
            width=0.5,
            palette=palette,
            showfliers=False,
        )
    else:  # bar
        sns.barplot(
            data=df,
            x=x,
            y=metric,
            order=order,
            hue=x,
            ax=ax,
            palette=palette,
            errorbar="sd",
        )
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)

    if grid_ax is not None:
        ax.grid(axis=grid_ax, alpha=0.3)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    sns.despine()
    return ax


def metric_grid(
    df: pd.DataFrame,
    metrics: List[str],
    *,
    order: Optional[List[str]] = None,
    n_cols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    **kwargs,
):
    """
    Display multiple metrics in a grid of boxplots.
    """
    n = len(metrics)
    n_cols = max(1, n_cols)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize or (4 * n_cols, 4 * n_rows),
        sharex=True,
        layout="constrained",
    )
    axs_flat = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    for ax, metric in zip(axs_flat, metrics):
        metric_box(df, metric, order=order, ax=ax, **kwargs)
        # ax.set_title(metric) # metric_box already sets title

    # Hide unused axes
    for ax in axs_flat[len(metrics) :]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_confidence(
    ts: np.ndarray,
    c: np.ndarray,
    y: Union[int, Sequence[int]],
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    *,
    high_conf: Optional[Sequence[int]] = None,
    ave_time: float = 0.0,
    model_name: str = "",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    thresh_line: Optional[float] = None,
    freq: int = 100,
    **kwargs,
):
    """
    Single-panel plot with LEFT y-axis = acceleration (g),
    RIGHT y-axis = confidence [0,1].
    """
    x = np.arange(len(ts))
    fig, ax = plt.subplots(figsize=(12, 3), dpi=200)

    # --- Left axis: acceleration ---
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Acceleration (g)")
    ax.set_ylim(min(-0.1, np.min(ts) * 1.05), np.max(ts) + 0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Plot faint signal in background
    ax.plot(x, ts, lw=0.8, color="0.25", alpha=0.2, label="Acceleration", zorder=1)

    # --- Right axis: confidence ---
    ax_r = ax.twinx()
    ax_r.set_ylim(0, 1.05)  # slight buffer for text

    # turn OFF the right ticks/labels/spine on ax_r (cleaner look)
    ax_r.tick_params(right=False, labelright=False)
    ax_r.spines["right"].set_visible(False)

    # Base line for legend entry
    ax_r.plot(x, c, lw=2.0, color="tab:blue", label="Confidence", zorder=2, alpha=0)

    # Colored segments by confidence value
    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(c[:-1]))
    segments = [[(x[i], c[i]), (x[i + 1], c[i + 1])] for i in range(len(c) - 1)]
    ax_r.add_collection(
        mcoll.LineCollection(
            segments, colors=colors, linewidths=1.5, alpha=0.9, zorder=2
        )
    )

    # --- Threshold Lines & Annotations ---
    # Determine text position (end of plot)
    text_x = len(ts) * 0.98

    # Default Threshold (0.5)
    ax_r.axhline(0.5, color="gray", lw=1.0, ls=":", zorder=1)
    ax_r.text(text_x, 0.51, r"$\tau=0.5$", color="gray", fontsize=8, ha="right")

    # Tuned Threshold
    if thresh_line is not None and abs(thresh_line - 0.5) > 0.01:
        ax_r.axhline(thresh_line, color="tab:green", lw=1.5, ls="--", zorder=2)
        ax_r.text(
            text_x,
            thresh_line + 0.02,
            f"$\\tau={thresh_line:.2f}$ (tuned)",
            color="tab:green",
            fontsize=8,
            ha="right",
            fontweight="bold",
        )

    # --- Highlights (Predicted Falls) ---
    if high_conf is not None:
        for h in high_conf:
            # Mark the point on the confidence line
            ax_r.plot(
                h,
                c[h],
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.5,
                zorder=3,
                label="Detection",
            )

    # --- Ground Truth Events ---
    # Normalize y to list
    if isinstance(y, int):
        events = [] if y == -1 else [y]
    else:
        events = [e for e in y if e != -1]

    for i, event_idx in enumerate(events):
        label = "True Event" if i == 0 else None
        ax.axvline(
            x=event_idx,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=label,
            zorder=3,
        )

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_r, fraction=0.03, pad=0.02)
    cbar.set_label("Confidence")

    # Mark thresholds on colorbar
    ticks = [0.5]
    colors_ticks = ["gray"]
    if thresh_line is not None and abs(thresh_line - 0.5) > 0.01:
        ticks.append(thresh_line)
        colors_ticks.append("tab:green")

    for t, col in zip(ticks, colors_ticks):
        cbar.ax.hlines(t, 0, 1, colors=col, lw=1.5)
        cbar.ax.plot(
            -0.15, t, marker=r"$\triangleright$", color=col, markersize=6, clip_on=False
        )

    # --- Legend (Combined) ---
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()

    # Remove duplicates if multiple detections plotted
    by_label = dict(zip(l1 + l2, h1 + h2))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper left",
        ncols=3,
        frameon=False,
        fontsize=9,
    )

    # --- Formatting ---
    # Convert X-axis samples to Seconds
    def seconds_formatter(x_val, pos):
        return f"{int(x_val / freq)}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(seconds_formatter))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))
    ax.set_xlabel("Time (s)")

    # Title
    stats = f"TP:{tp} FP:{fp} TN:{tn} FN:{fn} | Time: {ave_time:.2f} µs/sample"
    ax.set_title(f"{title or ''}  {stats}  {model_name}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_detection(
    ts: np.ndarray,
    y: int,
    c: np.ndarray,
    cm: np.ndarray,
    high_conf: Optional[np.ndarray],
    ave_time: float,
    **kwargs,
):
    """
    Wrapper for plot_confidence that handles conditional plotting logic.

    Kwargs:
    - plot (bool): Plot always.
    - plot_errors (bool): Plot only if FP > 0 or FN > 0.
    """
    tn, fp, fn, tp = cm.ravel()

    should_plot = kwargs.get("plot", False)
    plot_errors = kwargs.get("plot_errors", False)

    has_error = (fp > 0) or (fn > 0)

    if should_plot or (plot_errors and has_error):
        # Forward valid kwargs to plot_confidence
        valid_keys = plot_confidence.__code__.co_varnames
        plot_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

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
            **plot_kwargs,
        )


def critical_difference(
    df: pd.DataFrame,
    metric: str = "f1-score",
    pivot_column: str = "model",
    *,
    alpha: float = 0.05,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """
    Draw a Demšar critical-difference diagram.
    Requires 'aeon' package.
    """
    if not _AEON_AVAILABLE:
        warnings.warn(
            "The 'aeon' package is required for critical_difference plots. "
            "Pip install aeon to use this feature."
        )
        return

    # Pivot: Index=Fold/Dataset, Columns=Models, Values=Metric
    # We assume 'fold' or 'seed' exists in df to act as the index
    index_col = "seed" if "seed" in df.columns else "fold"
    if index_col not in df.columns:
        # Fallback: create a dummy index if multiple rows per model exist
        df = df.copy()
        df[index_col] = df.groupby(pivot_column).cumcount()

    pivot = df.pivot_table(
        index=index_col, columns=pivot_column, values=metric, aggfunc="mean"
    )

    methods = pivot.columns.tolist()
    results = pivot.values  # shape (n_datasets, n_methods)

    # AEON expects (n_estimators, n_datasets)
    # So we might need to transpose depending on version,
    # but typically cd() takes (n_datasets, n_estimators)?
    # Actually aeon.visualisation.plot_critical_difference docs say:
    # scores : np.array of shape (n_classifiers, n_datasets)
    # So we transpose.

    plt.figure(figsize=(10, 6))
    cd(results.T, methods, alpha=alpha)

    plt.title(title or f"CD diagram – {metric} (α={alpha})")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def save_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = ["Normal", "Event"],
    save_path: Optional[str] = None,
):
    """Plot and save a simple heatmap for a Confusion Matrix."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_grouped_stacked(
    real_df: pd.DataFrame,
    dummy_df: pd.DataFrame,
    metrics: List[str],
    figsize=(8, 5),
    save_path: Optional[str] = None,
):
    """
    Plot grouped bar chart showing:
    1. Improvement over Dummy Baseline (Base Improvement).
    2. Improvement from Tuning (Threshold Tuning Effect).
    """
    records = []

    # Ensure thresh column exists (if not, assume all are 0.5)
    if "thresh" not in real_df.columns:
        real_df["thresh"] = 0.5
    if "thresh" not in dummy_df.columns:
        dummy_df["thresh"] = 0.5

    for model, g in real_df.groupby("model"):
        # Identify "Base" (untuned) vs "Tuned"
        # We assume untuned is thresh == 0.5 or the first entry if only one exists
        base = g[g["thresh"] == 0.5]
        tuned = g[g["thresh"] != 0.5]

        # Get Dummy Baseline (usually averaged)
        dummy_row = dummy_df.mean(numeric_only=True)

        if base.empty and not tuned.empty:
            # Only tuned version exists
            v0 = tuned[metrics].mean().iloc[0]  # Fallback
            v1 = v0
        elif not base.empty:
            v0 = (
                base[metrics].mean().iloc[0]
            )  # Use first metric as proxy? No, need loop.
        else:
            continue

        for m in metrics:
            val_base = base[m].mean() if not base.empty else tuned[m].mean()
            val_tuned = tuned[m].mean() if not tuned.empty else val_base
            val_dummy = dummy_row[m]

            # Prevent div by zero
            if val_dummy == 0:
                val_dummy = 1e-6

            base_improv = 100 * (val_base - val_dummy) / val_dummy
            tuning_effect = 100 * (val_tuned - val_base) / val_base

            records.append(
                {
                    "model": model,
                    "metric": m,
                    "base_improvement": base_improv,
                    "tuning_effect": tuning_effect,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        print("No data to plot in grouped_stacked.")
        return

    # Plotting Logic
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    model_order = df.groupby("model")["base_improvement"].mean().sort_values().index
    metrics_palette = sns.color_palette("tab10", n_colors=len(metrics))

    bar_width = 0.8 / len(metrics)
    x = np.arange(len(model_order))

    # Background for negative improvement
    ax.axhspan(ax.get_ylim()[0], 0, facecolor="mistyrose", alpha=0.3, zorder=0)

    for i, metric in enumerate(metrics):
        sub = df[df["metric"] == metric].set_index("model").reindex(model_order)
        offset = (i - len(metrics) / 2 + 0.5) * bar_width

        # Base bars
        ax.bar(
            x + offset,
            sub["base_improvement"],
            width=bar_width,
            color=metrics_palette[i],
            label=metric,
            zorder=2,
        )

        # Stacked tuning
        # Only stack if tuning improved things, otherwise it overlaps weirdly
        # (Simplified visualization for now)
        ax.bar(
            x + offset,
            sub["tuning_effect"],
            width=bar_width,
            bottom=sub["base_improvement"],
            color=metrics_palette[i],
            alpha=0.5,
            hatch="//",
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.set_ylabel("% Improvement vs Dummy")
    ax.axhline(0, color="gray", lw=1)
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def window_bar(
    df: pd.DataFrame,
    metric: str = "f1-score",
    x: str = "window_size",
    hue: str = "model",
    *,
    order: Sequence[int] | None = None,
    errorbar = "sd",
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
        errorbar=errorbar,
        palette=palette,
        capsize=0.12,
        ax=ax,
        err_kws={'linewidth': 1},
    )
    ax.set_xlabel("Window size (s)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by window and model")
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sns.despine()

    if legend_out:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax
