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
from sklearn.metrics import make_scorer, precision_score, recall_score, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


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
    # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    sns.despine()

    if legend_out:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax

def plot_grouped_stacked(
    real_df, dummy_df, metrics,
    figsize=(7, 5), dpi=300, save_path=None
):
    """
    Plot grouped bar chart for improvement vs DummyADL and stacked tuning effect.
    Negative region is shaded light red.
    """

    records = []
    for model, g in real_df.groupby("model"):
        base = g.query("thresh==0.5")
        tuned = g.query("thresh!=0.5")
        dummy_base = dummy_df.query("thresh==0.5")

        for m in metrics:
            v0 = base[m].values[0]
            vd = dummy_base[m].values[0]
            v1 = tuned[m].values[0] if not tuned.empty else v0

            base_improv = 100 * (v0 - vd) / vd  # improvement vs DummyADL
            tuning_effect = 100 * (v1 - v0) / v0  # change from tuning

            records.append({
                "model": model,
                "metric": m,
                "base_improvement": base_improv,
                "tuning_effect": tuning_effect
            })

    df = pd.DataFrame(records)

    # Sort models by average base improvement
    model_order = df.groupby("model")["base_improvement"].mean().sort_values().index
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    palette = dict(zip(metrics, sns.color_palette("tab10", n_colors=len(metrics))))
    bar_width = 0.35
    x = np.arange(len(model_order))

    # Shade negative region
    ax.axhspan(ax.get_ylim()[0], 0, facecolor="mistyrose", alpha=0.4, zorder=0)

    for i, metric in enumerate(metrics):
        subdf = df[df["metric"] == metric].sort_values("model")
        base_vals = subdf["base_improvement"].values
        tuning_vals = subdf["tuning_effect"].values

        # Base bars
        ax.bar(
            x + (i - 0.5) * bar_width, base_vals, width=bar_width,
            color=palette[metric], label=metric if i == 0 else "", zorder=2
        )

        # Stacked tuning effect
        bottoms = base_vals * (tuning_vals >= 0)
        ax.bar(
            x + (i - 0.5) * bar_width, tuning_vals, width=bar_width,
            bottom=bottoms, color=palette[metric], alpha=0.5, zorder=3
        )

        # Annotate tuning effect
        for xi, b, t in zip(x, base_vals, tuning_vals):
            ax.text(
                xi + (i - 0.5) * bar_width,
                b + t + (2 if t >= 0 else -2),
                f"{t:+.0f}%",
                ha="center", va="bottom" if t >= 0 else "top",
                fontsize=8
            )

    ax.axhline(0, color="gray", linewidth=1, zorder=4)
    ax.set_ylabel("% change vs DummyADL")
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=45, ha="right")
    ax.legend(title="Metric")
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_model_helpfulness(real_df, dummy_base, metrics, figsize=(6, 4), dpi=300, save_path=None):
    """
    Simple barplot showing % improvement vs DummyADL for given metrics.
    """

    records = []
    for model, g in real_df.groupby("model"):
        base = g.query("thresh==0.5")    # untuned

        for m in metrics:
            v0 = base[m].values[0]
            vd = dummy_base[m].values[0]
            pct_improv = 100 * (vd - v0) / vd
            records.append({"model": model, "metric": m, "pct_improvement": pct_improv})

    df = pd.DataFrame(records)

    # Sort models by average improvement
    model_order = df.groupby("model")["pct_improvement"].mean().sort_values().index
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    palette = dict(zip(metrics, sns.color_palette("tab10", n_colors=len(metrics))))
    sns.barplot(data=df, x="model", y="pct_improvement", hue="metric", palette=palette, ax=ax)

    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("% change vs DummyADL")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="Metric")
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def fpr_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn, fp, _, _ = cm.ravel()
    tnr = tn / (tn + fp)
    return 1 - tnr

# function to plot precision-recall curve and ROC curve for tuned and untuned models
def plot_precision_recall_roc(untuned_model, tuned_model, X_test, y_test, cv_results):
    scoring = {
        "precision": make_scorer(precision_score),
        "recall": make_scorer(recall_score),
        "fpr": make_scorer(fpr_score),
        "tpr": make_scorer(recall_score),
    }
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), dpi=200, layout="constrained")
    linestyles = ("dashed", "dotted")
    markerstyles = ("o", ">")
    colors = ("tab:blue", "tab:orange")
    names = ("Vanilla GBDT", "Tuned GBDT")
    for idx, (est, linestyle, marker, color, name) in enumerate(
        zip((untuned_model, tuned_model), linestyles, markerstyles, colors, names)
    ):
        decision_threshold = getattr(est, "best_threshold_", 0.5)
        PrecisionRecallDisplay.from_estimator(
            est,
            X_test,
            y_test,
            linestyle=linestyle,
            color=color,
            ax=axs[0],
            name=name,
        )
        axs[0].plot(
            scoring["recall"](est, X_test, y_test),
            scoring["precision"](est, X_test, y_test),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )
        RocCurveDisplay.from_estimator(
            est,
            X_test,
            y_test,
            curve_kwargs=dict(linestyle=linestyle, color=color),
            ax=axs[1],
            name=name,
            plot_chance_level=idx == 1,
        )
        axs[1].plot(
            scoring["fpr"](est, X_test, y_test),
            scoring["tpr"](est, X_test, y_test),
            marker,
            markersize=10,
            color=color,
            label=f"Cut-off point at probability of {decision_threshold:.2f}",
        )

    axs[0].set_title("Precision-Recall curve")
    axs[0].legend()
    axs[1].set_title("ROC curve")
    axs[1].legend()

    axs[2].plot(
        tuned_model.cv_results_["thresholds"],
        tuned_model.cv_results_["scores"],
        color="tab:orange",
    )
    axs[2].plot(
        tuned_model.best_threshold_,
        tuned_model.best_score_,
        "o",
        markersize=10,
        color="tab:orange",
        label="Optimal cut-off point for the gain metric",
    )
    axs[2].legend()
    axs[2].set_xlabel("Decision threshold (probability)")
    axs[2].set_ylabel("Objective score (using cost-matrix)")
    axs[2].set_title("Objective score as a function of the decision threshold")