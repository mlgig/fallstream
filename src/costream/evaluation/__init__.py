from .tester import run_experiment, ModelSpec, train_models, evaluate_models
from .event_detection import evaluate_recording, FalsePositive
from .presegmented import evaluate_presegmented
from .cross_validation import run_subject_cv, aggregate_cv_results
from .probability_cache import CacheKey, CachedEntry, ProbabilityCache
from .checks import assert_adl_exclusion_zone, assert_gate_applied_identically
from .visualization import (
    plot_confidence, 
    plot_detection, 
    metric_box, 
    metric_grid,
    plot_grouped_stacked,
    critical_difference,
    window_bar,
)

__all__ = [
    "run_experiment",
    "ModelSpec",
    "train_models",
    "evaluate_models",
    "evaluate_recording",
    "FalsePositive",
    "evaluate_presegmented",
    "assert_adl_exclusion_zone",
    "assert_gate_applied_identically",
    "run_subject_cv",
    "aggregate_cv_results",
    "CacheKey",
    "CachedEntry",
    "ProbabilityCache",
    "plot_confidence",
    "plot_detection",
    "metric_box",
    "metric_grid",
    "plot_grouped_stacked",
    "critical_difference",
    "window_bar",
]