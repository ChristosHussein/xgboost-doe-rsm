"""xgb_doe_xai - Explainable Sequential Design of Experiments for XGBoost Hyperparameter Optimization.

A unified framework synthesizing 2nd-order Response Surface Methodology (RSM/ANOVA)
and Shapley Additive Explanations (SHAP) under stochastic nuisance blocking,
calibrated latency profiling, and strictly isolated holdout verification.
"""

from __future__ import annotations

from xgb_doe_xai.benchmarks import (
    EvaluationResult,
    compute_derringer_suich_desirability,
    compute_hypervolume_2d,
    extract_pareto_front,
    run_matched_motpe_benchmark,
    train_and_evaluate_config,
)
from xgb_doe_xai.config import (
    DEFAULT_FACTORS,
    FACTOR_LABELS,
    FACTOR_NAMES,
    BenchmarkConfig,
    FactorSpec,
    code_configuration,
    uncode_configuration,
)
from xgb_doe_xai.data import (
    DatasetPartitions,
    create_strictly_isolated_partitions,
    generate_synthetic_sentiment_corpus,
)
from xgb_doe_xai.design import (
    DesignPoint,
    generate_fccd_points,
    generate_multiblock_fccd,
    select_adaptive_infill_points,
)
from xgb_doe_xai.profiler import LatencyProfileResult, WarmedLatencyProfiler
from xgb_doe_xai.reporting import (
    render_latex_anova_table,
    render_markdown_anova_table,
    render_markdown_comparison_summary,
)
from xgb_doe_xai.surrogates import (
    AnovaTableResult,
    GaussianProcessSurrogate,
    PolynomialRSMSurrogate,
)
from xgb_doe_xai.xai import (
    AnovaAttributions,
    AnovaShapComparisonResult,
    ShapAttributions,
    compare_anova_vs_shap,
    compute_surrogate_shap_attributions,
    extract_anova_attributions,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_FACTORS",
    "FACTOR_LABELS",
    "FACTOR_NAMES",
    "BenchmarkConfig",
    "FactorSpec",
    "code_configuration",
    "uncode_configuration",
    "WarmedLatencyProfiler",
    "LatencyProfileResult",
    "DatasetPartitions",
    "create_strictly_isolated_partitions",
    "generate_synthetic_sentiment_corpus",
    "DesignPoint",
    "generate_fccd_points",
    "generate_multiblock_fccd",
    "select_adaptive_infill_points",
    "PolynomialRSMSurrogate",
    "GaussianProcessSurrogate",
    "AnovaTableResult",
    "extract_anova_attributions",
    "compute_surrogate_shap_attributions",
    "compare_anova_vs_shap",
    "AnovaAttributions",
    "ShapAttributions",
    "AnovaShapComparisonResult",
    "train_and_evaluate_config",
    "compute_derringer_suich_desirability",
    "extract_pareto_front",
    "compute_hypervolume_2d",
    "run_matched_motpe_benchmark",
    "EvaluationResult",
    "render_latex_anova_table",
    "render_markdown_anova_table",
    "render_markdown_comparison_summary",
]
