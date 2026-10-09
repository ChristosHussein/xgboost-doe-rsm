"""Pure statistical helpers for reproducible scientific comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class ParetoFront2D:
    points: Tuple[Point2D, ...]
    input_count: int
    unique_count: int
    eligible_count: int
    duplicate_count: int
    outside_reference_count: int
    dominated_count: int


@dataclass(frozen=True)
class Hypervolume2D:
    value: float
    reference: Point2D
    pareto: ParetoFront2D


def _points_2d(points: Iterable[Sequence[float]]) -> np.ndarray:
    try:
        values = np.asarray(list(points), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("points must be an iterable of numeric pairs") from exc
    if values.size == 0:
        return np.empty((0, 2), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    return values


def _reference_2d(reference: Sequence[float]) -> Point2D:
    values = np.asarray(reference, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("reference must be a finite numeric pair")
    return float(values[0]), float(values[1])


def pareto_front_2d_min(
    points: Iterable[Sequence[float]],
    reference: Optional[Sequence[float]] = None,
) -> ParetoFront2D:
    """Return the canonical weakly non-dominated front for two minimization objectives."""
    values = _points_2d(points)
    input_count = len(values)
    unique = np.unique(values, axis=0)
    unique_count = len(unique)

    outside_count = 0
    if reference is not None:
        ref = np.asarray(_reference_2d(reference))
        inside = np.all(unique <= ref, axis=1)
        outside_count = int((~inside).sum())
        unique = unique[inside]

    front = []
    best_second = np.inf
    for first, second in unique:  # np.unique is lexicographically sorted.
        if second < best_second:
            front.append((float(first), float(second)))
            best_second = second

    eligible_count = len(unique)
    return ParetoFront2D(
        points=tuple(front),
        input_count=input_count,
        unique_count=unique_count,
        eligible_count=eligible_count,
        duplicate_count=input_count - unique_count,
        outside_reference_count=outside_count,
        dominated_count=eligible_count - len(front),
    )


def hypervolume_2d_min(
    points: Iterable[Sequence[float]], reference: Sequence[float]
) -> Hypervolume2D:
    """Compute the union area dominated by a 2-D minimization front."""
    ref = _reference_2d(reference)
    pareto = pareto_front_2d_min(points, reference=ref)
    value = 0.0
    previous_second = ref[1]
    for first, second in pareto.points:
        value += (ref[0] - first) * (previous_second - second)
        previous_second = second
    return Hypervolume2D(float(value), ref, pareto)


def _paired_differences(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("paired samples must be one-dimensional and have equal length")
    if len(left) < 2:
        raise ValueError("at least two matched pairs are required")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("paired samples must contain only finite values")
    return left - right


def paired_difference_summary(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
) -> dict:
    """Summarize matched differences using the explicit sign convention ``a - b``."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    differences = _paired_differences(a, b)
    n_pairs = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    median = float(np.median(differences))
    q25, q75 = (float(x) for x in np.percentile(differences, [25, 75]))

    all_zero = bool(np.all(differences == 0.0))
    if sd == 0.0:
        standard_error = 0.0
        ci_low = ci_high = mean
        t_statistic = 0.0 if all_zero else None
        t_pvalue = 1.0 if all_zero else None
        cohens_dz = 0.0 if all_zero else None
    else:
        standard_error = sd / np.sqrt(n_pairs)
        critical = float(stats.t.ppf(0.5 + confidence / 2.0, n_pairs - 1))
        half_width = critical * standard_error
        ci_low, ci_high = mean - half_width, mean + half_width
        t_statistic = mean / standard_error
        t_pvalue = float(2.0 * stats.t.sf(abs(t_statistic), n_pairs - 1))
        cohens_dz = mean / sd

    if all_zero:
        wilcoxon_statistic, wilcoxon_pvalue = 0.0, 1.0
    else:
        wilcoxon = stats.wilcoxon(differences, alternative="two-sided", method="auto")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_pvalue = float(wilcoxon.pvalue)

    return {
        "difference": "a_minus_b",
        "n_pairs": n_pairs,
        "mean_difference": mean,
        "median_difference": median,
        "sd_difference": sd,
        "iqr_difference": q75 - q25,
        "standard_error": float(standard_error),
        "confidence_level": confidence,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "paired_t_statistic": None if t_statistic is None else float(t_statistic),
        "paired_t_pvalue": t_pvalue,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohens_dz": None if cohens_dz is None else float(cohens_dz),
    }


def paired_tost(
    a: Sequence[float],
    b: Sequence[float],
    margin: Optional[float],
    alpha: float = 0.05,
) -> dict:
    """Run paired two-one-sided equivalence tests for a predeclared symmetric margin."""
    if margin is None:
        return {
            "performed": False,
            "equivalent": None,
            "reason": "equivalence_margin_not_predeclared",
        }
    if not np.isfinite(margin) or margin <= 0.0:
        raise ValueError("margin must be a positive finite value")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie strictly between zero and 0.5")

    differences = _paired_differences(a, b)
    n_pairs = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    standard_error = sd / np.sqrt(n_pairs)
    ci_level = 1.0 - 2.0 * alpha

    if standard_error == 0.0:
        lower_pvalue = 0.0 if mean > -margin else (0.5 if mean == -margin else 1.0)
        upper_pvalue = 0.0 if mean < margin else (0.5 if mean == margin else 1.0)
        lower_statistic = upper_statistic = None
        ci_low = ci_high = mean
    else:
        lower_statistic = (mean + margin) / standard_error
        upper_statistic = (mean - margin) / standard_error
        lower_pvalue = float(stats.t.sf(lower_statistic, n_pairs - 1))
        upper_pvalue = float(stats.t.cdf(upper_statistic, n_pairs - 1))
        critical = float(stats.t.ppf(1.0 - alpha, n_pairs - 1))
        ci_low = mean - critical * standard_error
        ci_high = mean + critical * standard_error

    return {
        "performed": True,
        "equivalent": bool(lower_pvalue < alpha and upper_pvalue < alpha),
        "difference": "a_minus_b",
        "n_pairs": n_pairs,
        "mean_difference": mean,
        "margin": float(margin),
        "alpha": alpha,
        "ci_level": ci_level,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "lower_t_statistic": None if lower_statistic is None else float(lower_statistic),
        "lower_pvalue": float(lower_pvalue),
        "upper_t_statistic": None if upper_statistic is None else float(upper_statistic),
        "upper_pvalue": float(upper_pvalue),
        "tost_pvalue": float(max(lower_pvalue, upper_pvalue)),
    }


def regression_interval(
    prediction: float,
    leverage: float,
    residual_mean_square: float,
    residual_df: float,
    *,
    estimand: str,
    future_observations: int = 1,
    confidence: float = 0.95,
) -> dict:
    """Compute a labelled OLS interval for one explicitly chosen estimand."""
    if estimand not in {"surrogate_mean", "future_observation", "future_mean"}:
        raise ValueError(
            "estimand must be surrogate_mean, future_observation, or future_mean"
        )
    if leverage < 0.0 or residual_mean_square < 0.0 or residual_df <= 0.0:
        raise ValueError("leverage, residual mean square, and residual df are invalid")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isinstance(future_observations, int) or future_observations <= 0:
        raise ValueError("future_observations must be a positive integer")

    if estimand == "surrogate_mean":
        future_variance_multiplier = 0.0
    elif estimand == "future_observation":
        if future_observations != 1:
            raise ValueError("future_observation requires future_observations=1")
        future_variance_multiplier = 1.0
    else:
        future_variance_multiplier = 1.0 / future_observations
    variance = residual_mean_square * (leverage + future_variance_multiplier)
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, residual_df))
    half_width = critical * np.sqrt(variance)
    return {
        "estimand": estimand,
        "future_observations": (
            0 if estimand == "surrogate_mean" else future_observations
        ),
        "prediction": float(prediction),
        "leverage": float(leverage),
        "residual_mean_square": float(residual_mean_square),
        "degrees_of_freedom": float(residual_df),
        "confidence_level": float(confidence),
        "variance": float(variance),
        "lower": float(prediction - half_width),
        "upper": float(prediction + half_width),
    }


def satterthwaite_blocked_future_mean_interval(
    prediction: float,
    leverage: float,
    residual_mean_square: float,
    residual_df: float,
    block_mean_square: float,
    block_df: float,
    *,
    historical_runs_per_block: int,
    historical_block_count: int,
    future_observations: int,
    confidence: float = 0.95,
) -> dict:
    """Blocked interval for the mean of future runs under an explicit EMS model.

    This reproduces the historical expected-mean-square parameterization while
    exposing its assumptions and applying the zero block-variance boundary.
    """
    if leverage < 0.0 or residual_mean_square < 0.0 or residual_df <= 0.0:
        raise ValueError("invalid residual or leverage inputs")
    if block_mean_square < 0.0 or block_df <= 0.0:
        raise ValueError("invalid block mean square or degrees of freedom")
    if historical_runs_per_block <= 0 or historical_block_count <= 0:
        raise ValueError("historical block dimensions must be positive")
    if future_observations <= 0:
        raise ValueError("future_observations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")

    estimated_block_variance = max(
        0.0,
        (block_mean_square - residual_mean_square) / historical_runs_per_block,
    )
    boundary_applied = estimated_block_variance == 0.0
    if boundary_applied:
        variance = residual_mean_square * (
            leverage + 1.0 / future_observations
        )
        effective_df = float(residual_df)
        coefficients = {
            "residual_mean_square": leverage + 1.0 / future_observations,
            "block_mean_square": 0.0,
        }
    else:
        block_coefficient = (
            1.0 / future_observations + 1.0 / historical_block_count
        ) / historical_runs_per_block
        residual_coefficient = (
            leverage + 1.0 / future_observations - block_coefficient
        )
        if residual_coefficient < 0.0:
            raise ValueError("variance-component coefficients are not non-negative")
        variance = (
            residual_coefficient * residual_mean_square
            + block_coefficient * block_mean_square
        )
        denominator = (
            (residual_coefficient * residual_mean_square) ** 2 / residual_df
            + (block_coefficient * block_mean_square) ** 2 / block_df
        )
        effective_df = float(variance**2 / denominator)
        coefficients = {
            "residual_mean_square": float(residual_coefficient),
            "block_mean_square": float(block_coefficient),
        }
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, effective_df))
    half_width = critical * np.sqrt(variance)
    return {
        "estimand": "mean_of_future_confirmation_runs",
        "future_observations": int(future_observations),
        "prediction": float(prediction),
        "leverage": float(leverage),
        "confidence_level": float(confidence),
        "variance": float(variance),
        "effective_degrees_of_freedom": effective_df,
        "lower": float(prediction - half_width),
        "upper": float(prediction + half_width),
        "estimated_block_variance": float(estimated_block_variance),
        "zero_block_variance_boundary_applied": boundary_applied,
        "mean_square_coefficients": coefficients,
        "assumptions": [
            "Residual and block mean squares are treated as independent.",
            "The historical residual mean square includes any unmodelled structural discrepancy.",
            "Block variance is assumed transferable to the future confirmation runs.",
        ],
    }


def _design_matrix(name: str, values: Sequence[Sequence[float]], n_rows: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != n_rows or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty 2-D matrix with {n_rows} rows")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _model_summary(y: np.ndarray, matrix: np.ndarray) -> dict:
    rank = int(np.linalg.matrix_rank(matrix))
    coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
    residuals = y - matrix @ coefficients
    sse = float(residuals @ residuals)
    return {
        "column_count": int(matrix.shape[1]),
        "rank": rank,
        "rank_deficiency": int(matrix.shape[1] - rank),
        "residual_df": int(len(y) - rank),
        "sse": sse,
    }


def _component(ss: float, df: int) -> dict:
    return {"ss": float(ss), "df": int(df), "ms": float(ss / df)}


def _f_test(numerator: dict, denominator: dict, denominator_name: str) -> dict:
    if denominator["ms"] <= 0.0:
        return {
            "available": False,
            "f_statistic": None,
            "pvalue": None,
            "denominator": denominator_name,
            "reason": "zero_denominator_mean_square",
        }
    statistic = numerator["ms"] / denominator["ms"]
    return {
        "available": True,
        "f_statistic": float(statistic),
        "pvalue": float(stats.f.sf(statistic, numerator["df"], denominator["df"])),
        "numerator_df": numerator["df"],
        "denominator_df": denominator["df"],
        "denominator": denominator_name,
    }


def blocked_lack_of_fit_decomposition(
    response: Sequence[float],
    reduced_matrix: Sequence[Sequence[float]],
    additive_matrix: Sequence[Sequence[float]],
    cell_means_matrix: Sequence[Sequence[float]],
) -> dict:
    """Partition blocked-model residual SS using ranks of three nested models."""
    y = np.asarray(response, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("response must be a nonempty finite one-dimensional vector")
    reduced = _design_matrix("reduced_matrix", reduced_matrix, len(y))
    additive = _design_matrix("additive_matrix", additive_matrix, len(y))
    cell_means = _design_matrix("cell_means_matrix", cell_means_matrix, len(y))

    models = {
        "reduced": _model_summary(y, reduced),
        "additive": _model_summary(y, additive),
        "cell_means": _model_summary(y, cell_means),
    }
    reduced_rank = models["reduced"]["rank"]
    additive_rank = models["additive"]["rank"]
    cell_rank = models["cell_means"]["rank"]

    if np.linalg.matrix_rank(np.column_stack([additive, reduced])) != additive_rank:
        raise ValueError("reduced_matrix must be nested within additive_matrix")
    if np.linalg.matrix_rank(np.column_stack([cell_means, additive])) != cell_rank:
        raise ValueError("additive_matrix must be nested within cell_means_matrix")
    if not reduced_rank < additive_rank < cell_rank < len(y):
        raise ValueError("nested model ranks must increase and leave positive pure-error df")

    tolerance = 1e-10 * max(1.0, models["reduced"]["sse"])

    def difference(larger_sse: float, smaller_sse: float) -> float:
        value = larger_sse - smaller_sse
        if value < -tolerance:
            raise ValueError("nested model SSE increased unexpectedly")
        return max(0.0, value)

    structural = _component(
        difference(models["reduced"]["sse"], models["additive"]["sse"]),
        additive_rank - reduced_rank,
    )
    interaction = _component(
        difference(models["additive"]["sse"], models["cell_means"]["sse"]),
        cell_rank - additive_rank,
    )
    pure_error = _component(models["cell_means"]["sse"], len(y) - cell_rank)
    pooled_additive = _component(
        models["additive"]["sse"], len(y) - additive_rank
    )

    components = {
        "structural_lack_of_fit": structural,
        "treatment_by_block": interaction,
        "center_pure_error": pure_error,
        "pooled_additive_residual": pooled_additive,
    }
    tests = {
        "structural_vs_pooled_additive": _f_test(
            structural, pooled_additive, "pooled_additive_residual"
        ),
        "structural_vs_treatment_by_block": _f_test(
            structural, interaction, "treatment_by_block"
        ),
        "structural_vs_center_pure_error_sensitivity": _f_test(
            structural, pure_error, "center_pure_error"
        ),
        "interaction_vs_center_pure_error": _f_test(
            interaction, pure_error, "center_pure_error"
        ),
    }

    return {
        "n_observations": len(y),
        "models": models,
        "components": components,
        "tests": tests,
        "identity": {
            "df_components_sum": structural["df"] + interaction["df"] + pure_error["df"],
            "ss_components_sum": structural["ss"] + interaction["ss"] + pure_error["ss"],
        },
    }
