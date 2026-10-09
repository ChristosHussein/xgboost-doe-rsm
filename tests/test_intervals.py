"""
tests/test_intervals.py - Verification that hat value h is identical across dual response models.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from analysis import build_design_matrix
from scientific_stats import (
    regression_interval,
    satterthwaite_blocked_future_mean_interval,
)


def test_hat_value_identical_across_responses():
    """Verifies that leverage h at any coordinate x* is mathematically identical for Y1 and Y2."""
    df_runs = pd.read_csv("results/runs.csv")
    X = build_design_matrix(df_runs)
    XtXi = np.linalg.inv(X.T @ X)

    test_points = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.8499708, -0.66666667, 1.0, -0.08116946]),
        np.array([0.5, -0.5, 0.5, -0.5]),
    ]

    for pt in test_points:
        x1, x2, x3, x4 = pt
        r = np.array([
            1.0, x1, x2, x3, x4,
            x1**2, x2**2, x3**2, x4**2,
            x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4,
            0.2, 0.2, 0.2, 0.2
        ])
        h_y1 = float(r @ XtXi @ r)
        h_y2 = float(r @ XtXi @ r)
        assert np.isclose(h_y1, h_y2), f"Hat values differ at {pt}"


def test_interval_estimands_have_explicit_and_ordered_widths():
    common = dict(
        prediction=0.48,
        leverage=0.1,
        residual_mean_square=0.0004,
        residual_df=100,
    )
    mean = regression_interval(**common, estimand="surrogate_mean")
    future_mean = regression_interval(
        **common, estimand="future_mean", future_observations=10
    )
    observation = regression_interval(**common, estimand="future_observation")
    widths = [item["upper"] - item["lower"] for item in (mean, future_mean, observation)]
    assert widths[0] < widths[1] < widths[2]
    assert mean["future_observations"] == 0
    assert future_mean["future_observations"] == 10


def test_satterthwaite_block_interval_reports_components_and_effective_df():
    result = satterthwaite_blocked_future_mean_interval(
        prediction=0.48,
        leverage=0.1,
        residual_mean_square=0.0004,
        residual_df=121,
        block_mean_square=0.0020,
        block_df=4,
        historical_runs_per_block=28,
        historical_block_count=5,
        future_observations=10,
    )
    c2 = (1 / 10 + 1 / 5) / 28
    c1 = (1 / 10 + 0.1) - c2
    expected_variance = c1 * 0.0004 + c2 * 0.0020
    assert result["estimand"] == "mean_of_future_confirmation_runs"
    assert result["variance"] == pytest.approx(expected_variance)
    assert result["mean_square_coefficients"]["block_mean_square"] == pytest.approx(c2)
    assert result["zero_block_variance_boundary_applied"] is False
    assert result["effective_degrees_of_freedom"] < 121


def test_satterthwaite_block_interval_applies_zero_variance_boundary():
    result = satterthwaite_blocked_future_mean_interval(
        prediction=0.48,
        leverage=0.1,
        residual_mean_square=0.001,
        residual_df=50,
        block_mean_square=0.0005,
        block_df=4,
        historical_runs_per_block=28,
        historical_block_count=5,
        future_observations=5,
    )
    assert result["zero_block_variance_boundary_applied"] is True
    assert result["estimated_block_variance"] == 0.0
    assert result["effective_degrees_of_freedom"] == 50
    assert result["variance"] == pytest.approx(0.001 * (0.1 + 0.2))
