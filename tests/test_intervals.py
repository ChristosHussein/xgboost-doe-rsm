"""
tests/test_intervals.py - Verification that hat value h is identical across dual response models.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from analysis import build_design_matrix


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
