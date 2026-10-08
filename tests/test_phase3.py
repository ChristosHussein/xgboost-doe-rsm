"""
tests/test_phase3.py - Verification tests for Phase 3 canonical analysis and ridge analysis.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from analysis import build_design_matrix, canonical_decomposition, predict_block_averaged


def test_canonical_algebra_and_properties():
    """Tests trace == sum(eig), b + 2B x0 == 0, y0_hat consistency, and natural subsample bounds."""
    df_runs = pd.read_csv("results/runs.csv")
    X = build_design_matrix(df_runs)
    y = df_runs["val_rmse"]
    fit = sm.OLS(y, X).fit()

    b0, b, B = canonical_decomposition(fit)
    lam, M = np.linalg.eigh(B)

    # 1. Trace == sum of eigenvalues
    assert np.isclose(np.trace(B), lam.sum()), f"trace(B) {np.trace(B)} != sum(eigs) {lam.sum()}"

    # 2. Stationary point solves gradient = 0
    x0 = -0.5 * np.linalg.solve(B, b)
    grad = b + 2 * B @ x0
    assert np.allclose(grad, 0.0, atol=1e-10), f"b + 2B x0 != 0: {grad}"

    # 3. y0_hat consistency with block average evaluation
    y0_hat = b0 + 0.5 * float(b @ x0)
    pred_y0 = predict_block_averaged(fit, x0)
    assert np.isclose(y0_hat, pred_y0), f"y0_hat {y0_hat} != pred_y0 {pred_y0}"

    # 4. Phase 3 JSON output verification
    with open("results/phase3.json", "r", encoding="utf-8") as f:
        p3 = json.load(f)

    subsample_natural = p3["stationary_natural_clamped"]["subsample"]
    assert 0.50 <= subsample_natural <= 1.00, f"Clamped natural subsample {subsample_natural} out of [0.5, 1.0]"

    # 5. Type III SS cross-check for single-df terms
    XtXi = np.linalg.inv(X.T @ X)
    manual_ss3 = fit.params**2 / np.diag(XtXi)
    for k in ["x1", "x2", "x3", "x4", "x1_sq", "x2_sq"]:
        # Parameter t-stat squared * MSE should match manual SS3 / MSE
        t_stat_sq = (fit.tvalues[k])**2
        ratio = manual_ss3[k] / fit.mse_resid
        assert np.isclose(t_stat_sq, ratio, rtol=1e-5), f"SS3 mismatch on term {k}"
