"""
tests/test_lof.py - Verification tests for Lack of Fit nested model degrees of freedom.
"""

import json
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf


def test_lack_of_fit_degrees_of_freedom():
    """Asserts that the nested-model lack-of-fit test has exactly df_diff=10, df_resid=111."""
    df_runs = pd.read_csv("results/runs.csv")
    Q = ["x1", "x2", "x3", "x4"]
    for q in Q:
        df_runs[q + "_sq"] = df_runs[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_runs[f"{Q[i]}_{Q[j]}"] = df_runs[Q[i]] * df_runs[Q[j]]

    # Val RMSE
    red_y1 = smf.ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                     "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_runs).fit()
    sat_y1 = smf.ols("val_rmse ~ C(block) + C(point_id)", df_runs).fit()
    tab_y1 = sm.stats.anova_lm(red_y1, sat_y1)

    df_diff_y1 = int(tab_y1["df_diff"].iloc[1])
    df_resid_y1 = int(tab_y1["df_resid"].iloc[1])

    assert df_diff_y1 == 10, f"Expected df_diff=10 for LoF, got {df_diff_y1}"
    assert df_resid_y1 == 111, f"Expected df_resid=111 for Pure Error, got {df_resid_y1}"

    # Latency
    red_y2 = smf.ols("latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                     "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_runs).fit()
    sat_y2 = smf.ols("latency_us_median ~ C(block) + C(point_id)", df_runs).fit()
    tab_y2 = sm.stats.anova_lm(red_y2, sat_y2)

    df_diff_y2 = int(tab_y2["df_diff"].iloc[1])
    df_resid_y2 = int(tab_y2["df_resid"].iloc[1])

    assert df_diff_y2 == 10, f"Expected df_diff=10 for Y2 LoF, got {df_diff_y2}"
    assert df_resid_y2 == 111, f"Expected df_resid=111 for Y2 Pure Error, got {df_resid_y2}"


def test_lof_json_contents():
    """Verifies that results/lof.json records matching df values."""
    with open("results/lof.json", "r", encoding="utf-8") as f:
        lof = json.load(f)

    assert lof["Y1"]["df_LoF"] == 10
    assert lof["Y1"]["df_PE"] == 111
    assert lof["Y2"]["df_LoF"] == 10
    assert lof["Y2"]["df_PE"] == 111
