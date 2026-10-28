"""
tests/test_lof.py - Verification tests for Lack of Fit nested model degrees of freedom.
"""

import json
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from analysis import run_task3_lack_of_fit


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


def test_analysis_reports_rank_aware_blocked_lack_of_fit_components():
    df_runs = pd.read_csv("results/runs.csv")

    result = run_task3_lack_of_fit(df_runs)

    full_ranks = result["rank_diagnostics"]["full"]
    assert full_ranks["reduced"]["rank"] == 19
    assert full_ranks["additive"]["rank"] == 29
    assert full_ranks["cell_means"]["rank"] == 125

    y1 = result["Y1"]
    assert y1["structural_lack_of_fit"]["df"] == 10
    assert y1["treatment_by_block"]["df"] == 96
    assert y1["center_pure_error"]["df"] == 15
    assert y1["pooled_additive_residual"]["df"] == 111
    assert y1["pooled_additive_residual"]["ss"] == pytest.approx(
        y1["treatment_by_block"]["ss"] + y1["center_pure_error"]["ss"]
    )
    assert y1["historical_pooled_additive_test"]["denominator"] == "pooled_additive_residual"
    assert y1["center_pure_error_sensitivity"]["denominator"] == "center_pure_error"
    assert y1["legacy_compatibility"]["df_PE_alias"] == "pooled_additive_residual.df"

    y2 = result["Y2"]
    assert y2["structural_lack_of_fit"]["df"] == 10
    assert y2["treatment_by_block"]["df"] == 96
    assert y2["center_pure_error"]["df"] == 15


def test_restricted_lack_of_fit_exposes_aliasing_and_exact_components():
    df_runs = pd.read_csv("results/runs.csv")

    result = run_task3_lack_of_fit(df_runs)

    restricted_ranks = result["rank_diagnostics"]["restricted"]
    assert result["Y1_restricted"]["n_observations"] == 95
    assert restricted_ranks["reduced"]["column_count"] == 19
    assert restricted_ranks["reduced"]["rank"] == 18
    assert restricted_ranks["reduced"]["rank_deficiency"] == 1
    assert restricted_ranks["additive"]["rank"] == 20
    assert restricted_ranks["cell_means"]["rank"] == 80

    restricted = result["Y1_restricted"]
    assert restricted["structural_lack_of_fit"]["df"] == 2
    assert restricted["treatment_by_block"]["df"] == 60
    assert restricted["center_pure_error"]["df"] == 15
    assert restricted["pooled_additive_residual"]["df"] == 75
