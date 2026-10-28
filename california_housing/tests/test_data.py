"""
tests/test_data.py - Unit tests for Task 0 data integrity and factor coding.
"""

import os
import numpy as np
import pandas as pd
import pytest

from pipeline import encode_factors, decode_factors, generate_design_plan


def test_coding_roundtrip():
    """Verifies that coded <-> natural conversions round-trip correctly within tolerance."""
    # Test corners and center
    test_points = [
        np.array([-1.0, -1.0, -1.0, -1.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-0.5, 0.333333, 0.8, -0.2]),
    ]
    for pt in test_points:
        eta, depth, subsample, reg_lambda = decode_factors(pt, clip_domain=False)
        recoded = encode_factors(eta, depth, subsample, reg_lambda)
        np.testing.assert_allclose(recoded, pt, atol=1e-5, err_msg=f"Failed round-trip for point {pt}")


def test_design_plan_integrity():
    """Verifies properties of the generated 140-run experimental design plan."""
    plan = generate_design_plan()
    assert len(plan) == 140, f"Expected 140 runs, got {len(plan)}"

    df_plan = pd.DataFrame(plan)

    # 1. Check blocks
    assert sorted(df_plan["block"].unique()) == [1, 2, 3, 4, 5]
    for b in range(1, 6):
        assert len(df_plan[df_plan["block"] == b]) == 28, f"Block {b} does not have 28 runs"

    # 2. Check unique points (25 points total)
    assert len(df_plan["point_id"].unique()) == 25, "Expected 25 unique geometric point IDs"
    assert sorted(df_plan["point_id"].unique()) == list(range(1, 26))

    # 3. Check NO duplicate (point_id, seed)
    duplicates = df_plan.duplicated(subset=["point_id", "seed"])
    assert not duplicates.any(), f"Found duplicate (point_id, seed) rows:\n{df_plan[duplicates]}"

    # 4. Check run_order is a full permutation 1..140
    run_orders = sorted(df_plan["run_order"].values)
    assert run_orders == list(range(1, 141)), "run_order is not a valid permutation of 1..140"

    # 5. Check natural parameter ranges
    for _, row in df_plan.iterrows():
        x = np.array([row["x1"], row["x2"], row["x3"], row["x4"]])
        eta, depth, subsample, reg_lambda = decode_factors(x)
        assert 0.0099 <= eta <= 0.3001, f"eta {eta} out of range"
        assert depth in [3, 4, 5, 6, 7, 8, 9], f"depth {depth} out of range"
        assert 0.499 <= subsample <= 1.001, f"subsample {subsample} out of range"
        assert 0.099 <= reg_lambda <= 10.001, f"reg_lambda {reg_lambda} out of range"


def test_runs_csv_if_present():
    """If results/runs.csv exists, verify all columns and non-null values."""
    if not os.path.exists("results/runs.csv"):
        pytest.skip("results/runs.csv not yet generated")

    df = pd.read_csv("results/runs.csv")
    expected_cols = [
        "run_id", "phase", "point_id", "block", "seed", "replicate", "run_order",
        "x1", "x2", "x3", "x4", "eta", "depth", "subsample", "reg_lambda",
        "val_rmse", "test_rmse", "latency_us_median", "latency_us_iqr", "fit_time_s"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in results/runs.csv"

    assert len(df) == 140
    assert not df.isnull().any().any(), "Found null values in results/runs.csv"
    assert not df.duplicated(subset=["point_id", "seed"]).any(), "Duplicate (point_id, seed) found in CSV"
