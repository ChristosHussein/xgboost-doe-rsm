"""tests/test_xgb_doe_xai.py - Verification suite for the xgb_doe_xai package.

Tests:
1. Calibrated hyperparameter domain and coordinate coding/uncoding.
2. Unified warmed latency profiling protocol (100 warmup, 500 timed).
3. Strictly isolated cryptographic holdout partitioning and nuisance blocks.
4. FCCD design matrix and sequential adaptive infill generation.
5. Polynomial RSM surrogate, ANOVA decomposition, and prediction intervals.
6. ANOVA vs SHAP attributions and interaction cross-validation.
7. Pareto frontier extraction and 2D hypervolume calculation.
"""

from __future__ import annotations

import numpy as np
import pytest
import xgboost as xgb
from scipy import sparse

from xgb_doe_xai import (
    DEFAULT_FACTORS,
    BenchmarkConfig,
    GaussianProcessSurrogate,
    PolynomialRSMSurrogate,
    WarmedLatencyProfiler,
    code_configuration,
    compare_anova_vs_shap,
    compute_derringer_suich_desirability,
    compute_hypervolume_2d,
    compute_surrogate_shap_attributions,
    create_strictly_isolated_partitions,
    extract_anova_attributions,
    extract_pareto_front,
    generate_fccd_points,
    generate_multiblock_fccd,
    generate_synthetic_sentiment_corpus,
    render_markdown_anova_table,
    render_markdown_comparison_summary,
    select_adaptive_infill_points,
    train_and_evaluate_config,
    uncode_configuration,
)


def test_calibrated_hyperparameter_domain_and_coding_roundtrip():
    """Verifies that learning rate domain avoids underfitting and coding roundtrip is exact."""
    # Check calibrated eta range: [0.05, 0.30]
    eta_spec = DEFAULT_FACTORS["x1"]
    assert eta_spec.low == 0.05
    assert eta_spec.high == 0.30
    assert eta_spec.center == 0.175

    # Center coding
    center_nat = {"learning_rate": 0.175, "max_depth": 6, "subsample": 0.75, "colsample_bytree": 0.65}
    coded = code_configuration(center_nat)
    assert coded["x1"] == pytest.approx(0.0)
    assert coded["x2"] == pytest.approx(0.0)
    assert coded["x3"] == pytest.approx(0.0)
    assert coded["x4"] == pytest.approx(0.0)

    # Corners coding and uncoding roundtrip
    for c_val in [-1.0, 1.0]:
        test_coded = {"x1": c_val, "x2": c_val, "x3": c_val, "x4": c_val}
        nat = uncode_configuration(test_coded)
        recoded = code_configuration(nat)
        for k in ["x1", "x2", "x3", "x4"]:
            assert recoded[k] == pytest.approx(c_val)


def test_warmed_latency_profiler_protocol():
    """Verifies that profiler performs warm-up and timed inferences and returns stable stats."""
    # Create simple dummy booster for testing
    X_dummy = np.random.randn(20, 5)
    y_dummy = (X_dummy[:, 0] > 0).astype(int)
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=3, random_state=42)
    clf.fit(X_dummy, y_dummy)

    # Fast profiler settings for test
    profiler = WarmedLatencyProfiler(n_warmup=10, n_timed=20, measure_inplace=True)
    sample = X_dummy[:1]
    res = profiler.profile_booster(clf.get_booster(), sample)

    assert res.n_warmup == 10
    assert res.n_timed == 20
    assert res.predict_median_us > 0.0
    assert res.predict_mean_us > 0.0
    assert res.predict_iqm_us > 0.0
    assert res.predict_sd_us >= 0.0
    assert res.inplace_median_us is not None and res.inplace_median_us > 0.0


def test_strictly_isolated_partitions_and_cryptographic_hash():
    """Verifies that holdout partition is strictly isolated, hashed, and features are cleanly extracted."""
    texts, labels = generate_synthetic_sentiment_corpus(n_samples=200, random_seed=42)
    partitions = create_strictly_isolated_partitions(
        texts=texts,
        labels=labels,
        holdout_ratio=0.20,
        n_blocks=3,
        random_seed=42,
        max_features=200,
    )

    assert len(partitions.holdout_texts) == 40
    assert len(partitions.holdout_sha256) == 64  # SHA-256 hex length
    assert len(partitions.blocks) == 3
    assert partitions.holdout_features is not None
    assert partitions.holdout_features.shape == (40, 200)

    # Verify each block has train and val splits
    for b_tr_f, b_tr_l, b_val_f, b_val_l in partitions.block_features:
        assert b_tr_f.shape[1] == 200
        assert b_val_f.shape[1] == 200
        assert len(b_tr_l) + len(b_val_l) == 160


def test_fccd_design_matrix_geometry_and_adaptive_infill():
    """Verifies Face-Centered CCD point counts (16 corner, 8 axial, 2 center) and adaptive sampling."""
    pts = generate_fccd_points(n_center=2, block_id=0)
    # Total points = 2^4 + 2*4 + 2 = 16 + 8 + 2 = 26 points
    assert len(pts) == 26

    types = [p.point_type for p in pts]
    assert types.count("factorial") == 16
    assert types.count("axial") == 8
    assert types.count("center") == 2

    # Multiblock generation
    mb_pts = generate_multiblock_fccd(n_blocks=3, n_center_per_block=2)
    assert len(mb_pts) == 26 * 3

    # Adaptive infill selection
    def dummy_uncertainty(candidates: np.ndarray) -> np.ndarray:
        return np.linalg.norm(candidates, axis=1)

    infill = select_adaptive_infill_points(
        existing_points=pts,
        surrogate_uncertainty_fn=dummy_uncertainty,
        n_infill=3,
        random_seed=42,
    )
    assert len(infill) == 3
    for p in infill:
        assert p.point_type == "adaptive"
        for v in p.coded.values():
            assert -1.0 <= v <= 1.0


def test_polynomial_rsm_surrogate_and_anova_decomposition():
    """Verifies 2nd-order response surface fitting, ANOVA partition, and prediction intervals."""
    # Synthetic response with known linear, quadratic, and interaction effects
    rng = np.random.RandomState(42)
    pts = generate_multiblock_fccd(n_blocks=2, n_center_per_block=4)
    X_coded = np.array([p.vector for p in pts])
    block_ids = np.array([p.block_id for p in pts])

    # True surface: 0.50 + 0.10*x1 - 0.15*x2 + 0.05*x1^2 + 0.08*x2^2 + 0.12*x1*x2 + 0.02*block + noise
    x1, x2, x3, x4 = X_coded[:, 0], X_coded[:, 1], X_coded[:, 2], X_coded[:, 3]
    y = (
        0.50
        + 0.10 * x1
        - 0.15 * x2
        + 0.05 * (x1 ** 2)
        + 0.08 * (x2 ** 2)
        + 0.12 * x1 * x2
        + 0.03 * block_ids
        + rng.normal(0.0, 0.005, size=len(pts))
    )

    rsm = PolynomialRSMSurrogate(include_blocks=True)
    rsm.fit(X_coded, y, block_ids=block_ids)

    assert rsm.anova_ is not None
    assert rsm.anova_.r2 > 0.95
    assert rsm.anova_.adj_r2 > 0.90
    assert "  Main Effects" in rsm.anova_.source
    assert "  2-Factor Interactions" in rsm.anova_.source
    assert "  Curvature (Quadratic)" in rsm.anova_.source
    assert "  Lack of Fit" in rsm.anova_.source
    assert "  Pure Error" in rsm.anova_.source

    # Check Satterthwaite prediction interval coverage
    y_hat, pi_low, pi_high = rsm.predict_interval(X_coded[:5])
    assert np.all(pi_low <= y_hat)
    assert np.all(y_hat <= pi_high)

    # Check Markdown report rendering
    md_table = render_markdown_anova_table(rsm.anova_)
    assert "Source of Variation" in md_table
    assert "Model Fit:" in md_table


def test_anova_vs_shap_cross_validation_pipeline():
    """Verifies that ANOVA and SHAP extract matching factor rankings and interaction metrics."""
    rng = np.random.RandomState(42)
    pts = generate_fccd_points(n_center=4)
    X_coded = np.array([p.vector for p in pts])

    # Dominant factor is x2, followed by x1, with strong x1:x2 interaction
    x1, x2 = X_coded[:, 0], X_coded[:, 1]
    y = 0.40 - 0.25 * x2 + 0.12 * x1 + 0.15 * (x1 * x2) + rng.normal(0, 0.01, size=len(pts))

    rsm = PolynomialRSMSurrogate(include_blocks=False)
    rsm.fit(X_coded, y)

    anova_attr = extract_anova_attributions(rsm)
    # x2 should be top factor
    assert anova_attr.main_effect_ranking[0] == "x2"
    assert anova_attr.main_effect_ranking[1] == "x1"

    # Compute surrogate SHAP
    shap_attr = compute_surrogate_shap_attributions(
        surrogate_model=rsm,
        background_data=X_coded,
        eval_data=X_coded,
    )
    assert shap_attr.shap_ranking[0] == "x2"

    # Compare ANOVA and SHAP
    cmp_res = compare_anova_vs_shap(anova_attr, shap_attr)
    assert cmp_res.top_factor_anova == "x2"
    assert cmp_res.top_factor_shap == "x2"
    assert cmp_res.spearman_rho == pytest.approx(0.80, abs=0.01)
    assert cmp_res.top_interaction_anova == ("x1", "x2")

    summary_md = render_markdown_comparison_summary(cmp_res)
    assert "ANOVA vs SHAP Methodological Cross-Validation Summary" in summary_md


def test_pareto_front_and_hypervolume_calculation():
    """Verifies non-dominated Pareto front extraction and 2D hypervolume logic."""
    from xgb_doe_xai.profiler import LatencyProfileResult
    from xgb_doe_xai.benchmarks import EvaluationResult

    def make_res(cid: str, loss: float, lat: float) -> EvaluationResult:
        lp = LatencyProfileResult(
            predict_median_us=lat,
            predict_mean_us=lat,
            predict_iqm_us=lat,
            predict_sd_us=1.0,
            predict_p95_us=lat + 2.0,
            inplace_median_us=lat - 10.0,
            inplace_mean_us=lat - 10.0,
            inplace_iqm_us=lat - 10.0,
            n_warmup=10,
            n_timed=20,
            batch_size=1,
        )
        return EvaluationResult(
            config_id=cid,
            coded_params={"x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0},
            natural_params={"learning_rate": 0.175, "max_depth": 6, "subsample": 0.75, "colsample_bytree": 0.65},
            val_macro_f1=1.0 - loss,
            val_loss=loss,
            val_logloss=loss,
            latency=lp,
        )

    # 3 points: A is dominated by B, B and C form Pareto front
    # Point A: loss 0.30, lat 150
    # Point B: loss 0.20, lat 120 (dominates A)
    # Point C: loss 0.15, lat 180 (trade-off with B)
    evals = [
        make_res("A", 0.30, 150.0),
        make_res("B", 0.20, 120.0),
        make_res("C", 0.15, 180.0),
    ]

    pareto = extract_pareto_front(evals)
    pareto_ids = [p.config_id for p in pareto]
    assert "B" in pareto_ids
    assert "C" in pareto_ids
    assert "A" not in pareto_ids

    # Hypervolume calculation with reference (0.50, 200.0)
    pts = [(p.val_loss, p.latency.predict_median_us) for p in pareto]
    hv = compute_hypervolume_2d(pts, reference_point=(0.50, 200.0))
    assert hv > 0.0

    # Desirability formulation
    d_val = compute_derringer_suich_desirability(val_loss=0.20, latency_us=120.0)
    assert 0.0 <= d_val <= 1.0
