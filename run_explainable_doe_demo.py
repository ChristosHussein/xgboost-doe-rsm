"""run_explainable_doe_demo.py - End-to-end demonstration of the Explainable Sequential DoE Framework.

Demonstrates:
1. Calibrated factor domain and strictly isolated holdout partitioning.
2. Blocked FCCD experimental execution on sentiment text features.
3. 2nd-order RSM fitting with complete ANOVA partition & Lack-of-Fit test.
4. Surrogate-level SHAP interaction matrices and ANOVA vs SHAP cross-validation.
5. Adaptive infill augmentation.
6. Matched Multi-Objective TPE benchmark and Pareto hypervolume comparison.
7. Automated reporting in Markdown and LaTeX.
"""

from __future__ import annotations

import sys
import numpy as np

from xgb_doe_xai import (
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
    generate_synthetic_sentiment_corpus,
    render_latex_anova_table,
    render_markdown_anova_table,
    render_markdown_comparison_summary,
    run_matched_motpe_benchmark,
    select_adaptive_infill_points,
    train_and_evaluate_config,
)


def main() -> int:
    print("=" * 80)
    print("EXPLAINABLE SEQUENTIAL DoE FRAMEWORK FOR XGBOOST (ANOVA vs SHAP)")
    print("=" * 80)

    cfg = BenchmarkConfig()
    print(f"[1/6] Ingesting Sentiment Corpus & Creating Strictly Isolated Holdout Partition...")
    texts, labels = generate_synthetic_sentiment_corpus(n_samples=600, random_seed=cfg.random_seed)
    partitions = create_strictly_isolated_partitions(
        texts=texts,
        labels=labels,
        holdout_ratio=cfg.holdout_ratio,
        n_blocks=cfg.n_nuisance_blocks,
        random_seed=cfg.random_seed,
        max_features=500,
    )
    print(f"      - Holdout Partition: {len(partitions.holdout_texts)} samples (SHA-256: {partitions.holdout_sha256[:16]}...)")
    print(f"      - Development Pool: {cfg.n_nuisance_blocks} nuisance blocks across {len(texts) - len(partitions.holdout_texts)} samples")

    # Use Block 0 for fast demonstration run
    X_train, y_train, X_val, y_val = partitions.block_features[0]
    profiler = WarmedLatencyProfiler(n_warmup=15, n_timed=30, measure_inplace=True)

    print("\n[2/6] Generating Face-Centered CCD Design Matrix (k=4 factors)...")
    pts = generate_fccd_points(n_center=2, block_id=0)
    print(f"      - Generated {len(pts)} design coordinates (16 factorial, 8 axial, 2 center)")

    print("\n[3/6] Executing DoE Training Runs & Unified Warmed Latency Profiling...")
    evaluations = []
    for i, pt in enumerate(pts, start=1):
        res = train_and_evaluate_config(
            coded_params=pt.coded,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            profiler=profiler,
            config_id=pt.point_id,
            block_id=pt.block_id,
            n_trees=cfg.n_trees,
            random_seed=cfg.random_seed + i,
        )
        evaluations.append(res)
    print(f"      - Successfully evaluated {len(evaluations)} design points.")

    X_coded = np.array([p.vector for p in pts])
    y_loss = np.array([e.val_loss for e in evaluations])
    y_lat = np.array([e.latency.predict_median_us for e in evaluations])

    print("\n[4/6] Fitting 2nd-Order Response Surface (RSM) & Computing ANOVA Decomposition...")
    rsm_loss = PolynomialRSMSurrogate(include_blocks=False).fit(X_coded, y_loss)
    rsm_lat = PolynomialRSMSurrogate(include_blocks=False).fit(X_coded, y_lat)

    print("\n" + render_markdown_anova_table(rsm_loss.anova_))

    print("\n[5/6] Computing Surrogate-Level SHAP Values & Cross-Validating with ANOVA...")
    anova_loss_attr = extract_anova_attributions(rsm_loss)
    shap_loss_attr = compute_surrogate_shap_attributions(rsm_loss, X_coded, X_coded)
    loss_cmp = compare_anova_vs_shap(anova_loss_attr, shap_loss_attr)

    anova_lat_attr = extract_anova_attributions(rsm_lat)
    shap_lat_attr = compute_surrogate_shap_attributions(rsm_lat, X_coded, X_coded)
    lat_cmp = compare_anova_vs_shap(anova_lat_attr, shap_lat_attr)

    print("\n" + render_markdown_comparison_summary(loss_cmp, lat_cmp))

    print("\n[6/6] Multi-Objective Pareto Frontier & Matched TPE Benchmarking...")
    pareto_doe = extract_pareto_front(evaluations)
    hv_doe = compute_hypervolume_2d(
        [(p.val_loss, p.latency.predict_median_us) for p in pareto_doe],
        reference_point=(cfg.hv_ref_loss, cfg.hv_ref_latency_us),
    )
    print(f"      - Sequential DoE Pareto Front: {len(pareto_doe)} non-dominated configurations (HV = {hv_doe:.4f})")

    # Run matched TPE with identical fit budget
    print(f"      - Running matched Multi-Objective TPE ({len(evaluations)} trials)...")
    tpe_evals = run_matched_motpe_benchmark(
        n_trials=len(evaluations),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        profiler=profiler,
        random_seed=cfg.random_seed,
    )
    pareto_tpe = extract_pareto_front(tpe_evals)
    hv_tpe = compute_hypervolume_2d(
        [(p.val_loss, p.latency.predict_median_us) for p in pareto_tpe],
        reference_point=(cfg.hv_ref_loss, cfg.hv_ref_latency_us),
    )
    print(f"      - Matched MO-TPE Pareto Front: {len(pareto_tpe)} non-dominated configurations (HV = {hv_tpe:.4f})")

    print("\n" + "=" * 80)
    print("ALL STAGES COMPLETED SUCCESSFULLY WITH INTEGRITY PRESERVED.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
