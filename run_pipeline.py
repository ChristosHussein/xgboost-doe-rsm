"""
run_pipeline.py - Master Execution Script for the Sequential DOE Project
========================================================================
Executes all 5 methodological phases from end-to-end:
1. Screening & Curvature Assessment (2^4 Full Factorial with 4 Center Points x 5 Blocks = 100 runs).
2. Central Composite Design Augmentation (8 Axial Points x 5 Blocks = 40 runs -> 140 total runs).
3. Canonical & Ridge Analysis (B-matrix spectral decomposition, eigenvalues, stationary point).
4. Multi-Objective Desirability Optimization (Derringer-Suich compromise optimum x*).
5. Confirmation Trials (5 runs at x*) and Empirical Benchmarks (Random Search & Optuna TPE, 140 trials each).
Generates all 300 DPI visual artifacts, formatted ANOVA tables, and statistical summaries.
Zero dummy data policy: 100% genuine model fits and holdout test evaluations.
"""

import os
import json
import time
import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from pipeline import (
    DataManager,
    generate_phase1_design,
    generate_phase2_axial_design,
    execute_design_runs,
    evaluate_model,
    run_random_search_benchmark,
    run_optuna_tpe_benchmark,
    BLOCK_SEEDS,
)
from analysis import (
    fit_first_order_model,
    assess_curvature_phase1,
    fit_second_order_model,
    decompose_pure_error_lack_of_fit,
    compute_icc_and_tukey,
    perform_residual_diagnostics,
    canonical_analysis,
    optimize_multi_objective_desirability,
)
from plots import (
    plot_residual_diagnostics,
    plot_response_surface_2d_3d,
    plot_pareto_front_and_desirability,
    plot_efficiency_comparison,
)

logger = logging.getLogger("DOE-Master")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    figs_dir = os.path.join(base_dir, "figures")
    tables_dir = os.path.join(base_dir, "tables")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 80)
    print("STARTING SCIENTIFIC-GRADE DOE HYPERPARAMETER OPTIMIZATION PIPELINE")
    print("Author: Montgomery DOE Experimental Framework (Chapters 5, 9, 10, 14)")
    print("Dataset: California Housing (Live Sklearn Fetch)")
    print("Model: XGBoost Regressor")
    print("Nuisance Blocking: 5 Stochastic Random Seeds")
    print("=" * 80)

    # Initialize Data Manager
    data_mgr = DataManager(test_size=0.20, fixed_seed=42)

    # =========================================================================
    # PHASE 1: SCREENING & CURVATURE ASSESSMENT (2^4 + 4 Center x 5 Blocks = 100 runs)
    # =========================================================================
    print("\n--- PHASE 1: SCREENING & CURVATURE ASSESSMENT ---")
    phase1_design = generate_phase1_design()
    print(f"Generated {len(phase1_design)} Phase 1 design runs across 5 blocks.")
    
    t0_p1 = time.time()
    df_phase1 = execute_design_runs(phase1_design, data_mgr)
    p1_duration = time.time() - t0_p1
    print(f"Phase 1 execution complete in {p1_duration:.1f}s.")
    
    # Save Phase 1 data
    df_phase1.to_csv(os.path.join(data_dir, "phase1_runs.csv"), index=False)

    # Fit Phase 1 First-Order + 2FI Model
    model_p1_y1, anova_p1_y1 = fit_first_order_model(df_phase1, response_col="Y1_RMSE")
    model_p1_y2, anova_p1_y2 = fit_first_order_model(df_phase1, response_col="Y2_Latency_us")
    
    anova_p1_y1.to_csv(os.path.join(tables_dir, "anova_phase1_rmse.csv"))
    anova_p1_y2.to_csv(os.path.join(tables_dir, "anova_phase1_latency.csv"))
    
    # Curvature Assessment
    curv_y1 = assess_curvature_phase1(df_phase1, response_col="Y1_RMSE")
    curv_y2 = assess_curvature_phase1(df_phase1, response_col="Y2_Latency_us")
    
    print("\nPhase 1 Curvature Check (RMSE Y1):")
    print(f"  Factorial Mean Y1 = {curv_y1['mean_Factorial']:.4f}")
    print(f"  Center Mean Y1    = {curv_y1['mean_Center']:.4f}")
    print(f"  Difference (F-C)  = {curv_y1['diff_F_minus_C']:.4f}")
    print(f"  SS_Curvature      = {curv_y1['SS_Curvature']:.6f}")
    print(f"  F-statistic       = {curv_y1['F_Curvature']:.4f}, p-value = {curv_y1['p_value']:.4e}")
    print(f"  Significant Curvature Present? {curv_y1['curvature_significant']}")

    # Phase 1 ICC and Block Nuisance
    icc_p1_y1, tukey_p1_y1 = compute_icc_and_tukey(df_phase1, response_col="Y1_RMSE")
    print(f"\nPhase 1 Intraclass Correlation Coefficient (ICC) for Block Effect = {icc_p1_y1:.4f}")

    # =========================================================================
    # PHASE 2: CENTRAL COMPOSITE DESIGN (CCD) AUGMENTATION (8 Star x 5 Blocks = 40 runs)
    # =========================================================================
    print("\n--- PHASE 2: CENTRAL COMPOSITE DESIGN AUGMENTATION ---")
    axial_design = generate_phase2_axial_design(start_run_id=101)
    print(f"Generated {len(axial_design)} axial (star) design runs across 5 blocks.")
    
    t0_p2 = time.time()
    df_axial = execute_design_runs(axial_design, data_mgr)
    p2_duration = time.time() - t0_p2
    print(f"Phase 2 axial execution complete in {p2_duration:.1f}s.")

    # Combine Phase 1 + Phase 2 into full CCD dataset (140 runs)
    df_ccd = pd.concat([df_phase1, df_axial], ignore_index=True)
    df_ccd.to_csv(os.path.join(data_dir, "ccd_runs.csv"), index=False)
    print(f"Full CCD dataset combined: Total {len(df_ccd)} runs.")

    # Fit Full Second-Order Response Surface Model
    model_ccd_y1, anova_ccd_y1 = fit_second_order_model(df_ccd, response_col="Y1_RMSE")
    model_ccd_y2, anova_ccd_y2 = fit_second_order_model(df_ccd, response_col="Y2_Latency_us")

    anova_ccd_y1.to_csv(os.path.join(tables_dir, "anova_ccd_second_order_rmse.csv"))
    anova_ccd_y2.to_csv(os.path.join(tables_dir, "anova_ccd_second_order_latency.csv"))

    # Decompose Residual SS into Pure Error and Lack of Fit
    lof_y1 = decompose_pure_error_lack_of_fit(df_ccd, model_ccd_y1, response_col="Y1_RMSE")
    lof_y2 = decompose_pure_error_lack_of_fit(df_ccd, model_ccd_y2, response_col="Y2_Latency_us")

    print("\nLack-of-Fit Decomposition for Second-Order Model (RMSE Y1):")
    print(f"  Residual SS   = {lof_y1['SS_Residual']:.6f} (DF = {lof_y1['DF_Residual']})")
    print(f"  Pure Error SS = {lof_y1['SS_Pure_Error']:.6f} (DF = {lof_y1['DF_Pure_Error']})")
    print(f"  Lack of Fit SS= {lof_y1['SS_Lack_of_Fit']:.6f} (DF = {lof_y1['DF_Lack_of_Fit']})")
    print(f"  F_LoF = {lof_y1['F_Lack_of_Fit']:.4f}, p-value = {lof_y1['p_value_LoF']:.4f}")
    print(f"  Model Adequate (No Significant Lack of Fit)? {not lof_y1['lack_of_fit_significant']}")

    # CCD ICC and Block Effect
    icc_ccd_y1, tukey_ccd_y1 = compute_icc_and_tukey(df_ccd, response_col="Y1_RMSE")
    print(f"\nCCD Intraclass Correlation Coefficient (ICC) for Block Effect = {icc_ccd_y1:.4f}")

    # =========================================================================
    # RESIDUAL DIAGNOSTICS
    # =========================================================================
    print("\n--- RESIDUAL DIAGNOSTICS & ASSUMPTION VERIFICATION ---")
    diag_y1 = perform_residual_diagnostics(model_ccd_y1, df_ccd, response_col="Y1_RMSE")
    print(f"  Shapiro-Wilk Normality Test: W = {diag_y1['shapiro_stat']:.4f}, p-value = {diag_y1['shapiro_pval']:.4f} (Normal: {diag_y1['residuals_normal']})")
    print(f"  Breusch-Pagan Homoscedasticity Test: LM = {diag_y1['breusch_pagan_stat']:.4f}, p-value = {diag_y1['breusch_pagan_pval']:.4f} (Homoscedastic: {diag_y1['homoscedastic_bp']})")
    print(f"  Levene Homoscedasticity Test across Blocks: W = {diag_y1['levene_stat']:.4f}, p-value = {diag_y1['levene_pval']:.4f} (Homoscedastic: {diag_y1['homoscedastic_levene']})")

    # Render 4-in-1 Diagnostics Panel
    plot_residual_diagnostics(diag_y1, os.path.join(figs_dir, "diagnostics_panel_4in1.png"))

    # =========================================================================
    # PHASE 3: CANONICAL & RIDGE ANALYSIS
    # =========================================================================
    print("\n--- PHASE 3: CANONICAL & RIDGE ANALYSIS ---")
    canonical_y1 = canonical_analysis(model_ccd_y1)
    x0 = canonical_y1["stationary_point_coded"]
    nat0 = canonical_y1["stationary_point_natural"]
    eigenvals = canonical_y1["eigenvalues"]
    
    print("Stationary Point x0 (Coded Space):")
    print(f"  x1 = {x0[0]:.4f}, x2 = {x0[1]:.4f}, x3 = {x0[2]:.4f}, x4 = {x0[3]:.4f}")
    print("Stationary Point (Natural Hyperparameters):")
    print(f"  eta = {nat0['eta']:.4f}, depth = {nat0['depth']}, subsample = {nat0['subsample']:.4f}, reg_lambda = {nat0['reg_lambda']:.4f}")
    print(f"Predicted Response at Stationary Point y0 = {canonical_y1['predicted_response_at_x0']:.4f}")
    print(f"Eigenvalues of B matrix: {eigenvals}")
    print(f"Surface Classification: {canonical_y1['surface_type']}")
    print(f"Stationary Point Inside Operational Domain [-1, 1]^4? {canonical_y1['inside_operational_domain']}")

    # Render 2D & 3D Response Surface Plots for RMSE and Latency
    factor_labels = {
        "x1": "Factor A: Learning Rate $x_1$ ($\ln(\eta)$)",
        "x2": "Factor B: Max Tree Depth $x_2$",
        "x3": "Factor C: Subsample Fraction $x_3$",
        "x4": "Factor D: L2 Regularization $x_4$ ($\ln(\lambda)$)",
    }
    # For RMSE: plot Depth (x2) vs. Learning Rate (x1), holding x3 and x4 at stationary point or center
    fixed_rmse = {"x3": float(np.clip(x0[2], -1, 1)), "x4": float(np.clip(x0[3], -1, 1))}
    plot_response_surface_2d_3d(
        model_results=model_ccd_y1,
        factor_x_name="x1",
        factor_y_name="x2",
        fixed_factors=fixed_rmse,
        factor_labels=factor_labels,
        response_label="Holdout Test RMSE ($Y_1$)",
        save_path=os.path.join(figs_dir, "response_surface_rmse_2d_3d.png"),
        stationary_coord=(x0[0], x0[1]),
    )

    # For Latency: plot Depth (x2) vs. Subsample (x3)
    fixed_lat = {"x1": 0.0, "x4": 0.0}
    plot_response_surface_2d_3d(
        model_results=model_ccd_y2,
        factor_x_name="x3",
        factor_y_name="x2",
        fixed_factors=fixed_lat,
        factor_labels=factor_labels,
        response_label="Inference Latency $\mu s$ ($Y_2$)",
        save_path=os.path.join(figs_dir, "response_surface_latency_2d_3d.png"),
    )

    # =========================================================================
    # PHASE 4: MULTI-OBJECTIVE DESIRABILITY OPTIMIZATION
    # =========================================================================
    print("\n--- PHASE 4: MULTI-OBJECTIVE DESIRABILITY OPTIMIZATION ---")
    opt_desirability = optimize_multi_objective_desirability(model_ccd_y1, model_ccd_y2, df_ccd)
    x_star = opt_desirability["x_star_coded"]
    nat_star = opt_desirability["x_star_natural"]
    
    print("Pareto-Optimal Compromise Coordinate x* (Coded Space):")
    print(f"  x1 = {x_star[0]:.4f}, x2 = {x_star[1]:.4f}, x3 = {x_star[2]:.4f}, x4 = {x_star[3]:.4f}")
    print("Pareto-Optimal Hyperparameters (Natural Space):")
    print(f"  eta = {nat_star['eta']:.4f}, depth = {nat_star['depth']}, subsample = {nat_star['subsample']:.4f}, reg_lambda = {nat_star['reg_lambda']:.4f}")
    print(f"Predicted Y1 (RMSE) at x* = {opt_desirability['pred_Y1_RMSE']:.4f}, 95% CI = [{opt_desirability['Y1_CI_95'][0]:.4f}, {opt_desirability['Y1_CI_95'][1]:.4f}]")
    print(f"Predicted Y2 (Latency) at x* = {opt_desirability['pred_Y2_Latency_us']:.2f} us")
    print(f"Individual Desirabilities: d1(RMSE) = {opt_desirability['desirability_d1']:.4f}, d2(Latency) = {opt_desirability['desirability_d2']:.4f}")
    print(f"Composite Desirability D(x*) = {opt_desirability['overall_desirability_D']:.4f}")

    # Render Pareto Front & Desirability Plot
    plot_pareto_front_and_desirability(df_ccd, opt_desirability, os.path.join(figs_dir, "desirability_pareto_front.png"))

    # =========================================================================
    # PHASE 5: EMPIRICAL BENCHMARK & CONFIRMATION
    # =========================================================================
    print("\n--- PHASE 5: EMPIRICAL BENCHMARK & CONFIRMATION ---")
    print("Running 5 confirmation trials at predicted optimum x* across all seed blocks...")
    confirm_records = []
    for block_idx, seed in enumerate(BLOCK_SEEDS, start=1):
        rmse_c, lat_c, fit_c = evaluate_model(
            eta=nat_star["eta"],
            depth=nat_star["depth"],
            subsample=nat_star["subsample"],
            reg_lambda=nat_star["reg_lambda"],
            seed=seed,
            data_mgr=data_mgr,
        )
        confirm_records.append({
            "trial": block_idx,
            "block": block_idx,
            "seed": seed,
            "eta": nat_star["eta"],
            "depth": nat_star["depth"],
            "subsample": nat_star["subsample"],
            "reg_lambda": nat_star["reg_lambda"],
            "Y1_RMSE": rmse_c,
            "Y2_Latency_us": lat_c,
            "fit_time_sec": fit_c,
        })
    df_confirm = pd.DataFrame(confirm_records)
    df_confirm.to_csv(os.path.join(data_dir, "confirmation_runs.csv"), index=False)

    mean_confirm_y1 = float(df_confirm["Y1_RMSE"].mean())
    std_confirm_y1 = float(df_confirm["Y1_RMSE"].std())
    mean_confirm_y2 = float(df_confirm["Y2_Latency_us"].mean())
    std_confirm_y2 = float(df_confirm["Y2_Latency_us"].std())

    ci_low, ci_high = opt_desirability["Y1_CI_95"]
    inside_ci = bool(ci_low <= mean_confirm_y1 <= ci_high)

    print(f"Confirmation Empirical Mean Y1 (RMSE) = {mean_confirm_y1:.4f} +/- {std_confirm_y1:.4f}")
    print(f"Predicted 95% Confidence Interval       = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Confirmation Mean Inside Predicted CI? {inside_ci}")
    print(f"Confirmation Empirical Mean Y2 (Latency) = {mean_confirm_y2:.2f} +/- {std_confirm_y2:.2f} us")

    # Run Unguided Random Search Benchmark (140 trials)
    print("\nExecuting Unguided Random Search Benchmark (Budget = 140 evaluations)...")
    df_rand = run_random_search_benchmark(n_trials=140, data_mgr=data_mgr, seed=42)
    df_rand.to_csv(os.path.join(data_dir, "benchmark_random.csv"), index=False)

    # Run Bayesian Optimization (Optuna TPE Benchmark, 140 trials)
    print("\nExecuting Bayesian Optimization Benchmark (Optuna TPE, Budget = 140 evaluations)...")
    df_opt = run_optuna_tpe_benchmark(n_trials=140, data_mgr=data_mgr, seed=42)
    df_opt.to_csv(os.path.join(data_dir, "benchmark_optuna.csv"), index=False)

    # Render Efficiency Comparison Curve
    plot_efficiency_comparison(df_ccd, df_rand, df_opt, os.path.join(figs_dir, "efficiency_comparison_curve.png"))

    # Summary Benchmark Metrics
    best_ccd_rmse = float(df_ccd["Y1_RMSE"].min())
    best_rand_rmse = float(df_rand["Y1_RMSE"].min())
    best_opt_rmse = float(df_opt["Y1_RMSE"].min())

    print("\n--- BENCHMARK COMPARISON SUMMARY (140 Function Evaluations) ---")
    print(f"  DOE-CCD Best RMSE:          {best_ccd_rmse:.4f}")
    print(f"  Random Search Best RMSE:    {best_rand_rmse:.4f}")
    print(f"  Optuna TPE Best RMSE:       {best_opt_rmse:.4f}")

    # Build and Save Complete Results JSON
    summary_results = {
        "dataset": "California Housing",
        "model": "XGBoost Regressor (n_estimators=100)",
        "seed_blocks": BLOCK_SEEDS,
        "phase1": {
            "n_runs": len(df_phase1),
            "curvature": curv_y1,
            "icc_block": icc_p1_y1,
        },
        "phase2_ccd": {
            "n_runs": len(df_ccd),
            "lack_of_fit_rmse": lof_y1,
            "lack_of_fit_latency": lof_y2,
            "icc_block": icc_ccd_y1,
        },
        "residual_diagnostics": {
            "shapiro_stat": diag_y1["shapiro_stat"],
            "shapiro_pval": diag_y1["shapiro_pval"],
            "breusch_pagan_stat": diag_y1["breusch_pagan_stat"],
            "breusch_pagan_pval": diag_y1["breusch_pagan_pval"],
            "levene_stat": diag_y1["levene_stat"],
            "levene_pval": diag_y1["levene_pval"],
        },
        "canonical_analysis": {
            "stationary_point_coded": list(canonical_y1["stationary_point_coded"]),
            "stationary_point_natural": canonical_y1["stationary_point_natural"],
            "predicted_y0": canonical_y1["predicted_response_at_x0"],
            "eigenvalues": list(canonical_y1["eigenvalues"]),
            "surface_type": canonical_y1["surface_type"],
            "inside_operational_domain": canonical_y1["inside_operational_domain"],
        },
        "multi_objective_optimum": {
            "x_star_coded": list(opt_desirability["x_star_coded"]),
            "x_star_natural": opt_desirability["x_star_natural"],
            "pred_Y1_RMSE": opt_desirability["pred_Y1_RMSE"],
            "Y1_CI_95": list(opt_desirability["Y1_CI_95"]),
            "pred_Y2_Latency_us": opt_desirability["pred_Y2_Latency_us"],
            "desirability_d1": opt_desirability["desirability_d1"],
            "desirability_d2": opt_desirability["desirability_d2"],
            "overall_desirability_D": opt_desirability["overall_desirability_D"],
        },
        "confirmation_trials": {
            "n_trials": len(df_confirm),
            "mean_rmse": mean_confirm_y1,
            "std_rmse": std_confirm_y1,
            "mean_latency_us": mean_confirm_y2,
            "std_latency_us": std_confirm_y2,
            "inside_predicted_ci": inside_ci,
        },
        "benchmarks": {
            "best_rmse_ccd": best_ccd_rmse,
            "best_rmse_random": best_rand_rmse,
            "best_rmse_optuna": best_opt_rmse,
            "median_rmse_ccd": float(df_ccd["Y1_RMSE"].median()),
            "median_rmse_random": float(df_rand["Y1_RMSE"].median()),
            "median_rmse_optuna": float(df_opt["Y1_RMSE"].median()),
        }
    }

    with open(os.path.join(base_dir, "results_summary.json"), "w") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[Master] Complete results summary written to {os.path.join(base_dir, 'results_summary.json')}")
    print("=" * 80)
    print("ALL EXPERIMENTAL PHASES AND BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
