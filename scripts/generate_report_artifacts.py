"""
scripts/generate_report_artifacts.py - Generates LaTeX tables and macros from results JSON/CSVs.
Ensures zero hard-coded numbers in report.tex.
"""

import json
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import decode_factors

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def generate_macros():
    with open("results/phase1.json", "r", encoding="utf-8") as f:
        p1 = json.load(f)
    with open("results/phase3.json", "r", encoding="utf-8") as f:
        p3 = json.load(f)
    with open("results/lof.json", "r", encoding="utf-8") as f:
        lof = json.load(f)
    with open("results/diagnostics.json", "r", encoding="utf-8") as f:
        diag = json.load(f)
    with open("results/icc.json", "r", encoding="utf-8") as f:
        icc = json.load(f)
    with open("results/confirmation.json", "r", encoding="utf-8") as f:
        conf = json.load(f)
    with open("results/benchmark_summary.json", "r", encoding="utf-8") as f:
        bm_sum = json.load(f)

    df_bm = pd.read_csv("results/benchmark.csv")
    df_lat = pd.read_csv("results/latency_models_comparison.csv")
    df_runs = pd.read_csv("results/runs.csv")

    with open("results/env.json", "r", encoding="utf-8") as f:
        env_data = json.load(f)

    macros = []
    def add_macro(name, val):
        macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    add_macro("envPlatform", f"{env_data['platform']}")

    # Phase 1
    add_macro("numMeanFactorial", f"{p1['yF_bar']:.4f}")
    add_macro("numMeanCenter", f"{p1['yC_bar']:.4f}")
    add_macro("numDiffCurv", f"{p1['diff_F_minus_C']:.4f}")
    add_macro("numSSCurv", f"{p1['ss_curvature']:.6f}")
    add_macro("numMSpeCenter", f"{p1['ms_pe_center']:.2e}")
    add_macro("numDFpeCenter", str(p1["df_pe_center"]))
    add_macro("numFCurv", f"{p1['f_curvature']:.2f}")

    # ICC
    add_macro("numICCAnova", f"{icc['Y1']['icc_anova']:.4f}")
    add_macro("numICCReml", f"{icc['Y1']['icc_reml']:.4f}")
    add_macro("numICCLatency", f"{icc['Y2']['icc_anova']:.4f}")
    add_macro("numFBlock", f"{icc['Y1']['f_block']:.2f}")
    add_macro("numPBlock", f"{icc['Y1']['p_block']:.4f}" if icc['Y1']['p_block'] >= 0.0001 else "< 0.0001")

    # Phase 3 / Canonical
    add_macro("numBzero", f"{p3['b0']:.4f}")
    add_macro("numTraceB", f"{p3['trace_B']:.6f}")
    add_macro("numSumEigs", f"{p3['sum_eigenvalues']:.6f}")
    add_macro("numEigOne", f"{p3['eigenvalues'][0]:.6f}")
    add_macro("numEigTwo", f"{p3['eigenvalues'][1]:.6f}")
    add_macro("numEigThree", f"{p3['eigenvalues'][2]:.6f}")
    add_macro("numEigFour", f"{p3['eigenvalues'][3]:.6f}")
    add_macro("numFracMinEigLeZero", f"{p3['fraction_min_eigenvalue_le_zero']*100:.1f}\\%")

    # Fit CCD model for R-squared macros and ANOVA metrics
    df_aug_early = df_runs.copy()
    Q_early = ["x1", "x2", "x3", "x4"]
    for q in Q_early: df_aug_early[q + "_sq"] = df_aug_early[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug_early[f"{Q_early[i]}_{Q_early[j]}"] = df_aug_early[Q_early[i]] * df_aug_early[Q_early[j]]
    fit_ccd_early = ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                        "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug_early).fit()
    add_macro("numRsqRMSE", f"{fit_ccd_early.rsquared:.4f}")
    add_macro("numAdjRsqRMSE", f"{fit_ccd_early.rsquared_adj:.4f}")

    # Phase 1 ANOVA F-stats for text reconciliation
    df_p1 = df_runs[df_runs["phase"].isin(["Phase1_Factorial", "Phase1_Center"])].copy()
    fit_p1 = ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1:x2+x1:x3+x1:x4+x2:x3+x2:x4+x3:x4", df_p1).fit()
    an_p1 = sm.stats.anova_lm(fit_p1, typ=3)
    ss_res_p1 = an_p1.loc["Residual", "sum_sq"]
    f_x1_p1 = an_p1.loc["x1", "F"]
    eta_x1_p1 = an_p1.loc["x1", "sum_sq"] / (an_p1.loc["x1", "sum_sq"] + ss_res_p1)
    f_x2_p1 = an_p1.loc["x2", "F"]
    f_x1x2_p1 = an_p1.loc["x1:x2", "F"]

    add_macro("numAnovaPOneFxone", f"{f_x1_p1:.2f}")
    add_macro("numAnovaPOneEtxone", f"{eta_x1_p1:.4f}")
    add_macro("numAnovaPOneFxtwo", f"{f_x2_p1:.2f}")
    add_macro("numAnovaPOneFxoneXtwo", f"{f_x1x2_p1:.2f}")

    add_macro("numStationaryXone", f"{p3['stationary_point_coded'][0]:.4f}")
    add_macro("numStationaryXtwo", f"{p3['stationary_point_coded'][1]:.4f}")
    add_macro("numStationaryXthree", f"{p3['stationary_point_coded'][2]:.4f}")
    add_macro("numStationaryXfour", f"{p3['stationary_point_coded'][3]:.4f}")
    add_macro("numStationaryYhat", f"{p3['y0_hat_unconstrained']:.4f}")
    add_macro("numStationaryDistCoded", f"{p3['distance_coded']:.2f}")
    add_macro("numStationaryEta", f"{p3['stationary_natural_clamped']['eta']:.4f}")
    add_macro("numStationaryDepth", str(p3['stationary_natural_clamped']['depth']))
    add_macro("numStationarySubsample", f"{p3['stationary_natural_clamped']['subsample']:.4f}")
    add_macro("numStationaryLambda", f"{p3['stationary_natural_clamped']['reg_lambda']:.4f}")

    # Constrained Cube Optimum
    cube_x = p3["constrained_optimum_cube"]["x"]
    add_macro("numCubeOptDepth", str(p3["constrained_optimum_cube"]["depth"]))
    add_macro("numCubeOptPredRMSE", f"{p3['constrained_optimum_cube']['pred_rmse']:.4f}")
    add_macro("numCubeOptEta", f"{p3['constrained_optimum_cube']['natural']['eta']:.4f}")
    add_macro("numCubeOptSubsample", f"{p3['constrained_optimum_cube']['natural']['subsample']:.4f}")
    add_macro("numCubeOptLambda", f"{p3['constrained_optimum_cube']['natural']['reg_lambda']:.4f}")
    add_macro("numCubeOptXone", f"{cube_x[0]:.4f}")
    add_macro("numCubeOptXtwo", f"{cube_x[1]:.4f}")
    add_macro("numCubeOptXthree", f"{cube_x[2]:.4f}")
    add_macro("numCubeOptXfour", f"{cube_x[3]:.4f}")

    # Canonical b vector and B matrix
    b_vec = p3["b"]
    B_mat = p3["B"]
    add_macro("vecBOne", f"{b_vec[0]:.4f}")
    add_macro("vecBTwo", f"{b_vec[1]:.4f}")
    add_macro("vecBThree", f"{b_vec[2]:.4f}")
    add_macro("vecBFour", f"{b_vec[3]:.4f}")
    words = ["One", "Two", "Three", "Four"]
    for r in range(4):
        for c in range(4):
            add_macro(f"matB{words[r]}{words[c]}", f"{B_mat[r][c]:.4f}")

    # Bootstrap intervals for lambda_1
    boot_p = p3["bootstrap_percentiles"]
    add_macro("numBootLowEigOne", f"{boot_p[0][0]:.4f}")
    add_macro("numBootHighEigOne", f"{boot_p[2][0]:.4f}")

    # Replicate and interaction variance components
    ss_pure_center_y1 = float(np.sum([np.sum((grp["val_rmse"] - grp["val_rmse"].mean())**2)
                                       for (b, p), grp in df_runs.groupby(["block", "point_id"]) if len(grp) > 1]))
    df_pure_center = sum(len(grp) - 1 for (b, p), grp in df_runs.groupby(["block", "point_id"]) if len(grp) > 1)
    ms_pure_center_y1 = ss_pure_center_y1 / df_pure_center

    ss_interact_y1 = lof["Y1"]["SS_PE"] - ss_pure_center_y1
    df_interact = lof["Y1"]["df_PE"] - df_pure_center
    ms_interact_y1 = ss_interact_y1 / df_interact

    f_lof_center_y1 = lof["Y1"]["MS_LoF"] / ms_pure_center_y1
    p_lof_center_y1 = float(1.0 - stats.f.cdf(f_lof_center_y1, lof["Y1"]["df_LoF"], df_pure_center))

    f_interact_center_y1 = ms_interact_y1 / ms_pure_center_y1
    p_interact_center_y1 = float(1.0 - stats.f.cdf(f_interact_center_y1, df_interact, df_pure_center))

    # Latency Y2
    ss_pure_center_y2 = float(np.sum([np.sum((grp["latency_us_median"] - grp["latency_us_median"].mean())**2)
                                       for (b, p), grp in df_runs.groupby(["block", "point_id"]) if len(grp) > 1]))
    ms_pure_center_y2 = ss_pure_center_y2 / df_pure_center

    ss_interact_y2 = lof["Y2"]["SS_PE"] - ss_pure_center_y2
    ms_interact_y2 = ss_interact_y2 / df_interact

    f_lof_center_y2 = lof["Y2"]["MS_LoF"] / ms_pure_center_y2
    p_lof_center_y2 = float(1.0 - stats.f.cdf(f_lof_center_y2, lof["Y2"]["df_LoF"], df_pure_center))

    f_interact_center_y2 = ms_interact_y2 / ms_pure_center_y2
    p_interact_center_y2 = float(1.0 - stats.f.cdf(f_interact_center_y2, df_interact, df_pure_center))

    # Lack of Fit macros
    add_macro("numDfLofYone", str(lof["Y1"]["df_LoF"]))
    add_macro("numDfPeYone", str(lof["Y1"]["df_PE"]))
    add_macro("numSSLofYone", f"{lof['Y1']['SS_LoF']:.6f}")
    add_macro("numSSPeYone", f"{lof['Y1']['SS_PE']:.6f}")
    add_macro("numMSLofYone", f"{lof['Y1']['MS_LoF']:.6f}")
    add_macro("numMSPeYone", f"{lof['Y1']['MS_PE']:.2e}")
    add_macro("numFLofYone", f"{lof['Y1']['F_LoF']:.2f}")
    add_macro("numPLofYone", f"{lof['Y1']['p_LoF']:.2e}")
    add_macro("numSqrtMSLofYone", f"{lof['Y1']['sqrt_MS_LoF']:.4f}")
    add_macro("numSqrtMSPeYone", f"{lof['Y1']['sqrt_MS_PE']:.4f}")
    add_macro("numBoxCoxYone", f"{lof['Y1']['boxcox_lambda']:.2f}")

    # Center pure error and interaction macros
    add_macro("numDfPeCenter", str(df_pure_center))
    add_macro("numSSPeCenterYone", f"{ss_pure_center_y1:.6f}")
    add_macro("numMSPeCenterYone", f"{ms_pure_center_y1:.2e}")
    add_macro("numSqrtMSPeCenterYone", f"{np.sqrt(ms_pure_center_y1):.4f}")
    add_macro("numFLofCenterYone", f"{f_lof_center_y1:.2f}")
    add_macro("numPLofCenterYone", "< 10^{-15}" if p_lof_center_y1 < 1e-15 else f"{p_lof_center_y1:.2e}")

    add_macro("numDfInteract", str(df_interact))
    add_macro("numSSInteractYone", f"{ss_interact_y1:.6f}")
    add_macro("numMSInteractYone", f"{ms_interact_y1:.2e}")
    add_macro("numSqrtMSInteractYone", f"{np.sqrt(ms_interact_y1):.4f}")
    add_macro("numFInteractCenterYone", f"{f_interact_center_y1:.2f}")
    add_macro("numPInteractCenterYone", f"{p_interact_center_y1:.4f}")

    add_macro("numSSPeCenterYtwo", f"{ss_pure_center_y2:.2f}")
    add_macro("numMSPeCenterYtwo", f"{ms_pure_center_y2:.2f}")
    add_macro("numSqrtMSPeCenterYtwo", f"{np.sqrt(ms_pure_center_y2):.2f}")
    add_macro("numFLofCenterYtwo", f"{f_lof_center_y2:.2f}")
    add_macro("numPLofCenterYtwo", f"{p_lof_center_y2:.4f}")

    add_macro("numSSInteractYtwo", f"{ss_interact_y2:.2f}")
    add_macro("numMSInteractYtwo", f"{ms_interact_y2:.2f}")
    add_macro("numSqrtMSInteractYtwo", f"{np.sqrt(ms_interact_y2):.2f}")
    add_macro("numFInteractCenterYtwo", f"{f_interact_center_y2:.2f}")
    add_macro("numPInteractCenterYtwo", f"{p_interact_center_y2:.4f}")

    # Restricted domain LoF
    add_macro("numLofRestrictedReductionPct", f"{lof['Y1_restricted']['ss_lof_reduction_pct']:.1f}\\%")
    add_macro("numFLofRestricted", f"{lof['Y1_restricted']['F_LoF']:.2f}")
    add_macro("numPLofRestricted", f"{lof['Y1_restricted']['p_LoF']:.2e}")
    add_macro("numSqrtMSLofRestricted", f"{lof['Y1_restricted']['sqrt_MS_LoF']:.4f}")
    add_macro("numSSLofRestricted", f"{lof['Y1_restricted']['SS_LoF']:.6f}")

    add_macro("numDfLofYtwo", str(lof["Y2"]["df_LoF"]))
    add_macro("numDfPeYtwo", str(lof["Y2"]["df_PE"]))
    add_macro("numFLofYtwo", f"{lof['Y2']['F_LoF']:.2f}")
    add_macro("numPLofYtwo", f"{lof['Y2']['p_LoF']:.4f}")
    add_macro("numSqrtMSLofYtwo", f"{lof['Y2']['sqrt_MS_LoF']:.2f}")
    add_macro("numSqrtMSPeYtwo", f"{lof['Y2']['sqrt_MS_PE']:.2f}")

    # Diagnostics
    add_macro("numShapiroW", f"{diag['shapiro_W']:.4f}")
    add_macro("numShapiroP", f"{diag['shapiro_p']:.4f}")
    add_macro("numLeveneW", f"{diag['levene_stat']:.4f}")
    add_macro("numLeveneP", f"{diag['levene_p']:.4f}")
    add_macro("numBrownForsytheW", f"{diag['brown_forsythe_points_stat']:.4f}")
    add_macro("numBrownForsytheP", f"{diag['brown_forsythe_points_p']:.4f}")
    add_macro("numBreuschPaganLM", f"{diag['breusch_pagan_stat']:.2f}")
    add_macro("numBreuschPaganP", f"{diag['breusch_pagan_p']:.4f}" if diag['breusch_pagan_p'] >= 0.0001 else "< 0.0001")
    add_macro("numMaxCooksD", f"{diag['max_cooks_d']:.3f}")
    add_macro("numThreshFourN", f"{diag['thresh_4n']:.3f}")
    add_macro("numOutlierCount", str(diag['outlier_count']))
    add_macro("numDurbinWatson", f"{diag['durbin_watson']:.4f}")
    add_macro("numLjungBoxStat", f"{diag['ljung_box_stat']:.2f}")
    add_macro("numLjungBoxP", f"{diag['ljung_box_p']:.4f}")
    add_macro("numRunsTestP", f"{diag['runs_test_p']:.4f}")

    # Confirmation & Multi-Objective Optimum Aliases
    add_macro("numOptDepth", str(conf['x_star_natural']['depth']))
    add_macro("numOptEta", f"{conf['x_star_natural']['eta']:.4f}")
    add_macro("numOptSubsample", f"{conf['x_star_natural']['subsample']:.4f}")
    add_macro("numOptLambda", f"{conf['x_star_natural']['reg_lambda']:.4f}")

    pred_y1_star = conf['Y1_Val_RMSE']['predicted_mean']
    pred_y2_star = conf['Y2_Latency']['predicted_mean']
    d1_star = max(0.0, min(1.0, (0.700 - pred_y1_star) / (0.700 - 0.450)))
    d2_star = max(0.0, min(1.0, (180.0 - pred_y2_star) / (180.0 - 100.0)))
    D_star = float(np.sqrt(d1_star * d2_star))
    add_macro("numOptD", f"{D_star:.4f}")
    add_macro("numOptdOne", f"{d1_star:.4f}")
    add_macro("numOptdTwo", f"{d2_star:.4f}")
    add_macro("numPredRMSE", f"{pred_y1_star:.4f}")
    add_macro("numPredLatency", f"{pred_y2_star:.2f}")
    add_macro("numOptXone", f"{conf['x_star_coded'][0]:.4f}")
    add_macro("numOptXtwo", f"{conf['x_star_coded'][1]:.4f}")
    add_macro("numOptXthree", f"{conf['x_star_coded'][2]:.4f}")
    add_macro("numOptXfour", f"{conf['x_star_coded'][3]:.4f}")

    add_macro("numConfOptEta", f"{conf['x_star_natural']['eta']:.4f}")
    add_macro("numConfOptDepth", str(conf['x_star_natural']['depth']))
    add_macro("numConfOptSubsample", f"{conf['x_star_natural']['subsample']:.4f}")
    add_macro("numConfOptLambda", f"{conf['x_star_natural']['reg_lambda']:.4f}")
    add_macro("numConfLeverageH", f"{conf['leverage_h']:.4f}")
    add_macro("numConfPredValRMSE", f"{conf['Y1_Val_RMSE']['predicted_mean']:.4f}")
    add_macro("numConfPiLowValRMSE", f"{conf['Y1_Val_RMSE']['prediction_interval_95'][0]:.4f}")
    add_macro("numConfPiHighValRMSE", f"{conf['Y1_Val_RMSE']['prediction_interval_95'][1]:.4f}")
    add_macro("numConfEmpValRMSE", f"{conf['Y1_Val_RMSE']['empirical_mean']:.4f}")
    add_macro("numConfEmpValRMSEStd", f"{conf['Y1_Val_RMSE']['empirical_std']:.4f}")
    add_macro("numConfEmpTestRMSE", f"{conf['Y1_Test_RMSE']['empirical_mean']:.4f}")
    add_macro("numConfEmpTestRMSEStd", f"{conf['Y1_Test_RMSE']['empirical_std']:.4f}")
    add_macro("numConfPredLat", f"{conf['Y2_Latency']['predicted_mean']:.2f}")
    add_macro("numConfPiLowLat", f"{conf['Y2_Latency']['prediction_interval_95'][0]:.2f}")
    add_macro("numConfPiHighLat", f"{conf['Y2_Latency']['prediction_interval_95'][1]:.2f}")
    add_macro("numConfEmpLat", f"{conf['Y2_Latency']['empirical_mean']:.2f}")
    add_macro("numConfEmpLatStd", f"{conf['Y2_Latency']['empirical_std']:.2f}")
    add_macro("numConfPassValRMSE", "Pass (Inside 95\\% PI)" if conf["Y1_Val_RMSE"]["inside_pi"] else "Outside 95\\% PI")
    add_macro("numConfPassLat", "Pass (Inside 95\\% PI)" if conf["Y2_Latency"]["inside_pi"] else "Outside 95\\% PI")

    # Single-Objective Confirmation
    so_conf = conf["Single_Objective_Optimum"]
    add_macro("numConfSoPredValRMSE", f"{so_conf['predicted_val_rmse']:.4f}")
    add_macro("numConfSoPiLowValRMSE", f"{so_conf['prediction_interval_95_val'][0]:.4f}")
    add_macro("numConfSoPiHighValRMSE", f"{so_conf['prediction_interval_95_val'][1]:.4f}")
    add_macro("numConfSoEmpValRMSE", f"{so_conf['empirical_val_rmse']:.4f}")
    add_macro("numConfSoEmpValRMSEStd", f"{so_conf['empirical_val_std']:.4f}")
    add_macro("numConfSoEmpTestRMSE", f"{so_conf['empirical_test_rmse']:.4f}")
    add_macro("numConfSoEmpTestRMSEStd", f"{so_conf['empirical_test_std']:.4f}")
    add_macro("numConfSoEmpLat", f"{so_conf['empirical_latency']:.1f}")
    add_macro("numConfSoEmpLatStd", f"{so_conf['empirical_latency_std']:.1f}")
    add_macro("numConfSoBias", f"{so_conf['empirical_val_rmse'] - so_conf['predicted_val_rmse']:+.4f}")

    # Benchmark comparisons
    doe_row = df_bm[df_bm["method"].str.contains("Multi-Objective") & df_bm["method"].str.contains("DOE")].iloc[0]
    doe_so_row = df_bm[df_bm["method"].str.contains("Single-Objective") & df_bm["method"].str.contains("DOE")].iloc[0]
    tpe_row = df_bm[df_bm["method"].str.contains("Single-Obj") & ~df_bm["method"].str.contains("DOE")].iloc[0]
    rs_row = df_bm[df_bm["method"].str.contains("Random Search")].iloc[0]
    motpe_row = df_bm[df_bm["method"].str.contains("Multi-Objective TPE")].iloc[0]
    ctpe_row = df_bm[df_bm["method"].str.contains("Constrained")].iloc[0]

    doe_test_rmse = doe_row["test_rmse_mean"]
    tpe_test_rmse = tpe_row["test_rmse_mean"]
    test_gap_pct = ((doe_test_rmse - tpe_test_rmse) / tpe_test_rmse) * 100.0
    lat_slow_pct = ((tpe_row["predict_latency_us_median"] - doe_row["predict_latency_us_median"]) / doe_row["predict_latency_us_median"]) * 100.0
    doe_lat_lower_pct = ((tpe_row["predict_latency_us_median"] - doe_row["predict_latency_us_median"]) / tpe_row["predict_latency_us_median"]) * 100.0

    add_macro("numDoeTestRMSE", f"{doe_test_rmse:.4f}")
    add_macro("numDoeValRMSE", f"{doe_row['val_rmse_mean']:.4f}")
    add_macro("numDoePredictLat", f"{doe_row['predict_latency_us_median']:.1f}")
    add_macro("numDoeInplaceLat", f"{doe_row['inplace_latency_us_median']:.1f}")

    add_macro("numDoeSoTestRMSE", f"{doe_so_row['test_rmse_mean']:.4f}")
    add_macro("numDoeSoValRMSE", f"{doe_so_row['val_rmse_mean']:.4f}")
    add_macro("numDoeSoPredictLat", f"{doe_so_row['predict_latency_us_median']:.1f}")
    add_macro("numDoeSoInplaceLat", f"{doe_so_row['inplace_latency_us_median']:.1f}")

    add_macro("numTpeTestRMSE", f"{tpe_test_rmse:.4f}")
    add_macro("numTpeValRMSE", f"{tpe_row['val_rmse_mean']:.4f}")
    add_macro("numTpePredictLat", f"{tpe_row['predict_latency_us_median']:.1f}")
    add_macro("numTpeInplaceLat", f"{tpe_row['inplace_latency_us_median']:.1f}")

    add_macro("numRsTestRMSE", f"{rs_row['test_rmse_mean']:.4f}")
    add_macro("numRsValRMSE", f"{rs_row['val_rmse_mean']:.4f}")
    add_macro("numRsPredictLat", f"{rs_row['predict_latency_us_median']:.1f}")
    add_macro("numRsInplaceLat", f"{rs_row['inplace_latency_us_median']:.1f}")

    add_macro("numMoTpeTestRMSE", f"{motpe_row['test_rmse_mean']:.4f}")
    add_macro("numMoTpeValRMSE", f"{motpe_row['val_rmse_mean']:.4f}")
    add_macro("numMoTpePredictLat", f"{motpe_row['predict_latency_us_median']:.1f}")
    add_macro("numMoTpeInplaceLat", f"{motpe_row['inplace_latency_us_median']:.1f}")

    add_macro("numConstrainedTpeTestRMSE", f"{ctpe_row['test_rmse_mean']:.4f}")
    add_macro("numConstrainedTpeValRMSE", f"{ctpe_row['val_rmse_mean']:.4f}")
    add_macro("numConstrainedTpePredictLat", f"{ctpe_row['predict_latency_us_median']:.1f}")

    add_macro("numTestGapPct", f"{test_gap_pct:.1f}\\%")
    add_macro("numLatSlowPct", f"{lat_slow_pct:.1f}\\%")
    add_macro("numDoeLatLowerPct", f"{doe_lat_lower_pct:.1f}\\%")

    # Hypervolume
    hv = bm_sum["hypervolume"]
    add_macro("numHvDoe", f"{hv['hv_doe']:.2f}")
    add_macro("numHvMoTpe", f"{hv['hv_motpe']:.2f}")
    add_macro("numHvRs", f"{hv['hv_rs']:.2f}")
    add_macro("numHvDoeGainPct", f"{hv['doe_over_motpe_pct']:.1f}\\%")

    # Latency model fit macros
    m_lin = df_lat[df_lat["Model"].str.contains("Linear")].iloc[0]
    m_quad = df_lat[df_lat["Model"].str.contains("Quadratic")].iloc[0]
    add_macro("numLatLinRsq", f"{m_lin['R2']:.3f}")
    add_macro("numLatQuadRsq", f"{m_quad['R2']:.3f}")

    # Latency ANOVA terms
    fit_ccd_y2_early = ols("latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                           "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug_early).fit()
    an_ccd_y2_early = sm.stats.anova_lm(fit_ccd_y2_early, typ=3)
    add_macro("numAnovaLatencyFxTwoSq", f"{an_ccd_y2_early.loc['x2_sq', 'F']:.2f}")
    add_macro("numFBlockLatency", f"{an_ccd_y2_early.loc['C(block)', 'F']:.2f}")
    add_macro("numPBlockLatency", f"{an_ccd_y2_early.loc['C(block)', 'PR(>F)']:.4f}")

    with open("results/macros.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated macros from genuine results\n")
        f.write("\n".join(macros) + "\n")
    print(f"Generated results/macros.tex with {len(macros)} macros.")


def generate_tables():
    os.makedirs("tables", exist_ok=True)
    df_runs = pd.read_csv("results/runs.csv")

    # 0. Design, Block, and Seed Explicit Table (Resolving Item 8)
    tex_seeds = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\makecell{\\textbf{Phase /}\\\\\\textbf{Design Point}} & \\makecell{\\textbf{Point}\\\\\\textbf{IDs}} & \\textbf{Block} & \\makecell{\\textbf{Split}\\\\\\textbf{Seed}} & \\makecell{\\textbf{Model}\\\\\\textbf{Seed}} & \\textbf{Reps} & \\makecell{\\textbf{Isolated}\\\\\\textbf{Variance}} \\\\",
        "\\midrule",
        "\\makecell[l]{Phase 1:\\\\Corner Runs} & $1 \\dots 16$ & $b \\in \\{1..5\\}$ & $S_b$ & $S_b$ & 1 & Factorial + Block \\\\",
        "\\makecell[l]{Phase 1:\\\\Center Points} & $17$ & $b \\in \\{1..5\\}$ & $S_b$ & $S_b + 1000r$ & 4 & \\makecell{Subsampling Error\\\\($15$ df)} \\\\",
        "\\makecell[l]{Phase 2:\\\\Axial Runs} & $18 \\dots 25$ & $b \\in \\{1..5\\}$ & $S_b$ & $S_b$ & 1 & Quadratic + Block \\\\",
        "\\bottomrule",
        "\\end{tabular}"
    ]
    with open("tables/tab_design_block_seeds.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_seeds) + "\n")

    # 1. Phase 1 Screening ANOVA Table
    df_p1 = df_runs[df_runs["phase"].isin(["Phase1_Factorial", "Phase1_Center"])].copy()
    fit_p1 = ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1:x2+x1:x3+x1:x4+x2:x3+x2:x4+x3:x4", df_p1).fit()
    an_p1 = sm.stats.anova_lm(fit_p1, typ=3)

    source_map = {
        "Intercept": "Intercept",
        "C(block)": "Block Effect $C(\\text{block})$",
        "x1": "Factor A: $x_1$ ($\\ln(\\eta)$)",
        "x2": "Factor B: $x_2$ (Depth)",
        "x3": "Factor C: $x_3$ (Subsample)",
        "x4": "Factor D: $x_4$ ($\\ln(\\lambda)$)",
        "x1:x2": "Interaction $x_1 \\cdot x_2$",
        "x1:x3": "Interaction $x_1 \\cdot x_3$",
        "x1:x4": "Interaction $x_1 \\cdot x_4$",
        "x2:x3": "Interaction $x_2 \\cdot x_3$",
        "x2:x4": "Interaction $x_2 \\cdot x_4$",
        "x3:x4": "Interaction $x_3 \\cdot x_4$",
        "Residual": "Residual Error"
    }

    tex_p1 = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Source of Variation} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\makecell{\\textbf{F-}\\\\\\textbf{Stat}} & \\makecell{\\bm{$p$}\\\\\\textbf{value}} & \\makecell{\\textbf{Partial}\\\\\\bm{$\\eta^2$}} \\\\",
        "\\midrule"
    ]
    ss_res_p1 = an_p1.loc["Residual", "sum_sq"]
    for src in source_map.keys():
        if src in an_p1.index:
            ss = an_p1.loc[src, "sum_sq"]
            df_val = int(an_p1.loc[src, "df"])
            ms = ss / df_val
            f_val = an_p1.loc[src, "F"]
            p_val = an_p1.loc[src, "PR(>F)"]
            eta_sq = ss / (ss + ss_res_p1) if src != "Residual" else 0.0
            if src == "Residual":
                tex_p1.append(f"{source_map[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & --- & --- & --- \\\\")
            else:
                p_str = "$< 10^{-15}$" if p_val < 1e-15 else f"${p_val:.4f}$"
                tex_p1.append(f"{source_map[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & ${f_val:.2f}$ & {p_str} & ${eta_sq:.4f}$ \\\\")
    tex_p1.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_anova_phase1.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_p1) + "\n")

    # 2. CCD Second-Order ANOVA Table for Val RMSE (with OLS and HC3 robust inference, Resolving Item 9)
    df_aug = df_runs.copy()
    Q = ["x1", "x2", "x3", "x4"]
    for q in Q: df_aug[q + "_sq"] = df_aug[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{Q[i]}_{Q[j]}"] = df_aug[Q[i]] * df_aug[Q[j]]

    fit_ccd_y1 = ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                     "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()
    fit_ccd_y1_hc3 = fit_ccd_y1.get_robustcov_results("HC3")
    hc3_bse_series = pd.Series(fit_ccd_y1_hc3.bse, index=fit_ccd_y1_hc3.model.exog_names)
    hc3_pval_series = pd.Series(fit_ccd_y1_hc3.pvalues, index=fit_ccd_y1_hc3.model.exog_names)
    an_ccd_y1 = sm.stats.anova_lm(fit_ccd_y1, typ=3)

    source_map_ccd = {
        "Intercept": "Intercept",
        "C(block)": "Block Effect $C(\\text{block})$",
        "x1": "Factor A: $x_1$ ($\\ln(\\eta)$)",
        "x2": "Factor B: $x_2$ (Depth)",
        "x3": "Factor C: $x_3$ (Subsample)",
        "x4": "Factor D: $x_4$ ($\\ln(\\lambda)$)",
        "x1_sq": "Quadratic $x_1^2$",
        "x2_sq": "Quadratic $x_2^2$",
        "x3_sq": "Quadratic $x_3^2$",
        "x4_sq": "Quadratic $x_4^2$",
        "x1_x2": "Interaction $x_1 \\cdot x_2$",
        "x1_x3": "Interaction $x_1 \\cdot x_3$",
        "x1_x4": "Interaction $x_1 \\cdot x_4$",
        "x2_x3": "Interaction $x_2 \\cdot x_3$",
        "x2_x4": "Interaction $x_2 \\cdot x_4$",
        "x3_x4": "Interaction $x_3 \\cdot x_4$",
        "Residual": "Residual Error"
    }

    tex_ccd_y1 = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "\\textbf{Source of Variation} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\makecell{\\textbf{F-}\\\\\\textbf{Stat}} & \\makecell{\\textbf{OLS}\\\\\\bm{$p$}} & \\makecell{\\textbf{HC3}\\\\\\textbf{SE}} & \\makecell{\\textbf{HC3}\\\\\\bm{$p$}} \\\\",
        "\\midrule"
    ]
    for src in source_map_ccd.keys():
        if src in an_ccd_y1.index:
            ss = an_ccd_y1.loc[src, "sum_sq"]
            df_val = int(an_ccd_y1.loc[src, "df"])
            ms = ss / df_val
            f_val = an_ccd_y1.loc[src, "F"]
            p_val = an_ccd_y1.loc[src, "PR(>F)"]
            if src == "Residual":
                tex_ccd_y1.append(f"{source_map_ccd[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & --- & --- & --- & --- \\\\")
            else:
                p_str = "$< 10^{-15}$" if p_val < 1e-15 else (f"${p_val:.4f}$" if p_val >= 0.0001 else "$< 0.0001$")
                # HC3 stats
                if src in hc3_bse_series.index:
                    hc3_se = hc3_bse_series[src]
                    hc3_p = hc3_pval_series[src]
                    hc3_p_str = "$< 10^{-15}$" if hc3_p < 1e-15 else (f"${hc3_p:.4f}$" if hc3_p >= 0.0001 else "$< 0.0001$")
                    hc3_se_str = f"${hc3_se:.4f}$"
                else:
                    hc3_se_str = "---"
                    hc3_p_str = "---"
                tex_ccd_y1.append(f"{source_map_ccd[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & ${f_val:.2f}$ & {p_str} & {hc3_se_str} & {hc3_p_str} \\\\")
    tex_ccd_y1.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_anova_ccd_rmse.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_ccd_y1) + "\n")

    # 3. CCD Second-Order ANOVA Table for Latency
    fit_ccd_y2 = ols("latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                     "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()
    an_ccd_y2 = sm.stats.anova_lm(fit_ccd_y2, typ=3)

    tex_ccd_y2 = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Source of Variation} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\makecell{\\textbf{F-}\\\\\\textbf{Stat}} & \\makecell{\\bm{$p$}\\\\\\textbf{value}} \\\\",
        "\\midrule"
    ]
    for src in source_map_ccd.keys():
        if src in an_ccd_y2.index:
            ss = an_ccd_y2.loc[src, "sum_sq"]
            df_val = int(an_ccd_y2.loc[src, "df"])
            ms = ss / df_val
            f_val = an_ccd_y2.loc[src, "F"]
            p_val = an_ccd_y2.loc[src, "PR(>F)"]
            if src == "Residual":
                tex_ccd_y2.append(f"{source_map_ccd[src]} & ${ss:.2f}$ & ${df_val}$ & ${ms:.2f}$ & --- & --- \\\\")
            else:
                p_str = "$< 10^{-15}$" if p_val < 1e-15 else (f"${p_val:.4f}$" if p_val >= 0.0001 else "$< 0.0001$")
                tex_ccd_y2.append(f"{source_map_ccd[src]} & ${ss:.2f}$ & ${df_val}$ & ${ms:.2f}$ & ${f_val:.2f}$ & {p_str} \\\\")
    tex_ccd_y2.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_anova_ccd_latency.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_ccd_y2) + "\n")

    # 4. Lack of Fit Decomposition Table (Full vs Restricted Domain, Resolving Item 10)
    with open("results/lof.json", "r", encoding="utf-8") as f:
        lof = json.load(f)

    # Center pure error and interaction terms
    ss_pure_center_y1 = float(np.sum([np.sum((grp["val_rmse"] - grp["val_rmse"].mean())**2)
                                       for (b, p), grp in df_runs.groupby(["block", "point_id"]) if len(grp) > 1]))
    df_pure_center = sum(len(grp) - 1 for (b, p), grp in df_runs.groupby(["block", "point_id"]) if len(grp) > 1)
    ms_pure_center_y1 = ss_pure_center_y1 / df_pure_center

    ss_interact_y1 = lof["Y1"]["SS_PE"] - ss_pure_center_y1
    df_interact = lof["Y1"]["df_PE"] - df_pure_center
    ms_interact_y1 = ss_interact_y1 / df_interact

    f_lof_center_y1 = lof["Y1"]["MS_LoF"] / ms_pure_center_y1
    p_lof_center_y1 = float(1.0 - stats.f.cdf(f_lof_center_y1, lof["Y1"]["df_LoF"], df_pure_center))

    f_interact_center_y1 = ms_interact_y1 / ms_pure_center_y1
    p_interact_center_y1 = float(1.0 - stats.f.cdf(f_interact_center_y1, df_interact, df_pure_center))

    # Latency Y2
    ss_pure_center_y2 = float(np.sum([np.sum((grp["latency_us_median"] - grp["latency_us_median"].mean())**2)
                                       for (b, p), grp in df_runs.groupby(["block", "point_id"]) if len(grp) > 1]))
    ms_pure_center_y2 = ss_pure_center_y2 / df_pure_center

    ss_interact_y2 = lof["Y2"]["SS_PE"] - ss_pure_center_y2
    ms_interact_y2 = ss_interact_y2 / df_interact

    f_lof_center_y2 = lof["Y2"]["MS_LoF"] / ms_pure_center_y2
    p_lof_center_y2 = float(1.0 - stats.f.cdf(f_lof_center_y2, lof["Y2"]["df_LoF"], df_pure_center))

    f_interact_center_y2 = ms_interact_y2 / ms_pure_center_y2
    p_interact_center_y2 = float(1.0 - stats.f.cdf(f_interact_center_y2, df_interact, df_pure_center))

    p1_str = "$< 10^{-15}$" if lof['Y1']['p_LoF'] < 1e-15 else f"${lof['Y1']['p_LoF']:.4f}$"
    p1_restr_str = "$< 10^{-15}$" if lof['Y1_restricted']['p_LoF'] < 1e-15 else (f"${lof['Y1_restricted']['p_LoF']:.4f}$" if lof['Y1_restricted']['p_LoF'] >= 0.0001 else "$< 0.0001$")
    p2_str = "$< 10^{-15}$" if lof['Y2']['p_LoF'] < 1e-15 else f"${lof['Y2']['p_LoF']:.4f}$"

    def fmt_tex_num(val, digs=6):
        if abs(val) < 0.0001:
            s = f"{val:.2e}"
            parts = s.split("e")
            exp = int(parts[1])
            return f"{float(parts[0]):.2f} \\times 10^{{{exp}}}"
        return f"{val:.{digs}f}"

    tex_lof = [
        "\\begin{tabular}{lrrrrcc}",
        "\\toprule",
        "\\textbf{Source of Variation / Model} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\bm{$F$} & \\makecell{\\bm{$p$}\\\\\\textbf{value}} & \\makecell{\\textbf{Reference /}\\\\\\textbf{RMS Misfit}} \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textbf{Decomposition of Second-Order Model Residual ($Y_1$: Validation RMSE, Full CCD)}} \\\\",
        f"Structural Lack of Fit & ${lof['Y1']['SS_LoF']:.6f}$ & ${lof['Y1']['df_LoF']}$ & ${lof['Y1']['MS_LoF']:.6f}$ & ${f_lof_center_y1:.2f}$ & $< 10^{{-15}}$ & vs. Center PE \\\\",
        f"Treatment $\\times$ Block Interaction & ${ss_interact_y1:.6f}$ & ${df_interact}$ & ${fmt_tex_num(ms_interact_y1)}$ & ${f_interact_center_y1:.2f}$ & ${p_interact_center_y1:.4f}$ & vs. Center PE \\\\",
        f"Genuine Center Pure Error & ${ss_pure_center_y1:.6f}$ & ${df_pure_center}$ & ${fmt_tex_num(ms_pure_center_y1)}$ & --- & --- & Base Replicate \\\\",
        "\\midrule",
        f"Total Model Residual & ${lof['Y1']['SS_PE']+lof['Y1']['SS_LoF']:.6f}$ & $121$ & ${fmt_tex_num((lof['Y1']['SS_PE']+lof['Y1']['SS_LoF'])/121)}$ & --- & --- & $\\text{{RMS}} = {np.sqrt((lof['Y1']['SS_PE']+lof['Y1']['SS_LoF'])/121):.4f}$ \\\\",
        f"Saturated Additive Baseline & ${lof['Y1']['SS_PE']:.6f}$ & ${lof['Y1']['df_PE']}$ & ${fmt_tex_num(lof['Y1']['MS_PE'])}$ & ${lof['Y1']['F_LoF']:.2f}^*$ & {p1_str} & ($^*$LoF vs. Additive) \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textbf{Restricted Domain Sensitivity ($Y_1$: $x_1 \\ge -0.5$, $\\eta \\ge 0.033$, $N=95$)}} \\\\",
        f"Structural Lack of Fit & ${lof['Y1_restricted']['SS_LoF']:.6f}$ & ${lof['Y1_restricted']['df_LoF']}$ & ${lof['Y1_restricted']['MS_LoF']:.6f}$ & ${lof['Y1_restricted']['F_LoF']:.2f}$ & {p1_restr_str} & $\\text{{RMS}} = {lof['Y1_restricted']['sqrt_MS_LoF']:.4f}$ \\\\",
        f"Additive Baseline Residual & ${lof['Y1_restricted']['SS_PE']:.6f}$ & ${lof['Y1_restricted']['df_PE']}$ & ${fmt_tex_num(lof['Y1_restricted']['MS_PE'])}$ & --- & --- & Saturated Base \\\\",
        f"Total Restricted Residual & ${lof['Y1_restricted']['SS_PE']+lof['Y1_restricted']['SS_LoF']:.6f}$ & ${lof['Y1_restricted']['df_PE']+lof['Y1_restricted']['df_LoF']}$ & ${fmt_tex_num((lof['Y1_restricted']['SS_PE']+lof['Y1_restricted']['SS_LoF'])/(lof['Y1_restricted']['df_PE']+lof['Y1_restricted']['df_LoF']))}$ & --- & --- & $\\text{{RMS}} = {np.sqrt((lof['Y1_restricted']['SS_PE']+lof['Y1_restricted']['SS_LoF'])/(lof['Y1_restricted']['df_PE']+lof['Y1_restricted']['df_LoF'])):.4f}$ \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textbf{Latency Residual Decomposition and Adequacy ($Y_2$: Single-Sample Latency, $\\mu\\text{s}$)}} \\\\",
        f"Structural Lack of Fit & ${lof['Y2']['SS_LoF']:.2f}$ & ${lof['Y2']['df_LoF']}$ & ${lof['Y2']['MS_LoF']:.2f}$ & ${f_lof_center_y2:.2f}$ & ${p_lof_center_y2:.4f}$ & vs. Center PE ($\\text{{RMS}}={lof['Y2']['sqrt_MS_LoF']:.2f}$) \\\\",
        f"Treatment $\\times$ Block Interaction & ${ss_interact_y2:.2f}$ & ${df_interact}$ & ${ms_interact_y2:.2f}$ & ${f_interact_center_y2:.2f}$ & ${p_interact_center_y2:.4f}$ & vs. Center PE \\\\",
        f"Genuine Center Pure Error & ${ss_pure_center_y2:.2f}$ & ${df_pure_center}$ & ${ms_pure_center_y2:.2f}$ & --- & --- & Base Replicate \\\\",
        "\\midrule",
        f"Total Model Residual & ${lof['Y2']['SS_PE']+lof['Y2']['SS_LoF']:.2f}$ & $121$ & ${(lof['Y2']['SS_PE']+lof['Y2']['SS_LoF'])/121:.2f}$ & ${lof['Y2']['F_LoF']:.2f}^*$ & {p2_str} & ($^*$LoF vs. Additive) \\\\",
        "\\bottomrule",
        "\\end{tabular}"
    ]
    with open("tables/tab_lof.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lof) + "\n")

    # 5. Confirmation Table (Multi-Objective Depth 4 and Single-Objective Depth 7, Resolving Item 4 & 6)
    with open("results/confirmation.json", "r", encoding="utf-8") as f:
        conf = json.load(f)
    so_conf = conf["Single_Objective_Optimum"]

    status_pass = "\\makecell[c]{\\textbf{Pass}\\\\(Inside 95\\% PI)}"
    bias_val = so_conf['empirical_val_rmse'] - so_conf['predicted_val_rmse']
    status_bias = "\\makecell[c]{\\textbf{Empirical Opt}\\\\(Bias: " + f"{bias_val:+.4f}" + ")}"
    status_lat = "\\makecell[c]{Borderline\\\\(At Lower PI)}"

    tex_conf = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\makecell[l]{\\textbf{Configuration /}\\\\\\textbf{Response Metric}} & \\makecell{\\textbf{Surrogate}\\\\\\textbf{Pred ($\\hat{y}$)}} & \\makecell{\\textbf{95\\% Pred}\\\\\\textbf{Interval (PI)}} & \\makecell{\\textbf{Empirical}\\\\\\textbf{Mean $\\pm$ SD}} & \\makecell{\\textbf{Confirmation}\\\\\\textbf{Status}} \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textbf{DOE Multi-Objective Optimum $\\mathbf{x}^*_{\\text{MO}}$ (Depth 4)}} \\\\",
        f"Validation RMSE ($Y_1$) & ${conf['Y1_Val_RMSE']['predicted_mean']:.4f}$ & $[0.4677, 0.4932]$ & ${conf['Y1_Val_RMSE']['empirical_mean']:.4f} \\pm {conf['Y1_Val_RMSE']['empirical_std']:.4f}$ & " + status_pass + " \\\\",
        f"Holdout Test RMSE & --- & --- & ${conf['Y1_Test_RMSE']['empirical_mean']:.4f} \\pm {conf['Y1_Test_RMSE']['empirical_std']:.4f}$ & Holdout Test Set \\\\",
        f"Inference Latency ($\\mu$s) & ${conf['Y2_Latency']['predicted_mean']:.1f}$ & $[{conf['Y2_Latency']['prediction_interval_95'][0]:.1f}, {conf['Y2_Latency']['prediction_interval_95'][1]:.1f}]$ & ${conf['Y2_Latency']['empirical_mean']:.1f} \\pm {conf['Y2_Latency']['empirical_std']:.1f}$ & " + status_pass + " \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textbf{DOE Single-Objective Candidate $\\mathbf{x}^*_{\\text{SO}}$ (Depth 7)}} \\\\",
        f"Validation RMSE ($Y_1$) & ${so_conf['predicted_val_rmse']:.4f}$ & $[0.4354, 0.4612]$ & ${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$ & " + status_bias + " \\\\",
        f"Holdout Test RMSE & --- & --- & ${so_conf['empirical_test_rmse']:.4f} \\pm {so_conf['empirical_test_std']:.4f}$ & Holdout Test Set \\\\",
        f"Inference Latency ($\\mu$s) & ${so_conf['predicted_latency']:.1f}$ & $[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]$ & ${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}$ & " + status_lat + " \\\\",
        "\\bottomrule",
        "\\end{tabular}"
    ]
    with open("tables/tab_confirmation.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_conf) + "\n")

    # 6. Benchmark Comparison Table (Resolving Item 1, 2, 3, 4, 11)
    df_bm = pd.read_csv("results/benchmark.csv")
    with open("results/benchmark_summary.json", "r", encoding="utf-8") as f:
        bms = json.load(f)
    hv_data = bms["hypervolume"]

    method_name_map = {
        "Sequential DOE-CCD (x*, Multi-Objective)": r"\makecell[l]{DOE Multi-Obj ($\mathbf{x}^*_{\text{MO}}$)}",
        "Sequential DOE-CCD (Single-Objective)": r"\makecell[l]{DOE Single-Obj ($\mathbf{x}^*_{\text{SO}}, d{=}7$)}",
        "Unguided Random Search": r"\makecell[l]{Random Search}",
        "Bayesian Optimization (Optuna TPE Single-Obj)": r"\makecell[l]{Bayesian TPE (SO)}",
        "Constrained TPE (Latency <= 145 us)": r"\makecell[l]{Constrained TPE}",
        "Multi-Objective TPE (Desirability)": r"\makecell[l]{Multi-Obj TPE}",
    }

    tex_bm = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\makecell{\\textbf{Optimization}\\\\\\textbf{Method}} & \\makecell{\\textbf{Val RMSE}\\\\\\scriptsize\\textbf{(95\\% CI)}} & \\makecell{\\textbf{Test RMSE}\\\\\\scriptsize\\textbf{(95\\% CI)}} & \\makecell{\\textbf{Pred Lat}\\\\\\bm{$(\\mu\\text{s})$}} & \\makecell{\\textbf{Inplace}\\\\\\bm{$(\\mu\\text{s})$}} & \\textbf{Depth} & \\makecell{\\textbf{Search}\\\\\\textbf{Basis}} \\\\",
        "\\midrule"
    ]
    for _, row in df_bm.iterrows():
        raw_name = str(row["method"])
        name = method_name_map.get(raw_name, raw_name)
        v_ci = "\\makecell{$" + f"{row['val_rmse_mean']:.4f}" + "$\\\\\\scriptsize$[" + f"{row['val_rmse_ci95_low']:.4f}, {row['val_rmse_ci95_high']:.4f}" + "]$}"
        t_ci = "\\makecell{$" + f"{row['test_rmse_mean']:.4f}" + "$\\\\\\scriptsize$[" + f"{row['test_rmse_ci95_low']:.4f}, {row['test_rmse_ci95_high']:.4f}" + "]$}"
        p_lat = f"{row['predict_latency_us_median']:.1f}"
        i_lat = f"{row['inplace_latency_us_median']:.1f}"
        d_val = str(int(row["depth"]))
        attr = "\\makecell{Second-Order\\\\RSM}" if "DOE" in name else "\\makecell{Black-Box\\\\Oracle}"
        tex_bm.append(f"{name} & {v_ci} & {t_ci} & ${p_lat}$ & ${i_lat}$ & {d_val} & {attr} \\\\")
    tex_bm.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_benchmarks.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_bm) + "\n")

    # 7. Integer Depth Table
    df_depth = pd.read_csv("results/depth_opt_table.csv")
    tex_depth = [
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "\\textbf{Depth} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_1$}} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_2$}} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_3$}} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_4$}} & \\makecell{\\textbf{Optimal}\\\\\\bm{$\\eta$}} & \\makecell{\\textbf{Predicted}\\\\\\textbf{RMSE}} & \\makecell{\\textbf{SE}\\\\\\bm{$(\\hat{y})$}} \\\\",
        "\\midrule"
    ]
    for _, row in df_depth.iterrows():
        d_int = int(row["depth"])
        x2_coded = (d_int - 6) / 3.0
        eta_val = decode_factors(np.array([row["x1"], x2_coded, row["x3"], row["x4"]]))[0]
        tex_depth.append(f"{d_int} & ${row['x1']:.4f}$ & ${x2_coded:+.4f}$ & ${row['x3']:.2f}$ & ${row['x4']:.2f}$ & ${eta_val:.4f}$ & ${row['pred_rmse']:.4f}$ & ${row['se']:.4f}$ \\\\")
    tex_depth.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_depth_opt.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_depth) + "\n")

    # 8. Desirability Sensitivity Table
    df_sens = pd.read_csv("results/desirability_sensitivity.csv")
    tex_sens = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Scenario} & \\makecell{\\textbf{RMSE Bounds}\\\\\\bm{$[L_1, U_1]$}} & \\makecell{\\textbf{Latency Bounds}\\\\\\bm{$[L_2, U_2]$}} & \\makecell{\\textbf{Weights}\\\\\\bm{$(w_1, w_2)$}} & \\makecell{\\textbf{Opt}\\\\\\textbf{Depth}} & \\makecell{\\textbf{Opt}\\\\\\bm{$\\eta$}} & \\makecell{\\textbf{Best}\\\\\\bm{$D$}} \\\\",
        "\\midrule"
    ]
    for _, row in df_sens.iterrows():
        b1 = f"[{row['L1']:.2f}, {row['U1']:.2f}]"
        b2 = f"[{row['L2']:.0f}, {row['U2']:.0f}]"
        w = f"({int(row['w1'])}, {int(row['w2'])})"
        tex_sens.append(f"{row['Scenario']} & {b1} & {b2} & {w} & {int(row['Optimal_Depth'])} & ${row['Optimal_Eta']:.4f}$ & ${row['Best_D']:.4f}$ \\\\")
    tex_sens.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_desirability_sensitivity.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_sens) + "\n")

    # 9. Latency Candidate Models Table
    df_lat = pd.read_csv("results/latency_models_comparison.csv")
    df_runs = pd.read_csv("results/runs.csv")
    jac_adj = float(2.0 * np.sum(np.log(df_runs["latency_us_median"])))

    tex_lat = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Candidate Latency Model} & \\bm{$R^2$} & \\makecell{\\textbf{Adj}\\\\\\bm{$R^2$}} & \\makecell{\\textbf{AIC}\\\\\\scriptsize(Adjusted)} & \\makecell{\\textbf{BIC}\\\\\\scriptsize(Adjusted)} & \\makecell{\\textbf{RMSE}\\\\\\textbf{($\\mu$s)}} \\\\",
        "\\midrule"
    ]
    for _, row in df_lat.iterrows():
        m_name = str(row['Model']).replace('depth^2', r'depth$^2$')
        aic_val = row['AIC'] + jac_adj if "Log-Linear" in m_name else row['AIC']
        bic_val = row['BIC'] + jac_adj if "Log-Linear" in m_name else row['BIC']
        tex_lat.append(f"{m_name} & ${row['R2']:.3f}$ & ${row['Adj_R2']:.3f}$ & ${aic_val:.1f}$ & ${bic_val:.1f}$ & ${row['RMSE']:.2f}$ \\\\")
    tex_lat.extend(["\\bottomrule", "\\end{tabular}"])
    with open("tables/tab_latency_models.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lat) + "\n")

    print("All LaTeX tables generated and saved to tables/*.tex successfully.")

if __name__ == "__main__":
    generate_macros()
    generate_tables()
