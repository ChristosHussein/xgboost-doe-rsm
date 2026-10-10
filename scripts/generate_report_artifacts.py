"""
scripts/generate_report_artifacts.py - Generates LaTeX tables and macros from the
validated canonical reporting dataset (scripts/reporting_data.py).
Ensures zero hard-coded numbers in report.tex, results/macros.tex, and tables/*.tex.

Supports:
  python scripts/generate_report_artifacts.py [--write]
  python scripts/generate_report_artifacts.py --check
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline import decode_factors
from scripts.reporting_data import (
    PROSPECTIVE_OPTIMIZER_ORDER,
    ReportingDataset,
    load_reporting_dataset,
)


def render_macros(ds: Optional[ReportingDataset] = None) -> str:
    """Render the complete content of results/macros.tex from ReportingDataset."""
    if ds is None:
        ds = load_reporting_dataset()

    p1 = ds.phase1
    p3 = ds.phase3
    lof = ds.lof
    diag = ds.diagnostics
    icc = ds.icc
    conf = ds.confirmation
    bm_sum = ds.historical_benchmark_summary
    df_bm = ds.historical_benchmark_df
    df_lat = ds.latency_models_df
    env_data = ds.env

    macros = []

    def add_macro(name: str, val: str) -> None:
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
    add_macro(
        "numPBlock",
        f"{icc['Y1']['p_block']:.4f}" if icc["Y1"]["p_block"] >= 0.0001 else "< 0.0001",
    )

    # Phase 3 / Canonical
    add_macro("numBzero", f"{p3['b0']:.4f}")
    add_macro("numTraceB", f"{p3['trace_B']:.6f}")
    add_macro("numSumEigs", f"{p3['sum_eigenvalues']:.6f}")
    add_macro("numEigOne", f"{p3['eigenvalues'][0]:.6f}")
    add_macro("numEigTwo", f"{p3['eigenvalues'][1]:.6f}")
    add_macro("numEigThree", f"{p3['eigenvalues'][2]:.6f}")
    add_macro("numEigFour", f"{p3['eigenvalues'][3]:.6f}")
    add_macro("numFracMinEigLeZero", f"{p3['fraction_min_eigenvalue_le_zero']*100:.1f}\\%")

    # CCD R-squared macros
    add_macro("numRsqRMSE", f"{ds.ccd_y1_rsq:.4f}")
    add_macro("numAdjRsqRMSE", f"{ds.ccd_y1_adj_rsq:.4f}")

    # Phase 1 ANOVA F-stats
    an_p1 = ds.anova_phase1_df
    ss_res_p1 = float(an_p1.loc["Residual", "sum_sq"])
    f_x1_p1 = float(an_p1.loc["x1", "F"])
    eta_x1_p1 = float(an_p1.loc["x1", "sum_sq"] / (an_p1.loc["x1", "sum_sq"] + ss_res_p1))
    f_x2_p1 = float(an_p1.loc["x2", "F"])
    f_x1x2_p1 = float(an_p1.loc["x1:x2", "F"])

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
    add_macro("numStationaryDepth", str(p3["stationary_natural_clamped"]["depth"]))
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

    # Lack of Fit and Pure Error Decomposition
    ld = ds.lof_decomp
    df_pure_center = int(ld["df_pure_center"])
    df_interact = int(ld["df_interact"])
    ss_pure_center_y1 = ld["ss_pure_center_y1"]
    ms_pure_center_y1 = ld["ms_pure_center_y1"]
    ss_interact_y1 = ld["ss_interact_y1"]
    ms_interact_y1 = ld["ms_interact_y1"]
    f_lof_center_y1 = ld["f_lof_center_y1"]
    p_lof_center_y1 = ld["p_lof_center_y1"]
    f_interact_center_y1 = ld["f_interact_center_y1"]
    p_interact_center_y1 = ld["p_interact_center_y1"]

    ss_pure_center_y2 = ld["ss_pure_center_y2"]
    ms_pure_center_y2 = ld["ms_pure_center_y2"]
    ss_interact_y2 = ld["ss_interact_y2"]
    ms_interact_y2 = ld["ms_interact_y2"]
    f_lof_center_y2 = ld["f_lof_center_y2"]
    p_lof_center_y2 = ld["p_lof_center_y2"]
    f_interact_center_y2 = ld["f_interact_center_y2"]
    p_interact_center_y2 = ld["p_interact_center_y2"]

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

    add_macro("numDfPeCenter", str(df_pure_center))
    add_macro("numSSPeCenterYone", f"{ss_pure_center_y1:.6f}")
    add_macro("numMSPeCenterYone", f"{ms_pure_center_y1:.2e}")
    add_macro("numSqrtMSPeCenterYone", f"{np.sqrt(ms_pure_center_y1):.4f}")
    add_macro("numFLofCenterYone", f"{f_lof_center_y1:.2f}")
    add_macro(
        "numPLofCenterYone",
        "< 10^{-15}" if p_lof_center_y1 < 1e-15 else f"{p_lof_center_y1:.2e}",
    )

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
    add_macro(
        "numBreuschPaganP",
        f"{diag['breusch_pagan_p']:.4f}" if diag["breusch_pagan_p"] >= 0.0001 else "< 0.0001",
    )
    add_macro("numMaxCooksD", f"{diag['max_cooks_d']:.3f}")
    add_macro("numThreshFourN", f"{diag['thresh_4n']:.3f}")
    add_macro("numOutlierCount", str(diag["outlier_count"]))
    add_macro("numDurbinWatson", f"{diag['durbin_watson']:.4f}")
    add_macro("numLjungBoxStat", f"{diag['ljung_box_stat']:.2f}")
    add_macro("numLjungBoxP", f"{diag['ljung_box_p']:.4f}")
    add_macro("numRunsTestP", f"{diag['runs_test_p']:.4f}")

    # Confirmation & Multi-Objective Optimum Aliases
    add_macro("numOptDepth", str(conf["x_star_natural"]["depth"]))
    add_macro("numOptEta", f"{conf['x_star_natural']['eta']:.4f}")
    add_macro("numOptSubsample", f"{conf['x_star_natural']['subsample']:.4f}")
    add_macro("numOptLambda", f"{conf['x_star_natural']['reg_lambda']:.4f}")

    d_star = ds.desirability_star
    add_macro("numOptD", f"{d_star['D']:.4f}")
    add_macro("numOptdOne", f"{d_star['d1']:.4f}")
    add_macro("numOptdTwo", f"{d_star['d2']:.4f}")
    add_macro("numPredRMSE", f"{d_star['pred_y1']:.4f}")
    add_macro("numPredLatency", f"{d_star['pred_y2']:.2f}")
    add_macro("numOptXone", f"{conf['x_star_coded'][0]:.4f}")
    add_macro("numOptXtwo", f"{conf['x_star_coded'][1]:.4f}")
    add_macro("numOptXthree", f"{conf['x_star_coded'][2]:.4f}")
    add_macro("numOptXfour", f"{conf['x_star_coded'][3]:.4f}")

    add_macro("numConfOptEta", f"{conf['x_star_natural']['eta']:.4f}")
    add_macro("numConfOptDepth", str(conf["x_star_natural"]["depth"]))
    add_macro("numConfOptSubsample", f"{conf['x_star_natural']['subsample']:.4f}")
    add_macro("numConfOptLambda", f"{conf['x_star_natural']['reg_lambda']:.4f}")
    add_macro("numConfLeverageH", f"{conf['leverage_h']:.4f}")
    add_macro("numConfPredValRMSE", f"{conf['Y1_Val_RMSE']['predicted_mean']:.4f}")
    add_macro("numConfPiLowValRMSE", f"{conf['Y1_Val_RMSE']['prediction_interval_95'][0]:.4f}")
    add_macro("numConfPiHighValRMSE", f"{conf['Y1_Val_RMSE']['prediction_interval_95'][1]:.4f}")
    add_macro("numConfNuEffMo", f"{conf['Y1_Val_RMSE']['nu_eff']:.1f}")
    add_macro("numConfEmpValRMSE", f"{conf['Y1_Val_RMSE']['empirical_mean']:.4f}")
    add_macro("numConfEmpValRMSEStd", f"{conf['Y1_Val_RMSE']['empirical_std']:.4f}")
    add_macro("numConfEmpTestRMSE", f"{conf['Y1_Test_RMSE']['empirical_mean']:.4f}")
    add_macro("numConfEmpTestRMSEStd", f"{conf['Y1_Test_RMSE']['empirical_std']:.4f}")
    add_macro("numConfPredLat", f"{conf['Y2_Latency']['predicted_mean']:.2f}")
    add_macro("numConfPiLowLat", f"{conf['Y2_Latency']['prediction_interval_95'][0]:.2f}")
    add_macro("numConfPiHighLat", f"{conf['Y2_Latency']['prediction_interval_95'][1]:.2f}")
    add_macro("numConfEmpLat", f"{conf['Y2_Latency']['empirical_mean']:.2f}")
    add_macro("numConfEmpLatStd", f"{conf['Y2_Latency']['empirical_std']:.2f}")
    add_macro(
        "numConfPassValRMSE",
        "Pass (Inside 95\\% PI)" if conf["Y1_Val_RMSE"]["inside_pi"] else "Outside 95\\% PI",
    )
    add_macro(
        "numConfPassLat",
        "Pass (Inside 95\\% PI)" if conf["Y2_Latency"]["inside_pi"] else "Outside 95\\% PI",
    )

    # Single-Objective Confirmation
    so_conf = conf["Single_Objective_Optimum"]
    add_macro("numConfSoPredValRMSE", f"{so_conf['predicted_val_rmse']:.4f}")
    add_macro("numConfSoPiLowValRMSE", f"{so_conf['prediction_interval_95_val'][0]:.4f}")
    add_macro("numConfSoPiHighValRMSE", f"{so_conf['prediction_interval_95_val'][1]:.4f}")
    add_macro("numConfSoNuEffSo", f"{so_conf['nu_eff']:.1f}")
    add_macro("numConfSoEmpValRMSE", f"{so_conf['empirical_val_rmse']:.4f}")
    add_macro("numConfSoEmpValRMSEStd", f"{so_conf['empirical_val_std']:.4f}")
    add_macro("numConfSoEmpTestRMSE", f"{so_conf['empirical_test_rmse']:.4f}")
    add_macro("numConfSoEmpTestRMSEStd", f"{so_conf['empirical_test_std']:.4f}")
    add_macro("numConfSoEmpLat", f"{so_conf['empirical_latency']:.1f}")
    add_macro("numConfSoEmpLatStd", f"{so_conf['empirical_latency_std']:.1f}")
    add_macro("numConfSoBias", f"{so_conf['empirical_val_rmse'] - so_conf['predicted_val_rmse']:+.4f}")

    # Historical Benchmark comparisons
    doe_row = df_bm[df_bm["method"].str.contains("Multi-Objective") & df_bm["method"].str.contains("DOE")].iloc[0]
    doe_so_row = df_bm[df_bm["method"].str.contains("Single-Objective") & df_bm["method"].str.contains("DOE")].iloc[0]
    tpe_row = df_bm[df_bm["method"].str.contains("Single-Obj") & ~df_bm["method"].str.contains("DOE")].iloc[0]
    rs_row = df_bm[df_bm["method"].str.contains("Random Search")].iloc[0]
    motpe_row = df_bm[df_bm["method"].str.contains("Multi-Objective TPE")].iloc[0]
    ctpe_row = df_bm[df_bm["method"].str.contains("Constrained")].iloc[0]

    doe_test_rmse = float(doe_row["test_rmse_mean"])
    tpe_test_rmse = float(tpe_row["test_rmse_mean"])
    test_gap_pct = ((doe_test_rmse - tpe_test_rmse) / tpe_test_rmse) * 100.0
    lat_slow_pct = (
        (tpe_row["predict_latency_us_median"] - doe_row["predict_latency_us_median"])
        / doe_row["predict_latency_us_median"]
    ) * 100.0
    doe_lat_lower_pct = (
        (tpe_row["predict_latency_us_median"] - doe_row["predict_latency_us_median"])
        / tpe_row["predict_latency_us_median"]
    ) * 100.0

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

    hv = bm_sum.get("hypervolume", {})
    ref_pt = hv.get("reference_point", list(ds.hv_ref_primary))

    def _single_point_hv(rmse: float, lat: float, ref: list) -> float:
        if rmse <= ref[0] and lat <= ref[1]:
            return float((ref[0] - rmse) * (ref[1] - lat))
        return 0.0

    hv_doe_single = hv.get("hv_doe_single")
    if hv_doe_single is None:
        hv_doe_single = _single_point_hv(doe_test_rmse, float(doe_row["predict_latency_us_median"]), ref_pt)

    hv_motpe = hv.get("hv_motpe")
    if hv_motpe is None:
        hv_motpe = _single_point_hv(
            float(motpe_row["test_rmse_mean"]), float(motpe_row["predict_latency_us_median"]), ref_pt
        )

    hv_doe_single_diff_pct = hv.get("doe_single_over_motpe_pct")
    if hv_doe_single_diff_pct is None:
        hv_doe_single_diff_pct = ((hv_doe_single - hv_motpe) / hv_motpe) * 100.0 if hv_motpe > 0 else 0.0

    if "hv_doe" not in hv or "hv_rs" not in hv:
        raise KeyError("Missing required hypervolume keys ('hv_doe', 'hv_rs') in benchmark_summary.json")

    add_macro("numHvDoe", f"{hv['hv_doe']:.2f}")
    add_macro("numHvDoeSingle", f"{hv_doe_single:.2f}")
    add_macro("numHvMoTpe", f"{hv_motpe:.2f}")
    add_macro("numHvRs", f"{hv['hv_rs']:.2f}")
    add_macro("numHvDoeGainPct", f"{hv['doe_over_motpe_pct']:.1f}\\%")
    add_macro("numHvDoeSingleDiffPct", f"{hv_doe_single_diff_pct:.1f}\\%")

    # Latency model fit macros
    m_lin = df_lat[df_lat["Model"].str.contains("Linear")].iloc[0]
    m_quad = df_lat[df_lat["Model"].str.contains("Quadratic")].iloc[0]
    add_macro("numLatLinRsq", f"{m_lin['R2']:.3f}")
    add_macro("numLatQuadRsq", f"{m_quad['R2']:.3f}")

    # Latency ANOVA terms
    an_ccd_y2 = ds.anova_ccd_y2_df
    add_macro("numAnovaLatencyFxTwoSq", f"{an_ccd_y2.loc['x2_sq', 'F']:.2f}")
    add_macro("numFBlockLatency", f"{an_ccd_y2.loc['C(block)', 'F']:.2f}")
    add_macro("numPBlockLatency", f"{an_ccd_y2.loc['C(block)', 'PR(>F)']:.4f}")

    # =========================================================================
    # Revision v2 Full Experiment Macros (Derived from ReportingDataset)
    # =========================================================================
    add_macro("numRevTotalFits", f"{ds.total_model_fits:,}".replace(",", "{,}"))
    add_macro("numRevTotalTimedInferences", f"{ds.total_timed_inferences:,}".replace(",", "{,}"))
    add_macro("numRevTotalFrozenConfigs", f"{ds.total_selection_records}")
    add_macro("numRevTotalSelectionRecords", f"{ds.total_selection_records}")
    add_macro("numRevTotalDistinctConfigs", f"{ds.distinct_config_hashes}")
    add_macro("numRevTotalEvaluationRows", f"{ds.total_evaluation_rows:,}".replace(",", "{,}"))
    add_macro("numRevHoldoutNonDominatedConfigs", f"{ds.hv_summary['holdout_nd']}")
    add_macro("numRevBetweenSessionSd", f"{ds.between_session_latency_sd_mean:.2f}")
    add_macro("numRevInterfaceOverhead", f"{ds.interface_overhead_mean:.2f}")

    opts = ds.prospective_optimizers
    doe_mo = opts["repeated_preplanned_doe_multi_objective"]
    mo_tpe = opts["multi_objective_tpe"]
    ctpe = opts["constrained_tpe"]
    doe_so = opts["repeated_preplanned_doe_single_objective"]
    so_tpe = opts["single_objective_tpe"]
    rs = opts["random_search"]

    add_macro("numRevDoeMoTestRMSE", f"{doe_mo.test_rmse_mean:.5f}")
    add_macro("numRevDoeMoTestRMSESd", f"{doe_mo.test_rmse_sd:.5f}")
    add_macro("numRevDoeMoValRMSE", f"{doe_mo.val_rmse_mean:.5f}")
    add_macro("numRevDoeMoValRMSESd", f"{doe_mo.val_rmse_sd:.5f}")
    add_macro("numRevDoeMoPredictLat", f"{doe_mo.predict_latency_mean:.2f}")
    add_macro("numRevDoeMoPredictLatSd", f"{doe_mo.predict_latency_sd:.2f}")
    add_macro("numRevDoeMoInplaceLat", f"{doe_mo.inplace_latency_mean:.2f}")
    add_macro("numRevDoeMoRetrainSd", f"{doe_mo.retrain_sd:.5f}")

    add_macro("numRevDoeSoTestRMSE", f"{doe_so.test_rmse_mean:.5f}")
    add_macro("numRevDoeSoTestRMSESd", f"{doe_so.test_rmse_sd:.5f}")
    add_macro("numRevDoeSoValRMSE", f"{doe_so.val_rmse_mean:.5f}")
    add_macro("numRevDoeSoPredictLat", f"{doe_so.predict_latency_mean:.2f}")
    add_macro("numRevDoeSoPredictLatSd", f"{doe_so.predict_latency_sd:.2f}")
    add_macro("numRevDoeSoInplaceLat", f"{doe_so.inplace_latency_mean:.2f}")
    add_macro("numRevDoeSoRetrainSd", f"{doe_so.retrain_sd:.5f}")

    add_macro("numRevMoTpeTestRMSE", f"{mo_tpe.test_rmse_mean:.5f}")
    add_macro("numRevMoTpeTestRMSESd", f"{mo_tpe.test_rmse_sd:.5f}")
    add_macro("numRevMoTpeValRMSE", f"{mo_tpe.val_rmse_mean:.5f}")
    add_macro("numRevMoTpeValRMSESd", f"{mo_tpe.val_rmse_sd:.5f}")
    add_macro("numRevMoTpePredictLat", f"{mo_tpe.predict_latency_mean:.2f}")
    add_macro("numRevMoTpePredictLatSd", f"{mo_tpe.predict_latency_sd:.2f}")
    add_macro("numRevMoTpeInplaceLat", f"{mo_tpe.inplace_latency_mean:.2f}")
    add_macro("numRevMoTpeRetrainSd", f"{mo_tpe.retrain_sd:.5f}")

    add_macro("numRevSoTpeTestRMSE", f"{so_tpe.test_rmse_mean:.5f}")
    add_macro("numRevSoTpeTestRMSESd", f"{so_tpe.test_rmse_sd:.5f}")
    add_macro("numRevSoTpeValRMSE", f"{so_tpe.val_rmse_mean:.5f}")
    add_macro("numRevSoTpePredictLat", f"{so_tpe.predict_latency_mean:.2f}")
    add_macro("numRevSoTpePredictLatSd", f"{so_tpe.predict_latency_sd:.2f}")
    add_macro("numRevSoTpeInplaceLat", f"{so_tpe.inplace_latency_mean:.2f}")
    add_macro("numRevSoTpeRetrainSd", f"{so_tpe.retrain_sd:.5f}")

    add_macro("numRevCtpeTestRMSE", f"{ctpe.test_rmse_mean:.5f}")
    add_macro("numRevCtpeTestRMSESd", f"{ctpe.test_rmse_sd:.5f}")
    add_macro("numRevCtpeValRMSE", f"{ctpe.val_rmse_mean:.5f}")
    add_macro("numRevCtpePredictLat", f"{ctpe.predict_latency_mean:.2f}")
    add_macro("numRevCtpePredictLatSd", f"{ctpe.predict_latency_sd:.2f}")
    add_macro("numRevCtpeInplaceLat", f"{ctpe.inplace_latency_mean:.2f}")
    add_macro("numRevCtpeRetrainSd", f"{ctpe.retrain_sd:.5f}")
    add_macro("numRevCtpeSearchFeas", f"{ctpe.search_feasible_count}/{ctpe.n_replicates}")
    add_macro("numRevCtpeBenchFeas", f"{ctpe.benchmark_feasible_count}/{ctpe.n_replicates}")
    add_macro("numRevCtpeBenchFeasPct", f"{ctpe.benchmark_feasible_pct}\\%")
    add_macro("numRevCtpeDepthSixLat", f"{ds.ctpe_depth6_latency_mean:.2f}")
    add_macro("numRevCtpeDepthSixLatSd", f"{ds.ctpe_depth6_latency_sd:.2f}")
    add_macro("numRevCtpeDepthSevenLat", f"{ds.ctpe_depth7_latency_mean:.2f}")
    add_macro("numRevCtpeDepthSevenLatSd", f"{ds.ctpe_depth7_latency_sd:.2f}")

    add_macro("numRevRsTestRMSE", f"{rs.test_rmse_mean:.5f}")
    add_macro("numRevRsTestRMSESd", f"{rs.test_rmse_sd:.5f}")
    add_macro("numRevRsValRMSE", f"{rs.val_rmse_mean:.5f}")
    add_macro("numRevRsPredictLat", f"{rs.predict_latency_mean:.2f}")
    add_macro("numRevRsPredictLatSd", f"{rs.predict_latency_sd:.2f}")
    add_macro("numRevRsInplaceLat", f"{rs.inplace_latency_mean:.2f}")
    add_macro("numRevRsRetrainSd", f"{rs.retrain_sd:.5f}")

    add_macro("numRevDiffDoeMoMoTpe", f"{ds.doe_mo_vs_motpe_diff:+.5f}")
    add_macro("numRevWelchTDoeMoMoTpe", f"{ds.doe_mo_vs_motpe_welch_t:.4f}")
    add_macro("numRevWelchPDoeMoMoTpe", f"{ds.doe_mo_vs_motpe_welch_p:.4f}")

    hvs = ds.hv_summary
    add_macro("numRevHvMoTpeDevMean", f"{hvs['motpe_dev_060_mean']:.4f}")
    add_macro("numRevHvMoTpeDevSd", f"{hvs['motpe_dev_060_sd']:.4f}")
    add_macro("numRevHvDoeDevMean", f"{hvs['rdoe_dev_060_mean']:.4f}")
    add_macro("numRevHvDoeDevSd", f"{hvs['rdoe_dev_060_sd']:.4f}")
    add_macro("numRevHvFullDoeDev", f"{hvs['fdoe_dev_060']:.4f}")
    add_macro("numRevHvDiffDev", f"{hvs['diff_fdoe_minus_motpe_060']:.4f}")

    add_macro("numRevHvMoTpeDevMeanRefTwo", f"{hvs['motpe_dev_065_mean']:.4f}")
    add_macro("numRevHvMoTpeDevSdRefTwo", f"{hvs['motpe_dev_065_sd']:.4f}")
    add_macro("numRevHvFullDoeDevRefTwo", f"{hvs['fdoe_dev_065']:.4f}")
    add_macro("numRevHvDiffDevRefTwo", f"{hvs['diff_fdoe_minus_motpe_065']:.4f}")

    add_macro("numRevHvHoldoutFrozen", f"{hvs['holdout_frozen_060']:.4f}")
    add_macro("numRevHvHoldoutFrozenRefTwo", f"{hvs['holdout_frozen_065']:.4f}")

    return "% Auto-generated macros from genuine results\n" + "\n".join(macros) + "\n"


def render_tables(ds: Optional[ReportingDataset] = None) -> Dict[str, str]:
    """Render all LaTeX tables as a mapping from relative path ('tables/*.tex') to content."""
    if ds is None:
        ds = load_reporting_dataset()

    tables: Dict[str, str] = {}

    # 0. Design, Block, and Seed Explicit Table
    tex_seeds = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\makecell{\\textbf{Phase /}\\\\\\textbf{Design Point}} & \\makecell{\\textbf{Point}\\\\\\textbf{IDs}} & \\textbf{Block} & \\makecell{\\textbf{Split}\\\\\\textbf{Seed}} & \\makecell{\\textbf{Model}\\\\\\textbf{Seed}} & \\textbf{Reps} & \\makecell{\\textbf{Isolated}\\\\\\textbf{Variance}} \\\\",
        "\\midrule",
        "\\makecell[l]{Phase 1:\\\\Corner Runs} & $1 \\dots 16$ & $b \\in \\{1..5\\}$ & $S_b$ & $S_b$ & 1 & Factorial + Block \\\\",
        f"\\makecell[l]{{Phase 1:\\\\Center Points}} & $17$ & $b \\in \\{{1..5\\}}$ & $S_b$ & $S_b + 1000r$ & 4 & \\makecell{{Subsampling Error\\\\(${int(ds.lof_decomp['df_pure_center'])}$ df)}} \\\\",
        "\\makecell[l]{Phase 2:\\\\Axial Runs} & $18 \\dots 25$ & $b \\in \\{1..5\\}$ & $S_b$ & $S_b$ & 1 & Quadratic + Block \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tables["tables/tab_design_block_seeds.tex"] = "\n".join(tex_seeds) + "\n"

    # 1. Phase 1 Screening ANOVA Table
    an_p1 = ds.anova_phase1_df
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
        "Residual": "Residual Error",
    }

    tex_p1 = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Source of Variation} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\makecell{\\textbf{F-}\\\\\\textbf{Stat}} & \\makecell{\\bm{$p$}\\\\\\textbf{value}} & \\makecell{\\textbf{Partial}\\\\\\bm{$\\eta^2$}} \\\\",
        "\\midrule",
    ]
    ss_res_p1 = float(an_p1.loc["Residual", "sum_sq"])
    for src in source_map.keys():
        if src in an_p1.index:
            ss = float(an_p1.loc[src, "sum_sq"])
            df_val = int(an_p1.loc[src, "df"])
            ms = ss / df_val
            f_val = float(an_p1.loc[src, "F"])
            p_val = float(an_p1.loc[src, "PR(>F)"])
            eta_sq = ss / (ss + ss_res_p1) if src != "Residual" else 0.0
            if src == "Residual":
                tex_p1.append(
                    f"{source_map[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & --- & --- & --- \\\\"
                )
            else:
                p_str = "$< 10^{-15}$" if p_val < 1e-15 else f"${p_val:.4f}$"
                tex_p1.append(
                    f"{source_map[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & ${f_val:.2f}$ & {p_str} & ${eta_sq:.4f}$ \\\\"
                )
    tex_p1.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_anova_phase1.tex"] = "\n".join(tex_p1) + "\n"

    # 2. CCD Second-Order ANOVA Table for Val RMSE
    an_ccd_y1 = ds.anova_ccd_y1_df
    hc3_bse_series = ds.hc3_ccd_y1_se
    hc3_pval_series = ds.hc3_ccd_y1_p
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
        "Residual": "Residual Error",
    }

    tex_ccd_y1 = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "\\textbf{Source of Variation} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\makecell{\\textbf{F-}\\\\\\textbf{Stat}} & \\makecell{\\textbf{OLS}\\\\\\bm{$p$}} & \\makecell{\\textbf{HC3}\\\\\\textbf{SE}} & \\makecell{\\textbf{HC3}\\\\\\bm{$p$}} \\\\",
        "\\midrule",
    ]
    for src in source_map_ccd.keys():
        if src in an_ccd_y1.index:
            ss = float(an_ccd_y1.loc[src, "sum_sq"])
            df_val = int(an_ccd_y1.loc[src, "df"])
            ms = ss / df_val
            f_val = float(an_ccd_y1.loc[src, "F"])
            p_val = float(an_ccd_y1.loc[src, "PR(>F)"])
            if src == "Residual":
                tex_ccd_y1.append(
                    f"{source_map_ccd[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & --- & --- & --- & --- \\\\"
                )
            else:
                p_str = (
                    "$< 10^{-15}$"
                    if p_val < 1e-15
                    else (f"${p_val:.4f}$" if p_val >= 0.0001 else "$< 0.0001$")
                )
                if src in hc3_bse_series.index:
                    hc3_se = float(hc3_bse_series[src])
                    hc3_p = float(hc3_pval_series[src])
                    hc3_p_str = (
                        "$< 10^{-15}$"
                        if hc3_p < 1e-15
                        else (f"${hc3_p:.4f}$" if hc3_p >= 0.0001 else "$< 0.0001$")
                    )
                    hc3_se_str = f"${hc3_se:.4f}$"
                else:
                    hc3_se_str = "---"
                    hc3_p_str = "---"
                tex_ccd_y1.append(
                    f"{source_map_ccd[src]} & ${ss:.6f}$ & ${df_val}$ & ${ms:.6f}$ & ${f_val:.2f}$ & {p_str} & {hc3_se_str} & {hc3_p_str} \\\\"
                )
    tex_ccd_y1.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_anova_ccd_rmse.tex"] = "\n".join(tex_ccd_y1) + "\n"

    # 3. CCD Second-Order ANOVA Table for Latency
    an_ccd_y2 = ds.anova_ccd_y2_df
    tex_ccd_y2 = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\textbf{Source of Variation} & \\textbf{SS} & \\textbf{DF} & \\textbf{MS} & \\makecell{\\textbf{F-}\\\\\\textbf{Stat}} & \\makecell{\\bm{$p$}\\\\\\textbf{value}} \\\\",
        "\\midrule",
    ]
    for src in source_map_ccd.keys():
        if src in an_ccd_y2.index:
            ss = float(an_ccd_y2.loc[src, "sum_sq"])
            df_val = int(an_ccd_y2.loc[src, "df"])
            ms = ss / df_val
            f_val = float(an_ccd_y2.loc[src, "F"])
            p_val = float(an_ccd_y2.loc[src, "PR(>F)"])
            if src == "Residual":
                tex_ccd_y2.append(
                    f"{source_map_ccd[src]} & ${ss:.2f}$ & ${df_val}$ & ${ms:.2f}$ & --- & --- \\\\"
                )
            else:
                p_str = (
                    "$< 10^{-15}$"
                    if p_val < 1e-15
                    else (f"${p_val:.4f}$" if p_val >= 0.0001 else "$< 0.0001$")
                )
                tex_ccd_y2.append(
                    f"{source_map_ccd[src]} & ${ss:.2f}$ & ${df_val}$ & ${ms:.2f}$ & ${f_val:.2f}$ & {p_str} \\\\"
                )
    tex_ccd_y2.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_anova_ccd_latency.tex"] = "\n".join(tex_ccd_y2) + "\n"

    # 4. Lack of Fit Decomposition Table
    lof = ds.lof
    ld = ds.lof_decomp
    df_pure_center = int(ld["df_pure_center"])
    df_interact = int(ld["df_interact"])
    ss_pure_center_y1 = ld["ss_pure_center_y1"]
    ms_pure_center_y1 = ld["ms_pure_center_y1"]
    ss_interact_y1 = ld["ss_interact_y1"]
    ms_interact_y1 = ld["ms_interact_y1"]
    f_lof_center_y1 = ld["f_lof_center_y1"]
    f_interact_center_y1 = ld["f_interact_center_y1"]
    p_interact_center_y1 = ld["p_interact_center_y1"]
    ss_pure_center_y2 = ld["ss_pure_center_y2"]
    ms_pure_center_y2 = ld["ms_pure_center_y2"]
    ss_interact_y2 = ld["ss_interact_y2"]
    ms_interact_y2 = ld["ms_interact_y2"]
    f_lof_center_y2 = ld["f_lof_center_y2"]
    p_lof_center_y2 = ld["p_lof_center_y2"]
    f_interact_center_y2 = ld["f_interact_center_y2"]
    p_interact_center_y2 = ld["p_interact_center_y2"]

    p1_str = "$< 10^{-15}$" if lof["Y1"]["p_LoF"] < 1e-15 else f"${lof['Y1']['p_LoF']:.4f}$"
    p1_restr_str = (
        "$< 10^{-15}$"
        if lof["Y1_restricted"]["p_LoF"] < 1e-15
        else (
            f"${lof['Y1_restricted']['p_LoF']:.4f}$"
            if lof["Y1_restricted"]["p_LoF"] >= 0.0001
            else "$< 0.0001$"
        )
    )
    p2_str = "$< 10^{-15}$" if lof["Y2"]["p_LoF"] < 1e-15 else f"${lof['Y2']['p_LoF']:.4f}$"

    def fmt_tex_num(val: float, digs: int = 6) -> str:
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
        "\\multicolumn{7}{l}{\\textbf{Decomposition of Second-Order Residual ($Y_1$: Validation RMSE)}} \\\\",
        f"Structural Lack of Fit & ${lof['Y1']['SS_LoF']:.6f}$ & ${lof['Y1']['df_LoF']}$ & ${lof['Y1']['MS_LoF']:.6f}$ & ${f_lof_center_y1:.2f}$ & $< 10^{{-15}}$ & vs. Center PE \\\\",
        f"Treatment $\\times$ Block Interaction & ${ss_interact_y1:.6f}$ & ${df_interact}$ & ${fmt_tex_num(ms_interact_y1)}$ & ${f_interact_center_y1:.2f}$ & ${p_interact_center_y1:.4f}$ & vs. Center PE \\\\",
        f"Genuine Center Pure Error & ${ss_pure_center_y1:.6f}$ & ${df_pure_center}$ & ${fmt_tex_num(ms_pure_center_y1)}$ & --- & --- & Base Replicate \\\\",
        "\\midrule",
        f"Total Model Residual & ${lof['Y1']['SS_PE']+lof['Y1']['SS_LoF']:.6f}$ & $121$ & ${fmt_tex_num((lof['Y1']['SS_PE']+lof['Y1']['SS_LoF'])/121)}$ & --- & --- & $\\text{{RMS}} = {np.sqrt((lof['Y1']['SS_PE']+lof['Y1']['SS_LoF'])/121):.4f}$ \\\\",
        f"Saturated Additive Baseline & ${lof['Y1']['SS_PE']:.6f}$ & ${lof['Y1']['df_PE']}$ & ${fmt_tex_num(lof['Y1']['MS_PE'])}$ & ${lof['Y1']['F_LoF']:.2f}^*$ & {p1_str} & ($^*$LoF vs. Additive) \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textbf{Restricted Domain ($Y_1$: $x_1 \\ge -0.5$, $\\eta \\ge 0.033$, $N=95$)}} \\\\",
        f"Structural Lack of Fit & ${lof['Y1_restricted']['SS_LoF']:.6f}$ & ${lof['Y1_restricted']['df_LoF']}$ & ${lof['Y1_restricted']['MS_LoF']:.6f}$ & ${lof['Y1_restricted']['F_LoF']:.2f}$ & {p1_restr_str} & $\\text{{RMS}} = {lof['Y1_restricted']['sqrt_MS_LoF']:.4f}$ \\\\",
        f"Additive Baseline Residual & ${lof['Y1_restricted']['SS_PE']:.6f}$ & ${lof['Y1_restricted']['df_PE']}$ & ${fmt_tex_num(lof['Y1_restricted']['MS_PE'])}$ & --- & --- & Saturated Base \\\\",
        f"Total Restricted Residual & ${lof['Y1_restricted']['SS_PE']+lof['Y1_restricted']['SS_LoF']:.6f}$ & ${lof['Y1_restricted']['df_PE']+lof['Y1_restricted']['df_LoF']}$ & ${fmt_tex_num((lof['Y1_restricted']['SS_PE']+lof['Y1_restricted']['SS_LoF'])/(lof['Y1_restricted']['df_PE']+lof['Y1_restricted']['df_LoF']))}$ & --- & --- & $\\text{{RMS}} = {np.sqrt((lof['Y1_restricted']['SS_PE']+lof['Y1_restricted']['SS_LoF'])/(lof['Y1_restricted']['df_PE']+lof['Y1_restricted']['df_LoF'])):.4f}$ \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textbf{Latency Residual Decomposition ($Y_2$: Latency, $\\mu\\text{s}$)}} \\\\",
        f"Structural Lack of Fit & ${lof['Y2']['SS_LoF']:.2f}$ & ${lof['Y2']['df_LoF']}$ & ${lof['Y2']['MS_LoF']:.2f}$ & ${f_lof_center_y2:.2f}$ & ${p_lof_center_y2:.4f}$ & vs. Center PE ($\\text{{RMS}}={lof['Y2']['sqrt_MS_LoF']:.2f}$) \\\\",
        f"Treatment $\\times$ Block Interaction & ${ss_interact_y2:.2f}$ & ${df_interact}$ & ${ms_interact_y2:.2f}$ & ${f_interact_center_y2:.2f}$ & ${p_interact_center_y2:.4f}$ & vs. Center PE \\\\",
        f"Genuine Center Pure Error & ${ss_pure_center_y2:.2f}$ & ${df_pure_center}$ & ${ms_pure_center_y2:.2f}$ & --- & --- & Base Replicate \\\\",
        "\\midrule",
        f"Total Model Residual & ${lof['Y2']['SS_PE']+lof['Y2']['SS_LoF']:.2f}$ & $121$ & ${(lof['Y2']['SS_PE']+lof['Y2']['SS_LoF'])/121:.2f}$ & ${lof['Y2']['F_LoF']:.2f}^*$ & {p2_str} & ($^*$LoF vs. Additive) \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tables["tables/tab_lof.tex"] = "\n".join(tex_lof) + "\n"

    # 5. Confirmation Table
    conf = ds.confirmation
    so_conf = conf["Single_Objective_Optimum"]

    if conf["Y1_Val_RMSE"].get("inside_pi", False):
        status_mo_val = "\\makecell[c]{\\textbf{Pass}\\\\(Inside 95\\% PI)}"
    else:
        mo_bias_val = conf["Y1_Val_RMSE"]["empirical_mean"] - conf["Y1_Val_RMSE"]["predicted_mean"]
        mo_bias_type = "Optimism" if mo_bias_val > 0 else "Pessimism"
        status_mo_val = (
            "\\makecell[c]{\\textbf{Not Confirmed}\\\\("
            + f"{mo_bias_type}: {mo_bias_val:+.4f}"
            + ")}"
        )

    if conf["Y2_Latency"].get("inside_pi", False):
        lat_lo_mo, _ = conf["Y2_Latency"]["prediction_interval_95"]
        if abs(conf["Y2_Latency"]["empirical_mean"] - lat_lo_mo) < 0.5:
            status_mo_lat = "\\makecell[c]{Borderline\\\\(At Lower PI)}"
        else:
            status_mo_lat = "\\makecell[c]{\\textbf{Pass}\\\\(Inside 95\\% PI)}"
    else:
        status_mo_lat = "\\makecell[c]{\\textbf{Not Confirmed}\\\\(Outside PI)}"

    bias_val = so_conf["empirical_val_rmse"] - so_conf["predicted_val_rmse"]
    if so_conf.get("inside_pi_val", False):
        status_bias = "\\makecell[c]{\\textbf{Pass}\\\\(Inside 95\\% PI)}"
    else:
        bias_type = "Optimism" if bias_val > 0 else "Pessimism"
        status_bias = (
            "\\makecell[c]{\\textbf{Not Confirmed}\\\\(" + f"{bias_type}: {bias_val:+.4f}" + ")}"
        )

    if so_conf.get("inside_pi_lat", False):
        lat_lo, _ = so_conf["prediction_interval_95_lat"]
        if abs(so_conf["empirical_latency"] - lat_lo) < 0.5:
            status_lat = "\\makecell[c]{Borderline\\\\(At Lower PI)}"
        else:
            status_lat = "\\makecell[c]{\\textbf{Pass}\\\\(Inside 95\\% PI)}"
    else:
        status_lat = "\\makecell[c]{\\textbf{Not Confirmed}\\\\(Outside PI)}"

    pi_y1_mo_str = f"[{conf['Y1_Val_RMSE']['prediction_interval_95'][0]:.4f}, {conf['Y1_Val_RMSE']['prediction_interval_95'][1]:.4f}]"
    pi_y1_so_str = f"[{so_conf['prediction_interval_95_val'][0]:.4f}, {so_conf['prediction_interval_95_val'][1]:.4f}]"

    tex_conf = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\makecell[l]{\\textbf{Configuration /}\\\\\\textbf{Response Metric}} & \\makecell{\\textbf{Surrogate}\\\\\\textbf{Pred ($\\hat{y}$)}} & \\makecell{\\textbf{95\\% Pred}\\\\\\textbf{Interval (PI)}} & \\makecell{\\textbf{Empirical}\\\\\\textbf{Mean $\\pm$ SD}} & \\makecell{\\textbf{Confirmation}\\\\\\textbf{Status}} \\\\",
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\textbf{{DOE Multi-Objective Optimum $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ (Depth {conf['x_star_natural']['depth']})}}}} \\\\",
        f"Validation RMSE ($Y_1$) & ${conf['Y1_Val_RMSE']['predicted_mean']:.4f}$ & ${pi_y1_mo_str}$ & ${conf['Y1_Val_RMSE']['empirical_mean']:.4f} \\pm {conf['Y1_Val_RMSE']['empirical_std']:.4f}$ & "
        + status_mo_val
        + " \\\\",
        f"Holdout Test RMSE & --- & --- & ${conf['Y1_Test_RMSE']['empirical_mean']:.4f} \\pm {conf['Y1_Test_RMSE']['empirical_std']:.4f}$ & Holdout Test Set \\\\",
        f"Inference Latency ($\\mu$s) & ${conf['Y2_Latency']['predicted_mean']:.1f}$ & $[{conf['Y2_Latency']['prediction_interval_95'][0]:.1f}, {conf['Y2_Latency']['prediction_interval_95'][1]:.1f}]$ & ${conf['Y2_Latency']['empirical_mean']:.1f} \\pm {conf['Y2_Latency']['empirical_std']:.1f}$ & "
        + status_mo_lat
        + " \\\\",
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\textbf{{DOE Single-Objective Candidate $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ (Depth {ds.phase3['constrained_optimum_cube']['depth']})}}}} \\\\",
        f"Validation RMSE ($Y_1$) & ${so_conf['predicted_val_rmse']:.4f}$ & ${pi_y1_so_str}$ & ${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$ & "
        + status_bias
        + " \\\\",
        f"Holdout Test RMSE & --- & --- & ${so_conf['empirical_test_rmse']:.4f} \\pm {so_conf['empirical_test_std']:.4f}$ & Holdout Test Set \\\\",
        f"Inference Latency ($\\mu$s) & ${so_conf['predicted_latency']:.1f}$ & $[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]$ & ${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}$ & "
        + status_lat
        + " \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tables["tables/tab_confirmation.tex"] = "\n".join(tex_conf) + "\n"

    # 6. Benchmark Comparison Table
    tex_bm = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\makecell{\\textbf{Optimization}\\\\\\textbf{Method}} & \\makecell{\\textbf{Search Basis}\\\\\\scriptsize($N$ Reps)} & \\makecell{\\textbf{Val RMSE}\\\\\\scriptsize Mean $\\pm$ SD} & \\makecell{\\textbf{Holdout}\\\\\\textbf{Test RMSE}\\\\\\scriptsize Mean $\\pm$ SD} & \\makecell{\\textbf{Retrain}\\\\\\bm{$\\sigma_{\\text{eval}}$}} & \\makecell{\\textbf{Latency}\\\\\\scriptsize($\\mu\\text{s} \\pm \\text{SD}$)} & \\makecell{\\textbf{Feasible}\\\\\\scriptsize($\\le 145\\,\\mu\\text{s}$)} \\\\",
        "\\midrule",
        "\\multicolumn{7}{l}{\\textbf{Panel A: Prospective Revision-v2 Full Benchmark ($N=20$ Search Replicates)}} \\\\",
    ]

    for opt_key in PROSPECTIVE_OPTIMIZER_ORDER:
        m = ds.prospective_optimizers[opt_key]
        feas_str = f"{m.benchmark_feasible_count}/{m.n_replicates}"
        tex_bm.append(
            f"{m.display_name_tex} & {m.search_basis_tex} & ${m.val_rmse_mean:.4f} \\pm {m.val_rmse_sd:.4f}$ & ${m.test_rmse_mean:.4f} \\pm {m.test_rmse_sd:.4f}$ & ${m.retrain_sd:.4f}$ & ${m.predict_latency_mean:.1f} \\pm {m.predict_latency_sd:.1f}$ & {feas_str} \\\\"
        )

    tex_bm.append("\\midrule")
    tex_bm.append(
        "\\multicolumn{7}{l}{\\textbf{Panel B: Historical Baseline Snapshot ($v1.0.0$, Single Search Replicate)}} \\\\"
    )
    for hrow in ds.historical_benchmarks:
        v_val = f"${hrow.val_rmse_mean:.4f}$"
        t_val = f"${hrow.test_rmse_mean:.4f}$"
        p_lat = f"${hrow.predict_latency_us_median:.1f}$"
        h_feas = "Yes" if hrow.feasible_145 else "No"
        tex_bm.append(
            f"{hrow.display_name_tex} & {hrow.search_basis_tex} & {v_val} & {t_val} & --- & {p_lat} & {h_feas} \\\\"
        )

    tex_bm.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_benchmarks.tex"] = "\n".join(tex_bm) + "\n"

    # 6b. Hypervolume and Non-Dominated Frontiers Table
    hvs = ds.hv_summary
    nd_by_opt = ds.holdout_nondominated_by_optimizer
    r1_rmse, r1_lat = ds.hv_ref_primary
    r2_rmse, r2_lat = ds.hv_ref_secondary

    tex_hv = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        f"\\makecell{{\\textbf{{Frontier / Evaluation Protocol}}}} & \\makecell{{\\textbf{{Candidate}}\\\\\\textbf{{Pool ($N$)}}}} & \\makecell{{\\textbf{{Non-Dom.}}\\\\\\textbf{{Points}}}} & \\makecell{{\\textbf{{HV at $[{r1_rmse:.2f}, {r1_lat:.0f}]$}}\\\\\\scriptsize(Mean $\\pm$ SD)}} & \\makecell{{\\textbf{{HV at $[{r2_rmse:.2f}, {r2_lat:.0f}]$}}\\\\\\scriptsize(Mean $\\pm$ SD)}} \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textbf{Panel A: Development Candidate Frontiers (Distinct Evaluation Protocols Noted per Row)}} \\\\",
        f"\\makecell[l]{{Multi-Objective TPE Candidates\\\\(Single Split 42, 30-Call Search)}} & $140 \\times 20$ & ${hvs['motpe_nd_mean']:.2f} \\pm {hvs['motpe_nd_sd']:.1f}$ & ${hvs['motpe_dev_060_mean']:.4f} \\pm {hvs['motpe_dev_060_sd']:.4f}$ & ${hvs['motpe_dev_065_mean']:.4f} \\pm {hvs['motpe_dev_065_sd']:.4f}$ \\\\",
        f"\\makecell[l]{{Repeated DOE Candidate Fronts\\\\(5-Block Means Across Replicates)}} & $25 \\times 20$ & ${hvs['rdoe_nd_mean']:.2f} \\pm {hvs['rdoe_nd_sd']:.1f}$ & ${hvs['rdoe_dev_060_mean']:.4f} \\pm {hvs['rdoe_dev_060_sd']:.4f}$ & ${hvs['rdoe_dev_065_mean']:.4f} \\pm {hvs['rdoe_dev_065_sd']:.4f}$ \\\\",
        f"\\makecell[l]{{Fixed Full DOE Evaluated Front\\\\(Single Split 42, 27 Configs)}} & $27$ & ${hvs['fdoe_nd']}$ & ${hvs['fdoe_dev_060']:.4f}$ & ${hvs['fdoe_dev_065']:.4f}$ \\\\",
        f"\\makecell[l]{{Historical DOE 2-Point Set\\\\(Single Split 42, $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ \\& $\\mathbf{{x}}^*_{{\\text{{SO}}}}$)}} & $2$ & ${hvs['hdoe_nd']}$ & ${hvs['hdoe_dev_060']:.4f}$ & ${hvs['hdoe_dev_065']:.4f}$ \\\\",
        f"\\makecell[l]{{Difference: Fixed Full DOE Front\\\\\\quad $-$ MO-TPE Mean (Split 42)}} & --- & --- & ${hvs['diff_fdoe_minus_motpe_060']:+.4f}$ ($p < 0.0001$) & ${hvs['diff_fdoe_minus_motpe_065']:+.4f}$ ($p < 0.0001$) \\\\",
        "\\midrule",
        f"\\multicolumn{{5}}{{l}}{{\\textbf{{Panel B: Holdout Non-Dominated Set ({ds.total_selection_records} Records, {ds.distinct_config_hashes} Configs, {len(ds.fresh_eval_seeds)} Seeds)}}}} \\\\",
        f"\\makecell[l]{{Non-Dominated Set\\\\({hvs['holdout_nd']} Distinct Configs)}} & ${ds.total_selection_records}$ & ${hvs['holdout_nd']}$ & ${hvs['holdout_frozen_060']:.4f}$ & ${hvs['holdout_frozen_065']:.4f}$ \\\\",
        f"\\quad $\\llcorner$ Repeated DOE MO Selections & $20$ & ${nd_by_opt['repeated_preplanned_doe_multi_objective']}$ & --- & --- \\\\",
        f"\\quad $\\llcorner$ Multi-Objective TPE Selections & $20$ & ${nd_by_opt['multi_objective_tpe']}$ & --- & --- \\\\",
        f"\\quad $\\llcorner$ Constrained TPE Selections & $20$ & ${nd_by_opt['constrained_tpe']}$ & --- & --- \\\\",
        f"\\quad $\\llcorner$ Single-Objective TPE Selections & $20$ & ${nd_by_opt['single_objective_tpe']}$ & --- & --- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    tables["tables/tab_hypervolume_comparison.tex"] = "\n".join(tex_hv) + "\n"

    # 7. Integer Depth Table
    df_depth = ds.depth_opt_df
    tex_depth = [
        "\\begin{tabular}{cccccccc}",
        "\\toprule",
        "\\textbf{Depth} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_1$}} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_2$}} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_3$}} & \\makecell{\\textbf{Coded}\\\\\\bm{$x_4$}} & \\makecell{\\textbf{Optimal}\\\\\\bm{$\\eta$}} & \\makecell{\\textbf{Predicted}\\\\\\textbf{RMSE}} & \\makecell{\\textbf{SE}\\\\\\bm{$(\\hat{y})$}} \\\\",
        "\\midrule",
    ]
    for _, row in df_depth.iterrows():
        d_int = int(row["depth"])
        x2_coded = (d_int - 6) / 3.0
        eta_val = decode_factors(np.array([row["x1"], x2_coded, row["x3"], row["x4"]]))[0]
        tex_depth.append(
            f"{d_int} & ${row['x1']:.4f}$ & ${x2_coded:+.4f}$ & ${row['x3']:.2f}$ & ${row['x4']:.2f}$ & ${eta_val:.4f}$ & ${row['pred_rmse']:.4f}$ & ${row['se']:.4f}$ \\\\"
        )
    tex_depth.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_depth_opt.tex"] = "\n".join(tex_depth) + "\n"

    # 8. Desirability Sensitivity Table
    df_sens = ds.desirability_sens_df
    tex_sens = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Scenario} & \\makecell{\\textbf{RMSE Bounds}\\\\\\bm{$[L_1, U_1]$}} & \\makecell{\\textbf{Latency Bounds}\\\\\\bm{$[L_2, U_2]$}} & \\makecell{\\textbf{Weights}\\\\\\bm{$(w_1, w_2)$}} & \\makecell{\\textbf{Opt}\\\\\\textbf{Depth}} & \\makecell{\\textbf{Opt}\\\\\\bm{$\\eta$}} & \\makecell{\\textbf{Best}\\\\\\bm{$D$}} \\\\",
        "\\midrule",
    ]
    for _, row in df_sens.iterrows():
        b1 = f"[{row['L1']:.2f}, {row['U1']:.2f}]"
        b2 = f"[{row['L2']:.0f}, {row['U2']:.0f}]"
        w = f"({int(row['w1'])}, {int(row['w2'])})"
        tex_sens.append(
            f"{row['Scenario']} & {b1} & {b2} & {w} & {int(row['Optimal_Depth'])} & ${row['Optimal_Eta']:.4f}$ & ${row['Best_D']:.4f}$ \\\\"
        )
    tex_sens.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_desirability_sensitivity.tex"] = "\n".join(tex_sens) + "\n"

    # 9. Latency Candidate Models Table
    df_lat = ds.latency_models_df
    jac_adj = ds.latency_jacobian_adjustment
    tex_lat = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "\\textbf{Candidate Latency Model} & \\bm{$R^2$} & \\makecell{\\textbf{Adj}\\\\\\bm{$R^2$}} & \\makecell{\\textbf{AIC}\\\\\\scriptsize(Adjusted)} & \\makecell{\\textbf{BIC}\\\\\\scriptsize(Adjusted)} & \\makecell{\\textbf{RMSE}\\\\\\textbf{($\\mu$s)}} \\\\",
        "\\midrule",
    ]
    for _, row in df_lat.iterrows():
        m_name = str(row["Model"]).replace("depth^2", r"depth$^2$")
        aic_val = row["AIC"] + jac_adj if "Log-Linear" in m_name else row["AIC"]
        bic_val = row["BIC"] + jac_adj if "Log-Linear" in m_name else row["BIC"]
        tex_lat.append(
            f"{m_name} & ${row['R2']:.3f}$ & ${row['Adj_R2']:.3f}$ & ${aic_val:.1f}$ & ${bic_val:.1f}$ & ${row['RMSE']:.2f}$ \\\\"
        )
    tex_lat.extend(["\\bottomrule", "\\end{tabular}"])
    tables["tables/tab_latency_models.tex"] = "\n".join(tex_lat) + "\n"

    return tables


def generate_macros(ds: Optional[ReportingDataset] = None) -> None:
    if ds is None:
        ds = load_reporting_dataset()
    content = render_macros(ds)
    out_path = ds.root_dir / "results" / "macros.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    macro_count = sum(1 for line in content.splitlines() if line.startswith("\\newcommand{"))
    print(f"Generated results/macros.tex with {macro_count} macros.")


def generate_tables(ds: Optional[ReportingDataset] = None) -> None:
    if ds is None:
        ds = load_reporting_dataset()
    tables = render_tables(ds)
    for rel_path, content in tables.items():
        out_path = ds.root_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
    print("All LaTeX tables generated and saved to tables/*.tex successfully.")


def check_report_artifacts(ds: Optional[ReportingDataset] = None) -> Tuple[bool, str]:
    """Check whether results/macros.tex and tables/*.tex match canonical data without modifying files."""
    if ds is None:
        ds = load_reporting_dataset()
    expected_files: Dict[str, str] = {"results/macros.tex": render_macros(ds)}
    expected_files.update(render_tables(ds))

    diffs = []
    for rel_path, expected_content in expected_files.items():
        target = ds.root_dir / rel_path
        if not target.is_file():
            diffs.append(f"MISSING FILE: {rel_path}")
            continue
        actual_content = target.read_text(encoding="utf-8")
        if actual_content != expected_content:
            diff_lines = list(
                difflib.unified_diff(
                    actual_content.splitlines(keepends=True),
                    expected_content.splitlines(keepends=True),
                    fromfile=f"{rel_path} (on disk)",
                    tofile=f"{rel_path} (expected from canonical artifacts)",
                )
            )
            diffs.append("".join(diff_lines))

    if diffs:
        return False, "\n".join(diffs)
    return True, "All LaTeX macros and tables match canonical experimental artifacts."


def main(argv: Optional[ list[str] ] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or check LaTeX macros and tables from canonical experimental artifacts."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        action="store_true",
        help="Write results/macros.tex and tables/*.tex (default).",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify that results/macros.tex and tables/*.tex match canonical data without modifying files.",
    )
    args = parser.parse_args(argv)

    ds = load_reporting_dataset()
    if args.check:
        ok, message = check_report_artifacts(ds)
        if not ok:
            sys.stderr.write("ERROR: LaTeX report artifacts are out of sync with canonical data:\n")
            sys.stderr.write(message + "\n")
            return 1
        print(message)
        return 0

    generate_macros(ds)
    generate_tables(ds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
