"""
scripts/generate_research_reporting.py - Deterministic generator and verifier for REPORT.md,
docs/generated/scientific_results_manifest.json, and manuscript-level numerical auditing.

Consumes scripts/reporting_data.py as the single source of truth so REPORT.md, report.tex,
results/macros.tex, tables/*.tex, and docs/generated/scientific_results_manifest.json agree
by construction.

Supports:
  python scripts/generate_research_reporting.py --write
  python scripts/generate_research_reporting.py --check
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline import decode_factors
from scripts.generate_report_artifacts import (
    check_report_artifacts,
    generate_macros,
    generate_tables,
    render_macros,
)
from scripts.reporting_data import (
    PROSPECTIVE_OPTIMIZER_ORDER,
    ReportingDataError,
    ReportingDataset,
    build_scientific_results_manifest,
    load_reporting_dataset,
)

SECTION_IDS: Tuple[str, ...] = (
    "EXEC_SUMMARY",
    "SECTION_1_FACTORS_AND_BLOCKS",
    "SECTION_2_PHASE1",
    "SECTION_3_PHASE2",
    "SECTION_4_PHASE3",
    "SECTION_5_PHASE4",
    "SECTION_6_PHASE5",
    "SECTION_7_BENCHMARKS",
    "SECTION_8_9_DISCUSSION_AND_ARTIFACTS",
)


def _begin_marker(section_id: str) -> str:
    return f"<!-- BEGIN AUTO-GENERATED: {section_id} -->"


def _end_marker(section_id: str) -> str:
    return f"<!-- END AUTO-GENERATED: {section_id} -->"


def _fmt_sci_md(val: float, digits: int = 2) -> str:
    s = f"{val:.{digits}e}"
    mant, exp_s = s.split("e")
    exp = int(exp_s)
    return f"{float(mant):.{digits}f} \\times 10^{{{exp}}}"


def render_report_sections(ds: ReportingDataset) -> Dict[str, str]:
    """Render each auto-generated section of REPORT.md deterministically from ReportingDataset."""
    p1 = ds.phase1
    p3 = ds.phase3
    lof = ds.lof
    ld = ds.lof_decomp
    diag = ds.diagnostics
    icc = ds.icc
    conf = ds.confirmation
    so_conf = conf["Single_Objective_Optimum"]
    d_star = ds.desirability_star
    opts = ds.prospective_optimizers
    hvs = ds.hv_summary
    nd_by_opt = ds.holdout_nondominated_by_optimizer

    block_seeds_str = ", ".join(str(s) for s in ds.block_seeds)
    conf_seeds_str = ", ".join(str(s) for s in ds.confirmation_seeds)
    n_conf = len(ds.confirmation_seeds)

    an_p1 = ds.anova_phase1_df
    ss_res_p1 = float(an_p1.loc["Residual", "sum_sq"])
    f_x1_p1 = float(an_p1.loc["x1", "F"])
    eta_x1_p1 = float(an_p1.loc["x1", "sum_sq"] / (an_p1.loc["x1", "sum_sq"] + ss_res_p1))
    f_x2_p1 = float(an_p1.loc["x2", "F"])
    eta_x2_p1 = float(an_p1.loc["x2", "sum_sq"] / (an_p1.loc["x2", "sum_sq"] + ss_res_p1))
    f_x1x2_p1 = float(an_p1.loc["x1:x2", "F"])
    eta_x1x2_p1 = float(an_p1.loc["x1:x2", "sum_sq"] / (an_p1.loc["x1:x2", "sum_sq"] + ss_res_p1))

    an_ccd_y1 = ds.anova_ccd_y1_df
    an_ccd_y2 = ds.anova_ccd_y2_df
    f_block_ccd = float(icc["Y1"]["f_block"])
    icc_anova_pct = float(icc["Y1"]["icc_anova"] * 100.0)
    icc_reml = float(icc["Y1"]["icc_reml"])
    sigma_block = float(np.sqrt(max(0.0, icc["Y1"]["s2b_anova"])))

    eigs = p3["eigenvalues"]
    boot_p = p3["bootstrap_percentiles"]
    cube = p3["constrained_optimum_cube"]
    cube_x = cube["x"]
    cube_nat = cube["natural"]

    mo_coded = conf["x_star_coded"]
    mo_nat = conf["x_star_natural"]
    mo_val = conf["Y1_Val_RMSE"]
    mo_test = conf["Y1_Test_RMSE"]
    mo_lat = conf["Y2_Latency"]
    so_bias = float(so_conf["empirical_val_rmse"] - so_conf["predicted_val_rmse"])
    so_above_pi = float(so_conf["empirical_val_rmse"] - so_conf["prediction_interval_95_val"][1])

    doe_mo = opts["repeated_preplanned_doe_multi_objective"]
    mo_tpe = opts["multi_objective_tpe"]
    ctpe = opts["constrained_tpe"]
    doe_so = opts["repeated_preplanned_doe_single_objective"]
    so_tpe = opts["single_objective_tpe"]
    rs = opts["random_search"]

    r1_rmse, r1_lat = ds.hv_ref_primary
    r2_rmse, r2_lat = ds.hv_ref_secondary

    sections: Dict[str, str] = {}

    # 1. EXEC_SUMMARY
    sections["EXEC_SUMMARY"] = "\n".join(
        [
            f"This report presents a comprehensive Design of Experiments (DOE) and Response Surface Methodology (RSM) study applied to the multi-objective hyperparameter optimization of an **XGBoost Regressor** on the **California Housing** benchmark dataset ($N = 20,640$ observations, $8$ continuous features). Following the methodology of Douglas C. Montgomery's *Design and Analysis of Experiments* (9th ed., 2017, Chapters 5, 9, 10, and 14), we treat stochastic machine learning variability—arising from random train/validation partitioning and row subsampling—as a **nuisance factor** controlled via a **Randomized Complete Block Design (RCBD)** across $b = {len(ds.block_seeds)}$ seed blocks ($\\mathcal{{S}} = \\{{{block_seeds_str}\\}}$).",
            "",
            f"All primary DOE phases consist of **{140 + n_conf} genuine model training and evaluation runs** (100 in Phase 1, 40 axial augmentations in Phase 2, and {n_conf} confirmation trials in Phase 5 for $\\mathbf{{x}}^*_{{\\text{{MO}}}}$, supplemented by {n_conf} confirmation trials for $\\mathbf{{x}}^*_{{\\text{{SO}}}}$), alongside a prospective 20-replicate benchmark campaign comprising **{ds.total_model_fits:,} XGBoost model fits** and **{ds.total_timed_inferences:,} timed single-sample inferences**:",
            "",
            "1. **Holdout Isolation and Dataset History**: The 20,640 dataset observations are partitioned into an 80% development pool ($N_{\\text{dev}} = 16,512$) and a 20% external holdout test set ($N_{\\text{test}} = 4,128$). Within each block $b$, the development pool is split 75/25 into training ($N_{\\text{train}} = 12,384$) and validation ($N_{\\text{val}} = 4,128$) sets. Response $Y_1$ is **Development Validation RMSE** on the 25% validation split. Whereas the historical `v1.0.0` baseline experiments recorded holdout test results during development, the revised `Revision-v2` pipeline programmatically prevented access to those test labels during all search iterations, surrogate fitting, curvature testing, lack-of-fit analysis, and desirability optimization until winning configurations were frozen in `finalized_selections.json`. These safeguards enforce programmatic pipeline isolation during the revised search, although they cannot retroactively undo earlier historical exposure of the same dataset.",
            f"2. **Phase 1 ($2^4$ Full Factorial + Center Points in {len(ds.block_seeds)} Blocks, $N_1 = 100$ runs)**: Screening identifies **Learning Rate** ($\\ln \\eta$, $F = {f_x1_p1:.2f}, p < 10^{{-15}}, \\eta^2_p = {eta_x1_p1:.4f}$) and **Max Tree Depth** ($F = {f_x2_p1:.2f}, p < 0.0001, \\eta^2_p = {eta_x2_p1:.4f}$), along with their interaction $x_1 x_2$ ($F = {f_x1x2_p1:.2f}, p < 0.0001, \\eta^2_p = {eta_x1x2_p1:.4f}$), as the dominant drivers of validation RMSE. Single-degree-of-freedom curvature testing against within-block center-point pure error ($\\text{{df}} = {p1['df_pe_center']}, \\text{{MS}}_{{\\text{{PE, center}}}} = {_fmt_sci_md(p1['ms_pe_center'])}$) reveals strong quadratic curvature ($F_{{\\text{{Curv}}}} = {p1['f_curvature']:,.2f}, p < 10^{{-15}}$), where the factorial corner mean ($\\bar{{y}}_F = {p1['yF_bar']:.4f}$) exceeds the center mean ($\\bar{{y}}_C = {p1['yC_bar']:.4f}$) by ${p1['diff_F_minus_C']:.4f}$ RMSE.",
            f"3. **Phase 2 (Face-Centered Central Composite Design, $\\alpha = 1.0$, $N_{{\\text{{CCD}}}} = 140$ runs)**: Augmenting with $40$ axial points fits a full second-order polynomial ($R^2 = {ds.ccd_y1_rsq:.4f}, \\text{{Adj }} R^2 = {ds.ccd_y1_adj_rsq:.4f}$). Once quadratic curvature is modeled, the RCBD block effect becomes highly significant ($F = {f_block_ccd:.2f}, p < 0.0001$), absorbing **ICC = {icc_anova_pct:.2f}%** ($\\text{{ICC}}_{{\\text{{REML}}}} = {icc_reml:.4f}, \\sigma_{{\\text{{block}}}} \\approx {sigma_block:.4f}$ RMSE) of unexplained residual variance. Formal lack-of-fit decomposition shows statistically significant structural polynomial misfit ($F_{{\\text{{LoF}}}} = {lof['Y1']['F_LoF']:.2f}, p = {_fmt_sci_md(lof['Y1']['p_LoF'])}$ against the {lof['Y1']['df_PE']}-df saturated additive baseline; $F_{{\\text{{LoF}}}} = {ld['f_lof_center_y1']:.2f}, p < 10^{{-15}}$ against {int(ld['df_pure_center'])}-df center pure error; RMS misfit $= {lof['Y1']['sqrt_MS_LoF']:.4f}$ RMSE), demonstrating that second-order polynomials serve as local guidance maps rather than globally exact estimators of gradient boosting loss.",
            f"4. **Phase 3 (Canonical & Ridge Analysis)**: Spectral decomposition of $\\hat{{\\mathbf{{B}}}}$ yields three positive eigenvalues and one near-zero eigenvalue ($\\lambda = \\{{{eigs[0]:.6f}, {eigs[1]:.6f}, {eigs[2]:.6f}, {eigs[3]:.6f}\\}}$; wild bootstrap 95% CI for $\\lambda_1$: $[{boot_p[0][0]:.4f}, {boot_p[2][0]:+.4f}]$, with ${p3['fraction_min_eigenvalue_le_zero']*100:.1f}\\%$ of resamples $\\le 0$), characterizing a **stationary/rising ridge system**. The unconstrained stationary point ($\\|\\mathbf{{x}}_0\\| = {p3['distance_coded']:.2f}$) lies outside $[-1, +1]^4$; constrained optimization over valid integer depths identifies the single-objective candidate $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ at depth ${cube['depth']}$ ($x_1 = {cube_x[0]:.4f}, x_2 = {cube_x[1]:.4f}, x_3 = {cube_x[2]:.4f}, x_4 = {cube_x[3]:.4f}$, i.e., $\\eta = {cube_nat['eta']:.4f}, \\text{{depth}} = {cube['depth']}, \\text{{subsample}} = {cube_nat['subsample']:.4f}, \\lambda = {cube_nat['reg_lambda']:.4f}$) with predicted validation RMSE $\\hat{{y}} = {cube['pred_rmse']:.4f}$.",
            f"5. **Phase 4 (Derringer-Suich Multi-Objective Desirability)**: Balancing Validation RMSE ($Y_1 \\in [0.450, 0.700]$) and Single-Sample Inference Latency ($Y_2 \\in [100.0, 180.0]\\,\\mu\\text{{s}}$) yields an interior compromise coordinate $\\mathbf{{x}}^*_{{\\text{{MO}}}} = [{mo_coded[0]:.4f}, {mo_coded[1]:.4f}, {mo_coded[2]:.4f}, {mo_coded[3]:.4f}]^T$ ($\\eta = {mo_nat['eta']:.4f}, \\text{{depth}} = {mo_nat['depth']}, \\text{{subsample}} = {mo_nat['subsample']:.4f}, \\lambda = {mo_nat['reg_lambda']:.4f}$), achieving composite desirability **$D = {d_star['D']:.4f}$** ($d_1 = {d_star['d1']:.4f}, d_2 = {d_star['d2']:.4f}$) with predicted validation RMSE $\\hat{{Y}}_1 = {d_star['pred_y1']:.4f}$ and predicted latency $\\hat{{Y}}_2 = {d_star['pred_y2']:.2f}\\,\\mu\\text{{s}}$.",
            f"6. **Phase 5 (Confirmation & Comparative Benchmarks)**: Across **{n_conf} confirmation trials** ($m = {n_conf}$ fresh seeds $\\mathcal{{S}}_{{\\text{{conf}}}} = \\{{{conf_seeds_str}\\}}$), $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ achieves empirical validation RMSE ${mo_val['empirical_mean']:.4f} \\pm {mo_val['empirical_std']:.4f}$ (falling inside the Satterthwaite 95% prediction interval $[{mo_val['prediction_interval_95'][0]:.4f}, {mo_val['prediction_interval_95'][1]:.4f}]$), empirical latency ${mo_lat['empirical_mean']:.2f} \\pm {mo_lat['empirical_std']:.2f}\\,\\mu\\text{{s}}$ (${mo_lat['empirical_mean']:.1f} \\pm {mo_lat['empirical_std']:.1f}\\,\\mu\\text{{s}}$, inside $[{mo_lat['prediction_interval_95'][0]:.2f}, {mo_lat['prediction_interval_95'][1]:.2f}]\\,\\mu\\text{{s}}$), and external holdout test RMSE ${mo_test['empirical_mean']:.4f} \\pm {mo_test['empirical_std']:.4f}$. At the single-objective depth-{cube['depth']} candidate $\\mathbf{{x}}^*_{{\\text{{SO}}}}$, empirical validation RMSE is ${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$—lying ${so_above_pi:.4f}$ above the Satterthwaite 95% PI $[{so_conf['prediction_interval_95_val'][0]:.4f}, {so_conf['prediction_interval_95_val'][1]:.4f}]$ due to ${so_bias:+.4f}$ polynomial optimism bias—while achieving external holdout test RMSE ${so_conf['empirical_test_rmse']:.4f} \\pm {so_conf['empirical_test_std']:.4f}$ and borderline latency ${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}\\,\\mu\\text{{s}}$ (at the lower bound of $[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]\\,\\mu\\text{{s}}$). In the 20-replicate prospective benchmark campaign, Single-Objective TPE achieves the lowest observed test RMSE (${so_tpe.test_rmse_mean:.5f} \\pm {so_tpe.test_rmse_sd:.5f}$) at high latency (${so_tpe.predict_latency_mean:.2f} \\pm {so_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$), whereas Repeated DOE Single-Objective yields deterministic selection (${doe_so.test_rmse_mean:.5f} \\pm {doe_so.test_rmse_sd:.5f}$, retraining $\\sigma_{{\\text{{eval}}}} = {doe_so.retrain_sd:.5f}$) at ${ds.doe_so_vs_sotpe_latency_reduction_pct:.1f}\\%$ lower latency (${doe_so.predict_latency_mean:.2f} \\pm {doe_so.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$). In multi-objective optimization, Repeated DOE ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$) and Multi-Objective TPE show no statistically significant difference in holdout test RMSE (${doe_mo.test_rmse_mean:.5f} \\pm {doe_mo.test_rmse_sd:.5f}$ vs. ${mo_tpe.test_rmse_mean:.5f} \\pm {mo_tpe.test_rmse_sd:.5f}$, difference ${ds.doe_mo_vs_motpe_diff:+.5f}, p = {ds.doe_mo_vs_motpe_welch_p:.4f}$) at comparable latency (${doe_mo.predict_latency_mean:.2f} \\pm {doe_mo.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ vs. ${mo_tpe.predict_latency_mean:.2f} \\pm {mo_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$) and $100\\%$ ($20/20$) constraint feasibility.",
        ]
    )

    # 2. SECTION_1_FACTORS_AND_BLOCKS
    f_cfg = ds.config["factors"]
    x1_min, x1_max = float(f_cfg["x1"]["min"]), float(f_cfg["x1"]["max"])
    x1_cent = float(np.sqrt(x1_min * x1_max))
    x2_min, x2_max = int(f_cfg["x2"]["min"]), int(f_cfg["x2"]["max"])
    x2_cent = int((x2_min + x2_max) // 2)
    x3_min, x3_max = float(f_cfg["x3"]["min"]), float(f_cfg["x3"]["max"])
    x3_cent = 0.5 * (x3_min + x3_max)
    x4_min, x4_max = float(f_cfg["x4"]["min"]), float(f_cfg["x4"]["max"])
    x4_cent = float(np.sqrt(x4_min * x4_max))

    sections["SECTION_1_FACTORS_AND_BLOCKS"] = "\n".join(
        [
            "### 1.1 Hyperparameter Space & Natural-to-Coded Transformations",
            "",
            "We investigate $k = 4$ hyperparameters of an XGBoost Regressor (`n_estimators = 100`, `objective = 'reg:squarederror'`, `n_jobs = 1` for inference). Factors spanning orders of magnitude ($x_1$ and $x_4$) are mapped via natural logarithmic transformations prior to linear coding into $[-1, +1]$:",
            "",
            "| Coded Factor | Parameter Name | Symbol | Scale Transformation | Low ($-1$) | Center ($0$) | High ($+1$) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **$x_1$ (Factor A)** | `learning_rate` | $\\eta$ | $\\xi_1 = \\ln(\\eta)$ | ${x1_min:.4f}$ | ${x1_cent:.4f}$ | ${x1_max:.4f}$ |",
            f"| **$x_2$ (Factor B)** | `max_depth` | $d$ | Linear (Integer) | ${x2_min}$ | ${x2_cent}$ | ${x2_max}$ |",
            f"| **$x_3$ (Factor C)** | `subsample` | $s$ | Linear | ${x3_min:.4f}$ | ${x3_cent:.4f}$ | ${x3_max:.4f}$ |",
            f"| **$x_4$ (Factor D)** | `reg_lambda` | $\\lambda$ | $\\xi_4 = \\ln(\\lambda)$ | ${x4_min:.4f}$ | ${x4_cent:.4f}$ | ${x4_max:.4f}$ |",
            "",
            "The dimensionless coded variables $x_i \\in [-1, +1]$ are defined by:",
            "$$x_1 = \\frac{\\ln(\\eta) - (-2.9046)}{1.7006}, \\quad x_2 = \\frac{d - 6}{3}, \\quad x_3 = \\frac{s - 0.75}{0.25}, \\quad x_4 = \\frac{\\ln(\\lambda) - 0}{2.3026}$$",
            "",
            "### 1.2 Nuisance Blocking (RCBD) and Dual Responses",
            "",
            f"Each of the $b = {len(ds.block_seeds)}$ blocks ($\\mathcal{{S}} = \\{{{block_seeds_str}\\}}$) defines a deterministic 75/25 train/validation split of the 16,512-sample development pool via split seed $S_b$. For factorial corner points ($1 \\dots 16$) and axial points ($18 \\dots 25$), the XGBoost model seed equals $S_b$. For the $n_C = 4$ center-point replicates within each block, the train/validation split is held fixed at $S_b$ while the XGBoost subsampling seed is varied as $S_b + r \\times 1000$ ($r \\in \\{{0, 1, 2, 3\\}}$), isolating **{int(ld['df_pure_center'])} degrees of freedom of within-block subsampling pure error** from **4 degrees of freedom of block-to-block data-split variance**.",
            "",
            "Two response variables are recorded per run:",
            "- **Response $Y_1$ (Development Validation RMSE)**: Root Mean Squared Error on the 25% validation partition ($N_{\\text{val}} = 4,128$). Unlike the historical `v1.0.0` pipeline, which logged holdout test metrics during development, the revised pipeline prevented access to the 20% external holdout test partition ($N_{\\text{test}} = 4,128$) until candidate configurations were frozen in `finalized_selections.json`.",
            "- **Response $Y_2$ (Single-Sample Inference Latency, $\\mu\\text{s}$)**: Mean execution time in microseconds per single-row prediction on CPU Core 0 (`SetProcessAffinityMask = 1`, `n_jobs = 1`), timed over 1,000 single-sample calls after 100 untimed warmup calls (50 per prediction interface: `predict` and `inplace_predict`) using `time.perf_counter_ns()`.",
        ]
    )

    # 3. SECTION_2_PHASE1
    p1_source_map = [
        ("Intercept", "**Intercept**"),
        ("C(block)", "**Block Effect $C(\\text{block})$**"),
        ("x1", "**Factor A: $x_1$ ($\\ln \\eta$)**"),
        ("x2", "**Factor B: $x_2$ (`max_depth`)**"),
        ("x3", "**Factor C: $x_3$ (`subsample`)**"),
        ("x4", "**Factor D: $x_4$ ($\\ln \\lambda$)**"),
        ("x1:x2", "**Interaction $x_1 \\cdot x_2$**"),
        ("x1:x3", "**Interaction $x_1 \\cdot x_3$**"),
        ("x1:x4", "**Interaction $x_1 \\cdot x_4$**"),
        ("x2:x3", "**Interaction $x_2 \\cdot x_3$**"),
        ("x2:x4", "**Interaction $x_2 \\cdot x_4$**"),
        ("x3:x4", "**Interaction $x_3 \\cdot x_4$**"),
        ("Residual", "**Residual Error**"),
    ]
    p1_rows = [
        "### 2.1 Phase 1 ANOVA (Type III Sum of Squares)",
        "",
        f"Phase 1 evaluates $2^4 = 16$ factorial corners plus $n_C = 4$ center replicates across $b = {len(ds.block_seeds)}$ blocks ($N_1 = 100$ runs). Fitting the first-order model with two-factor interactions yields:",
        "",
        "| Source of Variation | Sum of Squares (SS) | DF | Mean Square (MS) | $F$-Value | $p$-Value | Partial $\\eta^2$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for src_key, src_label in p1_source_map:
        ss = float(an_p1.loc[src_key, "sum_sq"])
        df_v = int(an_p1.loc[src_key, "df"])
        ms = ss / df_v
        if src_key == "Residual":
            p1_rows.append(f"| {src_label} | ${ss:.6f}$ | ${df_v}$ | ${ms:.6f}$ | — | — | — |")
        else:
            fv = float(an_p1.loc[src_key, "F"])
            pv = float(an_p1.loc[src_key, "PR(>F)"])
            eta_v = ss / (ss + ss_res_p1)
            p_str = "$< 10^{-15}$" if pv < 1e-15 else f"${pv:.4f}$"
            p1_rows.append(
                f"| {src_label} | ${ss:.6f}$ | ${df_v}$ | ${ms:.6f}$ | ${fv:.2f}$ | {p_str} | ${eta_v:.4f}$ |"
            )

    p1_rows.extend(
        [
            "",
            "### 2.2 Single-Degree-of-Freedom Curvature Test",
            "",
            f"To test $H_0: \\sum_{{i=1}}^4 \\beta_{{ii}} = 0$, we compare the mean validation RMSE of the $n_F = 80$ factorial corner runs against the $n_C = 20$ center-point runs against within-block center-point pure error ($\\text{{df}} = {p1['df_pe_center']}$):",
            f"- **Factorial Mean ($\\bar{{y}}_F$)**: ${p1['yF_bar']:.4f}$",
            f"- **Center Point Mean ($\\bar{{y}}_C$)**: ${p1['yC_bar']:.4f}$",
            f"- **Curvature Contrast ($\\bar{{y}}_F - \\bar{{y}}_C$)**: ${p1['diff_F_minus_C']:+.4f}$ RMSE",
            f"- **Curvature Sum of Squares**: $\\text{{SS}}_{{\\text{{Curv}}}} = \\frac{{n_F n_C}}{{n_F + n_C}}(\\bar{{y}}_F - \\bar{{y}}_C)^2 = {p1['ss_curvature']:.6f}$",
            f"- **Within-Block Center Pure Error**: $\\text{{SS}}_{{\\text{{PE, center}}}} = {ld['ss_pure_center_y1']:.6f}$, $\\text{{df}}_{{\\text{{PE}}}} = {p1['df_pe_center']}$, $\\text{{MS}}_{{\\text{{PE, center}}}} = {_fmt_sci_md(p1['ms_pe_center'])}$",
            "- **Curvature $F$-Statistic**:",
            f"$$F_{{\\text{{Curv}}}} = \\frac{{\\text{{SS}}_{{\\text{{Curv}}}}}}{{\\text{{MS}}_{{\\text{{PE, center}}}}}} = \\frac{{{p1['ss_curvature']:.6f}}}{{{_fmt_sci_md(p1['ms_pe_center'])}}} = {p1['f_curvature']:,.2f} \\quad (p < 10^{{-15}})$$",
            "*(Note: If evaluated against overall pure error pooled across all replicated design coordinates, $\\text{df} = 83, \\text{MS}_{\\text{PE, All}} = 6.98 \\times 10^{-5}$, the $F$-statistic is $F \\approx 4,140, p < 10^{-15}$. Both confirm severe quadratic curvature requiring Phase 2 CCD augmentation.)*",
        ]
    )
    sections["SECTION_2_PHASE1"] = "\n".join(p1_rows)

    # 4. SECTION_3_PHASE2
    ccd_source_map = [
        ("Intercept", "**Intercept**"),
        ("C(block)", "**Block Effect $C(\\text{block})$**"),
        ("x1", "**Factor A: $x_1$ ($\\ln \\eta$)**"),
        ("x2", "**Factor B: $x_2$ (Depth)**"),
        ("x3", "**Factor C: $x_3$ (Subsample)**"),
        ("x4", "**Factor D: $x_4$ ($\\ln \\lambda$)**"),
        ("x1_sq", "**Quadratic $x_1^2$**"),
        ("x2_sq", "**Quadratic $x_2^2$**"),
        ("x3_sq", "**Quadratic $x_3^2$**"),
        ("x4_sq", "**Quadratic $x_4^2$**"),
        ("x1_x2", "**Interaction $x_1 \\cdot x_2$**"),
        ("x1_x3", "**Interaction $x_1 \\cdot x_3$**"),
        ("x1_x4", "**Interaction $x_1 \\cdot x_4$**"),
        ("x2_x3", "**Interaction $x_2 \\cdot x_3$**"),
        ("x2_x4", "**Interaction $x_2 \\cdot x_4$**"),
        ("x3_x4", "**Interaction $x_3 \\cdot x_4$**"),
        ("Residual", "**Residual Error**"),
    ]
    p2_rows = [
        "### 3.1 Second-Order Response Surface ANOVA ($Y_1$: Validation RMSE and $Y_2$: Inference Latency)",
        "",
        f"Augmenting Phase 1 with $2k = 8$ axial points ($\\alpha = 1.0$) across {len(ds.block_seeds)} blocks yields $N = 140$ runs ($28$ runs/block). The fitted second-order validation RMSE model achieves $R^2 = {ds.ccd_y1_rsq:.4f}$ and $\\text{{Adjusted }} R^2 = {ds.ccd_y1_adj_rsq:.4f}$:",
        "",
        "| Source of Variation | SS | DF | MS | $F$-Stat | OLS $p$ | HC3 SE | HC3 $p$ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for src_key, src_label in ccd_source_map:
        ss = float(an_ccd_y1.loc[src_key, "sum_sq"])
        df_v = int(an_ccd_y1.loc[src_key, "df"])
        ms = ss / df_v
        if src_key == "Residual":
            p2_rows.append(f"| {src_label} | ${ss:.6f}$ | ${df_v}$ | ${ms:.6f}$ | — | — | — | — |")
        else:
            fv = float(an_ccd_y1.loc[src_key, "F"])
            pv = float(an_ccd_y1.loc[src_key, "PR(>F)"])
            p_str = "$< 10^{-15}$" if pv < 1e-15 else (f"${pv:.4f}$" if pv >= 0.0001 else "$< 0.0001$")
            if src_key in ds.hc3_ccd_y1_se.index:
                hc3_se = float(ds.hc3_ccd_y1_se[src_key])
                hc3_p = float(ds.hc3_ccd_y1_p[src_key])
                hc3_se_s = f"${hc3_se:.4f}$"
                hc3_p_s = (
                    "$< 10^{-15}$"
                    if hc3_p < 1e-15
                    else (f"${hc3_p:.4f}$" if hc3_p >= 0.0001 else "$< 0.0001$")
                )
            else:
                hc3_se_s = "—"
                hc3_p_s = "—"
            p2_rows.append(
                f"| {src_label} | ${ss:.6f}$ | ${df_v}$ | ${ms:.6f}$ | ${fv:.2f}$ | {p_str} | {hc3_se_s} | {hc3_p_s} |"
            )

    f_x2_y2 = float(an_ccd_y2.loc["x2", "F"])
    f_x2sq_y2 = float(an_ccd_y2.loc["x2_sq", "F"])
    f_blk_y2 = float(an_ccd_y2.loc["C(block)", "F"])
    p_blk_y2 = float(an_ccd_y2.loc["C(block)", "PR(>F)"])
    ms_e_p1 = float(an_p1.loc["Residual", "sum_sq"] / an_p1.loc["Residual", "df"])
    ms_e_p2 = float(an_ccd_y1.loc["Residual", "sum_sq"] / an_ccd_y1.loc["Residual", "df"])
    f_blk_p1 = float(an_p1.loc["C(block)", "F"])
    p_blk_p1 = float(an_p1.loc["C(block)", "PR(>F)"])

    tot_ss_y1 = float(lof["Y1"]["SS_PE"] + lof["Y1"]["SS_LoF"])
    tot_ms_y1 = tot_ss_y1 / 121.0
    tot_rms_y1 = float(np.sqrt(tot_ms_y1))
    restr = lof["Y1_restricted"]
    tot_ss_r = float(restr["SS_PE"] + restr["SS_LoF"])
    tot_df_r = int(restr["df_PE"] + restr["df_LoF"])
    tot_ms_r = tot_ss_r / tot_df_r
    tot_rms_r = float(np.sqrt(tot_ms_r))
    tot_ss_y2 = float(lof["Y2"]["SS_PE"] + lof["Y2"]["SS_LoF"])
    tot_ms_y2 = tot_ss_y2 / 121.0

    p2_rows.extend(
        [
            "",
            f"For **Response $Y_2$ (Inference Latency, $\\mu\\text{{s/sample}}$)**, the second-order ANOVA shows that tree depth ($x_2$, $F = {f_x2_y2:.2f}, p < 10^{{-15}}$) and its quadratic term ($x_2^2$, $F = {f_x2sq_y2:.2f}, p < 0.0001$) govern inference latency, while block effects are negligible ($F = {f_blk_y2:.2f}, p = {p_blk_y2:.4f}, \\text{{ICC}} = {icc['Y2']['icc_anova']:.4f}$).",
            "",
            "### 3.2 Variance Shielding and Block Intraclass Correlation (ICC)",
            "",
            f"In Phase 1, unmodeled quadratic curvature inflated the residual mean square ($\\text{{MS}}_E = {ms_e_p1:.6f}$), masking block differences ($F = {f_blk_p1:.2f}, p = {p_blk_p1:.4f}$). In Phase 2, accounting for quadratic terms reduces $\\text{{MS}}_E$ by $40\\times$ to ${ms_e_p2:.6f}$, exposing highly significant seed-to-seed variance ($F = {f_block_ccd:.2f}, p < 0.0001$). The **Block Intraclass Correlation Coefficient** is:",
            f"$$\\text{{ICC}} = \\frac{{\\sigma^2_{{\\text{{block}}}}}}{{\\sigma^2_{{\\text{{block}}}} + \\sigma^2_\\epsilon}} = {icc['Y1']['icc_anova']:.4f} \\quad ({icc_anova_pct:.2f}\\%, \\quad \\text{{ICC}}_{{\\text{{REML}}}} = {icc_reml:.4f}, \\quad \\sigma_{{\\text{{block}}}} \\approx {sigma_block:.4f}\\text{{ RMSE}})$$",
            "",
            "### 3.3 Exact Multi-Scale Lack-of-Fit Decomposition and Residual Diagnostics",
            "",
            f"Partitioning the $121$ residual degrees of freedom into **Structural Lack of Fit** (${lof['Y1']['df_LoF']}\\text{{ df}}$), **Treatment $\\times$ Block Interaction** (${int(ld['df_interact'])}\\text{{ df}}$), and **Genuine Center Pure Error** (${int(ld['df_pure_center'])}\\text{{ df}}$) yields:",
            "",
            "| Source of Variation / Model | SS | DF | MS | $F$ | $p$-Value | Reference / RMS Misfit |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
            "| **Decomposition of Second-Order Residual ($Y_1$: Validation RMSE)** | | | | | | |",
            f"| $\\quad$ Structural Lack of Fit | ${lof['Y1']['SS_LoF']:.6f}$ | ${lof['Y1']['df_LoF']}$ | ${lof['Y1']['MS_LoF']:.6f}$ | ${ld['f_lof_center_y1']:.2f}$ | $< 10^{{-15}}$ | vs. Center PE |",
            f"| $\\quad$ Treatment $\\times$ Block Interaction | ${ld['ss_interact_y1']:.6f}$ | ${int(ld['df_interact'])}$ | ${_fmt_sci_md(ld['ms_interact_y1'])}$ | ${ld['f_interact_center_y1']:.2f}$ | ${ld['p_interact_center_y1']:.4f}$ | vs. Center PE |",
            f"| $\\quad$ Genuine Center Pure Error | ${ld['ss_pure_center_y1']:.6f}$ | ${int(ld['df_pure_center'])}$ | ${_fmt_sci_md(ld['ms_pure_center_y1'])}$ | — | — | Base Replicate |",
            f"| $\\quad$ Total Model Residual | ${tot_ss_y1:.6f}$ | $121$ | ${_fmt_sci_md(tot_ms_y1)}$ | — | — | $\\text{{RMS}} = {tot_rms_y1:.4f}$ |",
            f"| $\\quad$ Saturated Additive Baseline | ${lof['Y1']['SS_PE']:.6f}$ | ${lof['Y1']['df_PE']}$ | ${_fmt_sci_md(lof['Y1']['MS_PE'])}$ | ${lof['Y1']['F_LoF']:.2f}^*$ | $< 10^{{-15}}$ | ($^*$LoF vs. Additive, $p = {_fmt_sci_md(lof['Y1']['p_LoF'])}$, $\\text{{RMS}}_{{\\text{{LoF}}}} = {lof['Y1']['sqrt_MS_LoF']:.4f}$) |",
            "| **Restricted Domain ($Y_1$: $x_1 \\ge -0.5$, $\\eta \\ge 0.033$, $N=95$)** | | | | | | |",
            f"| $\\quad$ Structural Lack of Fit | ${restr['SS_LoF']:.6f}$ | ${restr['df_LoF']}$ | ${restr['MS_LoF']:.6f}$ | ${restr['F_LoF']:.2f}$ | ${restr['p_LoF']:.4f}$ | $\\text{{RMS}} = {restr['sqrt_MS_LoF']:.4f}$ (${restr['ss_lof_reduction_pct']:.1f}\\%$ SS reduction) |",
            f"| $\\quad$ Additive Baseline Residual | ${restr['SS_PE']:.6f}$ | ${restr['df_PE']}$ | ${_fmt_sci_md(restr['MS_PE'])}$ | — | — | Saturated Base |",
            f"| $\\quad$ Total Restricted Residual | ${tot_ss_r:.6f}$ | ${tot_df_r}$ | ${_fmt_sci_md(tot_ms_r)}$ | — | — | $\\text{{RMS}} = {tot_rms_r:.4f}$ |",
            "| **Latency Residual Decomposition ($Y_2$: Latency, $\\mu\\text{s}$)** | | | | | | |",
            f"| $\\quad$ Structural Lack of Fit | ${lof['Y2']['SS_LoF']:.2f}$ | ${lof['Y2']['df_LoF']}$ | ${lof['Y2']['MS_LoF']:.2f}$ | ${ld['f_lof_center_y2']:.2f}$ | ${ld['p_lof_center_y2']:.4f}$ | vs. Center PE ($\\text{{RMS}} = {lof['Y2']['sqrt_MS_LoF']:.2f}$) |",
            f"| $\\quad$ Treatment $\\times$ Block Interaction | ${ld['ss_interact_y2']:.2f}$ | ${int(ld['df_interact'])}$ | ${ld['ms_interact_y2']:.2f}$ | ${ld['f_interact_center_y2']:.2f}$ | ${ld['p_interact_center_y2']:.4f}$ | vs. Center PE |",
            f"| $\\quad$ Genuine Center Pure Error | ${ld['ss_pure_center_y2']:.2f}$ | ${int(ld['df_pure_center'])}$ | ${ld['ms_pure_center_y2']:.2f}$ | — | — | Base Replicate |",
            f"| $\\quad$ Total Model Residual | ${tot_ss_y2:.2f}$ | $121$ | ${tot_ms_y2:.2f}$ | ${lof['Y2']['F_LoF']:.2f}^*$ | ${lof['Y2']['p_LoF']:.4f}$ | ($^*$LoF vs. Additive) |",
            "",
            "Residual adequacy diagnostics for $Y_1$:",
            f"- **Normality**: Shapiro-Wilk $W = {diag['shapiro_W']:.4f}, p = {diag['shapiro_p']:.4f}$ (normal residuals).",
            f"- **Homoscedasticity**: Levene across blocks $W = {diag['levene_stat']:.4f}, p = {diag['levene_p']:.4f}$; Brown-Forsythe across design groups $W = {diag['brown_forsythe_points_stat']:.4f}, p = {diag['brown_forsythe_points_p']:.4f}$; Breusch-Pagan against fitted values $\\text{{LM}} = {diag['breusch_pagan_stat']:.2f}, p < 0.0001$ (reflecting multi-scale variance across the factor domain, addressed via HC3 robust standard errors).",
            f"- **Independence & Influence**: Durbin-Watson $DW = {diag['durbin_watson']:.4f}$, Ljung-Box $Q = {diag['ljung_box_stat']:.2f}, p = {diag['ljung_box_p']:.4f}$, Runs test $p = {diag['runs_test_p']:.4f}$, maximum Cook's distance $D_{{\\max}} = {diag['max_cooks_d']:.3f} < 1.0$.",
        ]
    )
    sections["SECTION_3_PHASE2"] = "\n".join(p2_rows)

    # 5. SECTION_4_PHASE3
    b_vec = p3["b"]
    st_pt = p3["stationary_point_coded"]
    p3_rows = [
        f"Writing the second-order validation RMSE surface as $\\hat{{y}}(\\mathbf{{x}}) = b_0 + \\mathbf{{x}}^T \\mathbf{{b}} + \\mathbf{{x}}^T \\hat{{\\mathbf{{B}}}} \\mathbf{{x}}$ gives $b_0 = {p3['b0']:.4f}$ and linear gradient $\\mathbf{{b}} = [{b_vec[0]:.4f}, {b_vec[1]:.4f}, {b_vec[2]:.4f}, {b_vec[3]:.4f}]^T$. Solving $\\mathbf{{x}}_0 = -\\frac{{1}}{{2}}\\hat{{\\mathbf{{B}}}}^{{-1}}\\mathbf{{b}}$ yields an unconstrained stationary point at $\\mathbf{{x}}_0 = [{st_pt[0]:.4f}, {st_pt[1]:.4f}, {st_pt[2]:.4f}, {st_pt[3]:.4f}]^T$ ($\\|\\mathbf{{x}}_0\\|_2 = {p3['distance_coded']:.2f}$, $\\hat{{y}}_0 = {p3['y0_hat_unconstrained']:.4f}$), which lies far outside $[-1, +1]^4$ due to flat regularization and subsampling curvature.",
        "",
        "Spectral decomposition $\\hat{\\mathbf{B}} = \\mathbf{V}\\bm{\\Lambda}\\mathbf{V}^T$ yields eigenvalues:",
        f"$$\\lambda_1 = {eigs[0]:.6f}, \\quad \\lambda_2 = {eigs[1]:.6f}, \\quad \\lambda_3 = {eigs[2]:.6f}, \\quad \\lambda_4 = {eigs[3]:.6f} \\quad (\\text{{trace}}(\\hat{{\\mathbf{{B}}}}) = {p3['trace_B']:.6f})$$",
        f"Rademacher wild bootstrap (2,000 replications) places the 95% confidence interval for $\\lambda_1$ at $[{boot_p[0][0]:.4f}, {boot_p[2][0]:+.4f}]$, with ${p3['fraction_min_eigenvalue_le_zero']*100:.1f}\\%$ of bootstrap resamples yielding $\\min(\\lambda) \\le 0$. Because the confidence interval for $\\lambda_1$ straddles zero, the response surface forms a **stationary/rising ridge system** along the $L_2$ regularization ($x_4$) and subsample ($x_3$) axes, while learning rate ($x_1$) and tree depth ($x_2$) exhibit steep positive convexity.",
        "",
        "Constrained optimization within $\\mathcal{D} = [-1, +1]^4$ restricted to valid integer depths $d \\in \\{3, \\dots, 9\\}$ yields:",
        "",
        "| Depth | Coded $x_1$ | Coded $x_2$ | Coded $x_3$ | Coded $x_4$ | Optimal $\\eta$ | Predicted RMSE | $\\text{SE}(\\hat{y})$ |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in ds.depth_opt_df.iterrows():
        d_int = int(row["depth"])
        x2_c = (d_int - 6) / 3.0
        eta_v = decode_factors(np.array([row["x1"], x2_c, row["x3"], row["x4"]]))[0]
        if d_int == int(cube["depth"]):
            p3_rows.append(
                f"| **${d_int}$** | **${row['x1']:.4f}$** | **${x2_c:+.4f}$** | **${row['x3']:.2f}$** | **${row['x4']:.2f}$** | **${eta_v:.4f}$** | **${row['pred_rmse']:.4f}$** | **${row['se']:.4f}$** |"
            )
        else:
            p3_rows.append(
                f"| ${d_int}$ | ${row['x1']:.4f}$ | ${x2_c:+.4f}$ | ${row['x3']:.2f}$ | ${row['x4']:.2f}$ | ${eta_v:.4f}$ | ${row['pred_rmse']:.4f}$ | ${row['se']:.4f}$ |"
            )
    p3_rows.extend(
        [
            "",
            f"The **Single-Objective RSM Candidate ($\\mathbf{{x}}^*_{{\\text{{SO}}}}$)** occurs at **depth $d = {cube['depth']}$** ($[{cube_x[0]:.4f}, {cube_x[1]:.4f}, {cube_x[2]:.4f}, {cube_x[3]:.4f}]^T \\implies \\eta = {cube_nat['eta']:.4f}, d = {cube['depth']}, s = {cube_nat['subsample']:.4f}, \\lambda = {cube_nat['reg_lambda']:.4f}$), with predicted validation RMSE $\\hat{{y}}(\\mathbf{{x}}^*_{{\\text{{SO}}}}) = {cube['pred_rmse']:.4f}$ and Satterthwaite 95% PI $[{so_conf['prediction_interval_95_val'][0]:.4f}, {so_conf['prediction_interval_95_val'][1]:.4f}]$.",
        ]
    )
    sections["SECTION_4_PHASE3"] = "\n".join(p3_rows)

    # 6. SECTION_5_PHASE4
    sections["SECTION_5_PHASE4"] = "\n".join(
        [
            "To simultaneously minimize **Validation RMSE ($Y_1$)** and **Single-Sample Inference Latency ($Y_2$)**, we apply one-sided Derringer-Suich transformations ($s_1 = s_2 = 1.0$, equal weights $w_1 = w_2 = 1.0$) with operational specification bounds $Y_1 \\in [L_1, U_1] = [0.450, 0.700]$ and $Y_2 \\in [L_2, U_2] = [100.0, 180.0]\\,\\mu\\text{s}$, maximizing $D(\\mathbf{x}) = \\sqrt{d_1(\\hat{Y}_1(\\mathbf{x})) \\cdot d_2(\\hat{Y}_2(\\mathbf{x}))}$ over valid integer depths:",
            "",
            f"- **Coded Compromise Coordinate ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$)**: $[{mo_coded[0]:.4f}, \\ {mo_coded[1]:.4f}, \\ {mo_coded[2]:.4f}, \\ {mo_coded[3]:.4f}]^T$",
            f"- **Natural Hyperparameters**: `learning_rate` $\\eta = {mo_nat['eta']:.4f}$, `max_depth` $d = {mo_nat['depth']}$, `subsample` $s = {mo_nat['subsample']:.4f}$, `reg_lambda` $\\lambda = {mo_nat['reg_lambda']:.4f}$",
            f"- **Surrogate Predictions**: $\\hat{{Y}}_1 = {d_star['pred_y1']:.4f}$ Validation RMSE, $\\hat{{Y}}_2 = {d_star['pred_y2']:.2f}\\,\\mu\\text{{s}}$ latency",
            f"- **Individual & Composite Desirabilities**: $d_1 = {d_star['d1']:.4f}$, $d_2 = {d_star['d2']:.4f}$, **$D = {d_star['D']:.4f}$**",
        ]
    )

    # 7. SECTION_6_PHASE5
    sections["SECTION_6_PHASE5"] = "\n".join(
        [
            f"Both DOE candidate coordinates ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$ at depth {mo_nat['depth']} and $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ at depth {cube['depth']}) were evaluated across **$m = {n_conf}$ fresh random seeds** ($\\mathcal{{S}}_{{\\text{{conf}}}} = \\{{{conf_seeds_str}\\}}$, matching `config.yaml` and `results/confirmation_runs.csv`) against Satterthwaite-adjusted 95% prediction intervals ($\\nu_{{\\text{{eff}}}} = {mo_val['nu_eff']:.1f}$ for $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ at $h_0 = {conf['leverage_h']:.4f}$; $\\nu_{{\\text{{eff}}}} = {so_conf['nu_eff']:.1f}$ for $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ at $h_0 = {so_conf['leverage_h']:.4f}$):",
            "",
            f"| Configuration / Response Metric | Surrogate Pred ($\\hat{{y}}$) | 95% Pred Interval (PI) | Empirical Mean $\\pm$ SD ($m = {n_conf}$) | Confirmation Status |",
            "| :--- | :---: | :---: | :---: | :--- |",
            f"| **DOE Multi-Objective Optimum $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ (Depth {mo_nat['depth']})** | | | | |",
            f"| Validation RMSE ($Y_1$) | ${mo_val['predicted_mean']:.4f}$ | $[{mo_val['prediction_interval_95'][0]:.4f}, {mo_val['prediction_interval_95'][1]:.4f}]$ | ${mo_val['empirical_mean']:.4f} \\pm {mo_val['empirical_std']:.4f}$ | **Pass** (Inside 95% PI) |",
            f"| Holdout Test RMSE | — | — | ${mo_test['empirical_mean']:.4f} \\pm {mo_test['empirical_std']:.4f}$ | Holdout Test Set |",
            f"| Inference Latency ($\\mu\\text{{s}}$) | ${mo_lat['predicted_mean']:.1f}$ (${mo_lat['predicted_mean']:.2f}$) | $[{mo_lat['prediction_interval_95'][0]:.1f}, {mo_lat['prediction_interval_95'][1]:.1f}]$ ($[{mo_lat['prediction_interval_95'][0]:.2f}, {mo_lat['prediction_interval_95'][1]:.2f}]$) | ${mo_lat['empirical_mean']:.1f} \\pm {mo_lat['empirical_std']:.1f}$ (${mo_lat['empirical_mean']:.2f} \\pm {mo_lat['empirical_std']:.2f}$) | **Pass** (Inside 95% PI) |",
            f"| **DOE Single-Objective Candidate $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ (Depth {cube['depth']})** | | | | |",
            f"| Validation RMSE ($Y_1$) | ${so_conf['predicted_val_rmse']:.4f}$ | $[{so_conf['prediction_interval_95_val'][0]:.4f}, {so_conf['prediction_interval_95_val'][1]:.4f}]$ | ${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$ | **Not Confirmed** (Optimism: ${so_bias:+.4f}$) |",
            f"| Holdout Test RMSE | — | — | ${so_conf['empirical_test_rmse']:.4f} \\pm {so_conf['empirical_test_std']:.4f}$ | Holdout Test Set |",
            f"| Inference Latency ($\\mu\\text{{s}}$) | ${so_conf['predicted_latency']:.1f}$ (${so_conf['predicted_latency']:.2f}$) | $[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]$ ($[{so_conf['prediction_interval_95_lat'][0]:.2f}, {so_conf['prediction_interval_95_lat'][1]:.2f}]$) | ${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}$ (${so_conf['empirical_latency']:.2f} \\pm {so_conf['empirical_latency_std']:.2f}$) | **Borderline** (At Lower PI) |",
            "",
            f"At $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ (depth {mo_nat['depth']}), both validation RMSE (${mo_val['empirical_mean']:.4f} \\pm {mo_val['empirical_std']:.4f}$) and single-sample inference latency (${mo_lat['empirical_mean']:.2f} \\pm {mo_lat['empirical_std']:.2f}\\,\\mu\\text{{s}}$) fall directly inside their Satterthwaite 95% prediction intervals. At $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ (depth {cube['depth']}), empirical validation RMSE (${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$) lies ${so_above_pi:.4f}$ above the upper prediction interval bound (${so_conf['prediction_interval_95_val'][1]:.4f}$), confirming that quadratic interpolation across depths $\\{{3, 6, 9\\}}$ overestimates accuracy gains at depth {cube['depth']} by ${so_bias:+.4f}$ RMSE ($6.8\\times \\text{{SE}}(\\hat{{y}})$), while empirical latency (${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}\\,\\mu\\text{{s}}$) sits on the lower bound of its 95% prediction interval ($[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]\\,\\mu\\text{{s}}$).",
        ]
    )

    # 8. SECTION_7_BENCHMARKS
    s7_rows = [
        f"To evaluate the DOE + RSM methodology against modern heuristic and Bayesian optimizers under strictly equal evaluation budgets (140 model evaluations per search), we executed a prospective benchmark campaign (`results/revision_v2/full_run_001/`) comprising **{ds.total_model_fits:,} XGBoost model fits** and **{ds.total_timed_inferences:,} timed single-sample inferences** across $N = 20$ independent search replicates per optimizer.",
        "",
        f"All **{ds.total_selection_records} winning selection records** (representing **{ds.distinct_config_hashes} distinct hyperparameter configurations** by SHA-256 hash and yielding **{ds.total_evaluation_rows:,} final evaluation rows** in `final_evaluations.csv`) were cryptographically frozen in `finalized_selections.json` prior to generalization testing across {len(ds.fresh_eval_seeds)} fresh retraining seeds ($\\mathcal{{S}}_{{\\text{{eval}}}} = \\{{{ds.fresh_eval_seeds[0]}, {ds.fresh_eval_seeds[1]}, \\dots, {ds.fresh_eval_seeds[-1]}\\}}$) on the external holdout test set ($N = 4,128$). Whereas the historical `v1.0.0` baseline logged holdout test metrics during exploratory development, the `Revision-v2` pipeline programmatically prevented access to holdout test labels until `finalized_selections.json` was frozen (enforcing pipeline isolation during the revised search, while recognizing that the same dataset was previously evaluated in the historical baseline).",
        "",
        "### 7.1 Holdout Generalization and Latency Comparison",
        "",
        "| Optimization Method | Search Basis ($N$ Reps) | Val RMSE (Mean $\\pm$ SD) | Holdout Test RMSE (Mean $\\pm$ SD) | Holdout Test RMSE [95% CI] | Retrain $\\sigma_{\\text{eval}}$ | `predict` Latency ($\\mu\\text{s} \\pm \\text{SD}$) | `inplace_predict` ($\\mu\\text{s} \\pm \\text{SD}$) | Feasible ($\\le 145\\,\\mu\\text{s}$) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        "| **Panel A: Prospective Revision-v2 Full Benchmark ($N = 20$ Search Replicates)** | | | | | | | | |",
    ]
    for opt_key in PROSPECTIVE_OPTIMIZER_ORDER:
        m = opts[opt_key]
        s7_rows.append(
            f"| **{m.display_name_md}** | {m.search_basis_md} | ${m.val_rmse_mean:.5f} \\pm {m.val_rmse_sd:.5f}$ (${m.val_rmse_mean:.4f} \\pm {m.val_rmse_sd:.4f}$) | ${m.test_rmse_mean:.5f} \\pm {m.test_rmse_sd:.5f}$ (${m.test_rmse_mean:.4f} \\pm {m.test_rmse_sd:.4f}$) | $[{m.test_rmse_ci_low:.5f}, {m.test_rmse_ci_high:.5f}]$ | ${m.retrain_sd:.5f}$ (${m.retrain_sd:.4f}$) | ${m.predict_latency_mean:.2f} \\pm {m.predict_latency_sd:.2f}$ (${m.predict_latency_mean:.1f} \\pm {m.predict_latency_sd:.1f}$) | ${m.inplace_latency_mean:.2f} \\pm {m.inplace_latency_sd:.2f}$ | {m.benchmark_feasible_count}/{m.n_replicates} ({m.benchmark_feasible_pct}%) |"
        )
    s7_rows.append("| **Panel B: Historical Baseline Snapshot (`v1.0.0`, Single Search Replicate)** | | | | | | | | |")
    for hrow in ds.historical_benchmarks:
        inp_s = "—" if hrow.inplace_latency_us_median is None else f"${hrow.inplace_latency_us_median:.1f}$"
        feas_s = "Yes" if hrow.feasible_145 else "No"
        s7_rows.append(
            f"| **{hrow.display_name_md}** | {hrow.search_basis_md} | ${hrow.val_rmse_mean:.4f}$ | ${hrow.test_rmse_mean:.4f}$ | — | — | ${hrow.predict_latency_us_median:.1f}$ | {inp_s} | {feas_s} |"
        )

    s7_rows.extend(
        [
            "",
            "### 7.2 Multi-Objective Pareto Hypervolume Comparison",
            "",
            f"| Frontier / Evaluation Protocol | Candidate Pool ($N$) | Non-Dom. Points | HV at $[{r1_rmse:.2f}, {r1_lat:.0f}]$ (Mean $\\pm$ SD) | HV at $[{r2_rmse:.2f}, {r2_lat:.0f}]$ (Mean $\\pm$ SD) |",
            "| :--- | :---: | :---: | :---: | :---: |",
            "| **Panel A: Development Candidate Frontiers (Distinct Evaluation Protocols Noted per Row)** | | | | |",
            f"| **Multi-Objective TPE Candidates (Single Split 42, 30-Call Search)** | $140 \\times 20$ | ${hvs['motpe_nd_mean']:.2f} \\pm {hvs['motpe_nd_sd']:.1f}$ | ${hvs['motpe_dev_060_mean']:.4f} \\pm {hvs['motpe_dev_060_sd']:.4f}$ | ${hvs['motpe_dev_065_mean']:.4f} \\pm {hvs['motpe_dev_065_sd']:.4f}$ |",
            f"| **Repeated DOE Candidate Fronts (5-Block Means Across Replicates)** | $25 \\times 20$ | ${hvs['rdoe_nd_mean']:.2f} \\pm {hvs['rdoe_nd_sd']:.1f}$ | ${hvs['rdoe_dev_060_mean']:.4f} \\pm {hvs['rdoe_dev_060_sd']:.4f}$ | ${hvs['rdoe_dev_065_mean']:.4f} \\pm {hvs['rdoe_dev_065_sd']:.4f}$ |",
            f"| **Fixed Full DOE Evaluated Front (Single Split 42, 27 Configs)** | $27$ | ${hvs['fdoe_nd']}$ | ${hvs['fdoe_dev_060']:.4f}$ | ${hvs['fdoe_dev_065']:.4f}$ |",
            f"| **Historical DOE 2-Point Set (Single Split 42, $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ & $\\mathbf{{x}}^*_{{\\text{{SO}}}}$)** | $2$ | ${hvs['hdoe_nd']}$ | ${hvs['hdoe_dev_060']:.4f}$ | ${hvs['hdoe_dev_065']:.4f}$ |",
            f"| **Difference: Fixed Full DOE Front $-$ MO-TPE Mean (Split 42)** | — | — | ${hvs['diff_fdoe_minus_motpe_060']:+.4f}$ ($p < 0.0001$) | ${hvs['diff_fdoe_minus_motpe_065']:+.4f}$ ($p < 0.0001$) |",
            f"| **Panel B: Holdout Non-Dominated Set ({ds.total_selection_records} Records, {ds.distinct_config_hashes} Configs, {len(ds.fresh_eval_seeds)} Seeds)** | | | | |",
            f"| **Non-Dominated Set ({hvs['holdout_nd']} Distinct Configs)** | ${ds.total_selection_records}$ | ${hvs['holdout_nd']}$ | ${hvs['holdout_frozen_060']:.4f}$ | ${hvs['holdout_frozen_065']:.4f}$ |",
            f"| $\\quad \\llcorner$ Repeated DOE MO Selections | $20$ | ${nd_by_opt['repeated_preplanned_doe_multi_objective']}$ | — | — |",
            f"| $\\quad \\llcorner$ Multi-Objective TPE Selections | $20$ | ${nd_by_opt['multi_objective_tpe']}$ | — | — |",
            f"| $\\quad \\llcorner$ Constrained TPE Selections | $20$ | ${nd_by_opt['constrained_tpe']}$ | — | — |",
            f"| $\\quad \\llcorner$ Single-Objective TPE Selections | $20$ | ${nd_by_opt['single_objective_tpe']}$ | — | — |",
            "",
            "### 7.3 Key Scientific Conclusions from the Benchmark Campaign",
            "",
            f"1. **Single-Objective Accuracy vs. Latency**: Single-Objective TPE achieves the numerically lowest observed mean holdout test RMSE (${so_tpe.test_rmse_mean:.5f} \\pm {so_tpe.test_rmse_sd:.5f}$) by concentrating searches in deeper trees ($d \\in \\{{7, 8, 9\\}}$), incurring ${so_tpe.predict_latency_mean:.2f} \\pm {so_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ inference latency (`inplace_predict`: ${so_tpe.inplace_latency_mean:.2f} \\pm {so_tpe.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$). Neither Single-Objective TPE nor Repeated DOE Single-Objective satisfies the $\\le 145\\,\\mu\\text{{s}}$ latency constraint ($0/20$ feasible). Repeated DOE Single-Objective ($\\mathbf{{x}}^*_{{\\text{{SO}}}}$, depth {cube['depth']}) selects the exact same configuration deterministically across all 20 replicates (between-search $\\text{{SD}} = {doe_so.test_rmse_sd:.5f}$, retraining $\\sigma_{{\\text{{eval}}}} = {doe_so.retrain_sd:.5f}$), achieving ${doe_so.test_rmse_mean:.5f}$ Test RMSE at ${doe_so.predict_latency_mean:.2f} \\pm {doe_so.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ inference latency (`inplace_predict`: ${doe_so.inplace_latency_mean:.2f} \\pm {doe_so.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$, ${ds.doe_so_vs_sotpe_latency_reduction_pct:.1f}\\%$ lower latency than SO-TPE).",
            f"2. **Multi-Objective Performance Comparison**: In the latency-constrained multi-objective regime, Repeated DOE ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$, depth {mo_nat['depth']}) achieves a mean Test RMSE of ${doe_mo.test_rmse_mean:.5f} \\pm {doe_mo.test_rmse_sd:.5f}$ at ${doe_mo.predict_latency_mean:.2f} \\pm {doe_mo.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ latency (`inplace_predict`: ${doe_mo.inplace_latency_mean:.2f} \\pm {doe_mo.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$), compared to ${mo_tpe.test_rmse_mean:.5f} \\pm {mo_tpe.test_rmse_sd:.5f}$ at ${mo_tpe.predict_latency_mean:.2f} \\pm {mo_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ (`inplace_predict`: ${mo_tpe.inplace_latency_mean:.2f} \\pm {mo_tpe.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$) for Multi-Objective TPE. The difference of ${ds.doe_mo_vs_motpe_diff:+.5f}$ RMSE is statistically non-significant under both Welch's two-sample $t$-test ($t = {ds.doe_mo_vs_motpe_welch_t:.4f}, p = {ds.doe_mo_vs_motpe_welch_p:.4f}$) and paired $t$-testing across search replicates ($t = {ds.doe_mo_vs_motpe_paired_t:.3f}, p = {ds.doe_mo_vs_motpe_paired_p:.4f}$). Both methods achieve $20/20$ ($100\\%$) benchmark constraint feasibility.",
            f"3. **Development Hypervolume vs. Holdout Non-Dominated Set**: On development split 42 under the online 30-call timing protocol, MO-TPE candidate fronts achieve a mean hypervolume of ${hvs['motpe_dev_060_mean']:.4f} \\pm {hvs['motpe_dev_060_sd']:.4f}$ at reference $[{r1_rmse:.2f}, {r1_lat:.1f}]$ (${hvs['motpe_dev_065_mean']:.4f} \\pm {hvs['motpe_dev_065_sd']:.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.1f}]$), exceeding the fixed 27-point DOE candidate frontier (${hvs['fdoe_dev_060']:.4f}$, difference ${hvs['diff_fdoe_minus_motpe_060']:+.4f}, p < 0.0001$; at $[{r2_rmse:.2f}, {r2_lat:.1f}]$, ${hvs['fdoe_dev_065']:.4f}$, difference ${hvs['diff_fdoe_minus_motpe_065']:+.4f}, p < 0.0001$), while Repeated DOE 5-block means achieve ${hvs['rdoe_dev_060_mean']:.4f} \\pm {hvs['rdoe_dev_060_sd']:.4f}$ (${hvs['rdoe_dev_065_mean']:.4f} \\pm {hvs['rdoe_dev_065_sd']:.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.1f}]$). This demonstrates a higher observed candidate hypervolume on the evaluated development split under 30-call search timing, without implying universal algorithmic superiority across unconstrained domains or unseen splits. When the {ds.total_selection_records} frozen selection records ({ds.distinct_config_hashes} distinct configurations) are evaluated on the external holdout test set across {len(ds.fresh_eval_seeds)} retraining seeds, the empirical non-dominated set among the evaluated frozen configurations consists of **{hvs['holdout_nd']} distinct configurations** ($\\text{{HV}} = {hvs['holdout_frozen_060']:.4f}$ at $[{r1_rmse:.2f}, {r1_lat:.1f}]$ and ${hvs['holdout_frozen_065']:.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.1f}]$) spanning both paradigms: **{nd_by_opt['repeated_preplanned_doe_multi_objective']} Repeated DOE MO, {nd_by_opt['multi_objective_tpe']} MO-TPE, {nd_by_opt['constrained_tpe']} Constrained TPE, and {nd_by_opt['single_objective_tpe']} SO-TPE**.",
            f"4. **Constrained TPE Feasibility Degradation**: While Constrained TPE satisfies $\\le 145\\,\\mu\\text{{s}}$ on ${ctpe.search_feasible_count}/{ctpe.n_replicates}$ searches during online 30-call search timing, only **${ctpe.benchmark_feasible_count}/{ctpe.n_replicates}$ (${ctpe.benchmark_feasible_pct}\\%$)** remain feasible under 1,000-call benchmark verification (${ds.ctpe_depth6_feasible}/{ds.ctpe_depth6_count}$ depth-6 selections feasible at ${ds.ctpe_depth6_latency_mean:.2f} \\pm {ds.ctpe_depth6_latency_sd:.2f}\\,\\mu\\text{{s}}$; ${ds.ctpe_depth7_feasible}/{ds.ctpe_depth7_count}$ depth-7 selections feasible at ${ds.ctpe_depth7_latency_mean:.2f} \\pm {ds.ctpe_depth7_latency_sd:.2f}\\,\\mu\\text{{s}}$). Plausible factors include noisy 30-call search measurements near the boundary, protocol differences (30 calls without warmup vs. 1,000 calls with warmup), and selection bias, rather than a single conclusively isolated cause.",
            "5. **Historical Latency Comparison Caveat**: Both historical `v1.0.0` scripts (`phase5_confirmation.py` and `phase5_benchmarks.py`) applied Win32 CPU core 0 affinity pinning. Consequently, the shift between historical single-run latency snapshots and the prospective benchmark session cannot be conclusively attributed to unpinned execution or a single identified environmental factor.",
        ]
    )
    sections["SECTION_7_BENCHMARKS"] = "\n".join(s7_rows)

    # 9. SECTION_8_9_DISCUSSION_AND_ARTIFACTS
    macro_count = sum(
        1 for line in render_macros(ds).splitlines() if line.startswith("\\newcommand")
    )
    sections["SECTION_8_9_DISCUSSION_AND_ARTIFACTS"] = "\n".join(
        [
            f"1. **Parametric Attribution vs. Adaptive Search**: DOE + RSM decomposes variance across main effects, interactions, quadratic terms, and seed blocks, and provides formal hypothesis tests for curvature ($F = {p1['f_curvature']:,.2f}$) and lack of fit ($F = {lof['Y1']['F_LoF']:.2f}$). Conversely, Bayesian optimization (TPE) adapts dynamically to non-polynomial basins without parametric assumptions.",
            f"2. **Value of Stochastic Nuisance Blocking**: Blocking across {len(ds.block_seeds)} data-partition seeds absorbed $\\text{{ICC}} = {icc_anova_pct:.2f}\\%$ of residual variance ($\\sigma_{{\\text{{block}}}} \\approx {sigma_block:.4f}$ RMSE), preventing seed noise from confounding hyperparameter comparisons.",
            f"3. **Structural Lack of Fit and Surrogate Optimism**: Because decision tree ensembles exhibit diminishing returns at deeper levels ($d \\ge 6$), a second-order polynomial interpolated across $d \\in \\{{3, 6, 9\\}}$ under-predicts validation RMSE at depth {cube['depth']} by ${so_bias:+.4f}$ RMSE. Satterthwaite prediction intervals widen for variance heterogeneity across degrees of freedom but cannot correct deterministic polynomial bias.",
            "4. **Dimensionality Scaling**: While a 4-factor FCCD requires $2^4 + 2(4) + 4 = 28$ runs per block, full factorials scale as $2^k$. For higher-dimensional spaces ($k > 6$), Resolution IV/V fractional factorials ($2^{k-p}$), Box-Behnken designs, or hybrid DOE-screening + Bayesian refinement pipelines are recommended.",
            "",
            "---",
            "",
            "## 9. Repository Artifacts and Reproducibility",
            "",
            "- **LaTeX Manuscript & Compiled PDF**: `report.tex` and `report.pdf` (22 pages, compiled via Tectonic with zero unresolved references or layout overflows).",
            f"- **Auto-Generated Statistical Macros & Tables**: `results/macros.tex` ({macro_count} macros) and `tables/*.tex`, generated deterministically via `python scripts/generate_report_artifacts.py`.",
            "- **Single-Source-of-Truth Reporting & Manifest**: `scripts/reporting_data.py`, `scripts/generate_research_reporting.py`, and `docs/generated/scientific_results_manifest.json`.",
            "- **Prospective Benchmark Evidence**: `results/revision_v2/full_run_001/` (`finalized_selections.json`, `final_evaluations.csv`, `final_summary.csv`, `optimizer_summary.json`, `hypervolume.json`, `paired_comparisons.json`, `computational_budget.json`, `run_manifest.json`).",
            "- **Verification & Audit Suite**: `python scripts/audit_scientific_consistency.py`, `python scripts/generate_research_reporting.py --check`, `python scripts/generate_report_artifacts.py --check`, and `pytest` (automated integrity, statistical, and numerical table consistency checks).",
        ]
    )

    return sections


def render_report_markdown(ds: Optional[ReportingDataset] = None) -> str:
    """Render the complete REPORT.md document with explicit auto-generated block markers."""
    if ds is None:
        ds = load_reporting_dataset()
    sec = render_report_sections(ds)

    def _wrap(sid: str) -> str:
        return f"{_begin_marker(sid)}\n{sec[sid]}\n{_end_marker(sid)}"

    lines = [
        "# Sequential Response Surface Methodology and Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking",
        "",
        "**Author:** Christos Chousein Sounios",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        _wrap("EXEC_SUMMARY"),
        "",
        "---",
        "",
        "## 1. Experimental Factors, Coding, and Blocking Architecture",
        "",
        _wrap("SECTION_1_FACTORS_AND_BLOCKS"),
        "",
        "---",
        "",
        "## 2. Phase 1: $2^4$ Factorial Screening and Curvature Test",
        "",
        _wrap("SECTION_2_PHASE1"),
        "",
        "---",
        "",
        "## 3. Phase 2: Face-Centered Central Composite Design (FCCD) and Model Adequacy",
        "",
        _wrap("SECTION_3_PHASE2"),
        "",
        "---",
        "",
        "## 4. Phase 3: Canonical Spectral Analysis and Ridge Optimization",
        "",
        _wrap("SECTION_4_PHASE3"),
        "",
        "---",
        "",
        "## 5. Phase 4: Multi-Objective Derringer-Suich Desirability",
        "",
        _wrap("SECTION_5_PHASE4"),
        "",
        "---",
        "",
        "## 6. Phase 5: Empirical Confirmation Trials ($m = 10$ Fresh Seeds)",
        "",
        _wrap("SECTION_6_PHASE5"),
        "",
        "---",
        "",
        "## 7. Prospective Multi-Replicate Benchmark Campaign (`Revision-v2`)",
        "",
        _wrap("SECTION_7_BENCHMARKS"),
        "",
        "---",
        "",
        "## 8. Discussion and Methodological Limitations",
        "",
        _wrap("SECTION_8_9_DISCUSSION_AND_ARTIFACTS"),
        "",
    ]
    return "\n".join(lines)


def _extract_marked_sections(md_text: str) -> Tuple[Dict[str, str], str]:
    """Extract all auto-generated blocks and return (blocks_dict, text_outside_blocks)."""
    found: Dict[str, str] = {}
    outside_text = md_text
    for sid in SECTION_IDS:
        b_tag = _begin_marker(sid)
        e_tag = _end_marker(sid)
        b_count = md_text.count(b_tag)
        e_count = md_text.count(e_tag)
        if b_count != 1 or e_count != 1:
            raise ReportingDataError(
                f"REPORT.md missing or duplicate markers for section '{sid}' "
                f"(begin_count={b_count}, end_count={e_count})"
            )
        b_idx = md_text.index(b_tag)
        e_idx = md_text.index(e_tag)
        if e_idx <= b_idx:
            raise ReportingDataError(f"REPORT.md end marker precedes begin marker for '{sid}'")
        inner = md_text[b_idx + len(b_tag) : e_idx].strip("\n")
        found[sid] = inner
        outside_text = outside_text.replace(md_text[b_idx : e_idx + len(e_tag)], "")
    return found, outside_text


def audit_manuscript_integrity(ds: Optional[ReportingDataset] = None) -> List[str]:
    """Perform manuscript-level numerical and methodological auditing across REPORT.md and report.tex (Task 9)."""
    if ds is None:
        ds = load_reporting_dataset()

    errors: List[str] = []
    report_md_path = ds.root_dir / "REPORT.md"
    report_tex_path = ds.root_dir / "report.tex"
    macros_tex_path = ds.root_dir / "results" / "macros.tex"
    manifest_path = ds.root_dir / "docs" / "generated" / "scientific_results_manifest.json"

    for p in (report_md_path, report_tex_path, macros_tex_path, manifest_path):
        if not p.is_file():
            errors.append(f"Missing required document/manifest: {p.relative_to(ds.root_dir)}")
    if errors:
        return errors

    md_text = report_md_path.read_text(encoding="utf-8")
    tex_text = report_tex_path.read_text(encoding="utf-8")
    macros_text = macros_tex_path.read_text(encoding="utf-8")

    # 1. Verify REPORT.md markers and outside-block text
    try:
        actual_blocks, outside_md = _extract_marked_sections(md_text)
        expected_blocks = render_report_sections(ds)
        for sid in SECTION_IDS:
            if actual_blocks[sid] != expected_blocks[sid]:
                diff = "".join(
                    difflib.unified_diff(
                        actual_blocks[sid].splitlines(keepends=True),
                        expected_blocks[sid].splitlines(keepends=True),
                        fromfile=f"REPORT.md:{sid} (on disk)",
                        tofile=f"REPORT.md:{sid} (expected)",
                    )
                )
                errors.append(f"Stale or manually edited auto-generated section '{sid}' in REPORT.md:\n{diff}")
        # Ensure no floating-point empirical values appear outside auto-generated blocks in REPORT.md
        outside_floats = re.findall(r"\b\d+\.\d+\b", outside_md)
        if outside_floats:
            errors.append(
                f"Untraced floating-point numbers found outside auto-generated sections in REPORT.md: {outside_floats}"
            )
    except ReportingDataError as exc:
        errors.append(str(exc))

    # 2. Verify confirmation seeds in REPORT.md and report.tex match config.yaml
    expected_conf_str = ", ".join(str(s) for s in ds.confirmation_seeds)
    if expected_conf_str not in md_text:
        errors.append(
            f"Confirmation seeds '{expected_conf_str}' not found in REPORT.md"
        )
    if expected_conf_str not in tex_text:
        errors.append(
            f"Confirmation seeds '{expected_conf_str}' not found in report.tex"
        )

    # 3. Verify hypervolume reference points in REPORT.md and report.tex
    r1_rmse, r1_lat = ds.hv_ref_primary
    r2_rmse, r2_lat = ds.hv_ref_secondary
    for doc_name, doc_content in [("REPORT.md", md_text), ("report.tex", tex_text)]:
        if f"[{r1_rmse:.2f}, {r1_lat:.0f}]" not in doc_content and f"[{r1_rmse:.2f}, {r1_lat:.1f}]" not in doc_content:
            errors.append(f"Primary hypervolume reference point [{r1_rmse:.2f}, {r1_lat:.1f}] missing from {doc_name}")
        if f"[{r2_rmse:.2f}, {r2_lat:.0f}]" not in doc_content and f"[{r2_rmse:.2f}, {r2_lat:.1f}]" not in doc_content:
            errors.append(f"Secondary hypervolume reference point [{r2_rmse:.2f}, {r2_lat:.1f}] missing from {doc_name}")

    # 4. Verify all \num... and \vecB... and \matB... macros used in report.tex are defined in results/macros.tex
    defined_macros = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+)\}", macros_text)
    )
    used_macros = set(re.findall(r"\\((?:num|vecB|matB|env)[A-Za-z]+)\b", tex_text))
    undefined = sorted(used_macros - set(defined_macros.keys()))
    if undefined:
        errors.append(f"Macros used in report.tex but undefined in results/macros.tex: {undefined}")

    # 5. Check that no unsupported TOST equivalence claims appear when rmse_equivalence_margin is None
    if ds.config["revision_v2"].get("rmse_equivalence_margin") is None:
        forbidden_equiv = re.compile(r"\b(?:TOST\s+confirmed|statistically\s+equivalent|formal\s+equivalence\s+established)\b", re.IGNORECASE)
        for doc_name, doc_content in [("REPORT.md", md_text), ("report.tex", tex_text)]:
            m_eq = forbidden_equiv.search(doc_content)
            if m_eq:
                errors.append(
                    f"Unsupported equivalence claim '{m_eq.group(0)}' found in {doc_name} while rmse_equivalence_margin is null"
                )

    return errors


def write_research_reporting(ds: Optional[ReportingDataset] = None) -> None:
    """Write REPORT.md, docs/generated/scientific_results_manifest.json, and LaTeX artifacts."""
    if ds is None:
        ds = load_reporting_dataset()

    # Sync LaTeX macros and tables
    generate_macros(ds)
    generate_tables(ds)

    # Write REPORT.md
    md_content = render_report_markdown(ds)
    report_md_path = ds.root_dir / "REPORT.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Generated REPORT.md with {len(SECTION_IDS)} auto-generated sections.")

    # Write docs/generated/scientific_results_manifest.json
    manifest_obj = build_scientific_results_manifest(ds)
    manifest_path = ds.root_dir / "docs" / "generated" / "scientific_results_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, indent=2)
        f.write("\n")
    print(
        f"Generated docs/generated/scientific_results_manifest.json "
        f"({manifest_obj['summary_counts']['total_metrics_tracked']} metrics tracked)."
    )


def check_research_reporting(ds: Optional[ReportingDataset] = None) -> Tuple[bool, str]:
    """Non-mutating check of REPORT.md, manifest, LaTeX macros/tables, and manuscript integrity."""
    try:
        if ds is None:
            ds = load_reporting_dataset()
    except ReportingDataError as exc:
        return False, f"Canonical reporting dataset validation failed: {exc}"

    diagnostics: List[str] = []

    # 1. Check LaTeX macros and tables
    ok_tex, msg_tex = check_report_artifacts(ds)
    if not ok_tex:
        diagnostics.append(msg_tex)

    # 2. Check REPORT.md full content and section markers
    report_md_path = ds.root_dir / "REPORT.md"
    expected_md = render_report_markdown(ds)
    if not report_md_path.is_file():
        diagnostics.append("MISSING FILE: REPORT.md")
    else:
        actual_md = report_md_path.read_text(encoding="utf-8")
        if actual_md != expected_md:
            diff = "".join(
                difflib.unified_diff(
                    actual_md.splitlines(keepends=True),
                    expected_md.splitlines(keepends=True),
                    fromfile="REPORT.md (on disk)",
                    tofile="REPORT.md (expected from canonical artifacts)",
                )
            )
            diagnostics.append(f"REPORT.md does not match canonical generated output:\n{diff}")

    # 3. Check docs/generated/scientific_results_manifest.json
    manifest_path = ds.root_dir / "docs" / "generated" / "scientific_results_manifest.json"
    expected_manifest_str = json.dumps(build_scientific_results_manifest(ds), indent=2) + "\n"
    if not manifest_path.is_file():
        diagnostics.append("MISSING FILE: docs/generated/scientific_results_manifest.json")
    else:
        actual_manifest_str = manifest_path.read_text(encoding="utf-8")
        if actual_manifest_str != expected_manifest_str:
            diff = "".join(
                difflib.unified_diff(
                    actual_manifest_str.splitlines(keepends=True),
                    expected_manifest_str.splitlines(keepends=True),
                    fromfile="docs/generated/scientific_results_manifest.json (on disk)",
                    tofile="docs/generated/scientific_results_manifest.json (expected)",
                )
            )
            diagnostics.append(
                f"docs/generated/scientific_results_manifest.json is out of sync:\n{diff}"
            )

    # 4. Run manuscript-level numerical auditing
    audit_errors = audit_manuscript_integrity(ds)
    diagnostics.extend(audit_errors)

    if diagnostics:
        return False, "\n\n".join(diagnostics)
    return (
        True,
        "All reporting outputs (REPORT.md, results/macros.tex, tables/*.tex, "
        "docs/generated/scientific_results_manifest.json) and manuscript checks passed.",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify REPORT.md, scientific_results_manifest.json, and LaTeX artifacts."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--write",
        action="store_true",
        help="Deterministically write REPORT.md, manifest, and LaTeX artifacts (default).",
    )
    group.add_argument(
        "--check",
        action="store_true",
        help="Verify all controlled outputs without modifying any files (exits non-zero on mismatch).",
    )
    args = parser.parse_args(argv)

    if args.check:
        ok, message = check_research_reporting()
        if not ok:
            sys.stderr.write("ERROR: Research reporting integrity check failed:\n")
            sys.stderr.write(message + "\n")
            return 1
        print(message)
        return 0

    write_research_reporting()
    ok, message = check_research_reporting()
    if not ok:
        sys.stderr.write("ERROR: Post-write verification failed:\n")
        sys.stderr.write(message + "\n")
        return 1
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
