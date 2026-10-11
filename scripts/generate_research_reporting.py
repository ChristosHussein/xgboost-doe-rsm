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
    cs = ds.curvature_sensitivity
    curv_pct_inc = (float(p1["diff_F_minus_C"]) / float(p1["yC_bar"])) * 100.0
    cb = ds.computational_budget
    mt = ds.method_timing_summary
    hpb = ds.holdout_pareto_bootstrap

    sections: Dict[str, str] = {}

    # 1. EXEC_SUMMARY
    sections["EXEC_SUMMARY"] = "\n".join(
        [
            f"This report presents a Design of Experiments (DOE) and Response Surface Methodology (RSM) study applied to the multi-objective hyperparameter optimization of an **XGBoost Regressor** on the **California Housing** benchmark dataset ($N = 20,640$ observations, $8$ continuous features). Following the methodology of Douglas C. Montgomery's *Design and Analysis of Experiments* (9th ed., 2017, Chapters 4, 5, 6, 7, 10, 11, and 13), we treat stochastic machine learning variability—arising from random train/validation partitioning and row subsampling—as a **nuisance factor** controlled via a **Randomized Complete Block Design (RCBD)** across $b = {len(ds.block_seeds)}$ seed blocks ($\\mathcal{{S}} = \\{{{block_seeds_str}\\}}$).",
            "",
            f"Rather than an adaptive sequential search, the historical DOE campaign executes a **pre-planned, globally randomized 140-run Face-Centered Central Composite Design (FCCD)** analyzed in two hierarchical stages ($100$ factorial-plus-center runs in Phase 1 and $40$ axial runs in Phase 2, with ${ds.doe_axial_before_run100}$ of the $40$ axial runs occurring at `run_order <= 100`), followed by **{n_conf} confirmation trials** in Phase 5 for $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ and **{n_conf} confirmation trials** for $\\mathbf{{x}}^*_{{\\text{{SO}}}}$, alongside a prospective 20-replicate benchmark campaign comprising **{ds.total_model_fits:,} XGBoost model fits** and **{ds.total_timed_inferences:,} timed single-sample inferences** ({ds.total_all_inferences:,} total single-row prediction calls including {ds.total_warmup_inferences:,} warmup calls):",
            "",
            "1. **Holdout Isolation and Dataset History**: The 20,640 dataset observations are partitioned into an 80% development pool ($N_{\\text{dev}} = 16,512$) and a 20% external holdout test set ($N_{\\text{test}} = 4,128$). Within each block $b$, the development pool is split 75/25 into training ($N_{\\text{train}} = 12,384$) and validation ($N_{\\text{val}} = 4,128$) sets. Response $Y_1$ is **Development Validation RMSE** on the 25% validation split. Whereas the historical `v1.0.0` baseline experiments recorded holdout test results during development, the revised `Revision-v2` pipeline programmatically prevented access to those test labels during all search iterations, surrogate fitting, curvature testing, lack-of-fit analysis, and desirability optimization until winning configurations were frozen in `finalized_selections.json`. These safeguards enforce programmatic pipeline isolation during the revised search, although they cannot retroactively undo earlier historical exposure of the same dataset.",
            f"2. **Phase 1 ($2^4$ Full Factorial + Center Points in {len(ds.block_seeds)} Blocks, $N_1 = 100$ runs)**: Screening identifies **Learning Rate** ($\\ln \\eta$, $F = {f_x1_p1:.2f}, p < 10^{{-15}}, \\eta^2_p = {eta_x1_p1:.4f}$) and **Max Tree Depth** ($F = {f_x2_p1:.2f}, p < 0.0001, \\eta^2_p = {eta_x2_p1:.4f}$), along with their interaction $x_1 x_2$ ($F = {f_x1x2_p1:.2f}, p < 0.0001, \\eta^2_p = {eta_x1x2_p1:.4f}$), as the dominant drivers of validation RMSE. Single-degree-of-freedom curvature testing against within-block center-point pure error ($\\text{{df}} = {p1['df_pe_center']}, \\text{{MS}}_{{\\text{{PE, center}}}} = {_fmt_sci_md(p1['ms_pe_center'])}$) yields $F_{{\\text{{Curv}}}} = {p1['f_curvature']:,.2f}$ ($p < 10^{{-15}}$), where the factorial corner mean ($\\bar{{y}}_F = {p1['yF_bar']:.4f}$) exceeds the center mean ($\\bar{{y}}_C = {p1['yC_bar']:.4f}$) by ${p1['diff_F_minus_C']:.4f}$ RMSE (${curv_pct_inc:+.2f}\\%$). Because within-block center replicates vary only row-subsampling seeds at fixed split $S_b$, sensitivity checks across six broader error denominators—including block-adjusted pooled coordinate error ($F = {cs['f_block_adj_pooled']:,.2f}, \\text{{df}} = {int(cs['df_block_adj_pooled'])}$), Treatment $\\times$ Block interaction ($F = {cs['f_txb_p1']:,.2f}, \\text{{df}} = {int(cs['df_txb_p1'])}$), 5-block paired contrast ($t = {cs['t_block_paired']:.2f}, F = {cs['f_block_paired']:,.2f}, \\text{{df}} = {int(cs['df_block_paired'])}, p = {_fmt_sci_md(cs['p_block_paired'])}$), and unadjusted first-order+2FI model residual ($F = {cs['f_2fi_resid']:.2f}, \\text{{df}} = {int(cs['df_2fi_resid'])}, p = {_fmt_sci_md(cs['p_2fi_resid'])}$)—confirm that quadratic curvature is robust to error-term specification while establishing that rejecting $H_0: \\sum \\beta_{{ii}} = 0$ detects severe boundary curvature rather than validating global quadratic adequacy.",
            f"3. **Phase 2 (Face-Centered Central Composite Design, $\\alpha = 1.0$, $N_{{\\text{{CCD}}}} = 140$ runs)**: Including the $40$ axial runs fits a full second-order polynomial ($R^2 = {ds.ccd_y1_rsq:.4f}, \\text{{Adj }} R^2 = {ds.ccd_y1_adj_rsq:.4f}$). Once quadratic curvature is modeled, the RCBD block effect becomes highly significant ($F = {f_block_ccd:.2f}, p < 0.0001$), absorbing **ICC = {icc_anova_pct:.2f}%** ($\\text{{ICC}}_{{\\text{{REML}}}} = {icc_reml:.4f}, \\sigma_{{\\text{{block}}}} \\approx {sigma_block:.4f}$ RMSE) of unexplained residual variance. Formal lack-of-fit decomposition shows statistically significant structural polynomial misfit ($F_{{\\text{{LoF}}}} = {lof['Y1']['F_LoF']:.2f}, p = {_fmt_sci_md(lof['Y1']['p_LoF'])}$ against the {lof['Y1']['df_PE']}-df saturated additive baseline; $F_{{\\text{{LoF}}}} = {ld['f_lof_center_y1']:.2f}, p < 10^{{-15}}$ against {int(ld['df_pure_center'])}-df center pure error; RMS misfit $= {lof['Y1']['sqrt_MS_LoF']:.4f}$ RMSE), demonstrating that second-order polynomials serve as local guidance maps rather than globally exact estimators of gradient boosting loss.",
            f"4. **Phase 3 (Canonical & Ridge Analysis)**: Spectral decomposition of $\\hat{{\\mathbf{{B}}}}$ yields three positive eigenvalues and one near-zero eigenvalue ($\\lambda = \\{{{eigs[0]:.6f}, {eigs[1]:.6f}, {eigs[2]:.6f}, {eigs[3]:.6f}\\}}$; wild bootstrap 95% CI for $\\lambda_1$: $[{boot_p[0][0]:.4f}, {boot_p[2][0]:+.4f}]$, with ${p3['fraction_min_eigenvalue_le_zero']*100:.1f}\\%$ of resamples $\\le 0$), characterizing a **stationary/rising ridge system**. The unconstrained stationary point ($\\|\\mathbf{{x}}_0\\| = {p3['distance_coded']:.2f}$) lies outside $[-1, +1]^4$; constrained optimization over valid integer depths identifies the single-objective candidate $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ at depth ${cube['depth']}$ ($x_1 = {cube_x[0]:.4f}, x_2 = {cube_x[1]:.4f}, x_3 = {cube_x[2]:.4f}, x_4 = {cube_x[3]:.4f}$, i.e., $\\eta = {cube_nat['eta']:.4f}, \\text{{depth}} = {cube['depth']}, \\text{{subsample}} = {cube_nat['subsample']:.4f}, \\lambda = {cube_nat['reg_lambda']:.4f}$) with predicted validation RMSE $\\hat{{y}} = {cube['pred_rmse']:.4f}$.",
            f"5. **Phase 4 (Derringer-Suich Multi-Objective Desirability)**: Balancing Validation RMSE ($Y_1 \\in [0.450, 0.700]$) and Single-Sample Inference Latency ($Y_2 \\in [100.0, 180.0]\\,\\mu\\text{{s}}$) yields an interior compromise coordinate $\\mathbf{{x}}^*_{{\\text{{MO}}}} = [{mo_coded[0]:.4f}, {mo_coded[1]:.4f}, {mo_coded[2]:.4f}, {mo_coded[3]:.4f}]^T$ ($\\eta = {mo_nat['eta']:.4f}, \\text{{depth}} = {mo_nat['depth']}, \\text{{subsample}} = {mo_nat['subsample']:.4f}, \\lambda = {mo_nat['reg_lambda']:.4f}$), achieving composite desirability **$D = {d_star['D']:.4f}$** ($d_1 = {d_star['d1']:.4f}, d_2 = {d_star['d2']:.4f}$) with predicted validation RMSE $\\hat{{Y}}_1 = {d_star['pred_y1']:.4f}$ and predicted latency $\\hat{{Y}}_2 = {d_star['pred_y2']:.2f}\\,\\mu\\text{{s}}$.",
            f"6. **Phase 5 (Confirmation & Comparative Benchmarks)**: Across **{n_conf} confirmation trials** ($m = {n_conf}$ fresh seeds $\\mathcal{{S}}_{{\\text{{conf}}}} = \\{{{conf_seeds_str}\\}}$), $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ achieves empirical validation RMSE ${mo_val['empirical_mean']:.4f} \\pm {mo_val['empirical_std']:.4f}$ (falling inside the Satterthwaite 95% prediction interval $[{mo_val['prediction_interval_95'][0]:.4f}, {mo_val['prediction_interval_95'][1]:.4f}]$), empirical latency ${mo_lat['empirical_mean']:.2f} \\pm {mo_lat['empirical_std']:.2f}\\,\\mu\\text{{s}}$ (${mo_lat['empirical_mean']:.1f} \\pm {mo_lat['empirical_std']:.1f}\\,\\mu\\text{{s}}$, inside $[{mo_lat['prediction_interval_95'][0]:.2f}, {mo_lat['prediction_interval_95'][1]:.2f}]\\,\\mu\\text{{s}}$), and external holdout test RMSE ${mo_test['empirical_mean']:.4f} \\pm {mo_test['empirical_std']:.4f}$. At the single-objective depth-{cube['depth']} candidate $\\mathbf{{x}}^*_{{\\text{{SO}}}}$, empirical validation RMSE is ${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$—lying ${so_above_pi:.4f}$ above the Satterthwaite 95% PI $[{so_conf['prediction_interval_95_val'][0]:.4f}, {so_conf['prediction_interval_95_val'][1]:.4f}]$ due to ${so_bias:+.4f}$ polynomial optimism bias—while achieving external holdout test RMSE ${so_conf['empirical_test_rmse']:.4f} \\pm {so_conf['empirical_test_std']:.4f}$ and borderline latency ${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}\\,\\mu\\text{{s}}$ (at the lower bound of $[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]\\,\\mu\\text{{s}}$). In the 20-replicate prospective benchmark campaign, Single-Objective TPE achieves the lowest observed test RMSE (${so_tpe.test_rmse_mean:.5f} \\pm {so_tpe.test_rmse_sd:.5f}$) at high latency (${so_tpe.predict_latency_mean:.2f} \\pm {so_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$), whereas Repeated DOE Single-Objective yields deterministic boundary selection (${doe_so.test_rmse_mean:.5f} \\pm {doe_so.test_rmse_sd:.5f}$, retraining $\\sigma_{{\\text{{eval}}}} = {doe_so.retrain_sd:.5f}$) at ${ds.doe_so_vs_sotpe_latency_reduction_pct:.1f}\\%$ lower latency (${doe_so.predict_latency_mean:.2f} \\pm {doe_so.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$). In multi-objective optimization, Repeated DOE ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$) and Multi-Objective TPE show no statistically significant difference in holdout test RMSE (${doe_mo.test_rmse_mean:.5f} \\pm {doe_mo.test_rmse_sd:.5f}$ vs. ${mo_tpe.test_rmse_mean:.5f} \\pm {mo_tpe.test_rmse_sd:.5f}$, difference ${ds.doe_mo_vs_motpe_diff:+.5f}$, primary Welch $p = {ds.doe_mo_vs_motpe_welch_p:.4f}$) at comparable latency (${doe_mo.predict_latency_mean:.2f} \\pm {doe_mo.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ vs. ${mo_tpe.predict_latency_mean:.2f} \\pm {mo_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$) and $100\\%$ ($20/20$) constraint feasibility.",
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

    x1_ln_mid = 0.5 * (np.log(x1_min) + np.log(x1_max))
    x1_ln_half = 0.5 * (np.log(x1_max) - np.log(x1_min))
    x2_half = (x2_max - x2_min) // 2
    x3_half = 0.5 * (x3_max - x3_min)
    x4_ln_mid = int(round(0.5 * (np.log(x4_min) + np.log(x4_max))))
    x4_ln_half = 0.5 * (np.log(x4_max) - np.log(x4_min))

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
            f"$$x_1 = \\frac{{\\ln(\\eta) - ({x1_ln_mid:.4f})}}{{{x1_ln_half:.4f}}}, \\quad x_2 = \\frac{{d - {x2_cent}}}{{{x2_half}}}, \\quad x_3 = \\frac{{s - {x3_cent:.2f}}}{{{x3_half:.2f}}}, \\quad x_4 = \\frac{{\\ln(\\lambda) - {x4_ln_mid}}}{{{x4_ln_half:.4f}}}$$",
            "",
            "### 1.2 Nuisance Blocking (RCBD) and Dual Responses",
            "",
            f"Each of the $b = {len(ds.block_seeds)}$ blocks ($\\mathcal{{S}} = \\{{{block_seeds_str}\\}}$) defines a deterministic 75/25 train/validation split of the 16,512-sample development pool via split seed $S_b$. For factorial corner points ($1 \\dots 16$) and axial points ($18 \\dots 25$), the XGBoost model seed equals $S_b$. For the $n_C = 4$ center-point replicates within each block, the train/validation split is held fixed at $S_b$ while the XGBoost subsampling seed is varied as $S_b + r \\times 1000$ ($r \\in \\{{0, 1, 2, 3\\}}$), isolating **{int(ld['df_pure_center'])} degrees of freedom of within-block subsampling pure error** from **4 degrees of freedom of block-to-block data-split variance**.",
            "",
            "Two response variables are recorded per run:",
            "- **Response $Y_1$ (Development Validation RMSE)**: Root Mean Squared Error on the 25% validation partition ($N_{\\text{val}} = 4,128$). Unlike the historical `v1.0.0` pipeline, which logged holdout test metrics during development, the revised pipeline prevented access to the 20% external holdout test partition ($N_{\\text{test}} = 4,128$) until candidate configurations were frozen in `finalized_selections.json`.",
            "- **Response $Y_2$ (Single-Sample Inference Latency, $\\mu\\text{s}$)**: Mean execution time in microseconds per single-row prediction on CPU Core 0 (`SetProcessAffinityMask = 1`, `n_jobs = 1`), timed using `time.perf_counter_ns()`.",
            "",
            "### 1.3 Pre-Planned Design Chronology, Evaluation Budget Allocation, and Timing Protocols",
            "",
            f"- **Pre-Planned Chronology vs. Hierarchical Analysis**: In `phase1_2_doe.py`, all $28 \\times 5 = 140$ runs (comprising $16$ factorial corners, $4$ center-point replicates, and $8$ axial points per block) were constructed upfront and globally shuffled via `rng = np.random.default_rng(42)` prior to execution (`run_order` $1 \\dots 140$). Consequently, **${ds.doe_axial_before_run100}$ of the $40$ `Phase2_Axial` runs were executed at `run_order <= 100`**, interleaved with `Phase1_Factorial` runs. The designation of Phase 1 ($N_1 = 100$) and Phase 2 ($N_{{\\text{{CCD}}}} = 140$) therefore represents a **two-stage hierarchical statistical analysis** of a pre-planned Face-Centered CCD rather than an adaptive sequential experiment conditioned on intermediate observations.",
            f"- **Fixed-Coordinate Replication vs. Adaptive Search Budget**: Under the matched 140-evaluation budget, the blocked FCCD evaluates **25 distinct hyperparameter coordinates** replicated across $5$ seed blocks ($25 \\times 5 = 125$ coordinate-block evaluations plus $15$ additional within-block center replicates $= 140$ fits). Because a standard FCCD allocates $8$ factorial corners and $1$ axial point per block to the lower learning-rate boundary $x_1 = -1.0$ ($\\eta = 0.01$), exactly **${ds.doe_low_eta_run_count}/140$ runs (${ds.doe_low_eta_run_pct:.2f}\\%$) evaluate $\\eta = 0.01$**, where 100-tree ensembles severely underfit (mean validation RMSE ${ds.doe_low_eta_mean_rmse:.4f}$ vs. ${ds.doe_high_eta_mean_rmse:.4f}$ for $x_1 > -1.0$). By contrast, adaptive Bayesian optimizers (SO-TPE, MO-TPE, Constrained TPE) and Random Search evaluate **140 distinct hyperparameter configurations** on a single split without cross-block replication, trading variance-decomposition power and split-blocking robustness for 5.6-fold denser coordinate exploration.",
            f"- **Latency Measurement Protocols and Interface Overhead (`predict` vs. `inplace_predict`)**: Latency measurements follow three distinct protocol tiers that must not be conflated: (i) **Historical DOE & Confirmation (`results/runs.csv`, `results/confirmation_runs*.csv`, `results/benchmark.csv`)**: 50 warmup calls per interface followed by 1,000 timed single-row calls per model fit; (ii) **Prospective Online Search (`optimizer_trials.csv`, `doe_selection_runs.csv`)**: 0 warmup calls and 30 timed `predict` calls per candidate fit to bound search wall-clock time; and (iii) **Prospective Holdout Evaluation (`final_evaluations.csv`)**: 25 warmup calls per interface (50 total) followed by 500 timed single-row calls per model across 2 independent timing sessions ($1,000$ timed calls/seed). Across all four empirical datasets, high-level `Booster.predict(DMatrix)` incurs a consistent data-marshaling overhead relative to `XGBRegressor.inplace_predict` on raw NumPy arrays: **${ds.hist_runs_interface_overhead_mean:.2f} \\pm {ds.hist_runs_interface_overhead_sd:.2f}\\,\\mu\\text{{s}}$** (range ${ds.hist_runs_interface_overhead_min:.2f}$–${ds.hist_runs_interface_overhead_max:.2f}\\,\\mu\\text{{s}}$) across the 140 historical DOE runs, **${ds.hist_conf_mo_interface_overhead_mean:.2f} \\pm {ds.hist_conf_mo_interface_overhead_sd:.2f}\\,\\mu\\text{{s}}$** ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$) and **${ds.hist_conf_so_interface_overhead_mean:.2f} \\pm {ds.hist_conf_so_interface_overhead_sd:.2f}\\,\\mu\\text{{s}}$** ($\\mathbf{{x}}^*_{{\\text{{SO}}}}$) in historical confirmation, **${ds.hist_bench_interface_overhead_mean:.2f} \\pm {ds.hist_bench_interface_overhead_sd:.2f}\\,\\mu\\text{{s}}$** (range ${ds.hist_bench_interface_overhead_min:.2f}$–${ds.hist_bench_interface_overhead_max:.2f}\\,\\mu\\text{{s}}$) in the 5-method historical benchmark, and **${ds.interface_overhead_mean:.2f} \\pm {ds.interface_overhead_sd:.2f}\\,\\mu\\text{{s}}$** (range ${ds.interface_overhead_min:.2f}$–${ds.interface_overhead_max:.2f}\\,\\mu\\text{{s}}$) across the 50-configuration Revision-v2 interface audit (`latency_interface_overhead.json`).",
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
        f"Phase 1 analyzes the $2^4 = 16$ factorial corners plus $n_C = 4$ center replicates across $b = {len(ds.block_seeds)}$ blocks ($N_1 = 100$ runs). Fitting the first-order model with two-factor interactions yields:",
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

    df_p1_runs = ds.runs_df[ds.runs_df["phase"].astype(str).str.contains("1")].copy()
    df_p1_runs["coord"] = list(
        zip(df_p1_runs["x1"], df_p1_runs["x2"], df_p1_runs["x3"], df_p1_runs["x4"])
    )
    ss_pe_all = float(
        df_p1_runs.groupby("coord")["val_rmse"]
        .apply(lambda s: ((s - s.mean()) ** 2).sum())
        .sum()
    )
    df_pe_all = int(len(df_p1_runs) - df_p1_runs["coord"].nunique())
    ms_pe_all = ss_pe_all / df_pe_all
    f_curv_all = float(p1["ss_curvature"]) / ms_pe_all

    p1_rows.extend(
        [
            "",
            "### 2.2 Single-Degree-of-Freedom Curvature Test",
            "",
            f"To test $H_0: \\sum_{{i=1}}^4 \\beta_{{ii}} = 0$, we compare the mean validation RMSE of the $n_F = 80$ factorial corner runs against the $n_C = 20$ center-point runs against within-block center-point pure error ($\\text{{df}} = {p1['df_pe_center']}$):",
            f"- **Factorial Mean ($\\bar{{y}}_F$)**: ${p1['yF_bar']:.4f}$",
            f"- **Center Point Mean ($\\bar{{y}}_C$)**: ${p1['yC_bar']:.4f}$",
            f"- **Curvature Contrast ($\\bar{{y}}_F - \\bar{{y}}_C$)**: ${p1['diff_F_minus_C']:+.4f}$ RMSE (${curv_pct_inc:+.2f}\\%$ relative to $\\bar{{y}}_C$; block-paired 95% CI $[{cs['ci95_block_paired'][0]:+.4f}, {cs['ci95_block_paired'][1]:+.4f}]$)",
            f"- **Curvature Sum of Squares**: $\\text{{SS}}_{{\\text{{Curv}}}} = \\frac{{n_F n_C}}{{n_F + n_C}}(\\bar{{y}}_F - \\bar{{y}}_C)^2 = {p1['ss_curvature']:.6f}$",
            f"- **Within-Block Center Pure Error**: $\\text{{SS}}_{{\\text{{PE, center}}}} = {ld['ss_pure_center_y1']:.6f}$, $\\text{{df}}_{{\\text{{PE}}}} = {p1['df_pe_center']}$, $\\text{{MS}}_{{\\text{{PE, center}}}} = {_fmt_sci_md(p1['ms_pe_center'])}$",
            "- **Curvature $F$-Statistic**:",
            f"$$F_{{\\text{{Curv}}}} = \\frac{{\\text{{SS}}_{{\\text{{Curv}}}}}}{{\\text{{MS}}_{{\\text{{PE, center}}}}}} = \\frac{{{p1['ss_curvature']:.6f}}}{{{_fmt_sci_md(p1['ms_pe_center'])}}} = {p1['f_curvature']:,.2f} \\quad (p < 10^{{-15}})$$",
            "",
            "### 2.3 Sensitivity of Curvature Test Across Alternative Error Denominators",
            "",
            f"Because the 4 center replicates per block hold the 75/25 data split fixed at $S_b$ and vary only the row-subsampling seed ($s = 0.75$), $\\text{{MS}}_{{\\text{{PE, center}}}} = {_fmt_sci_md(p1['ms_pe_center'])}$ ($\\text{{SD}} \\approx {np.sqrt(p1['ms_pe_center']):.5f}$ RMSE) measures micro-scale algorithmic subsampling variability rather than cross-split interaction or higher-order model error. Comparing macro-scale hyperparameter contrast ($\\text{{SS}}_{{\\text{{Curv}}}} = {p1['ss_curvature']:.6f}$, driven by underfitting at $\\eta = 0.01$) against micro-scale subsampling noise mechanically produces an $F$-ratio of $O(10^5)$. To verify that curvature detection does not depend on this narrow denominator, the table below evaluates $\\text{{SS}}_{{\\text{{Curv}}}}$ across six increasingly conservative error structures:",
            "",
            "| Error Denominator Specification | Error DF | Error SS | Error MS | Curvature $F$-Statistic | $p$-Value | Variance Scope Isolated |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
            f"| **1. Within-Block Center Pure Error ($\\text{{MS}}_{{\\text{{PE, center}}}}$)** | ${int(cs['df_pe_center'])}$ | ${ld['ss_pure_center_y1']:.6f}$ | ${_fmt_sci_md(cs['ms_pe_center'])}$ | ${cs['f_pe_center']:,.2f}$ | $< 10^{{-15}}$ | Subsampling seed noise at fixed split $S_b$ |",
            f"| **2. Block-Adjusted Pooled Coordinate Error** | ${int(cs['df_block_adj_pooled'])}$ | ${cs['ss_block_adj_pooled']:.6f}$ | ${_fmt_sci_md(cs['ms_block_adj_pooled'])}$ | ${cs['f_block_adj_pooled']:,.2f}$ | $< 10^{{-15}}$ | Center PE ($15$ df) + Treatment $\\times$ Block ($64$ df) |",
            f"| **3. Phase 1 Treatment $\\times$ Block Interaction** | ${int(cs['df_txb_p1'])}$ | ${cs['ss_txb_p1']:.6f}$ | ${_fmt_sci_md(cs['ms_txb_p1'])}$ | ${cs['f_txb_p1']:,.2f}$ | $< 10^{{-15}}$ | Coordinate-by-split interaction across $17 \\times 5$ cells |",
            f"| **4. Block-Paired Contrast ($\\bar{{y}}_{{F,b}} - \\bar{{y}}_{{C,b}}$ across 5 Blocks)** | ${int(cs['df_block_paired'])}$ | — | $\\text{{SE}} = {cs['se_block_paired']:.4f}$ | ${cs['f_block_paired']:,.2f}$ ($t = {cs['t_block_paired']:.2f}$) | ${_fmt_sci_md(cs['p_block_paired'])}$ | Direct across-block pairing ($95\\%$ CI $[{cs['ci95_block_paired'][0]:+.4f}, {cs['ci95_block_paired'][1]:+.4f}]$) |",
            f"| **5. Unadjusted Coordinate Pure Error ($\\text{{MS}}_{{\\text{{PE, All}}}}$)** | ${df_pe_all}$ | ${ss_pe_all:.6f}$ | ${_fmt_sci_md(ms_pe_all)}$ | ${cs['f_pe_coord_unadj']:,.2f}$ ($\\approx {f_curv_all:,.0f}$) | $< 10^{{-15}}$ | Pooled within-coordinate variance (includes block main effect) |",
            f"| **6. First-Order + 2FI + Curvature Residual** | ${int(cs['df_2fi_curv_resid'])}$ | ${cs['ss_2fi_curv_resid']:.6f}$ | ${_fmt_sci_md(cs['ms_2fi_curv_resid'])}$ | ${cs['f_2fi_curv_resid']:,.2f}$ | $< 10^{{-15}}$ | Higher-order factorial interactions ($3$FI+$4$FI) + TxB + PE |",
            f"| **7. Unadjusted First-Order + 2FI Residual ($\\text{{MS}}_E$)** | ${int(cs['df_2fi_resid'])}$ | ${ss_res_p1:.6f}$ | ${cs['ms_2fi_resid']:.6f}$ | ${cs['f_2fi_resid']:.2f}$ | ${_fmt_sci_md(cs['p_2fi_resid'])}$ | Total Phase 1 residual before separating curvature |",
            "",
            "*Methodological Distinction*: Rejecting $H_0: \\sum_{{i=1}}^4 \\beta_{{ii}} = 0$ confirms that a flat first-order plane is inadequate over $\\eta \\in [0.01, 0.30]$; it does **not** imply that a second-order polynomial is free of higher-order misfit. Surrogate predictive adequacy is evaluated separately via Phase 2 lack-of-fit testing (Section 3.3) and Phase 5 confirmation intervals (Section 6).",
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
        f"Incorporating the $2k = 8$ axial points ($\\alpha = 1.0$) across {len(ds.block_seeds)} blocks yields $N = 140$ runs ($28$ runs/block across $25$ distinct coordinates). The fitted second-order validation RMSE model achieves $R^2 = {ds.ccd_y1_rsq:.4f}$ and $\\text{{Adjusted }} R^2 = {ds.ccd_y1_adj_rsq:.4f}$:",
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
    mse_reduction_factor = int(round(ms_e_p1 / ms_e_p2))
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
            f"In Phase 1, unmodeled quadratic curvature inflated the residual mean square ($\\text{{MS}}_E = {ms_e_p1:.6f}$), masking block differences ($F = {f_blk_p1:.2f}, p = {p_blk_p1:.4f}$). In Phase 2, accounting for quadratic terms reduces $\\text{{MS}}_E$ by ${mse_reduction_factor}\\times$ to ${ms_e_p2:.6f}$, exposing highly significant seed-to-seed variance ($F = {f_block_ccd:.2f}, p < 0.0001$). The **Block Intraclass Correlation Coefficient** is:",
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
            "Residual adequacy diagnostics for $Y_1$ (explicitly distinguishing grouping factors):",
            f"- **Normality**: Shapiro-Wilk $W = {diag['shapiro_W']:.4f}, p = {diag['shapiro_p']:.4f}$ on the 140 second-order OLS residuals (consistent with Gaussian residuals once polynomial terms are fitted).",
            f"- **Homoscedasticity Across Grouping Factors vs. Fitted Values**: Mean-centered Levene test across the **5 seed blocks** (`block` $1 \\dots 5$) yields $W = {diag['levene_stat']:.4f}, p = {diag['levene_p']:.4f}$, confirming equal residual variance across data-split blocks. Median-centered Levene (Brown-Forsythe) test across the **25 distinct design coordinates** yields $W = {diag['brown_forsythe_points_stat']:.4f}, p = {diag['brown_forsythe_points_p']:.4f}$, because within each coordinate the 5 block residuals share similar spread after block adjustment. By contrast, the Breusch-Pagan test against continuous fitted values $\\hat{{y}}$ is highly significant ($\\text{{LM}} = {diag['breusch_pagan_stat']:.2f}, p < 0.0001$), because deterministic cell-mean polynomial misfit ($\\bar{{y}}_{{i\\cdot}} - \\hat{{y}}_i$) is concentrated at the high-RMSE $\\eta = 0.01$ boundary rather than uniform across $\\hat{{y}}$. We therefore report White/MacKinnon-White HC3 heteroscedasticity-robust standard errors alongside OLS $F$-tests in Table 3.1.",
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
    se_so_depth = float(
        ds.depth_opt_df[ds.depth_opt_df["depth"] == int(cube["depth"])]["se"].iloc[0]
    )
    so_bias_se_mult = float(round(so_bias, 4) / round(se_so_depth, 4))
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
            f"At $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ (depth {mo_nat['depth']}), both validation RMSE (${mo_val['empirical_mean']:.4f} \\pm {mo_val['empirical_std']:.4f}$) and single-sample inference latency (${mo_lat['empirical_mean']:.2f} \\pm {mo_lat['empirical_std']:.2f}\\,\\mu\\text{{s}}$) fall directly inside their Satterthwaite 95% prediction intervals. At $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ (depth {cube['depth']}), empirical validation RMSE (${so_conf['empirical_val_rmse']:.4f} \\pm {so_conf['empirical_val_std']:.4f}$) lies ${so_above_pi:.4f}$ above the upper prediction interval bound (${so_conf['prediction_interval_95_val'][1]:.4f}$), confirming that quadratic interpolation across depths $\\{{3, 6, 9\\}}$ overestimates accuracy gains at depth {cube['depth']} by ${so_bias:+.4f}$ RMSE (${so_bias_se_mult:.1f}\\times \\text{{SE}}(\\hat{{y}})$), while empirical latency (${so_conf['empirical_latency']:.1f} \\pm {so_conf['empirical_latency_std']:.1f}\\,\\mu\\text{{s}}$) sits on the lower bound of its 95% prediction interval ($[{so_conf['prediction_interval_95_lat'][0]:.1f}, {so_conf['prediction_interval_95_lat'][1]:.1f}]\\,\\mu\\text{{s}}$).",
            "",
            "### 6.1 Provenance of Numerical Similarities Across Confirmation, Historical Benchmark, and Revision-v2 Evaluations",
            "",
            f"Several summary metrics across Table 6 and Table 7.1 agree to four decimal places (`0.4649`, `0.4832`, `0.4952`, `0.4959`, `0.4765`) for distinct structural or statistical reasons that must be distinguished:",
            f"1. **Shared Configuration Hash and Shared 20-Seed Evaluation Pool (`doe-so-historical` vs. `doe-so-rep-0..19` and `doe-mo-historical`)**: Both the historical benchmark script (`phase5_benchmarks.py`) and the Revision-v2 holdout evaluator (`revision_experiments.py`) evaluate frozen configurations across the exact same 20 fresh seeds $\\mathcal{{S}}_{{\\text{{eval}}}} = \\{{2001, \\dots, 2020\\}}$ on the 75/25 split of the 16,512-row development pool ($N_{{\\text{{train}}}} = 12,384, N_{{\\text{{val}}}} = 4,128$) and the 4,128-row external holdout test set. Moreover, in Revision-v2, all 20 Repeated DOE Single-Objective replicates select the exact same boundary grid configuration (`config_hash = a83dde54d6ca2ee9`: $\\eta = {cube_nat['eta']:.4f}, d = {cube['depth']}, s = {cube_nat['subsample']:.4f}, \\lambda = {cube_nat['reg_lambda']:.4f}$) as `doe-so-historical`. Because `revision_experiments.py` deduplicates model training by `config_hash` across seeds `2001..2020`, `doe-so-historical` and all 20 `doe-so-rep-r` records map to the exact same 20 trained XGBoost boosters, yielding identical full-precision means (`test_rmse = 0.464899`, `val_rmse = 0.476524`) and zero between-search variance ($\\text{{SD}} = 0.00000$). Similarly, `doe-mo-historical` (`config_hash = 6fb2b64e8bf55e25`: $\\eta = {mo_nat['eta']:.4f}, d = {mo_nat['depth']}, s = {mo_nat['subsample']:.4f}, \\lambda = {mo_nat['reg_lambda']:.4f}$) is evaluated on the same 20 seeds `2001..2020` in both `results/benchmark.csv` and `final_evaluations.csv`, yielding identical RMSEs (`test_rmse = 0.483226`, `val_rmse = 0.495871`) while differing in latency due to separate timing sessions.",
            f"2. **Disjoint Seed Sets Matching to Four Decimal Places (`confirmation_runs*.csv` on `505..1414` vs. Benchmark on `2001..2020`)**: Phase 5 confirmation evaluates $\\mathbf{{x}}^*_{{\\text{{SO}}}}$ and $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ across $m = 10$ seeds $\\mathcal{{S}}_{{\\text{{conf}}}} = \\{{505, 606, \\dots, 1414\\}}$, which is completely disjoint from $\\mathcal{{S}}_{{\\text{{eval}}}} = \\{{2001, \\dots, 2020\\}}$. At $\\mathbf{{x}}^*_{{\\text{{SO}}}}$, the 10-seed confirmation holdout test RMSE mean is `0.46489249` ($\\text{{SD}} = 0.003382$), whereas the 20-seed benchmark holdout test RMSE mean is `0.46489935` ($\\text{{SD}} = 0.003042$); both round to `${so_conf['empirical_test_rmse']:.4f}$` at four decimal places despite differing by $6.86 \\times 10^{{-6}}$ RMSE. Likewise, at $\\mathbf{{x}}^*_{{\\text{{MO}}}}$, the 10-seed confirmation holdout test RMSE mean is `0.48318864` ($\\text{{SD}} = 0.003497$, rounding to `${mo_test['empirical_mean']:.4f}$`) vs. the 20-seed historical benchmark mean `0.48322619` ($\\text{{SD}} = 0.002975$, also rounding to `0.4832`), while the 20 Repeated DOE MO replicates (which select 4 distinct grid points at $\\eta \\in \\{{0.20, 0.21, 0.22\\}}, d=4, s=0.90$) average `${doe_mo.test_rmse_mean:.5f}$` (`0.4837` at 4 decimal places) and `${doe_mo.val_rmse_mean:.5f}$` (`0.4952` at 4 decimal places, compared to `0.495186` for the 10-seed confirmation validation mean at $\\eta = 0.2169, d=4, s=1.00$).",
        ]
    )

    # 8. SECTION_7_BENCHMARKS
    s7_rows = [
        f"To evaluate the pre-planned DOE + RSM methodology against modern heuristic and Bayesian optimizers under strictly equal evaluation budgets (140 model evaluations per search), we executed a prospective benchmark campaign (`results/revision_v2/full_run_001/`) comprising **{ds.total_model_fits:,} XGBoost model fits** and **{ds.total_timed_inferences:,} timed single-sample inferences** ({ds.total_all_inferences:,} total prediction calls including {ds.total_warmup_inferences:,} warmup calls) across $N = 20$ independent search replicates per optimizer.",
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
            f"| **Difference: Fixed Full DOE Front $-$ MO-TPE Mean (Split 42)** | — | — | ${hvs['diff_fdoe_minus_motpe_060']:+.4f}$ (1-sample $p < 0.0001$) | ${hvs['diff_fdoe_minus_motpe_065']:+.4f}$ (1-sample $p < 0.0001$) |",
            f"| **Panel B: Holdout Non-Dominated Set ({ds.total_selection_records} Records, {ds.distinct_config_hashes} Configs, {len(ds.fresh_eval_seeds)} Seeds)** | | | | |",
            f"| **Non-Dominated Set ({hvs['holdout_nd']} Distinct Configs)** | ${ds.total_selection_records}$ | ${hvs['holdout_nd']}$ | ${hvs['holdout_frozen_060']:.4f}$ | ${hvs['holdout_frozen_065']:.4f}$ |",
            f"| **Holdout Frontier Seed-Bootstrap ($B=2,000$ Resamples over 20 Seeds)** | ${ds.total_selection_records}$ | ${hpb['seed_only_nd_mean']:.2f} \\pm {hpb['seed_only_nd_sd']:.2f}$ | ${hpb['seed_only_060_mean']:.4f} \\pm {hpb['seed_only_060_sd']:.4f}$ ($[{hpb['seed_only_060_ci95'][0]:.4f}, {hpb['seed_only_060_ci95'][1]:.4f}]$) | ${hpb['seed_only_065_mean']:.4f} \\pm {hpb['seed_only_065_sd']:.4f}$ ($[{hpb['seed_only_065_ci95'][0]:.4f}, {hpb['seed_only_065_ci95'][1]:.4f}]$) |",
            f"| $\\quad \\llcorner$ Repeated DOE MO Selections | $20$ | ${nd_by_opt['repeated_preplanned_doe_multi_objective']}$ (Boot: ${hpb['by_optimizer']['repeated_preplanned_doe_multi_objective']['seed_only_present_pct']:.1f}\\%$) | — | — |",
            f"| $\\quad \\llcorner$ Multi-Objective TPE Selections | $20$ | ${nd_by_opt['multi_objective_tpe']}$ (Boot: ${hpb['by_optimizer']['multi_objective_tpe']['seed_only_present_pct']:.1f}\\%$) | — | — |",
            f"| $\\quad \\llcorner$ Constrained TPE Selections | $20$ | ${nd_by_opt['constrained_tpe']}$ (Boot: ${hpb['by_optimizer']['constrained_tpe']['seed_only_present_pct']:.1f}\\%$) | — | — |",
            f"| $\\quad \\llcorner$ Single-Objective TPE Selections | $20$ | ${nd_by_opt['single_objective_tpe']}$ (Boot: ${hpb['by_optimizer']['single_objective_tpe']['seed_only_present_pct']:.1f}\\%$) | — | — |",
            "",
            f"*Hypervolume Protocol Comparability Note*: Rows in Panel A and Panel B represent distinct evaluation protocols and candidate pools rather than a single homogeneous two-sample experiment: (i) **MO-TPE Candidates** ($20$ searches $\\times 140$ trials on fixed split seed $42$, timed with $30$ un-warmed calls/trial); (ii) **Repeated DOE Candidate Fronts** ($20$ searches $\\times 25$ FCCD coordinates averaged across $5$ disjoint block seeds in `3001..3100`, timed with $30$ un-warmed calls/fit); (iii) **Fixed Full DOE Evaluated Front** (a single deterministic 27-configuration set—25 FCCD points plus $\\mathbf{{x}}^*_{{\\text{{MO}}}}$ and $\\mathbf{{x}}^*_{{\\text{{SO}}}}$—evaluated once on split $42$ with $1,000$ warmed calls/fit); and (iv) **Holdout Non-Dominated Set** ($122$ frozen selection records across $95$ distinct configurations evaluated on the 4,128-row external holdout test set across $20$ fresh seeds `2001..2020` and $2$ timing sessions of $500$ warmed calls/fit). The Split-42 difference row (${hvs['diff_fdoe_minus_motpe_060']:+.4f}$ at $[{r1_rmse:.2f}, {r1_lat:.0f}]$; ${hvs['diff_fdoe_minus_motpe_065']:+.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.0f}]$) is a **one-sample comparison** of the 20 stochastic MO-TPE fronts against the single fixed 27-point DOE reference on split 42, not a two-sample test across independent data splits. Likewise, in the historical `v1.0.0` snapshot (`results/benchmark_summary.json`), the reported $18.38$ vs. $16.86$ ($+9.0\\%$) hypervolume comparison contrasted the full 25-point DOE design against a single selected MO-TPE configuration; when restricted to single deployed points ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$ vs. MO-TPE), historical hypervolume is $16.92$ vs. $16.86$ ($+0.3\\%$).",
            "",
            "### 7.3 Key Scientific Conclusions from the Benchmark Campaign",
            "",
            f"1. **Single-Objective Accuracy vs. Latency**: Single-Objective TPE achieves the numerically lowest observed mean holdout test RMSE (${so_tpe.test_rmse_mean:.5f} \\pm {so_tpe.test_rmse_sd:.5f}$) by concentrating searches in deeper trees ($d \\in \\{{7, 8, 9\\}}$), incurring ${so_tpe.predict_latency_mean:.2f} \\pm {so_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ inference latency (`inplace_predict`: ${so_tpe.inplace_latency_mean:.2f} \\pm {so_tpe.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$). Neither Single-Objective TPE nor Repeated DOE Single-Objective satisfies the $\\le 145\\,\\mu\\text{{s}}$ latency constraint ($0/20$ feasible). Repeated DOE Single-Objective ($\\mathbf{{x}}^*_{{\\text{{SO}}}}$, depth {cube['depth']}) selects the exact same boundary configuration deterministically across all 20 replicates (between-search $\\text{{SD}} = {doe_so.test_rmse_sd:.5f}$, conditional retraining $\\sigma_{{\\text{{eval}}}} = {doe_so.retrain_sd:.5f}$), achieving ${doe_so.test_rmse_mean:.5f}$ Test RMSE at ${doe_so.predict_latency_mean:.2f} \\pm {doe_so.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ inference latency (`inplace_predict`: ${doe_so.inplace_latency_mean:.2f} \\pm {doe_so.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$, ${ds.doe_so_vs_sotpe_latency_reduction_pct:.1f}\\%$ lower latency than SO-TPE).",
            f"2. **What Repeated DOE Replicates Represent & Separation of Selection vs. Retraining Variance**: Each of the $N = 20$ Repeated DOE replicates (`replicate_id` $0 \\dots 19$) executes a fresh 140-run blocked FCCD across 5 disjoint block seeds (`3001 + 5r` to `3005 + 5r`, spanning seeds `3001..3100`) drawn from the same 16,512-row California Housing development pool, fits OLS second-order surfaces for validation RMSE and latency, and selects candidates by evaluating the fitted surrogates over a deterministic **588-point candidate grid** ($21$ learning rates $\\eta \\in [0.05, 0.25]$ in steps of $0.01 \\times 7$ integer depths $d \\in \\{{3, \\dots, 9\\}} \\times 2$ subsample levels $s \\in \\{{0.9, 1.0\\}} \\times 2$ regularization levels $\\lambda \\in \\{{1.0, e\\}}$). Because the single-objective minimum on this grid consistently lies at the corner boundary ($\\eta = 0.08, d = 7, s = 1.0, \\lambda = 2.7183$), all 20 DOE-SO replicates select the same grid point, yielding between-search selection $\\text{{SD}} = 0.00000$; its non-zero uncertainty is captured entirely by the conditional retraining standard deviation $\\sigma_{{\\text{{eval}}}} = {doe_so.retrain_sd:.5f}$ across the 20 evaluation seeds `2001..2020`. For DOE-MO, desirability optimization selects among 4 adjacent interior grid points ($\\eta \\in \\{{0.20, 0.21, 0.22\\}}, d = 4, s = 0.90, \\lambda = 2.7183$, plus one replicate at $\\lambda = 1.0$), yielding a small between-search $\\text{{SD}} = {doe_mo.test_rmse_sd:.5f}$ alongside conditional retraining $\\sigma_{{\\text{{eval}}}} = {doe_mo.retrain_sd:.5f}$.",
            f"3. **Multi-Objective Statistical Comparison (Primary Unpaired Welch vs. Secondary Index-Paired Test)**: In the latency-constrained multi-objective regime, Repeated DOE ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$, depth {mo_nat['depth']}) achieves a mean Test RMSE of ${doe_mo.test_rmse_mean:.5f} \\pm {doe_mo.test_rmse_sd:.5f}$ at ${doe_mo.predict_latency_mean:.2f} \\pm {doe_mo.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ latency (`inplace_predict`: ${doe_mo.inplace_latency_mean:.2f} \\pm {doe_mo.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$), compared to ${mo_tpe.test_rmse_mean:.5f} \\pm {mo_tpe.test_rmse_sd:.5f}$ at ${mo_tpe.predict_latency_mean:.2f} \\pm {mo_tpe.predict_latency_sd:.2f}\\,\\mu\\text{{s}}$ (`inplace_predict`: ${mo_tpe.inplace_latency_mean:.2f} \\pm {mo_tpe.inplace_latency_sd:.2f}\\,\\mu\\text{{s}}$) for Multi-Objective TPE. Because replicate index $r \\in \\{{0, \\dots, 19\\}}$ pairs DOE block seeds `3001+5r..3005+5r` with Optuna sampler seed `1000+r` on split 42 rather than a shared random block, **Welch's two-sample $t$-test is the primary inferential comparison** ($\\Delta = {ds.doe_mo_vs_motpe_diff:+.5f}$ RMSE, $t = {ds.doe_mo_vs_motpe_welch_t:.4f}, p = {ds.doe_mo_vs_motpe_welch_p:.4f}$), while the replicate-index paired $t$-test ($t = {ds.doe_mo_vs_motpe_paired_t:.3f}, p = {ds.doe_mo_vs_motpe_paired_p:.4f}$) is reported as a secondary sensitivity check. Because no pre-registered equivalence margin was specified (`rmse_equivalence_margin: null`), this non-significant difference indicates no detected performance gap rather than formal statistical equivalence. Both methods achieve $20/20$ ($100\\%$) benchmark constraint feasibility.",
            f"4. **Development Hypervolume vs. Holdout Non-Dominated Set and Bootstrap Uncertainty**: On development split 42 under the online 30-call timing protocol, MO-TPE candidate fronts achieve a mean hypervolume of ${hvs['motpe_dev_060_mean']:.4f} \\pm {hvs['motpe_dev_060_sd']:.4f}$ at reference $[{r1_rmse:.2f}, {r1_lat:.1f}]$ (${hvs['motpe_dev_065_mean']:.4f} \\pm {hvs['motpe_dev_065_sd']:.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.1f}]$), exceeding the fixed 27-point DOE candidate frontier (${hvs['fdoe_dev_060']:.4f}$, descriptive 1-sample difference ${hvs['diff_fdoe_minus_motpe_060']:+.4f}, p < 0.0001$; at $[{r2_rmse:.2f}, {r2_lat:.1f}]$, ${hvs['fdoe_dev_065']:.4f}$, difference ${hvs['diff_fdoe_minus_motpe_065']:+.4f}, p < 0.0001$), while Repeated DOE 5-block means achieve ${hvs['rdoe_dev_060_mean']:.4f} \\pm {hvs['rdoe_dev_060_sd']:.4f}$ (${hvs['rdoe_dev_065_mean']:.4f} \\pm {hvs['rdoe_dev_065_sd']:.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.1f}]$). When the {ds.total_selection_records} frozen selection records ({ds.distinct_config_hashes} distinct configurations) are evaluated on the external holdout test set across {len(ds.fresh_eval_seeds)} retraining seeds, the empirical non-dominated set consists of **{hvs['holdout_nd']} distinct configurations** ($\\text{{HV}} = {hvs['holdout_frozen_060']:.4f}$ at $[{r1_rmse:.2f}, {r1_lat:.1f}]$ and ${hvs['holdout_frozen_065']:.4f}$ at $[{r2_rmse:.2f}, {r2_lat:.1f}]$): **{nd_by_opt['repeated_preplanned_doe_multi_objective']} Repeated DOE MO, {nd_by_opt['multi_objective_tpe']} MO-TPE, {nd_by_opt['constrained_tpe']} Constrained TPE, and {nd_by_opt['single_objective_tpe']} SO-TPE**. Paired cluster bootstrap resampling ($B = 2,000$ resamples over the 20 shared evaluation seeds `2001..2020`, `seed = 20261011`) yields a holdout frontier hypervolume of **${hpb['seed_only_060_mean']:.4f} \\pm {hpb['seed_only_060_sd']:.4f}$** ($95\\%$ CI $[{hpb['seed_only_060_ci95'][0]:.4f}, {hpb['seed_only_060_ci95'][1]:.4f}]$) at $[{r1_rmse:.2f}, {r1_lat:.0f}]$ and **${hpb['seed_only_065_mean']:.4f} \\pm {hpb['seed_only_065_sd']:.4f}$** ($95\\%$ CI $[{hpb['seed_only_065_ci95'][0]:.4f}, {hpb['seed_only_065_ci95'][1]:.4f}]$) at $[{r2_rmse:.2f}, {r2_lat:.0f}]$ (joint seed-plus-timing-session bootstrap: ${hpb['joint_060_mean']:.4f} \\pm {hpb['joint_060_sd']:.4f}$, $95\\%$ CI $[{hpb['joint_060_ci95'][0]:.4f}, {hpb['joint_060_ci95'][1]:.4f}]$). Across the 2,000 seed-bootstrap frontiers, Repeated DOE MO, MO-TPE, and SO-TPE each contribute at least one non-dominated point in **${hpb['by_optimizer']['repeated_preplanned_doe_multi_objective']['seed_only_present_pct']:.1f}\\%$** of resamples (mean non-dominated points: ${hpb['by_optimizer']['repeated_preplanned_doe_multi_objective']['seed_only_mean_points']:.2f}$ DOE-MO, ${hpb['by_optimizer']['multi_objective_tpe']['seed_only_mean_points']:.2f}$ MO-TPE, ${hpb['by_optimizer']['single_objective_tpe']['seed_only_mean_points']:.2f}$ SO-TPE), while Constrained TPE contributes in **${hpb['by_optimizer']['constrained_tpe']['seed_only_present_pct']:.1f}\\%$** (mean ${hpb['by_optimizer']['constrained_tpe']['seed_only_mean_points']:.2f}$ points).",
            f"5. **Constrained TPE Feasibility Degradation**: While Constrained TPE satisfies $\\le 145\\,\\mu\\text{{s}}$ on ${ctpe.search_feasible_count}/{ctpe.n_replicates}$ searches during online 30-call search timing, only **${ctpe.benchmark_feasible_count}/{ctpe.n_replicates}$ (${ctpe.benchmark_feasible_pct}\\%$)** remain feasible under 1,000-call benchmark verification (${ds.ctpe_depth6_feasible}/{ds.ctpe_depth6_count}$ depth-6 selections feasible at ${ds.ctpe_depth6_latency_mean:.2f} \\pm {ds.ctpe_depth6_latency_sd:.2f}\\,\\mu\\text{{s}}$; ${ds.ctpe_depth7_feasible}/{ds.ctpe_depth7_count}$ depth-7 selections feasible at ${ds.ctpe_depth7_latency_mean:.2f} \\pm {ds.ctpe_depth7_latency_sd:.2f}\\,\\mu\\text{{s}}$). Plausible factors include noisy 30-call search measurements near the boundary, protocol differences (30 calls without warmup vs. 1,000 calls with warmup), and winner's-curse selection bias near the $145\\,\\mu\\text{{s}}$ threshold.",
            "6. **Historical Latency Comparison Caveat**: Both historical `v1.0.0` scripts (`phase5_confirmation.py` and `phase5_benchmarks.py`) applied Win32 CPU core 0 affinity pinning. Consequently, the shift between historical single-run latency snapshots and the prospective benchmark session reflects between-session system timing drift ($\\text{SD} = " + f"{ds.between_session_latency_sd_mean:.2f}\\,\\mu\\text{{s}}$ across repeated sessions) rather than unpinned execution.",
            "",
            "### 7.4 Computational Budget Ledger and Wall-Clock Cost",
            "",
            "The table below reconciles every model fit, distinct coordinate, timed single-row inference, untimed warmup call, and per-replicate search wall-clock duration across the historical `v1.0.0` baseline and the prospective `Revision-v2` campaign (`computational_budget.json` and `optimizer_replicates.csv`):",
            "",
            "| Phase / Method Component | Search Reps | XGBoost Fits | Distinct Coords | Timed Inferences | Warmup Calls | Total Inferences | Mean Fit Time (ms) | Wall Time / Rep (s, Mean $\\pm$ SD) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            "| **Panel A: Historical Single-Campaign Baseline (`v1.0.0`, 160 DOE + Confirmation Fits)** | | | | | | | | |",
            "| Phase 1: $2^4$ Factorial + 4 Center ($5$ Blocks) | $1$ | $100$ | $17$ | $100,000$ | $10,000$ | $110,000$ | — | — |",
            "| Phase 2: Face-Centered Axial ($5$ Blocks) | $1$ | $40$ | $8$ | $40,000$ | $4,000$ | $44,000$ | — | — |",
            "| Phase 5: Confirmation ($\\mathbf{x}^*_{\\text{MO}}$ & $\\mathbf{x}^*_{\\text{SO}}$, $m=10$) | $2$ | $20$ | $2$ | $20,000$ | $2,000$ | $22,000$ | — | — |",
            f"| **Panel B: Prospective Revision-v2 Campaign (Total = {ds.total_model_fits:,} Fits, {ds.total_timed_inferences:,} Timed Inferences)** | | | | | | | | |",
            f"| Repeated Pre-Planned DOE (MO + SO Shared) | $20$ | ${int(cb['repeated_doe_design_model_fits']):,}$ | $25/\\text{{rep}}$ | ${int(cb['repeated_doe_design_model_fits']) * 30:,}$ | ${int(cb['repeated_doe_design_model_fits']) * 10:,}$ | ${int(cb['repeated_doe_design_model_fits']) * 40:,}$ | ${mt['repeated_preplanned_doe_multi_objective']['train_ms_per_fit_mean']:.1f}$ | ${mt['repeated_preplanned_doe_multi_objective']['wall_clock_s_mean']:.2f} \\pm {mt['repeated_preplanned_doe_multi_objective']['wall_clock_s_sd']:.2f}$ |",
            f"| Multi-Objective TPE ($140$ Trials/Rep) | $20$ | ${int(cb['search_model_fits']) // 4:,}$ | $140/\\text{{rep}}$ | ${(int(cb['search_model_fits']) // 4) * 30:,}$ | ${(int(cb['search_model_fits']) // 4) * 10:,}$ | ${(int(cb['search_model_fits']) // 4) * 40:,}$ | ${mt['multi_objective_tpe']['train_ms_per_fit_mean']:.1f}$ | ${mt['multi_objective_tpe']['wall_clock_s_mean']:.2f} \\pm {mt['multi_objective_tpe']['wall_clock_s_sd']:.2f}$ |",
            f"| Constrained TPE ($140$ Trials/Rep) | $20$ | ${int(cb['search_model_fits']) // 4:,}$ | $140/\\text{{rep}}$ | ${(int(cb['search_model_fits']) // 4) * 30:,}$ | ${(int(cb['search_model_fits']) // 4) * 10:,}$ | ${(int(cb['search_model_fits']) // 4) * 40:,}$ | ${mt['constrained_tpe']['train_ms_per_fit_mean']:.1f}$ | ${mt['constrained_tpe']['wall_clock_s_mean']:.2f} \\pm {mt['constrained_tpe']['wall_clock_s_sd']:.2f}$ |",
            f"| Single-Objective TPE ($140$ Trials/Rep) | $20$ | ${int(cb['search_model_fits']) // 4:,}$ | $140/\\text{{rep}}$ | ${(int(cb['search_model_fits']) // 4) * 30:,}$ | ${(int(cb['search_model_fits']) // 4) * 10:,}$ | ${(int(cb['search_model_fits']) // 4) * 40:,}$ | ${mt['single_objective_tpe']['train_ms_per_fit_mean']:.1f}$ | ${mt['single_objective_tpe']['wall_clock_s_mean']:.2f} \\pm {mt['single_objective_tpe']['wall_clock_s_sd']:.2f}$ |",
            f"| Random Search ($140$ Trials/Rep) | $20$ | ${int(cb['search_model_fits']) // 4:,}$ | $140/\\text{{rep}}$ | ${(int(cb['search_model_fits']) // 4) * 30:,}$ | ${(int(cb['search_model_fits']) // 4) * 10:,}$ | ${(int(cb['search_model_fits']) // 4) * 40:,}$ | ${mt['random_search']['train_ms_per_fit_mean']:.1f}$ | ${mt['random_search']['wall_clock_s_mean']:.2f} \\pm {mt['random_search']['wall_clock_s_sd']:.2f}$ |",
            f"| Split-42 Full DOE Baseline ($25 + 2$ Opt) | $1$ | ${int(cb['matched_historical_doe_candidate_fits'])}$ | $27$ | ${int(cb['matched_historical_doe_candidate_fits']) * 30:,}$ | ${int(cb['matched_historical_doe_candidate_fits']) * 10:,}$ | ${int(cb['matched_historical_doe_candidate_fits']) * 40:,}$ | — | — |",
            f"| Primary Latency Sessions ($5 \\times 122$ Records) | $5$ | ${int(cb['primary_latency_refit_fits']):,}$ | ${ds.distinct_config_hashes}$ | ${int(cb['primary_latency_timed_prediction_calls_maximum']):,}$ | ${int(cb['primary_latency_warmup_prediction_calls_maximum']):,}$ | ${int(cb['primary_latency_timed_prediction_calls_maximum']) + int(cb['primary_latency_warmup_prediction_calls_maximum']):,}$ | — | — |",
            f"| Holdout Retraining ($122$ Records $\\times 20$ Seeds) | — | ${int(cb['final_retraining_fits']):,}$ | ${ds.distinct_config_hashes}$ | $0$ | $0$ | $0$ | — | — |",
            f"| **Prospective Campaign Total** | — | **{ds.total_model_fits:,}** | — | **{ds.total_timed_inferences:,}** | **{ds.total_warmup_inferences:,}** | **{ds.total_all_inferences:,}** | — | — |",
        ]
    )
    sections["SECTION_7_BENCHMARKS"] = "\n".join(s7_rows)

    # 9. SECTION_8_9_DISCUSSION_AND_ARTIFACTS
    macro_count = sum(
        1 for line in render_macros(ds).splitlines() if line.startswith("\\newcommand")
    )
    sections["SECTION_8_9_DISCUSSION_AND_ARTIFACTS"] = "\n".join(
        [
            f"1. **Parametric Variance Attribution vs. Adaptive Coordinate Exploration**: Pre-planned blocked DOE + RSM and adaptive Bayesian optimization (TPE) answer complementary questions under a fixed 140-evaluation budget. Blocked FCCD replicates 25 structured coordinates across 5 splits to decompose variance into main effects, interactions, quadratic curvature ($F = {p1['f_curvature']:,.2f}$ against center pure error; $F = {cs['f_block_adj_pooled']:,.2f}$ against 79-df block-adjusted pooled error), seed-block variance ($\\text{{ICC}} = {icc_anova_pct:.2f}\\%$), and structural lack of fit ($F = {lof['Y1']['F_LoF']:.2f}$), while yielding deterministic or near-deterministic compromise selections ($\\mathbf{{x}}^*_{{\\text{{MO}}}}$ and $\\mathbf{{x}}^*_{{\\text{{SO}}}}$). Conversely, TPE concentrates all 140 evaluations into distinct configurations without replication, achieving lower unconstrained single-objective test RMSE (${so_tpe.test_rmse_mean:.5f}$ vs. ${doe_so.test_rmse_mean:.5f}$) and higher split-42 candidate hypervolume (${hvs['motpe_dev_060_mean']:.4f}$ vs. ${hvs['fdoe_dev_060']:.4f}$) by avoiding repeated evaluations at the high-loss $\\eta = 0.01$ boundary (${ds.doe_low_eta_run_count}/140 = {ds.doe_low_eta_run_pct:.2f}\\%$ of FCCD runs) and by exploring deeper trees ($d \\in \\{{7, 8, 9\\}}$) at higher training wall-clock cost (${mt['single_objective_tpe']['wall_clock_s_mean']:.2f}\\,\\text{{s}}$ vs. ${mt['repeated_preplanned_doe_multi_objective']['wall_clock_s_mean']:.2f}\\,\\text{{s}}$ per search).",
            f"2. **Value of Stochastic Nuisance Blocking**: Blocking across {len(ds.block_seeds)} data-partition seeds absorbed $\\text{{ICC}} = {icc_anova_pct:.2f}\\%$ of residual variance ($\\sigma_{{\\text{{block}}}} \\approx {sigma_block:.4f}$ RMSE), preventing partition noise from confounding hyperparameter comparisons.",
            f"3. **Structural Lack of Fit and Surrogate Optimism**: Because decision tree ensembles exhibit diminishing returns at deeper levels ($d \\ge 6$) and sharp exponential penalties at $\\eta = 0.01$, a global second-order polynomial interpolated across $d \\in \\{{3, 6, 9\\}}$ exhibits statistically significant lack of fit ($\\text{{RMS}} = {lof['Y1']['sqrt_MS_LoF']:.4f}$ RMSE) and under-predicts validation RMSE at depth {cube['depth']} by ${so_bias:+.4f}$ RMSE. Satterthwaite prediction intervals widen for variance heterogeneity across degrees of freedom but cannot correct deterministic polynomial bias.",
            f"4. **External Validity and Scope Limitations**: All empirical conclusions are scoped to a single tabular regression dataset (California Housing, $N = 20,640$), a single gradient boosting library (`XGBoost 3.2.0`, `n_estimators = 100`), a 4-hyperparameter domain ($\\eta, d, s, \\lambda$), and single-core Windows CPU inference timing (`Intel Core i7-12650H`). While the 12-point holdout Pareto frontier spans both Repeated DOE MO and TPE selections (with ${hpb['by_optimizer']['repeated_preplanned_doe_multi_objective']['seed_only_present_pct']:.1f}\\%$ bootstrap presence for DOE-MO, MO-TPE, and SO-TPE), generalizing these findings to higher-dimensional spaces ($k > 6$), larger datasets, or GPU/batch serving requires fractional factorial screening ($2^{{k-p}}$), sequential region zooming, or hybrid DOE + Bayesian pipelines (`docs/proposed_followup_experiments.md`).",
            "",
            "---",
            "",
            "## 9. Repository Artifacts, Provenance, and AI-Assistance Disclosure",
            "",
            "- **LaTeX Manuscript & Compiled PDF**: `report.tex` and `report.pdf` (compiled via Tectonic with zero unresolved references or layout overflows).",
            f"- **Auto-Generated Statistical Macros & Tables**: `results/macros.tex` ({macro_count} macros) and `tables/*.tex`, generated deterministically via `python scripts/generate_report_artifacts.py`.",
            "- **Single-Source-of-Truth Reporting & Manifest**: `scripts/reporting_data.py`, `scripts/generate_research_reporting.py`, `docs/generated/scientific_results_manifest.json`, `docs/scientific_claims_audit.md`, and `docs/proposed_followup_experiments.md`.",
            "- **Immutable Evidence Provenance**: Historical `v1.0.0` baseline artifacts (`results/runs.csv`, `results/phase1.json`, `results/phase3.json`, `results/confirmation.json`, `results/benchmark.csv`) were committed at SHA `cd63135d2b70a0ee324555c11b67a541b32b98a4`; prospective `Revision-v2` campaign artifacts (`results/revision_v2/full_run_001/`) were committed at SHA `421cf24229f1f5fba26cdb3103ec989f9a427c96` (`run_manifest.json` records the pre-commit parent SHA `45fe6f7ca8eb66b479edfb68d4b7c0c154a5d624` at campaign launch). All 40 canonical evidence files are protected by SHA-256 verification in `docs/evidence_manifest.json`.",
            "- **Verification & Audit Suite**: `python scripts/audit_scientific_consistency.py`, `python scripts/generate_research_reporting.py --check`, `python scripts/generate_report_artifacts.py --check`, and `pytest`.",
            "- **AI-Assistance Disclosure**: AI coding and scientific auditing assistants (OpenAI Codex and Google DeepMind Antigravity) were used for pipeline refactoring, deterministic reporting-layer generation, statistical sensitivity verification, and manuscript auditing against frozen experimental artifacts. All experimental designs, hyperparameter bounds, frozen evidence files, and scientific conclusions were verified against primary row-level records by the author.",
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
        "# Pre-Planned Response Surface Methodology and Face-Centered Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking",
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


def _allowed_report_tex_specification_floats(ds: ReportingDataset) -> set[str]:
    """Build the exact source-derived set of design/specification constants permitted as literals in report.tex."""
    f_cfg = ds.config["factors"]
    x1_min, x1_max = float(f_cfg["x1"]["min"]), float(f_cfg["x1"]["max"])
    x1_cent = float(np.sqrt(x1_min * x1_max))
    x3_min, x3_max = float(f_cfg["x3"]["min"]), float(f_cfg["x3"]["max"])
    x3_cent = 0.5 * (x3_min + x3_max)
    x4_min, x4_max = float(f_cfg["x4"]["min"]), float(f_cfg["x4"]["max"])
    x4_cent = float(np.sqrt(x4_min * x4_max))

    d_cfg = ds.config["desirability"]
    y1_l, y1_u = float(d_cfg["Y1_RMSE"]["L"]), float(d_cfg["Y1_RMSE"]["U"])
    y2_l, y2_u = float(d_cfg["Y2_Latency"]["L"]), float(d_cfg["Y2_Latency"]["U"])

    r1_rmse, r1_lat = ds.hv_ref_primary
    r2_rmse, r2_lat = ds.hv_ref_secondary

    allowed = {
        # Factor bounds & centers from config.yaml
        f"{x1_min:.2f}",
        f"{x1_max:.2f}",
        f"{x1_min:.4f}",
        f"{x1_cent:.4f}",
        f"{x1_max:.4f}",
        f"{abs(np.log(x1_min)):.4f}",
        f"{abs(np.log(x1_max)):.4f}",
        f"{x3_min:.2f}",
        f"{x3_cent:.2f}",
        f"{x3_max:.2f}",
        f"{x3_min:.4f}",
        f"{x3_cent:.4f}",
        f"{x3_max:.4f}",
        f"{x4_min:.2f}",
        f"{x4_max:.2f}",
        f"{x4_min:.4f}",
        f"{x4_cent:.4f}",
        f"{x4_max:.4f}",
        f"{abs(np.log(x4_min)):.4f}",
        f"{abs(np.log(x4_max)):.4f}",
        # Desirability specification bounds & grid levels from config.yaml
        f"{y1_l:.3f}",
        f"{y1_u:.3f}",
        f"{y2_l:.1f}",
        f"{y2_u:.1f}",
        "0.20",
        "0.21",
        "0.22",
        "0.25",
        "0.30",
        "0.08",
        "0.9",
        "0.90",
        # Hypervolume reference points from config.yaml
        f"{r1_rmse:.2f}",
        f"{r1_lat:.1f}",
        f"{r2_rmse:.2f}",
        f"{r2_lat:.1f}",
        # Methodological / statistical thresholds & restricted-domain cutoff
        "0.0",
        "1.0",
        "0.5",
        "0.033",
        "0.05",
        "0.025",
        "0.0001",
        "0.005",
        "0.015",
    }
    return allowed


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

    # 4. Verify all \num... and \vecB... and \matB... and \eigVec... macros used in report.tex are defined in results/macros.tex
    defined_macros = dict(
        re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.+)\}", macros_text)
    )
    used_macros = set(re.findall(r"\\((?:num|vecB|matB|eigVec|env)[A-Za-z]+)\b", tex_text))
    undefined = sorted(used_macros - set(defined_macros.keys()))
    if undefined:
        errors.append(f"Macros used in report.tex but undefined in results/macros.tex: {undefined}")

    # 5. Enforce that no empirical result floating-point numbers are hardcoded in report.tex prose
    allowed_tex_floats = _allowed_report_tex_specification_floats(ds)
    unpermitted_tex_floats: List[str] = []
    for line_no, raw_line in enumerate(tex_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped.startswith("%"):
            continue
        # Strip inline LaTeX comments
        line_clean = re.sub(r"(?<!\\)%.*$", "", raw_line)
        # Strip LaTeX layout/formatting commands and explicit software version tags
        line_clean = re.sub(r"\\renewcommand\{\\arraystretch\}\{[0-9.]+\}", "", line_clean)
        line_clean = re.sub(
            r"\b(?:v1\.0\.0|Python\s+3\.11(?:\.9)?|XGBoost\s+3\.2(?:\.0)?|Scikit-Learn\s+1\.8(?:\.0)?|Statsmodels\s+0\.15(?:\.0)?|Scipy\s+1\.17(?:\.1)?|Optuna\s+5\.0(?:\.0)?)\b",
            "",
            line_clean,
            flags=re.IGNORECASE,
        )
        for m in re.finditer(r"(?<![A-Za-z\\])\d+\.\d+(?![A-Za-z])", line_clean):
            # Ignore layout lengths like 0.48\textwidth, 0.35em, 1.2pt
            tail = line_clean[m.start() : m.end() + 18]
            if re.match(r"^\d+\.\d+\s*\\?(?:textwidth|linewidth|columnwidth|em|pt|mm|cm|in|ex)\b", tail):
                continue
            val_str = m.group(0)
            if val_str not in allowed_tex_floats:
                unpermitted_tex_floats.append(f"L{line_no}: {val_str}")
    if unpermitted_tex_floats:
        errors.append(
            "Hardcoded empirical floating-point numbers found in report.tex prose "
            f"(must use generated macros from results/macros.tex): {unpermitted_tex_floats}"
        )

    # 6. Check that no unsupported TOST equivalence claims appear when rmse_equivalence_margin is None
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
