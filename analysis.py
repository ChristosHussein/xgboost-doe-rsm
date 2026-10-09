"""
analysis.py - Rigorous Statistical Analysis & Response Surface Decomposition
=============================================================================
Implements Tasks 1-5 per fix.md:
  Task 1: Programmatic matrix generation, canonical decomposition, wild bootstrap, classification.
  Task 2: Box-constrained optimization over the cube (depth as integer), ridge trace, depth table.
  Task 3: Nested-model Lack-of-Fit F-test (df 10, 111), Box-Cox transformations, effect sizes.
  Task 4: Diagnostics (Shapiro-Wilk, Levene, Breusch-Pagan, Brown-Forsythe, HC3, Cook's D, Durbin-Watson).
  Task 5: REML MixedLM and ANOVA ICC estimators for block stochastic nuisance.
"""

import json
import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
import yaml

from pipeline import decode_factors, encode_factors, CONFIG
from scientific_stats import blocked_lack_of_fit_decomposition

Q = ["x1", "x2", "x3", "x4"]


def build_design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Builds full second-order design matrix with block indicators."""
    X = pd.DataFrame(index=df.index)
    for q in Q:
        X[q] = df[q]
    for q in Q:
        X[q + "_sq"] = df[q] ** 2
    for i in range(4):
        for j in range(i + 1, 4):
            X[f"{Q[i]}_{Q[j]}"] = df[Q[i]] * df[Q[j]]
    blk = pd.get_dummies(df["block"], prefix="blk", drop_first=True).astype(float)
    return sm.add_constant(pd.concat([X, blk], axis=1))


def canonical_decomposition(fit: sm.regression.linear_model.RegressionResultsWrapper) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Extracts b0 (block-averaged intercept), linear vector b, and symmetric B matrix.
    Off-diagonals of B are strictly beta_ij / 2.
    """
    p = fit.params
    b = p[Q].values
    B = np.diag(p[[q + "_sq" for q in Q]].values)
    for i in range(4):
        for j in range(i + 1, 4):
            B[i, j] = B[j, i] = p[f"{Q[i]}_{Q[j]}"] / 2.0

    blk_cols = [c for c in p.index if c.startswith("blk_")]
    b0 = float(p["const"] + (p[blk_cols].sum() if len(blk_cols) > 0 else 0.0) / 5.0)
    return b0, b, B


def predict_block_averaged(fit: sm.regression.linear_model.RegressionResultsWrapper, x: np.ndarray) -> float:
    """Evaluates the fitted model at coded coordinate x averaging block effects equally."""
    x1, x2, x3, x4 = x
    row = {
        "const": 1.0,
        "x1": x1, "x2": x2, "x3": x3, "x4": x4,
        "x1_sq": x1**2, "x2_sq": x2**2, "x3_sq": x3**2, "x4_sq": x4**2,
        "x1_x2": x1*x2, "x1_x3": x1*x3, "x1_x4": x1*x4,
        "x2_x3": x2*x3, "x2_x4": x2*x4, "x3_x4": x3*x4,
    }
    for b in [2, 3, 4, 5]:
        row[f"blk_{b}"] = 1.0 / 5.0
    val = sum(fit.params[k] * row[k] for k in fit.params.index)
    return float(val)


def prediction_se(fit: sm.regression.linear_model.RegressionResultsWrapper, XtXi: np.ndarray, x: np.ndarray) -> float:
    """Computes standard error of prediction SE(y_hat) at coded coordinate x."""
    x1, x2, x3, x4 = x
    row = [1.0, x1, x2, x3, x4, x1**2, x2**2, x3**2, x4**2,
           x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4,
           0.2, 0.2, 0.2, 0.2]
    r = np.array(row)
    return float(np.sqrt(fit.mse_resid * (r @ XtXi @ r)))


def wild_bootstrap_eigenvalues(X: pd.DataFrame, y: pd.Series, fit: sm.regression.linear_model.RegressionResultsWrapper,
                               n_reps: int = 2000, seed: int = 0) -> Tuple[np.ndarray, float]:
    """Performs Rademacher wild bootstrap on residuals to quantify eigenvalue uncertainty under heteroscedasticity."""
    rng = np.random.default_rng(seed)
    res = fit.resid.values
    fv = fit.fittedvalues.values
    out = []
    for _ in range(n_reps):
        boot_y = fv + res * rng.choice([-1.0, 1.0], size=len(res))
        f_boot = sm.OLS(boot_y, X).fit()
        _, _, B_boot = canonical_decomposition(f_boot)
        out.append(np.linalg.eigvalsh(B_boot))
    out = np.array(out)
    percentiles = np.percentile(out, [2.5, 50.0, 97.5], axis=0)
    frac_le_zero = float((out[:, 0] <= 0).mean())
    return percentiles, frac_le_zero


def run_phase1_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Phase 1: Screening ANOVA and Curvature test."""
    df_p1 = df[df["phase"].str.startswith("Phase1")].copy()
    fact = df_p1[df_p1["phase"] == "Phase1_Factorial"]
    cent = df_p1[df_p1["phase"] == "Phase1_Center"]

    yF = fact["val_rmse"].values
    yC = cent["val_rmse"].values
    nF = len(yF)
    nC = len(yC)
    yF_bar = float(np.mean(yF))
    yC_bar = float(np.mean(yC))
    diff = yF_bar - yC_bar
    ss_curv = float((nF * nC * (diff**2)) / (nF + nC))

    pe_center = cent.groupby("block")["val_rmse"].apply(lambda s: np.sum((s - s.mean())**2)).sum()
    df_pe_center = nC - len(cent["block"].unique())  # 20 - 5 = 15
    ms_pe_center = float(pe_center / df_pe_center)
    f_curv = float(ss_curv / ms_pe_center)
    p_curv = float(1.0 - stats.f.cdf(f_curv, 1, df_pe_center))

    # Phase 1 interaction model
    formula = ("val_rmse ~ C(block) + x1+x2+x3+x4 + "
               "x1:x2 + x1:x3 + x1:x4 + x2:x3 + x2:x4 + x3:x4")
    fit_p1 = smf.ols(formula, df_p1).fit()
    anova_p1 = sm.stats.anova_lm(fit_p1, typ=3)

    return {
        "n_factorial": nF,
        "n_center": nC,
        "yF_bar": yF_bar,
        "yC_bar": yC_bar,
        "diff_F_minus_C": diff,
        "ss_curvature": ss_curv,
        "ms_pe_center": ms_pe_center,
        "df_pe_center": df_pe_center,
        "f_curvature": f_curv,
        "p_curvature": p_curv,
        "curvature_significant": bool(p_curv < 0.05),
        "anova_table": anova_p1,
    }


def run_phase3_canonical_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Task 1 & Task 2: Second-order canonical analysis, ridge analysis, box optimization."""
    X = build_design_matrix(df)
    y = df["val_rmse"]
    fit = sm.OLS(y, X).fit()
    XtXi = np.linalg.inv(X.T @ X)

    b0, b, B = canonical_decomposition(fit)
    lam, M = np.linalg.eigh(B)

    # Verification assertions
    assert np.isclose(np.trace(B), lam.sum()), "Trace(B) does not equal sum of eigenvalues"
    x0 = -0.5 * np.linalg.solve(B, b)
    assert np.allclose(b + 2 * B @ x0, 0, atol=1e-10), "x0 does not solve gradient=0"
    y0_hat = b0 + 0.5 * float(b @ x0)
    assert np.isclose(y0_hat, predict_block_averaged(fit, x0)), "y0_hat does not match predict_block_averaged"

    # Type III SS cross-check
    ss3_manual = fit.params**2 / np.diag(XtXi)
    df_aug = df.copy()
    for q in Q: df_aug[q + "_sq"] = df_aug[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{Q[i]}_{Q[j]}"] = df_aug[Q[i]] * df_aug[Q[j]]
    fit_formula = smf.ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                          "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()
    anova_tab = sm.stats.anova_lm(fit_formula, typ=3)

    # Wild bootstrap
    boot_pcts, frac_le_zero = wild_bootstrap_eigenvalues(X, y, fit, n_reps=2000, seed=42)

    # Classification rule
    if boot_pcts[0, 0] > 0:
        surface_type = "Convex / Unique Local Minimum"
    else:
        surface_type = "Stationary or Rising Ridge (curvature in minor directions not distinguishable from zero)"

    # Extrapolation flag
    max_abs_x0 = float(np.max(np.abs(x0)))
    is_extrapolated = bool(max_abs_x0 > 1.0)
    dist_coded = float(np.linalg.norm(x0))

    # Natural mapping of x0 (with clamped natural values for unconstrained extrapolation)
    eta_0, depth_0, sub_0, lam_0 = decode_factors(x0, clip_domain=True)

    # Task 2: Box-constrained optimization over the cube [-1, 1]^4 with integer depth 3..9
    depth_results = []
    rng = np.random.default_rng(0)
    best_overall_rmse = 1e9
    best_overall_coord = None
    best_overall_depth = None

    for d in range(3, 10):
        x2 = (d - 6.0) / 3.0
        best_fun = 1e9
        best_z = None
        for _ in range(30):
            init = rng.uniform(-1, 1, 3)
            res = minimize(
                lambda z: predict_block_averaged(fit, np.array([z[0], x2, z[1], z[2]])),
                init,
                bounds=[(-1.0, 1.0)] * 3,
                method="L-BFGS-B"
            )
            if res.fun < best_fun:
                best_fun = res.fun
                best_z = res.x
        x_d = np.array([best_z[0], x2, best_z[1], best_z[2]])
        se_d = prediction_se(fit, XtXi, x_d)
        depth_results.append({
            "depth": d,
            "x1": float(best_z[0]),
            "x2": float(x2),
            "x3": float(best_z[1]),
            "x4": float(best_z[2]),
            "pred_rmse": float(best_fun),
            "se": float(se_d),
        })
        if best_fun < best_overall_rmse:
            best_overall_rmse = best_fun
            best_overall_coord = x_d
            best_overall_depth = d

    df_depth_opt = pd.DataFrame(depth_results)

    # Task 2: Ridge trace for R in linspace(0, 2, 21)
    ridge_records = []
    for R in np.linspace(0.0, 2.0, 21):
        if R == 0.0:
            x_r = np.zeros(4)
            y_r = predict_block_averaged(fit, x_r)
            se_r = prediction_se(fit, XtXi, x_r)
        else:
            cons = ({"type": "ineq", "fun": lambda x, R=R: R**2 - np.sum(x**2)})
            r_opt = minimize(
                lambda x: predict_block_averaged(fit, x),
                np.zeros(4),
                constraints=cons,
                method="SLSQP"
            )
            x_r = r_opt.x
            y_r = float(r_opt.fun)
            se_r = prediction_se(fit, XtXi, x_r)
        ridge_records.append({
            "R": float(R),
            "x1": float(x_r[0]),
            "x2": float(x_r[1]),
            "x3": float(x_r[2]),
            "x4": float(x_r[3]),
            "y_hat": float(y_r),
            "se": float(se_r),
        })
    df_ridge = pd.DataFrame(ridge_records)

    eta_box, depth_box, sub_box, lam_box = decode_factors(best_overall_coord, clip_domain=True)

    return {
        "b0": b0,
        "b": b.tolist(),
        "B": B.tolist(),
        "eigenvalues": lam.tolist(),
        "eigenvectors": M.tolist(),
        "trace_B": float(np.trace(B)),
        "sum_eigenvalues": float(lam.sum()),
        "stationary_point_coded": x0.tolist(),
        "y0_hat_unconstrained": y0_hat,
        "is_extrapolated": is_extrapolated,
        "distance_coded": dist_coded,
        "stationary_natural_clamped": {
            "eta": eta_0, "depth": depth_0, "subsample": sub_0, "reg_lambda": lam_0
        },
        "surface_classification": surface_type,
        "bootstrap_percentiles": boot_pcts.tolist(),
        "fraction_min_eigenvalue_le_zero": frac_le_zero,
        "constrained_optimum_cube": {
            "x": best_overall_coord.tolist(),
            "pred_rmse": float(best_overall_rmse),
            "depth": best_overall_depth,
            "natural": {
                "eta": eta_box, "depth": depth_box, "subsample": sub_box, "reg_lambda": lam_box
            }
        },
        "integer_depth_table": df_depth_opt,
        "ridge_table": df_ridge,
        "fit": fit,
        "XtXi": XtXi,
    }


def run_task3_lack_of_fit(df: pd.DataFrame) -> Dict[str, Any]:
    """Rank-aware blocked lack-of-fit decomposition for both responses."""
    df_aug = df.copy()
    for q in Q:
        df_aug[q + "_sq"] = df_aug[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{Q[i]}_{Q[j]}"] = df_aug[Q[i]] * df_aug[Q[j]]

    df_aug_restr = df_aug[df_aug["x1"] >= -0.5].copy()

    def matrices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        reduced = build_design_matrix(frame).to_numpy(dtype=float)
        additive = np.column_stack([
            np.ones(len(frame)),
            pd.get_dummies(frame["block"], drop_first=True, dtype=float).to_numpy(),
            pd.get_dummies(frame["point_id"], drop_first=True, dtype=float).to_numpy(),
        ])
        cell_key = frame["block"].astype(str) + ":" + frame["point_id"].astype(str)
        cell_means = pd.get_dummies(cell_key, dtype=float).to_numpy()
        return reduced, additive, cell_means

    full_matrices = matrices(df_aug)
    restricted_matrices = matrices(df_aug_restr)
    full_y1 = blocked_lack_of_fit_decomposition(
        df_aug["val_rmse"].to_numpy(), *full_matrices
    )
    full_y2 = blocked_lack_of_fit_decomposition(
        df_aug["latency_us_median"].to_numpy(), *full_matrices
    )
    restricted_y1 = blocked_lack_of_fit_decomposition(
        df_aug_restr["val_rmse"].to_numpy(), *restricted_matrices
    )

    def response_payload(decomposition: Dict[str, Any], boxcox_lambda=None) -> Dict[str, Any]:
        components = decomposition["components"]
        historical = decomposition["tests"]["structural_vs_pooled_additive"]
        sensitivity = decomposition["tests"]["structural_vs_center_pure_error_sensitivity"]
        structural = components["structural_lack_of_fit"]
        pooled = components["pooled_additive_residual"]
        payload = {
            **components,
            "historical_pooled_additive_test": historical,
            "center_pure_error_sensitivity": sensitivity,
            "interaction_test": decomposition["tests"]["interaction_vs_center_pure_error"],
            "structural_vs_interaction_test": decomposition["tests"]["structural_vs_treatment_by_block"],
            "identity": decomposition["identity"],
            "n_observations": decomposition["n_observations"],
            "assumptions": {
                "historical_pooled_additive_test": (
                    "Uses treatment-by-block interaction plus center-cell residual as the denominator."
                ),
                "center_pure_error_sensitivity": (
                    "Sensitivity only; requires center-point seed variance to represent the full domain."
                ),
            },
            "legacy_compatibility": {
                "df_PE_alias": "pooled_additive_residual.df",
                "SS_PE_alias": "pooled_additive_residual.ss",
            },
            # Stable aliases retained for historical artifact readers. They are
            # explicitly the pooled additive residual, not pure error.
            "df_LoF": structural["df"],
            "df_PE": pooled["df"],
            "SS_LoF": structural["ss"],
            "SS_PE": pooled["ss"],
            "MS_LoF": structural["ms"],
            "MS_PE": pooled["ms"],
            "F_LoF": historical["f_statistic"],
            "p_LoF": historical["pvalue"],
            "sqrt_MS_LoF": float(np.sqrt(structural["ms"])),
            "sqrt_MS_PE": float(np.sqrt(pooled["ms"])),
        }
        if boxcox_lambda is not None:
            payload["boxcox_lambda"] = float(boxcox_lambda)
        return payload

    # Box-Cox
    _, lam_y1 = stats.boxcox(df["val_rmse"])
    _, lam_y2 = stats.boxcox(df["latency_us_median"])
    y1_payload = response_payload(full_y1, lam_y1)
    y2_payload = response_payload(full_y2, lam_y2)
    restricted_payload = response_payload(restricted_y1)
    restricted_payload["ss_lof_reduction_pct"] = float(
        (full_y1["components"]["structural_lack_of_fit"]["ss"]
         - restricted_y1["components"]["structural_lack_of_fit"]["ss"])
        / full_y1["components"]["structural_lack_of_fit"]["ss"]
        * 100.0
    )
    restricted_payload["reduction_interpretation"] = (
        "Descriptive change after refitting a restricted domain; not a causal contribution."
    )
    return {
        "rank_diagnostics": {
            "full": full_y1["models"],
            "restricted": restricted_y1["models"],
        },
        "Y1": y1_payload,
        "Y1_restricted": restricted_payload,
        "Y2": y2_payload,
    }


def run_task4_diagnostics(df: pd.DataFrame) -> Dict[str, Any]:
    """Task 4: Diagnostics, HC3 robust inference, outlier analysis, independence."""
    X = build_design_matrix(df)
    y = df["val_rmse"]
    fit = sm.OLS(y, X).fit()
    infl = fit.get_influence()
    r = infl.resid_studentized_external

    # Normality
    W, p_norm = stats.shapiro(r)

    # Levene across blocks
    groups_block = [fit.resid[df["block"] == b].values for b in sorted(df["block"].unique())]
    block_info = []
    for b_idx, g in enumerate(groups_block, 1):
        block_info.append({"block": b_idx, "n": len(g), "variance": float(np.var(g, ddof=1))})
    lev_stat, lev_pval = stats.levene(*groups_block, center="median")

    # Brown-Forsythe across design points (point_id)
    groups_point = [fit.resid[df["point_id"] == p].values for p in sorted(df["point_id"].unique())]
    bf_stat, bf_pval = stats.levene(*groups_point, center="median")

    # Breusch-Pagan across X
    bp_stat, bp_pval, _, _ = sm.stats.diagnostic.het_breuschpagan(fit.resid, X)

    # Cook's distance & outliers
    cooks = infl.cooks_distance[0]
    n = len(df)
    thresh_4n = 4.0 / n
    outlier_mask = (np.abs(r) > 3.0) | (cooks > thresh_4n)
    outliers_df = df[outlier_mask].copy()
    outliers_df["studentized_resid"] = r[outlier_mask]
    outliers_df["cooks_d"] = cooks[outlier_mask]

    # Refit without top outlier
    max_cook_idx = int(np.argmax(cooks))
    df_no_outlier = df.drop(index=max_cook_idx).reset_index(drop=True)
    X_no = build_design_matrix(df_no_outlier)
    fit_no = sm.OLS(df_no_outlier["val_rmse"], X_no).fit()

    # Independence via Durbin-Watson, Ljung-Box, and Runs Test in execution order
    order_idx = np.argsort(df["run_order"].values)
    ordered_resid = fit.resid.values[order_idx]
    dw_stat = float(durbin_watson(ordered_resid))
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.sandbox.stats.runs import runstest_1samp
    lb_res = acorr_ljungbox(ordered_resid, lags=[5, 10], return_df=True)
    lb_stat = float(lb_res["lb_stat"].iloc[0])
    lb_p = float(lb_res["lb_pvalue"].iloc[0])
    runs_z, runs_p = runstest_1samp(ordered_resid)

    # HC3 robust inference
    fit_hc3 = fit.get_robustcov_results("HC3")

    return {
        "shapiro_W": float(W),
        "shapiro_p": float(p_norm),
        "normality_rejected_at_05": bool(p_norm < 0.05),
        "normality_rejected_at_01": bool(p_norm < 0.01),
        "block_variances": block_info,
        "levene_stat": float(lev_stat),
        "levene_p": float(lev_pval),
        "brown_forsythe_points_stat": float(bf_stat),
        "brown_forsythe_points_p": float(bf_pval),
        "breusch_pagan_stat": float(bp_stat),
        "breusch_pagan_p": float(bp_pval),
        "max_cooks_d": float(np.max(cooks)),
        "thresh_4n": float(thresh_4n),
        "outlier_count": int(outlier_mask.sum()),
        "durbin_watson": dw_stat,
        "ljung_box_stat": lb_stat,
        "ljung_box_p": lb_p,
        "runs_test_p": float(runs_p),
        "ols_params": fit.params.to_dict(),
        "ols_bse": fit.bse.to_dict(),
        "ols_pvalues": fit.pvalues.to_dict(),
        "hc3_bse": pd.Series(fit_hc3.bse, index=fit.params.index).to_dict(),
        "hc3_pvalues": pd.Series(fit_hc3.pvalues, index=fit.params.index).to_dict(),
    }


def run_task5_icc(df: pd.DataFrame) -> Dict[str, Any]:
    """Task 5: REML MixedLM and ANOVA ICC estimators for block stochastic nuisance."""
    df_aug = df.copy()
    for q in Q:
        df_aug[q + "_sq"] = df_aug[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{Q[i]}_{Q[j]}"] = df_aug[Q[i]] * df_aug[Q[j]]

    # 1. Y1 Val RMSE
    # MixedLM (REML)
    m_y1 = smf.mixedlm("val_rmse ~ x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                       "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug, groups=df_aug["block"]).fit(reml=True)
    s2b_y1 = float(m_y1.cov_re.iloc[0, 0])
    s2e_y1 = float(m_y1.scale)
    icc_reml_y1 = float(s2b_y1 / (s2b_y1 + s2e_y1)) if (s2b_y1 + s2e_y1) > 0 else 0.0

    # Fixed OLS ANOVA
    fit_fixed_y1 = smf.ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                           "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()
    anova_y1 = sm.stats.anova_lm(fit_fixed_y1, typ=3)
    ss_blk_y1 = float(anova_y1.loc["C(block)", "sum_sq"])
    df_blk_y1 = float(anova_y1.loc["C(block)", "df"])
    ms_blk_y1 = ss_blk_y1 / df_blk_y1
    mse_y1 = float(fit_fixed_y1.mse_resid)
    s2b_anova_y1 = max(0.0, (ms_blk_y1 - mse_y1) / 28.0)
    icc_anova_y1 = float(s2b_anova_y1 / (s2b_anova_y1 + mse_y1)) if (s2b_anova_y1 + mse_y1) > 0 else 0.0

    # 2. Y2 Latency
    m_y2 = smf.mixedlm("latency_us_median ~ x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                       "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug, groups=df_aug["block"]).fit(reml=True)
    s2b_y2 = float(m_y2.cov_re.iloc[0, 0])
    s2e_y2 = float(m_y2.scale)
    icc_reml_y2 = float(s2b_y2 / (s2b_y2 + s2e_y2)) if (s2b_y2 + s2e_y2) > 0 else 0.0

    fit_fixed_y2 = smf.ols("latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                           "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()
    anova_y2 = sm.stats.anova_lm(fit_fixed_y2, typ=3)
    ss_blk_y2 = float(anova_y2.loc["C(block)", "sum_sq"])
    df_blk_y2 = float(anova_y2.loc["C(block)", "df"])
    ms_blk_y2 = ss_blk_y2 / df_blk_y2
    mse_y2 = float(fit_fixed_y2.mse_resid)
    s2b_anova_y2 = max(0.0, (ms_blk_y2 - mse_y2) / 28.0)
    icc_anova_y2 = float(s2b_anova_y2 / (s2b_anova_y2 + mse_y2)) if (s2b_anova_y2 + mse_y2) > 0 else 0.0

    return {
        "Y1": {
            "s2b_reml": s2b_y1, "s2e_reml": s2e_y1, "icc_reml": icc_reml_y1,
            "ms_block_anova": ms_blk_y1, "mse_anova": mse_y1, "s2b_anova": s2b_anova_y1, "icc_anova": icc_anova_y1,
            "f_block": float(anova_y1.loc["C(block)", "F"]),
            "p_block": float(anova_y1.loc["C(block)", "PR(>F)"]),
        },
        "Y2": {
            "s2b_reml": s2b_y2, "s2e_reml": s2e_y2, "icc_reml": icc_reml_y2,
            "ms_block_anova": ms_blk_y2, "mse_anova": mse_y2, "s2b_anova": s2b_anova_y2, "icc_anova": icc_anova_y2,
            "f_block": float(anova_y2.loc["C(block)", "F"]),
            "p_block": float(anova_y2.loc["C(block)", "PR(>F)"]),
        }
    }


def derringer_suich_desirability(y1: float, y2: float,
                                 L1: float, U1: float,
                                 L2: float, U2: float,
                                 s: float = 1.0, t: float = 1.0,
                                 w1: float = 1.0, w2: float = 1.0) -> Tuple[float, float, float]:
    """Computes individual and overall Derringer-Suich desirability for two responses."""
    if y1 <= L1:
        d1 = 1.0
    elif y1 >= U1:
        d1 = 0.0
    else:
        d1 = float(((U1 - y1) / (U1 - L1)) ** s)

    if y2 <= L2:
        d2 = 1.0
    elif y2 >= U2:
        d2 = 0.0
    else:
        d2 = float(((U2 - y2) / (U2 - L2)) ** t)

    D = float((d1 ** w1 * d2 ** w2) ** (1.0 / (w1 + w2)))
    return d1, d2, D


def compute_prediction_interval_mean(fit: sm.regression.linear_model.RegressionResultsWrapper,
                                     X: pd.DataFrame, x_row: np.ndarray, m: int, level: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Computes (y_hat, lower_PI, upper_PI, leverage_h) for the mean of m future confirmation runs.
    Uses: half = t * sqrt(MSE * (1/m + h)).
    """
    XtXi = np.linalg.inv(X.T @ X)
    h = float(x_row @ XtXi @ x_row)
    df_res = fit.df_resid
    t_val = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_res))
    yh = float(x_row @ fit.params.values)
    half = t_val * float(np.sqrt(fit.mse_resid * (1.0 / m + h)))
    return yh, yh - half, yh + half, h


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("tables", exist_ok=True)
    df_runs = pd.read_csv("results/runs.csv")

    p1_res = run_phase1_analysis(df_runs)
    p3_res = run_phase3_canonical_analysis(df_runs)
    lof_res = run_task3_lack_of_fit(df_runs)
    diag_res = run_task4_diagnostics(df_runs)
    icc_res = run_task5_icc(df_runs)

    with open("results/phase1.json", "w", encoding="utf-8") as f:
        # omit DataFrame from json
        json.dump({k: v for k, v in p1_res.items() if k != "anova_table"}, f, indent=2)

    with open("results/phase3.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in p3_res.items() if k not in ["fit", "XtXi", "integer_depth_table", "ridge_table"]}, f, indent=2)

    p3_res["integer_depth_table"].to_csv("results/depth_opt_table.csv", index=False)
    p3_res["ridge_table"].to_csv("results/ridge_table.csv", index=False)

    with open("results/lof.json", "w", encoding="utf-8") as f:
        json.dump(lof_res, f, indent=2)

    with open("results/diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diag_res, f, indent=2)

    with open("results/icc.json", "w", encoding="utf-8") as f:
        json.dump(icc_res, f, indent=2)

    print("All Tasks 1-5 analyses completed and saved to results/*.json successfully.")
