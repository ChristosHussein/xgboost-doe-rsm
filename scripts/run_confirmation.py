"""
scripts/run_confirmation.py - Task 6: Confirmation experiment at recommended optima.
Runs m=10 trials across fresh, disjoint seeds.
Evaluates:
  1. Multi-Objective Desirability Optimum x* (depth 4).
  2. Single-Objective Cube Optimum x_single* (depth 7).
Prediction interval explicitly incorporates block variance:
  Var(y_bar_m - y_hat) = MSE * (1/m + h) + sigma2_block * (1/m + 1/5)
"""

import json
import os
import sys

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import yaml

from pipeline import CaliforniaHousingDataManager, evaluate_model, decode_factors, pin_cpu_affinity, CONFIG

CONFIRMATION_SEEDS = CONFIG["seeds"]["confirmation_seeds"]

def execute_confirmation():
    pin_cpu_affinity()
    os.makedirs("results", exist_ok=True)
    df_runs = pd.read_csv("results/runs.csv")

    with open("results/icc.json", "r", encoding="utf-8") as f:
        icc = json.load(f)
    s2b_y1 = icc["Y1"].get("s2b_anova", icc["Y1"].get("s2b_reml", 0.0))
    s2b_y2 = icc["Y2"].get("s2b_anova", icc["Y2"].get("s2b_reml", 0.0))

    Q = ["x1", "x2", "x3", "x4"]
    X = pd.DataFrame(index=df_runs.index)
    for q in Q: X[q] = df_runs[q]
    for q in Q: X[q + "_sq"] = df_runs[q] ** 2
    for i in range(4):
        for j in range(i + 1, 4):
            X[f"{Q[i]}_{Q[j]}"] = df_runs[Q[i]] * df_runs[Q[j]]
    blk = pd.get_dummies(df_runs["block"], prefix="blk", drop_first=True).astype(float)
    X = sm.add_constant(pd.concat([X, blk], axis=1))

    fit_y1 = sm.OLS(df_runs["val_rmse"], X).fit()
    fit_y2 = sm.OLS(df_runs["latency_us_median"], X).fit()
    XtXi = np.linalg.inv(X.T @ X)

    m = len(CONFIRMATION_SEEDS)
    # Variance components from ANOVA for Satterthwaite prediction interval
    ms_e_y1 = fit_y1.mse_resid
    ms_blk_y1 = ms_e_y1 + 28.0 * s2b_y1
    c2 = (1.0 / m + 0.2) / 28.0

    data_mgr = CaliforniaHousingDataManager()

    # 1. Multi-Objective Optimum x* (Depth 4)
    x_star = np.array([0.8499708, -0.66666667, 1.0, -0.08116946])
    eta_star, depth_star, sub_star, lam_star = decode_factors(x_star)
    x1, x2, x3, x4 = x_star
    x_row_mo = np.array([1.0, x1, x2, x3, x4,
                         x1**2, x2**2, x3**2, x4**2,
                         x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4,
                         0.2, 0.2, 0.2, 0.2])
    h_mo = float(x_row_mo @ XtXi @ x_row_mo)

    c1_mo = (1.0 / m + h_mo) - c2
    var_pi_y1_mo = c1_mo * ms_e_y1 + c2 * ms_blk_y1
    nu_eff_y1_mo = (var_pi_y1_mo**2) / (((c1_mo * ms_e_y1)**2) / 121.0 + ((c2 * ms_blk_y1)**2) / 4.0)
    t_crit_y1_mo = float(stats.t.ppf(0.975, nu_eff_y1_mo))

    yh_y1_mo = float(x_row_mo @ fit_y1.params.values)
    half_y1_mo = t_crit_y1_mo * float(np.sqrt(var_pi_y1_mo))
    pi_y1_mo = (yh_y1_mo - half_y1_mo, yh_y1_mo + half_y1_mo)

    # Latency Y2
    ms_e_y2 = fit_y2.mse_resid
    ms_blk_y2 = ms_e_y2 + 28.0 * s2b_y2
    var_pi_y2_mo = c1_mo * ms_e_y2 + c2 * ms_blk_y2
    denom_nu_y2_mo = ((c1_mo * ms_e_y2)**2) / 121.0 + ((c2 * ms_blk_y2)**2) / 4.0
    nu_eff_y2_mo = (var_pi_y2_mo**2) / denom_nu_y2_mo if denom_nu_y2_mo > 0 else 121.0
    t_crit_y2_mo = float(stats.t.ppf(0.975, nu_eff_y2_mo))

    yh_y2_mo = float(x_row_mo @ fit_y2.params.values)
    half_y2_mo = t_crit_y2_mo * float(np.sqrt(var_pi_y2_mo))
    pi_y2_mo = (yh_y2_mo - half_y2_mo, yh_y2_mo + half_y2_mo)

    # 2. Single-Objective Optimum (Depth 7)
    with open("results/phase3.json", "r", encoding="utf-8") as f:
        p3 = json.load(f)
    x_single = np.array(p3["constrained_optimum_cube"]["x"])
    eta_so, depth_so, sub_so, lam_so = decode_factors(x_single)
    x1, x2, x3, x4 = x_single
    x_row_so = np.array([1.0, x1, x2, x3, x4,
                         x1**2, x2**2, x3**2, x4**2,
                         x1*x2, x1*x3, x1*x4, x2*x3, x2*x4, x3*x4,
                         0.2, 0.2, 0.2, 0.2])
    h_so = float(x_row_so @ XtXi @ x_row_so)

    c1_so = (1.0 / m + h_so) - c2
    var_pi_y1_so = c1_so * ms_e_y1 + c2 * ms_blk_y1
    nu_eff_y1_so = (var_pi_y1_so**2) / (((c1_so * ms_e_y1)**2) / 121.0 + ((c2 * ms_blk_y1)**2) / 4.0)
    t_crit_y1_so = float(stats.t.ppf(0.975, nu_eff_y1_so))

    yh_y1_so = float(x_row_so @ fit_y1.params.values)
    half_y1_so = t_crit_y1_so * float(np.sqrt(var_pi_y1_so))
    pi_y1_so = (yh_y1_so - half_y1_so, yh_y1_so + half_y1_so)

    var_pi_y2_so = c1_so * ms_e_y2 + c2 * ms_blk_y2
    denom_nu_y2_so = ((c1_so * ms_e_y2)**2) / 121.0 + ((c2 * ms_blk_y2)**2) / 4.0
    nu_eff_y2_so = (var_pi_y2_so**2) / denom_nu_y2_so if denom_nu_y2_so > 0 else 121.0
    t_crit_y2_so = float(stats.t.ppf(0.975, nu_eff_y2_so))

    yh_y2_so = float(x_row_so @ fit_y2.params.values)
    half_y2_so = t_crit_y2_so * float(np.sqrt(var_pi_y2_so))
    pi_y2_so = (yh_y2_so - half_y2_so, yh_y2_so + half_y2_so)

    csv_mo = "results/confirmation_runs.csv"
    csv_so = "results/confirmation_runs_single_obj.csv"

    if os.path.exists(csv_mo) and os.path.exists(csv_so):
        print(f"Loading existing confirmation runs from {csv_mo} and {csv_so}...")
        df_conf_mo = pd.read_csv(csv_mo)
        df_conf_so = pd.read_csv(csv_so)
    else:
        print(f"Running confirmation at x* across {m} fresh seeds...")
        records_mo = []
        records_so = []

        for seed in CONFIRMATION_SEEDS:
            eval_mo = evaluate_model(
                eta=eta_star,
                depth=depth_star,
                subsample=sub_star,
                reg_lambda=lam_star,
                seed=seed,
                data_mgr=data_mgr,
                measure_latency_details=True
            )
            records_mo.append({
                "seed": seed,
                "val_rmse": eval_mo["val_rmse"],
                "test_rmse": eval_mo["test_rmse"],
                "latency_us_median": eval_mo["latency_us_median"],
                "latency_us_iqr": eval_mo["latency_us_iqr"],
                "inplace_latency_us_median": eval_mo["inplace_latency_us_median"],
                "fit_time_s": eval_mo["fit_time_s"],
            })

            eval_so = evaluate_model(
                eta=eta_so,
                depth=depth_so,
                subsample=sub_so,
                reg_lambda=lam_so,
                seed=seed,
                data_mgr=data_mgr,
                measure_latency_details=True
            )
            records_so.append({
                "seed": seed,
                "val_rmse": eval_so["val_rmse"],
                "test_rmse": eval_so["test_rmse"],
                "latency_us_median": eval_so["latency_us_median"],
                "latency_us_iqr": eval_so["latency_us_iqr"],
                "inplace_latency_us_median": eval_so["inplace_latency_us_median"],
                "fit_time_s": eval_so["fit_time_s"],
            })

        df_conf_mo = pd.DataFrame(records_mo)
        df_conf_so = pd.DataFrame(records_so)
        df_conf_mo.to_csv(csv_mo, index=False)
        df_conf_so.to_csv(csv_so, index=False)

    emp_val_m_mo = float(df_conf_mo["val_rmse"].mean())
    emp_val_s_mo = float(df_conf_mo["val_rmse"].std())
    emp_test_m_mo = float(df_conf_mo["test_rmse"].mean())
    emp_test_s_mo = float(df_conf_mo["test_rmse"].std())
    emp_lat_m_mo = float(df_conf_mo["latency_us_median"].mean())
    emp_lat_s_mo = float(df_conf_mo["latency_us_median"].std())
    emp_inplat_m_mo = float(df_conf_mo["inplace_latency_us_median"].mean())
    emp_inplat_s_mo = float(df_conf_mo["inplace_latency_us_median"].std())

    emp_val_m_so = float(df_conf_so["val_rmse"].mean())
    emp_val_s_so = float(df_conf_so["val_rmse"].std())
    emp_test_m_so = float(df_conf_so["test_rmse"].mean())
    emp_test_s_so = float(df_conf_so["test_rmse"].std())
    emp_lat_m_so = float(df_conf_so["latency_us_median"].mean())
    emp_lat_s_so = float(df_conf_so["latency_us_median"].std())

    inside_val_mo = bool(pi_y1_mo[0] <= emp_val_m_mo <= pi_y1_mo[1])
    inside_lat_mo = bool(pi_y2_mo[0] <= emp_lat_m_mo <= pi_y2_mo[1])

    inside_val_so = bool(pi_y1_so[0] <= emp_val_m_so <= pi_y1_so[1])
    inside_lat_so = bool(pi_y2_so[0] <= emp_lat_m_so <= pi_y2_so[1])

    summary = {
        "x_star_coded": x_star.tolist(),
        "x_star_natural": {
            "eta": eta_star, "depth": depth_star, "subsample": sub_star, "reg_lambda": lam_star
        },
        "leverage_h": h_mo,
        "Y1_Val_RMSE": {
            "predicted_mean": yh_y1_mo,
            "prediction_interval_95": [float(pi_y1_mo[0]), float(pi_y1_mo[1])],
            "nu_eff": float(nu_eff_y1_mo),
            "empirical_mean": emp_val_m_mo,
            "empirical_std": emp_val_s_mo,
            "inside_pi": inside_val_mo,
        },
        "Y1_Test_RMSE": {
            "empirical_mean": emp_test_m_mo,
            "empirical_std": emp_test_s_mo,
        },
        "Y2_Latency": {
            "predicted_mean": yh_y2_mo,
            "prediction_interval_95": [float(pi_y2_mo[0]), float(pi_y2_mo[1])],
            "nu_eff": float(nu_eff_y2_mo),
            "empirical_mean": emp_lat_m_mo,
            "empirical_std": emp_lat_s_mo,
            "empirical_inplace_mean": emp_inplat_m_mo,
            "empirical_inplace_std": emp_inplat_s_mo,
            "inside_pi": inside_lat_mo,
        },
        "Single_Objective_Optimum": {
            "x_coded": x_single.tolist(),
            "x_natural": {
                "eta": eta_so, "depth": depth_so, "subsample": sub_so, "reg_lambda": lam_so
            },
            "leverage_h": h_so,
            "predicted_val_rmse": yh_y1_so,
            "prediction_interval_95_val": [float(pi_y1_so[0]), float(pi_y1_so[1])],
            "nu_eff": float(nu_eff_y1_so),
            "empirical_val_rmse": emp_val_m_so,
            "empirical_val_std": emp_val_s_so,
            "empirical_test_rmse": emp_test_m_so,
            "empirical_test_std": emp_test_s_so,
            "predicted_latency": yh_y2_so,
            "prediction_interval_95_lat": [float(pi_y2_so[0]), float(pi_y2_so[1])],
            "empirical_latency": emp_lat_m_so,
            "empirical_latency_std": emp_lat_s_so,
            "inside_pi_val": inside_val_so,
            "inside_pi_lat": inside_lat_so,
        }
    }

    with open("results/confirmation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Confirmation Summary (DOE x*, Depth 4) ---")
    print(f"Empirical Val RMSE: {emp_val_m_mo:.4f} +/- {emp_val_s_mo:.4f} (PI: [{pi_y1_mo[0]:.4f}, {pi_y1_mo[1]:.4f}] -> Pass: {inside_val_mo})")
    print(f"Empirical Test RMSE: {emp_test_m_mo:.4f} +/- {emp_test_s_mo:.4f}")
    print(f"Empirical Latency: {emp_lat_m_mo:.2f} +/- {emp_lat_s_mo:.2f} us (PI: [{pi_y2_mo[0]:.2f}, {pi_y2_mo[1]:.2f}] -> Pass: {inside_lat_mo})")

    print("\n--- Confirmation Summary (DOE Single-Obj, Depth 7) ---")
    print(f"Empirical Val RMSE: {emp_val_m_so:.4f} +/- {emp_val_s_so:.4f} (Predicted: {yh_y1_so:.4f}, bias: {emp_val_m_so - yh_y1_so:+.4f})")
    print(f"Empirical Test RMSE: {emp_test_m_so:.4f} +/- {emp_test_s_so:.4f}")
    print(f"Empirical Latency: {emp_lat_m_so:.2f} +/- {emp_lat_s_so:.2f} us")

    return summary

if __name__ == "__main__":
    execute_confirmation()
