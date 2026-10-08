"""
scripts/fit_latency_models.py - Task 8: Latency Modelling and Form Comparison
Fits:
  1. Linear: latency ~ depth
  2. Quadratic: latency ~ depth + depth^2
  3. Log-linear: ln(latency) ~ depth
  4. Mechanistic: latency ~ const + beta * depth (tree leaf traversal model)
Compares R^2, Adj R^2, AIC, BIC, and RMSE.
"""

import json
import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def analyze_latency_models(df_runs: pd.DataFrame) -> Dict[str, Any]:
    # 1. Linear model
    m_lin = smf.ols("latency_us_median ~ depth", df_runs).fit()

    # 2. Quadratic model
    m_quad = smf.ols("latency_us_median ~ depth + I(depth**2)", df_runs).fit()

    # 3. Log-linear model
    m_log = smf.ols("np.log(latency_us_median) ~ depth", df_runs).fit()

    # 4. Multi-factor linear model (depth + eta + subsample + lambda)
    m_multi = smf.ols("latency_us_median ~ depth + I(depth**2) + eta + subsample + reg_lambda", df_runs).fit()

    # 5. Full CCD second-order latency model from ANOVA
    df_aug = df_runs.copy()
    Q = ["x1", "x2", "x3", "x4"]
    for q in Q: df_aug[q + "_sq"] = df_aug[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{Q[i]}_{Q[j]}"] = df_aug[Q[i]] * df_aug[Q[j]]

    m_ccd = smf.ols("latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                    "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()

    comparison = [
        {
            "Model": "Linear (depth)",
            "R2": float(m_lin.rsquared),
            "Adj_R2": float(m_lin.rsquared_adj),
            "AIC": float(m_lin.aic),
            "BIC": float(m_lin.bic),
            "RMSE": float(np.sqrt(m_lin.mse_resid)),
            "Formula": "latency ~ depth"
        },
        {
            "Model": "Quadratic (depth + depth^2)",
            "R2": float(m_quad.rsquared),
            "Adj_R2": float(m_quad.rsquared_adj),
            "AIC": float(m_quad.aic),
            "BIC": float(m_quad.bic),
            "RMSE": float(np.sqrt(m_quad.mse_resid)),
            "Formula": "latency ~ depth + depth^2"
        },
        {
            "Model": "Log-Linear (ln(latency) ~ depth)",
            "R2": float(m_log.rsquared),
            "Adj_R2": float(m_log.rsquared_adj),
            "AIC": float(m_log.aic),
            "BIC": float(m_log.bic),
            "RMSE": float(np.sqrt(np.mean((df_runs["latency_us_median"] - np.exp(m_log.fittedvalues))**2))),
            "Formula": "ln(latency) ~ depth"
        },
        {
            "Model": "Multi-Factor (depth, depth^2, eta, sub, lam)",
            "R2": float(m_multi.rsquared),
            "Adj_R2": float(m_multi.rsquared_adj),
            "AIC": float(m_multi.aic),
            "BIC": float(m_multi.bic),
            "RMSE": float(np.sqrt(m_multi.mse_resid)),
            "Formula": "latency ~ depth + depth^2 + eta + subsample + lambda"
        },
        {
            "Model": "Full CCD Second-Order (All factors + Block)",
            "R2": float(m_ccd.rsquared),
            "Adj_R2": float(m_ccd.rsquared_adj),
            "AIC": float(m_ccd.aic),
            "BIC": float(m_ccd.bic),
            "RMSE": float(np.sqrt(m_ccd.mse_resid)),
            "Formula": "latency ~ C(block) + second-order surface"
        }
    ]

    df_comp = pd.DataFrame(comparison)
    df_comp.to_csv("results/latency_models_comparison.csv", index=False)

    anova_ccd_latency = sm.stats.anova_lm(m_ccd, typ=3)
    anova_ccd_latency.to_csv("results/anova_ccd_latency.csv")

    summary = {
        "comparison_table": comparison,
        "m_lin_params": m_lin.params.to_dict(),
        "m_quad_params": m_quad.params.to_dict(),
        "ccd_r2": float(m_ccd.rsquared),
        "ccd_adj_r2": float(m_ccd.rsquared_adj),
    }

    with open("results/latency_modeling.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

if __name__ == "__main__":
    df_runs = pd.read_csv("results/runs.csv")
    res = analyze_latency_models(df_runs)
    print("Latency modeling analysis complete:")
    print(pd.read_csv("results/latency_models_comparison.csv"))
