"""
plots.py - Publication-Quality Visualizations for Sequential RSM/CCD HPO
========================================================================
Implements Task 10 per fix.md:
  1. 4-in-1 Residual Diagnostics Panel (Studentized, Q-Q, Execution Order Drift, Cook's D).
  2. 2D Contour and 3D Wireframe Surface for RMSE passing through Constrained Optimum.
  3. Latency vs. Depth with 95% CI band and mechanistic curve comparison (replacing inert subsample slice).
  4. Multi-objective Pareto Front with DOE x*, Confirmation point, and baseline incumbents.
  5. Optimization Efficiency Convergence with 20-replicate median and IQR bands.
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import CONFIG, decode_factors
from analysis import build_design_matrix, canonical_decomposition, predict_block_averaged

# Publication matplotlib aesthetics
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 1.8,
    "figure.autolayout": True,
})


def plot_residual_diagnostics(df_runs: pd.DataFrame, save_path: str = "figures/diagnostics_panel_4in1.png"):
    """Renders 4-in-1 Residual Diagnostics Panel."""
    import statsmodels.api as sm
    X = build_design_matrix(df_runs)
    y = df_runs["val_rmse"]
    fit = sm.OLS(y, X).fit()
    infl = fit.get_influence()
    studentized = infl.resid_studentized_external
    fitted = fit.fittedvalues.values
    cooks_d = infl.cooks_distance[0]
    n = len(studentized)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    # (a) Studentized residuals vs. fitted values
    ax = axes[0, 0]
    ax.scatter(fitted, studentized, color="#1f77b4", edgecolor="black", alpha=0.75, s=45)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(2, color="crimson", linestyle=":", linewidth=1.2, label=r"$\pm 2\sigma$ Threshold")
    ax.axhline(-2, color="crimson", linestyle=":", linewidth=1.2)
    ax.axhline(3, color="darkred", linestyle="-.", linewidth=1.0, label=r"$\pm 3\sigma$ Outlier limit")
    ax.axhline(-3, color="darkred", linestyle="-.", linewidth=1.0)
    ax.set_title("(a) Studentized Residuals vs. Fitted Values", fontweight="bold")
    ax.set_xlabel(r"Fitted Values $\hat{y}$")
    ax.set_ylabel("Studentized Residuals")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # (b) Normal Q-Q probability plot
    ax = axes[0, 1]
    sorted_res = np.sort(studentized)
    theoretical_q = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    ax.scatter(theoretical_q, sorted_res, color="#2ca02c", edgecolor="black", alpha=0.8, s=45)
    line_x = np.linspace(-3.2, 3.2, 100)
    ax.plot(line_x, line_x, color="red", linestyle="--", linewidth=1.5, label="1:1 Theoretical Line")
    W, p_norm = stats.shapiro(studentized)
    ax.text(0.05, 0.90, f"Shapiro-Wilk W = {W:.4f}\np = {p_norm:.4f}", transform=ax.transAxes,
            fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_title("(b) Normal Q-Q Probability Plot", fontweight="bold")
    ax.set_xlabel("Theoretical Standard Normal Quantiles")
    ax.set_ylabel("Observed Studentized Residuals")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    # (c) Residuals vs. run execution order
    ax = axes[1, 0]
    order_idx = np.argsort(df_runs["run_order"].values)
    ordered_studentized = studentized[order_idx]
    order_seq = np.arange(1, n + 1)
    ax.scatter(order_seq, ordered_studentized, color="#9467bd", edgecolor="black", alpha=0.75, s=45)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    rolling_mean = pd.Series(ordered_studentized).rolling(window=10, min_periods=1, center=True).mean()
    ax.plot(order_seq, rolling_mean, color="darkorange", linewidth=2.0, label="10-run Rolling Mean")
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.sandbox.stats.runs import runstest_1samp
    dw_val = durbin_watson(fit.resid.values[order_idx])
    lb_res = acorr_ljungbox(fit.resid.values[order_idx], lags=[5], return_df=True)
    lb_p = float(lb_res["lb_pvalue"].iloc[0])
    _, runs_p = runstest_1samp(fit.resid.values[order_idx])
    ax.text(0.05, 0.85, f"Durbin-Watson = {dw_val:.4f}\nLjung-Box p = {lb_p:.4f}\nRuns Test p = {runs_p:.4f}", transform=ax.transAxes,
            fontsize=9.0, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.9))
    ax.set_title("(c) Residuals vs. Execution Run Order", fontweight="bold")
    ax.set_xlabel("Randomized Execution Sequence")
    ax.set_ylabel("Studentized Residuals")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9)

    # (d) Cook's Distance
    ax = axes[1, 1]
    ax.stem(np.arange(1, n + 1), cooks_d, linefmt="#d62728", markerfmt="o", basefmt="k-")
    thresh_4n = 4.0 / n
    ax.axhline(thresh_4n, color="blue", linestyle="--", linewidth=1.4, label=f"Threshold 4/n ({thresh_4n:.3f})")
    ax.set_ylim(0, 0.25)
    max_d = float(np.max(cooks_d))
    ax.text(0.96, 0.92, f"Critical D=1.0 (Max observed = {max_d:.3f})", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0", edgecolor="darkred", alpha=0.9))
    ax.set_title("(d) Cook's Distance (Influence & Leverage)", fontweight="bold")
    ax.set_xlabel("Observation Index")
    ax.set_ylabel(r"Cook's Distance $D_i$")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Residual Diagnostics to: {save_path}")


def plot_response_surface_rmse_2d_3d(df_runs: pd.DataFrame, save_path: str = "figures/response_surface_rmse_2d_3d.png"):
    """Renders 2D Contour Slice and 3D Surface for RMSE passing through the Constrained Optimum."""
    import statsmodels.api as sm
    X = build_design_matrix(df_runs)
    fit_y1 = sm.OLS(df_runs["val_rmse"], X).fit()
    b0, b, B = canonical_decomposition(fit_y1)
    x0 = -0.5 * np.linalg.solve(B, b)

    # Constrained optimum from analysis: depth 7 (x2=0.333), x3=1.0, x4=1.0
    fixed_x3 = 1.0
    fixed_x4 = 1.0

    grid_res = 60
    x1_lin = np.linspace(-1.0, 1.0, grid_res)
    x2_lin = np.linspace(-1.0, 1.0, grid_res)
    X1_grid, X2_grid = np.meshgrid(x1_lin, x2_lin)

    Z_pred = np.zeros((grid_res, grid_res))
    for i in range(grid_res):
        for j in range(grid_res):
            pt = np.array([X1_grid[i, j], X2_grid[i, j], fixed_x3, fixed_x4])
            Z_pred[i, j] = predict_block_averaged(fit_y1, pt)

    fig = plt.figure(figsize=(15, 6), dpi=300)

    # Left: 2D Contour Plot
    ax1 = fig.add_subplot(1, 2, 1)
    cp = ax1.contourf(X1_grid, X2_grid, Z_pred, levels=20, cmap="viridis", alpha=0.9)
    cbar = fig.colorbar(cp, ax=ax1, shrink=0.85)
    cbar.set_label(r"Validation RMSE ($Y_1$)", fontsize=11)
    lines = ax1.contour(X1_grid, X2_grid, Z_pred, levels=12, colors="black", linewidths=0.8, alpha=0.75)
    clabels = ax1.clabel(lines, inline=True, fontsize=8.5, fmt="%.3f", colors="black")
    for lbl in clabels:
        lbl.set_bbox(dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

    # Mark constrained optimum
    ax1.scatter([0.5972], [0.3333], color="cyan", marker="o", s=130, edgecolor="black", zorder=6,
                label=r"Constrained Optimum: $x^* = (0.60, 0.33)$ [Depth 7]")

    # Directional arrow toward unconstrained stationary point
    ax1.annotate(
        f"Stationary Point ({x0[0]:.2f}, {x0[1]:.2f}) outside (x3={x0[2]:.1f}) ↗",
        xy=(min(1.0, max(-1.0, x0[0])), min(1.0, max(-1.0, x0[1]))),
        xytext=(0.02, 0.88),
        arrowprops=dict(facecolor="red", edgecolor="black", shrink=0.08, width=1.5, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor="red", alpha=0.95),
        fontsize=9.0,
        fontweight="bold"
    )

    ax1.set_title(r"2D Response Surface Contours: Validation RMSE ($Y_1$)", fontweight="bold")
    ax1.set_xlabel(r"Factor A: Learning Rate $x_1$ ($\ln(\eta)$)")
    ax1.set_ylabel(r"Factor B: Max Tree Depth $x_2$")
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.5)

    # Right: 3D Surface Plot
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    surf = ax2.plot_surface(X1_grid, X2_grid, Z_pred, cmap="viridis", edgecolor="k", linewidth=0.2, alpha=0.85, antialiased=True)
    ax2.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax2.grid(True, linestyle=":", alpha=0.35)

    ax2.set_title(r"3D Response Surface: Validation RMSE ($Y_1$)", fontweight="bold")
    ax2.set_xlabel(r"Factor A: $x_1$ ($\ln(\eta)$)", labelpad=8)
    ax2.set_ylabel(r"Factor B: Depth $x_2$", labelpad=8)
    ax2.set_zlabel(r"Validation RMSE ($Y_1$)", labelpad=8)
    ax2.view_init(elev=28, azim=-125)
    ax2.set_box_aspect(None, zoom=1.15)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved RMSE Response Surface to: {save_path}")


def plot_latency_vs_depth(df_runs: pd.DataFrame, save_path: str = "figures/response_surface_latency_2d_3d.png"):
    """
    Task 8 / Task 10: Latency vs. Depth analysis.
    Replaces inert subsample slice with empirical latency vs. depth + 95% CI band + mechanistic fit.
    """
    import statsmodels.formula.api as smf

    m_lin = smf.ols("latency_us_median ~ depth", df_runs).fit()
    m_quad = smf.ols("latency_us_median ~ depth + I(depth**2)", df_runs).fit()

    depth_stats = df_runs.groupby("depth")["latency_us_median"].agg(["mean", "std", "count"]).reset_index()
    depth_stats["se"] = depth_stats["std"] / np.sqrt(depth_stats["count"])
    depth_stats["ci95_low"] = depth_stats["mean"] - 1.96 * depth_stats["se"]
    depth_stats["ci95_high"] = depth_stats["mean"] + 1.96 * depth_stats["se"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel 1: Empirical Latency vs Depth with fitted curves
    depth_grid = np.linspace(3, 9, 100)
    pred_lin = m_lin.predict(pd.DataFrame({"depth": depth_grid}))
    pred_quad = m_quad.predict(pd.DataFrame({"depth": depth_grid}))

    ax1.errorbar(depth_stats["depth"], depth_stats["mean"], yerr=1.96 * depth_stats["se"],
                 fmt="o", color="#1f77b4", ecolor="navy", elinewidth=2.0, capsize=5, capthick=2.0,
                 markersize=8, label="Empirical Mean across Runs (95% CI)", zorder=5)

    ax1.plot(depth_grid, pred_lin, color="green", linestyle="--", linewidth=2.0,
             label=f"Linear Fit ($R^2 = {m_lin.rsquared:.3f}$)")
    ax1.plot(depth_grid, pred_quad, color="red", linestyle="-", linewidth=2.2,
             label=f"Quadratic Fit ($R^2 = {m_quad.rsquared:.3f}$)")

    # Display linear decomposition note
    fixed_overhead = m_lin.params["Intercept"]
    slope = m_lin.params["depth"]
    ax1.text(0.05, 0.85, f"Linear Decomposition:\nLatency = {fixed_overhead:.1f} $\\mu$s + {slope:.1f} $\\mu$s/depth\nQuadratic fit captures super-linear leaf scaling",
             transform=ax1.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="gray", alpha=0.9))

    ax1.set_title("Inference Latency vs. Tree Depth ($n_{\\text{trees}}=100$)", fontweight="bold")
    ax1.set_xlabel("Maximum Tree Depth")
    ax1.set_ylabel(r"Inference Latency ($\mu$s/sample)")
    ax1.set_xticks(range(3, 10))
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, fontsize=9.5)

    # Panel 2: Latency distribution boxplot across depth
    depth_data = [df_runs[df_runs["depth"] == d]["latency_us_median"].values for d in range(3, 10)]
    bp = ax2.boxplot(depth_data, positions=list(range(3, 10)), patch_artist=True,
                     boxprops=dict(facecolor="#aec7e8", color="navy"),
                     medianprops=dict(color="darkred", linewidth=2.0))
    ax2.set_title("Latency Distribution Across Empirical Depth Settings", fontweight="bold")
    ax2.set_xlabel("Maximum Tree Depth")
    ax2.set_ylabel(r"Measured Latency ($\mu$s/sample)")
    ax2.grid(True, linestyle=":", alpha=0.6)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Latency vs Depth Analysis to: {save_path}")


def plot_pareto_front_and_desirability(df_runs: pd.DataFrame, save_path: str = "figures/desirability_pareto_front.png"):
    """
    Renders Multi-objective Pareto trade-off between Val RMSE and Latency,
    marking the DOE recommended optimum x*, confirmation mean, and baseline incumbents.
    """
    df_bm = pd.read_csv("results/benchmark.csv")
    with open("results/confirmation.json", "r", encoding="utf-8") as f:
        conf = json.load(f)

    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=300)

    # Scatter of all CCD runs
    y1_all = df_runs["val_rmse"].values
    y2_all = df_runs["latency_us_median"].values
    depths = df_runs["depth"].values

    sc = ax.scatter(y1_all, y2_all, c=depths, cmap="cividis", s=55, alpha=0.75,
                    edgecolor="black", linewidth=0.5, label="CCD Design Points")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Max Tree Depth (Integer)", fontsize=11)

    # Identify Pareto frontier of CCD design
    sorted_pts = sorted(zip(y1_all, y2_all), key=lambda p: p[0])
    pareto_y1 = []
    pareto_y2 = []
    min_y2 = 1e9
    for y1, y2 in sorted_pts:
        if y2 < min_y2:
            pareto_y1.append(y1)
            pareto_y2.append(y2)
            min_y2 = y2

    ax.step(pareto_y1, pareto_y2, where="post", color="red", linestyle="--", linewidth=1.8, label="Empirical Pareto Frontier")
    ax.scatter(pareto_y1, pareto_y2, color="red", marker="o", s=70, edgecolor="black", zorder=5)

    # 1. Mark DOE recommended x* (Depth 4)
    doe_row = df_bm[df_bm["method"].str.contains("Multi-Objective") & df_bm["method"].str.contains("DOE")].iloc[0]
    ax.scatter([doe_row["val_rmse_mean"]], [doe_row["predict_latency_us_median"]],
               color="gold", marker="*", s=260, edgecolor="black", linewidth=1.5, zorder=8,
               label=f"DOE Multi-Obj $x^*$ (Depth 4, {doe_row['predict_latency_us_median']:.1f} $\\mu$s)")

    # 2. Mark DOE Single-Objective (Depth 7)
    doe_so = df_bm[df_bm["method"].str.contains("Single-Objective") & df_bm["method"].str.contains("DOE")].iloc[0]
    ax.scatter([doe_so["val_rmse_mean"]], [doe_so["predict_latency_us_median"]],
               color="cyan", marker="^", s=160, edgecolor="black", linewidth=1.2, zorder=8,
               label=f"DOE Single-Obj (Depth 7, {doe_so['predict_latency_us_median']:.1f} $\\mu$s)")

    # 3. Mark Confirmation Point
    conf_y1 = conf["Y1_Val_RMSE"]["empirical_mean"]
    conf_y2 = conf["Y2_Latency"]["empirical_mean"]
    ax.scatter([conf_y1], [conf_y2], color="lime", marker="D", s=110, edgecolor="black", linewidth=1.2, zorder=8,
               label=f"Confirmation Mean ($n=10$, {conf_y2:.1f} $\\mu$s)")

    # 4. Mark Baseline Incumbents
    tpe_row = df_bm[df_bm["method"].str.contains("Single-Obj") & ~df_bm["method"].str.contains("DOE")].iloc[0]
    ax.scatter([tpe_row["val_rmse_mean"]], [tpe_row["predict_latency_us_median"]],
               color="magenta", marker="P", s=150, edgecolor="black", linewidth=1.2, zorder=8,
               label=f"TPE Single-Obj (Depth {int(tpe_row['depth'])}, {tpe_row['predict_latency_us_median']:.1f} $\\mu$s)")

    tpe_const = df_bm[df_bm["method"].str.contains("Constrained")].iloc[0]
    ax.scatter([tpe_const["val_rmse_mean"]], [tpe_const["predict_latency_us_median"]],
               color="darkorange", marker="X", s=140, edgecolor="black", linewidth=1.2, zorder=8,
               label=f"Constrained TPE (Depth {int(tpe_const['depth'])}, {tpe_const['predict_latency_us_median']:.1f} $\\mu$s)")

    tpe_mo = df_bm[df_bm["method"].str.contains("Multi-Objective TPE")].iloc[0]
    ax.scatter([tpe_mo["val_rmse_mean"]], [tpe_mo["predict_latency_us_median"]],
               color="blueviolet", marker="s", s=130, edgecolor="black", linewidth=1.2, zorder=8,
               label=f"MO-TPE (Depth {int(tpe_mo['depth'])}, {tpe_mo['predict_latency_us_median']:.1f} $\\mu$s)")

    ax.set_title("Multi-Objective Pareto Trade-off: Validation RMSE vs. Inference Latency", fontweight="bold")
    ax.set_xlabel(r"Validation RMSE ($Y_1$, Smaller is Better)")
    ax.set_ylabel(r"Inference Latency ($\mu$s/sample, Smaller is Better)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.0)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Pareto Front Plot to: {save_path}")


def plot_efficiency_comparison(save_path: str = "figures/efficiency_comparison_curve.png"):
    """
    Renders Optimization Efficiency Convergence:
    Median and IQR shaded bands across 20 replicate runs for Random Search and TPE,
    with DOE cumulative best in actual randomized run_order.
    """
    df_traj = pd.read_csv("results/benchmark_evals_trajectories.csv")
    evals = df_traj["eval_idx"].values
    df_runs = pd.read_csv("results/runs.csv")
    doe_cum_best = df_runs.sort_values("run_order")["val_rmse"].cummin().values

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 1. Random Search: Median and IQR band
    ax.plot(evals, df_traj["rs_median"], color="#d62728", linewidth=2.0, linestyle="--", label="Random Search (20-run Median)")
    ax.fill_between(evals, df_traj["rs_q25"], df_traj["rs_q75"], color="#d62728", alpha=0.18, label="Random Search IQR Band")

    # 2. TPE Single-Obj: Median and IQR band
    ax.plot(evals, df_traj["tpe_median"], color="#2ca02c", linewidth=2.2, linestyle="-.", label="Optuna TPE Single-Obj (20-run Median)")
    ax.fill_between(evals, df_traj["tpe_q25"], df_traj["tpe_q75"], color="#2ca02c", alpha=0.18, label="TPE IQR Band")

    # 3. DOE Cumulative Best in actual run_order
    ax.plot(evals, doe_cum_best, color="#1f77b4", linewidth=2.5, linestyle="-", label="DOE Design (Randomized Execution Order)")

    # Phase dividers for DOE
    ax.axvline(100, color="gray", linestyle=":", linewidth=1.2)
    ax.text(50, 0.54, "Phase 1: Screening ($2^4+n_C$)",
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))
    ax.text(120, 0.54, "Phase 2: Axial CCD",
            ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    # Zoomed Inset Axes (Runs 20 to 140)
    ax_ins = ax.inset_axes([0.48, 0.42, 0.48, 0.50])
    ax_ins.plot(evals, df_traj["rs_median"], color="#d62728", linewidth=1.8, linestyle="--")
    ax_ins.fill_between(evals, df_traj["rs_q25"], df_traj["rs_q75"], color="#d62728", alpha=0.18)
    ax_ins.plot(evals, df_traj["tpe_median"], color="#2ca02c", linewidth=2.0, linestyle="-.")
    ax_ins.fill_between(evals, df_traj["tpe_q25"], df_traj["tpe_q75"], color="#2ca02c", alpha=0.18)
    ax_ins.plot(evals, doe_cum_best, color="#1f77b4", linewidth=2.2)

    ax_ins.set_xlim(20, 140)
    ax_ins.set_ylim(0.455, 0.490)
    ax_ins.set_title("Zoomed Convergence (Runs 20-140)", fontsize=9, fontweight="bold")
    ax_ins.grid(True, linestyle=":", alpha=0.6)
    ax.indicate_inset_zoom(ax_ins, edgecolor="#555555", alpha=0.75)

    ax.set_title("Optimization Sample Efficiency Comparison (Cumulative Best Val RMSE vs. Budget)", fontweight="bold")
    ax.set_xlabel("Number of Function Evaluations (Genuine Model Training Runs)")
    ax.set_ylabel("Cumulative Best Validation RMSE")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.96), frameon=True, facecolor="white", framealpha=0.95, fontsize=9.0)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Efficiency Comparison to: {save_path}")


def render_all_plots():
    df_runs = pd.read_csv("results/runs.csv")
    plot_residual_diagnostics(df_runs, "figures/diagnostics_panel_4in1.png")
    plot_response_surface_rmse_2d_3d(df_runs, "figures/response_surface_rmse_2d_3d.png")
    plot_latency_vs_depth(df_runs, "figures/response_surface_latency_2d_3d.png")
    plot_pareto_front_and_desirability(df_runs, "figures/desirability_pareto_front.png")
    plot_efficiency_comparison("figures/efficiency_comparison_curve.png")
    print("All 5 publication plots rendered successfully.")

if __name__ == "__main__":
    render_all_plots()
