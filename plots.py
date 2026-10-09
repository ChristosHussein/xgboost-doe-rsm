"""Historical diagnostics and versioned revision-v2 comparison plots."""

import argparse
import json
import os
import sys
from pathlib import Path
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
from scientific_stats import pareto_front_2d_min

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
    ax1.scatter([0.5983], [0.3333], color="cyan", marker="o", s=130, edgecolor="black", zorder=6,
                label=r"Single-Obj Optimum: $\mathbf{x}^*_{\text{SO}} = (0.60, 0.33)$ [Depth 7]")

    # Directional arrow toward unconstrained stationary point
    ax1.annotate(
        f"Stationary Point ({x0[0]:.2f}, {x0[1]:.2f})\noutside domain ($x_3={x0[2]:.1f}$) ↗",
        xy=(min(1.0, max(-1.0, x0[0])), min(1.0, max(-1.0, x0[1]))),
        xytext=(-0.65, 0.78),
        arrowprops=dict(facecolor="red", edgecolor="black", shrink=0.08, width=1.5, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff5f5", edgecolor="red", alpha=0.95),
        fontsize=8.5,
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
    """Plot the historical DOE front from configuration-level block means only."""
    grouped = (
        df_runs.groupby(["point_id", "depth"], as_index=False)
        .agg(
            validation_rmse=("val_rmse", "mean"),
            validation_rmse_sd=("val_rmse", "std"),
            predict_latency_us=("latency_us_median", "mean"),
            predict_latency_us_sd=("latency_us_median", "std"),
        )
    )
    fig, ax = plt.subplots(figsize=(9.5, 7), dpi=300)
    sc = ax.scatter(
        grouped["validation_rmse"],
        grouped["predict_latency_us"],
        c=grouped["depth"],
        cmap="cividis",
        s=65,
        edgecolor="black",
        linewidth=0.5,
        label="DOE configuration means",
    )
    ax.errorbar(
        grouped["validation_rmse"],
        grouped["predict_latency_us"],
        xerr=grouped["validation_rmse_sd"],
        yerr=grouped["predict_latency_us_sd"],
        fmt="none",
        ecolor="0.55",
        alpha=0.45,
        linewidth=0.7,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("Maximum tree depth (integer)", fontsize=11)
    front = pareto_front_2d_min(
        grouped[["validation_rmse", "predict_latency_us"]].to_numpy()
    )
    if front.points:
        pareto = np.asarray(front.points)
        ax.step(
            pareto[:, 0],
            pareto[:, 1],
            where="post",
            color="red",
            linestyle="--",
            label="DOE configuration-mean Pareto front",
        )
    ax.set_title(
        "Historical DOE configuration means: validation RMSE and latency",
        fontweight="bold",
    )
    ax.set_xlabel(r"Validation RMSE ($Y_1$, Smaller is Better)")
    ax.set_ylabel(r"Historical-session predict latency ($\mu$s/sample)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.95)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved configuration-mean historical Pareto plot to: {save_path}")


def load_revision_pareto_data(run_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """Load and validate the two non-mixed estimands used by the revision plot."""
    run_dir = Path(run_dir)
    doe = pd.read_csv(run_dir / "doe_matched_candidate_front.csv").copy()
    trials = pd.read_csv(run_dir / "optimizer_trials.csv")
    final = pd.read_csv(run_dir / "final_summary.csv").copy()
    mo_tpe = trials.loc[
        trials["optimizer"].eq("multi_objective_tpe")
        & trials["trial_status"].eq("completed")
    ].copy()
    if doe.empty or mo_tpe.empty or final.empty:
        raise ValueError("Revision Pareto inputs must each contain at least one row")

    def one_complete_value(frame: pd.DataFrame, column: str, label: str):
        if column not in frame or frame[column].isna().any():
            raise ValueError(f"{label} requires complete {column} values")
        values = frame[column].astype(str)
        if values.str.strip().eq("").any() or values.nunique() != 1:
            raise ValueError(f"{label} requires exactly one complete {column}")
        return frame[column].iloc[0]

    development_protocol = one_complete_value(
        doe, "latency_protocol_id", "DOE development panel"
    )
    trial_protocol = one_complete_value(
        mo_tpe, "latency_protocol_id", "MO-TPE development panel"
    )
    if development_protocol != trial_protocol:
        raise ValueError(
            "Development Pareto panel requires one matching latency protocol for DOE and MO-TPE"
        )
    doe_split = one_complete_value(
        doe, "development_split_seed", "DOE development panel"
    )
    trial_split = one_complete_value(
        mo_tpe, "development_split_seed", "MO-TPE development panel"
    )
    if int(doe_split) != int(trial_split):
        raise ValueError(
            "Development Pareto panel requires DOE and MO-TPE rows from the same split"
        )
    one_complete_value(
        final, "latency_protocol_id", "Independent-evaluation panel"
    )

    for label, frame in (("DOE", doe), ("MO-TPE", mo_tpe)):
        for column in ("validation_rmse", "predict_latency_us"):
            numeric = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(numeric)):
                raise ValueError(f"{label} {column} values must be finite")
            frame[column] = numeric
    final_latency = pd.to_numeric(final["predict_latency_us"], errors="coerce")
    if not np.all(np.isfinite(final_latency.to_numpy(dtype=float))):
        raise ValueError("Independent predict_latency_us values must be finite")
    final["predict_latency_us"] = final_latency

    parsed = final.copy()
    parsed["validation_rmse_mean"] = parsed["validation_rmse"].map(
        lambda value: float(json.loads(value)["mean"])
    )
    parsed["validation_rmse_ci_low"] = parsed["validation_rmse"].map(
        lambda value: json.loads(value)["confidence_interval_95"][0]
    )
    parsed["validation_rmse_ci_high"] = parsed["validation_rmse"].map(
        lambda value: json.loads(value)["confidence_interval_95"][1]
    )
    interval_columns = [
        "validation_rmse_mean",
        "validation_rmse_ci_low",
        "validation_rmse_ci_high",
    ]
    if not np.all(
        np.isfinite(parsed[interval_columns].to_numpy(dtype=float))
    ):
        raise ValueError("Independent validation summaries require finite means and intervals")
    return {
        "doe_development": doe,
        "mo_tpe_development": mo_tpe,
        "independent_selected": parsed,
    }


def plot_revision_pareto_front(
    run_dir: str | Path,
    save_path: str = "figures/revision_v2_pareto_front.png",
) -> None:
    """Separate matched development fronts from frozen-selection evaluation."""
    data = load_revision_pareto_data(run_dir)
    doe = data["doe_development"]
    mo_tpe = data["mo_tpe_development"]
    independent = data["independent_selected"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)

    axes[0].scatter(
        doe["validation_rmse"],
        doe["predict_latency_us"],
        label="DOE evaluated candidates",
        alpha=0.72,
        edgecolor="black",
        linewidth=0.4,
    )
    axes[0].scatter(
        mo_tpe["validation_rmse"],
        mo_tpe["predict_latency_us"],
        label="MO-TPE trials",
        marker="s",
        alpha=0.72,
        edgecolor="black",
        linewidth=0.4,
    )
    doe_front = pareto_front_2d_min(
        doe[["validation_rmse", "predict_latency_us"]].to_numpy()
    )
    if doe_front.points:
        points = np.asarray(doe_front.points)
        axes[0].step(
            points[:, 0], points[:, 1], where="post", color="#1f77b4", label="DOE front"
        )
    for replicate_id, replicate in mo_tpe.groupby("replicate_id"):
        front = pareto_front_2d_min(
            replicate[["validation_rmse", "predict_latency_us"]].to_numpy()
        )
        if front.points:
            points = np.asarray(front.points)
            axes[0].step(
                points[:, 0],
                points[:, 1],
                where="post",
                alpha=0.75,
                label=f"MO-TPE front, replicate {int(replicate_id)}",
            )
    development_protocol = doe["latency_protocol_id"].iloc[0]
    axes[0].set_title("Development candidates on one matched split")
    axes[0].set_xlabel("Search-time validation RMSE")
    axes[0].set_ylabel(f"predict() latency (μs)\n{development_protocol}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    markers = {
        "historical_preplanned_doe_desirability": "*",
        "historical_preplanned_doe_single_objective": "^",
        "multi_objective_tpe": "s",
        "constrained_tpe": "X",
        "single_objective_tpe": "P",
        "random_search": "o",
    }
    for optimizer, group in independent.groupby("optimizer"):
        x = group["validation_rmse_mean"].to_numpy(dtype=float)
        y = group["predict_latency_us"].to_numpy(dtype=float)
        low = group["validation_rmse_ci_low"].to_numpy(dtype=float)
        high = group["validation_rmse_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack((x - low, high - x))
        yerr = group["predict_latency_session_standard_deviation_us"].fillna(0).to_numpy(dtype=float)
        axes[1].errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=markers.get(optimizer, "o"),
            linestyle="none",
            capsize=2,
            label=optimizer.replace("_", " "),
        )
    primary_protocol = independent["latency_protocol_id"].iloc[0]
    axes[1].set_title("Frozen selections: independent validation means")
    axes[1].set_xlabel("Independent validation RMSE mean (95% t interval)")
    axes[1].set_ylabel(
        f"predict() latency (μs; bars = between-session SD)\n{primary_protocol}"
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        "Validation-domain Pareto evidence; panels use distinct, explicitly labelled estimands"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=300)
    plt.close(fig)
    print(f"[Plot] Saved revision Pareto plot to: {destination}")


def plot_efficiency_comparison(save_path: str = "figures/efficiency_comparison_curve.png"):
    """Render the historical v1 efficiency figure for archival reproduction only.

    Median and IQR shaded bands across 20 replicate runs for Random Search and TPE,
    with single illustrative DOE trajectory in actual randomized run_order.

    This mixes a single DOE run-order trace with optimizer replicate summaries and
    is deliberately excluded from the revision-v2 rendering entry point.
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
    ax.plot(evals, doe_cum_best, color="#1f77b4", linewidth=2.5, linestyle="-", label="DOE Design (Single Illustrative Trajectory, Run Order)")

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


def render_revision_plots(revision_run_dir: str | Path):
    """Render the supported diagnostics plus the estimand-separated v2 Pareto figure."""
    df_runs = pd.read_csv("results/runs.csv")
    plot_residual_diagnostics(df_runs, "figures/diagnostics_panel_4in1.png")
    plot_response_surface_rmse_2d_3d(df_runs, "figures/response_surface_rmse_2d_3d.png")
    plot_latency_vs_depth(df_runs, "figures/response_surface_latency_2d_3d.png")
    plot_pareto_front_and_desirability(df_runs, "figures/desirability_pareto_front.png")
    plot_revision_pareto_front(
        revision_run_dir, "figures/revision_v2_pareto_front.png"
    )
    print("Revision diagnostics and estimand-separated Pareto plot rendered successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render plots from an explicit versioned revision-v2 run."
    )
    parser.add_argument("--revision-run-dir", required=True)
    arguments = parser.parse_args()
    render_revision_plots(arguments.revision_run_dir)
