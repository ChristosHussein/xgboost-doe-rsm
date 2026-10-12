"""Generate reader-focused methodology figures from canonical repository evidence.

The conceptual panels describe the implemented protocol. Empirical panels read only
the SHA-256-locked historical and Revision-v2 artifacts; they never refit XGBoost or
modify experimental evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RUNS = ROOT / "results" / "runs.csv"
FINAL_SUMMARY = ROOT / "results" / "revision_v2" / "full_run_001" / "final_summary.csv"

COLORS = {
    "blue": "#2369a8",
    "orange": "#e07a1f",
    "green": "#2c8c62",
    "red": "#bd3c3c",
    "purple": "#7453a6",
    "gray": "#66717e",
    "light": "#eef3f7",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "svg.hashsalt": "xgboost-doe-rsm-reader-report",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / name,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "scripts/generate_explanatory_figures.py", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _box(ax: plt.Axes, x: float, y: float, text: str, color: str, width: float = 0.13) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color="#17212b",
        linespacing=1.2,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": color, "edgecolor": "#4b5966", "linewidth": 0.9},
        transform=ax.transAxes,
    )


def research_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axis("off")
    labels = [
        "Question\naccuracy + speed",
        "Partition data\ndevelopment + holdout",
        "Pre-plan all\n140 DOE runs",
        "Repeat across\n5 seed blocks",
        "Measure RMSE\nand latency",
        "Fit and check\nresponse surfaces",
        "Select and freeze\ncandidates",
        "Confirm and\ncompare with TPE",
    ]
    positions = [
        (0.12, 0.72), (0.37, 0.72), (0.62, 0.72), (0.87, 0.72),
        (0.87, 0.34), (0.62, 0.34), (0.37, 0.34), (0.12, 0.34),
    ]
    fills = ["#dcecf8", "#eaf3fa", "#e8f2e9", "#e8f2e9", "#fff1dc", "#f5eafb", "#fde9e7", "#e7f4ef"]
    for i, ((x, y), label, fill) in enumerate(zip(positions, labels, fills)):
        _box(ax, x, y, label, fill)
        if i < len(labels) - 1:
            nx, ny = positions[i + 1]
            if y == ny:
                start_x = x + (0.075 if nx > x else -0.075)
                end_x = nx - (0.075 if nx > x else -0.075)
                ax.annotate("", xy=(end_x, ny), xytext=(start_x, y), xycoords=ax.transAxes,
                            arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.3})
            else:
                ax.annotate("", xy=(nx, ny + 0.075), xytext=(x, y - 0.075), xycoords=ax.transAxes,
                            arrowprops={"arrowstyle": "->", "color": COLORS["gray"], "lw": 1.3})
    ax.text(0.5, 0.08, "Historical execution: every FCCD run was scheduled up front and globally randomized; the analysis was hierarchical, not adaptive.",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#38434d")
    _save(fig, "explainer_research_pipeline.pdf")


def design_exploration() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    axial = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    ax.scatter(corners[:, 0], corners[:, 1], s=110, marker="s", color=COLORS["blue"], label="factorial corners")
    ax.scatter(axial[:, 0], axial[:, 1], s=130, marker="D", color=COLORS["orange"], label="face-centered axial points")
    ax.scatter([0], [0], s=170, marker="o", color=COLORS["green"], label="replicated center")
    for x in (-1, 0, 1):
        ax.axvline(x, color="#d9dee3", lw=0.7, zorder=0)
        ax.axhline(x, color="#d9dee3", lw=0.7, zorder=0)
    ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), xlabel="coded learning rate", ylabel="coded tree depth")
    ax.set_title("A two-factor projection of the four-factor FCCD")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.text(0.02, 0.02, "Conceptual projection: subsample and regularization dimensions are not shown.",
            transform=ax.transAxes, color=COLORS["gray"], fontsize=8)
    ax.set_aspect("equal")
    _save(fig, "explainer_design_projection.pdf")


def blocking_example(runs: pd.DataFrame) -> None:
    center = runs[runs["phase"].eq("Phase1_Center")].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    rng = np.random.default_rng(20261012)
    for block, grp in center.groupby("block", sort=True):
        x = np.full(len(grp), block, dtype=float) + rng.uniform(-0.08, 0.08, len(grp))
        ax.scatter(x, grp["val_rmse"], s=42, alpha=0.75, color=COLORS["blue"])
        ax.scatter(block, grp["val_rmse"].mean(), s=115, marker="D", color=COLORS["orange"], edgecolor="white", linewidth=0.8)
    ax.plot(center.groupby("block")["val_rmse"].mean(), color=COLORS["orange"], lw=1.3, alpha=0.8)
    ax.set(xlabel="data-split block", ylabel="validation RMSE", xticks=sorted(center["block"].unique()))
    ax.set_title("The same center configuration changes across data splits")
    ax.text(0.02, 0.03, "Dots: repeated model seeds within a split. Diamonds: block means. Lower RMSE is better.",
            transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
    _save(fig, "explainer_blocking.pdf")


def response_surface_fit(runs: pd.DataFrame) -> None:
    formula = (
        "val_rmse ~ C(block) + x1 + x2 + x3 + x4 + "
        "I(x1 ** 2) + I(x2 ** 2) + I(x3 ** 2) + I(x4 ** 2) + "
        "x1:x2 + x1:x3 + x1:x4 + x2:x3 + x2:x4 + x3:x4"
    )
    fit = smf.ols(formula, data=runs).fit()
    observed = runs["val_rmse"].to_numpy()
    predicted = fit.fittedvalues.to_numpy()
    residual = observed - predicted
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    lo = min(observed.min(), predicted.min())
    hi = max(observed.max(), predicted.max())
    axes[0].scatter(predicted, observed, s=25, alpha=0.65, color=COLORS["blue"])
    axes[0].plot([lo, hi], [lo, hi], "--", color=COLORS["gray"], lw=1.2)
    axes[0].set(xlabel="surface prediction", ylabel="observed validation RMSE", title="Observed measurements versus fitted surface")
    axes[1].scatter(runs["eta"], residual, s=25, alpha=0.65, color=COLORS["purple"])
    axes[1].axhline(0, color=COLORS["gray"], lw=1)
    axes[1].set_xscale("log")
    axes[1].set(xlabel="learning rate (log scale)", ylabel="observed minus fitted RMSE", title="Systematic residuals reveal lack of fit")
    fig.tight_layout()
    _save(fig, "explainer_response_surface.pdf")


def _json_mean(value: str) -> float:
    return float(json.loads(value)["mean"])


def tradeoff(final_summary: pd.DataFrame) -> None:
    data = final_summary.copy()
    data["test_mean"] = data["test_rmse"].map(_json_mean)
    labels = {
        "repeated_preplanned_doe_multi_objective": "Repeated DOE (multi-objective)",
        "repeated_preplanned_doe_single_objective": "Repeated DOE (accuracy only)",
        "multi_objective_tpe": "TPE (multi-objective)",
        "single_objective_tpe": "TPE (accuracy only)",
        "constrained_tpe": "TPE (latency constrained)",
        "random_search": "Random search",
    }
    palette = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["purple"], COLORS["gray"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for color, (method, label) in zip(palette, labels.items()):
        grp = data[data["optimizer"].eq(method)]
        if grp.empty:
            continue
        ax.scatter(grp["predict_latency_us"], grp["test_mean"], s=38, alpha=0.55, color=color, label=label)
        ax.scatter(grp["predict_latency_us"].mean(), grp["test_mean"].mean(), s=150, marker="*", color=color, edgecolor="black", linewidth=0.5)
    ax.axvline(145, color=COLORS["red"], linestyle="--", lw=1.1, label="latency constraint")
    ax.set(xlabel="verified single-sample latency (microseconds; lower is better)", ylabel="holdout test RMSE (lower is better)")
    ax.set_title("Accuracy and speed form a trade-off, not one universal winner")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    ax.text(0.02, 0.03, "Small points are frozen selections; stars are method means.", transform=ax.transAxes, fontsize=8, color=COLORS["gray"])
    _save(fig, "explainer_accuracy_latency.pdf")


def method_comparison() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.5))
    ax.axis("off")
    rows = [
        ("Blocked DOE", "25 planned coordinates", "5 split blocks", "effects, interactions, curvature"),
        ("Random search", "140 sampled coordinates", "one development split", "broad non-adaptive exploration"),
        ("TPE variants", "140 adaptive coordinates", "one development split", "concentrated optimization"),
    ]
    headers = ["Method", "Where the 140-fit budget goes", "Noise handling during search", "Primary information gained"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="left", colLoc="left", colWidths=[0.16, 0.27, 0.24, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.0)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#b8c2ca")
        cell.set_facecolor("#dcecf8" if row == 0 else ("#f7fafc" if row % 2 else "#edf3f6"))
        if row == 0:
            cell.set_text_props(weight="bold")
    ax.set_title("Equal fit counts do not imply equal information", pad=18)
    ax.text(0.5, 0.07,
            "Revision-v2 repeated each search 20 times and retrained frozen selections across 20 evaluation seeds.",
            ha="center", transform=ax.transAxes, fontsize=8.5, color=COLORS["gray"])
    _save(fig, "explainer_method_comparison.pdf")


def main() -> None:
    _style()
    runs = pd.read_csv(RUNS)
    final_summary = pd.read_csv(FINAL_SUMMARY)
    research_pipeline()
    design_exploration()
    blocking_example(runs)
    response_surface_fit(runs)
    tradeoff(final_summary)
    method_comparison()
    print("Generated six reader-focused PDF figures from canonical evidence and protocol metadata.")


if __name__ == "__main__":
    main()
