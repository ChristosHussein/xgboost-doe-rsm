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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
RUNS = ROOT / "results" / "runs.csv"
FINAL_SUMMARY = ROOT / "results" / "revision_v2" / "full_run_001" / "final_summary.csv"
OPTIMIZER_TRIALS = ROOT / "results" / "revision_v2" / "full_run_001" / "optimizer_trials.csv"
DOE_SELECTION_RUNS = ROOT / "results" / "revision_v2" / "full_run_001" / "doe_selection_runs.csv"
ARCHIVED_TRAJECTORIES = ROOT / "results" / "benchmark_evals_trajectories.csv"

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


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float,
    height: float,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=color,
        edgecolor="#4b5966",
        linewidth=0.9,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color="#17212b", linespacing=1.2, transform=ax.transAxes)


def research_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axis("off")
    box_width = 0.19
    box_height = 0.12
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
        (0.115, 0.72), (0.37, 0.72), (0.625, 0.72), (0.88, 0.72),
        (0.88, 0.34), (0.625, 0.34), (0.37, 0.34), (0.115, 0.34),
    ]
    fills = ["#dcecf8", "#eaf3fa", "#e8f2e9", "#e8f2e9", "#fff1dc", "#f5eafb", "#fde9e7", "#e7f4ef"]
    for (x, y), label, fill in zip(positions, labels, fills):
        _box(ax, x, y, label, fill, box_width, box_height)

    gap = 0.012
    for (x, y), (nx, ny) in zip(positions, positions[1:]):
        if y == ny:
            direction = 1 if nx > x else -1
            start = (x + direction * (box_width / 2 + gap), y)
            end = (nx - direction * (box_width / 2 + gap), ny)
        else:
            start = (x, y - box_height / 2 - gap)
            end = (nx, ny + box_height / 2 + gap)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                color=COLORS["gray"],
                linewidth=1.35,
                transform=ax.transAxes,
            )
        )
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


def efficiency_comparison(runs: pd.DataFrame) -> None:
    """Render a reader-facing version of the canonical efficiency evidence."""
    trials = pd.read_csv(OPTIMIZER_TRIALS)
    doe_runs = pd.read_csv(DOE_SELECTION_RUNS)
    archived = pd.read_csv(ARCHIVED_TRAJECTORIES)
    evaluations = np.arange(1, 141)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    styles = [
        ("single_objective_tpe", "TPE: accuracy", COLORS["green"], "-."),
        ("multi_objective_tpe", "TPE: accuracy + speed", COLORS["purple"], "-"),
        ("constrained_tpe", "TPE: latency constrained", COLORS["orange"], ":"),
        ("random_search", "Random search", COLORS["red"], "--"),
    ]
    repeated_curves = {}
    for key, label, color, linestyle in styles:
        subset = trials[trials["optimizer"].eq(key)].sort_values(["replicate_id", "trial_number"])
        matrix = np.vstack([
            group["validation_rmse"].cummin().to_numpy()[:140]
            for _, group in subset.groupby("replicate_id")
        ])
        median = np.median(matrix, axis=0)
        lower = np.percentile(matrix, 25, axis=0)
        upper = np.percentile(matrix, 75, axis=0)
        repeated_curves[key] = (median, lower, upper, color, linestyle)
        axes[0].plot(evaluations, median, color=color, lw=1.8, ls=linestyle, label=label)
        axes[0].fill_between(evaluations, lower, upper, color=color, alpha=0.12)

    ordered_doe = doe_runs.sort_values(["doe_replicate_id", "block_index", "order_within_block"])
    doe_matrix = np.vstack([
        group["validation_rmse"].cummin().to_numpy()[:140]
        for _, group in ordered_doe.groupby("doe_replicate_id")
    ])
    doe_median = np.median(doe_matrix, axis=0)
    doe_lower = np.percentile(doe_matrix, 25, axis=0)
    doe_upper = np.percentile(doe_matrix, 75, axis=0)
    axes[0].plot(evaluations, doe_median, color=COLORS["blue"], lw=2.1, label="Blocked DOE")
    axes[0].fill_between(evaluations, doe_lower, doe_upper, color=COLORS["blue"], alpha=0.14)
    axes[0].set(title="Repeated benchmark searches", xlabel="Cumulative model fits", ylabel="Best validation RMSE so far")

    initial_doe = runs.sort_values("run_order")["val_rmse"].cummin().to_numpy()
    axes[1].plot(evaluations, archived["rs_median"], color=COLORS["red"], lw=1.8, ls="--", label="Random search")
    axes[1].fill_between(evaluations, archived["rs_q25"], archived["rs_q75"], color=COLORS["red"], alpha=0.12)
    axes[1].plot(evaluations, archived["tpe_median"], color=COLORS["green"], lw=1.9, ls="-.", label="TPE: accuracy")
    axes[1].fill_between(evaluations, archived["tpe_q25"], archived["tpe_q75"], color=COLORS["green"], alpha=0.12)
    axes[1].plot(evaluations, initial_doe, color=COLORS["blue"], lw=2.1, label="Blocked DOE")
    axes[1].set(title="Initial experiment archive", xlabel="Cumulative model fits", ylabel="Best validation RMSE so far")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.set_xlim(1, 140)
        ax.legend(loc="upper right", fontsize=7.2, frameon=True)
        inset = ax.inset_axes([0.43, 0.22, 0.53, 0.42])
        inset.set_xlim(20, 140)
        inset.set_ylim(0.455, 0.500 if ax is axes[0] else 0.490)
        inset.grid(True, linestyle=":", alpha=0.4)
        inset.tick_params(labelsize=7)
        inset.set_title("Fits 20--140", fontsize=7.5)
        if ax is axes[0]:
            for median, lower, upper, color, linestyle in repeated_curves.values():
                inset.plot(evaluations, median, color=color, lw=1.2, ls=linestyle)
                inset.fill_between(evaluations, lower, upper, color=color, alpha=0.09)
            inset.plot(evaluations, doe_median, color=COLORS["blue"], lw=1.4)
        else:
            inset.plot(evaluations, archived["rs_median"], color=COLORS["red"], lw=1.2, ls="--")
            inset.plot(evaluations, archived["tpe_median"], color=COLORS["green"], lw=1.2, ls="-.")
            inset.plot(evaluations, initial_doe, color=COLORS["blue"], lw=1.4)

    fig.suptitle("Optimization progress under a common 140-fit budget", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "explainer_efficiency_comparison.pdf")


def main() -> None:
    _style()
    runs = pd.read_csv(RUNS)
    final_summary = pd.read_csv(FINAL_SUMMARY)
    research_pipeline()
    design_exploration()
    blocking_example(runs)
    efficiency_comparison(runs)
    print("Generated four reader-focused PDF figures from canonical evidence and protocol metadata.")


if __name__ == "__main__":
    main()
