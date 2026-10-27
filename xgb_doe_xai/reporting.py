"""xgb_doe_xai.reporting - Formats ANOVA tables, SHAP comparisons, and benchmark summaries.
"""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
from xgb_doe_xai.benchmarks import EvaluationResult
from xgb_doe_xai.surrogates import AnovaTableResult
from xgb_doe_xai.xai import AnovaShapComparisonResult


def _number(value):
    return f'{value:.4f}' if value is not None else 'UNDEFINED'


def render_latex_anova_table(anova: AnovaTableResult, caption: str = "ANOVA for Second-Order Response Surface") -> str:
    """Render publication-grade LaTeX table for ANOVA decomposition."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{tab:anova_rsm}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"\textbf{Source of Variation} & \textbf{DF} & \textbf{Sum of Squares} & \textbf{Mean Square} & \textbf{$F$-Statistic} & \textbf{$p$-Value} \\",
        r"\hline",
    ]
    for src, df, ss, ms, f, p in zip(
        anova.source, anova.df, anova.sum_sq, anova.mean_sq, anova.f_stat, anova.p_value
    ):
        f_str = f"{f:.2f}" if f is not None else "--"
        if p is None:
            p_str = "--"
        elif p < 0.0001:
            p_str = r"$< 10^{-4}$"
        else:
            p_str = f"{p:.4f}"

        # LaTeX formatting for indentation
        src_tex = src.replace("  ", r"\quad ")
        lines.append(f"{src_tex} & {df} & {ss:.6f} & {ms:.6f} & {f_str} & {p_str} \\\\")

    lines.extend([
        r"\hline",
        r"\multicolumn{6}{l}{\small $R^2 = " + _number(anova.r2) + r", \quad \text{Adjusted } R^2 = " + _number(anova.adj_r2) + r"$} \\",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def render_markdown_anova_table(anova: AnovaTableResult) -> str:
    """Render clean Markdown table for ANOVA decomposition."""
    lines = [
        "| Source of Variation | DF | Sum of Squares | Mean Square | F-Statistic | p-Value |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for src, df, ss, ms, f, p in zip(
        anova.source, anova.df, anova.sum_sq, anova.mean_sq, anova.f_stat, anova.p_value
    ):
        f_str = f"{f:.2f}" if f is not None else "-"
        p_str = f"{p:.4e}" if p is not None else "-"
        lines.append(f"| {src} | {df} | {ss:.6f} | {ms:.6f} | {f_str} | {p_str} |")

    lines.append(f"\n**Model Fit:** $R^2 = {_number(anova.r2)}$, Adjusted $R^2 = {_number(anova.adj_r2)}$")
    lines.append('Group effects are partial sums of squares, not generally an additive SS partition. '
                 'Undefined F/LoF tests indicate insufficient estimated error variance.')
    if anova.f_lof is not None and anova.p_lof is not None:
        lines.append(f"**Lack of Fit Test:** $F = {anova.f_lof:.2f}$ ($p = {anova.p_lof:.4e}$)")
    return "\n".join(lines)


def render_markdown_comparison_summary(loss_comparison, latency_comparison=None) -> str:
    """Descriptive agreement of different estimands, not equivalence or validation."""
    def display(value):
        if value is None:
            return "UNDEFINED"
        if isinstance(value, tuple):
            return ":".join(value)
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    lines = ["### ANOVA vs SHAP Methodological Cross-Validation Summary", "",
             "Descriptive comparison, not statistical validation or mathematical equivalence.",
             "ANOVA scores are partial SS; SHAP scores are mean absolute attributions.",
             "Exact interventional interactions use the declared empirical background.", "",
             "| Metric | Loss (1 - macro F1) | Latency (us) |",
             "| :--- | :--- | :--- |"]
    for label, key in [("Spearman rho", "spearman_rho"), ("Kendall tau", "kendall_tau"),
                       ("Interaction correlation", "interaction_correlation"),
                       ("Top factor ANOVA", "top_factor_anova"), ("Top factor SHAP", "top_factor_shap"),
                       ("Top interaction ANOVA", "top_interaction_anova"),
                       ("Top interaction SHAP", "top_interaction_shap")]:
        a = display(getattr(loss_comparison, key))
        b = display(getattr(latency_comparison, key)) if latency_comparison else "NOT RUN"
        lines.append(f"| {label} | {a} | {b} |")
    lines += ["", "Normalized-score differences (descriptive, without significance thresholds):"]
    for factor, value in loss_comparison.discrepancy_scores.items():
        lines.append(f"- {factor}: {value:.4f}")
    return "\n".join(lines)
