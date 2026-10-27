"""xgb_doe_xai.reporting - Formats ANOVA tables, SHAP comparisons, and benchmark summaries.
"""

from __future__ import annotations

from typing import List, Optional
import pandas as pd
from xgb_doe_xai.benchmarks import EvaluationResult
from xgb_doe_xai.surrogates import AnovaTableResult
from xgb_doe_xai.xai import AnovaShapComparisonResult


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
        r"\multicolumn{6}{l}{\small $R^2 = " + f"{anova.r2:.4f}" + r", \quad \text{Adjusted } R^2 = " + f"{anova.adj_r2:.4f}" + r"$} \\",
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

    lines.append(f"\n**Model Fit:** $R^2 = {anova.r2:.4f}$, Adjusted $R^2 = {anova.adj_r2:.4f}$")
    if anova.f_lof is not None and anova.p_lof is not None:
        lines.append(f"**Lack of Fit Test:** $F = {anova.f_lof:.2f}$ ($p = {anova.p_lof:.4e}$)")
    return "\n".join(lines)


def render_markdown_comparison_summary(
    loss_comparison: AnovaShapComparisonResult,
    latency_comparison: Optional[AnovaShapComparisonResult] = None,
) -> str:
    """Render ANOVA vs SHAP methodological comparison summary in Markdown."""
    lines = [
        "### ANOVA vs SHAP Methodological Cross-Validation Summary",
        "",
        r"| Metric / Criterion | Response: Loss ($1 - F_1$) | Response: Latency ($\mu$s) |",
        "| :--- | :--- | :--- |",
        f"| **Main Effect Spearman $\\rho$** | `{loss_comparison.spearman_rho:.4f}` | `{latency_comparison.spearman_rho:.4f}` |" if latency_comparison else f"| **Main Effect Spearman $\\rho$** | `{loss_comparison.spearman_rho:.4f}` | - |",
        f"| **Main Effect Kendall $\\tau$** | `{loss_comparison.kendall_tau:.4f}` | `{latency_comparison.kendall_tau:.4f}` |" if latency_comparison else f"| **Main Effect Kendall $\\tau$** | `{loss_comparison.kendall_tau:.4f}` | - |",
        f"| **Interaction Matrix Correlation** | `{loss_comparison.interaction_correlation:.4f}` | `{latency_comparison.interaction_correlation:.4f}` |" if latency_comparison else f"| **Interaction Matrix Correlation** | `{loss_comparison.interaction_correlation:.4f}` | - |",
        f"| **Top Factor (ANOVA)** | `{loss_comparison.top_factor_anova}` | `{latency_comparison.top_factor_anova}` |" if latency_comparison else f"| **Top Factor (ANOVA)** | `{loss_comparison.top_factor_anova}` | - |",
        f"| **Top Factor (SHAP)** | `{loss_comparison.top_factor_shap}` | `{latency_comparison.top_factor_shap}` |" if latency_comparison else f"| **Top Factor (SHAP)** | `{loss_comparison.top_factor_shap}` | - |",
        f"| **Dominant Interaction (ANOVA)** | `{loss_comparison.top_interaction_anova[0]}:{loss_comparison.top_interaction_anova[1]}` | `{latency_comparison.top_interaction_anova[0]}:{latency_comparison.top_interaction_anova[1]}` |" if latency_comparison else f"| **Dominant Interaction (ANOVA)** | `{loss_comparison.top_interaction_anova[0]}:{loss_comparison.top_interaction_anova[1]}` | - |",
        f"| **Dominant Interaction (SHAP)** | `{loss_comparison.top_interaction_shap[0]}:{loss_comparison.top_interaction_shap[1]}` | `{latency_comparison.top_interaction_shap[0]}:{latency_comparison.top_interaction_shap[1]}` |" if latency_comparison else f"| **Dominant Interaction (SHAP)** | `{loss_comparison.top_interaction_shap[0]}:{loss_comparison.top_interaction_shap[1]}` | - |",
        "",
        "**Discrepancy Diagnostics:**",
    ]
    for factor, disc in sorted(loss_comparison.discrepancy_scores.items()):
        status = "Aligned" if disc < 0.20 else "Moderate Discrepancy" if disc < 0.40 else "High Discrepancy"
        lines.append(f"- **{factor}**: Discrepancy score = `{disc:.4f}` ({status})")
    return "\n".join(lines)
