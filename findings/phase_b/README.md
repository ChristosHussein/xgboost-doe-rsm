# Experimental Findings & Deliverables

Generated on 2026-10-28T21:08:46.479573+00:00 from `imdb`.

## Contents:
- `tables/`:
  - `anova_table_val_loss.csv` & `.tex`: 2nd-order ANOVA decomposition for validation loss.
  - `anova_table_latency_us.csv` & `.tex`: 2nd-order ANOVA decomposition for latency.
  - `benchmark_comparison.csv` & `.tex`: Multi-objective Pareto, loss, and feasibility comparison across methods.
  - `confirmation_predictions.csv` & `.tex`: Independent confirmation and boundary bias on reserved coordinates.
- `figures/`:
  - `pareto_front_comparison.png`: Accuracy vs Latency trade-off and SLA threshold.
  - `anova_vs_shap_importance.png`: Comparative factor importance (ANOVA SS vs Mean |SHAP|).
  - `interaction_heatmaps.png`: Pairwise factor interaction matrices.
- `values/`:
  - `metrics.json`: Machine-readable store of all reconstructed values.
  - `macros.tex`: Ready-to-use LaTeX macros for scientific reporting.
