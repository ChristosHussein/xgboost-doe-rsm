# Research question and study design

Can designed experiments identify useful XGBoost settings, explain their effects, and guide further evaluations efficiently?

Phase A studies California Housing regression with a blocked central composite design. Response surfaces describe prediction error and latency; separate confirmation and repeated optimizer comparisons assess their practical usefulness. Curvature, local prediction errors, timing uncertainty and historical test exposure motivate the sentiment study.

Phase B compares static CCD, sequential CCD and multi-objective TPE on IMDb with shared folds and equal fit budgets. ANOVA is compared with surrogate SHAP for factor and interaction importance. Reserved configurations assess response predictions; local word-feature perturbations assess explanation fidelity.

The findings and methods are presented in [report/report.pdf](report/report.pdf). Housing RMSE and sentiment F1 describe different outcomes and are not pooled into a single accuracy score.
