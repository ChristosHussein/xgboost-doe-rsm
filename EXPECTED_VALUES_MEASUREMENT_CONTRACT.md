# Pre-Experiment Measurement Contract & Expected Values Registry

> **Document Status**: `PRE-EXPERIMENT REGISTERED CONTRACT` (Phase -1 as per Scientific Integrity Skill).  
> **Registration Date**: October 28, 2026.  
> **Scope**: Pre-specification of all metric IDs, source selectors, mathematical derivations, denominators, uncertainty estimators, expected bounds, and report claim bindings before running the full IMDb experimental campaign. No observed numbers are fabricated or claimed as empirical findings prior to experimental execution.

---

## 1. Study Invariants & Experimental Accounting

| Parameter | Planned Contract Value | Verification Constraint |
|---|---|---|
| **Target Dataset** | Full IMDb Sentiment Dataset (`imdb.csv`) | SHA-256 verified against `imdb.json` manifest |
| **Development Pool** | 25,000 documents (Stratified 5-fold cross-validation) | 20,000 train / 5,000 validation per fold |
| **Protected Test Set** | 25,000 documents (Untouched official test partition) | Strictly isolated behind `FinalTestEvaluator` gate |
| **Nuisance Blocks** | $B = 5$ stratified folds | Every configuration evaluated across all 5 blocks |
| **Search Replicates** | $R = 3$ independent search runs with randomized seeds | Seeds: 42, 43, 44 |
| **Methods Evaluated** | Static Face-Centered CCD, Sequential CCD, MO-TPE | Fair budget: 140 fits per method per replicate |
| **Fit Budget Ledger** | 3 methods $\times$ 3 replicates $\times$ 140 fits = 1,260 search fits | Strict equality: no method receives extra fits |
| **Confirmation Budget** | 12 coordinates $\times$ 5 blocks $\times$ 3 replicates = 180 fits/method | Reserved interior and boundary coordinates |
| **Primary Responses** | $Y_1$: Validation Loss ($1 - \text{Macro F1}$), $Y_2$: Latency ($\mu\text{s}/\text{doc}$) | Single-document warmed median latency on pinned CPU |

---

## 2. Canonical Claim Registry & Expected Output Metrics

Every metric below possesses a unique semantic `metric_id`, binding selector, formula, and expected mathematical bound. When the experiment runs, automated reporting will extract the observed numbers strictly from these source paths.

### 2.1 Model Quality & Performance Metrics

| Metric ID | Semantic Target | Source Selector | Derivation Formula | Expected Bound | Direction |
|---|---|---|---|---|---|
| `imdb.static_doe.val_macro_f1.mean` | Validation Macro F1 for Static CCD | `summary.json:methods.static_doe.replicates[*].best_observed_loss` | $1.0 - \frac{1}{R} \sum_{r=1}^R \min_{\mathbf{x}} \bar{L}_r(\mathbf{x})$ | $[0.840, 0.920]$ | Higher is better |
| `imdb.static_doe.val_macro_f1.sd` | Cross-replicate SD of Macro F1 | Same | Sample standard deviation ($ddof=1$) | $[0.001, 0.025]$ | Lower is better |
| `imdb.sequential_doe.val_macro_f1.mean` | Validation Macro F1 for Sequential CCD | `summary.json:methods.doe.replicates[*].best_observed_loss` | $1.0 - \frac{1}{R} \sum_{r=1}^R \min_{\mathbf{x}} \bar{L}_r(\mathbf{x})$ | $[0.850, 0.925]$ | Higher is better |
| `imdb.sequential_doe.val_macro_f1.sd` | Cross-replicate SD of Macro F1 | Same | Sample standard deviation ($ddof=1$) | $[0.001, 0.020]$ | Lower is better |
| `imdb.mo_tpe.val_macro_f1.mean` | Validation Macro F1 for MO-TPE | `summary.json:methods.tpe.replicates[*].best_observed_loss` | $1.0 - \frac{1}{R} \sum_{r=1}^R \min_{\mathbf{x}} \bar{L}_r(\mathbf{x})$ | $[0.850, 0.925]$ | Higher is better |
| `imdb.mo_tpe.val_macro_f1.sd` | Cross-replicate SD of Macro F1 | Same | Sample standard deviation ($ddof=1$) | $[0.002, 0.030]$ | Lower is better |
| `imdb.final_test.macro_f1` | Protected Holdout Test Macro F1 | `final_results.json:results[*].val_macro_f1` | Reconstructed from true test labels and predictions | $[0.840, 0.915]$ | Higher is better |
| `imdb.final_test.logloss` | Protected Holdout Test Log Loss | `final_results.json:results[*].val_logloss` | Cross-entropy loss on test set | $[0.300, 0.500]$ | Lower is better |

### 2.2 Latency & Feasibility Metrics

| Metric ID | Semantic Target | Source Selector | Derivation Formula | Expected Bound | Direction |
|---|---|---|---|---|---|
| `imdb.latency_threshold_us` | Latency SLA Threshold | `calibration.json:latency_threshold_us` | Median classic latency across calibration fits | $[100.0, 250.0]\,\mu\text{s}$ | Calibration invariant |
| `imdb.static_doe.latency_us.mean` | Selected Model Latency (Static) | `final_results.json:results[static_doe].latency` | Median over 500 timed single-document calls | $[110.0, 240.0]\,\mu\text{s}$ | Lower is better |
| `imdb.sequential_doe.latency_us.mean` | Selected Model Latency (Sequential) | `final_results.json:results[doe].latency` | Median over 500 timed single-document calls | $[110.0, 240.0]\,\mu\text{s}$ | Lower is better |
| `imdb.mo_tpe.latency_us.mean` | Selected Model Latency (TPE) | `final_results.json:results[tpe].latency` | Median over 500 timed single-document calls | $[110.0, 280.0]\,\mu\text{s}$ | Lower is better |
| `imdb.static_doe.feasibility_rate` | Search Feasibility Fraction (Static) | `summary.json:methods.static_doe.replicates[*].feasibility_rate` | $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{latency}_i \le \text{Threshold})$ | $[0.50, 0.95]$ | Higher is better |
| `imdb.sequential_doe.feasibility_rate` | Search Feasibility Fraction (Sequential) | `summary.json:methods.doe.replicates[*].feasibility_rate` | Same | $[0.50, 0.95]$ | Higher is better |
| `imdb.mo_tpe.feasibility_rate` | Search Feasibility Fraction (TPE) | `summary.json:methods.tpe.replicates[*].feasibility_rate` | Same | $[0.35, 0.85]$ | Higher is better |

### 2.3 Optimization Multi-Objective Metrics

| Metric ID | Semantic Target | Source Selector | Derivation Formula | Expected Bound | Direction |
|---|---|---|---|---|---|
| `imdb.static_doe.hypervolume.mean` | Mean 2D Hypervolume (Static) | `summary.json:methods.static_doe.hypervolume_mean` | Dominated area to ref $(1.0, 2\times \text{lat}_{\max})$ | $[25.0, 45.0]$ | Higher is better |
| `imdb.sequential_doe.hypervolume.mean` | Mean 2D Hypervolume (Sequential) | `summary.json:methods.doe.hypervolume_mean` | Dominated area to ref $(1.0, 2\times \text{lat}_{\max})$ | $[28.0, 48.0]$ | Higher is better |
| `imdb.mo_tpe.hypervolume.mean` | Mean 2D Hypervolume (TPE) | `summary.json:methods.tpe.hypervolume_mean` | Dominated area to ref $(1.0, 2\times \text{lat}_{\max})$ | $[28.0, 48.0]$ | Higher is better |
| `imdb.sequential_doe.regret` | Observed Reference Regret | `summary.json:methods.doe.replicates[*].observed_reference_regret` | $\min L_{\text{DOE}} - \min (L_{\text{DOE}}, L_{\text{TPE}})$ | $[0.000, 0.020]$ | Lower is better |

### 2.4 Independent Confirmation & Generalization Metrics

| Metric ID | Semantic Target | Source Selector | Derivation Formula | Expected Bound | Direction |
|---|---|---|---|---|---|
| `imdb.ols.val_loss.rmse` | OLS Confirmation RMSE | `confirmation.json:rmse` | $\sqrt{\frac{1}{M}\sum (y_m - \hat{y}_m)^2}$ on reserved coords | $[0.010, 0.060]$ | Lower is better |
| `imdb.ols.val_loss.pi_coverage` | OLS Prediction Interval Coverage | `confirmation.json:pi_coverage` | $\frac{1}{M}\sum \mathbb{I}(\hat{y}_{m,\text{low}} \le y_m \le \hat{y}_{m,\text{high}})$ | $[0.85, 0.98]$ | Nominal: 0.95 |
| `imdb.ols.boundary.bias` | Boundary Candidate Optimism Bias | `confirmation.json:boundary_bias` | $\frac{1}{M_B} \sum_{m \in B} (y_m - \hat{y}_m)$ on $x_i = \pm 1$ | $[-0.040, +0.010]$ | 0 = unbiased |
| `imdb.gp.val_loss.rmse` | Gaussian Process Confirmation RMSE | `gp_confirmation_*.json:rmse` | Forecast vs observed block-mean loss | $[0.008, 0.050]$ | Lower is better |
| `imdb.gp.val_loss.coverage` | GP Model-Based Interval Coverage | `gp_confirmation_*.json:coverage` | Fraction inside $\hat{\mu} \pm 1.96 \hat{\sigma}$ | $[0.80, 0.98]$ | Nominal: 0.95 |

### 2.5 Explainable AI (ANOVA vs SHAP) Metrics

| Metric ID | Semantic Target | Source Selector | Derivation Formula | Expected Bound | Direction |
|---|---|---|---|---|---|
| `imdb.anova_shap.val_loss.spearman_rho` | Factor Ranking Agreement | `snapshot_*.json:comparisons.rsm.spearman_rho` | Spearman rank correlation of main effects | $[0.600, 1.000]$ | 1.0 = perfect agreement |
| `imdb.anova_shap.val_loss.kendall_tau` | Factor Ranking Concordance | `snapshot_*.json:comparisons.rsm.kendall_tau` | Kendall rank correlation coefficient | $[0.500, 1.000]$ | 1.0 = perfect agreement |
| `imdb.anova_shap.interaction_correlation` | Pairwise Interaction Correlation | `snapshot_*.json:comparisons.rsm.interaction_correlation` | Pearson $r$ between ANOVA interaction SS and $\Phi_{ij}$ | $[0.550, 0.980]$ | 1.0 = perfect correlation |
| `imdb.anova.top_factor` | Most Influential Factor (ANOVA) | `snapshot_*.json:attribution.main_effect_ranking[0]` | Factor with highest partial sum of squares | `x1` or `x2` | Categorical |
| `imdb.shap.top_factor` | Most Influential Factor (SHAP) | `snapshot_*.json:shap.rsm.shap_ranking[0]` | Factor with highest mean absolute Shapley value | `x1` or `x2` | Categorical |

### 2.6 Local Word Explanation Metrics (Document-Level)

| Metric ID | Semantic Target | Source Selector | Derivation Formula | Expected Bound | Direction |
|---|---|---|---|---|---|
| `imdb.word_rsm.fidelity_rmse` | Local RSM Perturbation RMSE | `word_study/observations.json:documents[*].confirmation.rmse` | RMSE on 64 unmeasured retention weights | $[0.005, 0.080]$ | Lower is better |
| `imdb.word_rsm.r2` | Local RSM Explained Variance | Same | Coefficient of determination $R^2$ | $[0.850, 0.999]$ | Higher is better |
| `imdb.word_shap.interaction_present` | Significant Word Interactions | Same | Count of non-zero pairwise word SHAP values | $\ge 1$ pair / document | Boolean |

---

## 3. Pre-Run Acceptance Gate

Prior to initiating the long execution run on the full IMDb dataset:
1. `python run_experiment.py --stage smoke` has run cleanly with exit code 0.
2. The dataset file `data/imdb.csv` matches the SHA-256 hash registered in `data/imdb.json`.
3. Calibration selects tree count on development data only without accessing official test documents.
4. The exact output directories (`results/experiment_run/imdb` and `findings/`) are verified ready to receive measurements.
5. All 182 test suite checks pass without failure.
