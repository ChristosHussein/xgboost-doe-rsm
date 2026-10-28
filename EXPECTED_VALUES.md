# Pre-Experiment Measurement Contract & Expected Values Registry
## Unified Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees

> **Document Status**: `PRE-EXPERIMENT REGISTERED CONTRACT & PARAMETERS REGISTRY` (as per Scientific Integrity Protocol).  
> **Registration Date**: October 2026.  
> **Scope**: Pre-specification of all study invariants, parameters of interest, semantic metric IDs, mathematical derivations, expected pre-run bounds, and post-run output binding targets across **BOTH Phase A and Phase B** before full experimental replication.  
> **Integrity Rule**: No observed numbers are fabricated or claimed as empirical findings prior to running the respective experimental phase. When experiments execute, all values are automatically populated from the unified `findings/` directory.

---

## 1. Unified Project Architecture & Phase Mapping

This project represents a single unified investigation:
**Multi-Objective Hyperparameter Optimization for Gradient Boosted Trees under Stochastic Nuisance Blocking via Response Surface Methodology and Explainable AI**.

The investigation progresses across two procedural phases:
- **Phase A — Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking**:  
  Evaluated on the California Housing benchmark ($N = 20,640$ continuous observations, 8 features, 5 spatial nuisance blocks). Investigates factorial screening, first-order blocking, second-order CCD curvature resolution, canonical stationary analysis, and multi-objective desirability surfaces under pinned CPU latency constraints.
- **Phase B — High-Dimensional Sequential / Adaptive Central Composite Design with Explainable AI & Warmed Latency Calibration**:  
  Evaluated on the IMDb Sentiment Analysis benchmark ($N = 50,000$ documents, 10,000 sparse TF-IDF n-grams, 5 stratified cross-validation folds). Extends the methodology into high-dimensional NLP classification, resolving capacity underfitting via pre-experiment calibration, enforcing warmed single-core latency protocols, testing sequential coordinate augmentation, evaluating ANOVA vs SHAP interaction agreement, and deriving document-level word response surfaces.

---

## 2. Phase A: Continuous Tabular RSM & Stochastic Blocking

### 2.1 Study Invariants & Experimental Accounting (Phase A)

| Parameter | Planned Contract Value | Verification Constraint |
|---|---|---|
| **Benchmark Dataset** | California Housing Benchmark (`fetch_california_housing`) | $N=20,640$ records, 8 continuous features, standardized targets |
| **Data Partition** | 80% Development Pool ($N=16,512$), 20% Protected Holdout ($N=4,128$) | Holdout strictly isolated until final gated verification |
| **Nuisance Blocks** | $B = 5$ spatial / geographic coordinate blocks | Every design point evaluated across all 5 blocks |
| **Factors ($k=4$)** | $x_1$: `max_depth` $[3, 9]$, $x_2$: `learning_rate` $[0.05, 0.30]$, $x_3$: `subsample` $[0.50, 1.00]$, $x_4$: `reg_lambda` $[0.10, 10.0]$ | Log-scale coding for $\lambda$, linear coding for others |
| **Phase 1 Screen** | $2^{4-1}$ or $2^4$ factorial + 4 center replicates per block | 20 runs per block ($N=100$ total Phase 1 evaluations) |
| **Phase 2 Augmentation** | Axial / Star points ($\alpha = 1.0$ face-centered or spherical $\sqrt{k} = 2.0$) | Added strictly upon curvature detection ($p_{\text{curv}} < 0.05$) |
| **Phase 3 CCD** | Complete Central Composite Design ($N=30$ design points $\times 5$ blocks = 150 fits) | Full quadratic response surface with block nuisance terms |
| **Latency Protocol** | Pinned CPU execution on Core 0, single-row inference timing | 50 warmup calls, 1,000 timed evaluations per configuration |
| **Benchmark Baselines** | Single-Objective TPE, Multi-Objective TPE, Unguided Random Search | Fair equal evaluation budget (140 fits per method) |

### 2.2 Parameters of Interest & Expected Value Bounds (Phase A)

| Metric ID | Semantic Target | Derivation / Formula | Pre-Experiment Expected Bound | Post-Run Fill Source |
|---|---|---|---|---|
| `phase_a.phase1.f_curvature` | First-Order Curvature $F$-statistic | $F = \frac{(\bar{y}_{\text{fact}} - \bar{y}_{\text{ctr}})^2 / (1/n_F + 1/n_C)}{MS_{\text{pure\_error}}}$ | $[1,000, 150,000]$ (Highly significant) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.phase1.p_curvature` | Curvature Detection $p$-value | $P(F > F_{\text{obs}})$ | $< 0.0001$ (Curvature confirmed) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.blocking.icc_rmse` | Intraclass Correlation Coefficient (RMSE) | $\text{ICC} = \frac{\sigma^2_{\text{block}}}{\sigma^2_{\text{block}} + \sigma^2_{\text{residual}}}$ | $[0.25, 0.55]$ (Substantial block variance) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.blocking.f_block` | Nuisance Block $F$-statistic | $MS_{\text{block}} / MS_{\text{residual}}$ | $[10.0, 35.0]$ ($p < 0.001$) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.blocking.icc_latency` | Latency Block Variance Fraction | $\text{ICC}_{\text{lat}}$ across data blocks | $[0.00, 0.05]$ (Block-invariant latency) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.ccd.r2_rmse` | Second-Order Surface $R^2$ (RMSE) | $1 - SS_{\text{res}} / SS_{\text{tot}}$ | $[0.980, 0.999]$ | `findings/phase_a/values/diagnostics.json` |
| `phase_a.ccd.adj_r2_rmse` | Adjusted $R^2$ (RMSE) | $1 - (1 - R^2)\frac{n-1}{n-p-1}$ | $[0.975, 0.998]$ | `findings/phase_a/values/diagnostics.json` |
| `phase_a.ccd.stationary_yhat` | Stationary Point Predicted RMSE | $\hat{y}_0 = \beta_0 + \frac{1}{2}\mathbf{x}_0^T \mathbf{b}$ | $[0.400, 0.450]$ | `findings/phase_a/values/diagnostics.json` |
| `phase_a.ccd.stationary_dist` | Stationary Point Distance from Center | $\|\mathbf{x}_0\|_2$ in coded units | $> 1.0$ (Outside experimental cube) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.desirability.cube_opt_rmse` | In-Cube Constrained Optimum RMSE | $\min_{\mathbf{x} \in [-1, 1]^4} \hat{y}(\mathbf{x})$ | $[0.435, 0.465]$ | `findings/phase_a/values/macros.tex` |
| `phase_a.desirability.opt_latency` | In-Cube Optimum Latency ($\mu\text{s}$) | $\hat{y}_{\text{lat}}(\mathbf{x}^*_{\text{des}})$ | $[90.0, 160.0]\,\mu\text{s}$ | `findings/phase_a/values/macros.tex` |
| `phase_a.benchmark.hv_doe` | 2D Pareto Hypervolume (DOE-RSM) | Dominated area to ref $[0.60, 250\,\mu\text{s}]$ | $[15.0, 18.0]$ | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.benchmark.hv_motpe` | 2D Pareto Hypervolume (MO-TPE) | Dominated area to ref $[0.60, 250\,\mu\text{s}]$ | $[13.5, 16.0]$ | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.benchmark.hv_random` | 2D Pareto Hypervolume (Random) | Dominated area to ref $[0.60, 250\,\mu\text{s}]$ | $[5.0, 9.0]$ | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.benchmark.doe_over_motpe_pct` | Relative Hypervolume Advantage (%) | $\frac{HV_{\text{DOE}} - HV_{\text{MOTPE}}}{HV_{\text{MOTPE}}} \times 100\%$ | $[+5.0\%, +15.0\%]$ | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.holdout.pi_coverage` | Holdout 95% PI Empirical Coverage | $\frac{1}{M}\sum \mathbb{I}(y_m \in \text{PI}_{0.95})$ | $[0.90, 0.98]$ (Nominal 0.95) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.holdout.boundary_bias` | Boundary Optimism Bias | $\frac{1}{M_B}\sum_{m \in B} (y_m - \hat{y}_m)$ | $[-0.030, +0.010]$ | `findings/phase_a/values/diagnostics.json` |

---

## 3. Phase B: Sequential Adaptive CCD & Explainable AI

### 3.1 Study Invariants & Experimental Accounting (Phase B)

| Parameter | Planned Contract Value | Verification Constraint |
|---|---|---|
| **Target Dataset** | Full IMDb Sentiment Dataset (`data/imdb.csv`) | SHA-256 verified against `data/imdb.json` manifest |
| **Development Pool** | 25,000 documents (Stratified 5-fold cross-validation) | 20,000 train / 5,000 validation per fold |
| **Protected Test Set** | 25,000 documents (Official test partition) | Strictly isolated behind `FinalTestEvaluator` gate |
| **Nuisance Blocks** | $B = 5$ stratified folds | Every configuration evaluated across all 5 blocks |
| **Search Replicates** | $R = 3$ independent search runs with randomized seeds | Seeds: 42, 43, 44 |
| **Methods Evaluated** | Static Face-Centered CCD, Sequential Adaptive CCD, MO-TPE | Fair budget: 140 fits per method per replicate |
| **Fit Budget Ledger** | 3 methods $\times$ 3 replicates $\times$ 140 fits = 1,260 search fits | Strict equality: no method receives extra fits |
| **Confirmation Budget** | 12 coordinates $\times$ 5 blocks $\times$ 3 replicates = 180 fits/method | Reserved interior and boundary coordinates |
| **Primary Responses** | $Y_1$: Validation Loss ($1 - \text{Macro F1}$), $Y_2$: Latency ($\mu\text{s}/\text{doc}$) | Single-document warmed median latency on pinned CPU |

### 3.2 Parameters of Interest & Expected Value Bounds (Phase B)

| Metric ID | Semantic Target | Derivation / Formula | Pre-Experiment Expected Bound | Post-Run Fill Source |
|---|---|---|---|---|
| `phase_b.calibration.tree_count` | Selected Calibrated Tree Count | Capacity grid $[100, 300, 600]$ on dev pool | $300$ or $600$ trees | `findings/phase_b/values/metrics.json` |
| `phase_b.calibration.latency_sla_us` | Latency SLA Threshold | Median latency across calibration runs | $[100.0, 250.0]\,\mu\text{s}$ | `findings/phase_b/values/metrics.json` |
| `phase_b.static_doe.val_macro_f1.mean` | Validation Macro F1 for Static CCD | $1.0 - \frac{1}{R} \sum \min_{\mathbf{x}} \bar{L}_r(\mathbf{x})$ | $[0.840, 0.920]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.sequential_doe.val_macro_f1.mean`| Validation Macro F1 for Sequential CCD | $1.0 - \frac{1}{R} \sum \min_{\mathbf{x}} \bar{L}_r(\mathbf{x})$ | $[0.850, 0.925]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.mo_tpe.val_macro_f1.mean` | Validation Macro F1 for MO-TPE | $1.0 - \frac{1}{R} \sum \min_{\mathbf{x}} \bar{L}_r(\mathbf{x})$ | $[0.850, 0.925]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.final_test.macro_f1` | Protected Holdout Test Macro F1 | Reconstructed from true test labels | $[0.840, 0.915]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.sequential_doe.latency_us.mean` | Selected Model Latency (Sequential) | Median over 500 timed single-document calls | $[110.0, 240.0]\,\mu\text{s}$ | `findings/phase_b/values/metrics.json` |
| `phase_b.sequential_doe.feasibility_rate`| Search Feasibility Fraction | $\frac{1}{N} \sum \mathbb{I}(\text{latency}_i \le \text{Threshold})$ | $[0.50, 0.95]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.mo_tpe.feasibility_rate` | Search Feasibility Fraction (TPE) | $\frac{1}{N} \sum \mathbb{I}(\text{latency}_i \le \text{Threshold})$ | $[0.35, 0.85]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.sequential_doe.hypervolume` | Mean 2D Hypervolume (Sequential) | Dominated area to ref $[1.0, 2\times \text{lat}_{\max}]$ | $[28.0, 48.0]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.mo_tpe.hypervolume` | Mean 2D Hypervolume (MO-TPE) | Dominated area to ref $[1.0, 2\times \text{lat}_{\max}]$ | $[28.0, 48.0]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.ols.val_loss.rmse` | OLS Confirmation RMSE | $\sqrt{\frac{1}{M}\sum (y_m - \hat{y}_m)^2}$ on reserved coords | $[0.010, 0.060]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.ols.val_loss.pi_coverage` | OLS Prediction Interval Coverage | $\frac{1}{M}\sum \mathbb{I}(\hat{y}_{\text{low}} \le y \le \hat{y}_{\text{high}})$ | $[0.85, 0.98]$ (Nominal 0.95) | `findings/phase_b/values/metrics.json` |
| `phase_b.gp.val_loss.rmse` | Gaussian Process Confirmation RMSE | Forecast vs observed block-mean loss | $[0.008, 0.050]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.anova_shap.spearman_rho` | Factor Ranking Agreement | Spearman rank correlation of main effects | $[0.600, 1.000]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.anova_shap.kendall_tau` | Factor Ranking Concordance | Kendall rank correlation coefficient | $[0.500, 1.000]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.anova_shap.interaction_corr` | Pairwise Interaction Correlation | Pearson $r$ between ANOVA SS and $\Phi_{ij}$ | $[0.550, 0.980]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.word_rsm.fidelity_rmse` | Document-Level Word RSM RMSE | Confirmation RMSE on 64 word perturbations | $[0.005, 0.080]$ | `findings/phase_b/values/metrics.json` |
| `phase_b.word_rsm.r2` | Document-Level Word RSM $R^2$ | Coefficient of determination $R^2$ | $[0.850, 0.999]$ | `findings/phase_b/values/metrics.json` |

---

## 4. How to Execute & Populate Findings End-to-End

When a researcher clones this repository, the expected values above serve as the verifiable acceptance criteria. Running the experiment will overwrite or populate the actual observed values in `findings/`:

```powershell
# 1. Fast End-to-End Verification (Smoke test across both Phase A & Phase B)
python run_experiment.py --smoke

# 2. Run Phase A Full Replication Campaign (California Housing)
python run_experiment.py --phase a --mode full

# 3. Run Phase B Full Replication Campaign (Full IMDb Sentiment)
python run_experiment.py --phase b --stage all --dataset full_imdb --training-threads 4

# 4. Or Run Both Phases End-to-End in Sequence
python run_experiment.py --phase all
```

All empirical findings are automatically formatted and saved into:
- `findings/phase_a/tables/`, `findings/phase_a/figures/`, `findings/phase_a/values/`
- `findings/phase_b/tables/`, `findings/phase_b/figures/`, `findings/phase_b/values/`
