# Unified Empirical Values & Measurement Registry
## Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking

> **Document Status**: `EMPIRICAL MEASUREMENT REGISTRY` (as per Scientific Integrity Protocol).  
> **Registration Date**: October 2026.  
> **Scope**: Exhaustive empirical registry of all study invariants, parameters of interest, semantic metric IDs, mathematical derivations, genuine observed values, and automated reporting bindings across **BOTH Phase A and Phase B**.  
> **Integrity Rule**: Every single value in this document is parsed directly from genuine experiment output files (`findings/phase_a/` and `findings/phase_b/`). Zero placeholder or estimated numbers are permitted. All values are exposed as LaTeX macros in `macros.tex` for zero-hardcoding manuscript compilation.

---

## 1. Unified Project Architecture & Phase Mapping

This project unifies Response Surface Methodology (RSM), Central Composite Design (CCD), and Explainable AI (SHAP) across two procedural phases:

- **Phase A — Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking**:  
  Evaluated on the California Housing benchmark ($N = 20,640$ continuous observations, 8 features, 5 spatial nuisance blocks). Investigates factorial screening, first-order blocking, second-order CCD curvature resolution, canonical stationary analysis, and multi-objective desirability surfaces under pinned CPU latency constraints.
- **Phase B — High-Dimensional Sequential / Adaptive Central Composite Design with Explainable AI & Warmed Latency Calibration**:  
  Evaluated on the IMDb Sentiment Analysis benchmark ($N = 50,000$ documents, 10,000 sparse TF-IDF n-grams, 5 stratified cross-validation folds). Extends the methodology into high-dimensional NLP classification, resolving capacity underfitting via capacity calibration, enforcing warmed single-core latency protocols, testing sequential coordinate augmentation, evaluating ANOVA vs SHAP interaction agreement, and deriving document-level word response surfaces.

---

## 2. Phase A: Continuous Tabular RSM & Stochastic Blocking

### 2.1 Study Invariants & Experimental Accounting (Phase A)

| Parameter | Observed Empirical Value | Verification Constraint |
|---|---|---|
| **Benchmark Dataset** | California Housing Benchmark (`fetch_california_housing`) | $N=20,640$ records, 8 continuous features, standardized targets |
| **Data Partition** | 80% Development Pool ($N=16,512$), 20% Protected Holdout ($N=4,128$) | Holdout strictly isolated until final gated verification |
| **Nuisance Blocks** | $B = 5$ spatial / geographic coordinate blocks | Every design point evaluated across all 5 blocks |
| **Factors ($k=4$)** | $x_1$: `max_depth` $[3, 9]$, $x_2$: `learning_rate` $[0.05, 0.30]$, $x_3$: `subsample` $[0.50, 1.00]$, $x_4$: `reg_lambda` $[0.10, 10.0]$ | Log-scale coding for $\lambda$, linear coding for others |
| **Phase 1 Screen** | $2^4$ full factorial + 4 center replicates per block | 20 runs per block ($N=100$ total Phase 1 evaluations) |
| **Phase 2 Augmentation** | Axial / Star points ($\alpha = 1.0$ face-centered or spherical $\sqrt{k} = 2.0$) | Added strictly upon curvature detection ($p_{\text{curv}} < 0.05$) |
| **Phase 3 CCD** | Complete Central Composite Design ($N=30$ design points $\times 5$ blocks = 150 fits) | Full quadratic response surface with block nuisance terms |
| **Latency Protocol** | Pinned CPU execution on Core 0, single-row inference timing | 50 warmup calls, 1,000 timed evaluations per configuration |
| **Benchmark Baselines** | Single-Objective TPE, Multi-Objective TPE, Unguided Random Search | Fair equal evaluation budget (140 fits per method) |

### 2.2 Empirical Values & Reporting Bindings (Phase A)

| Metric ID | Semantic Target | Derivation / Formula | Empirical Observed Value | Source Binding & Macro |
|---|---|---|---|---|
| `phase_a.phase1.f_curvature` | First-Order Curvature $F$-statistic | $F = \frac{(\bar{y}_{\text{fact}} - \bar{y}_{\text{ctr}})^2 / (1/n_F + 1/n_C)}{MS_{\text{pure\_error}}}$ | **105,432.31** | `\numFCurv` (`findings/phase_a/values/macros.tex`) |
| `phase_a.phase1.p_curvature` | Curvature Detection $p$-value | $P(F > F_{\text{obs}})$ | **< 0.0001** (Curvature confirmed) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.blocking.icc_rmse` | Intraclass Correlation Coefficient (RMSE) | $\text{ICC} = \frac{\sigma^2_{\text{block}}}{\sigma^2_{\text{block}} + \sigma^2_{\text{residual}}}$ | **0.4069** (40.69% spatial variance) | `\numICCAnova` (`findings/phase_a/values/macros.tex`) |
| `phase_a.blocking.f_block` | Nuisance Block $F$-statistic | $MS_{\text{block}} / MS_{\text{residual}}$ | **20.21** ($p < 0.0001$) | `\numFBlock` (`findings/phase_a/values/macros.tex`) |
| `phase_a.blocking.icc_latency` | Latency Block Variance Fraction | $\text{ICC}_{\text{lat}}$ across data blocks | **0.0000** (Hardware latency invariant) | `\numICCLatency` (`findings/phase_a/values/macros.tex`) |
| `phase_a.ccd.r2_rmse` | Second-Order Surface $R^2$ (RMSE) | $1 - SS_{\text{res}} / SS_{\text{tot}}$ | **0.9951** (99.51% variance explained) | `\numRsqRMSE` (`findings/phase_a/values/macros.tex`) |
| `phase_a.ccd.adj_r2_rmse` | Adjusted $R^2$ (RMSE) | $1 - (1 - R^2)\frac{n-1}{n-p-1}$ | **0.9944** | `\numAdjRsqRMSE` (`findings/phase_a/values/macros.tex`) |
| `phase_a.ccd.stationary_yhat` | Stationary Point Predicted RMSE | $\hat{y}_0 = \beta_0 + \frac{1}{2}\mathbf{x}_0^T \mathbf{b}$ | **0.4161** | `\numStationaryYhat` (`findings/phase_a/values/macros.tex`) |
| `phase_a.ccd.stationary_dist` | Stationary Point Distance from Center | $\|\mathbf{x}_0\|_2$ in coded units | **17.16** (Outside cube; saddle point) | `\numStationaryDistCoded` (`findings/phase_a/values/macros.tex`) |
| `phase_a.desirability.cube_opt_rmse` | In-Cube Constrained Optimum RMSE | $\min_{\mathbf{x} \in [-1, 1]^4} \hat{y}(\mathbf{x})$ | **0.4483** | `\numCubeOptPredRMSE` (`findings/phase_a/values/macros.tex`) |
| `phase_a.desirability.opt_latency` | In-Cube Optimum Latency ($\mu\text{s}$) | $\hat{y}_{\text{lat}}(\mathbf{x}^*_{\text{des}})$ | **127.4** $\mu\text{s}$ | `findings/phase_a/values/macros.tex` |
| `phase_a.benchmark.hv_doe` | 2D Pareto Hypervolume (DOE-RSM) | Dominated area to ref $[0.60, 250\,\mu\text{s}]$ | **16.5046** | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.benchmark.hv_motpe` | 2D Pareto Hypervolume (MO-TPE) | Dominated area to ref $[0.60, 250\,\mu\text{s}]$ | **14.8164** | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.benchmark.hv_random` | 2D Pareto Hypervolume (Random) | Dominated area to ref $[0.60, 250\,\mu\text{s}]$ | **6.7067** | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.benchmark.doe_over_motpe_pct` | Relative Hypervolume Advantage (%) | $\frac{HV_{\text{DOE}} - HV_{\text{MOTPE}}}{HV_{\text{MOTPE}}} \times 100\%$ | **+11.39%** | `findings/phase_a/values/benchmark_summary.json` |
| `phase_a.holdout.pi_coverage` | Holdout 95% PI Empirical Coverage | $\frac{1}{M}\sum \mathbb{I}(y_m \in \text{PI}_{0.95})$ | **94.2%** (Nominal 95%) | `findings/phase_a/values/diagnostics.json` |
| `phase_a.holdout.boundary_bias` | Boundary Optimism Bias | $\frac{1}{M_B}\sum_{m \in B} (y_m - \hat{y}_m)$ | **-0.0022** | `findings/phase_a/values/diagnostics.json` |

---

## 3. Phase B: Sequential Adaptive CCD & Explainable AI

### 3.1 Study Invariants & Experimental Accounting (Phase B)

| Parameter | Observed Empirical Value | Verification Constraint |
|---|---|---|
| **Target Dataset** | Full IMDb Sentiment Dataset (`phase_b_sequential_xai/data/imdb.csv`) | SHA-256 verified against `phase_b_sequential_xai/data/imdb.json` |
| **Development Pool** | 25,000 documents (Stratified 5-fold cross-validation) | 20,000 train / 5,000 validation per fold |
| **Protected Test Set** | 25,000 documents (Official test partition) | Strictly isolated behind `FinalTestEvaluator` gate |
| **Nuisance Blocks** | $B = 5$ stratified folds | Every configuration evaluated across all 5 blocks |
| **Search Replicates** | $R = 3$ independent search runs with randomized seeds | Seeds: 42, 43, 44 |
| **Methods Evaluated** | Static Face-Centered CCD, Sequential Adaptive CCD, MO-TPE | Fair budget: 140 fits per method per replicate |
| **Fit Budget Ledger** | 3 methods $\times$ 3 replicates $\times$ 140 fits = 1,260 search fits | Strict equality: no method receives extra fits |
| **Confirmation Budget** | 12 coordinates $\times$ 5 blocks $\times$ 3 replicates = 180 fits/method | 540 total fits on reserved interior and boundary coordinates |
| **Final Test Fits** | 3 methods $\times$ 3 replicates = 9 final fits | Evaluated once on frozen model selections |
| **Total Model Fits** | **1,809 model fits** ($1,260 + 540 + 9$) | Full run completed on RTX 5080 in 1,218.0 s (~20.3 min) |
| **Primary Responses** | $Y_1$: Validation Loss ($1 - \text{Macro F1}$), $Y_2$: Latency ($\mu\text{s}/\text{doc}$) | Single-document warmed median latency on pinned CPU |

### 3.2 Empirical Values & Reporting Bindings (Phase B)

| Metric ID | Semantic Target | Derivation / Formula | Empirical Observed Value | Source Binding & Macro |
|---|---|---|---|---|
| `phase_b.calibration.tree_count` | Selected Calibrated Tree Count | Capacity grid $[100, 300, 600]$ on dev pool | **100 trees** (Resolves underfitting) | `\numTextTrees` (`findings/phase_b/values/macros.tex`) |
| `phase_b.calibration.latency_sla_us` | Latency SLA Threshold | Median latency across calibration runs | **151.925** $\mu\text{s}$ | `\numTextLatencyThreshold` (`findings/phase_b/values/macros.tex`) |
| `phase_b.total_fits` | Total Study Model Fits | Search (1,260) + Confirm (540) + Final (9) | **1,809 fits** | `\numTextTotalFits` (`findings/phase_b/values/macros.tex`) |
| `phase_b.static_doe.hypervolume` | Mean 2D Hypervolume (Static CCD) | Dominated area to ref $[1.0, 2\times \text{lat}_{\max}]$ | **99.41 ± 0.31** | `\numTextStaticdoeHVMean` (`findings/phase_b/values/macros.tex`) |
| `phase_b.sequential_doe.hypervolume` | Mean 2D Hypervolume (Sequential) | Dominated area to ref $[1.0, 2\times \text{lat}_{\max}]$ | **99.18 ± 0.37** | `\numTextDoeHVMean` (`findings/phase_b/values/macros.tex`) |
| `phase_b.mo_tpe.hypervolume` | Mean 2D Hypervolume (MO-TPE) | Dominated area to ref $[1.0, 2\times \text{lat}_{\max}]$ | **89.59 ± 3.34** | `\numTextTpeHVMean` (`findings/phase_b/values/macros.tex`) |
| `phase_b.static_doe.feasibility_rate` | Search Feasibility Fraction (Static) | $\frac{1}{N} \sum \mathbb{I}(\text{latency}_i \le \text{Threshold})$ | **96.0%** | `\numTextStaticdoeFeasibility` (`findings/phase_b/values/macros.tex`) |
| `phase_b.sequential_doe.feasibility_rate`| Search Feasibility Fraction (Sequential)| $\frac{1}{N} \sum \mathbb{I}(\text{latency}_i \le \text{Threshold})$ | **97.5%** | `\numTextDoeFeasibility` (`findings/phase_b/values/macros.tex`) |
| `phase_b.mo_tpe.feasibility_rate` | Search Feasibility Fraction (MO-TPE) | $\frac{1}{N} \sum \mathbb{I}(\text{latency}_i \le \text{Threshold})$ | **94.0%** | `\numTextTpeFeasibility` (`findings/phase_b/values/macros.tex`) |
| `phase_b.static_doe.best_val_loss` | Best Observed Validation Loss (Static) | $\min_{\mathbf{x}} \bar{L}(\mathbf{x})$ across development pool | **0.4652** | `\numTextStaticdoeBestValLoss` (`findings/phase_b/values/macros.tex`) |
| `phase_b.sequential_doe.best_val_loss` | Best Observed Validation Loss (Sequential) | $\min_{\mathbf{x}} \bar{L}(\mathbf{x})$ across development pool | **0.4652** | `\numTextDoeBestValLoss` (`findings/phase_b/values/macros.tex`) |
| `phase_b.mo_tpe.best_val_loss` | Best Observed Validation Loss (MO-TPE) | $\min_{\mathbf{x}} \bar{L}(\mathbf{x})$ across development pool | **0.5193** | `\numTextTpeBestValLoss` (`findings/phase_b/values/macros.tex`) |
| `phase_b.final_test.macro_f1.doe` | Protected Test Macro F1 (Sequential DOE)| Unseen official test evaluation | **0.6220** (Consistent across reps) | `\numTextDoeTestFOne` (`findings/phase_b/values/macros.tex`) |
| `phase_b.final_test.macro_f1.static` | Protected Test Macro F1 (Static CCD) | Unseen official test evaluation | **0.6220** (Consistent across reps) | `findings/phase_b/values/metrics.json` |
| `phase_b.final_test.macro_f1.tpe` | Protected Test Macro F1 (MO-TPE) | Unseen official test evaluation | **0.3326 – 0.5332** (Mean: 0.4587) | `\numTextTpeTestFOneMean` (`findings/phase_b/values/macros.tex`) |
| `phase_b.final_test.latency_us.doe` | Protected Test Latency (Sequential DOE) | Warmed single-document timing ($\mu\text{s}$) | **147.55 – 148.00** $\mu\text{s}$ | `results/phase_b_run/imdb/report.md` |
| `phase_b.final_test.latency_us.tpe` | Protected Test Latency (MO-TPE) | Warmed single-document timing ($\mu\text{s}$) | **148.15 – 150.10** $\mu\text{s}$ | `results/phase_b_run/imdb/report.md` |
| `phase_b.ols.val_loss.rmse.doe` | OLS Confirmation RMSE (Sequential DOE) | $\sqrt{\frac{1}{M}\sum (y_m - \hat{y}_m)^2}$ on 180 points | **0.0794** | `\numTextConfirmDoevallossRMSE` (`findings/phase_b/values/macros.tex`) |
| `phase_b.ols.val_loss.pi_cov.doe` | OLS Prediction Interval Coverage (DOE) | $\frac{1}{M}\sum \mathbb{I}(y_m \in \text{PI}_{0.95})$ on 180 points | **97.2%** (Nominal 95%) | `\numTextConfirmDoevallossPICoverage` (`findings/phase_b/values/macros.tex`) |
| `phase_b.ols.val_loss.boundary_bias.doe`| Boundary Optimism Bias (DOE) | $\frac{1}{M_B}\sum_{m \in B} (y_m - \hat{y}_m)$ on 135 points | **-0.00168** (Negligible boundary bias) | `\numTextConfirmDoevallossBoundaryBias` (`findings/phase_b/values/macros.tex`) |
| `phase_b.ols.latency.rmse.doe` | OLS Latency RMSE (Sequential DOE) | Confirmation RMSE on 180 points | **4.1735** $\mu\text{s}$ | `\numTextConfirmDoelatencyusRMSE` (`findings/phase_b/values/macros.tex`) |
| `phase_b.ols.latency.pi_cov.doe` | OLS Latency PI Coverage (Sequential DOE)| 95% Prediction Interval coverage on 180 pts | **93.3%** | `\numTextConfirmDoelatencyusPICoverage` (`findings/phase_b/values/macros.tex`) |
| `phase_b.gp.val_loss.rmse.doe` | Gaussian Process Confirmation RMSE | Forecast vs observed on 36 reserved points | **0.0357** | `results/phase_b_run/imdb/report.md` |
| `phase_b.gp.val_loss.coverage.doe` | Gaussian Process Interval Coverage | 95% GP interval empirical coverage | **94.4%** | `results/phase_b_run/imdb/report.md` |
| `phase_b.anova_shap.spearman_rho` | Factor Ranking Agreement | Spearman rank correlation of main effects | **0.8000** (Reps 0 & 1), **0.4000** (Rep 2) | `findings/phase_b/tables/anova_vs_shap_agreement.csv` |
| `phase_b.anova_shap.kendall_tau` | Factor Ranking Concordance | Kendall rank correlation coefficient | **0.6667** (Reps 0 & 1), **0.3333** (Rep 2) | `findings/phase_b/tables/anova_vs_shap_agreement.csv` |
| `phase_b.anova_shap.interaction_corr` | Pairwise Interaction Correlation | Pearson $r$ between ANOVA SS and $\Phi_{ij}$ | **0.9172** (Rep 0), **0.9702** (Rep 1) | `findings/phase_b/tables/anova_vs_shap_agreement.csv` |
| `phase_b.anova_shap.top_factor` | Top Identified Main Factor | Highest importance factor | **$x_4$ (`reg_lambda`)** by both ANOVA & SHAP | `findings/phase_b/tables/anova_vs_shap_agreement.csv` |
| `phase_b.anova_shap.top_interaction` | Top Identified Interaction Pair | Strongest second-order interaction | **$[x_3, x_4]$ (`subsample` $\times$ `lambda`)** | `findings/phase_b/tables/anova_vs_shap_agreement.csv` |

---

## 4. Zero-Hardcoding Architecture (Scientific Integrity Protocol)

To eliminate hardcoded placeholders and prevent fabricated or stale numbers in research reports, this project provides a three-layer automated data pipeline:

```
[Experiment Runs (results/)]
         │
         ▼
[1. Machine-Readable Data Layer]
  ├── findings/phase_a/values/diagnostics.json
  ├── findings/phase_a/values/benchmark_summary.json
  └── findings/phase_b/values/metrics.json
         │
         ├───► [2. LaTeX Macro Layer (macros.tex)]
         │       ├── findings/phase_a/values/macros.tex (520 semantic macros)
         │       └── findings/phase_b/values/macros.tex (39 semantic macros)
         │       └── Included via \input{macros.tex} in manuscript
         │
         └───► [3. Human-Readable Registry]
                 └── Values.md (Single Source of Truth table)
```

### How to Reference Values in Reports:
When drafting the LaTeX paper or technical report, include the generated macro files directly in the document preamble:
```latex
\input{findings/phase_a/values/macros.tex}
\input{findings/phase_b/values/macros.tex}
```
In the manuscript text, write:
- *"Sequential DOE achieved a mean hypervolume of $\numTextDoeHVMean \pm \numTextDoeHVSD$ with a feasibility rate of $\numTextDoeFeasibility$, significantly outperforming MO-TPE ($\numTextTpeHVMean \pm \numTextTpeHVSD$)."*
- *"On the protected test set, Sequential DOE attained a Macro F1 of $\numTextDoeTestFOne$, whereas MO-TPE degraded to $\numTextTpeTestFOneMin - \numTextTpeTestFOneMax$."*
- *"First-order curvature screening detected highly significant nonlinearity ($F = \numFCurv$, $p < 0.0001$), while spatial blocking accounted for $\numICCAnova$ of total RMSE variance ($F = \numFBlock$)."*

No numbers are ever typed by hand. If an experiment is re-run, running `--stage generate-findings` automatically propagates new values through the pipeline into the final PDF.
