# Phase B: High-Dimensional Sequential Adaptive CCD with Explainable AI & Warmed Latency Calibration
## Evaluated on the IMDb Sentiment Analysis NLP Benchmark

---

## 1. Procedural Overview & Scientific Scope

**Phase B** extends our Response Surface Methodology and blocking framework into high-dimensional, sparse machine learning classification:
> **Core Objective**: Investigating how **Sequential Face-Centered Central Composite Design (CCD)**, **Pre-Experiment Capacity Calibration**, and **Explainable AI (ANOVA vs SHAP)** optimize and interpret tree-based gradient boosted models (**XGBoost**) on sparse text data under multi-objective constraints (predictive loss $1 - \text{Macro F1}$ vs inference latency $\mu\text{s}/\text{doc}$).

### Benchmark Dataset & Partitioning Protocol
Evaluated on the full IMDb Sentiment Analysis NLP benchmark ($N = 50,000$ documents, 10,000 sparse TF-IDF unigram and bigram features):
- **Development Pool**: 25,000 documents, partitioned into $B = 5$ stratified cross-validation nuisance blocks (20,000 train / 5,000 validation per fold). All screening, model calibration, and sequential infill are conducted exclusively on this pool.
- **Protected Holdout Set**: 25,000 documents (the official test partition), strictly isolated behind the `FinalTestEvaluator` gate and evaluated only once on frozen final model selections to prevent data leakage and benchmark over-optimism.

---

## 2. Experimental Methodology & Mathematical Formulation

### 2.1 Hyperparameter Factors ($k=4$)
Four hyperparameters governing tree capacity, variance, and inference time are systematically coded into $[-1, +1]$:

| Factor | Parameter Name | Natural Range | Center ($0$) | Coding Transformation |
|---|---|---|---|---|
| $x_1$ | Learning Rate ($\eta$, `learning_rate`) | $[0.05, 0.30]$ | $0.175$ | Linear: $x_1 = (\eta - 0.175) / 0.125$ |
| $x_2$ | Maximum Depth (`max_depth`) | $[3, 9]$ | $6$ | Linear: $x_2 = (\text{depth} - 6) / 3$ (Integer-valued) |
| $x_3$ | Row Subsampling (`subsample`) | $[0.50, 1.00]$ | $0.75$ | Linear: $x_3 = (\text{subsample} - 0.75) / 0.25$ |
| $x_4$ | Leaf Regularization ($\lambda$, `reg_lambda`) | $[0.10, 10.00]$ | $1.00$ | $\log_{10}$ logarithmic scale |

### 2.2 Multi-Objective Targets & Latency Protocol
1. **Predictive Loss**: Validation Loss defined as $Y_1 = 1 - \text{Macro F1}$ (with cross-entropy log-loss as secondary diagnostic).
2. **Inference Latency**: Single-document prediction latency ($Y_2$ in $\mu\text{s}/\text{doc}$) measured on pinned CPU Core 0, enforcing 100 warmup calls followed by 500 timed single-document evaluations, eliminating multi-threading jitter and cache noise.

### 2.3 Procedural Stages
1. **Tree Capacity Calibration (`--stage calibrate`)**:
   - Scans a development grid ($[100, 300, 600]$ trees across $\eta \in [0.05, 0.30]$) to establish the minimum tree count required to resolve underfitting at low learning rates prior to optimization.
2. **Sparse CSR Canonicalization (`features.py`)**:
   - Guarantees canonical format and sorted column indices (`canonical_features()` via `.sum_duplicates()` and `.sort_indices()`) before passing sparse text matrices to XGBoost, preventing CUDA histogram kernel misindexing.
3. **Sequential Adaptive CCD vs Static CCD vs MO-TPE (`study.py`)**:
   - Compares Static Face-Centered CCD, Sequential Adaptive CCD (with sequential variance-reducing coordinate infill), and Optuna Multi-Objective TPE under strict equal evaluation budgets ($140$ fits per replicate $\times 3$ search replicates = $1,260$ search fits).
4. **Independent Out-of-Sample Confirmation (`surrogates.py`)**:
   - Evaluates $540$ confirmation fits on 12 unmeasured coordinates stratified across the domain interior and boundary faces ($x_i = \pm 1$) to test second-order OLS and Gaussian Process surrogates and detect polynomial boundary optimism bias.
5. **Reconciling ANOVA with SHAP (`xai.py`)**:
   - Calculates exact coalition Shapley values and second-order interaction terms $\Phi_{ij}$ on surrogates, comparing them against ANOVA partial sums of squares via Spearman rank correlation ($\rho$) and Kendall's tau ($\tau$).
6. **Document-Level Word Response Surfaces (`word_study.py`)**:
   - Evaluates a local Face-Centered CCD over the top 4 TF-IDF word retention weights $\mathbf{w} \in [0, 1]^4$ for individual documents, fitting local quadratic surfaces ($R^2 > 0.90$) to model word-word interaction effects.

---

## 3. Directory Layout & Module Architecture

```
phase_b_sequential_xai/
├── run_phase_b.py           <- Master unified CLI orchestrator for Phase B execution
├── README.md                <- This package architecture and developer manual
├── data/                    <- Prepared benchmark datasets & manifests (imdb.csv, imdb.json)
│
└── xgb_doe_xai/             <- Modular Python package:
    ├── config.py            <- Experiment configurations, factor boundaries & coding transforms
    ├── data.py              <- IMDb dataset streaming, stratified block partitioning & test isolation
    ├── design.py            <- Classical & Sequential Face-Centered CCD coordinate generators
    ├── execution.py         <- Pinned-CPU multi-objective XGBoost trainer & CV evaluator
    ├── features.py          <- Canonical sparse CSR matrix formatting and index sorting
    ├── profiler.py          <- Warmed single-thread latency profiler with hardware CPU pinning
    ├── surrogates.py        <- OLS quadratic response surfaces & Matern Gaussian Process surrogates
    ├── xai.py               <- Exact coalition SHAP, pairwise interactions & ANOVA SS decomposition
    ├── local_explanations.py<- Document-level word feature masking and local perturbation engine
    ├── word_study.py        <- Document-level word retention response surface study runner
    ├── benchmarks.py        <- Fair-budget Optuna MO-TPE and random search baselines
    ├── study.py             <- End-to-end experiment orchestrator across all methods and replicates
    └── study_reporting.py   <- Automated production of publication tables, figures, and LaTeX macros
```

---

## 4. Execution & Replication Guide

### 4.1 Running via Root Master CLI (Recommended)
From the repository root:
```powershell
# 1. Fast synthetic smoke verification (< 25 seconds)
python run_experiment.py --phase b --stage smoke

# 2. Complete end-to-end execution on full IMDb with GPU acceleration (RTX 5080)
python run_experiment.py --phase b --stage all --dataset full_imdb --device cuda --training-threads 4

# 3. Fallback to CPU execution
python run_experiment.py --phase b --stage all --dataset full_imdb --device cpu --training-threads 4
```

### 4.2 Running Directly within `phase_b_sequential_xai`
```powershell
cd phase_b_sequential_xai

# A. Fast Synthetic Smoke Demonstration
python run_phase_b.py --stage smoke

# B. Download and Partition the IMDb Dataset (50,000 docs)
python run_phase_b.py --stage prepare --data-dir data

# C. Tree Capacity & Latency SLA Calibration (Development pool only)
python run_phase_b.py --stage calibrate --data-dir data --output-dir results/calibration --device cuda

# D. Run Equal-Budget Benchmark (Static CCD vs Sequential CCD vs MO-TPE)
python run_phase_b.py --stage run --dataset full_imdb --data-dir data --calibration-dir results/calibration --output-dir results/experiment_run --device cuda --training-threads 4

# E. Document-Level Word Response Surface Study
python run_phase_b.py --stage words --data-dir data --output-dir results/experiment_run

# F. Automatically Generate All Deliverables (Tables, Figures, Values)
python run_phase_b.py --stage generate-findings --output-dir results/experiment_run --findings-dir ../findings/phase_b

# G. Complete One-Click Full Pipeline
python run_phase_b.py --stage all --dataset full_imdb --data-dir data --findings-dir ../findings/phase_b --device cuda --training-threads 4
```

---

## 5. Key Empirical Deliverables & Acceptance Verification

### 5.1 Downstream Reporting Layer (`findings/phase_b/`)
All downstream figures, publication tables, and numerical macros are exported to the central reporting layer:
- **Figures** (`findings/phase_b/figures/`):
  - `pareto_front_comparison.png`: Empirical Pareto frontiers across Static CCD, Sequential CCD, and MO-TPE.
  - `anova_vs_shap_importance.png`: Main effect ranking comparison between ANOVA partial SS and Shapley values.
  - `interaction_heatmaps.png`: Comparison of ANOVA $MS_{ij}$ and pairwise SHAP interaction matrices $\Phi_{ij}$.
  - `curvature_profiles.png`: Second-order curvature profiles across all 4 hyperparameters.
  - `confirmation_errors_boundary.png`: Error distributions comparing interior vs boundary coordinates.
- **Tables** (`findings/phase_b/tables/`):
  - `anova_table_val_loss.tex` & `.csv`: Second-order ANOVA decomposition for validation loss.
  - `anova_table_latency_us.tex` & `.csv`: Second-order ANOVA decomposition for inference latency.
  - `benchmark_comparison.tex` & `.csv`: Multi-objective comparison (loss, latency, feasibility, hypervolume).
  - `confirmation_predictions.tex` & `.csv`: Out-of-sample forecast accuracy and prediction interval coverage.
  - `anova_vs_shap_agreement.tex` & `.csv`: Spearman $\rho$ and Kendall $\tau$ rank concordance metrics.
- **Values & LaTeX Macros** (`findings/phase_b/values/`):
  - `metrics.json`: Semantic key-value registry of all observed experimental numbers.
  - `macros.tex`: 39 semantic LaTeX macros auto-populated for manuscript compilation.

### 5.2 Summary of Verified Empirical Values
*Cross-referenced with root [`Values.md`](../Values.md)*:
- Calibrated tree capacity: **100 trees** (Resolves underfitting)
- Latency SLA threshold: **$151.925\,\mu\text{s}$**
- Total study model fits: **1,809 fits** ($1,260$ search + $540$ confirmation + $9$ final)
- Mean 2D Hypervolume: Sequential DOE $= \mathbf{99.18 \pm 0.37}$, Static CCD $= \mathbf{99.41 \pm 0.31}$, MO-TPE $= \mathbf{89.59 \pm 3.34}$
- Feasibility rate: Sequential DOE $= \mathbf{97.5\%}$, Static CCD $= \mathbf{96.0\%}$, MO-TPE $= \mathbf{94.0\%}$
- Protected Test Macro F1: Sequential DOE $= \mathbf{0.6220}$ across all replicates, Static CCD $= \mathbf{0.6220}$, MO-TPE $= \mathbf{0.3326 - 0.5332}$
- 180-point OLS confirmation: Val Loss RMSE $= \mathbf{0.0794}$, 95% PI coverage $= \mathbf{97.2\%}$, Boundary bias $= \mathbf{-0.00168}$
- ANOVA vs SHAP agreement: Spearman $\rho = \mathbf{0.8000}$, Kendall $\tau = \mathbf{0.6667}$, Interaction $r = \mathbf{0.9172}$. Top factor: $x_4$ (`reg_lambda`); top interaction: $[x_3, x_4]$ (`subsample` $\times$ `reg_lambda`) identified identically by both methods.
