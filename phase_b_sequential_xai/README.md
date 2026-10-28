# Phase B: High-Dimensional Sequential Adaptive CCD with Explainable AI & Warmed Latency Calibration
## Evaluated on the IMDb Sentiment Analysis NLP Benchmark

---

## 1. Procedural Overview & Scientific Scope

**Phase B** expands our Response Surface Methodology framework into high-dimensional, sparse machine learning classification:
> Investigating how **Sequential Face-Centered Central Composite Design (CCD)**, **Tree Capacity Calibration**, and **Explainable AI (ANOVA vs SHAP)** optimize and interpret tree-based gradient boosted models (**XGBoost**) on sparse text data under multi-objective constraints (predictive loss $1 - \text{Macro F1}$ vs inference latency $\mu\text{s}/\text{doc}$).

The empirical campaign evaluates the full IMDb Sentiment dataset ($N = 50,000$ documents, 10,000 TF-IDF unigram and bigram features), divided into:
- **Development Pool**: 25,000 documents, partitioned into $B = 5$ stratified cross-validation nuisance blocks (20,000 train / 5,000 validation per fold).
- **Protected Test Partition**: 25,000 documents, strictly isolated behind the `FinalTestEvaluator` gate until all hyperparameter selections are frozen.

---

## 2. Resolving Methodological Challenges (Supervisor Directives)

Phase B directly operationalizes the directives established in `Remarks,.txt` and `Research approach.txt`:

1. **Tree Capacity Calibration (`--stage calibrate`)**:
   - Scans a pre-experiment development grid ($[100, 300, 600]$ trees across $\eta \in [0.05, 0.30]$) to select an optimal tree count that guarantees low learning rates do not underfit before search begins.
2. **Standardized Warmed Latency Protocol (`profiler.py`)**:
   - Replaces noisy, unwarmed timing with 100 warmup calls followed by 500 timed single-document evaluations on pinned CPU Core 0, eliminating multi-threading jitter and false feasibility claims.
3. **Protected Holdout Isolation Gate (`data.py`, `final_evaluation.py`)**:
   - Locks the 25,000-document test partition behind a cryptographic evaluation gate, preventing data leakage during screening and selection.
4. **Boundary Optimism Detection (`surrogates.py`)**:
   - Reserves 12 unmeasured validation coordinates stratified across the domain interior and boundary faces ($x_i = \pm 1$) to detect polynomial boundary optimism bias.
5. **Reconciling ANOVA with SHAP (`xai.py`)**:
   - Evaluates exact coalition Shapley values and pairwise interaction terms $\Phi_{ij}$ on surrogate models, comparing them against classical ANOVA partial sums of squares via Spearman rank correlation ($\rho$) and Kendall's tau ($\tau$).
6. **Sequential Design Augmentation (`design.py`)**:
   - Adaptively augments the initial fractional factorial core with infill coordinates chosen via predictive variance reduction and space-filling, matching the evaluation budget of Bayesian optimization (140 fits per replicate).
7. **Document-Level Word Response Surfaces (`word_study.py`)**:
   - Evaluates a local Face-Centered CCD over the top 4 TF-IDF word retention weights $\mathbf{w} \in [0, 1]^4$ for individual documents, fitting local quadratic surfaces ($R^2 > 0.90$) to explain word-word interactions.

---

## 3. Directory Layout & Package Architecture

```
phase_b_sequential_xai/
├── run_phase_b.py           <- Master unified CLI orchestrator for Phase B execution
├── README.md                <- This package architecture and developer manual
│
└── xgb_doe_xai/             <- Modular Python package:
    ├── config.py            <- Experiment configurations, factor boundaries & coding transforms
    ├── data.py              <- IMDb dataset streaming, stratified block partitioning & test isolation
    ├── design.py            <- Classical & Sequential Face-Centered CCD coordinate generators
    ├── execution.py         <- Pinned-CPU multi-objective XGBoost trainer & CV evaluator
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

## 4. Execution & Replication Commands

### 4.1 Running via Root Master CLI (Recommended)
From the repository root:
```powershell
# Fast synthetic smoke verification (< 15 seconds)
python run_experiment.py --phase b --stage smoke

# Complete end-to-end execution on full IMDb dataset
# Complete end-to-end execution on full IMDb with GPU acceleration (RTX 5080)
python run_experiment.py --phase b --stage all --dataset full_imdb --device cuda --training-threads 4

# Or fallback to CPU
python run_experiment.py --phase b --stage all --dataset full_imdb --device cpu --training-threads 4
```

### 4.2 Running Directly within `phase_b_sequential_xai`
```powershell
cd phase_b_sequential_xai

# A. Fast Synthetic Smoke Demonstration
python run_phase_b.py --stage smoke

# B. Download and Partition the IMDb Dataset (50,000 docs)
python run_phase_b.py --stage prepare --data-dir ../data

# C. Tree Capacity & Latency SLA Calibration (Development pool only)
python run_phase_b.py --stage calibrate --data-dir ../data --output-dir results/calibration

# D. Run Equal-Budget Benchmark (Static CCD vs Sequential CCD vs MO-TPE)
python run_phase_b.py --stage run --dataset full_imdb --data-dir ../data --calibration-dir results/calibration --output-dir results/experiment_run

# E. Document-Level Word Response Surface Study
python run_phase_b.py --stage words --data-dir ../data --output-dir results/experiment_run

# F. Automatically Generate All Deliverables (Tables, Figures, Values)
python run_phase_b.py --stage generate-findings --output-dir results/experiment_run --findings-dir ../findings/phase_b

# G. Complete One-Click Full Pipeline
python run_phase_b.py --stage all --dataset full_imdb --data-dir ../data --findings-dir ../findings/phase_b --training-threads 4
```

---

## 5. Output Findings & Publication Artifacts

All processed findings, tables, figures, and LaTeX macros for Phase B are automatically exported to `findings/phase_b/`:
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
- **Values** (`findings/phase_b/values/`):
  - `metrics.json`: Semantic key-value registry of all observed experimental numbers.
  - `macros.tex`: LaTeX macros auto-populated from `metrics.json` for manuscript compilation.
