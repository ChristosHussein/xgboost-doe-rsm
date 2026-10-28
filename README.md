# Multi-Objective Hyperparameter Optimization for Gradient Boosted Trees under Stochastic Nuisance Blocking via Response Surface Methodology and Explainable AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Reproducibility: Fully Automated](https://img.shields.io/badge/Reproducibility-Fully%20Automated-success.svg)](run_experiment.py)

A unified research repository investigating the intersection of **Design of Experiments (DOE)**, **Response Surface Methodology (RSM)**, and **Orthogonal Nuisance Blocking** with modern **Explainable Artificial Intelligence (XAI)** to optimize and explain hyperparameters in tree-based gradient boosted models (**XGBoost**) across competing objectives (predictive accuracy vs inference latency).

---

## 1. One Unified Project Across Two Procedural Phases

This project is organized into two sequential procedural phases:

- **Phase A — Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking**  
  *Benchmark*: California Housing ($N=20,640$, 8 continuous features, 5 spatial nuisance blocks).  
  *Focus*: Factorial screening, first-order blocking, second-order Central Composite Design (CCD) curvature resolution, canonical stationary and ridge analysis ($\mathbf{x}_0 = -\frac{1}{2}\mathbf{B}^{-1}\mathbf{b}$), and multi-objective Derringer-Suich desirability optimization under pinned-CPU latency constraints.

- **Phase B — High-Dimensional Sequential / Adaptive Central Composite Design with Explainable AI & Warmed Latency Calibration**  
  *Benchmark*: IMDb Sentiment Analysis ($N=50,000$ documents, 10,000 sparse TF-IDF n-grams, 5 stratified folds).  
  *Focus*: High-dimensional text classification, automated tree capacity calibration, standardized warmed single-core latency protocols, sequential adaptive coordinate augmentation under equal evaluation budgets, game-theoretic SHAP vs ANOVA variance concordance, and document-level word response surfaces.

---

## 2. Pristine Repository Structure & Deliverables

```
.
├── EXPECTED_VALUES.md           <- Pre-experiment expected values & measurement contract (Phases A & B)
├── EXPERIMENT_EXPLANATION.md    <- Comprehensive scientific explanation & mathematical formulations
├── README.md                    <- This repository guide & execution manual
├── requirements.txt             <- Python runtime dependencies
├── run_experiment.py            <- Root single-command CLI orchestrator for all phases
│
├── findings/                    <- Unified directory for all produced deliverables
│   ├── phase_a/                 <- Automated findings for Phase A (California Housing)
│   │   ├── figures/             <- Publication figures (Pareto fronts, 3D response surfaces, diagnostics)
│   │   ├── tables/              <- Publication tables (ANOVA, benchmarks, LOF, confirmation) [CSV & TeX]
│   │   └── values/              <- benchmark_summary.json, diagnostics.json, macros.tex
│   └── phase_b/                 <- Automated findings for Phase B (IMDb Sentiment)
│       ├── figures/             <- Publication figures (Pareto fronts, ANOVA vs SHAP, interaction heatmaps)
│       ├── tables/              <- Publication tables (ANOVA loss/latency, benchmarks, confirmation) [CSV & TeX]
│       └── values/              <- metrics.json, macros.tex
│
├── phase_a_rsm_blocking/        <- Minimal Phase A execution code & canonical results
│   ├── pipeline.py              <- First-order screening, blocking & optimization pipeline
│   ├── analysis.py              <- Second-order ANOVA, lack-of-fit & canonical stationary analysis
│   ├── latency.py               <- Pinned single-row latency measurement protocol
│   ├── final_evaluation.py      <- Holdout test set evaluation gate
│   ├── scientific_stats.py      <- Non-parametric Wilcoxon and paired statistical tests
│   ├── config.yaml              <- Experimental bounds and factor coding rules
│   ├── report.pdf               <- Peer-reviewed publication manuscript (Phase A)
│   ├── technical_supplement.pdf <- Technical supplement detailing mathematical proofs
│   ├── scripts/                 <- Benchmark and replication scripts (run_benchmarks.py, reproduce_all.py)
│   └── results/                 <- Canonical empirical runs (17,077 fits, full_run_001, runs.csv)
│
├── phase_b_sequential_xai/      <- Modular Phase B execution package & pipeline
│   ├── run_phase_b.py           <- Master CLI runner for Phase B
│   ├── xgb_doe_xai/             <- Modular Python package (design, XAI, surrogates, benchmarks, profiler)
│   └── README.md                <- Detailed Phase B architecture and developer guide
│
└── archive/                     <- Strictly local backup of non-deliverable auxiliary docs (git-ignored)
```

---

## 3. Quick Start & Execution Guide

The entire repository can be replicated or verified through `run_experiment.py`:

### A. Fast End-to-End Smoke Verification (< 25s)
Runs an automated synthetic verification across both Phase A and Phase B to confirm that all pipeline stages, mathematical solvers, XAI routines, and findings generators function seamlessly without error:
```powershell
python run_experiment.py --smoke
```

### B. Run Phase A: Continuous Tabular RSM & Blocking
Executes the full replication campaign on California Housing, fitting the factorial designs, second-order CCD, and benchmark baselines:
```powershell
python run_experiment.py --phase a --mode full
```

### C. Run Phase B: High-Dimensional Sequential CCD & XAI
Executes data preparation, capacity calibration, sequential adaptive CCD, Optuna MO-TPE, exact surrogate SHAP attributions, document-level word studies, and findings generation on the full IMDb dataset:
```powershell
python run_experiment.py --phase b --stage all --dataset full_imdb --training-threads 4
```

### D. Run Both Phases End-to-End
```powershell
python run_experiment.py --phase all
```

---

## 4. Key Scientific Documents

- [EXPECTED_VALUES.md](EXPECTED_VALUES.md): Full pre-experiment measurement registry detailing invariants, metric IDs, mathematical formulas, and expected bounds for both Phase A and Phase B.
- [EXPERIMENT_EXPLANATION.md](EXPERIMENT_EXPLANATION.md): Deep-dive document covering theoretical derivations, canonical analysis, ANOVA vs SHAP reconciliation, and experimental accounting.
