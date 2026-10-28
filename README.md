# Multi-Objective Hyperparameter Optimization for Gradient Boosted Trees under Stochastic Nuisance Blocking via Response Surface Methodology and Explainable AI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Reproducibility: Fully Automated](https://img.shields.io/badge/Reproducibility-Fully%20Automated-success.svg)](run_experiment.py)
[![CUDA: Accelerated](https://img.shields.io/badge/CUDA-RTX%20Accelerated-green.svg)](phase_b_sequential_xai)

A unified research repository investigating the intersection of **Design of Experiments (DOE)**, **Response Surface Methodology (RSM)**, and **Orthogonal Nuisance Blocking** with modern **Explainable Artificial Intelligence (XAI)** to optimize and explain hyperparameters in tree-based gradient boosted models (**XGBoost**) across competing objectives (predictive accuracy vs inference latency).

---

## 1. Unified Project Architecture Across Two Procedural Phases

This project investigates hyperparameter response surfaces across two procedural phases:

- **Phase A — Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking**  
  *Benchmark*: California Housing ($N = 20,640$ observations, 8 continuous features, 5 spatial nuisance blocks).  
  *Focus*: Factorial screening, first-order blocking, second-order Central Composite Design (CCD) curvature resolution ($F_{\text{curv}} = 105,432.31$), canonical stationary and ridge analysis ($\mathbf{x}_0 = -\frac{1}{2}\mathbf{B}^{-1}\mathbf{b}$), and multi-objective Derringer-Suich desirability optimization under pinned-CPU latency constraints.  
  *Documentation*: Detailed guide in [`phase_a_rsm_blocking/README.md`](phase_a_rsm_blocking/README.md).

- **Phase B — High-Dimensional Sequential / Adaptive Central Composite Design with Explainable AI & Warmed Latency Calibration**  
  *Benchmark*: IMDb Sentiment Analysis ($N = 50,000$ documents, 10,000 sparse TF-IDF n-grams, 5 stratified cross-validation folds).  
  *Focus*: High-dimensional text classification, pre-experiment tree capacity calibration, standardized warmed single-core latency protocols, sequential adaptive coordinate augmentation under equal evaluation budgets (140 fits per replicate), canonical sparse CSR matrix formatting, game-theoretic SHAP vs ANOVA variance concordance ($\rho = 0.8000$, $r = 0.9172$), and document-level word response surfaces.  
  *Documentation*: Detailed guide in [`phase_b_sequential_xai/README.md`](phase_b_sequential_xai/README.md).

---

## 2. Pristine Repository Structure & Deliverables

```
.
├── Values.md                    <- Empirical measurement registry detailing all observed metrics & macros
├── EXPERIMENT_EXPLANATION.md    <- Comprehensive scientific explanation & mathematical formulations
├── README.md                    <- Master repository manual and execution guide
├── requirements.txt             <- Python runtime dependencies
├── run_experiment.py            <- Root single-command CLI orchestrator for all phases
│
├── findings/                    <- Downstream publication deliverables (Strictly derived assets, no raw runs)
│   ├── phase_a/                 <- Automated findings for Phase A (California Housing)
│   │   ├── figures/             <- Publication figures (Pareto fronts, 3D response surfaces, diagnostics)
│   │   ├── tables/              <- Publication tables (ANOVA, benchmarks, LOF, confirmation) [CSV & TeX]
│   │   └── values/              <- diagnostics.json, benchmark_summary.json, macros.tex (520 macros)
│   ├── phase_b/                 <- Automated findings for Phase B (IMDb Sentiment)
│   │   ├── figures/             <- Publication figures (Pareto fronts, ANOVA vs SHAP, interaction heatmaps)
│   │   ├── tables/              <- Publication tables (ANOVA loss/latency, benchmarks, confirmation) [CSV & TeX]
│   │   └── values/              <- metrics.json, macros.tex (39 macros)
│   └── final_report/            <- Unified manuscript, compilation files, and verification audit scripts
│       ├── main.tex / main.pdf  <- Peer-reviewed publication manuscript
│       └── *.tex / *.json / *.py<- Audit verification and section tables
│
├── phase_a_rsm_blocking/        <- Phase A package: Continuous Tabular RSM & Blocking
│   ├── pipeline.py              <- First-order screening, blocking & optimization pipeline
│   ├── analysis.py              <- Second-order ANOVA, lack-of-fit & canonical stationary analysis
│   ├── latency.py               <- Pinned single-row latency measurement protocol (Core 0)
│   ├── final_evaluation.py      <- Holdout test set evaluation gate
│   ├── scientific_stats.py      <- Non-parametric Wilcoxon and paired statistical tests
│   ├── config.yaml              <- Experimental bounds and factor coding rules
│   ├── report.pdf / .tex        <- Peer-reviewed publication manuscript (Phase A)
│   ├── scripts/                 <- Phase A benchmark and replication utilities
│   └── results/                 <- Canonical empirical runs (17,077 fits, full_run_001, runs.csv)
│
├── phase_b_sequential_xai/      <- Phase B package: Sequential Adaptive CCD & Explainable AI
│   ├── run_phase_b.py           <- Master CLI orchestrator for Phase B
│   ├── data/                    <- Prepared benchmark datasets & manifests (imdb.csv, imdb.json)
│   ├── xgb_doe_xai/             <- Modular Python package (design, XAI, surrogates, benchmarks, profiler)
│   └── README.md                <- Detailed Phase B architecture and developer guide
│
├── scripts/                     <- Standalone hardware verification & convenience helpers (Optional)
│   ├── run_corrected_campaign.py<- Helper script for automated calibration + run execution
│   └── verify_cuda_pipeline.py  <- Standalone GPU training verification test
│
├── tests/                       <- Automated CI/CD regression test suite (Optional for reproduction)
│   └── test_training_inputs.py  <- Pytest suite enforcing canonical CSR formatting & model health
│
└── archive/                     <- Strictly local backup of non-deliverable auxiliary docs (git-ignored)
```

---

## 3. Detailed Execution & Replication Manual

The entire research programme is orchestrated through [`run_experiment.py`](run_experiment.py).

### 3.1 Fast End-to-End Smoke Verification (< 25s)
Runs an automated synthetic verification across both Phase A and Phase B to verify that all mathematical solvers, design generators, XAI attribution engines, and findings pipelines execute cleanly:
```powershell
python run_experiment.py --smoke
```

---

### 3.2 Running Phase A: Continuous Tabular RSM & Blocking
Executes the full replication campaign on California Housing, fitting the factorial designs, second-order CCD, and benchmark baselines on pinned CPU Core 0:
```powershell
python run_experiment.py --phase a --mode full
```

---

### 3.3 Running Phase B: High-Dimensional Sequential CCD & XAI

Phase B evaluates 50,000 text documents ($1,809$ total model fits across 5 folds and 3 replicates).

#### Option A: GPU Acceleration via CUDA (Recommended for Speed)
Using NVIDIA CUDA cores achieves a **~12x speedup** (~20 minutes on an NVIDIA GeForce RTX 5080 vs ~50 hours on CPU):
```powershell
python run_experiment.py --phase b --stage all --dataset full_imdb --device cuda --training-threads 4
```
> **Requirements**: NVIDIA GPU with updated drivers and CUDA support. Input sparse CSR matrices are automatically validated and canonicalized via `canonical_features()` (`sort_indices()` and `sum_duplicates()`) to guarantee healthy GPU histogram bin accumulation.

#### Option B: CPU Fallback
If running on a system without a dedicated CUDA GPU:
```powershell
python run_experiment.py --phase b --stage all --dataset full_imdb --device cpu --training-threads 4
```

#### Individual Stage Execution (Phase B)
You can also run Phase B stage-by-stage using `run_experiment.py`:
```powershell
# 1. Dataset verification & fold preparation
python run_experiment.py --phase b --stage prepare

# 2. Pre-experiment tree capacity & latency calibration (dev pool only)
python run_experiment.py --phase b --stage calibrate --device cuda

# 3. Fair-budget optimizer campaign (Static CCD vs Sequential CCD vs MO-TPE)
python run_experiment.py --phase b --stage run --dataset full_imdb --device cuda --training-threads 4

# 4. Document-level word response surface study
python run_experiment.py --phase b --stage words

# 5. Automatically generate all deliverables (tables, figures, macros)
python run_experiment.py --phase b --stage generate-findings
```

---

### 3.4 Running Both Phases End-to-End
To run the full scientific investigation across both Phase A and Phase B in sequence:
```powershell
python run_experiment.py --phase all
```

---

### 3.5 Evidence Verification & Integrity Check
To verify the cryptographic checksums and evidence ledger of any completed study directory without running model fits:
```powershell
python phase_b_sequential_xai/run_phase_b.py --check results/phase_b_run/imdb
```

---

## 4. Are the `scripts/` and `tests/` Folders Necessary to Run End-to-End?

**No. Neither `scripts/` nor `tests/` is required to run the experiment end to end.**

- **Core Orchestration**: The entire scientific lifecycle (data preparation, calibration, design generation, model fitting, confirmation, XAI evaluation, and findings extraction) is fully self-contained in `run_experiment.py`, `phase_a_rsm_blocking/`, and `phase_b_sequential_xai/`.
- **Purpose of `tests/`**:
  - Contains [`tests/test_training_inputs.py`](tests/test_training_inputs.py), an automated unit and regression test suite executed by GitHub Actions CI on every commit/PR.
  - Verifies that sparse text matrices maintain canonical CSR indexing (`has_sorted_indices == True`), preventing the CUDA histogram misindexing issue from ever returning.
- **Purpose of `scripts/`**:
  - Contains standalone diagnostic and convenience utilities (e.g. [`scripts/verify_cuda_pipeline.py`](scripts/verify_cuda_pipeline.py) and [`scripts/run_corrected_campaign.py`](scripts/run_corrected_campaign.py)).
  - Useful for independent hardware sanity checks, but completely bypassed when executing via `run_experiment.py`.

---

## 5. Raw Results vs Downstream Findings Architecture

To adhere strictly to the Scientific Integrity Protocol, this repository maintains an explicit separation between **raw trial logs** and the **downstream reporting layer**:

```
[Experiment Runs (results/)]
  ├── Raw evaluations.jsonl, model weights, checkpoints
  └── Ephemeral and git-ignored
         │
         ▼
[Downstream Reporting Layer (findings/)]
  ├── figures/   <- Rendered publication plots (.png)
  ├── tables/    <- Clean markdown and LaTeX summary tables (.csv, .tex)
  ├── values/    <- Summary JSONs & semantic macros (macros.tex) for reports
  └── final_report/ <- Peer-reviewed publication manuscript (main.tex, main.pdf)
```

- **`results/`**: Contains raw, per-iteration evaluation logs (`evaluations.jsonl`), raw checkpoints, and booster dumps. These are large, ephemeral files excluded from Git to prevent repository bloat.
- **`findings/`**: Contains **strictly** downstream artifacts derived from completed results. No raw training datasets or trial-by-trial logs reside in `findings/`. All reported numbers are bound to `macros.tex` for zero-hardcoding manuscript drafting.

---

## 6. Key Scientific Documents

- [Values.md](Values.md): The unified empirical measurement registry containing all observed numbers, formulas, and macro bindings across Phase A and Phase B.
- [EXPERIMENT_EXPLANATION.md](EXPERIMENT_EXPLANATION.md): Comprehensive theoretical reference covering mathematical formulations, canonical stationary derivations, ANOVA vs SHAP proofs, and experimental accounting.
- [Phase A Manual](phase_a_rsm_blocking/README.md): Detailed Phase A documentation, California Housing setup, and blocking analysis.
- [Phase B Manual](phase_b_sequential_xai/README.md): Detailed Phase B documentation, IMDb NLP setup, capacity calibration, and XAI interaction pipeline.
- [Final Report Manuscript](findings/final_report/main.pdf): The complete publication manuscript detailing the entire multi-phase investigation.
