# Explainable Sequential DOE for XGBoost HPO (IMDb Sentiment Analysis)

A modular, scientifically rigorous framework combining **Design of Experiments (Face-Centered CCD & RSM)** with **Explainable AI (ANOVA & SHAP)** for multi-objective hyperparameter optimization in **XGBoost** under stochastic nuisance blocking.

Directly addresses:
- `Original specs.txt`: Factorial screening & RSM for ML HPO; classical response surfaces vs ML models; Explainable AI.
- `Remarks,.txt` (and `Remarks.txt`):
  1. Calibrated capacity to resolve underfitting.
  2. Unified warmed latency measurement (100 warmups + 500 timed single-document calls, pinned CPU affinity) to resolve false feasibility.
  3. Strictly protected test set and reserved confirmation coordinates (interior vs boundary) to resolve boundary candidate optimism bias.
  4. ANOVA vs SHAP interaction pipeline.
  5. Sentiment analysis task (IMDb).
  6. Clean, modular Python package that automatically produces tables, values, and plots.
- `Research approach.txt`: Explainable Sequential DOE, ANOVA-SHAP agreement, Pareto benchmarking, and local word explanations.

---

## Deliverables & Repository Structure

- **`experiment_code/`**:
  - `run_experiment.py`: Single-file master orchestrator to prepare, calibrate, run, and export all findings in 1 command.
  - `README.md`: Codebase architecture and instructions.
- **`run_experiment.py`**: Root entry point shortcut delegating to `experiment_code/run_experiment.py`.
- **`xgb_doe_xai/`**: Clean, modular Python package containing tested modules:
  - `config.py`, `data.py`, `design.py`, `execution.py`, `profiler.py`, `surrogates.py`, `xai.py`, `local_explanations.py`, `word_study.py`, `benchmarks.py`, `study.py`, `study_reporting.py`.
- **`findings/`**: Automated deliverables folder:
  - `tables/`: ANOVA tables (`val_loss` & `latency_us`), benchmark comparisons, confirmation predictions, and ANOVA vs SHAP agreement (CSV and LaTeX).
  - `figures/`: Multi-objective Pareto frontiers, ANOVA SS vs SHAP importance, interaction heatmaps, curvature profiles, and boundary confirmation boxplots.
  - `values/`: Reconstructed `metrics.json` and publication-ready LaTeX `macros.tex`.
  - `README.md`: Guide to all findings.
- **`EXPERIMENT_EXPLANATION.md`**: Complete explanation file of the research experiment, mathematical formulations, Q1-Q5 research questions, and experimental protocol.
- **`EXPECTED_VALUES_MEASUREMENT_CONTRACT.md`**: Pre-experiment measurement contract & expected values registry as per the Scientific Integrity skill (Phase -1).
- **`archive/`**: Local archive preserving all legacy California Housing scripts, previous reports, and historical runs.
- **GitHub PR #5**: Preserves all report-writing publication tooling, LaTeX templates, PDF build scripts, and full verification test suites.

---

## Quick Start & Replication Commands

```powershell
# 1. Fast 25-Second Smoke Verification (Synthetic corpus)
python run_experiment.py --stage smoke

# 2. Data Preparation (Downloads and partitions full IMDb)
python run_experiment.py --stage prepare --data-dir data

# 3. Development-Only Calibration (Selects tree count and latency threshold)
python run_experiment.py --stage calibrate --data-dir data --output-dir results/calibration

# 4. Full Experiment Execution (Equal-budget benchmark on full IMDb)
python run_experiment.py --stage run --dataset full_imdb --data-dir data --calibration-dir results/calibration --output-dir results/experiment_run

# 5. Automated Production of All Tables, Figures, and Values
python run_experiment.py --stage generate-findings --output-dir results/experiment_run --findings-dir findings

# 6. One-Click Complete Pipeline (Prepare -> Calibrate -> Run -> Words -> Findings)
python run_experiment.py --stage all --dataset full_imdb --data-dir data --findings-dir findings
```

---

## Core Research Questions

- **Q1 (Explanation Agreement)**: Do ANOVA 2nd-order interaction terms and surrogate SHAP interaction values agree on factor importance and pairwise synergy?
- **Q2 (Generalization & Boundary Prediction)**: Can 2nd-order response surfaces and Gaussian Processes accurately predict validation performance on unseen hyperparameter coordinates, specifically interior vs boundary coordinates?
- **Q3 (Sequential Adaptation)**: Does adaptively augmenting the initial CCD with uncertainty/space-filling points improve predictive fidelity and optimization efficiency compared to static CCD?
- **Q4 (Multi-Objective Optimization Efficiency)**: How does Sequential DOE compare against Multi-Objective TPE in Pareto front quality (hypervolume), regret, and feasibility under strict latency constraints with an equal model-fit budget ($N=140$ fits per repeat)?
- **Q5 (Local Word Explanations)**: Can local response surfaces serve as interpretable surrogates for document-level word feature interactions, mirroring LIME but with explicit 2nd-order interaction and curvature terms?
