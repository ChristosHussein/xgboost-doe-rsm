# Explainable Sequential DOE for XGBoost HPO (IMDb Sentiment Analysis)

A modular, scientifically rigorous framework combining **Design of Experiments (Face-Centered CCD & RSM)** with **Explainable AI (ANOVA & SHAP)** for multi-objective hyperparameter optimization in **XGBoost** under stochastic nuisance blocking.

---

## Deliverables

1. **`experiment_code/`**:
   - `run_experiment.py`: Single master orchestrator script to prepare, calibrate, execute, and export the entire experiment.
   - `xgb_doe_xai/`: Modular Python package implementing the design, execution, surrogates, ANOVA, SHAP, and latency profiling.
   - `README.md`: Code documentation and module guide.
2. **`findings/`**:
   - `tables/`: Automated ANOVA tables (`val_loss` and `latency_us`), benchmark comparisons, confirmation predictions, and ANOVA vs SHAP agreement (CSV and LaTeX).
   - `figures/`: Multi-objective Pareto frontiers, factor importance comparisons, interaction heatmaps, curvature profiles, and boundary confirmation boxplots.
   - `values/`: Reconstructed `metrics.json` and publication-ready LaTeX `macros.tex`.
   - `README.md`: Summary guide to generated findings.
3. **`EXPERIMENT_EXPLANATION.md`**:
   - Complete scientific explanation of the research experiment, mathematical formulations of CCD and RSM, the 5 core research questions, and experimental protocols.
4. **`EXPECTED_VALUES.md`**:
   - Pre-experiment measurement contract and expected values registry documenting all metric IDs, source selectors, formulas, and bounds prior to data collection as per the scientific integrity standard.

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
