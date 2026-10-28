# Experiment Codebase: Explainable Sequential DOE for XGBoost HPO

This directory contains the unified execution code to run the complete research experiment.

## Architecture

- **`run_experiment.py`**: Unified single-file CLI orchestrator that controls all stages of the study:
  - `--stage prepare`: Downloads and verifies full IMDb and Yelp datasets with SHA-256 manifests.
  - `--stage calibrate`: Development-only tree capacity and latency SLA calibration.
  - `--stage run`: Executes equal-budget multi-objective comparison across Static CCD, Sequential CCD, and MO-TPE.
  - `--stage words`: Local word-level RSM and pairwise SHAP interaction study.
  - `--stage generate-findings`: Automatically extracts and saves all ANOVA tables, benchmark comparisons, Pareto plots, and LaTeX macros into `findings/`.
  - `--stage smoke`: Fast 25-second synthetic smoke demonstration.
  - `--stage all`: End-to-end execution of all stages in one click.
  - `--check`: Reconstructs and verifies all observations, hashes, and artifact integrity.

## Modular Package: `xgb_doe_xai`

The underlying modular logic is implemented in the clean, fully-tested `xgb_doe_xai` package:
- `config.py`: Benchmark and protocol configurations.
- `data.py`: Text loading, TF-IDF vectorization, nuisance-block partitioning, and protected test gate.
- `design.py`: Face-Centered Central Composite Design (CCD) and sequential adaptive infill rules.
- `execution.py`: Model fitting and validation scoring under cross-validation.
- `profiler.py`: Single-thread latency measurement with pinned CPU affinity.
- `surrogates.py`: Second-order quadratic polynomial response surfaces (OLS) and Matern Gaussian Processes.
- `xai.py`: 2nd-order ANOVA SS decomposition, exact coalition Shapley values, and pairwise SHAP interactions.
- `local_explanations.py` & `word_study.py`: Document-level word feature perturbation surrogate and pairwise SHAP interactions.
- `benchmarks.py`: Multi-objective TPE, random search, hypervolume, and regret calculation.
- `study.py` & `study_reporting.py`: Experiment runner, verification gate, and findings generator.

## Usage

```powershell
# Fast smoke verification:
python experiment_code/run_experiment.py --stage smoke

# Full IMDb experiment:
python experiment_code/run_experiment.py --stage all --dataset full_imdb --training-threads 4
```
