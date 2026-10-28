# Explainable Sequential Response Surface Methodology for XGBoost

A comprehensive, mathematically rigorous research repository combining **Sequential Design of Experiments (Face-Centered Central Composite Design & RSM)** with **Explainable AI (ANOVA & SHAP)** for multi-objective hyperparameter optimization in **XGBoost** under stochastic nuisance blocking.

The project encompasses two coordinated studies:
1. **Study 1 (California Housing Tabular Benchmark)**: Empirical study and peer-reviewed research manuscript on tabular regression under nuisance blocking, Pareto hypervolume optimization, and second-order latency modeling.
2. **Study 2 (IMDb Sentiment Analysis NLP Study)**: Expanding the methodology into high-dimensional sparse text classification, addressing tree capacity calibration, unified warmed single-core latency protocols, holdout test isolation gates, and ANOVA-SHAP interaction synergy.

---

## Deliverables & Repository Structure

```
.
├── california_housing/               <- Study 1: Complete Tabular Benchmark & Manuscript
│   ├── report.pdf                    <- Primary research manuscript (PDF)
│   ├── report.tex                    <- Manuscript LaTeX source (macro-bound)
│   ├── technical_supplement.pdf      <- Comprehensive technical supplement (PDF)
│   ├── technical_supplement.tex      <- Supplement LaTeX source
│   ├── REPORT.md                     <- Generated empirical research record
│   ├── pipeline.py, analysis.py      <- Core optimization & ANOVA execution engine
│   ├── plots.py, scientific_stats.py <- Publication diagnostics & inference
│   ├── scripts/                      <- Publication build, verification & audit tools
│   ├── tables/                       <- 16 LaTeX & CSV tables for California Housing
│   ├── figures/                      <- Publication figures, response surfaces, Pareto front
│   ├── results/                      <- Canonical frozen empirical runs (17,077 fits)
│   ├── docs/                         <- Scientific audits, claim bindings & provenance
│   └── tests/                        <- 26 statistical, protocol & integrity test suites
│
├── experiment_code/                  <- Study 2: Clean code to run the whole experiment
│   ├── run_experiment.py             <- Single master orchestrator script
│   ├── xgb_doe_xai/                  <- Modular Python package (design, XAI, profiling)
│   └── README.md                     <- Code architecture and execution manual
│
├── findings/                         <- Study 2: Automated findings, tables & figures
│   ├── tables/                       <- Automated ANOVA, benchmark & confirmation tables (CSV & TeX)
│   ├── figures/                      <- Publication plots: Pareto fronts, heatmaps, profiles (PNG)
│   ├── values/                       <- Reconstructed metrics.json and LaTeX macros.tex
│   ├── evidence_sources.yaml         <- Evidence registry schema
│   ├── measurement_contract.yaml     <- Metric binding contracts
│   └── README.md                     <- Summary of generated empirical findings
│
├── EXPERIMENT_EXPLANATION.md         <- Deliverable: Detailed explanation of the experiment
├── EXPECTED_VALUES.md                <- Deliverable: Pre-experiment expected values registry
├── run_experiment.py                 <- Root single-command entry point to run Study 2
├── requirements.txt                  <- Python environment dependencies
└── .gitignore                        <- Local archive/ and raw data/ exclusion
```

---

## Quick Start & Execution

### 1. Study 2: Sequential DoE NLP Experiment (IMDb Sentiment Analysis)

All operations can be run via the root entry point `run_experiment.py`:

```bash
# A. Fast 12-Second Smoke Verification (Synthetic corpus end-to-end)
python run_experiment.py --stage smoke

# B. Download & Preprocess Full IMDb Dataset (49,578 unique documents)
python run_experiment.py --stage prepare --data-dir data

# C. Tree Capacity & Latency Calibration (Resolves underfitting on development pool)
python run_experiment.py --stage calibrate --data-dir data --output-dir results/calibration

# D. Run Equal-Budget Multi-Objective Benchmark (Static CCD vs Sequential CCD vs MO-TPE)
python run_experiment.py --stage run --dataset full_imdb --data-dir data --calibration-dir results/calibration --output-dir results/experiment_run

# E. Automatically Regenerate All Deliverables (Tables, Figures, Values)
python run_experiment.py --stage generate-findings --output-dir results/experiment_run --findings-dir findings

# F. Single Command Complete Pipeline
python run_experiment.py --stage all --dataset full_imdb --data-dir data --findings-dir findings --training-threads 4
```

### 2. Study 1: California Housing Tabular Study Verification

All verification commands for Study 1 are executed inside `california_housing/`:

```bash
cd california_housing

# Run full test suite (182 tests)
python -m pytest tests/ test_verification.py -q

# Run repository-wide scientific consistency audit
python scripts/audit_scientific_consistency.py

# Verify artifact bindings and report synchronization
python scripts/generate_report_artifacts.py --check
python scripts/generate_research_reporting.py --check
python scripts/build_publication.py --check
```

---

## Deliverables Checklist

| Deliverable | Location | Description |
|---|---|---|
| **Whole Experiment Code** | [`experiment_code/`](experiment_code/) | Self-contained package (`xgb_doe_xai`) & orchestrator (`run_experiment.py`) |
| **Findings, Tables & Figures** | [`findings/`](findings/) | Automated `tables/` (CSV & TeX), `figures/` (PNG), `values/` (`metrics.json`, `macros.tex`) |
| **Experiment Explanation** | [`EXPERIMENT_EXPLANATION.md`](EXPERIMENT_EXPLANATION.md) | Formulations of CCD, RSM, ANOVA-SHAP alignment, and 5 research questions |
| **Expected Values Contract** | [`EXPECTED_VALUES.md`](EXPECTED_VALUES.md) | Pre-experiment measurement contract & value bounds per scientific integrity |
| **California Housing Study** | [`california_housing/`](california_housing/) | Complete Study 1 code, data, reports ([`report.pdf`](california_housing/report.pdf)), tables, and tests |
| **Root Runner** | [`run_experiment.py`](run_experiment.py) | 1-click execution entry point delegating to `experiment_code` |
