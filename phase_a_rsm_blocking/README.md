# Phase A: Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking
## Evaluated on the California Housing Tabular Benchmark

---

## 1. Procedural Overview & Scientific Scope

**Phase A** establishes the rigorous parametric foundation of our research project:
> Investigating how classical **Design of Experiments (DOE)**, **Response Surface Methodology (RSM)**, and **Orthogonal Nuisance Blocking** optimize and model tree-based gradient boosted models (**XGBoost**) under competing objectives: predictive performance (RMSE) versus inference latency ($\mu\text{s}/\text{row}$).

The empirical evaluation is conducted on the California Housing benchmark ($N = 20,640$ observations, 8 continuous features), partitioned into:
- **Development Pool**: 80% ($N = 16,512$ rows), stratified across $B = 5$ spatial nuisance blocks to model geographic and sampling heterogeneity.
- **Protected Holdout**: 20% ($N = 4,128$ rows), strictly isolated behind a one-time evaluation gate.

---

## 2. Experimental Methodology

### 2.1 Hyperparameter Factors ($k=4$)
Four continuous hyperparameters governing tree capacity, variance, and inference time are systematically varied:
- **$x_1$ — Maximum Depth (`max_depth`)**: Range $[3, 9]$, center $= 6$ (coded $[-1, +1]$).
- **$x_2$ — Learning Rate ($\eta$, `learning_rate`)**: Range $[0.05, 0.30]$, center $= 0.175$ (coded $[-1, +1]$).
- **$x_3$ — Row Subsampling (`subsample`)**: Range $[0.50, 1.00]$, center $= 0.75$ (coded $[-1, +1]$).
- **$x_4$ — Leaf Regularization ($\lambda$, `reg_lambda`)**: Range $[0.10, 10.00]$, center $= 1.00$ ($\log_{10}$ coded $[-1, +1]$).

### 2.2 Procedural Phases
1. **Phase 1: First-Order Factorial Screening & Curvature Test**:
   - $2^4$ factorial core plus 4 center-point replicates per block ($N = 20$ runs per block, $100$ total evaluations).
   - ANOVA decomposes total variance into factor main effects, two-factor interactions, and nuisance block shifts.
   - Significant lack-of-fit curvature ($F_{\text{curv}} = 105,432.31, p < 0.0001$) diagnoses first-order model inadequacy and triggers Phase 2.
   - Spatial blocking captures substantial nuisance variation ($\text{ICC}_{\text{ANOVA}} = 0.4069, F_{\text{block}} = 20.21, p < 0.0001$).
2. **Phase 2: Axial / Star Point Augmentation**:
   - Augmented with $2k = 8$ axial points at $\alpha = 1.0$ (Face-Centered CCD) or spherical $\alpha = 2.0$, plus additional center replicates.
3. **Phase 3: Second-Order Response Surface & Canonical Analysis**:
   - Fits a full quadratic response surface with block parameters:
     $$y = \beta_0 + \sum_{i=1}^4 \beta_i x_i + \sum_{i=1}^4 \beta_{ii} x_i^2 + \sum_{i < j} \beta_{ij} x_i x_j + \sum_{b=2}^5 \gamma_b \text{Block}_b + \varepsilon$$
   - Surface demonstrates outstanding predictive fidelity ($R^2 = 0.9951, R^2_{\text{adj}} = 0.9944$).
   - Canonical stationary analysis identifies a saddle point ($\mathbf{x}_0$ outside the operational cube), resolved via **ridge analysis** along radius $R = \|\mathbf{x}\|_2$.
4. **Multi-Objective Optimization**:
   - Derringer-Suich geometric desirability index ($D(\mathbf{x}) = \sqrt{d_{\text{RMSE}} \times d_{\text{Lat}}}$) locates the constrained in-cube optimum.
5. **Benchmark Comparison**:
   - Evaluated against Single-Objective TPE, Multi-Objective TPE (Optuna), and Unguided Random Search under an equal budget (140 evaluations per method across replicates).

---

## 3. Directory Layout & Module Architecture

```
phase_a_rsm_blocking/
├── pipeline.py              <- Main execution engine for Phase 1 screening, blocking & Phase 3 CCD
├── analysis.py              <- Second-order OLS, ANOVA partitioning, LOF tests & canonical stationary analysis
├── latency.py               <- Hardware-pinned CPU latency measurement protocol (Core 0, 50 warmups + 1000 timed calls)
├── final_evaluation.py      <- Isolated holdout test evaluation gate on protected test partition
├── scientific_stats.py      <- Inferential statistics (Paired t-tests, Wilcoxon signed-rank tests, bootstrap CIs)
├── desirability_provenance.py <- Verification of historical desirability formulas and selection stability
├── config.yaml              <- Configuration file defining factor bounds, coding transformations, and hardware pins
│
├── report.pdf / .tex        <- Peer-reviewed publication manuscript detailing complete Phase A findings
├── technical_supplement.pdf <- Technical supplement containing mathematical proofs, LOF derivations & audits
│
├── scripts/                 <- Benchmark and reproduction CLI tools
│   ├── run_benchmarks.py    <- Executes fair-budget benchmark comparisons (DOE vs MOTPE vs Random)
│   ├── reproduce_all.py     <- Master reproduction script running the full Phase A programme
│   ├── fit_latency_models.py<- Fits and compares first- and second-order latency response surfaces
│   ├── measure_latency.py   <- Standalone CPU latency benchmarking harness
│   └── research_publication.py <- Publication figure and table formatting utilities
│
└── results/                 <- Canonical empirical runs (17,077 model fits)
    ├── runs.csv             <- All Phase 1 and CCD model evaluation rows
    ├── phase1.json          <- First-order ANOVA and curvature diagnostics
    ├── phase3.json          <- Second-order surface coefficients, stationary analysis & ridge path
    ├── confirmation.json    <- Holdout prediction intervals and empirical coverage validation
    ├── benchmark.csv        <- Multi-objective optimizer benchmark logs
    ├── benchmark_summary.json <- Hypervolume, spread, and paired statistical comparison tests
    └── revision_v2/         <- Revision 2 canonical replication artifacts (full_run_001)
```

---

## 4. Execution & Replication Commands

### 4.1 Running via Root Master CLI (Recommended)
From the repository root:
```powershell
# Fast smoke verification of Phase A (< 15 seconds)
python run_experiment.py --phase a --mode smoke

# Full replication campaign (all replicates, benchmarks & holdout gate)
python run_experiment.py --phase a --mode full
```

### 4.2 Running Directly within `phase_a_rsm_blocking`
```powershell
cd phase_a_rsm_blocking

# 1. Run the benchmarks in smoke mode:
python scripts/run_benchmarks.py --mode smoke

# 2. Run full benchmark replication:
python scripts/run_benchmarks.py --mode full --confirm-full-budget

# 3. Fit second-order latency models:
python scripts/fit_latency_models.py
```

---

## 5. Output Findings & Publication Artifacts

All processed findings, tables, figures, and LaTeX macros for Phase A are saved in the unified root `findings/` directory:
- **Figures** (`findings/phase_a/figures/`): 2D/3D response surfaces, Pareto frontiers, 4-in-1 residual diagnostics panels, and research pipeline schematics.
- **Tables** (`findings/phase_a/tables/`): LaTeX and CSV tables for ANOVA phase 1, ANOVA CCD second-order (RMSE & Latency), lack-of-fit, confirmation intervals, and optimizer benchmarks.
- **Values** (`findings/phase_a/values/`): `diagnostics.json`, `benchmark_summary.json`, and macro-bound `macros.tex`.
