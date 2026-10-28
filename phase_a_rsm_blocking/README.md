# Phase A: Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking
## Evaluated on the California Housing Tabular Benchmark

---

## 1. Procedural Overview & Scientific Scope

**Phase A** establishes the rigorous parametric foundation of our research project:
> **Core Objective**: Investigating how classical **Design of Experiments (DOE)**, **Response Surface Methodology (RSM)**, and **Orthogonal Nuisance Blocking** optimize and model tree-based gradient boosted models (**XGBoost**) under competing objectives: predictive performance (RMSE) versus inference latency ($\mu\text{s}/\text{row}$) on continuous tabular data.

### Benchmark Dataset & Partitioning Protocol
Evaluated on the canonical California Housing tabular benchmark ($N = 20,640$ observations, 8 continuous features):
- **Development Pool**: 80% ($N = 16,512$ rows), partitioned across $B = 5$ spatial nuisance blocks via geographic coordinates to capture geographic and sampling heterogeneity.
- **Protected Holdout Set**: 20% ($N = 4,128$ rows), strictly isolated behind an access gate and evaluated exactly once to verify out-of-sample prediction intervals and prevent data leakage during surface fitting.

---

## 2. Experimental Methodology & Mathematical Formulation

### 2.1 Hyperparameter Factors ($k=4$)
Four continuous hyperparameters governing tree capacity, variance, and inference time are systematically coded into $[-1, +1]$:

| Factor | Parameter Name | Natural Range | Center ($0$) | Coding Transformation |
|---|---|---|---|---|
| $x_1$ | Maximum Depth (`max_depth`) | $[3, 9]$ | $6$ | Linear: $x_1 = (\text{depth} - 6) / 3$ |
| $x_2$ | Learning Rate ($\eta$, `learning_rate`) | $[0.05, 0.30]$ | $0.175$ | Linear: $x_2 = (\eta - 0.175) / 0.125$ |
| $x_3$ | Row Subsampling (`subsample`) | $[0.50, 1.00]$ | $0.75$ | Linear: $x_3 = (\text{subsample} - 0.75) / 0.25$ |
| $x_4$ | Leaf Regularization ($\lambda$, `reg_lambda`) | $[0.10, 10.00]$ | $1.00$ | $\log_{10}$ logarithmic scale |

### 2.2 Multi-Objective Targets & Latency Protocol
1. **Predictive Performance**: Root Mean Squared Error (RMSE) on standard-scaled target values.
2. **Inference Latency**: Single-row prediction latency measured on pinned CPU Core 0, enforcing 50 warmup calls followed by 1,000 timed evaluations per configuration to eliminate multi-threading jitter.

### 2.3 Procedural Stages
1. **Phase 1: Factorial Screening & Curvature Test**:
   - $2^4$ full factorial design (16 corners) plus 4 center-point replicates per block ($N = 20$ runs per block, $100$ total evaluations).
   - Lack-of-fit curvature test assesses non-linearity:
     $$F_{\text{curv}} = \frac{(\bar{y}_{\text{fact}} - \bar{y}_{\text{center}})^2 / (1/n_F + 1/n_C)}{MS_{\text{pure\_error}}} = 105,432.31 \quad (p < 0.0001)$$
   - Curvature significance formally proves first-order model inadequacy and justifies augmentation.
   - Spatial blocking captures significant nuisance variation ($\text{ICC} = 0.4069, F_{\text{block}} = 20.21, p < 0.0001$).
2. **Phase 2: Axial / Star Point Augmentation**:
   - Augmented with $2k = 8$ axial points at $\alpha = 1.0$ (Face-Centered CCD) or spherical $\alpha = 2.0$, plus additional center replicates to establish second-order orthogonality.
3. **Phase 3: Second-Order Response Surface & Canonical Analysis**:
   - Fits a full quadratic response surface with orthogonal block terms:
     $$y = \beta_0 + \sum_{i=1}^4 \beta_i x_i + \sum_{i=1}^4 \beta_{ii} x_i^2 + \sum_{i < j} \beta_{ij} x_i x_j + \sum_{b=2}^5 \gamma_b \text{Block}_b + \varepsilon$$
   - Predictive fidelity: $R^2 = 0.9951, R^2_{\text{adj}} = 0.9944$.
   - Canonical stationary analysis computes the unconstrained stationary point $\mathbf{x}_0 = -\frac{1}{2}\mathbf{B}^{-1}\mathbf{b}$. Because $\mathbf{x}_0$ lies outside the operational cube ($\|\mathbf{x}_0\|_2 = 17.16$), constrained optimization is resolved via **ridge analysis**.
4. **Multi-Objective Desirability Optimization**:
   - Derringer-Suich geometric desirability index ($D(\mathbf{x}) = \sqrt{d_{\text{RMSE}}(\mathbf{x}) \times d_{\text{Lat}}(\mathbf{x})}$) identifies the optimal trade-off ($\text{RMSE} = 0.4483$, Latency $= 127.4\,\mu\text{s}$).
5. **Fair-Budget Benchmark Baselines**:
   - Compared against Single-Objective TPE, Multi-Objective TPE (Optuna), and Unguided Random Search under equal evaluation budgets (140 evaluations per method across replicates).

---

## 3. Directory Layout & Module Architecture

```
phase_a_rsm_blocking/
├── pipeline.py              <- Main orchestrator for Phase 1 screening, blocking & Phase 3 CCD
├── analysis.py              <- Second-order OLS, ANOVA partitioning, LOF tests & canonical stationary analysis
├── latency.py               <- Hardware-pinned CPU latency measurement protocol (Core 0, 50 warmups + 1000 timed calls)
├── final_evaluation.py      <- Isolated holdout test evaluation gate on protected test partition
├── scientific_stats.py      <- Inferential statistics (Paired t-tests, Wilcoxon signed-rank tests, bootstrap CIs)
├── desirability_provenance.py <- Verification of historical desirability formulas and selection stability
├── config.yaml              <- Experimental bounds and factor coding rules
│
├── report.pdf / .tex        <- Peer-reviewed publication manuscript detailing complete Phase A findings
├── technical_supplement.pdf <- Technical supplement containing mathematical proofs, LOF derivations & audits
│
├── scripts/                 <- Benchmark and reproduction CLI utilities
│   ├── run_benchmarks.py    <- Executes fair-budget benchmark comparisons (DOE vs MOTPE vs Random)
│   ├── reproduce_all.py     <- Master reproduction script running the full Phase A programme
│   ├── fit_latency_models.py<- Fits and compares first- and second-order latency response surfaces
│   ├── measure_latency.py   <- Standalone CPU latency benchmarking harness
│   └── research_publication.py <- Publication figure and table formatting utilities
│
└── results/                 <- Canonical empirical runs (17,077 model fits; separate from findings/)
    ├── runs.csv             <- All Phase 1 and CCD model evaluation rows
    ├── phase1.json          <- First-order ANOVA and curvature diagnostics
    ├── phase3.json          <- Second-order surface coefficients, stationary analysis & ridge path
    ├── confirmation.json    <- Holdout prediction intervals and empirical coverage validation
    ├── benchmark.csv        <- Multi-objective optimizer benchmark logs
    └── benchmark_summary.json <- Hypervolume, spread, and paired statistical comparison tests
```

---

## 4. Execution & Replication Guide

### 4.1 Running via Root Master CLI (Recommended)
From the repository root:
```powershell
# 1. Fast smoke verification of Phase A (< 15 seconds)
python run_experiment.py --phase a --mode smoke

# 2. Full replication campaign (all replicates, benchmarks & holdout gate)
python run_experiment.py --phase a --mode full
```

### 4.2 Running Directly within `phase_a_rsm_blocking`
```powershell
cd phase_a_rsm_blocking

# 1. Run benchmarks in smoke mode:
python scripts/run_benchmarks.py --mode smoke

# 2. Run full benchmark replication:
python scripts/run_benchmarks.py --mode full --confirm-full-budget

# 3. Fit second-order latency response surfaces:
python scripts/fit_latency_models.py
```

---

## 5. Key Empirical Deliverables & Acceptance Verification

### 5.1 Downstream Reporting Layer (`findings/phase_a/`)
All downstream figures, publication tables, and numerical macros are exported to the central reporting layer:
- **Figures** (`findings/phase_a/figures/`):
  - `response_surface_rmse_2d_3d.png`: 2D contour maps and 3D response surface topologies.
  - `revision_v2_pareto_front.png`: Multi-objective Pareto frontiers comparing DOE vs MO-TPE.
  - `diagnostics_panel_4in1.png`: Residual vs fitted, normal Q-Q, Cook's distance, and residual vs block.
- **Tables** (`findings/phase_a/tables/`):
  - `anova_phase1_rmse.csv` & `.tex`: Phase 1 screening ANOVA decomposition.
  - `anova_ccd_second_order_rmse.csv` & `.tex`: Full second-order ANOVA with lack-of-fit.
  - `anova_ccd_second_order_latency.csv` & `.tex`: Latency surface ANOVA.
- **Values & LaTeX Macros** (`findings/phase_a/values/`):
  - `diagnostics.json`: Normality tests, Levene homoscedasticity, Durbin-Watson, and OLS coefficients.
  - `benchmark_summary.json`: Multi-objective hypervolume and paired statistical tests.
  - `macros.tex`: 520 semantic LaTeX macros auto-populated for publication drafting.

### 5.2 Summary of Verified Empirical Values
*Cross-referenced with root [`Values.md`](../Values.md)*:
- Curvature $F$-statistic: **105,432.31** ($p < 0.0001$)
- Spatial nuisance block: $\text{ICC} = \mathbf{0.4069}$ ($F = 20.21, p < 0.0001$)
- Latency block variance: $\text{ICC} = \mathbf{0.0000}$ (Hardware latency is spatially invariant)
- Second-order surface fit: $R^2 = \mathbf{0.9951}, R^2_{\text{adj}} = \mathbf{0.9944}$
- Constrained in-cube optimum: $\hat{y}_{\text{RMSE}} = \mathbf{0.4483}$, Latency $= \mathbf{127.4}\,\mu\text{s}$
- Pareto Hypervolume advantage: DOE-RSM $= \mathbf{16.5046}$ vs MO-TPE $= \mathbf{14.8164}$ (**+11.39%**)
- Holdout 95% PI coverage: **94.2%** (Nominal 95%)
