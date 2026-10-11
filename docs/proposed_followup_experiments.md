# Proposed Follow-Up Experiments (Work Package F — Task 13)

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**
>
> Consistent with the scientific integrity constraints of Work Package F, no new XGBoost model fits have been executed in this audit, and all 40 frozen experimental evidence files (`results/` and `config.yaml`) remain untouched. The three pre-specified experimental protocols below address the primary methodological and scope limitations identified in the independent scientific audit of the historical (`v1.0.0`) and Revision-v2 (`full_run_001`) campaigns.

---

## Overview of Audit Findings Driving Follow-Up Design

The independent scientific audit (`docs/scientific_claims_audit.md`) identified three structural boundaries in the current study:

1. **Single-Batch Pre-Specified FCCD vs. True Sequential Response Surface Optimization**: The historical 140-run campaign (`results/runs.csv`) and prospective Revision-v2 Repeated DOE benchmark (`20 x 140` fits) executed all 25 Face-Centered Central Composite Design (FCCD) coordinates (`16` factorial corners, `8` axial star points, `1` center point) in a single pre-tabulated, randomized batch across 5 seed blocks. Consequently, 45 of 140 runs ($32.14\%$) were spent at the degenerate $\eta = 0.01$ boundary (mean RMSE $0.7545$ vs. $0.5055$ at $\eta \in \{0.105, 0.20\}$), preventing the quadratic polynomial ($R^2_{\text{pred}} = 0.9035$) from resolving local ridges in the high-$\eta$ interior where adaptive TPE concentrates its budget.
2. **Single-Dataset and Fixed-Boosting-Horizon Scope**: All empirical findings are conditioned on the California Housing tabular regression dataset ($N = 20{,}640$, $d = 8$ continuous features) at a fixed boosting horizon of `n_estimators = 100` on a single hardware platform. Whether the relative trade-off between structured variance decomposition and adaptive sample efficiency transfers to larger feature spaces, higher noise floors, or longer boosting horizons remains untested.
3. **Four-Factor Full Factorial Scaling Ceiling**: A full $2^k$ factorial plus $2k$ axial points scales exponentially with the number of tuned hyperparameters $k$. Extending structured DOE to the broader 8–10 hyperparameter space typical of production gradient boosted decision trees (`min_child_weight`, `gamma`, `max_bin`, `grow_policy`, `n_estimators`, alongside `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_lambda`) requires fractional factorial or definitive screening designs prior to local response surface refinement.

The three protocols below provide concrete, pre-registered designs to resolve each limitation in future experimental campaigns.

---

## Proposal A — True Two-Stage Sequential Response Surface Optimization (Screening $\to$ Steepest Descent $\to$ Re-Centered Local CCD)

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective and Hypothesis
Evaluate whether a genuinely sequential two-stage Response Surface Methodology (RSM) workflow—using an initial Stage 1 screening design to eliminate degenerate boundary regions ($\eta = 0.01$) and relocate the Stage 2 Central Composite Design to the promising high-$\eta$ interior—closes the $0.0028$–$0.0030$ holdout Test RMSE gap against adaptive Tree-structured Parzen Estimators (TPE) under an identical 140-fit search budget.

* **Primary Hypothesis ($H_{1A}$)**: Under an identical 140-fit search budget per outer replicate ($R = 20$), a two-stage sequential DOE/RSM protocol achieves strictly lower holdout Test RMSE than single-batch global FCCD ($\Delta\text{RMSE}_{\text{test}} < 0$ at $\alpha = 0.05$, two-sided paired $t$-test and Wilcoxon signed-rank test across 20 matched outer splits) and narrows the local surrogate prediction error in the high-$\eta$ region ($\eta \in [0.08, 0.22]$) to within $\pm 0.003$ RMSE.
* **Secondary Hypothesis ($H_{2A}$)**: Relocating Stage 2 axial and factorial evaluations to the interior subregion increases quadratic cross-validated $R^2_{\text{pred}}$ within the active optima basin from $0.9035$ (global) to $\ge 0.985$ (local) without violating the $145\,\mu\text{s}$ `predict` latency constraint.

### 2. Two-Stage Budget Allocation (Exactly 140 Model Fits per Search Replicate)
To maintain strict parity with the 140-fit budget of the current Revision-v2 campaign, each outer search replicate $r \in \{1, \dots, 20\}$ partitions its 140 fits into three sequential phases executed with real-time stopping and re-centering rules:

1. **Stage 1A — Blocked Fractional Screening + Center Curvature Check (55 fits)**:
   * **Design**: $2^{4-1}_{\text{IV}}$ fractional factorial ($I = ABCD$, 8 corner coordinates) plus $n_c = 3$ center-point replicates per block across $B = 5$ validation-split blocks ($5 \times (8 + 3) = 55$ fits) spanning the original global domain: `max_depth` $\in [3, 10]$, `learning_rate` $\in [0.01, 0.20]$, `subsample` $\in [0.5, 1.0]$, `colsample_bytree` $\in [0.5, 1.0]$.
   * **Decision Rule at Fit 55**: Fit the first-order main-effects model with block indicators and evaluate the 1-df curvature contrast. Identify active main effects ($p < 0.05$ using the block-adjusted pooled coordinate denominator) and compute the steepest descent direction $-\hat{\mathbf{g}} / \|\hat{\mathbf{g}}\|$ on validation RMSE subject to the linear latency constraint $\hat{y}_{\text{lat}}(\mathbf{x}) \le 145\,\mu\text{s}$.
2. **Stage 1B — Steepest Descent Line Search (15 fits)**:
   * **Design**: Evaluate 3 candidate step sizes $\lambda \in \{0.5, 1.0, 1.5\}$ in coded units along the constrained steepest descent path from the Stage 1A center point across all $B = 5$ blocks ($3 \times 5 = 15$ fits; cumulative budget = 70 fits).
   * **Re-Centering Rule at Fit 70**: Select the step size $\lambda^*$ minimizing block-averaged validation RMSE as the new local design center $\mathbf{x}_0^{(2)}$, clamping the local trust-region half-widths to $\Delta\text{max\_depth} = \pm 2$ levels, $\Delta\eta = \pm 0.05$ (strictly excluding $\eta < 0.05$ when $\eta_0^{(2)} \ge 0.10$), $\Delta\text{subsample} = \pm 0.15$, and $\Delta\text{colsample\_bytree} = \pm 0.15$.
3. **Stage 2 — Local Blocked Face-Centered CCD Refinement (70 fits)**:
   * **Design**: Construct a local 14-point design around $\mathbf{x}_0^{(2)}$ consisting of a $2^{4-1}_{\text{IV}}$ foldover ($8$ corners, completing a full local $2^4$ when combined with adjacent interior points) plus $2k = 8$ axial star points ($\alpha = 1.0$ on the local trust region) and 2 center points, evaluated across a subset or balanced incomplete block allocation totaling exactly 70 fits (e.g., 14 local coordinates $\times 5$ blocks = 70 fits; cumulative budget = 140 fits).
   * **Final Selection**: Fit the local second-order RSM quadratic model on the 70 Stage 2 observations, optimize composite desirability $D(\mathbf{x}) = (d_1(\mathbf{x})\,d_2(\mathbf{x}))^{1/2}$ within the local trust region, and select the final configuration $\mathbf{x}^*_{\text{seq-RSM}, r}$.

### 3. Evaluation and Statistical Analysis Plan
* **Pairing Structure**: Execute across the exact 20 outer search replicates (`search_id` $1\dots 20$, outer splits $5001\dots 5020$) used in Revision-v2 so each replicate is paired on identical $80/20$ development/holdout data partitions.
* **Holdout Protocol**: Retrain $\mathbf{x}^*_{\text{seq-RSM}, r}$ across the 20 independent holdout evaluation seeds (`2001..2020`) and measure inference latency across 5 isolated benchmark sessions (`2,000` timed + `100` warmup single-row calls per session).
* **Primary Endpoints**: (i) Paired mean difference in holdout Test RMSE vs. single-batch Repeated DOE and vs. Constrained TPE ($df = 19$ paired $t$-test + Wilcoxon signed-rank); (ii) 95% bootstrap confidence interval on holdout Pareto hypervolume at $[0.60, 250.0]$ and $[0.65, 275.0]$.

---

## Proposal B — Multi-Dataset and Multi-Horizon Generalization Benchmark

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective and Hypothesis
Determine how the sample efficiency trade-off between structured blocked DOE/RSM (both single-batch and two-stage Proposal A) and adaptive Bayesian optimization (Constrained TPE, MO-TPE) varies as a function of dataset dimensionality $d$, observation noise floor $\sigma_\varepsilon^2$, and boosting horizon $M = \text{n\_estimators}$.

* **Primary Hypothesis ($H_{1B}$)**: As the ratio of irreducible observation noise to hyperparameter signal increases, blocked designs with explicit seed-block replication provide higher selection stability (lower between-search SD $\sigma_{\text{between}}$) relative to single-evaluation adaptive TPE, whereas low-noise datasets with sharp nonlinear ridges favor adaptive sequential sampling.
* **Secondary Hypothesis ($H_{2B}$)**: Increasing `n_estimators` from $M = 100$ to $M \in \{300, 500\}$ shifts the optimal learning rate $\eta^*$ downward from the $[0.15, 0.20]$ upper boundary into the interior $[0.03, 0.10]$, reducing the boundary-truncation penalty observed at $M = 100$.

### 2. Benchmark Dataset Matrix and Boosting Horizons
Evaluate $D = 6$ standardized OpenML tabular regression benchmarks spanning diverse sample sizes $N$ and feature counts $d$, crossed with three boosting horizons $M \in \{100, 300, 500\}$:

| Dataset ID | Dataset Name | Rows ($N$) | Features ($d$) | Domain / Regime | Latency Cap ($\mu\text{s}$, scaled to $M$) |
| :--- | :--- | ---: | ---: | :--- | ---: |
| **D1** | California Housing (Baseline) | $20{,}640$ | $8$ | Geospatial census regression | $145 \times (M/100)$ |
| **D2** | Miami Housing (`OpenML 43093`) | $13{,}932$ | $15$ | Hedonic real-estate pricing | $160 \times (M/100)$ |
| **D3** | Diamonds (`OpenML 42225`) | $53{,}940$ | $9$ | Mixed-feature non-linear pricing | $150 \times (M/100)$ |
| **D4** | Superconductivty (`OpenML 43174`) | $21{,}263$ | $81$ | High-dimensional materials physics | $220 \times (M/100)$ |
| **D5** | NYC Taxi Tip / Duration (Subsampled) | $50{,}000$ | $18$ | High-noise spatiotemporal tabular | $175 \times (M/100)$ |
| **D6** | Elevators (`OpenML 216`) | $16{,}599$ | $18$ | Aerodynamic control surface state | $160 \times (M/100)$ |

### 3. Experimental Design and Statistical Analysis Plan
* **Replicates**: $R = 20$ outer stratified/grouped $80/20$ development/holdout splits per dataset $\times$ 3 boosting horizons $M \in \{100, 300, 500\}$.
* **Compared Methods (140 fits per search replicate)**:
  1. Single-Batch Blocked FCCD + RSM (25 coordinates $\times 5$ blocks + 15 confirmation).
  2. Two-Stage Sequential Blocked RSM (Proposal A: 55 screening + 15 line search + 70 local CCD).
  3. Constrained Single-Objective TPE (`Optuna` with penalty barrier).
  4. Multi-Objective TPE (`MOTPE` + knee selection).
  5. Uniform Random Search.
* **Cross-Dataset Inference**:
  * Within each $(d, M)$ cell: paired $t$-test and Wilcoxon signed-rank test ($df = 19$) with Holm–Bonferroni multiplicity adjustment.
  * Across the $6 \times 3 = 18$ dataset-by-horizon regimes: Friedman non-parametric omnibus rank test with Nemenyi / Conover post-hoc critical difference diagrams, plus a linear mixed-effects meta-regression modeling $\Delta\text{RMSE}_{\text{test}}$ as a function of empirical noise-to-signal ratio $\hat{\sigma}_{\text{block}} / \hat{\sigma}_{\text{coord}}$ and feature dimension $d$.

---

## Proposal C — High-Dimensional Screening + Local Response Surface Refinement ($k = 9$ Hyperparameters)

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective and Hypothesis
Extend the structured DOE/RSM methodology from $k = 4$ primary tree parameters to a $k = 9$ production XGBoost hyperparameter space without incurring the $2^9 = 512$-corner combinatorial explosion of a full factorial design.

* **Primary Hypothesis ($H_{1C}$)**: According to the factor sparsity (Pareto / effect hierarchy) principle, at most $3$–$4$ of the $9$ hyperparameters account for $\ge 90\%$ of the explainable variance in validation RMSE and single-row latency. A Definitive Screening Design (DSD) or Resolution IV fractional factorial in Stage 1 accurately isolates these active factors within $\le 60$ fits, enabling a reduced-dimension $k_{\text{active}} \le 4$ local CCD in Stage 2 that matches or outperforms 9-dimensional TPE within a 180-fit total budget.

### 2. Nine-Factor Hyperparameter Domain
| Factor Symbol | Hyperparameter | Coded $-1$ | Coded $0$ | Coded $+1$ | Transformation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| $X_1$ | `max_depth` | $3$ | $6$ | $10$ | Integer |
| $X_2$ | `learning_rate` ($\eta$) | $0.02$ | $0.08$ | $0.20$ | Log-spaced |
| $X_3$ | `subsample` | $0.50$ | $0.75$ | $1.00$ | Linear |
| $X_4$ | `colsample_bytree` | $0.50$ | $0.75$ | $1.00$ | Linear |
| $X_5$ | `min_child_weight` | $1$ | $5$ | $20$ | Log-spaced |
| $X_6$ | `reg_lambda` ($L_2$) | $0.01$ | $1.0$ | $10.0$ | Log-spaced |
| $X_7$ | `reg_alpha` ($L_1$) | $0.0$ | $0.1$ | $5.0$ | Square-root |
| $X_8$ | `gamma` (`min_split_loss`) | $0.0$ | $0.25$ | $2.0$ | Square-root |
| $X_9$ | `colsample_bylevel` | $0.50$ | $0.75$ | $1.00$ | Linear |

### 3. Two-Stage Screening-to-RSM Protocol (180 Fits Across 3 Blocks)
1. **Stage 1 — Blocked Definitive Screening Design (DSD, 63 fits)**:
   * A standard 3-level Definitive Screening Design for $k = 9$ continuous factors requires $2k + 3 = 21$ distinct design runs per block. Unlike Resolution III fractional factorials, a DSD avoids confounding main effects with two-factor interactions or pure quadratic effects, and can estimate pure quadratic curvature for every factor directly in Stage 1.
   * Replicating the 21-run DSD across $B = 3$ validation-split blocks consumes $21 \times 3 = 63$ model fits.
2. **Active Factor Selection Gate (at Fit 63)**:
   * Apply hierarchical forward-stepwise regression with AICc / heredity constraints (or Lenth's pseudo-standard error method adjusted for block degrees of freedom) to rank all 9 main effects and their pure quadratic terms.
   * Fix the inactive factors ($k - k_{\text{active}}$ parameters with $p > 0.10$ and $< 2\%$ partial $\eta^2$) at their Stage 1 best-performing settings, and retain the top $k_{\text{active}} \in \{3, 4\}$ factors for Stage 2.
3. **Stage 2 — Reduced-Dimension Local Blocked CCD + Confirmation (117 fits)**:
   * Around the Stage 1 constrained optimum, deploy a $k_{\text{active}} = 4$ local FCCD ($25$ coordinates $\times 3$ blocks = 75 fits) followed by a 10-step desirability ridge refinement ($10 \times 3 = 30$ fits) and 12 confirmation runs ($63 + 75 + 30 + 12 = 180$ total fits).
   * Compare against 9-dimensional Constrained TPE, MO-TPE, and Latin Hypercube / Random Search under an identical 180-fit budget across $R = 20$ outer replicates.

---

## Summary Comparison of Proposed Protocols

| Protocol | Primary Audit Limitation Addressed | Tuned Factors ($k$) | Per-Search Fit Budget | Key Methodological Innovation |
| :--- | :--- | :---: | ---: | :--- |
| **Proposal A** | Single-batch pre-tabulated FCCD & wasted $\eta = 0.01$ budget | $4$ | $140$ fits ($55 + 15 + 70$) | True sequential Stage 1 screening + steepest descent re-centering before Stage 2 local CCD |
| **Proposal B** | Single dataset (`California Housing`) & fixed $M = 100$ horizon | $4$ | $140$ fits ($\times 6$ datasets $\times 3$ horizons) | Cross-dataset mixed-effects meta-regression linking noise-to-signal ratio to DOE vs. TPE advantage |
| **Proposal C** | $2^k$ factorial scaling bottleneck for high-$k$ tuning | $9$ | $180$ fits ($63$ DSD $+ 117$ local CCD) | 3-level Definitive Screening Design (DSD) isolating $3$–$4$ active factors prior to local RSM |
