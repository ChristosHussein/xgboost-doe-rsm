# Proposed Follow-Up Experiments (Work Package F / F.1 — Task 13 & Task 4)

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**
>
> Consistent with the scientific integrity constraints of Work Packages F and F.1, **no new XGBoost model fits have been executed** in this audit, and all 40 frozen experimental evidence files (`results/` and `config.yaml`, verified via `CANONICAL_EVIDENCE_SHA256` in `scripts/reporting_data.py` and `results/revision_v2/full_run_001/provenance.json`) remain untouched. The five pre-specified experimental protocols below address both the three originally requested follow-up studies—**(A) Constrained DOE/RSM optimization at the prespecified $145\,\mu\text{s}$ latency cap**, **(B) Matched-protocol DOE vs. TPE parity benchmarking**, and **(C) Truly independent external confirmation**—and the three methodological extensions (**Sequential Region-Zooming RSM**, **Multi-Dataset/Multi-Horizon Generalization**, and **9-Factor Definitive Screening**).

---

## Overview of Audit Findings Driving Follow-Up Design

The independent scientific audit (`docs/scientific_claims_audit.md`) identified five concrete methodological and scope boundaries in the historical (`v1.0.0`) and Revision-v2 (`results/revision_v2/full_run_001/`) campaigns:

1. **Soft Desirability Trade-Off vs. Hard $145\,\mu\text{s}$ Constraint Optimization**: In Phase 4 (`results/confirmation.json`) and Revision-v2 (`results/revision_v2/full_run_001/doe_selection_protocol.json`), Multi-Objective DOE used soft Derringer-Suich desirability over $Y_1 \in [0.450, 0.700]$ and $Y_2 \in [100.0, 180.0]\,\mu\text{s}$, selecting conservative depth-4/depth-5 models (holdout Test RMSE $0.48997 \pm 0.00732$ at $122.29 \pm 2.70\,\mu\text{s}$, $20/20$ feasible), whereas Constrained TPE targeted the hard $\le 145\,\mu\text{s}$ threshold using noisy 30-call un-warmed online timing (achieving Test RMSE $0.47036 \pm 0.00324$ at $144.27 \pm 7.06\,\mu\text{s}$, but dropping from $20/20$ search feasibility to $9/20 = 45\%$ benchmark feasibility because depth-7 selections averaged $150.45 \pm 0.67\,\mu\text{s}$ vs. depth-6 at $136.70 \pm 0.82\,\mu\text{s}$). A direct comparison of hard-constrained RSM against Constrained TPE under matched latency uncertainty buffers has not yet been executed.
2. **Protocol Asymmetry Between Blocked DOE and Single-Split TPE**: In Revision-v2, each Repeated DOE replicate $r \in \{0, \dots, 19\}$ evaluated 25 FCCD coordinates across 5 disjoint block seeds (`3001 + 5r` to `3005 + 5r`, spanning `3001..3100`) and selected winners from a predeclared 1,323-point candidate grid (`doe_selection_protocol.json`), whereas TPE and Random Search ran 140 trials on fixed development split seed `42` using Optuna sampler seeds (`1000 + r`, `2000 + r`, `3000 + r`, `4000 + r`). Consequently, replicate index $r$ was not a shared random data block (making unpaired Welch's $t$-test primary: $\Delta = -0.00258, t = -0.8849, p = 0.3826$), and no pre-registered equivalence margin (`rmse_equivalence_margin: null`) was defined.
3. **Prior Historical Exposure of the California Housing Holdout Split**: Although `Revision-v2` enforced strict programmatic isolation of the 4,128-row holdout test partition until `finalized_selections.json` was frozen, the same 80/20 split (`random_state = 42`) of California Housing was previously evaluated in historical `v1.0.0` (`results/runs.csv`, `results/benchmark.csv`).
4. **Single-Batch Pre-Planned FCCD vs. True Sequential Region Zooming**: All 140 FCCD runs (`16` corners, `8` axial points, `4` center replicates per block across 5 blocks) were generated upfront in a single pre-planned batch (`30/40` axial runs executed at `run_order <= 100`). Because `45/140` runs (`32.14%`) sat on the $\eta = 0.01$ face (mean validation RMSE `0.7545` vs. `0.5055` for $x_1 > -1.0$), global quadratic interpolation ($R^2 = 0.9951, \text{Adj } R^2 = 0.9944, \text{RMS}_{\text{LoF}} = 0.0297$) exhibited $+0.0217$ RMSE optimism bias at depth 7 ($0.4483$ predicted vs. $0.4700 \pm 0.0093$ empirical) and trailed unconstrained SO-TPE by $+0.00168$ Test RMSE ($0.46859$ vs. $0.46691$).
5. **Single-Dataset, Fixed-Horizon ($M=100$), and Four-Factor ($k=4$) Scope**: Generalizability beyond California Housing ($N = 20,640, d = 8$), `n_estimators = 100`, and $k = 4$ hyperparameters requires multi-dataset, multi-horizon, and Definitive Screening Design (DSD) extensions.

---

## Protocol A — Constrained DOE/RSM Optimization at the Prespecified $145\,\mu\text{s}$ Latency Cap

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective and Hypothesis
Directly test whether optimizing the fitted second-order RSM validation RMSE surface $\hat{Y}_1(\mathbf{x})$ subject to a **hard upper prediction-bound latency constraint** at $145\,\mu\text{s}$ matches the holdout Test RMSE of Constrained TPE ($0.47036$) while maintaining $\ge 95\%$ benchmark constraint feasibility (eliminating the $20/20 \to 9/20$ feasibility collapse observed in Constrained TPE).

* **Primary Hypothesis ($H_{1A}$)**: Solving the chance-constrained RSM program
  $$\min_{\mathbf{x} \in [-1, +1]^4,\; d \in \{3,\dots,9\}} \hat{Y}_1(\mathbf{x}) \quad \text{subject to} \quad \hat{Y}_2(\mathbf{x}) + t_{0.95,\, \nu_{\text{eff}}(\mathbf{x})} \, \text{SE}(\hat{Y}_2(\mathbf{x})) \le 145.0\,\mu\text{s}$$
  selects depth-6 interior coordinates ($\eta \in [0.14, 0.22], d = 6, s = 1.0$) that achieve lower holdout Test RMSE than soft Derringer-Suich desirability ($\mathbf{x}^*_{\text{MO}}$ at depth 4: $0.48997$) while preserving $\ge 19/20$ ($95\%$) 1,000-call benchmark feasibility.

### 2. Experimental Design (Zero Additional Search Fits Required on Existing Replicates + Prospective Warmed Protocol)
1. **Arm A1 — Post-Hoc Constrained RSM vs. Soft Desirability on the 20 Existing Blocked FCCD Replicates**:
   * Using the 20 fitted 140-run blocked FCCD surfaces from `doe_selection_runs.csv` (block seeds `3001..3100`), compare three selection rules over the 1,323-point grid and continuous SLSQP: (i) soft Derringer-Suich desirability ($\hat{Y}_1 \in [0.450, 0.700], \hat{Y}_2 \in [100, 180]\,\mu\text{s}$), (ii) nominal hard-constrained RSM ($\hat{Y}_2(\mathbf{x}) \le 145\,\mu\text{s}$), and (iii) upper-confidence-bound constrained RSM ($\hat{Y}_2(\mathbf{x}) + t_{0.95,\nu_{\text{eff}}}\text{SE}(\hat{Y}_2(\mathbf{x})) \le 145\,\mu\text{s}$).
2. **Arm A2 — Warmed Online Timing Protocol for Constrained TPE vs. Constrained RSM**:
   * Replace the un-warmed 30-call search timing protocol with a warmed 25-warmup + 100-timed-call `predict` protocol during search so online latency standard errors drop below $1.5\,\mu\text{s}$, testing whether Constrained TPE's depth-7 boundary violations ($150.45 \pm 0.67\,\mu\text{s}$) were caused by 30-call timing noise and lack of warmup.

---

## Protocol B — Matched-Protocol DOE vs. TPE Parity Benchmark

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective and Hypothesis
Eliminate all four protocol asymmetries between Blocked DOE and TPE identified in the audit (different development splits, continuous vs. 1,323-point grid candidate spaces, 30-call vs. 1,000-call timing, and unpaired replicate indices) and test formal statistical equivalence under a pre-registered Two One-Sided Tests (TOST) margin.

* **Pre-Registered Equivalence Margin**: $\Delta_{\text{TOST}} = \pm 0.0025$ Holdout Test RMSE (approximately $0.8\times$ the conditional retraining seed standard deviation $\sigma_{\text{eval}} \approx 0.00304$–$0.00358$ RMSE).

### 2. Matched Experimental Architecture ($R = 20$ Truly Paired Outer Blocks)
1. **Shared Outer Split & Block Structure**:
   * For each outer replicate $r \in \{0, \dots, 19\}$, define a common 5-split validation block quintuplet $\mathcal{B}_r = (S_{r,1}, \dots, S_{r,5})$ using seeds `6001 + 5r` through `6005 + 5r` (a fresh seed range distinct from historical `42, 101, 202, 303, 404`, confirmation `505..1414`, evaluation `2001..2020`, and Revision-v2 `3001..3100`).
   * **Matched Blocked DOE ($140$ fits)**: Evaluates the 25 FCCD coordinates across the 5 splits in $\mathcal{B}_r$ ($140$ fits).
   * **Matched 5-Block TPE ($28 \times 5 = 140$ fits)**: Evaluates 28 adaptive TPE configurations across the exact same 5 splits $\mathcal{B}_r$ ($28 \times 5 = 140$ fits), alongside **Single-Split TPE ($140 \times 1 = 140$ fits)** on $S_{r,1}$ to isolate the exact effect of 5-block replication vs. 140-coordinate exploration on the same outer replicate $r$.
2. **Unified Candidate Domain & Objective Function**:
   * Evaluate both continuous SLSQP and the 1,323-point grid for RSM, and run both Derringer-Suich desirability maximization and hard-constrained ($\le 145\,\mu\text{s}$) optimization for both DOE and TPE.
3. **Paired Inferential Analysis**:
   * Because outer replicate $r$ shares the exact same validation splits $\mathcal{B}_r$, paired $t$-tests ($\text{df} = 19$), Wilcoxon signed-rank tests, and TOST equivalence tests ($[-\Delta_{\text{TOST}}, +\Delta_{\text{TOST}}] = [-0.0025, +0.0025]$) are exact by construction.

---

## Protocol C — Truly Independent Confirmation and External Holdout Validation

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective
Provide confirmation and generalization evidence on data partitions that have **zero prior historical exposure** across `v1.0.0` or `Revision-v2`.

### 2. Two-Tier Independent Validation Design
1. **Tier C1 — Fresh Outer Holdout Partitions on California Housing**:
   * Draw $R = 20$ fresh outer 80/20 development/holdout partitions of California Housing using outer split seeds `7001..7020` (never used in any prior experiment, whereas historical `v1.0.0` and `Revision-v2` both used outer split seed `42` for the 80/20 development/holdout partition).
   * Lock all candidate configurations in a pre-committed manifest before running a single evaluation on the outer 20% holdout test fold.
2. **Tier C2 — Untouched External Tabular Regression Benchmarks**:
   * Execute the frozen pipeline end-to-end on untouched external tabular datasets (Miami Housing `OpenML 43093`, Diamonds `OpenML 42225`, Superconductivity `OpenML 43174`, and Elevators `OpenML 216`) where no exploratory tuning has ever been performed.

---

## Protocol D — True Two-Stage Sequential / Region-Zooming Response Surface Optimization

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Scientific Objective and Hypothesis
Evaluate whether a genuinely sequential two-stage Response Surface Methodology (RSM) workflow—using Stage 1 fractional screening and steepest descent to eliminate the degenerate $\eta = 0.01$ boundary (`45/140 = 32.14%` of global FCCD runs) and re-centering Stage 2 in the high-$\eta$ interior—eliminates the $+0.0217$ RMSE depth-7 polynomial optimism bias and closes the $+0.00168$ Test RMSE single-objective gap against SO-TPE under an **exact 140-fit budget**.

### 2. Exact Two-Stage Budget Arithmetic (Strictly 140 Model Fits per Search Replicate)
To maintain strict parity with the 140-fit budget of Revision-v2 while respecting exact factorial/CCD point counts, each search replicate $r \in \{0, \dots, 19\}$ allocates its **140 model fits** across three sequential stages:

| Stage | Design Structure | Distinct Coordinates | Blocks ($B$) | Fits per Block | Stage Fits | Cumulative Fits |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1A: Fractional Screening** | $2^{4-1}_{\text{IV}}$ ($I=ABCD$, 8 corners) + 2 center replicates | $9$ ($8$ corners + $1$ center) | $5$ | $8 + 2 = 10$ | $50$ | $50$ |
| **Stage 1B: Steepest-Descent Line Search** | 3 coded step sizes $\lambda \in \{0.5, 1.0, 1.5\}$ along constrained gradient | $3$ | $5$ | $3$ | $15$ | $65$ |
| **Stage 2: Re-Centered Local FCCD** | Full 4-factor local FCCD ($2^4 = 16$ corners + $2k = 8$ axial + $1$ center) | $25$ | $3$ | $16 + 8 + 1 = 25$ | $75$ | **$140$** |
| **Alternative 5-Block Stage 2 Variant** | *If 5 blocks are retained in Stage 2*: Stage 1B uses 2 steps ($2 \times 5 = 10$ fits, cum. $60$), and Stage 2 runs a reduced-axial local CCD ($2^{4-1}_{\text{IV}}$ $8$ corners + $2k_{\text{active}} = 6$ axial points on the 3 active factors + $2$ centers $= 16$ runs/block $\times 5$ blocks $= 80$ fits) | $15$ | $5$ | $8 + 6 + 2 = 16$ | $80$ ($+10$ Stage 1B) | **$140$** |

1. **Stage 1A — Blocked Fractional Screening + Center Curvature Check ($5 \times 10 = 50$ fits)**:
   * **Design**: $2^{4-1}_{\text{IV}}$ fractional factorial ($I = ABCD$, 8 corner coordinates) plus $n_C = 2$ center-point replicates per block across $B = 5$ validation-split blocks ($5 \times (8 + 2) = 50$ fits) over the original 4-factor domain (`learning_rate` $\eta \in [0.01, 0.30]$, `max_depth` $d \in [3, 9]$, `subsample` $s \in [0.50, 1.00]$, `reg_lambda` $\lambda \in [0.10, 10.00]$).
   * **Decision Rule at Fit 50**: Fit the first-order model with block indicators, test 1-df curvature against the 5-df within-block center pure error and 4-df paired block contrast, and compute the constrained steepest-descent direction $-\hat{\mathbf{g}} / \|\hat{\mathbf{g}}\|_2$ on validation RMSE subject to $\hat{Y}_2(\mathbf{x}) \le 145\,\mu\text{s}$.
2. **Stage 1B — Steepest-Descent Line Search ($5 \times 3 = 15$ fits; cumulative = 65 fits)**:
   * **Design**: Evaluate 3 candidate step sizes $\lambda \in \{0.5, 1.0, 1.5\}$ in coded units along the constrained steepest-descent ray across all $B = 5$ blocks ($3 \times 5 = 15$ fits).
   * **Re-Centering Rule at Fit 65**: Set the step size $\lambda^*$ minimizing block-mean validation RMSE as the new local center $\mathbf{x}_0^{(2)}$, shrinking the local trust-region half-widths to $\eta \in [0.06, 0.28]$ (strictly excluding the degenerate $\eta = 0.01$ boundary), $\Delta d = \pm 2$ integer levels (e.g., $d \in \{4, 6, 8\}$ or $\{5, 7, 9\}$), $\Delta s = \pm 0.15$, and $\Delta \ln(\lambda) = \pm 1.15$.
3. **Stage 2 — Re-Centered Local Face-Centered CCD Refinement ($3 \times 25 = 75$ fits; cumulative = 140 fits)**:
   * **Design**: Around $\mathbf{x}_0^{(2)}$, execute a full 25-coordinate local FCCD ($2^4 = 16$ factorial corners + $2k = 8$ axial star points at $\alpha = 1.0$ + $1$ center point $= 25$ runs/block) across the first $B_2 = 3$ validation-split blocks ($25 \times 3 = 75$ fits; $50 + 15 + 75 = 140$ total fits). (In the 5-block sensitivity variant, Stage 1B evaluates 2 step sizes for $10$ fits and Stage 2 evaluates $8$ corners $+ 6$ active-factor axial points $+ 2$ centers $= 16$ runs/block across all $5$ blocks for $80$ fits; $50 + 10 + 80 = 140$ total fits.)

---

## Protocol E — Multi-Dataset ($D=6$), Multi-Horizon ($M \in \{100, 300, 500\}$), and 9-Factor Definitive Screening Extensions

> **STATUS: PROPOSED FUTURE WORK — NOT EXECUTED IN CURRENT REPOSITORY**

### 1. Multi-Dataset and Multi-Horizon Generalization Matrix ($D = 6 \times 3$ Horizons)
Cross $D = 6$ standardized OpenML tabular regression benchmarks (`California Housing`, `Miami Housing OpenML 43093`, `Diamonds OpenML 42225`, `Superconductivity OpenML 43174`, `NYC Taxi Duration`, and `Elevators OpenML 216`) with three boosting horizons $M \in \{100, 300, 500\}$ (`n_estimators`) under latency caps scaled by $M/100$. Across the $6 \times 3 = 18$ regimes, compare Single-Batch Blocked FCCD, Two-Stage Sequential RSM (Protocol D), Constrained RSM (Protocol A), Constrained TPE, MO-TPE, and Random Search using Friedman omnibus rank tests and mixed-effects meta-regression against empirical split-noise ICC.

### 2. High-Dimensional 9-Factor Definitive Screening Design ($k = 9$, 180 Fits Across 3 Blocks)
To extend structured DOE from $k = 4$ to $k = 9$ XGBoost hyperparameters (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `gamma`, `colsample_bylevel`) without a $2^9 = 512$-corner full factorial:
1. **Stage 1 — Blocked Definitive Screening Design (DSD, $21 \times 3 = 63$ fits)**: A 3-level DSD for $k = 9$ factors requires $2k + 3 = 21$ runs per block, where main effects are orthogonal to two-factor interactions and pure quadratic effects are estimable in Stage 1. Replicating across $B = 3$ blocks uses $63$ fits.
2. **Active Factor Selection Gate (at Fit 63)**: Retain the top $k_{\text{active}} = 4$ factors via hierarchical AICc / Lenth's method and fix the remaining 5 inactive factors.
3. **Stage 2 — Reduced-Dimension Local Blocked FCCD + Ridge & Confirmation ($75 + 30 + 12 = 117$ fits; Total = 180 fits)**: Deploy a 25-point local FCCD on the 4 active factors across 3 blocks ($25 \times 3 = 75$ fits), a 10-step desirability ridge trace ($10 \times 3 = 30$ fits), and 12 confirmation runs ($63 + 75 + 30 + 12 = 180$ total fits), benchmarked against 9D TPE and Random Search at 180 fits.

---

## Summary Comparison of Proposed Follow-Up Protocols

| Protocol | Primary Audit Limitation Addressed | Tuned Factors ($k$) | Per-Search Fit Budget | Key Methodological Innovation |
| :--- | :--- | :---: | ---: | :--- |
| **Protocol A** | Soft desirability vs. hard $\le 145\,\mu\text{s}$ constraint & 30-call cTPE infeasibility | $4$ | $140$ fits | Upper-prediction-bound constrained RSM ($\hat{Y}_2 + t_{0.95}\text{SE}(\hat{Y}_2) \le 145\,\mu\text{s}$) + warmed search timing |
| **Protocol B** | Split/grid/timing asymmetry & unpaired replicate indices between DOE and TPE | $4$ | $140$ fits | Shared 5-block outer splits (`6001..6100`), unified grid/continuous domain, pre-registered TOST ($\pm 0.0025$ RMSE) |
| **Protocol C** | Historical exposure of California Housing split `42` | $4$ | $140$ fits | Fresh outer holdout splits (`7001..7020`) + untouched external OpenML regression datasets |
| **Protocol D** | Pre-planned single-batch FCCD & `45/140` runs wasted at $\eta = 0.01$ | $4$ | $140$ fits ($50 + 15 + 75$) | True sequential Stage 1A $2^{4-1}_{\text{IV}}$ screening + Stage 1B steepest descent + Stage 2 local 25-pt FCCD |
| **Protocol E** | Single dataset, fixed $M = 100$ trees, and $k = 4$ dimensionality | $4$ & $9$ | $140$ / $180$ fits | $6 \times 3$ dataset-horizon matrix + 9-factor Definitive Screening Design ($63$ DSD $+ 75$ local FCCD $+ 42$ ridge/conf) |
