# Pre-Planned Response Surface Methodology and Face-Centered Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking

**Author:** Christos Chousein Sounios

---

## Executive Summary

<!-- BEGIN AUTO-GENERATED: EXEC_SUMMARY -->
This report presents a Design of Experiments (DOE) and Response Surface Methodology (RSM) study applied to the multi-objective hyperparameter optimization of an **XGBoost Regressor** on the **California Housing** benchmark dataset ($N = 20,640$ observations, $8$ continuous features). Following the methodology of Douglas C. Montgomery's *Design and Analysis of Experiments* (9th ed., 2017, Chapters 4, 5, 6, 7, 10, 11, and 13), we treat stochastic machine learning variability—arising from random train/validation partitioning and row subsampling—as a **nuisance factor** controlled via a **Randomized Complete Block Design (RCBD)** across $b = 5$ seed blocks ($\mathcal{S} = \{42, 101, 202, 303, 404\}$).

Rather than an adaptive sequential search, the historical DOE campaign executes a **pre-planned, globally randomized 140-run Face-Centered Central Composite Design (FCCD)** analyzed in two hierarchical stages ($100$ factorial-plus-center runs in Phase 1 and $40$ axial runs in Phase 2, with $30$ of the $40$ axial runs occurring at `run_order <= 100`), followed by **10 confirmation trials** in Phase 5 for $\mathbf{x}^*_{\text{MO}}$ and **10 confirmation trials** for $\mathbf{x}^*_{\text{SO}}$, alongside a prospective 20-replicate benchmark campaign comprising **17,077 XGBoost model fits** and **1,640,810 timed single-sample inferences** (1,842,080 total single-row prediction calls including 201,270 warmup calls):

1. **Holdout Isolation and Dataset History**: The 20,640 dataset observations are partitioned into an 80% development pool ($N_{\text{dev}} = 16,512$) and a 20% external holdout test set ($N_{\text{test}} = 4,128$). Within each block $b$, the development pool is split 75/25 into training ($N_{\text{train}} = 12,384$) and validation ($N_{\text{val}} = 4,128$) sets. Response $Y_1$ is **Development Validation RMSE** on the 25% validation split. Whereas the historical `v1.0.0` baseline experiments recorded holdout test results during development, the revised `Revision-v2` pipeline programmatically prevented access to those test labels during all search iterations, surrogate fitting, curvature testing, lack-of-fit analysis, and desirability optimization until winning configurations were frozen in `finalized_selections.json`. These safeguards enforce programmatic pipeline isolation during the revised search, although they cannot retroactively undo earlier historical exposure of the same dataset.
2. **Phase 1 ($2^4$ Full Factorial + Center Points in 5 Blocks, $N_1 = 100$ runs)**: Screening identifies **Learning Rate** ($\ln \eta$, $F = 357.40, p < 10^{-15}, \eta^2_p = 0.8079$) and **Max Tree Depth** ($F = 23.57, p < 0.0001, \eta^2_p = 0.2171$), along with their interaction $x_1 x_2$ ($F = 26.32, p < 0.0001, \eta^2_p = 0.2364$), as the dominant drivers of validation RMSE. Single-degree-of-freedom curvature testing against within-block center-point pure error ($\text{df} = 15, \text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$) yields $F_{\text{Curv}} = 105,432.31$ ($p < 10^{-15}$), where the factorial corner mean ($\bar{y}_F = 0.6333$) exceeds the center mean ($\bar{y}_C = 0.4989$) by $0.1343$ RMSE ($+26.93\%$). Because within-block center replicates vary only row-subsampling seeds at fixed split $S_b$, sensitivity checks across six broader error denominators—including block-adjusted pooled coordinate error ($F = 20,235.41, \text{df} = 79$), Treatment $\times$ Block interaction ($F = 17,013.24, \text{df} = 64$), 5-block paired contrast ($t = 85.69, F = 7,342.69, \text{df} = 4, p = 1.11 \times 10^{-7}$), and unadjusted first-order+2FI model residual ($F = 82.86, \text{df} = 85, p = 3.34 \times 10^{-14}$)—confirm that quadratic curvature is robust to error-term specification while establishing that rejecting $H_0: \sum \beta_{ii} = 0$ detects severe boundary curvature rather than validating global quadratic adequacy.
3. **Phase 2 (Face-Centered Central Composite Design, $\alpha = 1.0$, $N_{\text{CCD}} = 140$ runs)**: Including the $40$ axial runs fits a full second-order polynomial ($R^2 = 0.9951, \text{Adj } R^2 = 0.9944$). Once quadratic curvature is modeled, the RCBD block effect becomes highly significant ($F = 20.21, p < 0.0001$), absorbing **ICC = 40.69%** ($\text{ICC}_{\text{REML}} = 0.4069, \sigma_{\text{block}} \approx 0.0077$ RMSE) of unexplained residual variance. Formal lack-of-fit decomposition shows statistically significant structural polynomial misfit ($F_{\text{LoF}} = 62.21, p = 7.78 \times 10^{-41}$ against the 111-df saturated additive baseline; $F_{\text{LoF}} = 323.10, p < 10^{-15}$ against 15-df center pure error; RMS misfit $= 0.0297$ RMSE), demonstrating that second-order polynomials serve as local guidance maps rather than globally exact estimators of gradient boosting loss.
4. **Phase 3 (Canonical & Ridge Analysis)**: Spectral decomposition of $\hat{\mathbf{B}}$ yields three positive eigenvalues and one near-zero eigenvalue ($\lambda = \{0.000057, 0.000873, 0.022759, 0.110196\}$; wild bootstrap 95% CI for $\lambda_1$: $[-0.0039, +0.0022]$, with $68.2\%$ of resamples $\le 0$), characterizing a **stationary/rising ridge system**. The unconstrained stationary point ($\|\mathbf{x}_0\| = 17.16$) lies outside $[-1, +1]^4$; constrained optimization over valid integer depths identifies the single-objective candidate $\mathbf{x}^*_{\text{SO}}$ at depth $7$ ($x_1 = 0.5983, x_2 = 0.3333, x_3 = 1.0000, x_4 = 1.0000$, i.e., $\eta = 0.1515, \text{depth} = 7, \text{subsample} = 1.0000, \lambda = 10.0000$) with predicted validation RMSE $\hat{y} = 0.4483$.
5. **Phase 4 (Derringer-Suich Multi-Objective Desirability)**: Balancing Validation RMSE ($Y_1 \in [0.450, 0.700]$) and Single-Sample Inference Latency ($Y_2 \in [100.0, 180.0]\,\mu\text{s}$) yields an interior compromise coordinate $\mathbf{x}^*_{\text{MO}} = [0.8500, -0.6667, 1.0000, -0.0812]^T$ ($\eta = 0.2324, \text{depth} = 4, \text{subsample} = 1.0000, \lambda = 0.8295$), achieving composite desirability **$D = 0.6782$** ($d_1 = 0.8783, d_2 = 0.5236$) with predicted validation RMSE $\hat{Y}_1 = 0.4804$ and predicted latency $\hat{Y}_2 = 138.11\,\mu\text{s}$.
6. **Phase 5 (Confirmation & Comparative Benchmarks)**: Across **10 confirmation trials** ($m = 10$ fresh seeds $\mathcal{S}_{\text{conf}} = \{505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414\}$), $\mathbf{x}^*_{\text{MO}}$ achieves empirical validation RMSE $0.4853 \pm 0.0115$ (falling inside the Satterthwaite 95% prediction interval $[0.4677, 0.4932]$), empirical latency $142.33 \pm 7.79\,\mu\text{s}$ ($142.3 \pm 7.8\,\mu\text{s}$, inside $[122.82, 153.40]\,\mu\text{s}$), and external holdout test RMSE $0.4888 \pm 0.0032$. At the single-objective depth-7 candidate $\mathbf{x}^*_{\text{SO}}$, empirical validation RMSE is $0.4700 \pm 0.0093$—lying $0.0088$ above the Satterthwaite 95% PI $[0.4354, 0.4612]$ due to $+0.0217$ polynomial optimism bias—while achieving external holdout test RMSE $0.4691 \pm 0.0034$ and borderline latency $171.1 \pm 10.9\,\mu\text{s}$ (at the lower bound of $[171.1, 202.9]\,\mu\text{s}$). In the 20-replicate prospective benchmark campaign, Single-Objective TPE achieves the lowest observed test RMSE ($0.46691 \pm 0.00116$) at high latency ($188.93 \pm 12.07\,\mu\text{s}$), whereas Repeated DOE Single-Objective yields deterministic boundary selection ($0.46859 \pm 0.00000$, retraining $\sigma_{\text{eval}} = 0.00304$) at $20.2\%$ lower latency ($150.74 \pm 0.72\,\mu\text{s}$). In multi-objective optimization, Repeated DOE ($\mathbf{x}^*_{\text{MO}}$) and Multi-Objective TPE show no statistically significant difference in holdout test RMSE ($0.48997 \pm 0.00732$ vs. $0.49255 \pm 0.01080$, difference $-0.00258$, primary Welch $p = 0.3826$) at comparable latency ($122.29 \pm 2.70\,\mu\text{s}$ vs. $121.46 \pm 3.75\,\mu\text{s}$) and $100\%$ ($20/20$) constraint feasibility.
<!-- END AUTO-GENERATED: EXEC_SUMMARY -->

---

## 1. Experimental Factors, Coding, and Blocking Architecture

<!-- BEGIN AUTO-GENERATED: SECTION_1_FACTORS_AND_BLOCKS -->
### 1.1 Hyperparameter Space & Natural-to-Coded Transformations

We investigate $k = 4$ hyperparameters of an XGBoost Regressor (`n_estimators = 100`, `objective = 'reg:squarederror'`, `n_jobs = 1` for inference). Factors spanning orders of magnitude ($x_1$ and $x_4$) are mapped via natural logarithmic transformations prior to linear coding into $[-1, +1]$:

| Coded Factor | Parameter Name | Symbol | Scale Transformation | Low ($-1$) | Center ($0$) | High ($+1$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$x_1$ (Factor A)** | `learning_rate` | $\eta$ | $\xi_1 = \ln(\eta)$ | $0.0100$ | $0.0548$ | $0.3000$ |
| **$x_2$ (Factor B)** | `max_depth` | $d$ | Linear (Integer) | $3$ | $6$ | $9$ |
| **$x_3$ (Factor C)** | `subsample` | $s$ | Linear | $0.5000$ | $0.7500$ | $1.0000$ |
| **$x_4$ (Factor D)** | `reg_lambda` | $\lambda$ | $\xi_4 = \ln(\lambda)$ | $0.1000$ | $1.0000$ | $10.0000$ |

The dimensionless coded variables $x_i \in [-1, +1]$ are defined by:
$$x_1 = \frac{\ln(\eta) - (-2.9046)}{1.7006}, \quad x_2 = \frac{d - 6}{3}, \quad x_3 = \frac{s - 0.75}{0.25}, \quad x_4 = \frac{\ln(\lambda) - 0}{2.3026}$$

### 1.2 Nuisance Blocking (RCBD) and Dual Responses

Each of the $b = 5$ blocks ($\mathcal{S} = \{42, 101, 202, 303, 404\}$) defines a deterministic 75/25 train/validation split of the 16,512-sample development pool via split seed $S_b$. For factorial corner points ($1 \dots 16$) and axial points ($18 \dots 25$), the XGBoost model seed equals $S_b$. For the $n_C = 4$ center-point replicates within each block, the train/validation split is held fixed at $S_b$ while the XGBoost subsampling seed is varied as $S_b + r \times 1000$ ($r \in \{0, 1, 2, 3\}$), isolating **15 degrees of freedom of within-block subsampling pure error** from **4 degrees of freedom of block-to-block data-split variance**.

Two response variables are recorded per run:
- **Response $Y_1$ (Development Validation RMSE)**: Root Mean Squared Error on the 25% validation partition ($N_{\text{val}} = 4,128$). Unlike the historical `v1.0.0` pipeline, which logged holdout test metrics during development, the revised pipeline prevented access to the 20% external holdout test partition ($N_{\text{test}} = 4,128$) until candidate configurations were frozen in `finalized_selections.json`.
- **Response $Y_2$ (Single-Sample Inference Latency, $\mu\text{s}$)**: Mean execution time in microseconds per single-row prediction on CPU Core 0 (`SetProcessAffinityMask = 1`, `n_jobs = 1`), timed using `time.perf_counter_ns()`.

### 1.3 Pre-Planned Design Chronology, Evaluation Budget Allocation, and Timing Protocols

- **Pre-Planned Chronology vs. Hierarchical Analysis**: In `pipeline.py`, all $28 \times 5 = 140$ runs (comprising $16$ factorial corners, $4$ center-point replicates, and $8$ axial points per block) were constructed upfront and globally shuffled via `rng = np.random.default_rng(42)` prior to execution (`run_order` $1 \dots 140$). Consequently, **$30$ of the $40$ `Phase2_Axial` runs were executed at `run_order <= 100`**, interleaved with `Phase1_Factorial` runs. The designation of Phase 1 ($N_1 = 100$) and Phase 2 ($N_{\text{CCD}} = 140$) therefore represents a **two-stage hierarchical statistical analysis** of a pre-planned Face-Centered CCD rather than an adaptive sequential experiment conditioned on intermediate observations.
- **Fixed-Coordinate Replication vs. Adaptive Search Budget**: Under the matched 140-evaluation budget, the blocked FCCD evaluates **25 distinct hyperparameter coordinates** replicated across $5$ seed blocks ($25 \times 5 = 125$ coordinate-block evaluations plus $15$ additional within-block center replicates $= 140$ fits). Because a standard FCCD allocates $8$ factorial corners and $1$ axial point per block to the lower learning-rate boundary $x_1 = -1.0$ ($\eta = 0.01$), exactly **$45/140$ runs ($32.14\%$) evaluate $\eta = 0.01$**, where 100-tree ensembles severely underfit (mean validation RMSE $0.7545$ vs. $0.5055$ for $x_1 > -1.0$). By contrast, adaptive Bayesian optimizers (SO-TPE, MO-TPE, Constrained TPE) and Random Search evaluate **140 distinct hyperparameter configurations** on a single split without cross-block replication, trading variance-decomposition power and split-blocking robustness for 5.6-fold denser coordinate exploration.
- **Latency Measurement Protocols and Interface Overhead (`predict` vs. `inplace_predict`)**: Latency measurements follow three distinct protocol tiers that must not be conflated: (i) **Historical DOE & Confirmation (`results/runs.csv`, `results/confirmation_runs*.csv`, `results/benchmark.csv`)**: 50 warmup calls per interface followed by 1,000 timed single-row calls per model fit; (ii) **Prospective Online Search (`optimizer_trials.csv`, `doe_selection_runs.csv`)**: 0 warmup calls and 30 timed `predict` calls per candidate fit to bound search wall-clock time; and (iii) **Prospective Holdout Evaluation (`final_evaluations.csv`)**: 25 warmup calls per interface (50 total) followed by 500 timed single-row calls per model across 2 independent timing sessions ($1,000$ timed calls/seed). Across all four empirical datasets, high-level `Booster.predict(DMatrix)` incurs a consistent data-marshaling overhead relative to `XGBRegressor.inplace_predict` on raw NumPy arrays: **$40.88 \pm 11.56\,\mu\text{s}$** (range $5.75$–$89.33\,\mu\text{s}$) across the 140 historical DOE runs, **$38.18 \pm 5.24\,\mu\text{s}$** ($\mathbf{x}^*_{\text{MO}}$) and **$36.85 \pm 7.75\,\mu\text{s}$** ($\mathbf{x}^*_{\text{SO}}$) in historical confirmation, **$32.07 \pm 1.16\,\mu\text{s}$** (range $30.89$–$33.96\,\mu\text{s}$) in the 5-method historical benchmark, and **$32.19 \pm 1.46\,\mu\text{s}$** (range $27.20$–$37.43\,\mu\text{s}$) across the 50-configuration Revision-v2 interface audit (`latency_interface_overhead.json`).
<!-- END AUTO-GENERATED: SECTION_1_FACTORS_AND_BLOCKS -->

---

## 2. Phase 1: $2^4$ Factorial Screening and Curvature Test

<!-- BEGIN AUTO-GENERATED: SECTION_2_PHASE1 -->
### 2.1 Phase 1 ANOVA (Type III Sum of Squares)

Phase 1 analyzes the $2^4 = 16$ factorial corners plus $n_C = 4$ center replicates across $b = 5$ blocks ($N_1 = 100$ runs). Fitting the first-order model with two-factor interactions yields:

| Source of Variation | Sum of Squares (SS) | DF | Mean Square (MS) | $F$-Value | $p$-Value | Partial $\eta^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | $7.309127$ | $1$ | $7.309127$ | $2097.15$ | $< 10^{-15}$ | $0.9610$ |
| **Block Effect $C(\text{block})$** | $0.004662$ | $4$ | $0.001166$ | $0.33$ | $0.8541$ | $0.0155$ |
| **Factor A: $x_1$ ($\ln \eta$)** | $1.245625$ | $1$ | $1.245625$ | $357.40$ | $< 10^{-15}$ | $0.8079$ |
| **Factor B: $x_2$ (`max_depth`)** | $0.082155$ | $1$ | $0.082155$ | $23.57$ | $0.0000$ | $0.2171$ |
| **Factor C: $x_3$ (`subsample`)** | $0.002406$ | $1$ | $0.002406$ | $0.69$ | $0.4084$ | $0.0081$ |
| **Factor D: $x_4$ ($\ln \lambda$)** | $0.000089$ | $1$ | $0.000089$ | $0.03$ | $0.8731$ | $0.0003$ |
| **Interaction $x_1 \cdot x_2$** | $0.091725$ | $1$ | $0.091725$ | $26.32$ | $0.0000$ | $0.2364$ |
| **Interaction $x_1 \cdot x_3$** | $0.001878$ | $1$ | $0.001878$ | $0.54$ | $0.4649$ | $0.0063$ |
| **Interaction $x_1 \cdot x_4$** | $0.007452$ | $1$ | $0.007452$ | $2.14$ | $0.1474$ | $0.0245$ |
| **Interaction $x_2 \cdot x_3$** | $0.001106$ | $1$ | $0.001106$ | $0.32$ | $0.5747$ | $0.0037$ |
| **Interaction $x_2 \cdot x_4$** | $0.000092$ | $1$ | $0.000092$ | $0.03$ | $0.8714$ | $0.0003$ |
| **Interaction $x_3 \cdot x_4$** | $0.000081$ | $1$ | $0.000081$ | $0.02$ | $0.8790$ | $0.0003$ |
| **Residual Error** | $0.296247$ | $85$ | $0.003485$ | — | — | — |

### 2.2 Single-Degree-of-Freedom Curvature Test

To test $H_0: \sum_{i=1}^4 \beta_{ii} = 0$, we compare the mean validation RMSE of the $n_F = 80$ factorial corner runs against the $n_C = 20$ center-point runs against within-block center-point pure error ($\text{df} = 15$):
- **Factorial Mean ($\bar{y}_F$)**: $0.6333$
- **Center Point Mean ($\bar{y}_C$)**: $0.4989$
- **Curvature Contrast ($\bar{y}_F - \bar{y}_C$)**: $+0.1343$ RMSE ($+26.93\%$ relative to $\bar{y}_C$; block-paired 95% CI $[+0.1300, +0.1387]$)
- **Curvature Sum of Squares**: $\text{SS}_{\text{Curv}} = \frac{n_F n_C}{n_F + n_C}(\bar{y}_F - \bar{y}_C)^2 = 0.288786$
- **Within-Block Center Pure Error**: $\text{SS}_{\text{PE, center}} = 0.000041$, $\text{df}_{\text{PE}} = 15$, $\text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$
- **Curvature $F$-Statistic**:
$$F_{\text{Curv}} = \frac{\text{SS}_{\text{Curv}}}{\text{MS}_{\text{PE, center}}} = \frac{0.288786}{2.74 \times 10^{-6}} = 105,432.31 \quad (p < 10^{-15})$$

### 2.3 Sensitivity of Curvature Test Across Alternative Error Denominators

Because the 4 center replicates per block hold the 75/25 data split fixed at $S_b$ and vary only the row-subsampling seed ($s = 0.75$), $\text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$ ($\text{SD} \approx 0.00166$ RMSE) measures micro-scale algorithmic subsampling variability rather than cross-split interaction or higher-order model error. Comparing macro-scale hyperparameter contrast ($\text{SS}_{\text{Curv}} = 0.288786$, driven by underfitting at $\eta = 0.01$) against micro-scale subsampling noise mechanically produces an $F$-ratio of $O(10^5)$. To verify that curvature detection does not depend on this narrow denominator, the table below evaluates $\text{SS}_{\text{Curv}}$ across six increasingly conservative error structures:

| Error Denominator Specification | Error DF | Error SS | Error MS | Curvature $F$-Statistic | $p$-Value | Variance Scope Isolated |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Within-Block Center Pure Error ($\text{MS}_{\text{PE, center}}$)** | $15$ | $0.000041$ | $2.74 \times 10^{-6}$ | $105,432.31$ | $< 10^{-15}$ | Subsampling seed noise at fixed split $S_b$ |
| **2. Block-Adjusted Pooled Coordinate Error** | $79$ | $0.001127$ | $1.43 \times 10^{-5}$ | $20,235.41$ | $< 10^{-15}$ | Center PE ($15$ df) + Treatment $\times$ Block ($64$ df) |
| **3. Phase 1 Treatment $\times$ Block Interaction** | $64$ | $0.001086$ | $1.70 \times 10^{-5}$ | $17,013.24$ | $< 10^{-15}$ | Coordinate-by-split interaction across $17 \times 5$ cells |
| **4. Block-Paired Contrast ($\bar{y}_{F,b} - \bar{y}_{C,b}$ across 5 Blocks)** | $4$ | — | $\text{SE} = 0.0016$ | $7,342.69$ ($t = 85.69$) | $1.11 \times 10^{-7}$ | Direct across-block pairing ($95\%$ CI $[+0.1300, +0.1387]$) |
| **5. Unadjusted Coordinate Pure Error ($\text{MS}_{\text{PE, All}}$)** | $83$ | $0.005790$ | $6.98 \times 10^{-5}$ | $4,140.06$ ($\approx 4,140$) | $< 10^{-15}$ | Pooled within-coordinate variance (includes block main effect) |
| **6. First-Order + 2FI + Curvature Residual** | $84$ | $0.007461$ | $8.88 \times 10^{-5}$ | $3,251.19$ | $< 10^{-15}$ | Higher-order factorial interactions ($3$FI+$4$FI) + TxB + PE |
| **7. Unadjusted First-Order + 2FI Residual ($\text{MS}_E$)** | $85$ | $0.296247$ | $0.003485$ | $82.86$ | $3.34 \times 10^{-14}$ | Total Phase 1 residual before separating curvature |

*Methodological Distinction*: Rejecting $H_0: \sum_{{i=1}}^4 \beta_{{ii}} = 0$ confirms that a flat first-order plane is inadequate over $\eta \in [0.01, 0.30]$; it does **not** imply that a second-order polynomial is free of higher-order misfit. Surrogate predictive adequacy is evaluated separately via Phase 2 lack-of-fit testing (Section 3.3) and Phase 5 confirmation intervals (Section 6).
<!-- END AUTO-GENERATED: SECTION_2_PHASE1 -->

---

## 3. Phase 2: Face-Centered Central Composite Design (FCCD) and Model Adequacy

<!-- BEGIN AUTO-GENERATED: SECTION_3_PHASE2 -->
### 3.1 Second-Order Response Surface ANOVA ($Y_1$: Validation RMSE and $Y_2$: Inference Latency)

Incorporating the $2k = 8$ axial points ($\alpha = 1.0$) across 5 blocks yields $N = 140$ runs ($28$ runs/block across $25$ distinct coordinates). The fitted second-order validation RMSE model achieves $R^2 = 0.9951$ and $\text{Adjusted } R^2 = 0.9944$:

| Source of Variation | SS | DF | MS | $F$-Stat | OLS $p$ | HC3 SE | HC3 $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | $4.718852$ | $1$ | $4.718852$ | $54749.42$ | $< 10^{-15}$ | $0.0021$ | $< 10^{-15}$ |
| **Block Effect $C(\text{block})$** | $0.006966$ | $4$ | $0.001742$ | $20.21$ | $< 0.0001$ | — | — |
| **Factor A: $x_1$ ($\ln \eta$)** | $1.388673$ | $1$ | $1.388673$ | $16111.76$ | $< 10^{-15}$ | $0.0012$ | $< 10^{-15}$ |
| **Factor B: $x_2$ (Depth)** | $0.101706$ | $1$ | $0.101706$ | $1180.02$ | $< 10^{-15}$ | $0.0013$ | $< 10^{-15}$ |
| **Factor C: $x_3$ (Subsample)** | $0.002013$ | $1$ | $0.002013$ | $23.36$ | $< 0.0001$ | $0.0012$ | $0.0001$ |
| **Factor D: $x_4$ ($\ln \lambda$)** | $0.000044$ | $1$ | $0.000044$ | $0.52$ | $0.4743$ | $0.0012$ | $0.5483$ |
| **Quadratic $x_1^2$** | $0.146265$ | $1$ | $0.146265$ | $1697.01$ | $< 10^{-15}$ | $0.0023$ | $< 10^{-15}$ |
| **Quadratic $x_2^2$** | $0.008779$ | $1$ | $0.008779$ | $101.85$ | $< 10^{-15}$ | $0.0035$ | $< 0.0001$ |
| **Quadratic $x_3^2$** | $0.000005$ | $1$ | $0.000005$ | $0.05$ | $0.8188$ | $0.0023$ | $0.8005$ |
| **Quadratic $x_4^2$** | $0.000006$ | $1$ | $0.000006$ | $0.08$ | $0.7844$ | $0.0020$ | $0.7197$ |
| **Interaction $x_1 \cdot x_2$** | $0.091725$ | $1$ | $0.091725$ | $1064.22$ | $< 10^{-15}$ | $0.0013$ | $< 10^{-15}$ |
| **Interaction $x_1 \cdot x_3$** | $0.001878$ | $1$ | $0.001878$ | $21.79$ | $< 0.0001$ | $0.0013$ | $0.0003$ |
| **Interaction $x_1 \cdot x_4$** | $0.007452$ | $1$ | $0.007452$ | $86.46$ | $< 10^{-15}$ | $0.0013$ | $< 0.0001$ |
| **Interaction $x_2 \cdot x_3$** | $0.001106$ | $1$ | $0.001106$ | $12.83$ | $0.0005$ | $0.0013$ | $0.0050$ |
| **Interaction $x_2 \cdot x_4$** | $0.000092$ | $1$ | $0.000092$ | $1.07$ | $0.3040$ | $0.0013$ | $0.4118$ |
| **Interaction $x_3 \cdot x_4$** | $0.000081$ | $1$ | $0.000081$ | $0.94$ | $0.3334$ | $0.0013$ | $0.4400$ |
| **Residual Error** | $0.010429$ | $121$ | $0.000086$ | — | — | — | — |

For **Response $Y_2$ (Inference Latency, $\mu\text{s/sample}$)**, the second-order ANOVA shows that tree depth ($x_2$, $F = 811.46, p < 10^{-15}$) and its quadratic term ($x_2^2$, $F = 39.86, p < 0.0001$) govern inference latency, while block effects are negligible ($F = 0.75, p = 0.5591, \text{ICC} = 0.0000$).

### 3.2 Variance Shielding and Block Intraclass Correlation (ICC)

In Phase 1, unmodeled quadratic curvature inflated the residual mean square ($\text{MS}_E = 0.003485$), masking block differences ($F = 0.33, p = 0.8541$). In Phase 2, accounting for quadratic terms reduces $\text{MS}_E$ by $40\times$ to $0.000086$, exposing highly significant seed-to-seed variance ($F = 20.21, p < 0.0001$). The **Block Intraclass Correlation Coefficient** is:
$$\text{ICC} = \frac{\sigma^2_{\text{block}}}{\sigma^2_{\text{block}} + \sigma^2_\epsilon} = 0.4069 \quad (40.69\%, \quad \text{ICC}_{\text{REML}} = 0.4069, \quad \sigma_{\text{block}} \approx 0.0077\text{ RMSE})$$

### 3.3 Exact Multi-Scale Lack-of-Fit Decomposition and Residual Diagnostics

Partitioning the $121$ residual degrees of freedom into **Structural Lack of Fit** ($10\text{ df}$), **Treatment $\times$ Block Interaction** ($96\text{ df}$), and **Genuine Center Pure Error** ($15\text{ df}$) yields:

| Source of Variation / Model | SS | DF | MS | $F$ | $p$-Value | Reference / RMS Misfit |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Decomposition of Second-Order Residual ($Y_1$: Validation RMSE)** | | | | | | |
| $\quad$ Structural Lack of Fit | $0.008850$ | $10$ | $0.000885$ | $323.10$ | $< 10^{-15}$ | vs. Center PE |
| $\quad$ Treatment $\times$ Block Interaction | $0.001538$ | $96$ | $1.60 \times 10^{-5}$ | $5.85$ | $0.0002$ | vs. Center PE |
| $\quad$ Genuine Center Pure Error | $0.000041$ | $15$ | $2.74 \times 10^{-6}$ | — | — | Base Replicate |
| $\quad$ Total Model Residual | $0.010429$ | $121$ | $8.62 \times 10^{-5}$ | — | — | $\text{RMS} = 0.0093$ |
| $\quad$ Saturated Additive Baseline | $0.001579$ | $111$ | $1.42 \times 10^{-5}$ | $62.21^*$ | $< 10^{-15}$ | ($^*$LoF vs. Additive, $p = 7.78 \times 10^{-41}$, $\text{RMS}_{\text{LoF}} = 0.0297$) |
| **Restricted Domain ($Y_1$: $x_1 \ge -0.5$, $\eta \ge 0.033$, $N=95$)** | | | | | | |
| $\quad$ Structural Lack of Fit | $0.000324$ | $2$ | $0.000162$ | $10.11$ | $0.0001$ | $\text{RMS} = 0.0127$ ($96.3\%$ SS reduction) |
| $\quad$ Additive Baseline Residual | $0.001203$ | $75$ | $1.60 \times 10^{-5}$ | — | — | Saturated Base |
| $\quad$ Total Restricted Residual | $0.001528$ | $77$ | $1.98 \times 10^{-5}$ | — | — | $\text{RMS} = 0.0045$ |
| **Latency Residual Decomposition ($Y_2$: Latency, $\mu\text{s}$)** | | | | | | |
| $\quad$ Structural Lack of Fit | $3322.37$ | $10$ | $332.24$ | $1.20$ | $0.3625$ | vs. Center PE ($\text{RMS} = 18.23$) |
| $\quad$ Treatment $\times$ Block Interaction | $27844.93$ | $96$ | $290.05$ | $1.05$ | $0.4906$ | vs. Center PE |
| $\quad$ Genuine Center Pure Error | $4148.89$ | $15$ | $276.59$ | — | — | Base Replicate |
| $\quad$ Total Model Residual | $35316.19$ | $121$ | $291.87$ | $1.15^*$ | $0.3305$ | ($^*$LoF vs. Additive) |

Residual adequacy diagnostics for $Y_1$ (explicitly distinguishing grouping factors):
- **Normality**: Shapiro-Wilk $W = 0.9971, p = 0.9947$ on the 140 second-order OLS residuals (consistent with Gaussian residuals once polynomial terms are fitted).
- **Homoscedasticity Across Grouping Factors vs. Fitted Values**: Mean-centered Levene test across the **5 seed blocks** (`block` $1 \dots 5$) yields $W = 0.1754, p = 0.9507$, confirming equal residual variance across data-split blocks. Median-centered Levene (Brown-Forsythe) test across the **25 distinct design coordinates** yields $W = 0.6872, p = 0.8552$, because within each coordinate the 5 block residuals share similar spread after block adjustment. The Breusch-Pagan test against continuous fitted values $\hat{y}$ provides evidence of heteroscedasticity ($\text{LM} = 70.10, p < 0.0001$), because deterministic cell-mean polynomial misfit ($\bar{y}_{i\cdot} - \hat{y}_i$) is concentrated at the high-RMSE $\eta = 0.01$ boundary rather than uniform across $\hat{y}$. HC3 heteroscedasticity-consistent standard errors are used separately for coefficient inference and do not eliminate model misspecification.
- **Independence & Influence**: Durbin-Watson $DW = 1.9918$, Ljung-Box $Q = 4.01, p = 0.5486$, Runs test $p = 0.8576$, maximum Cook's distance $D_{\max} = 0.097 < 1.0$.
<!-- END AUTO-GENERATED: SECTION_3_PHASE2 -->

---

## 4. Phase 3: Canonical Spectral Analysis and Ridge Optimization

<!-- BEGIN AUTO-GENERATED: SECTION_4_PHASE3 -->
Writing the second-order validation RMSE surface as $\hat{y}(\mathbf{x}) = b_0 + \mathbf{x}^T \mathbf{b} + \mathbf{x}^T \hat{\mathbf{B}} \mathbf{x}$ gives $b_0 = 0.4994$ and linear gradient $\mathbf{b} = [-0.1242, -0.0336, -0.0047, -0.0007]^T$. Solving $\mathbf{x}_0 = -\frac{1}{2}\hat{\mathbf{B}}^{-1}\mathbf{b}$ yields an unconstrained stationary point at $\mathbf{x}_0 = [0.4171, 1.3524, 15.6944, -6.8009]^T$ ($\|\mathbf{x}_0\|_2 = 17.16$, $\hat{y}_0 = 0.4161$), which lies far outside $[-1, +1]^4$ due to flat regularization and subsampling curvature.

Spectral decomposition $\hat{\mathbf{B}} = \mathbf{V}\bm{\Lambda}\mathbf{V}^T$ yields eigenvalues:
$$\lambda_1 = 0.000057, \quad \lambda_2 = 0.000873, \quad \lambda_3 = 0.022759, \quad \lambda_4 = 0.110196 \quad (\text{trace}(\hat{\mathbf{B}}) = 0.133886)$$
Rademacher wild bootstrap (2,000 replications) places the 95% confidence interval for $\lambda_1$ at $[-0.0039, +0.0022]$, with $68.2\%$ of bootstrap resamples yielding $\min(\lambda) \le 0$. Because the confidence interval for $\lambda_1$ straddles zero, the response surface forms a **stationary/rising ridge system** along the $L_2$ regularization ($x_4$) and subsample ($x_3$) axes, while learning rate ($x_1$) and tree depth ($x_2$) exhibit steep positive convexity.

Constrained optimization within $\mathcal{D} = [-1, +1]^4$ restricted to valid integer depths $d \in \{3, \dots, 9\}$ yields:

| Depth | Coded $x_1$ | Coded $x_2$ | Coded $x_3$ | Coded $x_4$ | Optimal $\eta$ | Predicted RMSE | $\text{SE}(\hat{y})$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $3$ | $0.8103$ | $-1.0000$ | $1.00$ | $1.00$ | $0.2173$ | $0.4909$ | $0.0032$ |
| $4$ | $0.7573$ | $-0.6667$ | $1.00$ | $1.00$ | $0.1985$ | $0.4724$ | $0.0030$ |
| $5$ | $0.7043$ | $-0.3333$ | $1.00$ | $1.00$ | $0.1814$ | $0.4592$ | $0.0032$ |
| $6$ | $0.6513$ | $+0.0000$ | $1.00$ | $1.00$ | $0.1658$ | $0.4511$ | $0.0033$ |
| **$7$** | **$0.5983$** | **$+0.3333$** | **$1.00$** | **$1.00$** | **$0.1515$** | **$0.4483$** | **$0.0032$** |
| $8$ | $0.5453$ | $+0.6667$ | $1.00$ | $1.00$ | $0.1384$ | $0.4506$ | $0.0031$ |
| $9$ | $0.4923$ | $+1.0000$ | $1.00$ | $1.00$ | $0.1265$ | $0.4582$ | $0.0034$ |

The **Single-Objective RSM Candidate ($\mathbf{x}^*_{\text{SO}}$)** occurs at **depth $d = 7$** ($[0.5983, 0.3333, 1.0000, 1.0000]^T \implies \eta = 0.1515, d = 7, s = 1.0000, \lambda = 10.0000$), with predicted validation RMSE $\hat{y}(\mathbf{x}^*_{\text{SO}}) = 0.4483$ and Satterthwaite 95% PI $[0.4354, 0.4612]$.
<!-- END AUTO-GENERATED: SECTION_4_PHASE3 -->

---

## 5. Phase 4: Multi-Objective Derringer-Suich Desirability

<!-- BEGIN AUTO-GENERATED: SECTION_5_PHASE4 -->
To simultaneously minimize **Validation RMSE ($Y_1$)** and **Single-Sample Inference Latency ($Y_2$)**, we apply one-sided Derringer-Suich transformations ($s_1 = s_2 = 1.0$, equal weights $w_1 = w_2 = 1.0$) with operational specification bounds $Y_1 \in [L_1, U_1] = [0.450, 0.700]$ and $Y_2 \in [L_2, U_2] = [100.0, 180.0]\,\mu\text{s}$, maximizing $D(\mathbf{x}) = \sqrt{d_1(\hat{Y}_1(\mathbf{x})) \cdot d_2(\hat{Y}_2(\mathbf{x}))}$ over valid integer depths:

- **Coded Compromise Coordinate ($\mathbf{x}^*_{\text{MO}}$)**: $[0.8500, \ -0.6667, \ 1.0000, \ -0.0812]^T$
- **Natural Hyperparameters**: `learning_rate` $\eta = 0.2324$, `max_depth` $d = 4$, `subsample` $s = 1.0000$, `reg_lambda` $\lambda = 0.8295$
- **Surrogate Predictions**: $\hat{Y}_1 = 0.4804$ Validation RMSE, $\hat{Y}_2 = 138.11\,\mu\text{s}$ latency
- **Individual & Composite Desirabilities**: $d_1 = 0.8783$, $d_2 = 0.5236$, **$D = 0.6782$**
<!-- END AUTO-GENERATED: SECTION_5_PHASE4 -->

---

## 6. Phase 5: Empirical Confirmation Trials ($m = 10$ Fresh Seeds)

<!-- BEGIN AUTO-GENERATED: SECTION_6_PHASE5 -->
Both DOE candidate coordinates ($\mathbf{x}^*_{\text{MO}}$ at depth 4 and $\mathbf{x}^*_{\text{SO}}$ at depth 7) were evaluated across **$m = 10$ fresh random seeds** ($\mathcal{S}_{\text{conf}} = \{505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414\}$, matching `config.yaml` and `results/confirmation_runs.csv`) against Satterthwaite-adjusted 95% prediction intervals ($\nu_{\text{eff}} = 14.0$ for $\mathbf{x}^*_{\text{MO}}$ at $h_0 = 0.1043$; $\nu_{\text{eff}} = 15.1$ for $\mathbf{x}^*_{\text{SO}}$ at $h_0 = 0.1210$):

| Configuration / Response Metric | Surrogate Pred ($\hat{y}$) | 95% Pred Interval (PI) | Empirical Mean $\pm$ SD ($m = 10$) | Confirmation Status |
| :--- | :---: | :---: | :---: | :--- |
| **DOE Multi-Objective Optimum $\mathbf{x}^*_{\text{MO}}$ (Depth 4)** | | | | |
| Validation RMSE ($Y_1$) | $0.4804$ | $[0.4677, 0.4932]$ | $0.4853 \pm 0.0115$ | **Pass** (Inside 95% PI) |
| Holdout Test RMSE | — | — | $0.4888 \pm 0.0032$ | Holdout Test Set |
| Inference Latency ($\mu\text{s}$) | $138.1$ ($138.11$) | $[122.8, 153.4]$ ($[122.82, 153.40]$) | $142.3 \pm 7.8$ ($142.33 \pm 7.79$) | **Pass** (Inside 95% PI) |
| **DOE Single-Objective Candidate $\mathbf{x}^*_{\text{SO}}$ (Depth 7)** | | | | |
| Validation RMSE ($Y_1$) | $0.4483$ | $[0.4354, 0.4612]$ | $0.4700 \pm 0.0093$ | **Not Confirmed** (Optimism: $+0.0217$) |
| Holdout Test RMSE | — | — | $0.4691 \pm 0.0034$ | Holdout Test Set |
| Inference Latency ($\mu\text{s}$) | $187.0$ ($187.00$) | $[171.1, 202.9]$ ($[171.10, 202.90]$) | $171.1 \pm 10.9$ ($171.12 \pm 10.95$) | **Borderline** (At Lower PI) |

At $\mathbf{x}^*_{\text{MO}}$ (depth 4), both validation RMSE ($0.4853 \pm 0.0115$) and single-sample inference latency ($142.33 \pm 7.79\,\mu\text{s}$) fall directly inside their Satterthwaite 95% prediction intervals. At $\mathbf{x}^*_{\text{SO}}$ (depth 7), empirical validation RMSE ($0.4700 \pm 0.0093$) lies $0.0088$ above the upper prediction interval bound ($0.4612$), confirming that quadratic interpolation across depths $\{3, 6, 9\}$ overestimates accuracy gains at depth 7 by $+0.0217$ RMSE ($6.8\times \text{SE}(\hat{y})$), while empirical latency ($171.1 \pm 10.9\,\mu\text{s}$) sits on the lower bound of its 95% prediction interval ($[171.1, 202.9]\,\mu\text{s}$).

### 6.1 Provenance of Numerical Similarities Across Confirmation, Historical Benchmark, and Revision-v2 Evaluations

Several summary metrics across Table 6 and Table 7.1 agree to three or four decimal places (`0.4686`/`0.4687`/`0.4690`/`0.4691`, `0.4853`, `0.4884`/`0.4888`, `0.4700`) for distinct structural or statistical reasons that must be distinguished:
1. **Shared Configuration Hash vs. Adjacent Grid Point Across 20-Seed Benchmark Evaluations (`doe-so-historical`, `doe-mo-historical`, and `doe-so-rep-0..19`)**: Both the historical benchmark script (`scripts/run_benchmarks.py`) and the Revision-v2 holdout evaluator (`final_evaluation.py`) evaluate frozen configurations across the exact same 20 fresh seeds $\mathcal{S}_{\text{eval}} = \{2001, \dots, 2020\}$ on the 75/25 split of the 16,512-row development pool ($N_{\text{train}} = 12,384, N_{\text{val}} = 4,128$) and the 4,128-row external holdout test set. Consequently, `doe-so-historical` (`config_hash = 0079fbae0ec5b7cb`: $\eta = 0.1515, d = 7, s = 1.0000, \lambda = 10.0000$) and `doe-mo-historical` (`config_hash = b05e8e5cb67e8668`: $\eta = 0.2324, d = 4, s = 1.0000, \lambda = 0.8295$) achieve identical full-precision RMSEs in `results/benchmark.csv` and `final_evaluations.csv` ($\mathbf{x}^*_{\text{SO}}$: `test_rmse = 0.469033`, `val_rmse = 0.468680`; $\mathbf{x}^*_{\text{MO}}$: `test_rmse = 0.488409`, `val_rmse = 0.482071`) while differing in latency due to separate timing sessions. In Revision-v2, all 20 Repeated DOE Single-Objective replicates select the exact same boundary grid configuration (`config_hash = 9d7c4e74ff156870`, `grid_index = 1052`: coded $[+0.6, +1/3, +1.0, +1.0]^T$, i.e., $\eta = 0.1519, d = 7, s = 1.0000, \lambda = 10.0000$), which differs from continuous SLSQP $\mathbf{x}^*_{\text{SO}}$ ($x_1 = +0.5983, \eta = 0.1515$) only by the $0.1$ coded grid step on $x_1$. Because `final_evaluation.py` deduplicates model training by `config_hash` across seeds `2001..2020`, all 20 `doe-so-rep-r` records map to the exact same 20 trained XGBoost boosters (`test_rmse = 0.468591`, `val_rmse = 0.468723`), yielding zero between-search variance ($\text{SD} = 0.00000$).
2. **Disjoint Seed Sets Matching to Three or Four Decimal Places (`confirmation_runs*.csv` on `505..1414` vs. Benchmark on `2001..2020`)**: Phase 5 confirmation evaluates $\mathbf{x}^*_{\text{SO}}$ and $\mathbf{x}^*_{\text{MO}}$ across $m = 10$ seeds $\mathcal{S}_{\text{conf}} = \{505, 606, \dots, 1414\}$, which is completely disjoint from $\mathcal{S}_{\text{eval}} = \{2001, \dots, 2020\}$. At $\mathbf{x}^*_{\text{SO}}$, the 10-seed confirmation holdout test RMSE mean is `0.469124` ($\text{SD} = 0.003437$, rounding to `$0.4691$`), whereas the 20-seed historical benchmark holdout test RMSE mean is `0.469033` ($\text{SD} = 0.002585$, rounding to `$0.4690$`, differing by $91.25 \times 10^{-6}$ RMSE) and the 20-seed Repeated DOE SO grid candidate mean is `0.468591` (`0.4686` test, `0.468723` or `0.4687` val, matching historical $\mathbf{x}^*_{\text{SO}}$ validation `0.468680` / `0.4687` to four decimal places). Likewise, at $\mathbf{x}^*_{\text{MO}}$, the 10-seed confirmation validation RMSE mean is `0.485286` (rounding to `$0.4853$`) and holdout test RMSE is `0.488774` ($\text{SD} = 0.003243$, rounding to `$0.4888$`) vs. the 20-seed historical benchmark (`0.488409`, $\text{SD} = 0.003191$, rounding to `$0.4884$` test and `0.482071` / `$0.4821$` val), while the 20 Repeated DOE MO replicates (which select 12 distinct grid configurations across $d \in \{4, 5\}$) average `$0.48528$` (`0.485283`, rounding to `$0.4853$` and matching the 10-seed confirmation validation mean `0.4853` to four decimal places) and `$0.48997$` (`0.4900` at 4 decimal places) on holdout test. Similarly, Constrained TPE validation RMSE (`0.469962`) and 10-seed $\mathbf{x}^*_{\text{SO}}$ confirmation validation RMSE (`0.470000`) both round to `0.4700`.
<!-- END AUTO-GENERATED: SECTION_6_PHASE5 -->

---

## 7. Prospective Multi-Replicate Benchmark Campaign (`Revision-v2`)

<!-- BEGIN AUTO-GENERATED: SECTION_7_BENCHMARKS -->
To evaluate the pre-planned DOE + RSM methodology against modern heuristic and Bayesian optimizers under strictly equal evaluation budgets (140 model evaluations per search), we executed a prospective benchmark campaign (`results/revision_v2/full_run_001/`) comprising **17,077 XGBoost model fits** and **1,640,810 timed single-sample inferences** (1,842,080 total prediction calls including 201,270 warmup calls) across $N = 20$ independent search replicates per optimizer.

All **122 winning selection records** (representing **95 distinct hyperparameter configurations** by SHA-256 hash and yielding **2,440 final evaluation rows** in `final_evaluations.csv`) were cryptographically frozen in `finalized_selections.json` prior to generalization testing across 20 fresh retraining seeds ($\mathcal{S}_{\text{eval}} = \{2001, 2002, \dots, 2020\}$) on the external holdout test set ($N = 4,128$). Whereas the historical `v1.0.0` baseline logged holdout test metrics during exploratory development, the `Revision-v2` pipeline programmatically prevented access to holdout test labels until `finalized_selections.json` was frozen (enforcing pipeline isolation during the revised search, while recognizing that the same dataset was previously evaluated in the historical baseline).

### 7.1 Holdout Generalization and Latency Comparison

| Optimization Method | Search Basis ($N$ Reps) | Val RMSE (Mean $\pm$ SD) | Holdout Test RMSE (Mean $\pm$ SD) | Holdout Test RMSE [95% CI] | Retrain $\sigma_{\text{eval}}$ | `predict` Latency ($\mu\text{s} \pm \text{SD}$) | `inplace_predict` ($\mu\text{s} \pm \text{SD}$) | Feasible ($\le 145\,\mu\text{s}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Panel A: Prospective Revision-v2 Full Benchmark ($N = 20$ Search Replicates)** | | | | | | | | |
| **Repeated DOE MO ($\mathbf{x}^*_{\text{MO}}$)** | RSM ($N=20$) | $0.48528 \pm 0.00554$ ($0.4853 \pm 0.0055$) | $0.48997 \pm 0.00732$ ($0.4900 \pm 0.0073$) | $[0.48654, 0.49339]$ | $0.00330$ ($0.0033$) | $122.29 \pm 2.70$ ($122.3 \pm 2.7$) | $89.73 \pm 2.73$ | 20/20 (100%) |
| **Multi-Objective TPE** | Parzen ($N=20$) | $0.48815 \pm 0.00934$ ($0.4881 \pm 0.0093$) | $0.49255 \pm 0.01080$ ($0.4926 \pm 0.0108$) | $[0.48749, 0.49761]$ | $0.00358$ ($0.0036$) | $121.46 \pm 3.75$ ($121.5 \pm 3.8$) | $88.84 \pm 3.76$ | 20/20 (100%) |
| **Constrained TPE ($\le 145\,\mu\text{s}$)** | Parzen ($N=20$) | $0.46996 \pm 0.00222$ ($0.4700 \pm 0.0022$) | $0.47036 \pm 0.00324$ ($0.4704 \pm 0.0032$) | $[0.46885, 0.47188]$ | $0.00341$ ($0.0034$) | $144.27 \pm 7.06$ ($144.3 \pm 7.1$) | $112.07 \pm 7.24$ | 9/20 (45%) |
| **Repeated DOE SO ($d = 7$)** | RSM ($N=20$) | $0.46872 \pm 0.00000$ ($0.4687 \pm 0.0000$) | $0.46859 \pm 0.00000$ ($0.4686 \pm 0.0000$) | $[0.46859, 0.46859]$ | $0.00304$ ($0.0030$) | $150.74 \pm 0.72$ ($150.7 \pm 0.7$) | $118.78 \pm 0.59$ | 0/20 (0%) |
| **Single-Objective TPE** | Parzen ($N=20$) | $0.46750 \pm 0.00082$ ($0.4675 \pm 0.0008$) | $0.46691 \pm 0.00116$ ($0.4669 \pm 0.0012$) | $[0.46637, 0.46746]$ | $0.00314$ ($0.0031$) | $188.93 \pm 12.07$ ($188.9 \pm 12.1$) | $157.14 \pm 12.11$ | 0/20 (0%) |
| **Unguided Random Search** | Uniform ($N=20$) | $0.46869 \pm 0.00143$ ($0.4687 \pm 0.0014$) | $0.46953 \pm 0.00207$ ($0.4695 \pm 0.0021$) | $[0.46856, 0.47050]$ | $0.00297$ ($0.0030$) | $181.12 \pm 19.02$ ($181.1 \pm 19.0$) | $149.05 \pm 18.81$ | 0/20 (0%) |
| **Panel B: Historical Baseline Snapshot (`v1.0.0`, Single Search Replicate)** | | | | | | | | |
| **Historical DOE MO ($\mathbf{x}^*_{\text{MO}}$)** | RSM (Single) | $0.4821$ | $0.4884$ | — | — | $119.8$ | $87.1$ | Yes |
| **Historical Multi-Obj TPE** | Parzen (Single) | $0.4812$ | $0.4870$ | — | — | $118.8$ | $87.6$ | Yes |
| **Historical Constrained TPE** | Parzen (Single) | $0.4783$ | $0.4823$ | — | — | $125.4$ | $94.1$ | Yes |
| **Historical DOE SO ($d = 7$)** | RSM (Single) | $0.4687$ | $0.4690$ | — | — | $147.8$ | $116.9$ | No |
| **Historical Bayesian TPE** | Parzen (Single) | $0.4735$ | $0.4726$ | — | — | $193.2$ | $160.9$ | No |
| **Historical Random Search** | Uniform (Single) | $0.4696$ | $0.4707$ | — | — | $198.1$ | $164.2$ | No |

### 7.2 Multi-Objective Pareto Hypervolume Comparison

| Frontier / Evaluation Protocol | Candidate Pool ($N$) | Non-Dom. Points | HV at $[0.60, 250]$ (Mean $\pm$ SD) | HV at $[0.65, 275]$ (Mean $\pm$ SD) |
| :--- | :---: | :---: | :---: | :---: |
| **Panel A: Development Candidate Frontiers (Distinct Evaluation Protocols Noted per Row)** | | | | |
| **Multi-Objective TPE Candidates (Single Split 42, 30-Call Search)** | $140 \times 20$ | $9.35 \pm 2.3$ | $18.3834 \pm 0.3301$ | $30.0535 \pm 0.4123$ |
| **Repeated DOE Candidate Fronts (5-Block Means Across Replicates)** | $25 \times 20$ | $6.10 \pm 1.4$ | $15.6128 \pm 0.7268$ | $26.6847 \pm 0.8786$ |
| **Fixed Full DOE Evaluated Front (Single Split 42, 27 Configs)** | $27$ | $4$ | $16.8585$ | $28.1430$ |
| **Historical DOE 2-Point Set (Single Split 42, $\mathbf{x}^*_{\text{MO}}$ & $\mathbf{x}^*_{\text{SO}}$)** | $2$ | $2$ | $16.0363$ | $26.9168$ |
| **Difference: Fixed Full DOE Front $-$ MO-TPE Mean (Split 42)** | — | — | $-1.5249$ (1-sample $p < 0.0001$) | $-1.9105$ (1-sample $p < 0.0001$) |
| **Panel B: Holdout Non-Dominated Set (122 Records, 95 Configs, 20 Seeds)** | | | | |
| **Non-Dominated Set (12 Distinct Configs)** | $122$ | $12$ | $17.6714$ | $28.9900$ |
| **Holdout Frontier Seed-Bootstrap ($B=2,000$ Resamples over 20 Seeds)** | $122$ | $13.42 \pm 1.46$ | $17.6923 \pm 0.0564$ ($[17.5848, 17.8061]$) | $29.0170 \pm 0.0688$ ($[28.8861, 29.1579]$) |
| $\quad \llcorner$ Repeated DOE MO Selections | $20$ | $3$ (Boot: $100.0\%$) | — | — |
| $\quad \llcorner$ Multi-Objective TPE Selections | $20$ | $3$ (Boot: $100.0\%$) | — | — |
| $\quad \llcorner$ Constrained TPE Selections | $20$ | $5$ (Boot: $100.0\%$) | — | — |
| $\quad \llcorner$ Single-Objective TPE Selections | $20$ | $1$ (Boot: $91.5\%$) | — | — |

*Hypervolume Protocol Comparability Note*: Rows in Panel A and Panel B represent distinct evaluation protocols and candidate pools rather than a single homogeneous two-sample experiment: (i) **MO-TPE Candidates** ($20$ searches $\times 140$ trials on fixed split seed $42$, timed with $30$ un-warmed calls/trial); (ii) **Repeated DOE Candidate Fronts** ($20$ searches $\times 25$ FCCD coordinates averaged across $5$ disjoint block seeds in `3001..3100`, timed with $30$ un-warmed calls/fit); (iii) **Fixed Full DOE Evaluated Front** (a single deterministic 27-configuration set—25 FCCD points plus $\mathbf{x}^*_{\text{MO}}$ and $\mathbf{x}^*_{\text{SO}}$—evaluated once on split $42$ with $1,000$ warmed calls/fit); and (iv) **Holdout Non-Dominated Set** ($122$ frozen selection records across $95$ distinct configurations evaluated on the 4,128-row external holdout test set across $20$ fresh seeds `2001..2020` and $2$ timing sessions of $500$ warmed calls/fit). The Split-42 difference row ($-1.5249$ at $[0.60, 250]$; $-1.9105$ at $[0.65, 275]$) is a **one-sample comparison** of the 20 stochastic MO-TPE fronts against the single fixed 27-point DOE reference on split 42, not a two-sample test across independent data splits. Likewise, in the historical `v1.0.0` snapshot (`results/benchmark_summary.json`), the reported $18.38$ vs. $16.86$ ($+9.0\%$) hypervolume comparison contrasted the full 25-point DOE design against a single selected MO-TPE configuration; when restricted to single deployed points ($\mathbf{x}^*_{\text{MO}}$ vs. MO-TPE), historical hypervolume is $16.92$ vs. $16.86$ ($+0.3\%$).

### 7.3 Key Scientific Conclusions from the Benchmark Campaign

1. **Single-Objective Accuracy vs. Latency**: Single-Objective TPE achieves the numerically lowest observed mean holdout test RMSE ($0.46691 \pm 0.00116$) by concentrating searches in deeper trees ($d \in \{7, 8, 9\}$), incurring $188.93 \pm 12.07\,\mu\text{s}$ inference latency (`inplace_predict`: $157.14 \pm 12.11\,\mu\text{s}$). Neither Single-Objective TPE nor Repeated DOE Single-Objective satisfies the $\le 145\,\mu\text{s}$ latency constraint ($0/20$ feasible). Repeated DOE Single-Objective ($\mathbf{x}^*_{\text{SO}}$, depth 7) selects the exact same boundary configuration deterministically across all 20 replicates (between-search $\text{SD} = 0.00000$, conditional retraining $\sigma_{\text{eval}} = 0.00304$), achieving $0.46859$ Test RMSE at $150.74 \pm 0.72\,\mu\text{s}$ inference latency (`inplace_predict`: $118.78 \pm 0.59\,\mu\text{s}$, $20.2\%$ lower latency than SO-TPE).
2. **What Repeated DOE Replicates Represent & Separation of Selection vs. Retraining Variance**: Each of the $N = 20$ Repeated DOE replicates (`replicate_id` $0 \dots 19$) executes a fresh 140-run blocked FCCD across 5 disjoint block seeds (`3001 + 5r` to `3005 + 5r`, spanning seeds `3001..3100`) drawn from the same 16,512-row California Housing development pool, fits OLS second-order surfaces for validation RMSE and latency, and selects candidates by evaluating the fitted surrogates over a deterministic **1,323-point candidate grid** (`doe_selection_protocol.json`: $21$ coded learning-rate levels $x_1 \in [-1, +1]$ in steps of $0.1 \times 7$ integer depths $d \in \{3, \dots, 9\} \times 3$ coded subsample levels $x_3 \in \{-1, 0, +1\} \times 3$ coded regularization levels $x_4 \in \{-1, 0, +1\}$). Because the single-objective minimum on this grid consistently lies at the same boundary coordinate (`grid_index = 1052`, coded $[+0.6, +1/3, +1.0, +1.0]^T \implies \eta = 0.1519, d = 7, s = 1.0000, \lambda = 10.0000$), all 20 DOE-SO replicates select the identical configuration hash (`9d7c4e74ff156870`), yielding between-search selection $\text{SD} = 0.00000$; its non-zero uncertainty is captured entirely by the conditional retraining standard deviation $\sigma_{\text{eval}} = 0.00304$ across the 20 evaluation seeds `2001..2020`. For DOE-MO, desirability optimization over feasible grid points selects across 12 distinct grid configurations in $d \in \{4, 5\}$, yielding between-search $\text{SD} = 0.00732$ alongside conditional retraining $\sigma_{\text{eval}} = 0.00330$.
3. **Multi-Objective Statistical Comparison (Primary Unpaired Welch vs. Secondary Index-Paired Test)**: In the latency-constrained multi-objective regime, Repeated DOE ($\mathbf{x}^*_{\text{MO}}$, depth 4) achieves a mean Test RMSE of $0.48997 \pm 0.00732$ at $122.29 \pm 2.70\,\mu\text{s}$ latency (`inplace_predict`: $89.73 \pm 2.73\,\mu\text{s}$), compared to $0.49255 \pm 0.01080$ at $121.46 \pm 3.75\,\mu\text{s}$ (`inplace_predict`: $88.84 \pm 3.76\,\mu\text{s}$) for Multi-Objective TPE. Because replicate index $r \in \{0, \dots, 19\}$ pairs DOE block seeds `3001+5r..3005+5r` with Optuna sampler seed `1000+r` on split 42 rather than a shared random block, **Welch's two-sample $t$-test is the primary inferential comparison** ($\Delta = -0.00258$ RMSE, $t = -0.8849, p = 0.3826$), while the replicate-index paired $t$-test ($t = -0.966, p = 0.3463$) is reported as a secondary sensitivity check. Because no pre-registered equivalence margin was specified (`rmse_equivalence_margin: null`), this non-significant difference indicates no detected performance gap rather than formal statistical equivalence. Both methods achieve $20/20$ ($100\%$) benchmark constraint feasibility.
4. **Development Hypervolume vs. Holdout Non-Dominated Set and Bootstrap Uncertainty**: On development split 42 under the online 30-call timing protocol, MO-TPE candidate fronts achieve a mean hypervolume of $18.3834 \pm 0.3301$ at reference $[0.60, 250.0]$ ($30.0535 \pm 0.4123$ at $[0.65, 275.0]$), exceeding the fixed 27-point DOE candidate frontier ($16.8585$, descriptive 1-sample difference $-1.5249, p < 0.0001$; at $[0.65, 275.0]$, $28.1430$, difference $-1.9105, p < 0.0001$), while Repeated DOE 5-block means achieve $15.6128 \pm 0.7268$ ($26.6847 \pm 0.8786$ at $[0.65, 275.0]$). When the 122 frozen selection records (95 distinct configurations) are evaluated on the external holdout test set across 20 retraining seeds, the empirical non-dominated set consists of **12 distinct configurations** ($\text{HV} = 17.6714$ at $[0.60, 250.0]$ and $28.9900$ at $[0.65, 275.0]$): **3 Repeated DOE MO, 3 MO-TPE, 5 Constrained TPE, and 1 SO-TPE**. Paired cluster bootstrap resampling ($B = 2,000$ resamples over the 20 shared evaluation seeds `2001..2020`, `seed = 20261011`) yields a holdout frontier hypervolume of **$17.6923 \pm 0.0564$** ($95\%$ CI $[17.5848, 17.8061]$) at $[0.60, 250]$ and **$29.0170 \pm 0.0688$** ($95\%$ CI $[28.8861, 29.1579]$) at $[0.65, 275]$ (joint seed-plus-timing-session bootstrap: $17.6890 \pm 0.0727$, $95\%$ CI $[17.5507, 17.8420]$). Across the 2,000 seed-bootstrap frontiers, Repeated DOE MO, MO-TPE, and Constrained TPE each contribute at least one non-dominated point in **$100.0\%$** of resamples (mean non-dominated points: $2.76$ DOE-MO, $3.16$ MO-TPE, $6.03$ Constrained TPE), while SO-TPE contributes in **$91.5\%$** (mean $1.38$ points).
5. **Constrained TPE Feasibility Degradation**: While Constrained TPE satisfies $\le 145\,\mu\text{s}$ on $20/20$ searches during online 30-call search timing, only **$9/20$ ($45\%$)** remain feasible under 1,000-call benchmark verification ($9/9$ depth-6 selections feasible at $136.70 \pm 0.82\,\mu\text{s}$; $0/11$ depth-7 selections feasible at $150.45 \pm 0.67\,\mu\text{s}$). Plausible factors include noisy 30-call search measurements near the boundary, protocol differences (30 calls without warmup vs. 1,000 calls with warmup), and winner's-curse selection bias near the $145\,\mu\text{s}$ threshold.
6. **Historical Latency Comparison Caveat**: Both historical `v1.0.0` scripts (`scripts/run_confirmation.py` and `scripts/run_benchmarks.py`) applied Win32 CPU core 0 affinity pinning. Consequently, the shift between historical single-run latency snapshots and the prospective benchmark session reflects between-session system timing drift ($\text{SD} = 1.28\,\mu\text{s}$ across repeated sessions) rather than unpinned execution.

### 7.4 Computational Budget Ledger and Wall-Clock Cost

The table below reconciles every model fit, distinct coordinate, timed single-row inference, untimed warmup call, and per-replicate search wall-clock duration across the historical `v1.0.0` baseline and the prospective `Revision-v2` campaign (`computational_budget.json` and `optimizer_replicates.csv`):

| Phase / Method Component | Search Reps | XGBoost Fits | Distinct Coords | Timed Inferences | Warmup Calls | Total Inferences | Mean Fit Time (ms) | Wall Time / Rep (s, Mean $\pm$ SD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Panel A: Historical Single-Campaign Baseline (`v1.0.0`, 160 DOE + Confirmation Fits)** | | | | | | | | |
| Phase 1: $2^4$ Factorial + 4 Center ($5$ Blocks) | $1$ | $100$ | $17$ | $100,000$ | $10,000$ | $110,000$ | — | — |
| Phase 2: Face-Centered Axial ($5$ Blocks) | $1$ | $40$ | $8$ | $40,000$ | $4,000$ | $44,000$ | — | — |
| Phase 5: Confirmation ($\mathbf{x}^*_{\text{MO}}$ & $\mathbf{x}^*_{\text{SO}}$, $m=10$) | $2$ | $20$ | $2$ | $20,000$ | $2,000$ | $22,000$ | — | — |
| **Panel B: Prospective Revision-v2 Campaign (Total = 17,077 Fits, 1,640,810 Timed Inferences)** | | | | | | | | |
| Repeated Pre-Planned DOE (MO + SO Shared) | $20$ | $2,800$ | $25/\text{rep}$ | $84,000$ | $28,000$ | $112,000$ | $238.6$ | $36.49 \pm 0.19$ |
| Multi-Objective TPE ($140$ Trials/Rep) | $20$ | $2,800$ | $140/\text{rep}$ | $84,000$ | $28,000$ | $112,000$ | $174.2$ | $26.24 \pm 1.80$ |
| Constrained TPE ($140$ Trials/Rep) | $20$ | $2,800$ | $140/\text{rep}$ | $84,000$ | $28,000$ | $112,000$ | $176.9$ | $26.50 \pm 2.70$ |
| Single-Objective TPE ($140$ Trials/Rep) | $20$ | $2,800$ | $140/\text{rep}$ | $84,000$ | $28,000$ | $112,000$ | $370.2$ | $54.03 \pm 3.05$ |
| Random Search ($140$ Trials/Rep) | $20$ | $2,800$ | $140/\text{rep}$ | $84,000$ | $28,000$ | $112,000$ | $221.9$ | $32.61 \pm 1.63$ |
| Split-42 Full DOE Baseline ($25 + 2$ Opt) | $1$ | $27$ | $27$ | $810$ | $270$ | $1,080$ | — | — |
| Primary Latency Sessions ($5 \times 122$ Records) | $5$ | $610$ | $95$ | $1,220,000$ | $61,000$ | $1,281,000$ | — | — |
| Holdout Retraining ($122$ Records $\times 20$ Seeds) | — | $2,440$ | $95$ | $0$ | $0$ | $0$ | — | — |
| **Prospective Campaign Total** | — | **17,077** | — | **1,640,810** | **201,270** | **1,842,080** | — | — |
<!-- END AUTO-GENERATED: SECTION_7_BENCHMARKS -->

---

## 8. Discussion and Methodological Limitations

<!-- BEGIN AUTO-GENERATED: SECTION_8_9_DISCUSSION_AND_ARTIFACTS -->
1. **Parametric Variance Attribution vs. Adaptive Coordinate Exploration**: Pre-planned blocked DOE + RSM and adaptive Bayesian optimization (TPE) answer complementary questions under a fixed 140-evaluation budget. Blocked FCCD replicates 25 structured coordinates across 5 splits to decompose variance into main effects, interactions, quadratic curvature ($F = 105,432.31$ against center pure error; $F = 20,235.41$ against 79-df block-adjusted pooled error), seed-block variance ($\text{ICC} = 40.69\%$), and structural lack of fit ($F = 62.21$), while yielding deterministic or near-deterministic compromise selections ($\mathbf{x}^*_{\text{MO}}$ and $\mathbf{x}^*_{\text{SO}}$). Conversely, TPE concentrates all 140 evaluations into distinct configurations without replication, achieving lower unconstrained single-objective test RMSE ($0.46691$ vs. $0.46859$) and higher split-42 candidate hypervolume ($18.3834$ vs. $16.8585$) by avoiding repeated evaluations at the high-loss $\eta = 0.01$ boundary ($45/140 = 32.14\%$ of FCCD runs) and by exploring deeper trees ($d \in \{7, 8, 9\}$) at higher training wall-clock cost ($54.03\,\text{s}$ vs. $36.49\,\text{s}$ per search).
2. **Value of Stochastic Nuisance Blocking**: Blocking across 5 data-partition seeds absorbed $\text{ICC} = 40.69\%$ of residual variance ($\sigma_{\text{block}} \approx 0.0077$ RMSE), preventing partition noise from confounding hyperparameter comparisons.
3. **Structural Lack of Fit and Surrogate Optimism**: Because decision tree ensembles exhibit diminishing returns at deeper levels ($d \ge 6$) and sharp exponential penalties at $\eta = 0.01$, a global second-order polynomial interpolated across $d \in \{3, 6, 9\}$ exhibits statistically significant lack of fit ($\text{RMS} = 0.0297$ RMSE) and under-predicts validation RMSE at depth 7 by $+0.0217$ RMSE. Satterthwaite prediction intervals widen for variance heterogeneity across degrees of freedom but cannot correct deterministic polynomial bias.
4. **External Validity and Scope Limitations**: All empirical conclusions are scoped to a single tabular regression dataset (California Housing, $N = 20,640$), a single gradient boosting library (`XGBoost 3.2.0`, `n_estimators = 100`), a 4-hyperparameter domain ($\eta, d, s, \lambda$), and single-core Windows CPU inference timing (`Intel Core i7-12650H`). While the 12-point holdout Pareto frontier spans both Repeated DOE MO and TPE selections (with $100.0\%$ bootstrap presence for DOE-MO, MO-TPE, and Constrained TPE, and $91.5\%$ for SO-TPE), generalizing these findings to higher-dimensional spaces ($k > 6$), larger datasets, or GPU/batch serving requires fractional factorial screening ($2^{k-p}$), sequential region zooming, or hybrid DOE + Bayesian pipelines (`docs/proposed_followup_experiments.md`).

---

## 9. Repository Artifacts, Provenance, and AI-Assistance Disclosure

- **LaTeX Manuscript & Compiled PDF**: `report.tex` and `report.pdf` (compiled via Tectonic with zero unresolved references or layout overflows).
- **Auto-Generated Statistical Macros & Tables**: `results/macros.tex` (516 macros) and `tables/*.tex`, generated deterministically via `python scripts/generate_report_artifacts.py`.
- **Single-Source-of-Truth Reporting & Manifest**: `scripts/reporting_data.py`, `scripts/generate_research_reporting.py`, `docs/generated/scientific_results_manifest.json`, `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md`.
- **Immutable Evidence Provenance**: Historical `v1.0.0` baseline artifacts (`results/runs.csv`, `results/phase1.json`, `results/phase3.json`, `results/confirmation.json`, `results/benchmark.csv`) were committed at SHA `cd63135d2b70a0ee324555c11b67a541b32b98a4`; prospective `Revision-v2` campaign artifacts (`results/revision_v2/full_run_001/`) were committed at SHA `421cf24229f1f5fba26cdb3103ec989f9a427c96` (`run_manifest.json` records the pre-commit parent SHA `45fe6f7ca8eb66b479edfb68d4b7c0c154a5d624` at campaign launch). All 40 canonical evidence files are protected by SHA-256 verification via `CANONICAL_EVIDENCE_SHA256` and `verify_canonical_evidence_hashes()` in `scripts/reporting_data.py` (with Revision-v2 artifact digests recorded in `results/revision_v2/full_run_001/provenance.json`).
- **Verification & Audit Suite**: `python scripts/audit_scientific_consistency.py`, `python scripts/generate_research_reporting.py --check`, `python scripts/generate_report_artifacts.py --check`, and `pytest`.
- **AI-Assistance Disclosure**: AI coding and scientific auditing assistants (OpenAI Codex and Google DeepMind Antigravity) were used for pipeline refactoring, deterministic reporting-layer generation, statistical sensitivity verification, and manuscript auditing against frozen experimental artifacts. All experimental designs, hyperparameter bounds, frozen evidence files, and scientific conclusions were verified against primary row-level records by the author.
<!-- END AUTO-GENERATED: SECTION_8_9_DISCUSSION_AND_ARTIFACTS -->
