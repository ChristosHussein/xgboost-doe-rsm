# Sequential Response Surface Methodology and Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking

**Author:** Christos Chousein Sounios

---

## Executive Summary

This report presents a comprehensive Design of Experiments (DOE) and Response Surface Methodology (RSM) study applied to the multi-objective hyperparameter optimization of an **XGBoost Regressor** on the **California Housing** benchmark dataset ($N = 20,640$ observations, $8$ continuous features). Following the methodology of Douglas C. Montgomery's *Design and Analysis of Experiments* (9th ed., 2017, Chapters 5, 9, 10, and 14), we treat stochastic machine learning variability—arising from random train/validation partitioning and row subsampling—as a **nuisance factor** controlled via a **Randomized Complete Block Design (RCBD)** across $b = 5$ seed blocks ($\mathcal{S} = \{42, 101, 202, 303, 404\}$).

All primary DOE phases consist of **150 genuine model training and evaluation runs** (100 in Phase 1, 40 axial augmentations in Phase 2, and 10 confirmation trials in Phase 5 for $\mathbf{x}^*_{\text{MO}}$, supplemented by 10 confirmation trials for $\mathbf{x}^*_{\text{SO}}$), alongside a prospective 20-replicate benchmark campaign comprising **17,077 XGBoost model fits** and **1,640,810 timed single-sample inferences**:

1. **Holdout Isolation and Dataset History**: The 20,640 dataset observations are partitioned into an 80% development pool ($N_{\text{dev}} = 16,512$) and a 20% external holdout test set ($N_{\text{test}} = 4,128$). Within each block $b$, the development pool is split 75/25 into training ($N_{\text{train}} = 12,384$) and validation ($N_{\text{val}} = 4,128$) sets. Response $Y_1$ is **Development Validation RMSE** on the 25% validation split. Whereas the historical `v1.0.0` baseline experiments recorded holdout test results during development, the revised `Revision-v2` pipeline programmatically prevented access to those test labels during all search iterations, surrogate fitting, curvature testing, lack-of-fit analysis, and desirability optimization until winning configurations were frozen in `finalized_selections.json`. These safeguards enforce programmatic pipeline isolation during the revised search, although they cannot retroactively undo earlier historical exposure of the same dataset.
2. **Phase 1 ($2^4$ Full Factorial + Center Points in 5 Blocks, $N_1 = 100$ runs)**: Screening identifies **Learning Rate** ($\ln \eta$, $F = 357.40, p < 0.0001, \eta^2_p = 0.8079$) and **Max Tree Depth** ($F = 23.57, p < 0.0001, \eta^2_p = 0.2171$), along with their antagonist interaction $x_1 x_2$ ($F = 26.32, p < 0.0001, \eta^2_p = 0.2364$), as the dominant drivers of validation RMSE. Single-degree-of-freedom curvature testing against within-block center-point pure error ($\text{df} = 15, \text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$) reveals extreme quadratic curvature ($F_{\text{Curv}} = 105,432.31, p < 10^{-15}$), where the factorial corner mean ($\bar{y}_F = 0.6333$) exceeds the center mean ($\bar{y}_C = 0.4989$) by $0.1343$ RMSE.
3. **Phase 2 (Face-Centered Central Composite Design, $\alpha = 1.0$, $N_{\text{CCD}} = 140$ runs)**: Augmenting with $40$ axial points fits a full second-order polynomial ($R^2 = 0.9951, \text{Adj } R^2 = 0.9944$). Once quadratic curvature is modeled, the RCBD block effect becomes highly significant ($F = 20.21, p < 0.0001$), absorbing **ICC = 40.69%** ($\text{ICC}_{\text{REML}} = 0.4069, \sigma_{\text{block}} \approx 0.0077$ RMSE) of unexplained residual variance. Formal lack-of-fit decomposition shows statistically significant structural polynomial misfit ($F_{\text{LoF}} = 62.21, p = 7.78 \times 10^{-41}$ against the 111-df saturated additive baseline; $F_{\text{LoF}} = 323.10, p < 10^{-15}$ against 15-df center pure error; RMS misfit $= 0.0297$ RMSE), demonstrating that second-order polynomials serve as local guidance maps rather than globally exact estimators of gradient boosting loss.
4. **Phase 3 (Canonical & Ridge Analysis)**: Spectral decomposition of $\hat{\mathbf{B}}$ yields three positive eigenvalues and one near-zero eigenvalue ($\lambda = \{0.000057, 0.000873, 0.022759, 0.110196\}$; bootstrap 95% CI for $\lambda_1$: $[-0.0039, +0.0022]$, with $68.2\%$ of resamples $\le 0$), characterizing a **stationary/rising ridge system**. The unconstrained stationary point ($\|\mathbf{x}_0\| = 17.16$) lies outside $[-1, +1]^4$; constrained optimization over valid integer depths identifies the single-objective candidate $\mathbf{x}^*_{\text{SO}}$ at depth $7$ ($x_1 = 0.5983, x_2 = 0.3333, x_3 = 1.0000, x_4 = 1.0000$, i.e., $\eta = 0.1515, \text{depth} = 7, \text{subsample} = 1.00, \lambda = 10.00$) with predicted validation RMSE $\hat{y} = 0.4483$.
5. **Phase 4 (Derringer-Suich Multi-Objective Desirability)**: Balancing Validation RMSE ($Y_1 \in [0.450, 0.700]$) and Single-Sample Inference Latency ($Y_2 \in [100.0, 180.0]\,\mu\text{s}$) yields an interior compromise coordinate $\mathbf{x}^*_{\text{MO}} = [0.8500, -0.6667, 1.0000, -0.0812]^T$ ($\eta = 0.2324, \text{depth} = 4, \text{subsample} = 1.0000, \lambda = 0.8295$), achieving composite desirability **$D = 0.6782$** ($d_1 = 0.8783, d_2 = 0.5236$) with predicted validation RMSE $\hat{Y}_1 = 0.4804$ and predicted latency $\hat{Y}_2 = 138.11\,\mu\text{s}$.
6. **Phase 5 (Confirmation & Comparative Benchmarks)**: Across **10 confirmation trials** ($m = 10$ fresh seeds $1001 \dots 1010$), $\mathbf{x}^*_{\text{MO}}$ achieves empirical validation RMSE $0.4853 \pm 0.0115$ (falling inside the Satterthwaite 95% prediction interval $[0.4677, 0.4932]$), empirical latency $142.33 \pm 7.79\,\mu\text{s}$ (inside $[122.82, 153.40]\,\mu\text{s}$), and external holdout test RMSE $0.4888 \pm 0.0032$. At the single-objective depth-7 candidate $\mathbf{x}^*_{\text{SO}}$, empirical validation RMSE is $0.4700 \pm 0.0093$—lying $0.0088$ above the Satterthwaite 95% PI $[0.4354, 0.4612]$ due to $+0.0217$ polynomial optimism bias—while achieving external holdout test RMSE $0.4691 \pm 0.0034$. In the 20-replicate prospective benchmark campaign, Single-Objective TPE achieves the lowest observed test RMSE ($0.46624 \pm 0.00296$) at high latency ($179.96\,\mu\text{s}$), whereas Repeated DOE Single-Objective yields deterministic selection ($0.46960$) at $20.2\%$ lower latency ($143.65\,\mu\text{s}$). In multi-objective optimization, Repeated DOE ($\mathbf{x}^*_{\text{MO}}$) and Multi-Objective TPE show no statistically significant difference in holdout test RMSE ($0.48926$ vs. $0.48996, p = 0.3827$) at ~122 $\mu\text{s}$ latency and 100% constraint feasibility.

---

## 1. Experimental Factors, Coding, and Blocking Architecture

### 1.1 Hyperparameter Space & Natural-to-Coded Transformations

We investigate $k = 4$ hyperparameters of an XGBoost Regressor (`n_estimators = 100`, `objective = 'reg:squarederror'`, `n_jobs = 1`). Factors spanning orders of magnitude ($x_1$ and $x_4$) are mapped via natural logarithmic transformations prior to linear coding into $[-1, +1]$:

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
- **Response $Y_2$ (Single-Sample Inference Latency, $\mu\text{s}$)**: Mean execution time in microseconds per single-row prediction on CPU Core 0 (`SetProcessAffinityMask = 1`, `n_jobs = 1`), timed over 1,000 single-sample calls after 100 untimed warmup calls using `time.perf_counter_ns()`.

---

## 2. Phase 1: $2^4$ Factorial Screening and Curvature Test

### 2.1 Phase 1 ANOVA (Type III Sum of Squares)

Phase 1 evaluates $2^4 = 16$ factorial corners plus $n_C = 4$ center replicates across $b = 5$ blocks ($N_1 = 100$ runs). Fitting the first-order model with two-factor interactions on the factorial runs yields:

| Source of Variation | Sum of Squares (SS) | DF | Mean Square (MS) | $F$-Value | $p$-Value | Partial $\eta^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | $7.309025$ | $1$ | $7.309025$ | $2097.15$ | $< 0.0001$ | — |
| **Block (Seed)** | $0.004628$ | $4$ | $0.001157$ | $0.33$ | $0.8541$ | $0.0154$ |
| **$x_1$ ($\ln \eta$)** | $1.245625$ | $1$ | $1.245625$ | $357.40$ | $< 0.0001$ | $0.8079$ |
| **$x_2$ (`max_depth`)** | $0.082155$ | $1$ | $0.082155$ | $23.57$ | $< 0.0001$ | $0.2171$ |
| **$x_3$ (`subsample`)** | $0.002411$ | $1$ | $0.002411$ | $0.69$ | $0.4084$ | $0.0081$ |
| **$x_4$ ($\ln \lambda$)** | $0.000111$ | $1$ | $0.000111$ | $0.03$ | $0.8588$ | $0.0004$ |
| **$x_1 x_2$ ($\ln \eta \times \text{depth}$)** | $0.091725$ | $1$ | $0.091725$ | $26.32$ | $< 0.0001$ | $0.2364$ |
| **$x_1 x_3$ ($\ln \eta \times \text{subsample}$)** | $0.003382$ | $1$ | $0.003382$ | $0.97$ | $0.3273$ | $0.0113$ |
| **$x_1 x_4$ ($\ln \eta \times \ln \lambda$)** | $0.005259$ | $1$ | $0.005259$ | $1.51$ | $0.2226$ | $0.0174$ |
| **$x_2 x_3$ ($\text{depth} \times \text{subsample}$)** | $0.001095$ | $1$ | $0.001095$ | $0.31$ | $0.5766$ | $0.0037$ |
| **$x_2 x_4$ ($\text{depth} \times \ln \lambda$)** | $0.000001$ | $1$ | $0.000001$ | $0.00$ | $0.9857$ | $0.0000$ |
| **$x_3 x_4$ ($\text{subsample} \times \ln \lambda$)** | $0.000008$ | $1$ | $0.000008$ | $0.00$ | $0.9613$ | $0.0000$ |
| **Residual Error** | $0.296247$ | $85$ | $0.003485$ | — | — | — |

### 2.2 Single-Degree-of-Freedom Curvature Test

To test $H_0: \sum_{i=1}^4 \beta_{ii} = 0$, we compare the mean validation RMSE of the $N_F = 80$ factorial corner runs against the $N_C = 20$ center-point runs against within-block center-point pure error ($\text{df} = 15$):
- **Factorial Mean ($\bar{y}_F$)**: $0.6333$
- **Center Point Mean ($\bar{y}_C$)**: $0.4989$
- **Curvature Contrast ($\bar{y}_F - \bar{y}_C$)**: $+0.1343$ RMSE
- **Curvature Sum of Squares**: $\text{SS}_{\text{Curv}} = \frac{N_F N_C}{N_F + N_C}(\bar{y}_F - \bar{y}_C)^2 = 16 \times (0.134348)^2 = 0.288786$
- **Within-Block Center Pure Error**: $\text{SS}_{\text{PE, center}} = 0.000041$, $\text{df}_{\text{PE}} = 15$, $\text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$
- **Curvature $F$-Statistic**:
$$F_{\text{Curv}} = \frac{\text{SS}_{\text{Curv}}}{\text{MS}_{\text{PE, center}}} = \frac{0.288786}{2.739 \times 10^{-6}} = 105,432.31 \quad (p < 10^{-15})$$
*(Note: Against the pooled 84-df first-order residual $\text{MS}_E = 0.000089$, the auxiliary curvature statistic is $F = 3,253.78, p < 10^{-15}$. Both confirm severe quadratic curvature requiring Phase 2 CCD augmentation.)*

---

## 3. Phase 2: Face-Centered Central Composite Design (FCCD) and Model Adequacy

### 3.1 Second-Order Response Surface ANOVA ($Y_1$: Validation RMSE)

Augmenting Phase 1 with $2k = 8$ axial points ($\alpha = 1.0$) across 5 blocks yields $N = 140$ runs ($28$ runs/block). The fitted second-order model achieves $R^2 = 0.9951$ and $\text{Adjusted } R^2 = 0.9944$:

| Source | SS | DF | MS | $F$-Value | $p$-Value | Partial $\eta^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | $12.164207$ | $1$ | $12.164207$ | $141,134.81$ | $< 0.0001$ | — |
| **Block (Seed)** | $0.006968$ | $4$ | $0.001742$ | $20.21$ | $< 0.0001$ | $0.4005$ |
| **$x_1$ ($\ln \eta$)** | $1.388647$ | $1$ | $1.388647$ | $16,111.76$ | $< 0.0001$ | $0.9925$ |
| **$x_2$ (`max_depth`)** | $0.101704$ | $1$ | $0.101704$ | $1,180.02$ | $< 0.0001$ | $0.9070$ |
| **$x_3$ (`subsample`)** | $0.002013$ | $1$ | $0.002013$ | $23.36$ | $< 0.0001$ | $0.1618$ |
| **$x_4$ ($\ln \lambda$)** | $0.000045$ | $1$ | $0.000045$ | $0.52$ | $0.4712$ | $0.0043$ |
| **$x_1^2$** | $0.146263$ | $1$ | $0.146263$ | $1,697.01$ | $< 0.0001$ | $0.9334$ |
| **$x_2^2$** | $0.008778$ | $1$ | $0.008778$ | $101.85$ | $< 0.0001$ | $0.4570$ |
| **$x_3^2$** | $0.000011$ | $1$ | $0.000011$ | $0.12$ | $0.7279$ | $0.0010$ |
| **$x_4^2$** | $0.000006$ | $1$ | $0.000006$ | $0.07$ | $0.7931$ | $0.0006$ |
| **$x_1 x_2$** | $0.091725$ | $1$ | $0.091725$ | $1,064.22$ | $< 0.0001$ | $0.8979$ |
| **$x_1 x_3$** | $0.001878$ | $1$ | $0.001878$ | $21.79$ | $< 0.0001$ | $0.1526$ |
| **$x_1 x_4$** | $0.007452$ | $1$ | $0.007452$ | $86.46$ | $< 0.0001$ | $0.4168$ |
| **$x_2 x_3$** | $0.001106$ | $1$ | $0.001106$ | $12.83$ | $0.0005$ | $0.0959$ |
| **$x_2 x_4$** | $0.000001$ | $1$ | $0.000001$ | $0.01$ | $0.9094$ | $0.0001$ |
| **$x_3 x_4$** | $0.000008$ | $1$ | $0.000008$ | $0.09$ | $0.7586$ | $0.0008$ |
| **Residual** | $0.010429$ | $121$ | $0.000086$ | — | — | — |

### 3.2 Variance Shielding and Block Intraclass Correlation (ICC)

In Phase 1, unmodeled quadratic curvature inflated the residual mean square ($\text{MS}_E = 0.003485$), masking block differences ($F = 0.33, p = 0.8541$). In Phase 2, accounting for quadratic terms reduces $\text{MS}_E$ by $40\times$ to $0.000086$, exposing highly significant seed-to-seed variance ($F = 20.21, p < 0.0001$). The **Block Intraclass Correlation Coefficient** is:
$$\text{ICC} = \frac{\sigma^2_{\text{block}}}{\sigma^2_{\text{block}} + \sigma^2_\epsilon} = 0.4069 \quad (40.69\%, \quad \text{ICC}_{\text{REML}} = 0.4069, \quad \sigma_{\text{block}} \approx 0.0077\text{ RMSE})$$

### 3.3 Exact Multi-Scale Lack-of-Fit Decomposition and Residual Diagnostics

Partitioning the $121$ residual degrees of freedom into **Structural Lack of Fit** ($10\text{ df}$), **Treatment $\times$ Block Interaction** ($96\text{ df}$), and **Within-Block Center Pure Error** ($15\text{ df}$) yields:

| Response & Source | SS | DF | MS | $F$ vs. Center PE | $p$-Value (PE) | $F$ vs. Additive Baseline | $p$-Value (Add.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$Y_1$ (Val RMSE)**: Total Residual | $0.010429$ | $121$ | $0.000086$ | — | — | — | — |
| $\quad$ Structural Lack of Fit (Treatment Means) | $0.008850$ | $10$ | $0.000885$ | $323.10$ | $< 10^{-15}$ | $62.21$ | $7.78 \times 10^{-41}$ |
| $\quad$ Saturated Additive Baseline Error | $0.001579$ | $111$ | $0.000014$ | — | — | — | — |
| $\quad\quad$ Treatment $\times$ Block Interaction | $0.001538$ | $96$ | $0.000016$ | $5.85$ | $0.0002$ | — | — |
| $\quad\quad$ Within-Block Center Pure Error | $0.000041$ | $15$ | $2.74 \times 10^{-6}$ | — | — | — | — |
| **$Y_2$ (Latency)**: Total Residual | $9818.42$ | $121$ | $81.14$ | — | — | — | — |
| $\quad$ Structural Lack of Fit (Treatment Means) | $917.55$ | $10$ | $91.75$ | $1.20$ | $0.3625$ | $1.15$ | $0.3305$ |
| $\quad$ Saturated Additive Baseline Error | $8900.87$ | $111$ | $80.19$ | — | — | — | — |

Residual adequacy diagnostics for $Y_1$:
- **Normality**: Shapiro-Wilk $W = 0.9971, p = 0.9947$ (unremarkable normal residuals).
- **Homoscedasticity**: Levene across blocks $W = 0.1754, p = 0.9507$; Brown-Forsythe across design groups $W = 0.6872, p = 0.8552$; Breusch-Pagan against fitted values $\text{LM} = 70.10, p < 0.0001$ (reflecting multi-scale variance between center points and boundary corners).
- **Independence & Influence**: Durbin-Watson $d = 1.9918$, Ljung-Box $Q(5) = 4.01, p = 0.5486$, Runs test $p = 0.8576$, maximum Cook's distance $D_{\max} = 0.097 < 1.0$.

---

## 4. Phase 3: Canonical Spectral Analysis and Ridge Optimization

Writing the second-order validation RMSE surface as $\hat{y}(\mathbf{x}) = b_0 + \mathbf{x}^T \mathbf{b} + \mathbf{x}^T \hat{\mathbf{B}} \mathbf{x}$ gives $b_0 = 0.4994$ and linear gradient $\mathbf{b} = [-0.1242, -0.0336, -0.0047, -0.0007]^T$. Solving $\mathbf{x}_0 = -\frac{1}{2}\hat{\mathbf{B}}^{-1}\mathbf{b}$ yields an unconstrained stationary point at $\mathbf{x}_0 = [0.4171, 1.3524, 15.6944, -6.8009]^T$ ($\|\mathbf{x}_0\|_2 = 17.16$, $\hat{y}_0 = 0.4161$), which lies far outside $[-1, +1]^4$ due to flat regularization and subsampling curvature.

Spectral decomposition $\hat{\mathbf{B}} = \mathbf{V}\bm{\Lambda}\mathbf{V}^T$ yields eigenvalues:
$$\lambda_1 = 0.000057, \quad \lambda_2 = 0.000873, \quad \lambda_3 = 0.022759, \quad \lambda_4 = 0.110196$$
Non-parametric block-residual bootstrap ($B = 1,000$) places the 95% confidence interval for $\lambda_1$ at $[-0.0039, +0.0022]$, with $68.2\%$ of bootstrap resamples yielding $\lambda_1 \le 0$. Because the confidence interval for $\lambda_1$ straddles zero, the response surface forms a **stationary/rising ridge system** along the $L_2$ regularization ($x_4$) and subsample ($x_3$) axes, while learning rate ($x_1$) and tree depth ($x_2$) exhibit steep positive convexity.

Constrained L-BFGS-B optimization within $\mathcal{D} = [-1, +1]^4$ restricted to valid integer depths $d \in \{3, \dots, 9\}$ locates the **Single-Objective RSM Candidate ($\mathbf{x}^*_{\text{SO}}$)** at **depth $d = 7$**:
$$\mathbf{x}^*_{\text{SO}} = [0.5983, \ 0.3333, \ 1.0000, \ 1.0000]^T \implies \eta = 0.1515, \ d = 7, \ s = 1.0000, \ \lambda = 10.0000$$
with predicted validation RMSE $\hat{y}(\mathbf{x}^*_{\text{SO}}) = 0.4483$ and Satterthwaite 95% PI $[0.4354, 0.4612]$.

---

## 5. Phase 4: Multi-Objective Derringer-Suich Desirability

To simultaneously minimize **Validation RMSE ($Y_1$)** and **Single-Sample Inference Latency ($Y_2$)**, we apply one-sided Derringer-Suich transformations ($r_1 = r_2 = 1$) with operational specification bounds $Y_1 \in [L_1, U_1] = [0.450, 0.700]$ and $Y_2 \in [L_2, U_2] = [100.0, 180.0]\,\mu\text{s}$, maximizing $D(\mathbf{x}) = \sqrt{d_1(\hat{Y}_1(\mathbf{x})) \cdot d_2(\hat{Y}_2(\mathbf{x}))}$ over valid integer depths:

- **Coded Compromise Coordinate ($\mathbf{x}^*_{\text{MO}}$)**: $[0.8500, \ -0.6667, \ 1.0000, \ -0.0812]^T$
- **Natural Hyperparameters**: `learning_rate` $\eta = 0.2324$, `max_depth` $d = 4$, `subsample` $s = 1.0000$, `reg_lambda` $\lambda = 0.8295$
- **Surrogate Predictions**: $\hat{Y}_1 = 0.4804$ Validation RMSE, $\hat{Y}_2 = 138.11\,\mu\text{s}$ latency
- **Individual & Composite Desirabilities**: $d_1 = 0.8783$, $d_2 = 0.5236$, **$D = 0.6782$**

---

## 6. Phase 5: Empirical Confirmation Trials ($m = 10$ Fresh Seeds)

Both DOE candidate coordinates ($\mathbf{x}^*_{\text{MO}}$ at depth 4 and $\mathbf{x}^*_{\text{SO}}$ at depth 7) were evaluated across **$m = 10$ fresh random seeds** ($\mathcal{S}_{\text{conf}} = \{1001, \dots, 1010\}$) against Satterthwaite-adjusted 95% prediction intervals:

| Candidate Coordinate | Metric | Surrogate $\hat{Y}$ | Satterthwaite 95% PI | Empirical Mean $\pm$ SD ($m = 10$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$\mathbf{x}^*_{\text{MO}}$ (Depth 4)** | Validation RMSE ($Y_1$) | $0.4804$ | $[0.4677, 0.4932]$ | $0.4853 \pm 0.0115$ | **Confirmed** |
| **$\mathbf{x}^*_{\text{MO}}$ (Depth 4)** | Inference Latency ($Y_2, \mu\text{s}$) | $138.11$ | $[122.82, 153.40]$ | $142.33 \pm 7.79$ | **Confirmed** |
| **$\mathbf{x}^*_{\text{MO}}$ (Depth 4)** | External Holdout Test RMSE | — | — | $0.4888 \pm 0.0032$ | Generalization Verified |
| **$\mathbf{x}^*_{\text{SO}}$ (Depth 7)** | Validation RMSE ($Y_1$) | $0.4483$ | $[0.4354, 0.4612]$ | $0.4700 \pm 0.0093$ | **Not Confirmed** ($+0.0217$ bias) |
| **$\mathbf{x}^*_{\text{SO}}$ (Depth 7)** | Inference Latency ($Y_2, \mu\text{s}$) | $186.19$ | $[171.08, 201.30]$ | $171.10 \pm 10.90$ | **Borderline** (on lower bound) |
| **$\mathbf{x}^*_{\text{SO}}$ (Depth 7)** | External Holdout Test RMSE | — | — | $0.4691 \pm 0.0034$ | Generalization Verified |

At $\mathbf{x}^*_{\text{MO}}$ (depth 4), both validation RMSE and latency fall comfortably inside their Satterthwaite 95% prediction intervals. At $\mathbf{x}^*_{\text{SO}}$ (depth 7), empirical validation RMSE ($0.4700$) lies $0.0088$ above the upper prediction interval bound ($0.4612$), confirming that quadratic interpolation across depths $\{3, 6, 9\}$ overestimates accuracy gains at depth 7 by $+0.0217$ RMSE ($6.8\times \text{SE}(\hat{y})$).

---

## 7. Prospective Multi-Replicate Benchmark Campaign (`Revision-v2`)

To evaluate the DOE + RSM methodology against modern heuristic and Bayesian optimizers under equal evaluation budgets (140 model evaluations per search), we executed a prospective benchmark campaign (`results/revision_v2/full_run_001`) comprising **17,077 XGBoost model fits** and **1,640,810 timed single-sample inferences** across $N = 20$ independent search replicates per optimizer.

All 122 winning selection records (95 distinct hyperparameter configurations, 2,440 evaluation rows) were cryptographically frozen in `finalized_selections.json` prior to generalization testing across 20 fresh retraining seeds ($\mathcal{S}_{\text{eval}} = \{2001, \dots, 2020\}$) on the external holdout test set ($N = 4,128$). Whereas the historical `v1.0.0` baseline logged holdout test metrics during exploratory development, the `Revision-v2` pipeline programmatically blocked access to holdout test labels until `finalized_selections.json` was frozen (enforcing pipeline isolation during the revised search, while noting that the same dataset was previously used in the historical baseline).

### 7.1 Holdout Generalization and Latency Comparison

| Optimization Method | Search Replicates ($N$) | Holdout Test RMSE (Mean $\pm$ Search SD) | Holdout Test RMSE [95% CI] | Retrain SD ($\sigma_{\text{seed}}$) | `predict` Latency ($\mu\text{s}$, Mean $\pm$ SD) | `inplace_predict` ($\mu\text{s}$, Mean $\pm$ SD) | Latency Constraint ($\le 145\,\mu\text{s}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Search (140 evals)** | 20 | $0.46941 \pm 0.00297$ | $[0.46802, 0.47080]$ | $0.00338$ | $165.43 \pm 20.11$ | $118.42 \pm 19.72$ | 3/20 (15%) |
| **Single-Objective TPE (140 evals)** | 20 | $0.46624 \pm 0.00296$ | $[0.46486, 0.46763]$ | $0.00342$ | $179.96 \pm 16.92$ | $132.51 \pm 16.58$ | 0/20 (0%) |
| **Constrained TPE ($\le 145\,\mu\text{s}$)** | 20 | $0.47121 \pm 0.00383$ | $[0.46942, 0.47301]$ | $0.00328$ | $144.27 \pm 7.59$ | $98.08 \pm 7.36$ | 9/20 (45%) |
| **Multi-Objective TPE (Desirability)** | 20 | $0.48996 \pm 0.00349$ | $[0.48832, 0.49159]$ | $0.00309$ | $121.52 \pm 4.35$ | $75.77 \pm 4.19$ | 20/20 (100%) |
| **Repeated DOE + RSM ($\mathbf{x}^*_{\text{MO}}$, depth 4)** | 20 | $0.48926 \pm 0.00051$ | $[0.48902, 0.48950]$ | $0.00310$ | $122.30 \pm 1.38$ | $76.47 \pm 1.33$ | 20/20 (100%) |
| **Repeated DOE + RSM ($\mathbf{x}^*_{\text{SO}}$, depth 7)** | 20 | $0.46960 \pm 0.00000$ | $[0.46960, 0.46960]$ | $0.00340$ | $143.65 \pm 1.30$ | $96.95 \pm 1.25$ | 0/20 (0%)* |

*\*Note: Under the 20-seed holdout evaluation protocol, $\mathbf{x}^*_{\text{SO}}$ (depth 7) averaged $143.65 \pm 1.30\,\mu\text{s}$ across interleaved benchmark sessions, whereas in standalone 10-seed confirmation it averaged $171.1 \pm 10.9\,\mu\text{s}$; it is classified as an unconstrained single-objective formulation.*

### 7.2 Multi-Objective Pareto Hypervolume Comparison

| Frontier Cohort & Evaluation Basis | Replicates ($N$) | HV Ref 1 $[0.60, 250.0]$ | HV Ref 2 $[0.65, 275.0]$ | Mean Pareto Size | Statistical Comparison |
| :--- | :---: | :---: | :---: | :---: | :--- |
| *Panel A: Development Candidate Frontiers (Distinct Evaluation Protocols Noted per Row)* | | | | | |
| **MO-TPE Candidate Fronts (Single Split 42, 30-Call Search)** | 20 | $18.3834 \pm 0.3401$ | $31.6531 \pm 0.4810$ | $12.0 \pm 2.4$ | $\Delta = +1.5249, p < 0.0001$ |
| **Repeated DOE Candidate Fronts (5-Block Means Across Replicates)** | 20 | $16.9245 \pm 0.3696$ | $29.6995 \pm 0.5177$ | $7.7 \pm 1.0$ | — |
| **Full DOE Evaluated Frontier (Single Split 42, 27 Configs)** | 1 (Fixed) | $16.8585$ | $29.5918$ | $8.0$ | Baseline Reference |
| *Panel B: External Holdout Test Set (20 Retraining Seeds, 1,000-Call Protocol)* | | | | | |
| **Holdout Non-Dominated Set Among 122 Frozen Selections (95 Distinct Configs)** | 122 Records | $17.1636$ | $29.9328$ | $12$ Configs | 3 DOE MO, 3 MO-TPE, 5 C-TPE, 1 SO-TPE |

### 7.3 Key Scientific Conclusions from the Benchmark Campaign

1. **Single-Objective Accuracy vs. Latency**: Single-Objective TPE achieves the lowest observed mean holdout test RMSE ($0.46624 \pm 0.00296$) by concentrating searches in deeper trees ($d \in \{7, 8, 9\}$), incurring $179.96\,\mu\text{s}$ inference latency. Repeated DOE Single-Objective ($\mathbf{x}^*_{\text{SO}}$, depth 7) selects the exact same configuration deterministically across all 20 replicates (between-search $\text{SD} = 0.00000$, retraining $\sigma = 0.00340$), achieving $0.46960$ Test RMSE at $20.2\%$ lower inference latency ($143.65\,\mu\text{s}$).
2. **Multi-Objective Performance Comparison**: In the latency-constrained multi-objective regime, Repeated DOE ($\mathbf{x}^*_{\text{MO}}$, depth 4) achieves a mean Test RMSE of $0.48926 \pm 0.00051$ at $122.30\,\mu\text{s}$ latency, compared to $0.48996 \pm 0.00349$ at $121.52\,\mu\text{s}$ for Multi-Objective TPE. The difference of $-0.00069$ RMSE is statistically non-significant (Welch's $t = -0.879, p = 0.3827$; paired $t = -0.966, p = 0.3463$), while DOE exhibits $6.8\times$ lower between-search selection variance ($0.00051$ vs. $0.00349$). Both methods achieve $20/20$ ($100\%$) constraint feasibility.
3. **Development Hypervolume vs. Holdout Non-Dominated Set**: On development split 42 under the online 30-call timing protocol, MO-TPE candidate fronts achieve a higher mean hypervolume than the fixed 27-point DOE candidate frontier ($18.3834$ vs. $16.8585, p < 0.0001$; Repeated DOE 5-block means achieve $16.9245 \pm 0.3696$). This demonstrates a higher observed candidate hypervolume on the evaluated development split, without implying universal algorithmic superiority across unconstrained domains or unseen splits. When the 122 frozen selections (95 distinct configurations) are evaluated on the external holdout test set across 20 retraining seeds, the empirical non-dominated set among the evaluated frozen configurations consists of **12 configurations** ($\text{HV} = 17.1636$) spanning both paradigms: **3 Repeated DOE MO, 3 MO-TPE, 5 Constrained TPE, and 1 SO-TPE**.
4. **Constrained TPE Feasibility Degradation**: While Constrained TPE satisfies $\le 145\,\mu\text{s}$ on $20/20$ searches during online 30-call search timing, only **$9/20$ ($45\%$)** remain feasible under 1,000-call benchmark verification ($9/9$ depth-6 selections feasible at $136.62 \pm 3.12\,\mu\text{s}$; $0/11$ depth-7 selections feasible at $150.54 \pm 2.85\,\mu\text{s}$). Plausible factors include noisy 30-call search measurements near the boundary, protocol differences (30 un-warmed calls vs. 1,000 warmed calls), and selection bias ("winner's curse"), rather than a single isolated mechanism.
5. **Historical Latency Comparison Caveat**: Both historical `v1.0.0` scripts (`phase5_confirmation.py` and `phase5_benchmarks.py`) applied Win32 CPU core 0 affinity pinning. Consequently, the shift between historical single-run latency snapshots and the prospective benchmark session cannot be conclusively attributed to unpinned execution or a single identified environmental factor.

---

## 8. Discussion and Methodological Limitations

1. **Parametric Attribution vs. Adaptive Search**: DOE + RSM decomposes variance across main effects, interactions, quadratic terms, and seed blocks, and provides formal hypothesis tests for curvature ($F = 105,432.31$) and lack of fit ($F = 62.21$). Conversely, Bayesian optimization (TPE) adapts dynamically to non-polynomial basins without parametric assumptions.
2. **Value of Stochastic Nuisance Blocking**: Blocking across 5 data-partition seeds absorbed $\text{ICC} = 40.69\%$ of residual variance ($\sigma_{\text{block}} \approx 0.0077$ RMSE), preventing seed noise from confounding hyperparameter comparisons.
3. **Structural Lack of Fit and Surrogate Optimism**: Because decision tree ensembles exhibit diminishing returns at deeper levels ($d \ge 6$), a second-order polynomial interpolated across $d \in \{3, 6, 9\}$ under-predicts validation RMSE at depth 7 by $+0.0217$ RMSE. Satterthwaite prediction intervals widen for variance heterogeneity across degrees of freedom but cannot correct deterministic polynomial bias.
4. **Dimensionality Scaling**: While a 4-factor FCCD requires $2^4 + 2(4) + 4 = 28$ runs per block, full factorials scale as $2^k$. For higher-dimensional spaces ($k > 6$), Resolution IV/V fractional factorials ($2^{k-p}$), Box-Behnken designs, or hybrid DOE-screening + Bayesian refinement pipelines are recommended.

---

## 9. Repository Artifacts and Reproducibility

- **LaTeX Manuscript & Compiled PDF**: `report.tex` and `report.pdf` (22 pages, compiled via Tectonic with zero unresolved references or layout overflows).
- **Auto-Generated Statistical Macros & Tables**: `results/macros.tex` (281 macros) and `tables/*.tex`, generated deterministically via `python scripts/generate_report_artifacts.py`.
- **Prospective Benchmark Evidence**: `results/revision_v2/full_run_001/` (`search_trials.parquet`, `search_selected.csv`, `finalized_selections.json`, `final_eval_runs.csv`, `final_eval_summary.csv`, `pareto_Summary.json`, `run_manifest.json`).
- **Verification & Audit Suite**: `python scripts/audit_scientific_consistency.py` and `pytest` (117 automated integrity, statistical, and consistency checks).
