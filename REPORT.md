# Sequential Response Surface Methodology and Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking

**Author:** Christos Chousein Sounios

---

## Executive Summary

This report presents a comprehensive Design of Experiments (DOE) and Response Surface Methodology (RSM) study applied to the multi-objective hyperparameter optimization of an **XGBoost Regressor** on the **California Housing** benchmark dataset ($N = 20,640$ observations, $8$ continuous features). Following the methodology of Douglas C. Montgomery's *Design and Analysis of Experiments* (9th ed., 2017, Chapters 5, 9, 10, and 14), we treat stochastic machine learning variability—arising from random train/validation partitioning and row subsampling—as a **nuisance factor** controlled via a **Randomized Complete Block Design (RCBD)** across $b = 5$ seed blocks ($\mathcal{S} = \{42, 101, 202, 303, 404\}$).

All primary DOE phases consist of **150 genuine model training and evaluation runs** (100 in Phase 1, 40 axial augmentations in Phase 2, and 10 confirmation trials in Phase 5 for $\mathbf{x}^*_{\text{MO}}$, supplemented by 10 confirmation trials for $\mathbf{x}^*_{\text{SO}}$), alongside a prospective 20-replicate benchmark campaign comprising **17,077 XGBoost model fits** and **1,640,810 timed single-sample inferences**:

1. **Holdout Isolation and Dataset History**: The 20,640 dataset observations are partitioned into an 80% development pool ($N_{\text{dev}} = 16,512$) and a 20% external holdout test set ($N_{\text{test}} = 4,128$). Within each block $b$, the development pool is split 75/25 into training ($N_{\text{train}} = 12,384$) and validation ($N_{\text{val}} = 4,128$) sets. Response $Y_1$ is **Development Validation RMSE** on the 25% validation split. Whereas the historical `v1.0.0` baseline experiments recorded holdout test results during development, the revised `Revision-v2` pipeline programmatically prevented access to those test labels during all search iterations, surrogate fitting, curvature testing, lack-of-fit analysis, and desirability optimization until winning configurations were frozen in `finalized_selections.json`. These safeguards enforce programmatic pipeline isolation during the revised search, although they cannot retroactively undo earlier historical exposure of the same dataset.
2. **Phase 1 ($2^4$ Full Factorial + Center Points in 5 Blocks, $N_1 = 100$ runs)**: Screening identifies **Learning Rate** ($\ln \eta$, $F = 357.40, p < 10^{-15}, \eta^2_p = 0.8079$) and **Max Tree Depth** ($F = 23.57, p < 0.0001, \eta^2_p = 0.2171$), along with their interaction $x_1 x_2$ ($F = 26.32, p < 0.0001, \eta^2_p = 0.2364$), as the dominant drivers of validation RMSE. Single-degree-of-freedom curvature testing against within-block center-point pure error ($\text{df} = 15, \text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$) reveals strong quadratic curvature ($F_{\text{Curv}} = 105,432.31, p < 10^{-15}$), where the factorial corner mean ($\bar{y}_F = 0.6333$) exceeds the center mean ($\bar{y}_C = 0.4989$) by $0.1343$ RMSE.
3. **Phase 2 (Face-Centered Central Composite Design, $\alpha = 1.0$, $N_{\text{CCD}} = 140$ runs)**: Augmenting with $40$ axial points fits a full second-order polynomial ($R^2 = 0.9951, \text{Adj } R^2 = 0.9944$). Once quadratic curvature is modeled, the RCBD block effect becomes highly significant ($F = 20.21, p < 0.0001$), absorbing **ICC = 40.69%** ($\text{ICC}_{\text{REML}} = 0.4069, \sigma_{\text{block}} \approx 0.0077$ RMSE) of unexplained residual variance. Formal lack-of-fit decomposition shows statistically significant structural polynomial misfit ($F_{\text{LoF}} = 62.21, p = 7.78 \times 10^{-41}$ against the 111-df saturated additive baseline; $F_{\text{LoF}} = 323.10, p < 10^{-15}$ against 15-df center pure error; RMS misfit $= 0.0297$ RMSE), demonstrating that second-order polynomials serve as local guidance maps rather than globally exact estimators of gradient boosting loss.
4. **Phase 3 (Canonical & Ridge Analysis)**: Spectral decomposition of $\hat{\mathbf{B}}$ yields three positive eigenvalues and one near-zero eigenvalue ($\lambda = \{0.000057, 0.000873, 0.022759, 0.110196\}$; wild bootstrap 95% CI for $\lambda_1$: $[-0.0039, +0.0022]$, with $68.2\%$ of resamples $\le 0$), characterizing a **stationary/rising ridge system**. The unconstrained stationary point ($\|\mathbf{x}_0\| = 17.16$) lies outside $[-1, +1]^4$; constrained optimization over valid integer depths identifies the single-objective candidate $\mathbf{x}^*_{\text{SO}}$ at depth $7$ ($x_1 = 0.5983, x_2 = 0.3333, x_3 = 1.0000, x_4 = 1.0000$, i.e., $\eta = 0.1515, \text{depth} = 7, \text{subsample} = 1.0000, \lambda = 10.0000$) with predicted validation RMSE $\hat{y} = 0.4483$.
5. **Phase 4 (Derringer-Suich Multi-Objective Desirability)**: Balancing Validation RMSE ($Y_1 \in [0.450, 0.700]$) and Single-Sample Inference Latency ($Y_2 \in [100.0, 180.0]\,\mu\text{s}$) yields an interior compromise coordinate $\mathbf{x}^*_{\text{MO}} = [0.8500, -0.6667, 1.0000, -0.0812]^T$ ($\eta = 0.2324, \text{depth} = 4, \text{subsample} = 1.0000, \lambda = 0.8295$), achieving composite desirability **$D = 0.6782$** ($d_1 = 0.8783, d_2 = 0.5236$) with predicted validation RMSE $\hat{Y}_1 = 0.4804$ and predicted latency $\hat{Y}_2 = 138.11\,\mu\text{s}$.
6. **Phase 5 (Confirmation & Comparative Benchmarks)**: Across **10 confirmation trials** ($m = 10$ fresh seeds $\mathcal{S}_{\text{conf}} = \{505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414\}$), $\mathbf{x}^*_{\text{MO}}$ achieves empirical validation RMSE $0.4853 \pm 0.0115$ (falling inside the Satterthwaite 95% prediction interval $[0.4677, 0.4932]$), empirical latency $142.33 \pm 7.79\,\mu\text{s}$ ($142.3 \pm 7.8\,\mu\text{s}$, inside $[122.82, 153.40]\,\mu\text{s}$), and external holdout test RMSE $0.4888 \pm 0.0032$. At the single-objective depth-7 candidate $\mathbf{x}^*_{\text{SO}}$, empirical validation RMSE is $0.4700 \pm 0.0093$—lying $0.0088$ above the Satterthwaite 95% PI $[0.4354, 0.4612]$ due to $+0.0217$ polynomial optimism bias—while achieving external holdout test RMSE $0.4691 \pm 0.0034$ and borderline latency $171.1 \pm 10.9\,\mu\text{s}$ (at the lower bound of $[171.1, 202.9]\,\mu\text{s}$). In the 20-replicate prospective benchmark campaign, Single-Objective TPE achieves the lowest observed test RMSE ($0.46691 \pm 0.00116$) at high latency ($188.93 \pm 12.07\,\mu\text{s}$), whereas Repeated DOE Single-Objective yields deterministic selection ($0.46859 \pm 0.00000$, retraining $\sigma_{\text{eval}} = 0.00304$) at $20.2\%$ lower latency ($150.74 \pm 0.72\,\mu\text{s}$). In multi-objective optimization, Repeated DOE ($\mathbf{x}^*_{\text{MO}}$) and Multi-Objective TPE show no statistically significant difference in holdout test RMSE ($0.48997 \pm 0.00732$ vs. $0.49255 \pm 0.01080$, difference $-0.00258, p = 0.3826$) at comparable latency ($122.29 \pm 2.70\,\mu\text{s}$ vs. $121.46 \pm 3.75\,\mu\text{s}$) and $100\%$ ($20/20$) constraint feasibility.

---

## 1. Experimental Factors, Coding, and Blocking Architecture

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
- **Response $Y_2$ (Single-Sample Inference Latency, $\mu\text{s}$)**: Mean execution time in microseconds per single-row prediction on CPU Core 0 (`SetProcessAffinityMask = 1`, `n_jobs = 1`), timed over 1,000 single-sample calls after 100 untimed warmup calls (50 per prediction interface: `predict` and `inplace_predict`) using `time.perf_counter_ns()`.

---

## 2. Phase 1: $2^4$ Factorial Screening and Curvature Test

### 2.1 Phase 1 ANOVA (Type III Sum of Squares)

Phase 1 evaluates $2^4 = 16$ factorial corners plus $n_C = 4$ center replicates across $b = 5$ blocks ($N_1 = 100$ runs). Fitting the first-order model with two-factor interactions yields:

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
- **Curvature Contrast ($\bar{y}_F - \bar{y}_C$)**: $+0.1343$ RMSE
- **Curvature Sum of Squares**: $\text{SS}_{\text{Curv}} = \frac{n_F n_C}{n_F + n_C}(\bar{y}_F - \bar{y}_C)^2 = 0.288786$
- **Within-Block Center Pure Error**: $\text{SS}_{\text{PE, center}} = 0.000041$, $\text{df}_{\text{PE}} = 15$, $\text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$
- **Curvature $F$-Statistic**:
$$F_{\text{Curv}} = \frac{\text{SS}_{\text{Curv}}}{\text{MS}_{\text{PE, center}}} = \frac{0.288786}{2.74 \times 10^{-6}} = 105,432.31 \quad (p < 10^{-15})$$
*(Note: If evaluated against overall pure error pooled across all replicated design coordinates, $\text{df} = 83, \text{MS}_{\text{PE, All}} = 6.98 \times 10^{-5}$, the $F$-statistic is $F \approx 4,140, p < 10^{-15}$. Both confirm severe quadratic curvature requiring Phase 2 CCD augmentation.)*

---

## 3. Phase 2: Face-Centered Central Composite Design (FCCD) and Model Adequacy

### 3.1 Second-Order Response Surface ANOVA ($Y_1$: Validation RMSE and $Y_2$: Inference Latency)

Augmenting Phase 1 with $2k = 8$ axial points ($\alpha = 1.0$) across 5 blocks yields $N = 140$ runs ($28$ runs/block). The fitted second-order validation RMSE model achieves $R^2 = 0.9951$ and $\text{Adjusted } R^2 = 0.9944$:

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

Residual adequacy diagnostics for $Y_1$:
- **Normality**: Shapiro-Wilk $W = 0.9971, p = 0.9947$ (normal residuals).
- **Homoscedasticity**: Levene across blocks $W = 0.1754, p = 0.9507$; Brown-Forsythe across design groups $W = 0.6872, p = 0.8552$; Breusch-Pagan against fitted values $\text{LM} = 70.10, p < 0.0001$ (reflecting multi-scale variance across the factor domain, addressed via HC3 robust standard errors).
- **Independence & Influence**: Durbin-Watson $DW = 1.9918$, Ljung-Box $Q = 4.01, p = 0.5486$, Runs test $p = 0.8576$, maximum Cook's distance $D_{\max} = 0.097 < 1.0$.

---

## 4. Phase 3: Canonical Spectral Analysis and Ridge Optimization

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

---

## 5. Phase 4: Multi-Objective Derringer-Suich Desirability

To simultaneously minimize **Validation RMSE ($Y_1$)** and **Single-Sample Inference Latency ($Y_2$)**, we apply one-sided Derringer-Suich transformations ($s_1 = s_2 = 1.0$, equal weights $w_1 = w_2 = 1.0$) with operational specification bounds $Y_1 \in [L_1, U_1] = [0.450, 0.700]$ and $Y_2 \in [L_2, U_2] = [100.0, 180.0]\,\mu\text{s}$, maximizing $D(\mathbf{x}) = \sqrt{d_1(\hat{Y}_1(\mathbf{x})) \cdot d_2(\hat{Y}_2(\mathbf{x}))}$ over valid integer depths:

- **Coded Compromise Coordinate ($\mathbf{x}^*_{\text{MO}}$)**: $[0.8500, \ -0.6667, \ 1.0000, \ -0.0812]^T$
- **Natural Hyperparameters**: `learning_rate` $\eta = 0.2324$, `max_depth` $d = 4$, `subsample` $s = 1.0000$, `reg_lambda` $\lambda = 0.8295$
- **Surrogate Predictions**: $\hat{Y}_1 = 0.4804$ Validation RMSE, $\hat{Y}_2 = 138.11\,\mu\text{s}$ latency
- **Individual & Composite Desirabilities**: $d_1 = 0.8783$, $d_2 = 0.5236$, **$D = 0.6782$**

---

## 6. Phase 5: Empirical Confirmation Trials ($m = 10$ Fresh Seeds)

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

---

## 7. Prospective Multi-Replicate Benchmark Campaign (`Revision-v2`)

To evaluate the DOE + RSM methodology against modern heuristic and Bayesian optimizers under strictly equal evaluation budgets (140 model evaluations per search), we executed a prospective benchmark campaign (`results/revision_v2/full_run_001/`) comprising **17,077 XGBoost model fits** and **1,640,810 timed single-sample inferences** across $N = 20$ independent search replicates per optimizer.

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
| **Historical Constrained TPE** | Parzen (Single) | $0.4783$ | $0.4823$ | — | — | $125.4$ | — | Yes |
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
| **Difference: Fixed Full DOE Front $-$ MO-TPE Mean (Split 42)** | — | — | $-1.5249$ ($p < 0.0001$) | $-1.9105$ ($p < 0.0001$) |
| **Panel B: Holdout Non-Dominated Set (122 Records, 95 Configs, 20 Seeds)** | | | | |
| **Non-Dominated Set (12 Distinct Configs)** | $122$ | $12$ | $17.6714$ | $28.9900$ |
| $\quad \llcorner$ Repeated DOE MO Selections | $20$ | $3$ | — | — |
| $\quad \llcorner$ Multi-Objective TPE Selections | $20$ | $3$ | — | — |
| $\quad \llcorner$ Constrained TPE Selections | $20$ | $5$ | — | — |
| $\quad \llcorner$ Single-Objective TPE Selections | $20$ | $1$ | — | — |

### 7.3 Key Scientific Conclusions from the Benchmark Campaign

1. **Single-Objective Accuracy vs. Latency**: Single-Objective TPE achieves the numerically lowest observed mean holdout test RMSE ($0.46691 \pm 0.00116$) by concentrating searches in deeper trees ($d \in \{7, 8, 9\}$), incurring $188.93 \pm 12.07\,\mu\text{s}$ inference latency (`inplace_predict`: $157.14 \pm 12.11\,\mu\text{s}$). Neither Single-Objective TPE nor Repeated DOE Single-Objective satisfies the $\le 145\,\mu\text{s}$ latency constraint ($0/20$ feasible). Repeated DOE Single-Objective ($\mathbf{x}^*_{\text{SO}}$, depth 7) selects the exact same configuration deterministically across all 20 replicates (between-search $\text{SD} = 0.00000$, retraining $\sigma_{\text{eval}} = 0.00304$), achieving $0.46859$ Test RMSE at $150.74 \pm 0.72\,\mu\text{s}$ inference latency (`inplace_predict`: $118.78 \pm 0.59\,\mu\text{s}$, $20.2\%$ lower latency than SO-TPE).
2. **Multi-Objective Performance Comparison**: In the latency-constrained multi-objective regime, Repeated DOE ($\mathbf{x}^*_{\text{MO}}$, depth 4) achieves a mean Test RMSE of $0.48997 \pm 0.00732$ at $122.29 \pm 2.70\,\mu\text{s}$ latency (`inplace_predict`: $89.73 \pm 2.73\,\mu\text{s}$), compared to $0.49255 \pm 0.01080$ at $121.46 \pm 3.75\,\mu\text{s}$ (`inplace_predict`: $88.84 \pm 3.76\,\mu\text{s}$) for Multi-Objective TPE. The difference of $-0.00258$ RMSE is statistically non-significant under both Welch's two-sample $t$-test ($t = -0.8849, p = 0.3826$) and paired $t$-testing across search replicates ($t = -0.966, p = 0.3463$). Both methods achieve $20/20$ ($100\%$) benchmark constraint feasibility.
3. **Development Hypervolume vs. Holdout Non-Dominated Set**: On development split 42 under the online 30-call timing protocol, MO-TPE candidate fronts achieve a mean hypervolume of $18.3834 \pm 0.3301$ at reference $[0.60, 250.0]$ ($30.0535 \pm 0.4123$ at $[0.65, 275.0]$), exceeding the fixed 27-point DOE candidate frontier ($16.8585$, difference $-1.5249, p < 0.0001$; at $[0.65, 275.0]$, $28.1430$, difference $-1.9105, p < 0.0001$), while Repeated DOE 5-block means achieve $15.6128 \pm 0.7268$ ($26.6847 \pm 0.8786$ at $[0.65, 275.0]$). This demonstrates a higher observed candidate hypervolume on the evaluated development split under 30-call search timing, without implying universal algorithmic superiority across unconstrained domains or unseen splits. When the 122 frozen selection records (95 distinct configurations) are evaluated on the external holdout test set across 20 retraining seeds, the empirical non-dominated set among the evaluated frozen configurations consists of **12 distinct configurations** ($\text{HV} = 17.6714$ at $[0.60, 250.0]$ and $28.9900$ at $[0.65, 275.0]$) spanning both paradigms: **3 Repeated DOE MO, 3 MO-TPE, 5 Constrained TPE, and 1 SO-TPE**.
4. **Constrained TPE Feasibility Degradation**: While Constrained TPE satisfies $\le 145\,\mu\text{s}$ on $20/20$ searches during online 30-call search timing, only **$9/20$ ($45\%$)** remain feasible under 1,000-call benchmark verification ($9/9$ depth-6 selections feasible at $136.70 \pm 0.82\,\mu\text{s}$; $0/11$ depth-7 selections feasible at $150.45 \pm 0.67\,\mu\text{s}$). Plausible factors include noisy 30-call search measurements near the boundary, protocol differences (30 calls without warmup vs. 1,000 calls with warmup), and selection bias, rather than a single conclusively isolated cause.
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
- **Auto-Generated Statistical Macros & Tables**: `results/macros.tex` (283 macros) and `tables/*.tex`, generated deterministically via `python scripts/generate_report_artifacts.py`.
- **Prospective Benchmark Evidence**: `results/revision_v2/full_run_001/` (`finalized_selections.json`, `final_evaluations.csv`, `final_summary.csv`, `optimizer_summary.json`, `hypervolume.json`, `paired_comparisons.json`, `computational_budget.json`, `run_manifest.json`).
- **Verification & Audit Suite**: `python scripts/audit_scientific_consistency.py` and `pytest` (automated integrity, statistical, and numerical table consistency checks).
