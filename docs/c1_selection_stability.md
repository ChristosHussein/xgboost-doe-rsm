# Audit D: Separation of DOE Selection Stability from Prediction Uncertainty

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Audited Dataset:** `results/revision_v2/full_run_001/`  
**Date:** October 10, 2026  

---

## 1. Executive Summary & Definite Forensic Finding

The full experiment reports:
- Repeated DOE Single-Objective Test RMSE: **$0.46859$**
- Between-search Standard Deviation: **$0.00000$**

**Audit Finding:**
1. **Source of Zero Between-Search Variance:** All 20 independent repetitions of Repeated DOE Single-Objective selected the **exact same hyperparameter configuration** ($\text{SHA-256: } \texttt{9d7c4e74...}$, $\eta = 0.15195$, $\text{max\_depth} = 7$, $\text{subsample} = 1.0$, $\lambda = 10.0$). Because final evaluation measures each frozen configuration across the same fixed set of 20 evaluation seeds ($2001$–$2020$) on the common holdout test set, the resulting mean test RMSE across those 20 seeds is algebraically identical for all 20 replicate rows.
2. **Non-Zero Predictive Uncertainty:** The underlying XGBoost model trained with that hyperparameter configuration **does not have zero uncertainty**. When retrained across the 20 evaluation seeds, its holdout test RMSE ranges from $0.4646$ to $0.4745$, with a **conditional retraining standard deviation of $0.003037$** ($95\%$ CI: $[0.4672, 0.4699]$).
3. **Mandatory Scientific Framing:** The metric $\text{SD} = 0.00000$ reflects **between-search selection stability** of the response surface optimum across nuisance block sets. It must never be interpreted as zero predictive uncertainty, zero generalization variance, or statistical infallibility.
4. **Approved Terminology:** The manuscript must describe this evidence accurately:
   > *"DOE single-objective selection was perfectly stable across all 20 repeated design seed sets, selecting the exact same boundary configuration ($d=7, \eta=0.152, s=1.0, \lambda=10.0$). Conditional on this selection, retraining variability across holdout evaluation seeds was $\text{SD} = 0.00304$."*

---

## 2. Multi-Level Variance Decomposition Framework

To prevent conflating search variance with retraining variance, the study establishes three separate, non-overlapping variance quantities:

```
Total Empirical Performance Variation
 ├── 1. Configuration-Selection Variability (Which hyperparameter points are chosen?)
 ├── 2. Between-Search Variability (How does expected performance shift across search replicates?)
 └── 3. Conditional Retraining Variability (How does holdout performance vary across data splits / seeds?)
```

---

## 3. Empirical Results Across All Optimization Methods

### D1. Configuration-Selection Variability

| Optimization Method | Total Replicates | Unique Configs Selected | Selection Entropy / Stability | Depth Distribution |
|---|---:|---:|---|---|
| **Repeated DOE Single-Objective** | 20 | **1** (100% identical) | Perfectly stable ($20/20$ replicate coincidence) | $100\%$ Depth 7 |
| **Repeated DOE Multi-Objective** | 20 | **12** ($60\%$ unique) | Moderate stability ($15/20$ select Depth 4, $5/20$ Depth 5) | $75\%$ Depth 4, $25\%$ Depth 5 |
| **Single-Objective TPE** | 20 | **20** ($100\%$ unique)| Stochastic exploration | $70\%$ Depth 9, $30\%$ Depth 8 |
| **Constrained TPE ($\le 145\,\mu\text{s}$)** | 20 | **20** ($100\%$ unique)| Stochastic exploration | $55\%$ Depth 7, $45\%$ Depth 6 |
| **Multi-Objective TPE** | 20 | **20** ($100\%$ unique)| Stochastic exploration | $50\%$ Depth 4, $25\%$ Depth 5, $25\%$ Depth 3 |
| **Random Search** | 20 | **20** ($100\%$ unique)| Purely stochastic | $45\%$ Depth 9, $40\%$ Depth 8, $15\%$ Depth 7 |

**Insight:** Because the central composite design enforces a fixed, rigid geometric grid, the single-objective quadratic response surface consistently estimated its unconstrained gradient pointing toward the upper-bound boundary point ($\text{max\_depth} = 7$, $\lambda = 10.0$, $\text{subsample} = 1.0$), resulting in identical analytical grid optima across all 20 nuisance block sets. In contrast, multi-objective desirability balancing RMSE and latency introduces subtle curvature sensitivity in the trade-off knee, producing 12 unique configurations across the 20 block sets.

---

### D2. Between-Search Performance Variability ($N = 20$ Search Replicates)

Measures the standard deviation of replicate-level mean test RMSE across the 20 independent optimization executions:

| Optimization Method | Replicate Mean Test RMSE | Between-Search SD | Between-Search IQR | Replicate Range [Min, Max] |
|---|---:|---:|---:|---|
| **Repeated DOE Single-Objective** | 0.46859 | **0.00000** | 0.00000 | [0.46859, 0.46859] |
| **Single-Objective TPE** | 0.46691 | **0.00116** | 0.00164 | [0.46505, 0.46968] |
| **Random Search** | 0.46953 | **0.00207** | 0.00282 | [0.46535, 0.47355] |
| **Constrained TPE** | 0.47036 | **0.00324** | 0.00445 | [0.46556, 0.47648] |
| **Repeated DOE Multi-Objective** | 0.48997 | **0.00732** | 0.01007 | [0.47511, 0.49969] |
| **Multi-Objective TPE** | 0.49255 | **0.01080** | 0.01509 | [0.47321, 0.50974] |

**Insight:** Between-search variance is lowest in single-objective search and highest in multi-objective trade-off selection:
- In single-objective search, both TPE ($\text{SD} = 0.00116$) and DOE ($\text{SD} = 0.00000$) converge reliably to the high-depth region.
- In multi-objective optimization, both DOE ($\text{SD} = 0.00732$) and MO-TPE ($\text{SD} = 0.01080$) exhibit higher between-search variation because small differences in surrogate curvature or sample trade-offs shift the chosen point along the Pareto curve (e.g., trading $2\,\mu\text{s}$ latency for $0.01$ RMSE).

---

### D3. Conditional Retraining Variability ($N = 20$ Evaluation Seeds per Configuration)

Measures the standard deviation of test RMSE across 20 distinct data splits and model initializations for each frozen configuration:

| Optimization Method | Configurations Evaluated | Mean Retraining SD | Min Retraining SD | Max Retraining SD |
|---|---:|---:|---:|---:|
| **Repeated DOE Single-Objective** | 20 (1 unique) | **0.003037** | 0.003037 | 0.003037 |
| **Single-Objective TPE** | 20 (20 unique)| **0.003141** | 0.002068 | 0.004093 |
| **Random Search** | 20 (20 unique)| **0.002975** | 0.002291 | 0.003595 |
| **Constrained TPE** | 20 (20 unique)| **0.003411** | 0.002362 | 0.004348 |
| **Repeated DOE Multi-Objective** | 20 (12 unique)| **0.003301** | 0.002331 | 0.004377 |
| **Multi-Objective TPE** | 20 (20 unique)| **0.003578** | 0.001963 | 0.006080 |

**Insight:**
- Conditional retraining variability is an inherent property of XGBoost model fitting under stochastic data partitioning and feature subsampling on the California Housing dataset.
- Across all methods and tree depths, **retraining standard deviation averages $\approx 0.0030 - 0.0036$**.
- DOE Single-Objective's configuration has a retraining SD of $0.003037$, exactly in line with all other configurations in the study.

---

## 4. Synthesis for Manuscript Text and Claims

| Claim / Statement | Scientific Status | Audit Recommendation |
|---|---|---|
| *"DOE has zero uncertainty."* | **False / Unsupported** | Strictly prohibited. |
| *"DOE single-objective selection was perfectly stable across 20 nuisance block sets."* | **Directly Verified** | Approved description of empirical evidence. |
| *"DOE Single-Objective has lower between-search variance than Random Search or TPE."* | **Directly Verified** | Approved, provided it is noted that this arises from deterministic boundary selection. |
| *"The performance of the selected DOE configuration is known without error."* | **False / Unsupported** | Strictly prohibited. Retraining standard deviation ($0.00304$) must be reported alongside mean ($0.46859$). |
