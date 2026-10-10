# Audit B: Computational Efficiency and Experimental Budget Reconciliation

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Audited Dataset:** `results/revision_v2/full_run_001/`  
**Reconciliation Artifact:** `results/revision_v2/scientific_audit_c1/budget_reconciliation.json`  
**Date:** October 10, 2026  

---

## 1. Executive Summary & Core Correction

Prior preliminary summaries claimed that Response Surface Methodology (DOE/RSM) achieves a "fivefold computational efficiency advantage" over Bayesian Optimization (TPE).

**Audit Finding: This claim is methodologically flawed and must be withdrawn.**
- In the full experimental revision, each complete repeated DOE replicate comprises **5 nuisance blocks $\times$ 28 model fits = 140 model fits**.
- Each TPE search replicate likewise executes **140 model fits** (140 sequential trials).
- Therefore, **the primary comparative protocol operates under exactly equal model-evaluation budgets (140 fits per replicate)**.
- In wall-clock execution time, **MO-TPE was actually faster than Repeated DOE (26.24 seconds vs 36.49 seconds)** on the benchmark hardware.
- The true methodological distinction is **allocation of evaluation budget**, not evaluation count:
  - DOE allocates 140 fits to **25 unique geometric configurations** with structured 5-fold replication across nuisance blocks, enabling formal estimation of nuisance block effects, curvature, and interaction terms.
  - TPE allocates 140 fits to **140 unique hyperparameter configurations** across the global 4D domain, prioritizing exploratory coverage and non-dominated Pareto frontier discovery.

---

## 2. Reconciled Computational Budget Accounting

The raw execution records and ledger files across `results/revision_v2/full_run_001/` were audited and reconciled:

### A. Total Model Fits Reconciliation ($N = 17,077$)

| Pipeline Stage | Experimental Unit | Derivation | Verified File | Actual Row Count |
|---|---|---|---|---:|
| **Optimizer Search Fits** | 4 optimizers $\times$ 20 replicates | $4 \times 20 \times 140$ trials | `optimizer_trials.csv` | 11,200 |
| **Repeated DOE Fits** | 20 replicates $\times$ 5 blocks | $20 \times 5 \times 28$ design runs | `doe_selection_runs.csv` | 2,800 |
| **Historical Matched DOE** | 25 coordinates + 2 selections | $25 + 2$ candidate evaluations | `doe_matched_candidate_front.csv` | 27 |
| **Primary Latency Refits** | 122 frozen configs $\times$ 5 sessions | $122 \times 5$ model fits | `latency_measurement.json` | 610 |
| **Final Holdout Retrainings**| 122 frozen configs $\times$ 20 seeds | $122 \times 20$ retrainings | `final_evaluations.csv` | 2,440 |
| **Total Model Fits** | | | | **17,077** |

### B. Resolution of the "54 Historical Fits" Narrative Discrepancy
- In previous session notes, a passing mention referenced "54 historical matched fits."
- **Audit Verification:** The schema in `computational_budget.json` records:
  - `matched_historical_doe_candidate_fits`: 27
  - `matched_historical_doe_coordinate_fits`: 25
  - `matched_historical_doe_selected_point_fits`: 2
- The 25 coordinate fits and 2 selected points sum to 27 total candidates. Adding $27 + (25 + 2)$ results in 54 through accidental double-counting.
- **Evidence:** `doe_matched_candidate_front.csv` contains exactly **27 rows**, and the execution log confirms that exactly **27 genuine fits** were performed. The mention of 54 was an informal arithmetic double-counting error, not an actual discrepancy in experimental execution.

### C. Latency Prediction Calls Reconciliation

| Latency Timing Mode | Measurements | Timed Calls / Meas. | Total Timed Calls | Warmup Calls / Meas. | Total Warmup Calls | Total Calls |
|---|---:|---:|---:|---:|---:|---:|
| **Online Search (4 opts)** | 11,200 | 30 | 336,000 | 10 | 112,000 | 448,000 |
| **Online Repeated DOE** | 2,800 | 30 | 84,000 | 10 | 28,000 | 112,000 |
| **Online Matched DOE** | 27 | 30 | 810 | 10 | 270 | 1,080 |
| **Primary Benchmark (5 sess $\times$ 2 APIs)**| 1,220 | 1,000 | 1,220,000 | 50 | 61,000 | 1,281,000 |
| **Total** | | | **1,640,810** | | **201,270** | **1,842,080** |

---

## 3. Fair Per-Replicate Efficiency Comparison (140-Fit Budget)

Comparing a single complete replicate of each optimization strategy under the equal 140-fit evaluation budget:

| Method | Total Fits | Unique Configs | Replications per Config | Mean Fit Time | Wall-Clock Time (Mean $\pm$ SD) | Latency Eval Time | Surrogate Overhead |
|---|---:|---:|---|---:|---:|---:|---:|
| **Repeated Pre-planned DOE** | 140 | 25 | 5-fold across blocks | 238.6 ms | $36.49\,\text{s} \pm 0.19\,\text{s}$ | 1.60 s | 1.48 s (OLS + grid) |
| **Multi-Objective TPE** | 140 | 140 | None (1 per trial) | 174.2 ms | $26.24\,\text{s} \pm 1.80\,\text{s}$ | 1.42 s | 0.43 s (KDE / Parzen) |
| **Constrained TPE** | 140 | 140 | None (1 per trial) | 176.9 ms | $26.50\,\text{s} \pm 2.70\,\text{s}$ | 1.50 s | 0.41 s (KDE / Parzen) |
| **Random Search** | 140 | 140 | None (1 per trial) | 221.9 ms | $32.61\,\text{s} \pm 1.63\,\text{s}$ | 1.53 s | 0.00 s |
| **Single-Objective TPE** | 140 | 140 | None (1 per trial) | 370.2 ms | $54.03\,\text{s} \pm 3.05\,\text{s}$ | 1.97 s | 0.38 s (KDE / Parzen) |

### Key Analytical Insights:
1. **Wall-Clock Optimization Efficiency:**
   - Multi-Objective TPE ran in **$26.24\,\text{s}$**, compared to Repeated DOE's **$36.49\,\text{s}$** (TPE was $28\%$ faster in wall-clock time).
   - This occurs because TPE evaluates smaller depths on average during multi-objective exploration ($174.2\,\text{ms}$ per fit vs $238.6\,\text{ms}$ for DOE's fixed factorial combinations that include depth 10 models).
   - Single-Objective TPE took **$54.03\,\text{s}$** because it quickly converged toward deeper trees (`max_depth = 8..10`), increasing individual training times ($370.2\,\text{ms}$ per fit).
2. **Surrogate Fitting Overhead:**
   - Fitting the OLS second-order response surface and searching a fine resolution grid ($10,000$ points) took only **$1.48\,\text{s}$** per DOE replicate.
   - Parzen window density estimation in Optuna took **$0.41 - 0.43\,\text{s}$** per 140-trial replicate.
   - Neither method is bottlenecked by surrogate mathematics; both are dominated by XGBoost gradient boosting fitting time ($> 90\%$ of runtime).

---

## 4. Definitional Framework for Efficiency Claims

To prevent ambiguous claims in the manuscript, the study must explicitly distinguish four distinct dimensions of efficiency:

1. **Model-Fit Efficiency:** The number of model evaluations required to reach an acceptable operating point. Under equal 140-fit budgets, neither method has an inherent count advantage.
2. **Unique-Design-Point Efficiency:** DOE evaluates only 25 unique geometric configurations, whereas TPE evaluates 140 unique points. When the search goal is finding multiple distinct non-dominated trade-offs across the global hyperparameter space, TPE is more point-efficient.
3. **Statistical Information per Evaluation:** DOE yields formal ANOVA decomposition, lack-of-fit testing ($F$-tests), main effects, quadratic curvature, and interaction coefficients with known degrees of freedom. TPE provides trial histories and Parzen density ratios, but no formal orthogonal variance decomposition.
4. **Noise-Control Capacity:** DOE’s structured replication (evaluating the same 25 coordinates across 5 nuisance blocks) enables explicit separation of hyperparameter effect from nuisance seed variation. TPE does not repeat configurations, confounding hyperparameter quality with seed noise on any given trial.

**Conclusion for Manuscript:** Rather than claiming "fivefold sample efficiency," the manuscript should accurately frame DOE as a **structured, block-replicated methodology that extracts formal second-order response surfaces and nuisance variance estimates from 25 design configurations under an equal 140-fit budget**.
