# Audit E: Method-Level Statistical Inference and Publication Claim Verification

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Audited Dataset:** `results/revision_v2/full_run_001/`  
**Date:** October 10, 2026  

---

## 1. Executive Summary & Inferential Boundaries

This audit rigorously verifies every statistical test, effect size calculation, constraint feasibility proportion, and candidate publication claim arising from the full scientific experiment.

### Key Inferential Conclusions:
1. **No Statistical Superiority for DOE MO over MO-TPE:** The observed Test RMSE difference between Repeated DOE MO ($0.48997$) and MO-TPE ($0.49255$) is only **$0.00258$** ($0.52\%$). Under a defensible unpaired Welch two-sample $t$-test across the independent search replicates, **$t = -0.885, p = 0.3826$** ($95\%$ CI: $[-0.00851, +0.00335]$); under a paired-by-index test, **$t = -0.966, p = 0.3463$**. This difference is **not statistically significant**. Any claim that DOE MO is statistically superior to MO-TPE on accuracy is **unsupported** and must be classified as purely descriptive.
2. **Constrained TPE Feasibility Discrepancy Explained:** Constrained TPE achieved **$20/20$ ($100\%$) feasibility during online search**, but only **$9/20$ ($45\%$) feasibility under the independent primary benchmark protocol**:
   - Every selection with $\text{max\_depth} = 6$ ($9/9$) remained fully feasible in benchmark timing (mean latency: $136.62\,\mu\text{s}$).
   - Every selection with $\text{max\_depth} = 7$ ($11/11$) slipped under the $145\,\mu\text{s}$ constraint during noisy 30-call search measurements (mean search latency: $140.94\,\mu\text{s}$), but genuinely averaged **$150.54\,\mu\text{s}$** under the 5-session 1,000-call benchmark protocol.
   - The true benchmark feasibility proportion is **$45.0\%$ ($95\%$ Wilson score interval: $[25.8\%, 65.8\%]$)**.
3. **No Retraining Seed Pseudo-Replication:** Retraining seeds ($2001$–$2020$) represent conditional fits of frozen configurations on a shared development partition and holdout test set. They do not constitute independent method-level search replicates. All inferential comparisons must use the independent search replicates ($N=20$) as the experimental unit.

---

## 2. Reconstructed Performance Summaries & Effect Sizes

Evaluating the $N=20$ independent search replicates for each optimization strategy on the external holdout test set:

| Comparison | Anchor Method | Comparator Method | Anchor Mean (SD) | Comparator Mean (SD) | Mean Diff (Anchor - Comp) | 95% CI of Difference | Welch $t$-stat | Welch $p$-value | Practical Effect Size |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| **DOE MO vs MO-TPE** | Repeated DOE MO | MO-TPE | 0.48997 (0.00732) | 0.49255 (0.01080) | **-0.00258** | [-0.00851, +0.00335] | -0.885 | 0.3826 | Negligible ($0.52\%$) |
| **DOE SO vs SO-TPE** | Repeated DOE SO | SO-TPE | 0.46859 (0.00000) | 0.46691 (0.00116) | **+0.00168** | [+0.00113, +0.00222] | +6.465 | $< 0.0001$ | Small ($0.36\%$, TPE lower) |
| **DOE SO vs Random** | Repeated DOE SO | Random Search | 0.46859 (0.00000) | 0.46953 (0.00207) | **-0.00094** | [-0.00191, +0.00003] | -2.033 | 0.0563 | Negligible ($0.20\%$) |
| **DOE MO vs Constrained**| Repeated DOE MO | Constrained TPE | 0.48997 (0.00732) | 0.47036 (0.00324) | **+0.01961** | [+0.01594, +0.02328] | +10.871 | $< 0.0001$ | Substantial ($4.17\%$, CTPE lower) |

### Contextual Interpretation:
- **Accuracy vs Latency Trade-Off:** While Constrained TPE achieved lower Test RMSE than Repeated DOE MO ($-0.01961$), it pushed right to the boundary of latency ($144.27\,\mu\text{s}$ mean, with $55\%$ violating the constraint). In contrast, Repeated DOE MO selected operating points deeper in the feasible region ($122.29\,\mu\text{s}$ mean latency, $100\%$ feasible).
- **Single-Objective Optimization:** SO-TPE reached the lowest absolute error in the study ($0.46691$), but required $188.93\,\mu\text{s}$ latency ($+43.93\,\mu\text{s}$ beyond the $145\,\mu\text{s}$ constraint).

---

## 3. Forensic Investigation of Constrained TPE Feasibility

| Selection ID | Max Depth | Online Search Latency | Benchmark Predict Latency | Latency Shift | Benchmark Inplace Latency | Benchmark Feasible ($\le 145\,\mu\text{s}$)? |
|---|---:|---:|---:|---:|---:|---|
| `constrained_tpe-rep-0` | 7 | 142.54 µs | 150.54 µs | +8.00 µs | 118.58 µs | **Violation** |
| `constrained_tpe-rep-1` | 6 | 125.77 µs | 135.34 µs | +9.57 µs | 103.40 µs | Feasible |
| `constrained_tpe-rep-2` | 6 | 138.41 µs | 136.76 µs | -1.65 µs | 104.77 µs | Feasible |
| `constrained_tpe-rep-3` | 7 | 142.83 µs | 151.49 µs | +8.66 µs | 119.55 µs | **Violation** |
| `constrained_tpe-rep-4` | 7 | 141.91 µs | 149.80 µs | +7.89 µs | 119.27 µs | **Violation** |
| `constrained_tpe-rep-5` | 6 | 143.09 µs | 137.53 µs | -5.56 µs | 105.22 µs | Feasible |
| `constrained_tpe-rep-6` | 6 | 130.14 µs | 138.00 µs | +7.86 µs | 104.62 µs | Feasible |
| `constrained_tpe-rep-7` | 7 | 138.07 µs | 151.73 µs | +13.66 µs | 118.86 µs | **Violation** |
| `constrained_tpe-rep-8` | 6 | 128.86 µs | 136.63 µs | +7.77 µs | 104.62 µs | Feasible |
| `constrained_tpe-rep-9` | 6 | 128.25 µs | 135.96 µs | +7.71 µs | 104.63 µs | Feasible |
| `constrained_tpe-rep-10` | 7 | 138.08 µs | 150.23 µs | +12.15 µs | 118.06 µs | **Violation** |
| `constrained_tpe-rep-11` | 6 | 143.33 µs | 137.33 µs | -6.00 µs | 104.08 µs | Feasible |
| `constrained_tpe-rep-12` | 7 | 144.49 µs | 149.42 µs | +4.93 µs | 116.51 µs | **Violation** |
| `constrained_tpe-rep-13` | 7 | 140.22 µs | 150.12 µs | +9.90 µs | 117.75 µs | **Violation** |
| `constrained_tpe-rep-14` | 6 | 134.60 µs | 136.42 µs | +1.82 µs | 103.43 µs | Feasible |
| `constrained_tpe-rep-15` | 7 | 140.23 µs | 150.16 µs | +9.93 µs | 118.84 µs | **Violation** |
| `constrained_tpe-rep-16` | 7 | 138.65 µs | 150.69 µs | +12.04 µs | 119.29 µs | **Violation** |
| `constrained_tpe-rep-17` | 6 | 137.14 µs | 136.37 µs | -0.77 µs | 104.00 µs | Feasible |
| `constrained_tpe-rep-18` | 7 | 139.32 µs | 150.42 µs | +11.10 µs | 117.92 µs | **Violation** |
| `constrained_tpe-rep-19` | 7 | 144.04 µs | 150.42 µs | +6.38 µs | 117.97 µs | **Violation** |

### Mechanism of the Discrepancy:
- **Optimization Boundary Exploitation:** In constrained Bayesian optimization, the surrogate actively favors points near the feasibility boundary to maximize the primary objective (lower RMSE).
- **Sampling Noise Effect:** Because online search evaluated latency with only 30 calls, depth 7 models with true latency $\approx 150.5\,\mu\text{s}$ occasionally registered below $145\,\mu\text{s}$ ($138 - 144\,\mu\text{s}$) due to normal measurement variance. The optimizer capitalized on these downward fluctuations to select depth 7 models.
- **Benchmark Exposure:** When subjected to the multi-session 1,000-call benchmark protocol, the true latency of these depth 7 models was revealed, resulting in a $55\%$ violation rate.
- **Scientific Implication:** Constrained optimization with noisy constraints requires either conservative safety margins (e.g., constraining at $140\,\mu\text{s}$ instead of $145\,\mu\text{s}$) or larger online sampling sizes to prevent constraint-boundary boundary violations.

---

## 4. Master Publication Claim Ledger

Every candidate claim considered for the revised manuscript is classified according to empirical evidence:

| # | Candidate Manuscript Claim | Audit Classification | Evidence Base & Justification |
|---|---|---|---|
| **1** | *"DOE achieves a fivefold computational efficiency advantage over TPE."* | **Unsupported / Factually False** | Both methods consume 140 model fits per replicate in the primary benchmark protocol. (See Audit B). |
| **2** | *"Repeated DOE MO achieves lower Test RMSE than MO-TPE."* | **Descriptive Only** | Observed difference is only $0.00258$ ($p = 0.3826$ Welch, $p = 0.3463$ paired). Not statistically significant. |
| **3** | *"Repeated DOE Single-Objective has zero predictive uncertainty."* | **Unsupported / Factually False** | Selection stability was deterministic across the 20 block sets, but conditional retraining standard deviation is $0.00304$. (See Audit D). |
| **4** | *"Repeated DOE Single-Objective selection is perfectly stable across nuisance block sets."* | **Directly Verified** | All 20 independent searches selected the exact same hyperparameter configuration ($20/20$). |
| **5** | *"MO-TPE achieves significantly higher hypervolume on development candidate fronts than DOE."* | **Directly Verified** | MO-TPE achieved mean HV $18.3834$ vs Full DOE $16.8585$ under matched single-split protocol ($p < 0.0001$). |
| **6** | *"Candidate-level Pareto hypervolume comparisons between Repeated DOE and MO-TPE compare different estimands."* | **Directly Verified** | MO-TPE uses a single split per trial ($N=1$); Repeated DOE averages across 5 nuisance blocks ($N=5$). |
| **7** | *"Single-Objective TPE discovers the lowest unconstrained Test RMSE in the study."* | **Directly Verified** | SO-TPE achieved Test RMSE $0.46691 \pm 0.00116$, but violates the latency constraint ($188.93\,\mu\text{s}$). |
| **8** | *"Constrained TPE selections achieved 100% search-time feasibility but only 45% benchmark feasibility."* | **Directly Verified** | Depth 7 models exploited 30-call search noise ($140.9\,\mu\text{s}$ search mean), but exceed $145\,\mu\text{s}$ in 1,000-call benchmark ($150.5\,\mu\text{s}$). |
| **9** | *"The historical latency discrepancy between confirmation and benchmark was caused by predict vs inplace_predict confusion."* | **Unsupported** | Both historical scripts measured both APIs and both published `predict()`. Discrepancy was caused by session and environment drift. (See Audit C). |
| **10**| *"Both Repeated DOE MO and MO-TPE contribute non-dominated configurations to the global holdout Pareto front."* | **Directly Verified** | 12 configurations form the global front: 3 from DOE MO ($118.7 - 126.2\,\mu\text{s}$), 3 from MO-TPE ($116.1 - 120.2\,\mu\text{s}$), 5 from Constrained TPE ($136.0 - 150.4\,\mu\text{s}$), and 1 from SO-TPE ($171.3\,\mu\text{s}$). |
