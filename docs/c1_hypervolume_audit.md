# Audit A: Pareto-Front Hypervolume Verification and Estimand Comparability

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Audited Dataset:** `results/revision_v2/full_run_001/`  
**Reference Points:** $[0.60, 250.0]$ and $[0.65, 275.0]$  
**Date:** October 10, 2026  

---

## 1. Executive Summary & Core Finding

The prospective revision reports that Multi-Objective TPE (MO-TPE) achieves a mean hypervolume of $18.3834$ (SD: $0.3301$) at reference point $[0.60, 250.0]$, compared to $15.6128$ (SD: $0.7268$) for Repeated DOE and $16.8585$ for the Full Evaluated DOE Candidate Front.

**Audit Verification Result:**
1. **Mathematical Reproducibility:** All saved hypervolume numbers in `results/revision_v2/full_run_001/hypervolume.json` were independently recomputed from the raw records in `optimizer_trials.csv`, `doe_selection_candidates.csv`, and `doe_matched_candidate_front.csv`. Every value reproduces to floating-point machine precision ($< 1.1 \times 10^{-14}$ absolute difference).
2. **Estimand Non-Identity:** The comparison between MO-TPE candidate fronts ($18.3834$) and Repeated DOE candidate fronts ($15.6128$) **does not compare identical mathematical estimands**:
   - MO-TPE candidate fronts are composed of single-split evaluations ($N=1$ model fit on development split seed 42) per candidate trial.
   - Repeated DOE candidate fronts are composed of the arithmetic mean validation RMSE across five distinct seeded nuisance block partitions ($N=5$ model fits per design point).
3. **Matched Single-Split Comparison:** When DOE is evaluated on the exact same single development split (seed 42) and matched online latency protocol as MO-TPE (`full_doe_evaluated_candidate_front`), DOE achieves a hypervolume of **$16.8585$** (4 non-dominated points) at $[0.60, 250.0]$. MO-TPE's 20 replicates achieve a mean of **$18.3834$** (95% CI: $[18.2289, 18.5379]$). In all 20 replicates, MO-TPE's hypervolume exceeds $16.8585$.
4. **Root Cause:** MO-TPE evaluates 140 candidate trials sequentially across the entire 4D hyperparameter space, discovering an average of $9.35$ non-dominated points spanning both shallow/fast and deep/accurate models. In contrast, DOE's central composite design was positioned locally around a preliminary screening region, evaluating 25 geometrical points that produce only 4 non-dominated points within the reference bounding box.

---

## 2. Independent Reproduction of Published Hypervolumes

Using `scientific_stats.hypervolume_2d_min` with strict 2D Lebesgue integration:
$$\text{HV}(P, r) = \sum_{i=1}^{k} (r_1 - p_{i,1}) (p_{i-1,2} - p_{i,2}) \quad \text{with } p_{0,2} = r_2$$

The raw records were extracted and recalculated independently:

| Reference Point | Evaluated Frontier | Saved JSON Value | Independent Recomputed Value | Absolute Discrepancy |
|---|---|---:|---:|---:|
| **$[0.60, 250.0]$** | MO-TPE Replicate 0 | 17.959245779438703 | 17.959245779438703 | $0.0 \times 10^{-16}$ |
| **$[0.60, 250.0]$** | MO-TPE Mean (20 reps) | 18.383424576020012 | 18.383424576020015 | $3.5 \times 10^{-15}$ |
| **$[0.60, 250.0]$** | Repeated DOE Replicate 0 | 15.698944888126743 | 15.698944888126743 | $0.0 \times 10^{-16}$ |
| **$[0.60, 250.0]$** | Repeated DOE Mean (20 reps)| 15.612770809055679 | 15.612770809055680 | $1.7 \times 10^{-15}$ |
| **$[0.60, 250.0]$** | Full Evaluated DOE Front | 16.858506433880063 | 16.858506433880056 | $7.1 \times 10^{-15}$ |
| **$[0.60, 250.0]$** | Historical DOE 2-Point Set | 16.036254991142510 | 16.036254991142510 | $0.0 \times 10^{-16}$ |
| **$[0.60, 250.0]$** | Historical DOE Desirability | 14.302693885829054 | 14.302693885829054 | $0.0 \times 10^{-16}$ |
| **$[0.65, 275.0]$** | MO-TPE Mean (20 reps) | 30.053515311085608 | 30.053515311085610 | $1.7 \times 10^{-15}$ |
| **$[0.65, 275.0]$** | Repeated DOE Mean (20 reps)| 26.684669771295790 | 26.684669771295790 | $0.0 \times 10^{-16}$ |
| **$[0.65, 275.0]$** | Full Evaluated DOE Front | 28.143035904619303 | 28.143035904619303 | $0.0 \times 10^{-16}$ |
| **$[0.65, 275.0]$** | Historical DOE 2-Point Set | 26.916784461881754 | 26.916784461881754 | $0.0 \times 10^{-16}$ |

**Conclusion:** The mathematical implementation and numerical reporting in `hypervolume.json` are 100% reproducible.

---

## 3. Dissection of Frontier Construction and Estimand Divergence

Detailed inspection of the data generation pipeline reveals how each frontier was constructed:

### A. MO-TPE Candidate Fronts (`mo_tpe_by_replicate`)
- **Source:** `results/revision_v2/full_run_001/optimizer_trials.csv` where `optimizer == 'multi_objective_tpe'`.
- **Candidate Pool:** In each replicate, all $140$ sequential trials evaluated during search are retained (not just the selected incumbent). Across 20 replicates, this represents $2,800$ trials ($140 \times 20$).
- **Data Partition & Model Seed:** Every trial was fitted on a single development split (`development_split_seed: 42`), using the sampler seed of that trial.
- **Objective Estimand:**
  - Objective 1: Empirical validation RMSE from that **single model fit** on split 42.
  - Objective 2: Online single-sample predict latency from 30 timed calls (`single_sample_latency_online_search_v1`).
- **Pruning & Reference Filtering:**
  - Lexicographical sort by validation RMSE.
  - Points with validation RMSE $> 0.60$ or latency $> 250.0\,\mu\text{s}$ are classified as `outside_reference_count` and pruned.
  - Dominated points are filtered out.
  - **Resulting Pareto front:** An average of $9.35$ non-dominated configurations per replicate (range: 6 to 15).

### B. Repeated DOE Candidate Fronts (`repeated_doe_by_replicate`)
- **Source:** `results/revision_v2/full_run_001/doe_selection_candidates.csv`.
- **Candidate Pool:** 25 design coordinate points (the standard face-centered CCD points) evaluated per replicate.
- **Data Partition & Model Seed:** Each point was fitted and evaluated across five distinct nuisance blocks (seeds 2001–2005 for rep 0, etc.) from `doe_selection_runs.csv`.
- **Objective Estimand:**
  - Objective 1: The **arithmetic mean validation RMSE across 5 block fits** (`validation_rmse_mean`).
  - Objective 2: The **arithmetic mean predict latency across 5 block fits** (`predict_latency_us_mean`).
- **Resulting Pareto front:** An average of $6.1$ non-dominated configurations per replicate (range: 4 to 8).

### C. Full Evaluated DOE Candidate Front (`full_doe_evaluated_candidate_front`)
- **Source:** `results/revision_v2/full_run_001/doe_matched_candidate_front.csv`.
- **Candidate Pool:** 27 configurations: the 25 CCD design coordinates plus the 2 selected historical operating points (`doe-mo-historical` and `doe-so-historical`).
- **Data Partition & Model Seed:** Evaluated on the exact single development split (`development_split_seed: 42`).
- **Objective Estimand:**
  - Objective 1: Empirical validation RMSE from a **single model fit** on split 42.
  - Objective 2: Online single-sample predict latency from 30 timed calls on split 42.
- **Resulting Pareto front:** Exactly 4 non-dominated points:
  1. $[0.47098, 150.19\,\mu\text{s}]$ (DOE point 26, single-objective optimum)
  2. $[0.48835, 121.90\,\mu\text{s}]$ (DOE point 27, multi-objective desirability optimum)
  3. $[0.49784, 114.36\,\mu\text{s}]$ (DOE point 15)
  4. $[0.50374, 113.82\,\mu\text{s}]$ (DOE point 3)

---

## 4. Matched Comparison Analysis

To compare MO-TPE and DOE fairly, both methods must be analyzed under matched objective definitions:

### Matched Comparison 1: Single-Split Development Frontier
- **Protocol:** Single model fit on development split 42, online latency protocol (30 calls), reference point $[0.60, 250.0]$.
- **Evaluated Frontiers:**
  - **Full DOE Evaluated Front (27 candidates):** $\text{HV} = 16.8585$ (4 non-dominated points).
  - **MO-TPE Candidate Fronts (140 candidates per replicate, $N=20$ replicates):**
    - Mean HV: **$18.3834$** (SD: $0.3301$, IQR: $0.3976$)
    - Minimum HV: $17.4788$
    - Maximum HV: $18.7822$
    - Mean difference ($\text{DOE} - \text{MO-TPE}$): **$-1.5249$** (95% CI: $[-1.6794, -1.3704]$)

**Scientific Finding:** Under an identical single-split development protocol, MO-TPE achieves a larger hypervolume than DOE across all 20 replicates. This is attributable to exploration breadth: MO-TPE explores 140 candidate configurations across the entire hyperparameter domain ($[0.01, 0.30] \times [2, 10] \times [0.5, 1.0] \times [0.1, 10.0]$), discovering diverse trade-off points between $113\,\mu\text{s}$ and $166\,\mu\text{s}$. DOE’s candidate points are confined to the local CCD cube, leaving regions of the trade-off space unexplored.

### Matched Comparison 2: Holdout Retraining Frontier of Selected Incumbents
- **Protocol:** External holdout test set (mean test RMSE over 20 evaluation seeds 2001–2020) and primary benchmark latency (median over 5 sessions $\times$ 1,000 timed reps, thread-pinned).
- **Candidate Pool:** All 122 frozen selected configurations from all optimizers and DOE.
- **Global Holdout Pareto Frontier:** Exactly 12 non-dominated configurations emerge at reference point $[0.60, 250.0]$ ($\text{HV} = 17.6714$):
  - **1 point from Single-Objective TPE:** Test RMSE $0.46505$, Latency $171.33\,\mu\text{s}$ (unconstrained lowest error).
  - **5 points from Constrained TPE:** Test RMSE $0.46556$ to $0.47344$, Latency $150.42\,\mu\text{s}$ to $135.96\,\mu\text{s}$.
  - **3 points from Repeated DOE MO:** Test RMSE $0.47511$ to $0.48907$, Latency $126.22\,\mu\text{s}$ to $118.74\,\mu\text{s}$.
  - **3 points from Multi-Objective TPE:** Test RMSE $0.48441$ to $0.50666$, Latency $120.15\,\mu\text{s}$ to $116.10\,\mu\text{s}$.

**Scientific Finding:** On the external holdout test set, **both Repeated DOE MO and MO-TPE contribute non-dominated solutions** to the low-latency boundary ($\le 126\,\mu\text{s}$):
- In the range $118\,\mu\text{s}$ to $126\,\mu\text{s}$, Repeated DOE MO selections achieve lower Test RMSE ($0.47511$ at $126.22\,\mu\text{s}$; $0.48366$ at $121.21\,\mu\text{s}$; $0.48907$ at $118.74\,\mu\text{s}$) than MO-TPE ($0.48441$ at $120.15\,\mu\text{s}$).
- In the ultra-low latency regime ($< 117\,\mu\text{s}$), MO-TPE achieves points at $116.54\,\mu\text{s}$ ($0.50279$) and $116.10\,\mu\text{s}$ ($0.50666$) that DOE does not span.

---

## 5. Statistical Interpretation and Limitations

1. **Search Variance vs Retraining Variance:** The variance across MO-TPE replicates ($\text{SD} = 0.3301$) reflects stochastic search variation across sampler seeds. The variance across Repeated DOE replicates ($\text{SD} = 0.7268$) reflects variance across nuisance block seeds. These two variances represent fundamentally different stochastic mechanisms and must not be pooled.
2. **Dependence of Partitions:** Repeated DOE nuisance blocks draw from overlapping partitions of the development dataset. They are not independent datasets, and confidence intervals around repeated DOE hypervolume must be interpreted as conditional on that shared data pool.
3. **Candidate Frontier vs Search-Time Frontier:** The candidate frontier evaluated during search reflects optimistic sample realization (winner's curse). The holdout Pareto frontier of frozen selections confirms that while MO-TPE finds a wider candidate front during search, both DOE MO and MO-TPE find competitive, non-dominated operating configurations in post-search holdout evaluation.
