# Work Package C.1 Audit Methodology: Independent Scientific Consistency Audit and Publication Evidence Verification

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Audited Commit:** `842b531611ded9ff8495b88bf0c12790efe56b69`  
**Audited Dataset:** `results/revision_v2/full_run_001/`  
**Historical Baseline Tag:** `v1.0.0` (`cd63135d2b70a0ee324555c11b67a541b32b98a4`)  
**Audit Output Directory:** `results/revision_v2/scientific_audit_c1/`  
**Date:** October 10, 2026  

---

## 1. Executive Purpose and Governance Principles

The goal of Work Package C.1 is to conduct an independent, evidence-driven scientific audit of the completed full-scale experiments before regenerating any publication artifacts.

The completed experiment encompasses 17,077 genuine model fits, 1,640,810 timed latency predictions, and 20 independent search replicates per optimizer. The audit objective is not to favor or defend either Design of Experiments (DOE) or Tree-structured Parzen Estimator (TPE), but to ensure that every numerical claim, comparative table, figure, and inferential statement in the prospective manuscript is methodologically comparable, reproducible, statistically sound, and strictly bounded by empirical evidence.

### Non-Negotiable Operational Constraints
1. **Raw Result Immutability:** `results/revision_v2/full_run_001/`, historical baseline artifacts (`v1.0.0`), checkpoints, manifests, and logs remain strictly read-only and immutable. All audit artifacts, recomputed data structures, and diagnostic scripts are isolated in `results/revision_v2/scientific_audit_c1/` and `docs/`.
2. **Data-First Forensics:** All investigations rely on the persisted trial-level records (`optimizer_trials.csv`, 11,200 rows), design runs (`doe_selection_runs.csv`, 2,800 rows), timing sessions (`latency_measurement.json`), and evaluation tables (`final_evaluations.csv`, 2,440 rows).
3. **No Retrospective Optimization:** No seeds, reference points, desirability parameters, or evaluation procedures may be tuned or adjusted post-hoc to generate favorable conclusions.
4. **Holdout Test Set Isolation:** Holdout test data must never be consulted to select hyperparameters, adjust objective formulations, or choose winning methods.
5. **Separation of Software Greenness and Statistical Invariance:** Passing 106 automated tests verifies software execution and protocol integrity; it does not constitute statistical proof of algorithm superiority or generalization validity.
6. **Manuscript Freeze:** Neither `report.tex` nor `report.pdf` will be modified or regenerated during Work Package C.1. Pull Request #2 remains unmerged on GitHub.

---

## 2. Audit Scope and Module Breakdown

The audit is structured into five core investigations:

### Audit A: Comparable Pareto-Front Hypervolumes & Estimand Alignment
- **Problem:** `hypervolume.json` compares MO-TPE candidate fronts (evaluated on a single development split per trial) with repeated DOE candidate fronts (evaluated across an average of 5 seeded development partitions).
- **Plan:**
  1. Independently recompute all hypervolume metrics from raw candidate records to confirm exact mathematical reproducibility.
  2. Analyze the mathematical definition and candidate-set selection of both frontiers.
  3. Formulate a matched common evaluation protocol (identical dataset, split, seeds, timing API, and reference points).
  4. Differentiate between search-time candidate non-dominated fronts and independently evaluated non-dominated frontiers.
  5. Evaluate whether additional model fits are required, calculate their computational budget, and report revised comparisons or declare unresolved estimand boundaries.

### Audit B: Computational Efficiency & Budget Accounting Reconciliation
- **Problem:** Previous claims asserted a $5\times$ sample efficiency advantage for DOE based on 28 fits per block versus 140 TPE trials. However, a full repeated DOE experiment comprises 5 blocks $\times$ 28 fits = 140 fits, equal to TPE's 140 trials. Furthermore, the report mentioned 54 historical matched fits whereas the budget ledger logged 27.
- **Plan:**
  1. Reconcile the exact 17,077 model fits and 1,640,810 timed latency predictions across all files and ledgers.
  2. Disentangle model-fit efficiency, unique design configuration efficiency, optimization wall-clock time, and noise-reduction replication capacity.
  3. Output a formal reconciliation artifact: `results/revision_v2/scientific_audit_c1/budget_reconciliation.json`.

### Audit C: Historical Latency Discrepancy Forensics
- **Problem:** The v1.0.0 paper reported confirmation latency of $142.3\,\mu\text{s}$ (DOE-MO) and $171.1\,\mu\text{s}$ (DOE-SO) vs benchmark latency of $119.8\,\mu\text{s}$ (DOE-MO) and $147.8\,\mu\text{s}$ (DOE-SO). While the revision discovered a $32.19\,\mu\text{s}$ wrapper overhead (`predict` vs `inplace_predict`), this does not automatically prove what occurred in the historical code.
- **Plan:**
  1. Inspect the historical code directly at git tag `v1.0.0` (`pipeline.py`, `scripts/run_confirmation.py`, `scripts/run_benchmarks.py`, `scripts/measure_latency.py`).
  2. Reconstruct the exact historical prediction paths, data types (ndarray vs DataFrame vs DMatrix), threading configs, warmup schedules, and timing sessions.
  3. Explicitly report what is proven by code evidence and what remains an unprovable historical artifact.

### Audit D: DOE Selection Stability vs Predictive Uncertainty
- **Problem:** Repeated DOE single-objective reported between-search standard deviation of $0.00000$ because all 20 repeated searches selected the exact same hyperparameter configuration.
- **Plan:**
  1. Disentangle configuration-selection stability (which was perfectly deterministic across the 20 block repetitions) from prediction/retraining uncertainty across random seeds.
  2. Compute and document configuration-selection variability, between-search variability, and conditional retraining variability as three distinct quantities.

### Audit E: Method-Level Statistical Claims & Feasibility Verification
- **Problem:** Summary tables report small numerical differences between methods, and Constrained TPE achieved 20/20 feasibility during search but 9/20 under primary `predict` benchmark timing.
- **Plan:**
  1. Reconstruct all performance summaries and statistical distributions.
  2. Audit paired comparison methodology, experimental units, and equivalence assertions.
  3. Verify Constrained TPE online vs benchmark latency measurements.
  4. Classify every candidate publication claim into: *Directly verified*, *Verified with limitations*, *Descriptive only*, *Unsupported*, or *Requires additional evidence*.

---

## 3. Audit Deliverables Matrix

| Module | Core Deliverable File | Key Artifacts |
|---|---|---|
| **Audit A** | `docs/c1_hypervolume_audit.md` | Reproducibility scripts, estimand analysis, matched comparisons |
| **Audit B** | `docs/c1_computational_efficiency.md` | `results/revision_v2/scientific_audit_c1/budget_reconciliation.json` |
| **Audit C** | `docs/c1_latency_forensics.md` | `v1.0.0` code analysis, interface comparison table |
| **Audit D** | `docs/c1_selection_stability.md` | Selection distributions, variance decomposition |
| **Audit E** | `docs/c1_statistical_inference.md` | Claim audit ledger, effect size analysis, Constrained TPE audit |
| **Synthesis** | Final Summary Report | Scientific contribution and claim boundaries |
