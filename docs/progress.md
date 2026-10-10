# Progress Report: Scientific Revision Pipeline

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Base Commit:** `120fd4da95efa8e5d6e0b6a7e6fd074028d89f06`  
**Historical Baseline Tag:** `v1.0.0` (`cd63135d2b70a0ee324555c11b67a541b32b98a4`)  
**Date:** October 10, 2026  
**Status:** Work Package C complete; full empirical protocol executed, audited, and verified; ready for user review prior to manuscript compilation.

---

## 1. Executive Summary

This revision addresses all substantive reviewer and Codex audit findings for the research pipeline comparing Sequential Response Surface Methodology (RSM/CCD) and Bayesian Optimization (TPE, Constrained TPE, MO-TPE) on XGBoost under stochastic nuisance blocking.

All software defects, data leakage vectors, synthetic coordinate averaging, and latency measurement discrepancies have been repaired. A complete automated test suite (106 tests, 100% green), a GitHub Actions CI workflow, and an atomic per-replicate checkpointing/resume mechanism have been implemented and verified with a real end-to-end smoke reproduction run.

---

## 2. Completed Milestones & Architectural Fixes

### A. Strict Holdout Test-Set Isolation & Cryptographic Gating
- **Problem:** Historical `CaliforniaHousingDataManager.get_split()` returned train, validation, and holdout test sets simultaneously; `pipeline.evaluate_model()` logged `test_rmse` during experimental design exploration.
- **Correction:** 
  - Implemented `CaliforniaHousingDevelopmentDataManager.get_split()` returning strictly train and validation splits (`get_dev_split`). Holdout test data is completely inaccessible during search and DOE selection.
  - Implemented `final_evaluation.FinalTestEvaluator`: external test evaluation is locked behind cryptographic SHA-256 gating that requires frozen configurations in `finalized_selections.json`.
- **Verification:** `tests/test_holdout_isolation.py` and `tests/test_final_evaluation_gate.py`.

### B. Incumbent Selection Integrity & Trial-Level Logging
- **Problem:** Baselines previously constructed synthetic incumbent vectors using coordinate-wise medians (`np.median(..., axis=0)`), which could represent configurations that were never evaluated. Constrained TPE also fell back silently to infeasible configurations when latency constraints were violated.
- **Correction:**
  - Removed coordinate-wise median averaging; implemented `select_median_actual_incumbent()` which selects an actual winning trial from trial logs.
  - All optimizer trials (including infeasible trials and errors) are logged to `optimizer_trials.csv` with parameters, latency, objective, status, and feasibility flags.
  - Constrained TPE strictly marks replicates as infeasible if no trial satisfies $\le 145\,\mu\text{s}$ (no infeasible fallbacks).
  - Integer depth is parameterized as a discrete integer dimension (`trial.suggest_int`).
- **Verification:** `tests/test_benchmark_protocol.py`.

### C. Standardized Latency Profiling & Cross-Platform Thread Pinning
- **Problem:** Confirmation and benchmark latency were measured on disparate scales (142.3 µs vs 119.8 µs) with variable warmups and thread contention.
- **Correction:**
  - Implemented centralized `latency.py` protocol: 50 warmup iterations, 1,000 timed prediction repetitions.
  - Added CPU thread pinning (`os.sched_setaffinity` on Linux, Win32 `kernel32` process affinity on Windows).
  - Explicitly measures and separates `predict` vs `inplace_predict` interface overhead (`latency_interface_overhead.json`).
- **Verification:** `tests/test_latency_protocol.py` and `tests/test_latency_modeling.py`.

### D. Multi-Objective Candidate Fronts & Dual-Reference Hypervolume
- **Problem:** Historical reporting evaluated hypervolume between a 2-point DOE set and a single MO-TPE point.
- **Correction:**
  - Implemented candidate-level Pareto front extraction (`scientific_stats.py`) across the complete search space.
  - Calculated 2D hypervolume using common predeclared reference points ($[0.60, 250]$ and $[0.65, 275]$) with dominance clipping.
  - Preserved historical 2-point reference comparison for provenance disclosure while introducing fair candidate-front comparisons.
- **Verification:** `tests/test_scientific_stats.py` and `tests/test_revision_plots.py`.

### E. Resumable Atomic Checkpointing & CI/CD
- **Problem:** Full protocol requires ~51 minutes of CPU time (17,077 model fits); interrupted executions risked losing progress.
- **Correction:**
  - Implemented per-replicate JSON checkpointing in `scripts/run_benchmarks.py` under `<output-dir>/checkpoints/`.
  - Added `--resume` CLI flag to both `scripts/run_benchmarks.py` and `scripts/reproduce_all.py` to seamlessly continue interrupted runs.
  - Added GitHub Actions workflow `.github/workflows/test.yml` running unit tests across Ubuntu and Windows on pushes and PRs.
- **Verification:** `tests/test_revision_benchmark_integration.py` and `tests/test_revision_entrypoints.py`.

---

## 3. Software Verification & Smoke Test Results

### Unit and Integration Test Suite
```
python -m pytest -q
........................................................................ [ 67%]
..................................                                       [100%]
106 passed, 4 warnings in 3.89s
```
All 106 tests passed without errors.

### Real Smoke Run Execution (`results/revision_v2/smoke_run_001`)
- **Command:** `python scripts/reproduce_all.py --mode smoke --output-dir results/revision_v2/smoke_run_001`
- **Elapsed Time:** 62.0 seconds.
- **Generated Artifacts Verified (19 files):**
  1. `computational_budget.json`
  2. `optimizer_trials.csv`
  3. `optimizer_replicates.csv`
  4. `optimizer_summary.json`
  5. `doe_selection_runs.csv`
  6. `doe_selection_summary.json`
  7. `finalized_selections.json`
  8. `latency_measurement.json`
  9. `latency_interface_overhead.json`
  10. `final_evaluations.csv`
  11. `final_summary.csv`
  12. `paired_comparisons.json`
  13. `hypervolume.json`
  14. `provenance.json`
  15. `run_manifest.json`
  16. `fig_validation_tradeoff.png`
  17. `fig_test_tradeoff.png`
  18. `fig_hypervolume.png`
  19. `fig_latency_distribution.png`

All generated data structures were audited: no NaN or missing values in completed trials, strict holdout separation verified, configuration SHA-256 hashes matched, and smoke test status correctly tagged as `smoke_underpowered`.

---

### Continuous Integration (GitHub Actions) Results
- **Workflow:** `.github/workflows/test.yml` (Matrix: `ubuntu-latest`, `windows-latest` on Python 3.11).
- **Pull Request #2 Verification Run (ID 38005501561):**
  - `ubuntu-latest, Python 3.11`: **PASS** (54s) — 106 passed in 5.3s.
  - `windows-latest, Python 3.11`: **PASS** (2m 0s) — 106 passed in 4.9s.
- **Branch Push Verification Run (ID 38005496379, commit `a336f33`):**
  - `ubuntu-latest, Python 3.11`: **PASS** (38s).
  - `windows-latest, Python 3.11`: **PASS** (2m 4s).
- **CI Defects Resolved:**
  1. `actions/setup-python@v5` cache error: Added tested `requirements.txt` manifest and configured `cache-dependency-path: 'requirements.txt'`.
  2. Git baseline tag resolution error: `actions/checkout@v4` defaulted to shallow `fetch-depth: 1`, omitting the baseline tag `v1.0.0`. Added `fetch-depth: 0` to checkout step, restoring full tag history for cryptographic baseline comparison.

---

## 4. Current Work Package Status

| Package | Title | Status | Notes |
|---|---|---|---|
| **Package A** | Audit and Correctness | **Completed & Verified** | Holdout isolation, incumbent selection integrity, centralized latency. |
| **Package B** | Benchmark and Statistical Redesign | **Completed & Verified** | Candidate Pareto fronts, hypervolume, CI workflow green on Ubuntu & Windows, atomic checkpointing. |
| **Package C** | Full Scientific Experiments | **Completed & Verified** | Executed 17,077 model fits and 1,640,810 timed calls across 122 configurations with 0 errors. Audit script verified all 5 dimensions. |
| **Package C.1** | Scientific Audit & Evidence Verification | **Completed & Verified** | Qualified claim boundaries, audit script `audit_scientific_consistency.py`, 110/110 tests passing. |
| **Package D** | Publication Revision & PDF Build | **Completed & Verified** | LaTeX source revised, 275 macros & tables generated, `report.pdf` compiled, documentation verified. Ready for user review. |

---

## 5. Full Scientific Experiments Execution & Empirical Results (Work Package C)

The full experimental protocol was executed on Windows 10 (AMD64, AMD Ryzen Zen 5, 12 logical cores pinned to Core 0) with command:
```powershell
python scripts/run_benchmarks.py --mode full --output-dir results/revision_v2/full_run_001 --confirm-full-budget
```
- **Execution Summary:** 17,077 genuine XGBoost model fits, 1,640,810 timed latency prediction calls.
- **Duration:** 73.1 minutes wall-clock time.
- **Completed Checkpoints:** All 20 optimizer replicate JSONs, repeated DOE selection checkpoint, 5 primary latency sessions, and final evaluation rows.
- **Integrity Status:** 0 failed fits, 0 hash mismatches, 0 holdout leakage occurrences.

### A. Method-Level Performance Summary ($N=122$ Configurations, $2,440$ Holdout Retrainings)

| Method | Frozen Configs | Val RMSE Mean (Between SD) | Test RMSE Mean (Between SD) | Predict Latency Mean (SD) | Inplace Latency Mean | Feasible ($\le 145\,\mu\text{s}$) | Optimism (Val - Search) |
|---|---:|---|---|---|---|---:|---|
| **Single-Objective TPE** | 20 | 0.46750 (0.00082) | **0.46691 (0.00116)** | 188.93 µs (12.07) | 157.14 µs | 0/20 | +0.00516 |
| **Random Search** | 20 | 0.46869 (0.00143) | 0.46953 (0.00207) | 181.12 µs (19.02) | 149.05 µs | 0/20 | +0.00044 |
| **Repeated DOE Single-Obj** | 20 | 0.46872 (0.00000) | 0.46859 (0.00000) | 150.74 µs (0.72) | 118.78 µs | 0/20 | +0.02367 |
| **Constrained TPE** ($\le 145\,\mu\text{s}$) | 20 | 0.46996 (0.00222) | 0.47036 (0.00324) | 144.27 µs (7.06) | 112.07 µs | 9/20* | -0.00039 |
| **Repeated DOE Multi-Obj** | 20 | 0.48528 (0.00554) | 0.48997 (0.00732) | 122.29 µs (2.70) | 89.73 µs | **20/20** | +0.01430 |
| **Multi-Objective TPE** | 20 | 0.48815 (0.00934) | 0.49255 (0.01080) | 121.46 µs (3.75) | 88.84 µs | **20/20** | -0.00176 |
| **Historical DOE Desirability** | 1 | 0.48207 (—) | 0.48841 (—) | 120.42 µs (—) | 88.04 µs | 1/1 | — |
| **Historical DOE Single-Obj** | 1 | 0.46868 (—) | 0.46903 (—) | 148.62 µs (—) | 117.02 µs | 0/1 | — |

*\*Note on Constrained TPE latency:* During search, 100% of chosen incumbents satisfied the online search latency constraint ($\le 145\,\mu\text{s}$). Under the standardized 5-session primary benchmark protocol, predict latency averaged $144.27\,\mu\text{s}$ (9/20 replicates $\le 145\,\mu\text{s}$ on wrapper `predict`, and 20/20 replicates $\le 145\,\mu\text{s}$ at $112.07\,\mu\text{s}$ on core `inplace_predict`).

### B. Multi-Objective Hypervolume Comparison

Evaluated on candidate Pareto fronts under two pre-declared reference points:

| Evaluated Front | Points on Front | HV at $[0.60, 250.0]$ Mean (SD) | HV at $[0.65, 275.0]$ Mean (SD) |
|---|---:|---|---|
| **MO-TPE Candidate Front (20 reps)** | 7–12 | **18.3834 (0.3301)** | **30.0535 (0.4123)** |
| **Repeated DOE Candidate Front (20 reps)** | 4 | 15.6128 (0.7268) | 26.6847 (0.8786) |
| **Full DOE Evaluated Candidate Front** | 4 | 16.8585 | 28.1430 |
| **Historical DOE 2-Point Set** | 2 | 16.0363 | 26.9168 |
| **Historical DOE 1-Point (Desirability)** | 1 | 14.3027 | 24.7490 |

---

## 6. Scientific Findings & Objective Methodological Trade-offs

1. **Observed Pareto Hypervolume on Evaluated Development Split:**
   - **Candidate Hypervolume:** On the evaluated development split, MO-TPE achieves higher observed candidate-front hypervolume than the discrete DOE lattice (+2.77 units at $[0.60, 250]$) because its 140 sequential evaluations adaptively sample continuous configurations across the 4D domain ($[0.01, 0.30] \times [3, 9] \times [0.5, 1.0] \times [0.1, 10.0]$), discovering diverse non-dominated trade-offs. The face-centered CCD, while spanning the exact same factor domain, evaluates a discrete 25-point geometric lattice that yields 4 non-dominated candidate points on this split. This finding supports higher observed hypervolume on this development split; it does not establish universal algorithmic superiority across unseen data partitions.
   - **Single-Objective Absolute RMSE:** Single-Objective TPE attained the lowest overall Test RMSE ($0.46691 \pm 0.00116$), outperforming DOE Single-Objective ($0.46859 \pm 0.00000$). However, this required $188.93\,\mu\text{s}$ predict latency, which violates the latency constraint.

2. **DOE Methodological Characteristics under Equal 140-Fit Budget:**
   - **Budget Allocation:** Under an equal evaluation budget of 140 model fits per replicate, Repeated DOE allocates fits to 25 unique geometric lattice points with 5-fold replication across nuisance blocks, enabling formal estimation of block effects, lack-of-fit, and second-order response surfaces. TPE allocates all 140 fits to unique configurations for exploratory coverage.
   - **Targeted Desirability Point Selection:** Under strict multi-objective desirability ($\le 145\,\mu\text{s}$), Repeated DOE MO selected operating points with lower descriptive Test RMSE ($0.48997$ vs $0.49255$, $p = 0.38$, not statistically significant) and lower between-search variance ($0.00732$ vs $0.01080$) than MO-TPE at matched latency ($122.29\,\mu\text{s}$ vs $121.46\,\mu\text{s}$).
   - **Selection Stability:** Repeated DOE Single-Objective demonstrated zero between-search variance ($\text{SD} = 0.00000$) due to deterministic boundary selection across all 20 nuisance block sets, while maintaining non-zero retraining uncertainty across holdout evaluation seeds ($\text{SD} = 0.00304$).

3. **Latency Measurement & Interface Profiling:**
   - **Session Stability:** Thread pinning to Core 0 yielded a within-configuration between-session standard deviation of only $1.28\,\mu\text{s}$ across 5 sessions.
   - **Historical Discrepancy Status:** Code inspection of `v1.0.0` confirmed that both historical confirmation and benchmark scripts applied Win32 CPU pinning and evaluated both `predict()` and `inplace_predict()`. The historical $+22.5\,\mu\text{s}$ to $+23.3\,\mu\text{s}$ discrepancy cannot be conclusively attributed to unpinned execution or an identified environmental mechanism; it remains an unexplained historical discrepancy that Revision v2 resolves through standardized multi-session profiling.

---

## 7. Verification & Scientific Integrity Audit

Automated verification scripts (`scripts/audit_full_results.py` and `scripts/audit_scientific_consistency.py`) verified:
- **Holdout Isolation:** All 11,200 optimizer trials in `optimizer_trials.csv` contain zero holdout columns (`test_rmse` absent).
- **Cryptographic Gating:** All 122 frozen configurations in `finalized_selections.json` were hashed prior to holdout evaluation; zero hash mismatches occurred during evaluation.
- **Trial Logging:** All 2,800 constrained TPE trials logged feasibility flags (1,590 feasible, 1,210 infeasible). No infeasible configurations were selected as search winners.
- **Statistical Inference:** Pseudo-replicated t-tests over holdout retrainings are explicitly marked `inference_performed: false` in `paired_comparisons.json` to prevent invalid statistical claims.

---

## 8. Work Package C.1 Final Closeout Summary

Work Package C.1 (Independent Scientific Consistency Audit & Publication Evidence Verification) is officially closed:
1. **Hypervolume Conclusions Qualified:** Observed higher candidate hypervolume for MO-TPE ($18.3834$ vs $16.8585$ Full DOE, $p < 0.0001$) on development split 42 supports higher observed performance on that split, not universal algorithmic superiority.
2. **Factual Descriptions Corrected:** Face-centered CCD spans the complete 4D hypercube $[0.01, 0.30] \times [3, 9] \times [0.5, 1.0] \times [0.1, 10.0]$ across 25 lattice points; all depth ranges correctly noted as $3–9$ discrete integer (no depth 2 or 10).
3. **Constrained TPE Feasibility Qualified:** 9/20 benchmark feasibility verified; causal explanations qualified as hypotheses (sampling variability, protocol differences, selection effects).
4. **Deliverables Completed:** `docs/c1_final_assessment.md`, `scripts/audit_scientific_consistency.py`, and `tests/test_scientific_consistency_audit.py` created and verified by 110/110 passing pytest tests.

---

## 9. Work Package D Final Closeout Summary

Work Package D (Final Scientific Manuscript Revision & Publication Build) is officially complete:
1. **Manuscript Revised (`report.tex`):** Abstract, experimental architecture, response surface limitations, Phase 5 benchmarks, discussion, reproducibility, and conclusion revised to incorporate prospective Revision-v2 evidence ($17,077$ fits, $1,640,810$ predictions, 20 replicates) with qualified scientific claims.
2. **Automated Artifact Pipeline:** `scripts/generate_report_artifacts.py` programmatically populates 275 macros in `results/macros.tex`, Table 5 (`tables/tab_benchmarks.tex`), and Table 6 (`tables/tab_hypervolume_comparison.tex`). Zero hardcoded numbers in `report.tex`, enforced by `tests/test_no_hardcoded_numbers.py`.
3. **PDF Build (`report.pdf`):** Compiled successfully via Tectonic 0.17.0 (14 pages, 4.04 MB, exit code 0).
4. **Validation Ledger:** Detailed documentation produced in `docs/d_final_validation.md`.

**FINAL STATUS:** Work Packages A, B, C, C.1, and D are fully complete and verified. PR #2 remains unmerged on branch `codex/scientific-revision`. All changes are ready for final user review and CI verification.

