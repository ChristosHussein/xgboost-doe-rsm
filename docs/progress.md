# Progress Report: Scientific Revision Pipeline

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Base Commit:** `120fd4da95efa8e5d6e0b6a7e6fd074028d89f06`  
**Historical Baseline Tag:** `v1.0.0` (`cd63135d2b70a0ee324555c11b67a541b32b98a4`)  
**Date:** October 10, 2026  
**Status:** Verification and smoke testing complete; ready for full experimental protocol.

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
| **Package C** | Full Scientific Experiments | **Ready for Execution** | Pre-computed budget (17,077 fits, ~51 min), guarded by `--confirm-full-budget`. |
| **Package D** | Publication Revision | **Pending Package C** | Blocked on completion of Package C; PDF/macros will be updated from validated full results. |

---

## 5. Scientific Limitations & Experimental Caveats

Prior to executing the full experimental protocol, the following known methodological boundaries are documented:

1. **Historical Holdout Exposure (SA-001, SA-002):**
   - The historical v1.0.0 experimental run recorded test set RMSE during exploration. In revision v2, all development and optimizer search paths are strictly isolated via `CaliforniaHousingDevelopmentDataManager` and holdout evaluation is cryptographically gated. Historical results are preserved with this explicit disclosure; new comparative claims rely exclusively on gated development evaluations.
2. **Pre-Planned vs Adaptive Design (SA-010):**
   - The 140-run FCCD was pre-planned in fixed nuisance blocks rather than dynamically or adaptively navigated in real time. The revision clarifies this terminology throughout report and documentation.
3. **Latency Platform-Dependence:**
   - While thread pinning and interface overhead isolation eliminate within-session drift and process migration, physical microseconds vary across processor architectures and instruction sets (AMD Zen 5 desktop vs Intel Xeon cloud runner).
4. **Search vs Retraining Variance Separation:**
   - To avoid conflating retraining noise with search robustness, the pipeline explicitly decomposes variation into between-search replicate variance ($N=20$) and holdout retraining variance ($N=20$).

---

## 6. Revised Runtime Estimate & Readiness Statement

- **Estimated Total Fits:** 17,077 model fits.
- **Estimated Timed Latency Calls:** 1,640,810 calls.
- **Empirical Single-Core Timing:** ~134.4 ms/fit, ~127.9 µs/call.
- **Wall-Clock Runtime Estimate:** **~51 minutes (0.85 – 1.2 hours)**.
- **Memory & Storage:** $\ge 4\,\text{GB}$ RAM, $\approx 150\,\text{MB}$ disk storage.
- **Checkpoint Resilience:** Atomic checkpoints saved per-replicate in `<output-dir>/checkpoints/`; `--resume` supported.

**READINESS STATUS:** Both CI runners (Ubuntu and Windows) are 100% green. The pipeline is fully verified and ready for full experimental execution upon user authorization:
```powershell
python scripts/run_benchmarks.py --mode full --output-dir results/revision_v2/full_run_001 --confirm-full-budget
```

