# Execution Plan: Full Scientific Experiments (Package C)

**Document:** `docs/full_experiment_plan.md`  
**Revision:** v2 Protocol  
**Repository Branch:** `codex/scientific-revision`  
**Target Output Directory:** `results/revision_v2/full_run_001`  
**Status:** Software validated; awaiting authorization to execute.

---

## 1. Computational Budget Breakdown

The full scientific experiment protocol executes a complete, statistically powered evaluation with 20 search replicates per optimizer, 20 repeated DOE seed sets, 5 primary latency sessions, and 20 evaluation seeds for final test retraining.

| Component | Specification | Model Fits | Timed Latency Calls |
|---|---|---:|---:|
| **Optimizer Searches** | 4 methods $\times$ 20 replicates $\times$ 140 trials | 11,200 | 420,810 |
| **Repeated DOE Selection** | 20 block seed sets $\times$ 5 seeds $\times$ 28 design runs | 2,800 | 0 |
| **Historical DOE Candidates** | Matched candidate evaluation set | 27 | 0 |
| **Primary Latency Profiling** | 122 frozen configs $\times$ 5 sessions $\times$ 2 interfaces | 610 | 1,220,000 |
| **Final Gated Retraining** | 122 frozen configs $\times$ 20 holdout evaluation seeds | 2,440 | 0 |
| **Total Computational Workload** | | **17,077 fits** | **1,640,810 calls** |

### Empirical Timing & Wall-Clock Estimates
Empirical microbenchmarking on the test machine (AMD Ryzen 5 9600X 6-Core Processor @ 3.90 GHz):
- **XGBoost Fit Time:** ~134.4 ms per fit $\rightarrow 17,077 \times 0.1344\,\text{s} \approx 2,295\,\text{s}$ (~38.3 min).
- **Timed Latency Measurement:** ~127.9 µs per predict call, ~95.7 µs per inplace call $\rightarrow \approx 750\,\text{s}$ (~12.5 min).
- **Total Wall-Clock Estimate:** **~51 minutes (0.85 – 1.2 hours)**.

### Resource Requirements
- **CPU:** 1 dedicated CPU core (thread pinning applied via Win32 `kernel32` process affinity).
- **RAM:** Minimum 4 GB available system memory.
- **Disk Storage:** ~150 MB for trial logs, JSON checkpoints, evaluation tables, and figures.

---

## 2. Checkpoint & Resumption Architecture

To protect against interruptions (power loss, system updates, timeouts) during the 50+ minute execution, the pipeline implements an atomic per-replicate checkpointing architecture in `<output-dir>/checkpoints/`:

1. `optimizer_replicate_{0..19}.json`: Checkpoints each completed optimizer replicate (all 140 trials, incumbents, and Pareto fronts).
2. `repeated_doe_selection.json`: Checkpoints the 20 repeated DOE block-seed selections.
3. `primary_latency_sessions.json`: Checkpoints the 5 primary latency measurement sessions across all frozen configurations.
4. `final_evaluations_rows.json`: Checkpoints final holdout test evaluation rows.

If an execution is interrupted at any point, passing `--resume` immediately loads all existing checkpoints and resumes from the exact replicate or stage where execution stopped.

---

## 3. Pre-Flight Safeguards & Isolation Gates

Before launching the full budget, the following invariants are enforced:
1. **Holdout Isolation Gate:** `CaliforniaHousingDevelopmentDataManager` provides train/validation splits only. External holdout labels are inaccessible during optimization.
2. **Cryptographic Selection Gate:** Selections are exported to `finalized_selections.json` and hashed with SHA-256. `FinalTestEvaluator` rejects any evaluation attempt if hashes do not match.
3. **No Synthetic Medians:** Every selected configuration is guaranteed to exist as an actual trial in `optimizer_trials.csv`.
4. **Feasibility Integrity:** Constrained TPE rejects infeasible trials ($\le 145\,\mu\text{s}$ threshold); no fallback to infeasible configurations.
5. **Historical Immutability:** Historical baseline `v1.0.0` files remain strictly untouched; all new data is written to a clean directory under `results/revision_v2/`.

---

## 4. Execution Commands

### Recommended Launch Command (Direct Benchmark Runner)
```powershell
python scripts/run_benchmarks.py --mode full --output-dir results/revision_v2/full_run_001 --confirm-full-budget
```

### Full Reproduction Pipeline (Includes Pre-Test Verification)
```powershell
python scripts/reproduce_all.py --mode full --output-dir results/revision_v2/full_run_001 --confirm-full-budget
```

### Resuming an Interrupted Execution
```powershell
python scripts/run_benchmarks.py --mode full --output-dir results/revision_v2/full_run_001 --confirm-full-budget --resume
```

---

## 5. Post-Execution Workflow (Package D)

Upon successful completion of the full experiment run:
1. Review `results/revision_v2/full_run_001/final_summary.csv` and `paired_comparisons.json`.
2. Verify numerical integrity against historical and revision benchmarks.
3. Update manuscript macros in `results/macros.tex` dynamically from the full run artifacts.
4. Regenerate publication tables (`tables/tab_benchmarks.tex`) and figures.
5. Recompile `report.pdf` with LaTeX and update `REPORT.md`.
