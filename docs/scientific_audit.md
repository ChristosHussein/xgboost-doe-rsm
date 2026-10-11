# Scientific audit and revision ledger

## Scope and authority

This document records the controlled revision requested on 2026-10-09. The published historical baseline is the annotated tag `v1.0.0`, resolving to commit `cd63135d2b70a0ee324555c11b67a541b32b98a4`. That revision, rather than file timestamps, is authoritative for the manuscript and results published before this revision.

The baseline is preserved by Git and indexed by `docs/provenance/baseline_v1.0.0.json`. The manifest hashes every tracked file under `results/`, `data/`, `tables/`, and `figures/`, plus the manuscript, configuration, and summary files. Historical files are not to be overwritten by revised experiments. New outputs belong under a separately versioned results directory.

At audit start the working tree already contained modified benchmark outputs and source changes. Those files are preserved, but they are **unverified working outputs** and are not authoritative for either the historical paper or the revised study until reproduced under the new protocol and captured in a new manifest.

## Baseline verification

- Baseline commit: `cd63135d2b70a0ee324555c11b67a541b32b98a4`
- Historical snapshot: annotated tag `v1.0.0`
- Revision branch: `codex/scientific-revision`
- Baseline test command: `python -m pytest`
- Baseline result: **17 passed in 9.05 seconds** on 2026-10-09 before revision code was edited
- Python: 3.11.9 on Windows 10.0.26300, AMD64
- Key libraries: NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, statsmodels 0.15.0, XGBoost 3.2.0, Optuna 5.0.0, pytest 9.1.1

## Result-version findings

### Published manuscript baseline

The committed `report.tex`, `report.pdf`, `results/macros.tex`, `tables/tab_benchmarks.tex`, `results/benchmark.csv`, and `results/benchmark_summary.json` at `v1.0.0` form the historical publication snapshot. For example, the committed benchmark reports DOE-MO latency `119.8365` microseconds and DOE-SO latency `147.8415` microseconds, matching the rounded manuscript table values `119.8` and `147.8`.

### Legacy workflow

Top-level `results_summary.json` and files under `data/` were produced by the older `run_pipeline.py` workflow. Their numerical definitions differ materially from the current manuscript workflow: examples include curvature `F = 57,468.86`, six lack-of-fit degrees of freedom, and five confirmation trials. They are historical artifacts and are not authoritative inputs to `report.tex`.

### Dirty working benchmark outputs

At audit start, `results/benchmark.csv`, `results/benchmark_evals_trajectories.csv`, `results/benchmark_optimizer_incumbents.csv`, and `results/benchmark_summary.json` differed from `v1.0.0`. The dirty benchmark CSV, for example, reports a different selected MO-TPE configuration and test RMSE than the published snapshot. These changes were not accompanied by a clean provenance manifest or a regenerated manuscript, so they remain unverified and must not be substituted into the paper.

## Confirmed discrepancies and resolution status

| ID | Severity | Evidence | Required disposition | Resolution & Verification Status |
|---|---|---|---|---|
| SA-001 | Critical | `CaliforniaHousingDataManager.get_split()` returns train, validation, and external holdout arrays to every caller. | Introduce a development-only split interface that cannot expose holdout labels. | **Resolved & Verified.** `CaliforniaHousingDevelopmentDataManager.get_split()` strictly returns train and validation arrays. Verified by `tests/test_holdout_isolation.py`. |
| SA-002 | Critical | `pipeline.evaluate_model()` computes `test_rmse` unconditionally, and the 140-run DOE path logs it to `results/runs.csv`. | Remove holdout evaluation from all development runs. Preserve and disclose the historical exposure. | **Resolved & Verified.** Holdout evaluation stripped from search; development code records validation RMSE only; `optimizer_trials.csv` contains no holdout labels. Verified by `tests/test_holdout_isolation.py`. |
| SA-003 | Critical | Final benchmark and confirmation evaluation accepts in-memory coordinates without a persisted, validated selection freeze. | Require a finalized-selection manifest with canonical configuration hashes before final evaluation. | **Resolved & Verified.** `final_evaluation.py` enforces cryptographic SHA-256 verification against `finalized_selections.json` prior to evaluation. Verified by `tests/test_final_evaluation_gate.py`. |
| SA-004 | High | Published confirmation latency is 142.33 microseconds for DOE-MO and 171.12 for DOE-SO, while the baseline benchmark session reports 119.84 and 147.84. | Centralize the timing protocol and identify each historical session rather than combining its values. | **Resolved & Verified.** Implemented centralized `latency.py` protocol with thread pinning, warmup iterations, and interface overhead tracking. Verified by `tests/test_latency_protocol.py`. |
| SA-005 | High | Optimizer trial-level records are not retained; only trajectories and selected incumbents are stored. | Persist every trial, including status, timing, objective, feasibility, errors, and selection state. | **Resolved & Verified.** Complete optimizer trial logs persisted to `optimizer_trials.csv`. Infeasible trials and failures explicitly recorded. Verified by `tests/test_benchmark_protocol.py`. |
| SA-006 | High | Historical confidence intervals condition on one representative configuration and do not measure between-search variability. | Add repeated end-to-end evaluation and keep search and retraining variation separate. | **Resolved & Verified.** Repeated search replication protocol implemented in `scripts/run_benchmarks.py` and `scientific_stats.py`. Verified by `tests/test_scientific_stats.py`. |
| SA-007 | High | The published Pareto plot uses individual DOE run realizations, allowing favorable seed noise to define the frontier. | Use configuration-level estimates under matched timing and validation protocols. | **Resolved & Verified.** Frontier extraction uses configuration-level means and multi-seed evaluations. Verified by `tests/test_scientific_stats.py` and `tests/test_revision_plots.py`. |
| SA-008 | High | The historical two-point DOE hypervolume is compared with one selected MO-TPE point. | Preserve it as historical context; use complete candidate fronts for optimizer-quality claims. | **Resolved & Verified.** Candidate-level Pareto front extraction and dual reference point ($[0.60, 250]$ and $[0.65, 275]$) hypervolume computation implemented. Verified by `tests/test_scientific_stats.py`. |
| SA-009 | Medium | `results_summary.json` belongs to a legacy workflow but is not labelled as such in its filename or schema. | Keep it immutable and classify it explicitly in provenance and documentation. | **Resolved & Verified.** Baseline manifest `docs/provenance/baseline_v1.0.0.json` explicitly classifies legacy artifacts. |
| SA-010 | Medium | The pre-planned 140-run FCCD is analyzed in stages but was not operationally adaptive. | Use “pre-planned blocked factorial-plus-CCD” terminology unless a new adaptive experiment is executed. | **Resolved & Verified.** Terminology updated in documentation and reporting scripts. |

## Historical test-set caveat

The external holdout was evaluated and recorded during the original DOE executions. Removing the column now would not make the historical study untouched. The revision retains the raw historical record, documents the exposure, prevents future development-stage access, and uses a frozen-selection gate for any new final evaluation. Claims that depend on an untouched test set are limited accordingly.

## Work-package status

- **Package A — Audit and correctness:** **Completed and verified.** Test set isolation, holdout gating, incumbent selection integrity, discrete depth parameterization, and centralized latency timing protocol are implemented and verified by 106 automated tests.
- **Package B — Benchmark and statistical redesign:** **Completed and verified.** Multi-objective candidate front extraction, dual-reference hypervolume, block-aware variance decomposition, CI workflow (`.github/workflows/test.yml`), and atomic per-replicate checkpoint/resume mechanisms are fully implemented and verified via unit tests and smoke execution (`results/revision_v2/smoke_run_001`).
- **Package C — New scientific experiments:** **Completed and verified.** Executed full benchmark protocol (`results/revision_v2/full_run_001/`) with 17,077 model fits and 1,640,810 timed prediction calls across 122 frozen configurations. Completed all 20 replicates for each optimizer and repeated DOE. All 5 audit dimensions verified by `scripts/audit_full_results.py` (0 test leaks, 0 hash mismatches, 0 failed retrainings).
- **Package D — Publication revision:** **Awaiting user review of full results.** PR #2 remains unmerged and `report.tex`/`report.pdf` remain untouched until experimental results are reviewed and authorized.

Software tests alone establish implementation behavior, not statistical validity or generalization. Those evidence levels remain separate throughout this ledger.

