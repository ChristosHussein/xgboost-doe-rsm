# Work Package D.2 — Single Source of Truth, Automated Scientific Reporting, and Numerical Integrity Validation

## 1. Executive Summary

Work Package D.2 eliminates the possibility of inconsistent scientific numbers appearing across `REPORT.md`, `report.tex`, `results/macros.tex`, `tables/*.tex`, and `report.pdf` by replacing manual numerical transcription and string-presence tests with a **Single Source of Truth (SSOT)** reporting and verification architecture:

1. **Canonical Data Loader & Row-Level Reconstructor (`scripts/reporting_data.py`)**: Loads all authoritative artifacts from `config.yaml`, `results/`, and `results/revision_v2/full_run_001/`, enforces strict schema and seed-set validation, and independently recomputes every summary statistic and 2D Pareto hypervolume from raw trial- and evaluation-level CSVs (`confirmation_runs.csv`, `confirmation_runs_single_obj.csv`, `runs.csv`, `optimizer_trials.csv`, `doe_selection_runs.csv`, `doe_matched_candidate_front.csv`, `final_evaluations.csv`, and `final_summary.csv`). Any missing key, seed mismatch, or discrepancy exceeding tolerance raises `ReportingDataError` (fail-closed).
2. **Unified LaTeX Artifact Generator (`scripts/generate_report_artifacts.py`)**: Generates all 283 macros in `results/macros.tex` and all 11 publication tables in `tables/*.tex` exclusively from `scripts/reporting_data.py` with zero hardcoded numbers, supporting `--write` and non-mutating `--check` modes.
3. **Automated Research Report & Manifest Generator (`scripts/generate_research_reporting.py`)**: Deterministically generates all 9 numerical and narrative result sections of `REPORT.md` inside explicit `<!-- BEGIN AUTO-GENERATED: <SECTION_ID> -->` / `<!-- END AUTO-GENERATED: <SECTION_ID> -->` markers, emits `docs/generated/scientific_results_manifest.json` (58 tracked metrics), and audits `report.tex` and `REPORT.md` in `--check` mode.
4. **Numerical & Mutation Integrity Test Suite (`tests/test_reporting_integrity.py` & `tests/test_no_hardcoded_numbers.py`)**: Verifies row-level reconstructions, parses Markdown and LaTeX tables numerically against canonical artifacts, enforces zero untraced floats outside generated markers, and executes automated negative/mutation tests verifying that corrupting a Markdown table value, confirmation seed, hypervolume reference point, LaTeX cell, or claim guard causes `--check` to fail closed.

No XGBoost models were refit, and all canonical experimental files in `results/` and `config.yaml` remain byte-for-byte unchanged (`git diff --exit-code results/ config.yaml` exits `0`).

---

## 2. Single-Source-of-Truth Architecture

```
Canonical Evidence (Read-Only)
├── config.yaml
├── results/
│   ├── runs.csv, confirmation_runs.csv, confirmation_runs_single_obj.csv, benchmark.csv
│   ├── phase1_summary.json, canonical_analysis.json, desirability.json, confirmation.json
│   ├── icc_analysis.json, lack_of_fit_decomposition.json, residual_diagnostics.json
│   ├── latency_models.json, wild_bootstrap.json, desirability_sensitivity.csv
└── results/revision_v2/full_run_001/
    ├── finalized_selections.json, final_evaluations.csv, final_summary.csv
    ├── optimizer_trials.csv, optimizer_replicates.csv, optimizer_summary.json
    ├── doe_selection_runs.csv, doe_selection_summary.json, doe_matched_candidate_front.csv
    ├── hypervolume.json, paired_comparisons.json, computational_budget.json
    └── latency_measurement.json, latency_interface_overhead.json, run_manifest.json
          │
          ▼
scripts/reporting_data.py (Schema Validation + Independent Row-Level Reconstructions)
          │
          ├──► scripts/generate_report_artifacts.py (--write / --check)
          │      ├── results/macros.tex (283 LaTeX macros)
          │      └── tables/*.tex (11 LaTeX publication tables)
          │
          └──► scripts/generate_research_reporting.py (--write / --check)
                 ├── REPORT.md (9 auto-generated sections; 0 floats outside markers)
                 └── docs/generated/scientific_results_manifest.json (58 tracked metrics)
```

---

## 3. Canonical Data Sources & Independent Row-Level Reconstructions

`scripts/reporting_data.py` performs the following independent reconstructions on every load and fails closed (`ReportingDataError`) if any check deviates beyond tolerance:

| Verification Domain | Raw Source Artifact(s) | Reconstructed Target Artifact | Verified Agreement |
| :--- | :--- | :--- | :--- |
| **Phase 1 & 2 Block Seeds** | `config.yaml`, `results/runs.csv` | `design.random_seeds: [42, 100, 202, 303, 404]` | Exact match ($N = 100$ Phase 1, $N = 140$ Phase 2) |
| **Phase 5 Confirmation Seeds** | `config.yaml`, `results/confirmation_runs.csv`, `results/confirmation_runs_single_obj.csv` | `confirmation.random_seeds: [505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414]` | Exact 10-seed sequence match |
| **Phase 5 Confirmation Means & SDs** | `results/confirmation_runs.csv`, `results/confirmation_runs_single_obj.csv` | `results/confirmation.json` (`Y1_Val_RMSE`, `Y1_Test_RMSE`, `Y2_Latency`, `Y1_SingleObj_Val_RMSE`, `Y1_SingleObj_Test_RMSE`, `Y2_SingleObj_Latency`) | Max absolute difference $< 10^{-9}$ |
| **Frozen Selection Provenance** | `finalized_selections.json`, `final_evaluations.csv`, `final_summary.csv` | 122 selection records, 95 distinct SHA-256 `config_hash` values, $122 \times 20 = 2,440$ evaluation rows across seeds $2001\text{--}2020$ | Exact count & SHA-256 hash set match |
| **Selection-Level Aggregation** | `final_evaluations.csv` (2,440 rows) | `final_summary.csv` (122 rows: mean/SD of `val_rmse`, `test_rmse`, `predict_latency_us`, `inplace_latency_us`) | Max absolute difference $< 10^{-9}$ |
| **Optimizer-Level Aggregation** | `final_summary.csv` & `optimizer_replicates.csv` | `optimizer_summary.json` & `doe_selection_summary.json` (6 optimizers $\times$ 20 replicates: Val/Test RMSE mean, SD, median, IQR, 95% CI, $\sigma_{\text{eval}}$, latency mean/SD, search & benchmark feasibility counts) | Max absolute difference $< 10^{-9}$ |
| **Constrained TPE Feasibility Split** | `optimizer_replicates.csv`, `final_summary.csv` | 20/20 search feasible ($\le 145\,\mu\text{s}$, 30 calls); 9/20 benchmark feasible (1,000 calls): 9/9 depth 6 feasible ($136.70 \pm 0.82\,\mu\text{s}$), 0/11 depth 7 feasible ($150.45 \pm 0.67\,\mu\text{s}$) | Exact count & latency match |
| **Paired & Welch Statistical Comparisons** | `final_summary.csv` | `paired_comparisons.json` (Repeated DOE MO vs MO-TPE diff $-0.002582$, Welch $t = -0.8849, p = 0.3826$, paired $t = -0.9658, p = 0.3463$) | Max absolute difference $< 10^{-6}$ |
| **2D Pareto Hypervolumes** | `optimizer_trials.csv`, `doe_selection_runs.csv`, `doe_matched_candidate_front.csv`, `final_summary.csv` | `hypervolume.json` at reference points $[0.60, 250.0]$ and $[0.65, 275.0]$ across all 4 development protocols and the 12-point holdout non-dominated frontier (3 DOE MO, 3 MO-TPE, 5 cTPE, 1 SO-TPE) | Max absolute difference $< 10^{-9}$ |
| **Computational Budget & Latency Overhead** | `computational_budget.json`, `latency_measurement.json`, `latency_interface_overhead.json` | 17,077 model fits, 1,640,810 timed predictions, $32.27 \pm 0.90\,\mu\text{s}$ DataFrame wrapper overhead ($21.21 \pm 2.41\%$) | Max absolute difference $< 10^{-9}$ |

---

## 4. Auto-Generated `REPORT.md` Structure & Guardrails

`REPORT.md` is partitioned into 9 deterministic auto-generated sections managed by `scripts/generate_research_reporting.py`:

1. `EXEC_SUMMARY`
2. `SECTION_1_FACTORS_AND_BLOCKS`
3. `SECTION_2_PHASE1`
4. `SECTION_3_PHASE2`
5. `SECTION_4_PHASE3`
6. `SECTION_5_PHASE4`
7. `SECTION_6_PHASE5`
8. `SECTION_7_BENCHMARKS`
9. `SECTION_8_9_DISCUSSION_AND_ARTIFACTS`

Outside of these `<!-- BEGIN AUTO-GENERATED: ... -->` / `<!-- END AUTO-GENERATED: ... -->` blocks, `REPORT.md` contains only structural Markdown headings and horizontal rules. `scripts/generate_research_reporting.py --check` enforces that **zero floating-point numbers** appear outside auto-generated blocks.

---

## 5. Machine-Readable Manifest (`docs/generated/scientific_results_manifest.json`)

`docs/generated/scientific_results_manifest.json` tracks 58 canonical metrics across all experimental phases and prospective benchmarks. Each entry records:
- `metric_id`: Unique canonical identifier
- `source_file` and `source_key`: Exact authoritative file and JSON/CSV path
- `raw_value`: Full-precision float/int/list from `ReportingDataset`
- `formatted_values`: Exact formatted strings at 2dp, 4dp, 5dp, 6dp, etc.
- `latex_targets`: Exact macro names in `results/macros.tex` and table files in `tables/*.tex`
- `markdown_targets`: Exact section markers in `REPORT.md`

---

## 6. Mandatory Acceptance Tests (Task 13 Verification)

All three mandatory negative acceptance tests were executed against `python scripts/generate_research_reporting.py --check` to verify fail-closed behavior:

1. **Acceptance Test 1 — Corrupt a generated benchmark value in `REPORT.md`**:
   - **Action**: Temporarily replaced Repeated DOE MO Holdout Test RMSE `0.48997` with `0.99999` in `REPORT.md`.
   - **Result**: `python scripts/generate_research_reporting.py --check` exited with code `1` and printed a unified diff pinpointing the discrepancy in `REPORT.md`.
   - **Restoration**: Restored `REPORT.md` and verified exit code `0`.

2. **Acceptance Test 2 — Corrupt a confirmation seed in `results/confirmation_runs.csv`**:
   - **Action**: Temporarily replaced seed `505` with `999` in `results/confirmation_runs.csv`.
   - **Result**: `python scripts/generate_research_reporting.py --check` exited with code `1`:
     `Canonical reporting dataset validation failed: results/confirmation_runs.csv seeds (999, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414) do not match config.yaml (505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414)`.
   - **Restoration**: Restored `results/confirmation_runs.csv` and verified exit code `0`.

3. **Acceptance Test 3 — Corrupt a hypervolume reference point in `results/revision_v2/full_run_001/hypervolume.json`**:
   - **Action**: Temporarily replaced reference point `250.0` with `260.0` in `results/revision_v2/full_run_001/hypervolume.json`.
   - **Result**: `python scripts/generate_research_reporting.py --check` exited with code `1`:
     `Canonical reporting dataset validation failed: Missing required keys in hypervolume.json:development_domain.reference_points: ['[0.6, 250.0]']`.
   - **Restoration**: Restored `hypervolume.json` and verified exit code `0`.

In addition, these mutation tests are permanently encoded in `tests/test_reporting_integrity.py` (`test_mutation_corrupted_markdown_benchmark_value_fails_check`, `test_mutation_corrupted_confirmation_seeds_fails_closed`, `test_mutation_corrupted_hypervolume_reference_point_fails_closed`, and `test_mutation_stale_latex_or_missing_marker_or_untraced_float_fails`) and run on every CI build.

---

## 7. Final Verification Summary

| Verification Command | Result | Details |
| :--- | :---: | :--- |
| `python -m pytest -q` | **PASS** | `126 passed` (zero skipped tests) |
| `python scripts/audit_scientific_consistency.py` | **PASS** | All 7 raw-artifact consistency checks passed |
| `python scripts/generate_report_artifacts.py --check` | **PASS** | All 283 macros (`results/macros.tex`) and 11 tables (`tables/*.tex`) match canonical artifacts |
| `python scripts/generate_research_reporting.py --check` | **PASS** | `REPORT.md`, `docs/generated/scientific_results_manifest.json`, LaTeX artifacts, and manuscript claim checks passed |
| `git diff --exit-code results/ config.yaml` | **PASS** | Zero modifications to canonical experimental evidence |
