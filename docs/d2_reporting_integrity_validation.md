# Work Package D.2 — Single Source of Truth, Automated Scientific Reporting, and Numerical Integrity Validation

## 1. Executive Summary

Work Package D.2 eliminates the possibility of inconsistent scientific numbers appearing across `REPORT.md`, `report.tex`, `results/macros.tex`, `tables/*.tex`, and `report.pdf` by replacing manual numerical transcription and string-presence tests with a **Single Source of Truth (SSOT)** reporting and verification architecture:

1. **Canonical Data Loader, Cryptographic Hash Verifier, & Row-Level Reconstructor (`scripts/reporting_data.py`)**: Verifies the cryptographic SHA-256 digests of all 40 canonical experimental artifacts (`CANONICAL_EVIDENCE_SHA256` and `results/revision_v2/full_run_001/provenance.json`), loads all authoritative artifacts from `config.yaml`, `results/`, and `results/revision_v2/full_run_001/`, enforces strict schema and seed-set validation, and independently recomputes every summary statistic and 2D Pareto hypervolume from raw trial- and evaluation-level CSVs (`confirmation_runs.csv`, `confirmation_runs_single_obj.csv`, `runs.csv`, `optimizer_trials.csv`, `doe_selection_runs.csv`, `doe_matched_candidate_front.csv`, `final_evaluations.csv`, and `final_summary.csv`). Any tampered evidence file, missing key, seed mismatch, or discrepancy exceeding tolerance raises `ReportingDataError` (fail-closed).
2. **Unified LaTeX Artifact Generator (`scripts/generate_report_artifacts.py`)**: Generates all 362 macros in `results/macros.tex` (covering every empirical numerical claim in `report.tex` prose) and all 11 publication tables in `tables/*.tex` exclusively from `scripts/reporting_data.py` with zero hardcoded numbers, supporting `--write` and non-mutating `--check` modes.
3. **Automated Research Report & Manifest Generator (`scripts/generate_research_reporting.py`)**: Deterministically generates all 9 numerical and narrative result sections of `REPORT.md` inside explicit `<!-- BEGIN AUTO-GENERATED: <SECTION_ID> -->` / `<!-- END AUTO-GENERATED: <SECTION_ID> -->` markers, emits `docs/generated/scientific_results_manifest.json` (58 tracked metrics), and audits `report.tex` and `REPORT.md` in `--check` mode (including strict enforcement that no empirical result floating-point numbers are hardcoded in `report.tex` prose).
4. **Numerical & Mutation Integrity Test Suite (`tests/test_reporting_integrity.py` & `tests/test_no_hardcoded_numbers.py`)**: Verifies cryptographic SHA-256 evidence hashes, row-level reconstructions, parses Markdown and LaTeX tables numerically against canonical artifacts, enforces zero untraced floats outside generated markers in `REPORT.md` and zero hardcoded empirical floats in `report.tex` prose, and executes automated negative/mutation tests verifying that corrupting a Markdown table value, confirmation seed, hypervolume reference point, LaTeX prose float, LaTeX cell, canonical evidence file, or claim guard causes `--check` to fail closed.

No XGBoost models were refit, and all canonical experimental files in `results/` and `config.yaml` remain byte-for-byte unchanged (`git diff --exit-code results/ config.yaml`, `git diff --exit-code 842b531611ded9ff8495b88bf0c12790efe56b69 -- config.yaml results/revision_v2/full_run_001/`, and `git diff --exit-code v1.0.0 -- results/*.csv results/*.json` all exit `0`).

---

## 2. Single-Source-of-Truth Architecture

```
Canonical Evidence (Read-Only, Cryptographically Verified via SHA-256 & Immutable Git Refs)
├── config.yaml
├── results/ (Immutable v1.0.0 Tag Baseline)
│   ├── runs.csv, confirmation_runs.csv, confirmation_runs_single_obj.csv, benchmark.csv
│   ├── phase1.json, phase3.json, confirmation.json, lof.json, diagnostics.json, icc.json, env.json
│   ├── latency_modeling.json, latency_models_comparison.csv, anova_ccd_latency.csv
│   └── benchmark_evals_trajectories.csv, benchmark_optimizer_incumbents.csv, benchmark_summary.json, depth_opt_table.csv, desirability_sensitivity.csv, ridge_table.csv
└── results/revision_v2/full_run_001/ (Frozen Campaign Baseline 842b5316 + provenance.json)
    ├── finalized_selections.json, final_evaluations.csv, final_summary.csv
    ├── optimizer_trials.csv, optimizer_replicates.csv, optimizer_summary.json
    ├── doe_selection_runs.csv, doe_selection_summary.json, doe_matched_candidate_front.csv
    ├── hypervolume.json, paired_comparisons.json, computational_budget.json
    └── latency_measurement.json, latency_interface_overhead.json, run_manifest.json, provenance.json
          │
          ▼
scripts/reporting_data.py (40-File SHA-256 Check + Schema Validation + Row-Level Reconstructions)
          │
          ├──► scripts/generate_report_artifacts.py (--write / --check)
          │      ├── results/macros.tex (362 LaTeX macros)
          │      └── tables/*.tex (11 LaTeX publication tables)
          │
          └──► scripts/generate_research_reporting.py (--write / --check)
                 ├── REPORT.md (9 auto-generated sections; 0 floats outside markers)
                 ├── report.tex audit (0 hardcoded empirical floats in prose)
                 └── docs/generated/scientific_results_manifest.json (58 tracked metrics)
```

---

## 3. Canonical Data Sources & Independent Row-Level Reconstructions

`scripts/reporting_data.py` performs the following independent reconstructions on every load and fails closed (`ReportingDataError`) if any check deviates beyond tolerance:

| Verification Domain | Raw Source Artifact(s) | Reconstructed Target Artifact | Verified Agreement |
| :--- | :--- | :--- | :--- |
| **Cryptographic Evidence Hashes** | `config.yaml`, 20 `v1.0.0` `results/*` files, `provenance.json`, 18 `full_run_001/*` files | `CANONICAL_EVIDENCE_SHA256` & `provenance.json:artifact_sha256` (40 files total) | Exact SHA-256 digest match |
| **Phase 1 & 2 Block Seeds** | `config.yaml`, `results/runs.csv` | `design.random_seeds: [42, 101, 202, 303, 404]` | Exact match ($N = 100$ Phase 1, $N = 140$ Phase 2) |
| **Phase 5 Confirmation Seeds** | `config.yaml`, `results/confirmation_runs.csv`, `results/confirmation_runs_single_obj.csv` | `confirmation.random_seeds: [505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414]` | Exact 10-seed sequence match |
| **Phase 5 Confirmation Means & SDs** | `results/confirmation_runs.csv`, `results/confirmation_runs_single_obj.csv` | `results/confirmation.json` (`Y1_Val_RMSE`, `Y1_Test_RMSE`, `Y2_Latency`, `Y1_SingleObj_Val_RMSE`, `Y1_SingleObj_Test_RMSE`, `Y2_SingleObj_Latency`) | Max absolute difference $< 10^{-9}$ |
| **Frozen Selection Provenance** | `finalized_selections.json`, `final_evaluations.csv`, `final_summary.csv` | 122 selection records, 95 distinct SHA-256 `config_hash` values, $122 \times 20 = 2,440$ evaluation rows across seeds $2001\text{--}2020$ | Exact count & SHA-256 hash set match |
| **Selection-Level Aggregation** | `final_evaluations.csv` (2,440 rows) | `final_summary.csv` (122 rows: mean/SD of `val_rmse`, `test_rmse`, `predict_latency_us`, `inplace_latency_us`) | Max absolute difference $< 10^{-9}$ |
| **Optimizer-Level Aggregation** | `final_summary.csv` & `optimizer_replicates.csv` | `optimizer_summary.json` & `doe_selection_summary.json` (6 optimizers $\times$ 20 replicates: Val/Test RMSE mean, SD, median, IQR, 95% CI, $\sigma_{\text{eval}}$, latency mean/SD, search & benchmark feasibility counts) | Max absolute difference $< 10^{-9}$ |
| **Constrained TPE Feasibility Split** | `optimizer_replicates.csv`, `final_summary.csv` | 20/20 search feasible ($\le 145\,\mu\text{s}$, 30 calls); 9/20 benchmark feasible (1,000 calls): 9/9 depth 6 feasible ($136.70 \pm 0.82\,\mu\text{s}$), 0/11 depth 7 feasible ($150.45 \pm 0.67\,\mu\text{s}$) | Exact count & latency match |
| **Paired & Welch Statistical Comparisons** | `final_summary.csv` | `paired_comparisons.json` (Repeated DOE MO vs MO-TPE diff $-0.002582$, Welch $t = -0.8849, p = 0.3826$, paired $t = -0.9658, p = 0.3463$) | Max absolute difference $< 10^{-6}$ |
| **2D Pareto Hypervolumes** | `optimizer_trials.csv`, `doe_selection_runs.csv`, `doe_matched_candidate_front.csv`, `final_summary.csv` | `hypervolume.json` at reference points $[0.60, 250.0]$ and $[0.65, 275.0]$ across all 4 development protocols and the 12-point holdout non-dominated frontier (3 DOE MO, 3 MO-TPE, 5 cTPE, 1 SO-TPE) | Max absolute difference $< 10^{-9}$ |
| **Computational Budget & Latency Overhead** | `computational_budget.json`, `latency_measurement.json`, `latency_interface_overhead.json` | 17,077 model fits, 1,640,810 timed predictions, $32.19 \pm 1.46\,\mu\text{s}$ wrapper overhead across 610 session measurements | Max absolute difference $< 10^{-9}$ |

---

## 4. Auto-Generated `REPORT.md` Structure & `report.tex` Macro Guardrails

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

Outside of these `<!-- BEGIN AUTO-GENERATED: ... -->` / `<!-- END AUTO-GENERATED: ... -->` blocks, `REPORT.md` contains only structural Markdown headings and horizontal rules. `scripts/generate_research_reporting.py --check` enforces that:
- **Zero floating-point numbers** appear outside auto-generated blocks in `REPORT.md`.
- **Zero empirical result floating-point numbers** are hardcoded in `report.tex` prose (every empirical numerical claim in `report.tex` resolves via `\num...`, `\vecB...`, `\matB...`, or `\eigVec...` macros from `results/macros.tex`, or generated tables in `tables/*.tex`).

---

## 5. Machine-Readable Manifest (`docs/generated/scientific_results_manifest.json`)

`docs/generated/scientific_results_manifest.json` tracks 58 canonical metrics across all experimental phases and prospective benchmarks. Each entry records:
- `metric_id`: Unique canonical identifier
- `input_artifact_path` and `source_field_or_reconstruction`: Exact authoritative file and JSON/CSV path
- `value`: Full-precision float/int/list from `ReportingDataset`
- `formatted_value` and `precision`: Canonical formatted string and rounding precision
- `target_documents`: Exact macro names in `results/macros.tex`, table files in `tables/*.tex`, and section markers in `REPORT.md`

---

## 6. Mandatory Acceptance Tests (Task 13 Verification)

All negative acceptance tests are executed against `python scripts/generate_research_reporting.py --check` and encoded permanently in `tests/test_reporting_integrity.py`:

1. **Acceptance Test 1 — Corrupt a generated benchmark value in `REPORT.md`**:
   - **Action**: Temporarily replaced Repeated DOE MO Holdout Test RMSE `0.48997` with `0.99999` in `REPORT.md`.
   - **Result**: `python scripts/generate_research_reporting.py --check` exited with code `1` and printed a unified diff pinpointing the discrepancy in `REPORT.md`.
   - **Restoration**: Restored `REPORT.md` and verified exit code `0`.

2. **Acceptance Test 2 — Corrupt a confirmation seed in `results/confirmation_runs.csv`**:
   - **Action**: Temporarily replaced seed `505` with `999` in `results/confirmation_runs.csv`.
   - **Result**: `python scripts/generate_research_reporting.py --check` exited with code `1` (caught by both SHA-256 digest verification and seed-set validation).
   - **Restoration**: Restored `results/confirmation_runs.csv` and verified exit code `0`.

3. **Acceptance Test 3 — Corrupt a hypervolume reference point in `results/revision_v2/full_run_001/hypervolume.json`**:
   - **Action**: Temporarily replaced reference point `250.0` with `260.0` in `results/revision_v2/full_run_001/hypervolume.json`.
   - **Result**: `python scripts/generate_research_reporting.py --check` exited with code `1` (caught by both `provenance.json` SHA-256 verification and reference-point key validation).
   - **Restoration**: Restored `hypervolume.json` and verified exit code `0`.

4. **Acceptance Test 4 — Inject a hardcoded empirical float into `report.tex` prose**:
   - **Action**: Temporarily replaced `\numRevHvMoTpeDevMean` with `18.3834` in `report.tex`.
   - **Result**: `audit_manuscript_integrity` flagged `Hardcoded empirical floating-point numbers found in report.tex prose (must use generated macros from results/macros.tex): ['L479: 18.3834']`.

In addition, these mutation tests are permanently encoded in `tests/test_reporting_integrity.py` (`test_mutation_corrupted_markdown_benchmark_value_fails_check`, `test_mutation_corrupted_confirmation_seeds_fails_closed`, `test_mutation_corrupted_hypervolume_reference_point_fails_closed`, `test_mutation_stale_latex_or_missing_marker_or_untraced_float_fails`, and `test_cryptographic_evidence_hash_protection_fails_on_tampered_artifact`) and run on every CI build.

---

## 7. Final Verification Summary

| Verification Command | Result | Details |
| :--- | :---: | :--- |
| `python -m pytest -q` | **PASS** | `127 passed` (zero skipped tests) |
| `python scripts/audit_scientific_consistency.py` | **PASS** | All 8 raw-artifact and SHA-256 cryptographic consistency checks passed |
| `python scripts/generate_report_artifacts.py --check` | **PASS** | All 362 macros (`results/macros.tex`) and 11 tables (`tables/*.tex`) match canonical artifacts |
| `python scripts/generate_research_reporting.py --check` | **PASS** | `REPORT.md`, `docs/generated/scientific_results_manifest.json`, LaTeX artifacts, and `report.tex` prose checks passed |
| `git diff --exit-code results/ config.yaml` + immutable ref diffs (`v1.0.0`, `842b5316`) | **PASS** | Zero uncommitted or committed modifications to canonical experimental evidence |
