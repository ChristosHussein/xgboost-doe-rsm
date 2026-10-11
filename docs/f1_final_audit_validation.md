# Work Package F.1 — Final Scientific Audit Corrections and CI Stabilization Validation Report

**Repository:** `ChristosHussein/xgboost-doe-rsm`  
**Branch:** `codex/scientific-revision`  
**Audit Scope:** Cross-platform CI stabilization (`38161948306` / Ubuntu job `114536539457`), full numerical reconciliation across `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md`, deterministic audit-document generation and mutation testing, follow-up experimental protocol redesign (Protocols A–E), and repository path verification.

---

## 1. Root Cause and Resolution of Cross-Platform CI Failure

### 1.1 Diagnosis of GitHub Actions Run `38161948306` (Ubuntu Job `114536539457`)

In GitHub Actions pull-request workflow run `38161948306`, the Windows job passed all tests (`128 passed`), whereas the Ubuntu 24.04 job (`114536539457`) reported **`125 passed, 3 failed`**:

1. `tests/test_reporting_integrity.py::test_generate_report_artifacts_check_passes_on_clean_repo`
2. `tests/test_reporting_integrity.py::test_audit_scientific_consistency_passes_on_clean_repo`
3. `tests/test_reporting_integrity.py::test_work_package_f_scientific_audit_reconstructions_and_deliverables`

All three failures traced to a single floating-point representation difference in `docs/generated/scientific_results_manifest.json` under `metrics.phase1.curvature_f_block_adj_pooled`:

- **Committed manifest (Windows x86_64):** `"value": 20235.41158394235`
- **Reconstructed on Ubuntu x86_64:** `"value": 20235.411583942332`
- **Formatted reporting value on both platforms:** `"formatted": "20,235.41"`

### 1.2 Mathematical and Numerical Root Cause

In `scripts/reporting_data.py`, the block-adjusted pooled coordinate pure-error sum of squares (`ss_block_adj_pooled`, $\text{df} = 79$) was previously extracted from `float(fit_p1_cell.ssr)` where `fit_p1_cell = ols("val_rmse ~ C(block) + C(point_id)", data=df_p1).fit()`. Because `C(block) + C(point_id)` fits a 21-parameter design matrix via LAPACK/OpenBLAS QR/SVD routines, platform-specific SIMD instruction ordering between Windows and Linux produced a $1\text{ ULP}$ ($10^{-19}$) difference in the residual sum of squares (`0.0011260444...`). When `ss_curv = 0.2887857845` was divided by `ms_block_adj_pooled = ss_block_adj_pooled / 79.0` ($\approx 1.4253727 \times 10^{-5}$), that $1\text{ ULP}$ denominator difference amplified by $79 / 0.001126 \approx 7 \times 10^4$, shifting the 16th significant digit of $F_{\text{curv,block-pool}}$ by $1.8 \times 10^{-11}$ (`20235.41158394235` vs. `20235.411583942332`).

### 1.3 Principled Two-Part Fix in `scripts/reporting_data.py`

1. **Exact Orthogonal Algebraic Decomposition**: Because every Phase 1 design coordinate is replicated across the exact same 5 balanced seed blocks, the indicator column space of `C(point_id)` is strictly orthogonal to the centered block contrast space of `C(block)`. Consequently, the exact block-adjusted pooled pure-error sum of squares satisfies the algebraic identity:
   $$\text{SS}_{\text{PE,coord-block}} = \text{SS}_{\text{PE,coord-unadj}} - \text{SS}_{\text{block}}$$
   Replacing `float(fit_p1_cell.ssr)` with `ss_block_adj_pooled = float(ss_pe_coord_unadj - ss_block_p1)` in `scripts/reporting_data.py` (while verifying `_assert_close(ss_block_adj_pooled, float(fit_p1_cell.ssr), atol=1e-12)`) eliminates the 21-column LAPACK SVD solve from the denominator calculation.
2. **Deterministic Floating-Point Serialization Policy**: In `build_scientific_results_manifest(ds)` (`scripts/reporting_data.py`), floating-point `"value"` entries are normalized via `round(float(value), 8)` ($10^{-8}$ decimal precision), which is 4 to 6 orders of magnitude tighter than any manuscript display format (`2` to `4` decimal places) while invariant to IEEE-754 last-bit platform drift across operating systems. Exact checks are preserved without relaxation for all metric identifiers, source filenames, SHA-256 hashes, integer counts, seed lists, protocol strings, and formatted reporting values. Both `--check` modes (`scripts/generate_report_artifacts.py --check` and `scripts/generate_research_reporting.py --check`) remain strictly non-mutating.

---

## 2. Reconciliation of Scientific Audit Documents Against Canonical Reconstructions

Every numerical statement across `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md` was audited and reconciled against `ReportingDataset` (`scripts/reporting_data.py`). All three documents are now deterministically rendered from `ReportingDataset` so manual transcription errors cannot recur.

| Quantity | Previous Incorrect Audit Value | Verified Canonical Value (`ReportingDataset`) | Canonical Source Artifact & Reconstruction |
| :--- | :--- | :--- | :--- |
| Phase 1 curvature sum of squares ($\text{SS}_{\text{curv}}$, $\text{df}=1$) | `2.6602` | **`0.288786`** (`0.2887857845`) | `results/phase1.json` (`ss_curvature`) and 100-run Phase 1 OLS reconstruction in `scripts/reporting_data.py` |
| Phase 1 center-point pure-error mean square ($\text{MS}_{\text{PE,center}}$, $\text{df}=15$) | `2.523e-5` | **`2.7391e-6`** (`2.739066861e-6`) | `results/phase1.json` (`ms_pe_center`) across 20 center runs (`4` per block $\times 5$ blocks, $\text{SS}_{\text{PE,center}} = 4.10860 \times 10^{-5}$) |
| Phase 1 block-adjusted pooled coordinate pure-error MS ($\text{df}=79$) | `2.633e-4` | **`1.4254e-5`** (`1.425372698e-5`) | `scripts/reporting_data.py` (`ss_block_adj_pooled = 0.00112604`, yielding $F(1,79) = 20,235.41$) |
| Phase 1 unadjusted pooled coordinate pure-error MS ($\text{df}=83$) | `4.195e-4` | **`6.9760e-5`** (`6.976012048e-5`) | `scripts/reporting_data.py` (`ss_pe_coord_unadj = 0.00579009`, yielding $F(1,83) = 4,140.06$) |
| Repeated 20-campaign DOE MO holdout test RMSE ($\text{mean} \pm \text{SD}$) | `0.48373 ± 0.00511` | **`0.48997 ± 0.01325`** | `results/revision_v2/full_run_001/final_summary.csv` and `results/revision_v2/full_run_001/doe_selection_summary.json` |
| Repeated 20-campaign MO-TPE holdout test RMSE ($\text{mean} \pm \text{SD}$) | `0.48632 ± 0.00548` | **`0.49255 ± 0.00284`** | `results/revision_v2/full_run_001/final_summary.csv` and `results/revision_v2/full_run_001/optimizer_summary.json` (Welch $\Delta = -0.00258, t = -0.8518, p = 0.3826$; paired $t = -0.922, p = 0.3680$ in `results/revision_v2/full_run_001/paired_comparisons.json`) |
| Repeated 20-campaign MO-TPE development hypervolume SD (at `[0.60, 250.0]`) | `0.2584` | **`0.3301`** (`18.3834 ± 0.3301`) | `results/revision_v2/full_run_001/hypervolume.json` (`mo_tpe_hypervolume_distribution`, vs. Repeated DOE `15.6128 ± 0.4535`) |
| Holdout Pareto front hypervolume (primary reference `[0.60, 250.0]`) | `17.6486` | **`17.6714`** (seed-bootstrap **`17.6923 ± 0.2454`**, 95% CI `[17.2072, 18.1804]`) | `results/revision_v2/full_run_001/hypervolume.json` (`external_test_domain`) and `scripts/reporting_data.py` |
| Constrained TPE ($\le 145\,\mu\text{s}$) depth-specific latency distribution | Fabricated examples (`max_depth=4: 48.2 us`, `max_depth=6: 84.5 us`, `max_depth=7: 118.4 us`) | **9 depth-6 selections** (`9/9` benchmark-feasible at **`119.35 ± 9.75 us`**) and **11 depth-7 selections** (`0/11` benchmark-feasible at **`154.07 ± 4.02 us`**); overall `20/20` search-feasible $\to$ `9/20` (`45%`) benchmark-feasible, latency **`138.45 ± 18.81 us`**, holdout test RMSE **`0.47036 ± 0.00297`** | `results/revision_v2/full_run_001/finalized_selections.json` and `results/revision_v2/full_run_001/final_summary.csv` |
| Candidate-specific surrogate predictions and confirmation optimism | Misattributed across candidates | **$\mathbf{x}^*_{\text{MO}}$ (`max_depth=4`, $\eta=0.1449$, $s=1.0$, $\lambda=2.6607$):** pred val RMSE `0.4892`, empirical val `0.4893 ± 0.0024` ($\in [0.4648, 0.5137]$), test `0.4898 ± 0.0030`, pred lat `137.13 us`, empirical lat `135.75 ± 2.91 us`; **$\mathbf{x}^*_{\text{SO}}$ (`max_depth=7`, $\eta=0.2200$, $s=1.0$, $\lambda=10.0$):** pred val RMSE `0.4534`, empirical val `0.4751 ± 0.0023` ($\notin [0.4275, 0.4737]$, optimism bias **`+0.0217`** RMSE), test `0.4767 ± 0.0029`, pred lat `215.2 us`, empirical lat `179.96 ± 2.53 us` | `results/confirmation.json` and `results/phase3.json` |
| Historical vs. Repeated DOE discrete grid protocol | Undocumented in claims register | **1,323-point coded lattice** (`doe_grid_size = 1323`); historical single-campaign SO hash `0079fbae0ec5b7cb` ($\eta=0.2200$) vs. repeated 20-campaign SO hash `9d7c4e74ff156870` (grid index `1052`, $\eta=0.1519$); historical MO Knee hash `b05e8e5cb67e8668` vs. `12` unique grid selections across 20 repeated campaigns | `results/revision_v2/full_run_001/doe_selection_protocol.json` and `results/revision_v2/full_run_001/finalized_selections.json` |

Where historical or superseded values are discussed in `docs/scientific_claims_audit.md` (such as the obsolete pre-revision linear latency extrapolation $-23.22\,\mu\text{s}$ or the un-pre-registered TOST margin $\pm 0.005$), they are explicitly labeled with their historical source version and classified as **UNSUPPORTED** or **NEEDS QUALIFICATION**, never as `VERIFIED` or `SUPPORTED`.

---

## 3. Drift Protection and Mutation Testing for Audit Documents

To ensure that `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md` can never drift from canonical experimental artifacts:

1. **Automated Generation from `ReportingDataset`**: `scripts/generate_research_reporting.py` implements `render_scientific_claims_audit(ds)`, `render_f_scientific_review_validation(ds)`, and `render_proposed_followup_experiments(ds)`. Every empirical value in those three documents is interpolated directly from `ReportingDataset`.
2. **Non-Mutating Verification in `--check` Mode**: `audit_manuscript_integrity(ds)` in `scripts/generate_research_reporting.py` compares the on-disk contents of all three audit documents against `render_*(ds)` and fails with a unified diff if any number, table cell, or claim entry is modified out of band.
3. **Closed-Loop Mutation Tests**: `tests/test_reporting_integrity.py::test_mutation_corrupted_audit_docs_or_nonexistent_paths_fail_closed` mutates empirical values in each of the three audit documents (`0.288786 -> 2.6602`, `0.48997 -> 0.48373`, `0.47036 -> 0.46111`) and injects a nonexistent file reference, asserting that `check_research_reporting()` and `audit_manuscript_integrity()` fail closed with clear diagnostic messages.

---

## 4. Corrected Follow-Up Experiment Designs (`docs/proposed_followup_experiments.md`)

`docs/proposed_followup_experiments.md` was redesigned to fix the local CCD point-count arithmetic, distinguish outer split seeds from inner XGBoost seeds, and restore all five requested follow-up protocols:

1. **Protocol A — Constrained DOE/RSM Under the $145\,\mu\text{s}$ Latency Ceiling**: Replaces soft desirability trade-offs with hard upper-prediction-bound latency filtering ($\hat{Y}_2(\mathbf{x}) + t_{0.95,\nu}\text{SE}(\hat{Y}_2(\mathbf{x})) \le 145\,\mu\text{s}$) across the 20 existing Repeated DOE surfaces (`3001..3100`) and evaluates warmed search-time timing (`50` warmup + `200` timed calls) for both DOE and Constrained TPE under a 140-fit budget.
2. **Protocol B — Matched-Protocol Repeated-Campaign DOE vs. TPE Parity Benchmark**: Executes a truly paired $R = 20$ benchmark on shared 5-split validation blocks (`6001..6100`), crossing design strategy (blocked FCCD vs. adaptive TPE) with candidate extraction rule (continuous SLSQP polish vs. 1,323-point coded grid evaluation), with a pre-registered TOST equivalence margin $\Delta_{\text{TOST}} = \pm 0.0025$ RMSE.
3. **Protocol C — Truly Independent External Confirmation**: Evaluates frozen configurations on 20 fresh outer 80/20 California Housing partitions (`7001..7020`, disjoint from historical/Revision-v2 outer split `42`, confirmation `505..1414`, evaluation `2001..2020`, and DOE blocks `3001..3100`) as well as untouched OpenML tabular regression benchmarks (`Miami Housing`, `Diamonds`, `Superconductivity`, `Elevators`).
4. **Protocol D — True Two-Stage Sequential / Region-Zooming RSM Under an Exact 140-Evaluation Budget**:
   - Explicitly corrects the local CCD point count: a reduced $2^{4-1}_{\text{IV}}$ local CCD ($8$ corners $+ 8$ axial $+ 2$ centers) has $18$ runs/block (not $14$), and a full 4-factor local FCCD ($16$ corners $+ 8$ axial $+ 1$ center) has $25$ runs/block.
   - Allocates **strictly 140 XGBoost fits per search replicate**:
     - **Stage 1A (50 fits)**: $2^{4-1}_{\text{IV}}$ fractional factorial ($8$ corners $+ 2$ center replicates $= 10$ runs/block) across $B = 5$ blocks ($50$ fits).
     - **Stage 1B (15 fits)**: Constrained steepest-descent line search at $3$ step sizes across $B = 5$ blocks ($15$ fits; cumulative $65$ fits).
     - **Stage 2 (75 fits)**: Full 25-point re-centered local FCCD ($16$ corners $+ 8$ axial $+ 1$ center) excluding $\eta = 0.01$ across $B_2 = 3$ blocks ($25 \times 3 = 75$ fits; cumulative $\mathbf{50 + 15 + 75 = 140}$ fits), plus an explicit 5-block sensitivity variant ($50 + 10 + 80 = 140$ fits using $8$ corners $+ 6$ active-factor axial points $+ 2$ centers $= 16$ runs/block $\times 5$ blocks).
5. **Protocol E — Multi-Dataset ($D=6$), Multi-Horizon ($M \in \{100, 300, 500\}$), and 9-Factor Definitive Screening Design**: Extends the framework across 6 tabular datasets $\times 3$ boosting horizons and a 9-factor Definitive Screening Design ($k = 9$, $2k + 3 = 21$ runs/block $\times 3$ blocks $= 63$ Stage 1 fits $+ 75$ Stage 2 local FCCD fits $+ 42$ ridge/confirmation fits $= 180$ fits).

---

## 5. Corrected Evidence-Hash and Repository Path References

1. **Removal of Nonexistent File References**: Previous versions of the audit documents and manuscript referenced nonexistent files ("docs/canonical_evidence_hashes.json", "phase1_2_doe.py", "phase5_benchmarks.py", "phase5_confirmation.py", "revision_experiments.py", and "tables/tab_seeds.tex"). All references across `REPORT.md`, `report.tex`, `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md` have been updated to the actual repository paths:
   - `CANONICAL_EVIDENCE_SHA256` in `scripts/reporting_data.py` (verified by `verify_canonical_evidence_hashes()`)
   - `docs/generated/scientific_results_manifest.json`
   - `pipeline.py`, `run_pipeline.py`, `desirability_provenance.py`, `final_evaluation.py`
   - `scripts/run_benchmarks.py`, `scripts/run_confirmation.py`
   - `tables/tab_design_block_seeds.tex`
2. **Automated Repository Path Guard**: `_audit_referenced_repo_paths(ds, doc_name, doc_text)` in `scripts/generate_research_reporting.py` scans backtick-quoted repository paths across `REPORT.md` and all audit documents, failing `--check` if any referenced file or directory does not exist on disk.

---

## 6. Final Verification Summary

| Verification Gate | Command | Result |
| :--- | :--- | :--- |
| LaTeX Artifacts Non-Mutating Check | `python scripts/generate_report_artifacts.py --check` | **PASSED** (`results/macros.tex` and 15 `tables/*.tex` verified) |
| Full Research Reporting & Audit Docs Check | `python scripts/generate_research_reporting.py --check` | **PASSED** (`REPORT.md`, manifest, and all Work Package F/F.1 audit docs verified) |
| Standalone Scientific Consistency Auditor | `python scripts/audit_scientific_consistency.py` | **PASSED** (40 SHA-256 evidence hashes + 65 manifest metrics + audit docs verified) |
| Automated Pytest Suite (incl. Mutation Tests) | `python -m pytest -q` | **PASSED** (`129 passed`) |
| LaTeX Manuscript Compilation | `tectonic report.tex` | **PASSED** (`report.pdf` compiled cleanly) |
| Frozen Experimental Evidence Integrity | `git diff --exit-code results/phase1.json results/phase3.json results/confirmation.json results/runs.csv results/revision_v2/full_run_001/ config.yaml` | **PASSED** (0 bytes modified across all 40 frozen files) |

READY FOR INDEPENDENT MERGE REVIEW
