# Work Package F / F.1 — Independent Scientific and Methodological Audit Validation Report

**Repository:** `ChristosHussein/xgboost-doe-rsm`  
**Branch:** `codex/scientific-revision`  
**Baseline Commit:** `6b53dac85af261553a7870f4888d5ea7423d4b5e` (Work Package F) / `6050816e0357a98d77a37b2f58fcf7da2a0cf526` (Work Package F.1)  
**Canonical Campaign:** `results/revision_v2/full_run_001/` & Historical `v1.0.0` (`results/*.json`, `results/*.csv`)

---

## 1. Executive Summary

Work Package F (and its Work Package F.1 reconciliation) audited the scientific validity, methodological terminology, statistical assumptions, experimental comparisons, and scope conclusions of the study on top of the single-source-of-truth reporting architecture established in Work Package D.2.

Crucially, **zero new XGBoost model fits were executed** and **all 40 frozen experimental evidence files (`results/` and `config.yaml`) remain byte-identical** to the immutable SHA-256 baseline (`CANONICAL_EVIDENCE_SHA256` and `verify_canonical_evidence_hashes()` in `scripts/reporting_data.py` plus `results/revision_v2/full_run_001/provenance.json`). Every sensitivity metric, chronology reconstruction, latency overhead summary, computational budget breakdown, and holdout Pareto bootstrap interval is computed deterministically from the existing frozen artifacts in `scripts/reporting_data.py` and propagated automatically to `results/macros.tex`, `tables/*.tex`, `REPORT.md`, `report.tex`, `report.pdf`, `docs/generated/scientific_results_manifest.json`, `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md`.

---

## 2. Task-by-Task Audit Resolution Matrix

| Task | Audit Scope | Key Scientific Finding & Resolution | Primary Artifacts Updated |
| :--- | :--- | :--- | :--- |
| **Task 1** | Scientific Claims Register | Inventoried 22 material claims across abstract, introduction, methods, results, discussion, conclusion, tables, and captions (`SUPPORTED`, `NEEDS QUALIFICATION`, and `UNSUPPORTED` prior to correction; all 22 resolved and deterministically verified from `ReportingDataset`). | `docs/scientific_claims_audit.md` |
| **Task 2** | Experimental Design Terminology & Budget | Proved from `pipeline.py`, `run_pipeline.py`, and `results/runs.csv` (`30` of `40` axial runs occurred at `run_order <= 100`) that the 140-run FCCD was generated upfront as one pre-specified randomized batch. Removed misleading "Sequential RSM" from titles/prose and contrasted `25` distinct coordinates ($\times 5$ blocks + $15$ within-block center replicates) against `140` unique adaptive TPE coordinates. | `report.tex`, `REPORT.md`, `scripts/reporting_data.py` |
| **Task 3** | Headline Curvature Test Sensitivity | Reconstructed $F(1, 15) = 105,432.31$ (`SS_curv = 0.288786`, `MS_PE,center = 2.74 \times 10^{-6}` from within-block subsampling seeds at fixed split $S_b$) and added sensitivity analyses across 6 alternative error denominators ($F = 20,235.41$ on 79-df block-adjusted pooled coordinate error; $F = 17,013.24$ on 64-df $T\times B$ interaction; $F = 7,342.69, t = 85.69, p = 1.11 \times 10^{-7}$ on 4-df paired block contrast; $F = 82.86, p = 3.34 \times 10^{-14}$ on 85-df unadjusted 2FI residual). | `scripts/reporting_data.py`, `report.tex`, `REPORT.md` |
| **Task 4** | Statistical Assumptions & Inference | Explicitly documented grouping factors for Levene's test (5 seed blocks, $W = 0.1754, p = 0.9507$) vs. Brown–Forsythe (25 coordinates, $W = 0.6872, p = 0.8552$) vs. Breusch-Pagan ($\text{LM} = 70.10, p < 0.0001$), designated unpaired Welch's $t$-test as primary for Repeated DOE MO vs. MO-TPE ($0.48997$ vs. $0.49255$, $\Delta = -0.00258, t = -0.8849, p = 0.3826$) with index-paired $t$-test ($t = -0.966, p = 0.3463$) as secondary sensitivity, and replaced all equivalence phrasing with non-detection language. | `report.tex`, `REPORT.md`, `docs/scientific_claims_audit.md` |
| **Task 5** | Surrogate Adequacy & Boundary Discussion | Reconciled global $R^2 = 0.9951$ ($\text{Adj } R^2 = 0.9944$) against the $+0.0217$ RMSE surrogate optimism bias at the single-objective depth-7 candidate $\mathbf{x}^*_{\text{SO}}$ ($0.4483$ predicted vs. $0.4700 \pm 0.0093$ realized, lying $+0.0088$ above the 95% PI upper bound $0.4612$): `45` of `140` runs (`32.14%`) were spent at $\eta = 0.01$ (mean RMSE `0.7545` vs. `0.5055` for $x_1 > -1.0$). | `report.tex`, `REPORT.md`, `scripts/reporting_data.py` |
| **Task 6** | Latency Measurement & Interface Claims | Reconstructed exact `predict` minus `inplace_predict` overheads across all artifacts: `results/runs.csv` ($40.88 \pm 11.56\,\mu\text{s}$), `results/confirmation_runs*.csv` ($38.18 \pm 5.24\,\mu\text{s}$ MO; $36.85 \pm 7.75\,\mu\text{s}$ SO), `results/benchmark.csv` ($32.07 \pm 1.16\,\mu\text{s}$), and `results/revision_v2/full_run_001/latency_interface_overhead.json` ($32.19 \pm 1.46\,\mu\text{s}$). Distinguished development `10 + 30` timing from holdout `5 x (100 + 2,000)` timing. | `scripts/reporting_data.py`, `report.tex`, `REPORT.md` |
| **Task 7** | Baseline & Optimizer Fairness | Documented the protocol differences across `v1.0.0` Split-42 baselines and Revision-v2 20-replicate benchmarks, including Constrained TPE search-to-benchmark feasibility degradation ($20/20 \to 9/20 = 45\%$: depth 6 has $9/9$ feasible at $136.70 \pm 0.82\,\mu\text{s}$, depth 7 has $0/11$ feasible at $150.45 \pm 0.67\,\mu\text{s}$). | `report.tex`, `REPORT.md` |
| **Task 8** | Figure 6 Redesign & Audit | Redesigned `figures/efficiency_comparison_curve.png` in `plots.py` into a two-panel side-by-side figure: Panel (a) Prospective Revision-v2 20-replicate median validation RMSE trajectories ($\pm$ IQR); Panel (b) Historical `v1.0.0` single shuffled FCCD trace vs. Split-42 baselines with the misleading `run_order = 100` divider removed. | `plots.py`, `figures/efficiency_comparison_curve.png`, `report.tex` |
| **Task 9** | Computational Budget & Wall-Clock Accounting | Added complete accounting of all $17,077$ model fits (`14,027` online-timed fits, `610` primary latency refits, `2,440` holdout evaluation rows across `1,900` unique fits), `$1,640,810` timed calls, `$201,270` warmup calls (`1,842,080` total calls), and per-replicate search wall-clock durations (`36.49 ± 0.19 s` Repeated DOE, `26.24 ± 1.80 s` MO-TPE, `26.50 ± 2.70 s` Constrained TPE, `54.03 ± 3.05 s` SO-TPE, `32.61 ± 1.63 s` Random Search). | `tables/tab_computational_budget.tex`, `scripts/generate_report_artifacts.py`, `report.tex`, `REPORT.md` |
| **Task 10** | Multi-Objective & Pareto Claims | Implemented deterministic bootstrap ($B = 2,000$ resamples over the 20 holdout evaluation seeds `2001..2020`, `seed = 20261011`) for the pooled 12-point Pareto frontier: HV at $[0.60, 250.0]$ is $17.6923 \pm 0.0564$ (95% CI $[17.5848, 17.8061]$; point estimate $17.6714$); at $[0.65, 275.0]$ is $29.0170 \pm 0.0688$ (95% CI $[28.8861, 29.1579]$; point estimate $28.9900$); joint seed+session HV is $17.6890 \pm 0.0727$ ($[17.5507, 17.8420]$); optimizer frontier presence is $100.0\%$ (DOE-MO, MO-TPE, Constrained TPE) and $91.5\%$ (SO-TPE). | `scripts/reporting_data.py`, `tables/tab_hypervolume_comparison.tex`, `report.tex`, `REPORT.md` |
| **Task 11** | Scope & Generalizability | Qualified all conclusions to California Housing ($N = 20,640, d = 8$), `n_estimators = 100`, $k = 4$ tuned hyperparameters, and single-workstation timing. Framed DOE/RSM as complementary to (not a universal replacement for) Bayesian optimization. | `report.tex`, `REPORT.md` |
| **Task 12** | Manuscript Title, Abstract, & Narrative | Retitled both `report.tex` and `REPORT.md` to *"Pre-Planned Response Surface Methodology and Face-Centered Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking"*, and rewrote the abstract, methods, results, discussion, and conclusion to reflect all audit findings. | `report.tex`, `REPORT.md`, `report.pdf` |
| **Task 13** | Proposed Follow-Up Experiments | Designed five pre-specified protocols clearly marked as **Proposed Future Work — Not Executed in Current Repository**: Protocol A (Constrained DOE/RSM at $145\,\mu\text{s}$), Protocol B (Matched-Protocol DOE vs. TPE Parity Benchmark), Protocol C (Truly Independent External Confirmation), Protocol D (Two-Stage Sequential / Region-Zooming RSM with exact 140-fit budget arithmetic), and Protocol E (Multi-Dataset, Multi-Horizon, and 9-Factor DSD Extensions). | `docs/proposed_followup_experiments.md` |

---

## 3. Summary of Key Reconstructed Scientific Metrics

### 3.1 Curvature Test Sensitivity Across Error Denominators (`results/runs.csv`, $\text{SS}_{\text{Curv}} = 0.288786$)

| Error Term / Model Specification | Denominator $df$ | Error SS | Denominator MS | $F$-Statistic | $p$-Value | Scientific Interpretation |
| :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| Center-point pure error across 5 blocks | $15$ | $0.000041$ | $2.74 \times 10^{-6}$ | $105,432.31$ | $< 10^{-15}$ | Within-block subsampling seed variance at fixed split $S_b$ |
| Block-adjusted pooled coordinate pure error | $79$ | $0.001127$ | $1.43 \times 10^{-5}$ | $20,235.41$ | $< 10^{-15}$ | Center PE ($15$ df) + Phase 1 Treatment $\times$ Block ($64$ df) |
| Phase 1 Treatment $\times$ Block interaction | $64$ | $0.001086$ | $1.70 \times 10^{-5}$ | $17,013.24$ | $< 10^{-15}$ | Coordinate-by-seed-block interaction floor across $17 \times 5$ cells |
| Paired 5-block corner-mean vs. center contrast | $4$ | — | $\text{SE} = 0.0016$ | $7,342.69$ ($t = 85.69$) | $1.11 \times 10^{-7}$ | Block-level paired contrast ($\hat{\Delta}_{\text{curv}} = +0.1343$, 95% CI $[+0.1300, +0.1387]$) |
| Unadjusted pooled coordinate pure error | $83$ | $0.005790$ | $6.98 \times 10^{-5}$ | $4,140.06$ | $< 10^{-15}$ | Pooled within-coordinate variance without block adjustment |
| Block-adjusted first-order + 2FI + curvature residual | $84$ | $0.007461$ | $8.88 \times 10^{-5}$ | $3,251.19$ | $< 10^{-15}$ | Residual MS after fitting main effects, 2FIs, curvature indicator, and blocks |
| Block-adjusted first-order + 2FI residual | $85$ | $0.296247$ | $0.003485$ | $82.86$ | $3.34 \times 10^{-14}$ | Conservative lower bound absorbing omitted pure quadratic effects into residual |

### 3.2 Holdout Pooled Pareto Frontier Bootstrap (`results/revision_v2/full_run_001/final_evaluations.csv`)

* **Resampling Protocol**: $B = 2,000$ non-parametric cluster bootstrap replicates (`seed = 20261011`) resampling the 20 independent holdout evaluation seeds (`2001..2020`) with replacement across all 122 finalized selection records (95 unique configurations).
* **Primary Reference Point $[0.60, 250.0]$**: Point estimate $= 17.6714$; Bootstrap Mean $\pm$ SD $= 17.6923 \pm 0.0564$; 95% percentile CI $[17.5848, 17.8061]$.
* **Secondary Reference Point $[0.65, 275.0]$**: Point estimate $= 28.9900$; Bootstrap Mean $\pm$ SD $= 29.0170 \pm 0.0688$; 95% percentile CI $[28.8861, 29.1579]$.
* **Joint Seed + Session Resampling ($[0.60, 250.0]$)**: Bootstrap Mean $\pm$ SD $= 17.6890 \pm 0.0727$; 95% percentile CI $[17.5507, 17.8420]$.
* **Frontier Count & Optimizer Presence**: $13.42 \pm 1.46$ non-dominated configurations per resample (12 distinct configurations on the point-estimate frontier); optimizer presence on the frontier is $100.0\%$ for Repeated DOE (MO), $100.0\%$ for MO-TPE, $100.0\%$ for Constrained TPE, $91.5\%$ for SO-TPE, $3.9\%$ for Repeated DOE (SO), and $3.8\%$ for Random Search.

---

## 4. Verification & Evidence Protection Summary

All automated verification gates were executed and confirmed to pass with zero errors:

1. `python scripts/generate_report_artifacts.py --check` — **PASSED** (`results/macros.tex` and all 12 LaTeX tables in `tables/*.tex` match canonical reconstructions byte-for-byte).
2. `python scripts/generate_research_reporting.py --check` — **PASSED** (`REPORT.md`, `docs/generated/scientific_results_manifest.json`, `docs/scientific_claims_audit.md`, `docs/f_scientific_review_validation.md`, and `docs/proposed_followup_experiments.md` match canonical reconstructions byte-for-byte).
3. `python scripts/audit_scientific_consistency.py` — **PASSED** (All 40 frozen experimental evidence files match their trusted SHA-256 baseline hashes in `CANONICAL_EVIDENCE_SHA256` and `results/revision_v2/full_run_001/provenance.json`; all prose numerical tokens in `report.tex` and `REPORT.md` are verified against canonical macros/sources).
4. `python -m pytest -q` — **PASSED** (Full test suite including Work Package F and F.1 scientific audit and mutation assertions).
5. `tectonic report.tex` — **PASSED** (`report.pdf` compiled cleanly with zero undefined macros or references).
6. `git diff --exit-code results/ config.yaml` — **PASSED** (Zero modifications to frozen experimental evidence).

**READY FOR INDEPENDENT SCIENTIFIC REVIEW**
