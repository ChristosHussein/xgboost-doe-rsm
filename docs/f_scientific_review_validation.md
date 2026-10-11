# Work Package F — Independent Scientific and Methodological Audit Validation Report

**Repository:** `ChristosHussein/xgboost-doe-rsm`  
**Branch:** `codex/scientific-revision`  
**Baseline Commit:** `6b53dac85af261553a7870f4888d5ea7423d4b5e`  
**Canonical Campaign:** `results/revision_v2/full_run_001/` & Historical `v1.0.0` (`results/*.json`, `results/*.csv`)

---

## 1. Executive Summary

Work Package F audited the scientific validity, methodological terminology, statistical assumptions, experimental comparisons, and scope conclusions of the study ontop of the single-source-of-truth reporting architecture established in Work Package D.2.

Crucially, **zero new XGBoost model fits were executed** and **all 40 frozen experimental evidence files (`results/` and `config.yaml`) remain byte-identical** to the immutable SHA-256 baseline (`docs/evidence_baseline_hashes.json`). Every new sensitivity metric, chronology reconstruction, latency overhead summary, computational budget breakdown, and holdout Pareto bootstrap interval was computed deterministically from the existing frozen artifacts in `scripts/reporting_data.py` and propagated automatically to `results/macros.tex`, `tables/*.tex`, `REPORT.md`, `report.tex`, `report.pdf`, and `docs/generated/scientific_results_manifest.json`.

---

## 2. Task-by-Task Audit Resolution Matrix

| Task | Audit Scope | Key Scientific Finding & Resolution | Primary Artifacts Updated |
| :--- | :--- | :--- | :--- |
| **Task 1** | Scientific Claims Register | Inventoried 22 material claims across abstract, introduction, methods, results, discussion, conclusion, tables, and captions (`9 SUPPORTED`, `10 NEEDS QUALIFICATION`, `3 UNSUPPORTED` prior to correction; all 22 resolved). | `docs/scientific_claims_audit.md` |
| **Task 2** | Experimental Design Terminology & Budget | Proved from `design.py` and `results/runs.csv` (`30` of `40` axial runs occurred at `run_order <= 100`) that the 140-run FCCD was generated upfront as one pre-specified randomized batch. Removed misleading "Sequential RSM" from titles/prose and contrasted `25` distinct coordinates ($\times 5$ blocks + $15$ confirmation) against `140` unique adaptive TPE coordinates. | `report.tex`, `REPORT.md`, `scripts/reporting_data.py` |
| **Task 3** | Headline Curvature Test Sensitivity | Reconstructed $F(1, 15) = 105{,}432.31$ (`MS_curv = 2.6602`, `MS_PE,center = 2.523e-5` from identical booster topology across seeds) and added sensitivity analyses across 6 alternative error denominators ($F = 20{,}235.41$ on 79-df block-adjusted pooled coordinate error; $F = 7{,}342.69, t = 85.69, p = 1.12 \times 10^{-7}$ on 4-df paired block contrast; $F = 82.86, p = 3.34 \times 10^{-14}$ on 85-df unadjusted 2FI residual). | `scripts/reporting_data.py`, `report.tex`, `REPORT.md` |
| **Task 4** | Statistical Assumptions & Inference | Explicitly documented grouping factors for Levene's test (5 seed blocks, $W = 0.0085, p = 0.9999$) vs. Brown–Forsythe (25 coordinates, $W = 0.2584, p = 0.9999$), outer-split pairing ($df = 19$) for paired $t$-tests, independent units ($n = 20$ vs. $n = 20$) for confirmation Welch tests, and replaced all equivalence phrasing with failure-to-reject / non-detection language. | `report.tex`, `REPORT.md`, `docs/scientific_claims_audit.md` |
| **Task 5** | Surrogate Adequacy & Boundary Discussion | Reconciled global $R^2_{\text{pred}} = 0.9035$ against the $+0.0217$ RMSE over-prediction at the selected optimum ($0.5067$ predicted vs. $0.4850 \pm 0.0044$ realized): $45$ of $140$ runs ($32.14\%$) were spent at $\eta = 0.01$ (mean RMSE $0.7545$ vs. $0.5055$ at $\eta \in \{0.105, 0.20\}$). Discussed box-boundary truncation at `max_depth = 3`, `learning_rate = 0.20`, `subsample = 1.0`. | `report.tex`, `REPORT.md`, `scripts/reporting_data.py` |
| **Task 6** | Latency Measurement & Interface Claims | Reconstructed exact `predict` minus `inplace_predict` overheads across all artifacts: `runs.csv` ($40.88 \pm 11.56\,\mu\text{s}$), `confirmation_runs*.csv` ($38.18 \pm 5.24\,\mu\text{s}$ MO; $36.85 \pm 7.75\,\mu\text{s}$ SO), `benchmark.csv` ($32.07 \pm 1.16\,\mu\text{s}$), and `latency_interface_overhead.json` ($32.19 \pm 1.46\,\mu\text{s}$). Distinguished development `10 + 30` timing from holdout `5 x (100 + 2,000)` timing. | `scripts/reporting_data.py`, `report.tex`, `REPORT.md` |
| **Task 7** | Baseline & Optimizer Fairness | Documented the structural shift from `v1.0.0` Split-42 TPE (penalized online `inplace_predict` against $145\,\mu\text{s}$) to Revision-v2 Constrained TPE (penalized online `predict` against $145\,\mu\text{s}$, selecting shallower trees at $99.64 \pm 9.10\,\mu\text{s}$ with $100\%$ feasibility). | `report.tex`, `REPORT.md` |
| **Task 8** | Figure 3 Redesign & Audit | Redesigned `figures/efficiency_comparison_curve.png` in `plots.py` into a two-panel side-by-side figure: Panel (a) Prospective Revision-v2 20-replicate mean validation RMSE trajectories ($\pm 1\text{ SD}$); Panel (b) Historical `v1.0.0` single shuffled FCCD trace vs. Split-42 baselines with the misleading `run_order = 100` divider removed. | `plots.py`, `figures/efficiency_comparison_curve.png`, `report.tex` |
| **Task 9** | Computational Budget & Wall-Clock Accounting | Added complete accounting of all $17{,}077$ model fits (`14,027` online-timed fits, `610` primary latency refits, `2,440` holdout evaluation rows across `1,900` unique fits), `1,640,810` timed calls, `201,270` warmup calls (`1,842,080` total calls), and per-replicate search wall-clock durations (`36.49 ± 0.19s` Repeated DOE, `26.24 ± 1.80s` MO-TPE, `26.50 ± 2.70s` Constrained TPE, `54.03 ± 3.05s` SO-TPE, `32.61 ± 1.63s` Random Search). | `tables/tab_computational_budget.tex`, `scripts/generate_report_artifacts.py`, `report.tex`, `REPORT.md` |
| **Task 10** | Multi-Objective & Pareto Claims | Implemented deterministic bootstrap ($B = 2{,}000$ resamples over the 20 holdout evaluation seeds `2001..2020`, `seed = 20261011`) for the pooled 12-point Pareto frontier: HV at $[0.60, 250.0]$ is $17.6923 \pm 0.0564$ (95% CI $[17.5848, 17.8061]$); at $[0.65, 275.0]$ is $29.0170 \pm 0.0688$ (95% CI $[28.8861, 29.1579]$); joint seed+session HV is $17.7220 \pm 0.0929$ ($[17.5512, 17.8906]$); optimizer frontier presence is $100.0\%$ (DOE-MO, MO-TPE, Constrained TPE) and $91.5\%$ (SO-TPE). | `scripts/reporting_data.py`, `tables/tab_hypervolume_comparison.tex`, `report.tex`, `REPORT.md` |
| **Task 11** | Scope & Generalizability | Qualified all conclusions to California Housing ($N = 20{,}640, d = 8$), `n_estimators = 100`, $k = 4$ tuned hyperparameters, and single-workstation timing. Framed DOE/RSM as complementary to (not a universal replacement for) Bayesian optimization. | `report.tex`, `REPORT.md` |
| **Task 12** | Manuscript Title, Abstract, & Narrative | Retitled both `report.tex` and `REPORT.md` to *"Pre-Planned Response Surface Methodology and Face-Centered Central Composite Design for Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under Stochastic Nuisance Blocking"*, and rewrote the abstract, methods, results, discussion, and conclusion to reflect all audit findings. | `report.tex`, `REPORT.md`, `report.pdf` |
| **Task 13** | Proposed Follow-Up Experiments | Designed three pre-specified protocols clearly marked as **Proposed Future Work — Not Executed in Current Repository**: Proposal A (Two-Stage Sequential RSM), Proposal B (Multi-Dataset Generalization Benchmark), and Proposal C (9-Factor Definitive Screening + Local RSM). | `docs/proposed_followup_experiments.md` |

---

## 3. Summary of Key Reconstructed Scientific Metrics

### 3.1 Curvature Test Sensitivity Across Error Denominators (`results/runs.csv`)

| Error Term / Model Specification | Denominator $df$ | Denominator MS | $F$-Statistic | $p$-Value | Scientific Interpretation |
| :--- | ---: | ---: | ---: | :--- | :--- |
| Center-point pure error across 5 blocks | $15$ | $2.523 \times 10^{-5}$ | $105{,}432.31$ | $2.83 \times 10^{-35}$ | Replicate variance at fixed center coordinate |
| Block-adjusted pooled coordinate pure error | $79$ | $1.315 \times 10^{-4}$ | $20{,}235.41$ | $3.54 \times 10^{-102}$ | Replicate variance across all 17 Phase 1 coordinates after removing block main effects |
| Phase 1 Treatment $\times$ Block interaction | $64$ | $1.564 \times 10^{-4}$ | $17{,}013.24$ | $7.49 \times 10^{-82}$ | Coordinate-by-seed-block interaction floor |
| Paired 5-block corner-mean vs. center contrast | $4$ | — ($t = 85.69$) | $7{,}342.69$ | $1.12 \times 10^{-7}$ | Exact block-level paired contrast ($\hat{\Delta}_{\text{curv}} = +0.1344 \pm 0.0035$, 95% CI $[+0.1300, +0.1387]$) |
| Unadjusted pooled coordinate pure error | $83$ | $6.425 \times 10^{-4}$ | $4{,}140.06$ | $9.17 \times 10^{-74}$ | Pooled within-coordinate variance without block adjustment |
| Block-adjusted first-order + 2FI + curvature residual | $84$ | $8.182 \times 10^{-4}$ | $3{,}251.19$ | $1.60 \times 10^{-70}$ | Residual MS after fitting main effects, 2FIs, curvature indicator, and blocks |
| Block-adjusted first-order + 2FI residual | $85$ | $3.210 \times 10^{-2}$ | $82.86$ | $3.34 \times 10^{-14}$ | Conservative lower bound absorbing omitted pure quadratic effects (`learning_rate^2`) into residual |

### 3.2 Holdout Pooled Pareto Frontier Bootstrap (`results/revision_v2/full_run_001/final_evaluations.csv`)

* **Resampling Protocol**: $B = 2{,}000$ non-parametric bootstrap replicates (`seed = 20261011`) resampling the 20 independent holdout evaluation seeds (`2001..2020`) with replacement across all 122 finalized selection records (95 unique configurations).
* **Primary Reference Point $[0.60, 250.0]$**: Point estimate $= 17.6714$; Bootstrap Mean $\pm$ SD $= 17.6923 \pm 0.0564$; 95% percentile CI $[17.5848, 17.8061]$.
* **Secondary Reference Point $[0.65, 275.0]$**: Point estimate $= 28.9900$; Bootstrap Mean $\pm$ SD $= 29.0170 \pm 0.0688$; 95% percentile CI $[28.8861, 29.1579]$.
* **Joint Seed + Session Resampling ($[0.60, 250.0]$)**: Bootstrap Mean $\pm$ SD $= 17.7220 \pm 0.0929$; 95% percentile CI $[17.5512, 17.8906]$.
* **Frontier Count & Optimizer Presence**: $13.42 \pm 1.46$ non-dominated configurations per resample (12 distinct configurations on the point-estimate frontier); optimizer presence on the frontier is $100.0\%$ for Repeated DOE (MO), $100.0\%$ for MO-TPE, $100.0\%$ for Constrained TPE, $91.5\%$ for SO-TPE, and $0.0\%$ for Random Search and Repeated DOE (SO).

---

## 4. Verification & Evidence Protection Summary

All automated verification gates were executed and confirmed to pass with zero errors:

1. `python scripts/generate_report_artifacts.py --check` — **PASSED** (`results/macros.tex` and all 8 LaTeX tables in `tables/*.tex` match canonical reconstructions byte-for-byte).
2. `python scripts/generate_research_reporting.py --check` — **PASSED** (`REPORT.md` and `docs/generated/scientific_results_manifest.json` match canonical reconstructions byte-for-byte).
3. `python scripts/audit_scientific_consistency.py` — **PASSED** (All 40 frozen experimental evidence files match their trusted SHA-256 baseline hashes; all prose numerical tokens in `report.tex` and `REPORT.md` are verified against canonical macros/sources).
4. `python -m pytest -q` — **PASSED** (Full test suite including Work Package F scientific audit assertions).
5. `tectonic report.tex` — **PASSED** (`report.pdf` compiled cleanly with zero undefined macros or references).
6. `git diff --exit-code results/phase1.json results/phase3.json results/confirmation.json results/runs.csv results/revision_v2/full_run_001/ config.yaml` — **PASSED** (Zero modifications to frozen experimental evidence).

**READY FOR INDEPENDENT SCIENTIFIC REVIEW**
