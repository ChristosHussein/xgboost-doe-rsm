# Work Package F / F.1 — Scientific Claims Register and Methodological Audit

**Repository:** `ChristosHussein/xgboost-doe-rsm`  
**Branch:** `codex/scientific-revision`  
**Starting Commit:** `6b53dac85af261553a7870f4888d5ea7423d4b5e` (Work Package F) / `6050816e0357a98d77a37b2f58fcf7da2a0cf526` (Work Package F.1)  
**Canonical Evidence Lock:** `CANONICAL_EVIDENCE_SHA256` and `verify_canonical_evidence_hashes()` in `scripts/reporting_data.py` plus `results/revision_v2/full_run_001/provenance.json` (40 frozen artifacts verified by SHA-256)

---

## 1. Purpose and Audit Scope

Work Package D.2 established a single-source-of-truth reporting pipeline (`scripts/reporting_data.py`, `scripts/generate_report_artifacts.py`, and `scripts/generate_research_reporting.py`) ensuring that every numerical value in `report.tex`, `REPORT.md`, `results/macros.tex`, `tables/*.tex`, and `docs/generated/scientific_results_manifest.json` is deterministically reconstructed from raw experimental artifacts.

However, **numerical consistency with a saved JSON or CSV artifact is not equivalent to scientific validity**. A claim can reproduce a stored floating-point number to six decimal places while:
1. Mischaracterizing the experimental design chronology (e.g., calling a pre-planned 140-run FCCD "sequential");
2. Using an overly narrow error denominator without sensitivity analysis (e.g., $F_{\text{Curvature}} = 105,432.31$ against within-block center subsampling error);
3. Conflating grouping factors in residual diagnostics (e.g., Levene across 5 blocks vs. Brown-Forsythe across 25 coordinates vs. Breusch-Pagan across continuous $\hat{y}$);
4. Comparing multi-objective hypervolumes computed under incompatible candidate pools or latency timing protocols;
5. Treating replicate indices as paired experimental blocks when they represent independent RNG streams; or
6. Presenting a deterministic 12-point holdout non-dominated set without seed-sampling uncertainty.

This register inventories all **22 material scientific claims** across the Title, Abstract, Introduction, Methods, Phase 1–5 Results, Prospective Benchmark Tables/Figures, Discussion, Code Availability, and Conclusion, classifies each claim prior to Work Package F (`SUPPORTED`, `NEEDS QUALIFICATION`, `UNSUPPORTED`, or `UNVERIFIED`), and documents the exact methodological correction enforced in `report.tex` and `REPORT.md`. Furthermore, in Work Package F.1 this document itself is deterministically generated and verified from `ReportingDataset` via `scripts/generate_research_reporting.py` so that audit documentation can never drift from canonical source reconstructions.

---

## 2. Summary Classification Matrix

| Claim ID | Manuscript Location | Topic / Claim Summary | Pre-Audit Status | Post-Audit Disposition |
| :---: | :--- | :--- | :---: | :--- |
| **C-01** | Main Abstract and Secs. 1 & 3; Supplement Secs. 1--2 | Characterization of the 140-run FCCD campaign as "Sequential RSM" | **UNSUPPORTED** | Corrected to **Pre-Planned Blocked RSM & FCCD**; actual shuffled execution chronology documented (`30/40` axial runs before `run_order = 100`). |
| **C-02** | Main Secs. 3 & 7; Supplement Sec. 2 and Table 1; `REPORT.md` Sec. 1.3 | 140-evaluation budget equivalence between Blocked DOE (25 coords $\times$ 5 blocks) and TPE/Random Search (140 distinct coords) | **NEEDS QUALIFICATION** | Explicitly contrasted spatial exploration (25 vs. 140 distinct coordinates) and low-$\eta$ corner allocation (`45/140 = 32.14%` runs at $\eta = 0.01$). |
| **C-03** | Main Sec. 3; Supplement Sec. 2; `REPORT.md` Sec. 1.3 | Single-sample CPU inference latency protocol and `predict` vs. `inplace_predict` interface overhead | **NEEDS QUALIFICATION** | Documented all 3 timing protocols (1,000-call historical, 30-call online search, $2\times 500$-call warmed holdout) and quantified wrapper overhead (`32.19 ± 1.46 µs` in Revision-v2; `40.88 ± 11.56 µs` in historical DOE). |
| **C-04** | Main Sec. 4; Supplement Sec. 3 and Table 3; `REPORT.md` Sec. 2.1 | Phase 1 $2^4$ factorial screening dominance of $\ln(\eta)$ ($F = 357.40$), depth ($F = 23.57$), and $x_1 x_2$ ($F = 26.32$) | **SUPPORTED** | Verified via Type III OLS ANOVA on `results/runs.csv` ($N_1 = 100$ runs across 5 blocks, $\text{MS}_E = 0.003485$). |
| **C-05** | Main Sec. 4; Supplement Sec. 3; `REPORT.md` Secs. 2.2--2.3 | Phase 1 curvature test $F_{\text{Curvature}} = 105,432.31$ ($p < 10^{-15}$) | **NEEDS QUALIFICATION** | Explained micro-scale subsampling denominator ($\text{df}=15$, $\text{SS}_{\text{Curv}} = 0.288786$, $\text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$) and added 6-denominator sensitivity analysis ($F = 20,235.41$ pooled; $F = 17,013.24$ vs. $T\times B$; $F = 7,342.69, p = 1.11 \times 10^{-7}$ 5-block paired; $F = 82.86, p = 3.34 \times 10^{-14}$ vs. unadjusted 2FI residual). Distinguished non-planarity from quadratic adequacy. |
| **C-06** | Main Sec. 3; Supplement Secs. 2 & 4; `REPORT.md` Sec. 3.2 | Variance shielding and Block Intraclass Correlation $\text{ICC} = 0.4069$ ($40.69\%$, REML $0.4069$, $\sigma_{\text{block}} \approx 0.0077$ RMSE) | **SUPPORTED** | Reconstructed from `results/runs.csv` Phase 2 ANOVA ($\text{SS}_{\text{block}} = 0.006966, \text{MS}_{\text{block}} = 0.001742, \text{MS}_E = 0.000086, F = 20.21, p = 9.11 \times 10^{-13}$) and REML mixed model. |
| **C-07** | Main Sec. 4; Supplement Sec. 4 and Table 4; `REPORT.md` Sec. 3.1 | Phase 2 second-order FCCD model fit ($R^2 = 0.9951$, $\text{Adj } R^2 = 0.9944$) and HC3 robust standard errors | **NEEDS QUALIFICATION** | Qualified high $R^2$ as reflecting macro-scale contrast across $\eta \in [0.01, 0.30]$ while local lack of fit remains significant; noted HC3 adjusts SEs for heteroscedasticity but does not remove polynomial bias. |
| **C-08** | Main Sec. 4; Supplement Sec. 4 and Table 6; `REPORT.md` Sec. 3.3 | Multi-scale lack-of-fit decomposition ($121\text{ df} = 10\text{ df LoF} + 96\text{ df } T\times B + 15\text{ df Center PE}$) and restricted-domain refit ($x_1 \ge -0.5$) | **SUPPORTED** | Verified exact SS/DF/MS decomposition ($\text{SS}_{\text{LoF}} = 0.008850, F = 62.21$ vs. additive; $F = 323.10$ vs. center PE) and documented that dropping $x_1 = -1$ aliases $x_1^2 \equiv x_1$, reducing $\text{df}_{\text{LoF}}$ from 10 to 2 ($96.3\%$ SS reduction). |
| **C-09** | Main Sec. 4; Supplement Sec. 4, Tables 5 & 7, and latency figure; `REPORT.md` Sec. 3.1 | Second-order and quadratic-depth inference latency models ($R^2 = 0.8820$ and $0.8589$, $\text{AIC} = 1209.6$ and $1202.6$) and non-significant latency lack of fit ($F = 1.20, p = 0.3625$) | **SUPPORTED** | Verified from `results/runs.csv` and `results/latency_models_comparison.csv` with exact Jacobian adjustment ($+2\sum\ln y_i = 1,450.8$). |
| **C-10** | Main Sec. 4; Supplement Sec. 5 and diagnostics figure; `REPORT.md` Sec. 3.3 | Residual diagnostics: Shapiro-Wilk ($W = 0.9971, p = 0.9947$), Levene ($p = 0.9507$), Brown-Forsythe ($p = 0.8552$), Breusch-Pagan ($\text{LM} = 70.10, p < 0.0001$), Durbin-Watson ($DW = 1.9918$) | **NEEDS QUALIFICATION** | Explicitly specified grouping factors: mean-centered Levene across **5 seed blocks** ($W=0.1754, p=0.9507$), median-centered Brown-Forsythe across **25 design coordinates** ($W=0.6872, p=0.8552$), and reconciled both with significant Breusch-Pagan across continuous $\hat{y}$ due to cell-mean LoF at $\eta = 0.01$. |
| **C-11** | Main Sec. 5; Supplement Sec. 6, Table 8, and RMSE-surface figure; `REPORT.md` Sec. 4 | Canonical spectral decomposition ($\lambda_1 = 0.000057$, 95% wild bootstrap CI $[-0.0039, +0.0022]$, $68.2\%$ $\le 0$) and stationary/rising ridge classification | **SUPPORTED** | Verified eigenvalues, eigenvectors, unconstrained stationary point $\mathbf{x}_0 = [+0.4171, +1.3524, +15.6944, -6.8009]^T$ ($\|\mathbf{x}_0\|_2 = 17.16$), and constrained depth-7 candidate $\mathbf{x}^*_{\text{SO}}$ ($\hat{y} = 0.4483$). |
| **C-12** | Main Sec. 5; Supplement Sec. 7, Table 9, and desirability figure; `REPORT.md` Sec. 5 | Multi-objective Derringer-Suich desirability optimization selecting interior $\mathbf{x}^*_{\text{MO}}$ ($\eta = 0.2324, d = 4, s = 1.0000, \lambda = 0.8295, D = 0.6782$) | **SUPPORTED** | Verified against `results/confirmation.json` ($\hat{Y}_1 = 0.4804, \hat{Y}_2 = 138.11\,\mu\text{s}, d_1 = 0.8783, d_2 = 0.5236$) and `results/desirability_sensitivity.csv`. |
| **C-13** | Main Sec. 6 and confirmation table; Supplement Sec. 8; `REPORT.md` Sec. 6 | Phase 5 Satterthwaite confirmation intervals ($m=10$ seeds `505..1414`): pointwise confirmation at $\mathbf{x}^*_{\text{MO}}$ ($0.4853 \in [0.4677, 0.4932]$) and non-confirmation / polynomial optimism at $\mathbf{x}^*_{\text{SO}}$ ($0.4700 > 0.4612$, bias $+0.0217$) | **SUPPORTED** | Verified exact variance-component synthesis $\hat{V}_{\text{conf}} = c_1 \text{MS}_E + c_2 \text{MS}_{\text{block}}$, effective degrees of freedom ($\nu_{\text{eff}} = 14.0$ and $15.1$), and qualified pointwise confirmation vs. global surrogate accuracy. |
| **C-14** | Main Sec. 6 summary; Supplement Sec. 8 and Tables 10--11; `REPORT.md` Sec. 6.1 | Three- and four-decimal-place numerical similarities (`0.4686`/`0.4687`/`0.4690`/`0.4691`, `0.4853`, `0.4884`/`0.4888`, `0.4700`) across confirmation, historical benchmark, and Revision-v2 tables | **NEEDS QUALIFICATION** | Documented exact provenance: (1) shared configuration hashes (`0079fbae0ec5b7cb` and `b05e8e5cb67e8668`) on seeds `2001..2020`, plus deterministic grid selection (`9d7c4e74ff156870`, `grid_index = 1052`); (2) disjoint seed sets (`505..1414` vs. `2001..2020`) differing at the 5th–6th decimal place (`0.469124` vs. `0.469033`, $\Delta = 91.25\times 10^{-6}$). |
| **C-15** | Main Secs. 3, 6 & 8; Supplement Secs. 1 & 8; `REPORT.md` Sec. 7 | Holdout test set isolation history between historical `v1.0.0` and prospective `Revision-v2` | **SUPPORTED** | Explicitly stated that `v1.0.0` logged test RMSE during development, whereas `Revision-v2` enforced programmatic isolation until `finalized_selections.json` was frozen, without undoing prior historical exposure of the California Housing dataset. |
| **C-16** | Main Sec. 6; Supplement Sec. 8 and Table 11; `REPORT.md` Sec. 7.3 | What the 20 Repeated DOE replicates represent (block seeds `3001..3100` + deterministic 1,323-point grid selection) and why DOE-SO has between-search $\text{SD} = 0.00000$ vs. retraining $\sigma_{\text{eval}} = 0.00304$ | **NEEDS QUALIFICATION** | Explicitly documented the 1,323-point grid ($21 \times 7 \times 3 \times 3$ in `doe_selection_protocol.json`), why all 20 DOE-SO replicates select `grid_index = 1052` ($\eta=0.1519, d=7, s=1.0, \lambda=10.0$), and how between-search selection SD is separated from 20-seed retraining SD. |
| **C-17** | Main Abstract and Sec. 7; Supplement Sec. 8 and Table 11; `REPORT.md` Sec. 7.3 | Multi-objective holdout comparison: Repeated DOE MO ($0.48997 \pm 0.00732$) vs. MO-TPE ($0.49255 \pm 0.01080$), Welch $t = -0.8849, p = 0.3826$, paired $t = -0.966, p = 0.3463$ | **NEEDS QUALIFICATION** | Designated **unpaired Welch's two-sample $t$-test as the primary inferential test** (since replicate index $r$ pairs DOE seeds `3001+5r..3005+5r` with Optuna seed `1000+r` on split 42 rather than a shared random block) and paired $t$-test as a secondary sensitivity check; noted no pre-registered equivalence margin (`rmse_equivalence_margin: null`). |
| **C-18** | Main Sec. 7; Supplement Sec. 8 and Table 12; `REPORT.md` Sec. 7.2 | Development candidate hypervolume comparison (MO-TPE $18.3834 \pm 0.3301$ vs. Repeated DOE 5-block means $15.6128 \pm 0.7268$ vs. Fixed 27-point DOE front $16.8585$, diff $-1.5249, p < 0.0001$) and historical `v1.0.0` hypervolume ($18.38$ vs. $16.86$) | **NEEDS QUALIFICATION** | Distinguished all 4 protocol rows; qualified the Split-42 difference as a **descriptive 1-sample test** of 20 stochastic MO-TPE fronts against a single fixed 27-point DOE front on split 42; explained that historical `v1.0.0` compared a 25-point DOE front ($18.38$) against 1 MO-TPE point ($16.86$), whereas single-point comparison is $16.92$ vs. $16.86$ ($+0.3\%$). |
| **C-19** | Main Abstract and Sec. 7 with Pareto figure; Supplement Sec. 8 and Table 12; `REPORT.md` Secs. 7.2--7.3 | Holdout non-dominated set of 12 configurations ($\text{HV} = 17.6714$ at $[0.60, 250.0]$; $28.9900$ at $[0.65, 275.0]$): 3 DOE-MO, 3 MO-TPE, 5 Constrained TPE, 1 SO-TPE | **NEEDS QUALIFICATION** | Added paired cluster bootstrap ($B=2,000$ resamples over the 20 shared evaluation seeds `2001..2020`, `seed=20261011`): $\text{HV} = 17.6923 \pm 0.0564$ (95% CI $[17.5848, 17.8061]$) at $[0.60, 250.0]$, $29.0170 \pm 0.0688$ (95% CI $[28.8861, 29.1579]$) at $[0.65, 275.0]$, joint seed+session $\text{HV} = 17.6890 \pm 0.0727$ ($[17.5507, 17.8420]$), and per-optimizer frontier presence ($100.0\%$ DOE-MO, MO-TPE, Constrained TPE; $91.5\%$ SO-TPE). |
| **C-20** | Main Secs. 7 & 8; Supplement Sec. 8 and Table 11; `REPORT.md` Sec. 7.3 | Constrained TPE feasibility degradation from $20/20$ ($100\%$) during online 30-call search to $9/20$ ($45\%$) under 1,000-call benchmark verification (depth 6: $9/9$ feasible at $136.70 \pm 0.82\,\mu\text{s}$; depth 7: $0/11$ feasible at $150.45 \pm 0.67\,\mu\text{s}$) | **SUPPORTED** | Verified from `finalized_selections.json` and `final_summary.csv`; framed explanations (30-call noise, warmup protocol difference, winner's-curse selection near $145\,\mu\text{s}$) as plausible mechanisms. |
| **C-21** | Main Sec. 7; Supplement computational-budget table; `REPORT.md` Sec. 7.4 | Complete computational budget ($17,077$ model fits, $1,640,810$ timed single-row calls, $201,270$ warmup calls, $1,842,080$ total calls) and per-method search wall-clock times | **NEEDS QUALIFICATION** | Added Table 13 (`tables/tab_computational_budget.tex`) and Section 7.4 reconciling every fit, timed call, warmup call, mean fit duration (`174.2 ms` MO-TPE to `370.2 ms` SO-TPE; `238.6 ms` DOE), and replicate wall-clock time (`36.49 ± 0.19 s` DOE vs. `54.03 ± 3.05 s` SO-TPE). |
| **C-22** | Main Secs. 3 & 10; Supplement latency and code-availability sections; `REPORT.md` Secs. 7.3 & 9 | Historical vs. prospective latency differences and reproducibility provenance | **SUPPORTED** | Verified that both historical and prospective scripts applied Win32 CPU Core 0 affinity pinning (attributing differences to between-session drift, $\text{SD} = 1.28\,\mu\text{s}$), cited specific Montgomery chapters (4, 5, 6, 7, 10, 11, 13), recorded immutable commit SHAs (`cd63135d...`, `421cf242...`, `6b53dac8...`), retained detailed version hashes in repository provenance records, and kept the reader manuscript focused on scientifically relevant timing limitations. |

---

## 3. Detailed Claim-by-Claim Scientific Audit

### Claim C-01 — Study Chronology: Pre-Planned vs. Sequential DOE
- **Location**: Main Abstract and Sections 1 and 3; Supplement Sections 1 and 2; `REPORT.md` Executive Summary and Section 1.3.
- **Scientific Question**: Was the 140-run Face-Centered Central Composite Design executed as a two-stage adaptive sequence (running 100 factorial+center trials, testing curvature, and only then scheduling 40 axial trials), or as a single pre-planned 140-run design?
- **Experimental Evidence & Source Artifact**: Inspection of `pipeline.py` and `run_pipeline.py` (`generate_design()` and `run_experiments()`) and `results/runs.csv` (`phase` vs. `run_order` columns) shows that all 28 runs per block ($16$ factorial + $4$ center + $8$ axial) were generated up front and shuffled together across all 140 runs (`rng = np.random.default_rng(42)`). Specifically, **30 of the 40 `Phase2_Axial` runs have `run_order <= 100`** in `results/runs.csv`.
- **Pre-Audit Classification**: **UNSUPPORTED** (calling the executed historical campaign "sequential" in the title and prose conflated textbook RSM pedagogy with the actual simultaneous randomized execution).
- **Implemented Correction**:
  1. Removed "Sequential" from the title of both `report.tex` and `REPORT.md` (*"Pre-Planned Response Surface Methodology and Face-Centered Central Composite Design..."*).
  2. Added Section 2.2 in `report.tex` and Section 1.3 in `REPORT.md` explicitly documenting that all 140 runs were generated and randomized in a single pre-planned campaign where `30` of `40` axial runs executed before `run_order = 100`, while Phase 1 and Phase 2 represent a structured *hierarchical analytical* decomposition.
  3. Rebuilt `figures/efficiency_comparison_curve.png` (`plots.py`) into a two-panel figure removing the misleading vertical divider at `run_order = 100`.

### Claim C-02 — Evaluation Budget Allocation: 25 Replicated Coordinates vs. 140 Adaptive Coordinates
- **Location**: Main Sections 3 and 7; Supplement Section 2 and design-seed table; `REPORT.md` Section 1.3.
- **Scientific Question**: How does the 140-evaluation budget allocation differ structurally between the blocked FCCD and heuristic/Bayesian optimizers?
- **Experimental Evidence & Source Artifact**: `results/runs.csv` and `results/revision_v2/full_run_001/doe_selection_runs.csv` evaluate $m = 25$ distinct coded coordinates replicated across 5 seed blocks ($16\times 5 = 80$ corners, $1\times 4\times 5 = 20$ center runs, $8\times 5 = 40$ axial runs). Of these 140 runs, **45 runs (`32.14%`) lie on the $x_1 = -1$ ($\eta = 0.01$) face** ($8\times 5 = 40$ factorial corners + $1\times 5 = 5$ axial runs), where 100 boosting trees severely underfit (mean validation RMSE `0.7545` vs. `0.5055` for $x_1 \in \{0, +1\}$). By contrast, Optuna TPE and Random Search (`results/revision_v2/full_run_001/optimizer_trials.csv`) evaluate **140 distinct hyperparameter coordinates** on a single development split (seed 42).
- **Pre-Audit Classification**: **NEEDS QUALIFICATION**.
- **Implemented Correction**: Explicitly documented the 25-coordinate vs. 140-coordinate trade-off and the `45/140` (`32.14%`) low-$\eta$ budget allocation in `report.tex` (Section 2.2 & Discussion) and `REPORT.md` (Section 1.3 & Section 8).

### Claim C-03 — Single-Sample CPU Latency Protocols and Prediction Interface Overhead
- **Location**: Main Section 3; Supplement Section 2; `REPORT.md` Section 1.3.
- **Scientific Question**: How is inference latency measured across historical search, prospective online search, and final holdout evaluation, and how much overhead does `Booster.predict(DMatrix)` add over `XGBRegressor.inplace_predict`?
- **Experimental Evidence & Source Artifact**:
  - Historical DOE (`results/runs.csv`): 50 warmup per interface (100 total) + 1,000 timed single-row calls on CPU Core 0; `predict - inplace_predict` overhead is **`40.88 ± 11.56 µs`** (range `5.75`–`89.33 µs`).
  - Historical Confirmation (`results/confirmation_runs*.csv`): MO overhead **`38.18 ± 5.24 µs`**, SO overhead **`36.85 ± 7.75 µs`**.
  - Historical Benchmark (`results/benchmark.csv`): overhead **`32.07 ± 1.16 µs`** (range `30.89`–`33.96 µs`).
  - Prospective Revision-v2 (`results/revision_v2/full_run_001/latency_interface_overhead.json`, 610 session records across 122 selections $\times$ 5 sessions): overhead **`32.19 ± 1.46 µs`** (range `27.20`–`37.43 µs`).
  - Online search timing (`results/revision_v2/full_run_001/optimizer_trials.csv`, `results/revision_v2/full_run_001/doe_selection_runs.csv`): 10 warmup + 30 timed calls per fit (`online_30call_single_row_cpu0`).
- **Pre-Audit Classification**: **NEEDS QUALIFICATION**.
- **Implemented Correction**: Added explicit protocol tier breakdown and interface overhead reconstructions across all four artifact sources in `scripts/reporting_data.py`, `report.tex` (Section 2.4), and `REPORT.md` (Section 1.3).

### Claim C-04 — Phase 1 Factorial Main Effects and Two-Factor Interactions
- **Location**: Main Section 4; Supplement Section 3 and Phase-1 ANOVA table; `REPORT.md` Section 2.1.
- **Scientific Question**: Which hyperparameters and two-factor interactions drive validation RMSE in the $2^4$ factorial + center screening stage?
- **Experimental Evidence & Source Artifact**: `results/runs.csv` subset `Phase1_Factorial` ($n_F = 80$) and `Phase1_Center` ($n_C = 20$). Type III OLS ANOVA (`val_rmse ~ C(block) + (x1+x2+x3+x4)^2`, $\text{df}_{\text{res}} = 85$, $\text{SS}_E = 0.296247$, $\text{MS}_E = 0.003485$): Factor A ($\ln \eta$) accounts for $\text{SS} = 1.245625, F = 357.40, p < 10^{-15}, \eta_p^2 = 0.8079$; Interaction $x_1 x_2$ accounts for $\text{SS} = 0.091725, F = 26.32, p < 0.0001, \eta_p^2 = 0.2364$; Factor B (depth) accounts for $\text{SS} = 0.082155, F = 23.57, p < 0.0001, \eta_p^2 = 0.2171$; Block effect $C(\text{block})$ is masked by unmodeled curvature ($\text{SS} = 0.004662, F = 0.33, p = 0.8541$).
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Maintained exact Type III ANOVA reconstruction and verified macro bindings.

### Claim C-05 — Phase 1 Curvature Test ($F_{\text{Curvature}} = 105,432.31$) and Denominator Sensitivity
- **Location**: Main Section 4; Supplement Section 3; `REPORT.md` Sections 2.2--2.3.
- **Scientific Question**: Why is the single-df curvature $F$-statistic $O(10^5)$, does curvature significance depend on using within-block center subsampling error as the denominator, and does rejecting $H_0: \sum \beta_{ii} = 0$ prove that a quadratic surface is adequate?
- **Experimental Evidence & Source Artifact**: From `results/phase1.json` and `results/runs.csv` ($n_F = 80, \bar{y}_F = 0.633282$; $n_C = 20, \bar{y}_C = 0.498935$; $\bar{y}_F - \bar{y}_C = +0.134347$ RMSE or $+26.93\%$; $\text{SS}_{\text{Curv}} = 0.288786$):
  1. **Within-block center pure error ($\text{df} = 15$)**: $\text{SS}_{\text{PE, center}} = 0.000041$, $\text{MS}_{\text{PE, center}} = 2.74 \times 10^{-6}$ (`2.739066861e-06`, $\text{SD} = 0.00166$ RMSE, reflecting only row-subsampling seed noise at $s = 0.75$ on a fixed data split $S_b$), giving $F = 105,432.31$ ($p < 10^{-15}$).
  2. **Block-adjusted pooled coordinate error ($\text{df} = 79$)**: Pools 15-df center PE + 64-df Phase 1 Treatment $\times$ Block interaction ($\text{SS} = 0.001127, \text{MS} = 1.43 \times 10^{-5}$), giving $F = 20,235.41$ ($p < 10^{-15}$).
  3. **Phase 1 Treatment $\times$ Block interaction ($\text{df} = 64$)**: $\text{SS} = 0.001086, \text{MS}_{T\times B} = 1.70 \times 10^{-5}$, giving $F = 17,013.24$ ($p < 10^{-15}$).
  4. **5-Block paired contrast ($\text{df} = 4$, block as independent unit)**: Computing $d_b = \bar{y}_{F,b} - \bar{y}_{C,b}$ across the 5 blocks yields $\bar{d} = +0.1343$ RMSE, $\text{SE} = 0.0016$, 95% CI $[+0.1300, +0.1387]$, $t = 85.69$, $F = t^2 = 7,342.69$ ($p = 1.11 \times 10^{-7}$).
  5. **Unadjusted coordinate pure error ($\text{df} = 83$)**: $\text{SS} = 0.005790, \text{MS}_{\text{PE, All}} = 6.98 \times 10^{-5}$, giving $F = 4,140.06$ ($p < 10^{-15}$).
  6. **First-order + 2FI + curvature residual ($\text{df} = 84$)**: $\text{SS} = 0.007461, \text{MS} = 8.88 \times 10^{-5}$, giving $F = 3,251.19$ ($p < 10^{-15}$).
  7. **Unadjusted first-order + 2FI residual ($\text{df} = 85$)**: $\text{SS} = 0.296247, \text{MS}_E = 0.003485$, giving $F = 82.86$ ($p = 3.34 \times 10^{-14}$).
- **Pre-Audit Classification**: **NEEDS QUALIFICATION**.
- **Implemented Correction**: Added the complete 7-denominator sensitivity table in `REPORT.md` (Section 2.3) and itemized sensitivity analysis in `report.tex` (Section 3.1), explicitly explaining why $\text{MS}_{\text{PE, center}}$ is micro-scale subsampling noise and distinguishing non-planarity detection from quadratic adequacy.

### Claim C-06 — Stochastic Nuisance Blocking and Intraclass Correlation ($\text{ICC} = 40.69\%$)
- **Location**: Main Section 3; Supplement Sections 2 and 4; `REPORT.md` Section 3.2.
- **Scientific Question**: How much residual validation RMSE variance is attributable to random train/validation split partitioning across the 5 seed blocks?
- **Experimental Evidence & Source Artifact**: `results/icc.json` and `results/runs.csv` Phase 2 ANOVA: $\text{SS}_{\text{block}} = 0.006966, \text{MS}_{\text{block}} = 0.001742$, $\text{SS}_E = 0.010429, \text{MS}_E = 0.000086$, $F_{\text{block}} = 20.21$ ($p = 9.11 \times 10^{-13}$). ANOVA variance component $\hat{\sigma}^2_{\text{block}} = (\text{MS}_{\text{block}} - \text{MS}_E)/28 = 5.91 \times 10^{-5}$ ($\hat{\sigma}_{\text{block}} = 0.0077$ RMSE), giving $\text{ICC}_{\text{ANOVA}} = 0.4069$ ($40.69\%$) and $\text{ICC}_{\text{REML}} = 0.4069$.
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Verified exact reconstruction and explained why Phase 1 unmodeled curvature ($\text{MS}_E = 0.003485$) masked block effects ($F = 0.33, p = 0.8541$) until quadratic terms reduced $\text{MS}_E$ by $40\times$ to $0.000086$ in Phase 2.

### Claim C-07 — Phase 2 Second-Order FCCD Fit ($R^2 = 0.9951$) and HC3 Robust Covariance
- **Location**: Main Section 4; Supplement Section 4 and second-order RMSE ANOVA table; `REPORT.md` Section 3.1.
- **Scientific Question**: Does the 19-parameter blocked second-order polynomial capture the primary response topography, and are factor inferences robust to heteroscedasticity?
- **Experimental Evidence & Source Artifact**: `results/runs.csv` ($N = 140$, $\text{df}_{\text{res}} = 121$). OLS $R^2 = 0.9951$, $\text{Adj } R^2 = 0.9944$. Under MacKinnon-White HC3 robust standard errors, $x_1, x_2, x_3, x_1^2, x_2^2, x_1 x_2, x_1 x_3$, and $x_1 x_4$ remain statistically significant.
- **Pre-Audit Classification**: **NEEDS QUALIFICATION**.
- **Implemented Correction**: Qualified that $R^2 = 0.9951$ reflects macro-scale variation across $\eta \in [0.01, 0.30]$ rather than absence of local approximation bias, and noted that HC3 standard errors correct parameter inference for heteroscedasticity without eliminating deterministic polynomial bias in prediction intervals.

### Claim C-08 — Lack-of-Fit Variance Decomposition (121 df) and Restricted-Domain Aliasing
- **Location**: Main Section 4; Supplement Section 4 and lack-of-fit table; `REPORT.md` Section 3.3.
- **Scientific Question**: How do the 121 residual degrees of freedom partition into structural lack of fit, treatment-by-block interaction, and center pure error, and what happens when refitting on $x_1 \ge -0.5$?
- **Experimental Evidence & Source Artifact**: `results/lof.json` and `results/runs.csv`:
  - Total residual ($121\text{ df}, \text{SS} = 0.010429, \text{MS} = 8.62 \times 10^{-5}$) $=$ Structural LoF ($10\text{ df}, \text{SS} = 0.008850, \text{MS} = 0.000885, \sqrt{\text{MS}_{\text{LoF}}} = 0.0297$) $+$ Treatment $\times$ Block interaction ($96\text{ df}, \text{SS} = 0.001538, \text{MS} = 1.60 \times 10^{-5}$) $+$ Within-block center pure error ($15\text{ df}, \text{SS} = 0.000041, \text{MS} = 2.74 \times 10^{-6}$).
  - LoF is significant against both center PE ($F = 323.10, p < 10^{-15}$) and the 111-df additive baseline ($\text{SS}_{\text{PE}} = 0.001579, \text{MS}_{\text{PE}} = 1.42 \times 10^{-5}, F = 62.21, p = 7.78 \times 10^{-41}$).
  - Dropping $x_1 = -1$ ($N = 95$) reduces $\text{SS}_{\text{LoF}}$ by $96.3\%$ to $0.000324$ ($\text{MS} = 0.000162, \text{RMS} = 0.0127$), but leaves only two levels of $x_1 \in \{0, 1\}$, aliasing $x_1^2 \equiv x_1$ and reducing $\text{df}_{\text{LoF}}$ from 10 to 2 ($F = 10.11, p = 0.0001$).
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Preserved the complete multi-scale decomposition and aliasing explanation in both documents.

### Claim C-09 — Quadratic Latency Model Adequacy and Jacobian-Adjusted AIC
- **Location**: Main Section 4; Supplement Section 4, latency tables, and latency-surface figure; `REPORT.md` Section 3.1.
- **Scientific Question**: What functional form best describes single-sample CPU inference latency across tree depths $d \in \{3, 6, 9\}$?
- **Experimental Evidence & Source Artifact**: `results/latency_models_comparison.csv`, `results/lof.json`, and `results/runs.csv`. The full second-order latency model achieves $R^2 = 0.8820$ ($\text{Adj } R^2 = 0.8644, \text{AIC} = 1209.6$), while the parsimonious quadratic-depth model achieves $R^2 = 0.8589$ ($\text{AIC} = 1202.6$, compared to linear-depth $R^2 = 0.7915, \text{AIC} = 1255.2$ and Jacobian-adjusted log-linear $\text{AIC} = -232.7 + 1,450.8 = 1218.1$). In the second-order ANOVA, linear depth ($x_2$: $F = 811.46, p < 10^{-15}$) and quadratic depth ($x_2^2$: $F = 39.86, p < 0.0001$) govern latency, while block effects ($F = 0.75, p = 0.5591, \text{ICC} = 0.0000$) and lack of fit ($F = 1.20, p = 0.3625$ vs. center PE; $F = 1.15, p = 0.3305$ vs. additive baseline) are non-significant.
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Verified Jacobian likelihood adjustment ($+2\sum\ln y_i = 1,450.8$) and added interface overhead context.

### Claim C-10 — Residual Homoscedasticity: Reconciling Levene, Brown-Forsythe, and Breusch-Pagan
- **Location**: Main Section 4; Supplement Section 5 and diagnostics figure; `REPORT.md` Section 3.3.
- **Scientific Question**: Why do Levene ($p = 0.9507$) and Brown-Forsythe ($p = 0.8552$) fail to reject homoscedasticity while Breusch-Pagan ($\text{LM} = 70.10, p < 0.0001$) is highly significant?
- **Experimental Evidence & Source Artifact**: Direct reconstruction from `results/runs.csv` and `results/diagnostics.json`:
  - `levene_stat = 0.1754, levene_p = 0.9507`: `scipy.stats.levene(*groups, center='mean')` grouped across the **5 seed blocks** (`block` $1\dots5$, 28 runs/block). This confirms that residual variance is uniform across data splits.
  - `brown_forsythe_points_stat = 0.6872, brown_forsythe_points_p = 0.8552`: `scipy.stats.levene(*groups, center='median')` grouped across the **25 distinct design coordinates** (`point_id`, 5 block observations per non-center coordinate and 20 at the center). Because all 5 block observations at a given coordinate $i$ share the same deterministic polynomial misfit $(\bar{y}_{i\cdot} - \hat{y}_i)$, centering each cell at its cell median removes the cell-mean lack-of-fit offset and tests only within-cell cross-block spread, which is homogeneous across the 25 coordinates.
  - `breusch_pagan_stat = 70.10, breusch_pagan_p = 4.35 \times 10^{-8}`: Regresses raw squared OLS residuals $e_{ij}^2 = [(\bar{y}_{i\cdot} - \hat{y}_i) + (y_{ij} - \bar{y}_{i\cdot})]^2$ on the continuous model regressors. Because deterministic lack of fit $(\bar{y}_{i\cdot} - \hat{y}_i)^2$ is concentrated at the corners and the $\eta = 0.01$ face, $e_{ij}^2$ varies systematically with $\hat{y}$.
- **Pre-Audit Classification**: **NEEDS QUALIFICATION** (previously did not state which grouping factor each test used or why Brown-Forsythe removes cell-mean LoF).
- **Implemented Correction**: Explicitly documented the exact grouping factors (5 blocks vs. 25 coordinates vs. continuous regressors) and the algebraic decomposition $e_{ij} = (\bar{y}_{i\cdot} - \hat{y}_i) + (y_{ij} - \bar{y}_{i\cdot})$ in `report.tex` (Section 5) and `REPORT.md` (Section 3.3).

### Claim C-11 — Canonical Spectral Decomposition and Stationary/Rising Ridge Classification
- **Location**: Main Section 5; Supplement Section 6, depth table, and RMSE-surface figure; `REPORT.md` Section 4.
- **Scientific Question**: What is the global curvature geometry of the fitted second-order validation RMSE surface?
- **Experimental Evidence & Source Artifact**: `results/phase3.json`. Eigenvalues of $\hat{\mathbf{B}}$ are $\lambda = [0.000057, 0.000873, 0.022759, 0.110196]$ ($\text{trace}(\hat{\mathbf{B}}) = 0.133886$). Rademacher wild bootstrap ($B=2,000$) yields a 95% CI for $\lambda_1$ of $[-0.0039, +0.0022]$ with $68.2\%$ of resamples $\le 0$, confirming a flat ridge along subsample ($x_3$) and regularization ($x_4$) and an exterior stationary point ($\mathbf{x}_0 = [+0.4171, +1.3524, +15.6944, -6.8009]^T, \|\mathbf{x}_0\|_2 = 17.16, \hat{y}_0 = 0.4161$). Constrained optimization over integer depths $d \in \{3, \dots, 9\}$ identifies $\mathbf{x}^*_{\text{SO}}$ at depth 7 ($[+0.5983, +0.3333, +1.0000, +1.0000]^T \implies \eta = 0.1515, d = 7, s = 1.0000, \lambda = 10.0000, \hat{y} = 0.4483$).
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Verified all spectral macros and table entries.

### Claim C-12 — Multi-Objective Derringer-Suich Desirability Compromise ($\mathbf{x}^*_{\text{MO}}$)
- **Location**: Main Section 5; Supplement Section 7, sensitivity table, and desirability figure; `REPORT.md` Section 5.
- **Scientific Question**: Which operating coordinate balances validation RMSE ($[0.450, 0.700]$) and single-sample latency ($[100.0, 180.0]\,\mu\text{s}$) while avoiding boundary extrapolation?
- **Experimental Evidence & Source Artifact**: `results/confirmation.json` and `results/desirability_sensitivity.csv`. At $\mathbf{x}^*_{\text{MO}} = [+0.8500, -0.6667, +1.0000, -0.0812]^T$ ($\eta = 0.2324, d = 4, s = 1.0000, \lambda = 0.8295$), predicted responses are $\hat{Y}_1 = 0.4804$ and $\hat{Y}_2 = 138.11\,\mu\text{s}$ ($d_1 = 0.8783, d_2 = 0.5236, D = 0.6782$).
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Verified interior desirability sweep macros and sensitivity table.

### Claim C-13 — Phase 5 Satterthwaite Confirmation Intervals ($m = 10$ Fresh Seeds)
- **Location**: Main Section 6 and confirmation table; Supplement Section 8; `REPORT.md` Section 6.
- **Scientific Question**: Do empirical confirmation runs across $m = 10$ fresh seeds (`505..1414`) validate the second-order surrogate predictions at $\mathbf{x}^*_{\text{MO}}$ (depth 4) and $\mathbf{x}^*_{\text{SO}}$ (depth 7)?
- **Experimental Evidence & Source Artifact**: `results/confirmation.json`, `results/confirmation_runs.csv`, and `results/confirmation_runs_single_obj.csv`:
  - At $\mathbf{x}^*_{\text{MO}}$ ($h_0 = 0.1043, \nu_{\text{eff}} = 14.0$), empirical validation RMSE is $0.4853 \pm 0.0115 \in [0.4677, 0.4932]$ (confirmed; holdout test RMSE $0.4888 \pm 0.0032$), and empirical latency is $142.33 \pm 7.79\,\mu\text{s}$ ($142.3 \pm 7.8\,\mu\text{s} \in [122.8, 153.4]\,\mu\text{s}$, confirmed).
  - At $\mathbf{x}^*_{\text{SO}}$ ($h_0 = 0.1210, \nu_{\text{eff}} = 15.1$), empirical validation RMSE is $0.4700 \pm 0.0093 \notin [0.4354, 0.4612]$ (exceeds upper PI bound by $+0.0088$; surrogate optimism bias $+0.0217$ RMSE or $6.8\times \text{SE}(\hat{y})$; holdout test RMSE $0.4691 \pm 0.0034$), while empirical latency is $171.1 \pm 10.9\,\mu\text{s}$ ($171.12 \pm 10.95\,\mu\text{s}$) at the lower PI bound $[171.1, 202.9]\,\mu\text{s}$ (borderline).
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Explicitly noted that Satterthwaite intervals pool structural lack of fit into $\text{MS}_E$ and do not protect against local deterministic polynomial bias.

### Claim C-14 — Provenance of Three- and Four-Decimal-Place Numerical Similarities Across Tables
- **Location**: Main Section 6 summary; Supplement Section 8 and confirmation/benchmark tables; `REPORT.md` Section 6.1.
- **Scientific Question**: Why do numbers such as `0.4686`/`0.4687`/`0.4690`/`0.4691`, `0.4853`, `0.4884`/`0.4888`, and `0.4700` appear across confirmation ($m=10$ seeds `505..1414`), historical benchmark ($m=20$ seeds `2001..2020`), and Revision-v2 tables?
- **Experimental Evidence & Source Artifact**: Reconstructed from `results/confirmation_runs*.csv`, `results/benchmark.csv`, `results/revision_v2/full_run_001/finalized_selections.json`, and `results/revision_v2/full_run_001/final_evaluations.csv`:
  1. **Shared Configuration Hash vs. Adjacent Grid Point Across 20-Seed Benchmark Evaluations (`2001..2020`)**: Both `results/benchmark.csv` and `results/revision_v2/full_run_001/final_evaluations.csv` evaluate `doe-so-historical` (`config_hash = 0079fbae0ec5b7cb`: $\eta = 0.1515, d = 7, s = 1.0, \lambda = 10.0$) and `doe-mo-historical` (`config_hash = b05e8e5cb67e8668`: $\eta = 0.2324, d = 4, s = 1.0, \lambda = 0.8295$) on the exact same 20 seeds `2001..2020`, yielding identical full-precision RMSEs (`test_rmse = 0.469033, val_rmse = 0.468680` for $\mathbf{x}^*_{\text{SO}}$; `test_rmse = 0.488409, val_rmse = 0.482071` for $\mathbf{x}^*_{\text{MO}}$). All 20 `doe-so-rep-0..19` selections choose the adjacent predeclared grid point `grid_index = 1052` (`config_hash = 9d7c4e74ff156870`: coded $[+0.6, +1/3, +1.0, +1.0]^T, \eta = 0.1519, d = 7, s = 1.0, \lambda = 10.0$), which is deduplicated across all 20 replicates on seeds `2001..2020` (`test_rmse = 0.468591, val_rmse = 0.468723`, between-search $\text{SD} = 0.00000$).
  2. **Disjoint Seed Sets (`505..1414` vs. `2001..2020`)**: At $\mathbf{x}^*_{\text{SO}}$, the 10-seed confirmation holdout test RMSE mean is `0.469124` ($\text{SD} = 0.003437$, rounding to `0.4691`) vs. the 20-seed historical benchmark mean `0.469033` ($\text{SD} = 0.002585$, rounding to `0.4690`, differing by $91.25\times 10^{-6}$ RMSE). At $\mathbf{x}^*_{\text{MO}}$, the 10-seed confirmation holdout test RMSE mean is `0.488774` ($\text{SD} = 0.003243$, rounding to `0.4888`) vs. the 20-seed historical benchmark mean `0.488409` (`0.4884`), while the 20-seed Repeated DOE MO validation mean `0.485283` (`0.48528`) and the 10-seed confirmation validation mean `0.485286` both round to `0.4853` at four decimal places.
- **Pre-Audit Classification**: **NEEDS QUALIFICATION** (without explicit explanation, three/four-decimal similarities across tables could be misread as copy-pasted or leaked numbers).
- **Implemented Correction**: Added Section 8.2 in `report.tex` and Section 6.1 in `REPORT.md` documenting the exact 6-decimal-place values, configuration hashes, and seed sets.

### Claim C-15 — Holdout Test Set Isolation History
- **Location**: Main Sections 3, 6, and 8; Supplement Sections 1 and 8; `REPORT.md` Section 7.
- **Scientific Question**: Was the 4,128-row external holdout test set strictly untouched across the entire history of the project?
- **Experimental Evidence & Source Artifact**: In historical `v1.0.0` (`pipeline.py`, `results/runs.csv`), `test_rmse` was logged during exploratory runs, whereas in `Revision-v2` (`final_evaluation.py`, `results/revision_v2/full_run_001/optimizer_trials.csv`, `results/revision_v2/full_run_001/doe_selection_runs.csv`), test labels were excluded from all search functions until `results/revision_v2/full_run_001/finalized_selections.json` was frozen.
- **Pre-Audit Classification**: **SUPPORTED** (already qualified in D.2, preserved in Work Package F/F.1).
- **Implemented Correction**: Maintained explicit disclosure that `Revision-v2` programmatic safeguards isolate the revised search pipeline but cannot undo prior historical exposure of the dataset.

### Claim C-16 — Meaning of the 20 Repeated DOE Replicates and 1,323-Point Grid Selection
- **Location**: Main Section 6; Supplement Section 8 and benchmark table; `REPORT.md` Section 7.3.
- **Scientific Question**: What varies across the $N = 20$ Repeated DOE replicates, how are $\mathbf{x}^*_{\text{MO}}$ and $\mathbf{x}^*_{\text{SO}}$ selected in each replicate, and why is between-search SD zero for DOE-SO?
- **Experimental Evidence & Source Artifact**: Inspection of `desirability_provenance.py`, `results/revision_v2/full_run_001/doe_selection_protocol.json`, and `results/revision_v2/full_run_001/doe_selection_replicates.csv`:
  - Each replicate $r \in \{0, \dots, 19\}$ draws 5 fresh block seeds `3001 + 5r` through `3005 + 5r` (100 distinct block seeds `3001..3100` across the 20 replicates), runs the 140-fit blocked FCCD, fits second-order OLS models for validation RMSE and online latency, and evaluates them on a deterministic **1,323-point candidate grid** ($21$ coded learning-rate levels $x_1 \in [-1, +1]$ in steps of $0.1 \times 7$ integer depths $d \in \{3, \dots, 9\} \times 3$ coded subsample levels $x_3 \in \{-1, 0, +1\} \times 3$ coded regularization levels $x_4 \in \{-1, 0, +1\}$).
  - For DOE-SO, the grid minimum on all 20 5-block surfaces lies at the same boundary grid point `grid_index = 1052` ($[+0.6, +1/3, +1.0, +1.0]^T \implies \eta = 0.1519, d = 7, s = 1.0, \lambda = 10.0$), producing 1 unique `config_hash` (`9d7c4e74ff156870`) across all 20 replicates (`between-search SD = 0.00000`), while retraining across the 20 evaluation seeds `2001..2020` yields $\sigma_{\text{eval}} = 0.00304$.
  - For DOE-MO, desirability maximization over feasible grid points selects across 12 distinct grid configurations in $d \in \{4, 5\}$, yielding between-search $\text{SD} = 0.00732$ and retraining $\sigma_{\text{eval}} = 0.00330$.
- **Pre-Audit Classification**: **NEEDS QUALIFICATION**.
- **Implemented Correction**: Explicitly documented the 1,323-point grid specification, block seed range `3001..3100`, and variance decomposition in `report.tex` (Section 8.3, Finding 2) and `REPORT.md` (Section 7.3, Finding 2).

### Claim C-17 — Statistical Comparison of Repeated DOE MO vs. Multi-Objective TPE
- **Location**: Main Abstract and Section 7; Supplement Section 8 and benchmark table; `REPORT.md` Section 7.3.
- **Scientific Question**: Which statistical test is primary when comparing Repeated DOE MO ($N=20$) and Multi-Objective TPE ($N=20$), and what does a non-significant $p$-value establish?
- **Experimental Evidence & Source Artifact**: Replicate $r \in \{0, \dots, 19\}$ for Repeated DOE uses block seeds `3001+5r..3005+5r` (holdout test RMSE $0.48997 \pm 0.00732$, `predict` latency $122.29 \pm 2.70\,\mu\text{s}$), whereas replicate $r$ for MO-TPE uses Optuna sampler seed `1000+r` on fixed development split 42 (holdout test RMSE $0.49255 \pm 0.01080$, `predict` latency $121.46 \pm 3.75\,\mu\text{s}$). Because replicate index $r$ does **not** represent a shared random data split or common experimental block, the two samples are independent rather than naturally paired.
  - **Primary Test (Welch's two-sample $t$-test)**: $\Delta = -0.00258$ RMSE, $t = -0.8849, p = 0.3826$.
  - **Secondary Sensitivity Check (Index-paired $t$-test in `paired_comparisons.json`)**: $t = -0.966, \text{df} = 19, p = 0.3463$.
- **Pre-Audit Classification**: **NEEDS QUALIFICATION** (previously listed Welch and paired $t$-tests on equal footing without explaining why replicate index $r$ is not a true block pairing).
- **Implemented Correction**: Designated Welch's two-sample $t$-test as the primary inferential comparison and index-paired $t$-testing as a secondary sensitivity check in `report.tex` and `REPORT.md`, and reiterated that non-significance indicates no detected difference rather than formal equivalence (`rmse_equivalence_margin: null`).

### Claim C-18 — Development Candidate Hypervolume Comparability Across Protocols
- **Location**: Main Section 7; Supplement Section 8 and hypervolume table; `REPORT.md` Section 7.2.
- **Scientific Question**: Are the development hypervolume rows in Panel A of Table 12 directly comparable two-sample experiments?
- **Experimental Evidence & Source Artifact**: `results/revision_v2/full_run_001/hypervolume.json` and `results/benchmark_summary.json`:
  - Row 1 (`MO-TPE Candidates`): 20 searches $\times$ 140 adaptive trials on single split 42, timed with 30 calls/trial ($\text{HV} = 18.3834 \pm 0.3301$ at $[0.60, 250.0]$; $30.0535 \pm 0.4123$ at $[0.65, 275.0]$).
  - Row 2 (`Repeated DOE Candidate Fronts`): 20 searches $\times$ 25 FCCD coordinates averaged over 5 block seeds (`3001..3100`), timed with 30 calls/fit ($\text{HV} = 15.6128 \pm 0.7268$ at $[0.60, 250.0]$; $26.6847 \pm 0.8786$ at $[0.65, 275.0]$).
  - Row 3 (`Fixed Full DOE Evaluated Front`): 1 deterministic 27-point set (25 FCCD points + $\mathbf{x}^*_{\text{MO}} + \mathbf{x}^*_{\text{SO}}$) on split 42 ($\text{HV} = 16.8585$ at $[0.60, 250.0]$; $28.1430$ at $[0.65, 275.0]$).
  - Row 5 (`Difference: Fixed Full DOE Front - MO-TPE Mean`): $-1.5249$ at $[0.60, 250.0]$ ($-1.9105$ at $[0.65, 275.0]$; $p < 0.0001$). Because the 27-point DOE front on split 42 is a single deterministic constant ($N=1$), this $p$-value is a **one-sample $t$-test** of the 20 MO-TPE runs against a fixed constant on split 42, not a two-sample test across independent splits.
  - Historical `v1.0.0` (`results/benchmark_summary.json`): Reported $\text{HV}_{\text{DOE}} = 18.38$ vs. $\text{HV}_{\text{MO-TPE}} = 16.86$ ($+9.0\%$) by comparing all 25 DOE design points against a single selected MO-TPE point; single-point comparison ($\mathbf{x}^*_{\text{MO}}$ vs. MO-TPE) is $16.92$ vs. $16.86$ ($+0.3\%$).
- **Pre-Audit Classification**: **NEEDS QUALIFICATION**.
- **Implemented Correction**: Updated `tables/tab_hypervolume_comparison.tex`, `report.tex` (Section 8.3), and `REPORT.md` (Section 7.2) to label the Split-42 difference as `(1-sample p < 0.0001)` and explain all protocol and candidate-pool differences.

### Claim C-19 — Holdout Non-Dominated Set and Paired Seed-Bootstrap Uncertainty
- **Location**: Main Abstract and Section 7 with Pareto figure; Supplement Section 8 and hypervolume table; `REPORT.md` Sections 7.2--7.3.
- **Scientific Question**: How sensitive is the 12-configuration empirical holdout non-dominated set ($\text{HV} = 17.6714$ at $[0.60, 250.0]$ and $28.9900$ at $[0.65, 275.0]$) to resampling the 20 evaluation seeds (`2001..2020`) and 5 latency sessions?
- **Experimental Evidence & Source Artifact**: Reconstructed in `scripts/reporting_data.py` (`_compute_holdout_pareto_bootstrap`, $B = 2,000$ paired cluster bootstrap resamples over the 20 evaluation seeds `2001..2020` and 5 latency sessions, `seed = 20261011`):
  - **Seed-Bootstrap Holdout Hypervolume at $[0.60, 250.0]$**: **`17.6923 ± 0.0564`**, 95% CI **`[17.5848, 17.8061]`**, mean non-dominated frontier size **`13.42 ± 1.46`**.
  - **Seed-Bootstrap Holdout Hypervolume at $[0.65, 275.0]$**: **`29.0170 ± 0.0688`**, 95% CI **`[28.8861, 29.1579]`**.
  - **Joint Seed + Latency-Session Bootstrap at $[0.60, 250.0]$**: **`17.6890 ± 0.0727`**, 95% CI **`[17.5507, 17.8420]`**.
  - **Optimizer Frontier Presence Across $B = 2,000$ Seed Resamples**:
    - Repeated DOE MO: present in **`100.0%`** of bootstrap frontiers (3 point-estimate points; mean `2.76` points).
    - Multi-Objective TPE: present in **`100.0%`** of bootstrap frontiers (3 point-estimate points; mean `3.16` points).
    - Constrained TPE: present in **`100.0%`** of bootstrap frontiers (5 point-estimate points; mean `6.03` points).
    - Single-Objective TPE: present in **`91.5%`** of bootstrap frontiers (1 point-estimate point; mean `1.38` points).
- **Pre-Audit Classification**: **NEEDS QUALIFICATION** (previously reported only the point-estimate 12-configuration frontier without resampling uncertainty).
- **Implemented Correction**: Added the Holdout Frontier Seed-Bootstrap row and per-optimizer bootstrap presence rates to `tables/tab_hypervolume_comparison.tex`, `report.tex` (Section 8.3), `REPORT.md` (Section 7.2–7.3), and `docs/generated/scientific_results_manifest.json`.

### Claim C-20 — Constrained TPE Feasibility Degradation ($20/20 \to 9/20$)
- **Location**: Main Sections 7 and 8; Supplement Section 8 and benchmark table; `REPORT.md` Section 7.3.
- **Scientific Question**: Why does Constrained TPE ($\le 145\,\mu\text{s}$) drop from $20/20$ ($100\%$) search-time feasibility to $9/20$ ($45\%$) benchmark feasibility?
- **Experimental Evidence & Source Artifact**: `results/revision_v2/full_run_001/finalized_selections.json` and `results/revision_v2/full_run_001/final_summary.csv`. All 9 depth-6 selections remain feasible ($9/9$ at $136.70 \pm 0.82\,\mu\text{s}$), whereas all 11 depth-7 selections exceed $145\,\mu\text{s}$ under 1,000-call warmed verification ($0/11$ feasible at $150.45 \pm 0.67\,\mu\text{s}$; overall mean $144.27 \pm 7.06\,\mu\text{s}$) after passing the noisy 30-call online check during search.
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Maintained exact depth-stratified breakdown and qualified the mechanisms (30-call measurement variance, warmup protocol differences, and winner's-curse selection near the constraint boundary).

### Claim C-21 — Complete Computational Budget Ledger and Wall-Clock Cost
- **Location**: Main Section 7; Supplement computational-budget table; `REPORT.md` Section 7.4.
- **Scientific Question**: How do the 17,077 XGBoost fits, 1,640,810 timed calls, 201,270 warmup calls, and per-replicate wall-clock times break down across optimizers and evaluation stages?
- **Experimental Evidence & Source Artifact**: `results/revision_v2/full_run_001/computational_budget.json`, `results/revision_v2/full_run_001/optimizer_replicates.csv`, `results/revision_v2/full_run_001/optimizer_trials.csv`, `results/revision_v2/full_run_001/doe_selection_replicates.csv`, and `results/revision_v2/full_run_001/doe_selection_runs.csv`:
  - Search fits: $2,800$ Repeated DOE + $4 \times 2,800 = 11,200$ optimizer trials + $27$ Split-42 DOE baseline = $14,027$ online-timed fits ($420,810$ timed calls + $140,270$ warmup calls).
  - Primary latency verification: $5\text{ sessions} \times 122\text{ records} = 610$ refits ($1,220,000$ timed calls + $61,000$ warmup calls across 2 interfaces).
  - Final holdout retraining: $122\text{ records} \times 20\text{ seeds} = 2,440$ evaluation rows (deduplicated across 95 distinct hashes $\times 20 = 1,900$ unique booster trainings).
  - Total: **$17,077$ model fits**, **$1,640,810$ timed calls**, **$201,270$ warmup calls**, **$1,842,080$ total single-row calls**.
  - Per-replicate search wall-clock time and mean training duration per fit: Repeated DOE **`36.49 ± 0.19 s`** (`729.76 s` total, `238.6 ms/fit`), MO-TPE **`26.24 ± 1.80 s`** (`524.87 s` total, `174.2 ms/fit`), Constrained TPE **`26.50 ± 2.70 s`** (`529.99 s` total, `176.9 ms/fit`), Random Search **`32.61 ± 1.63 s`** (`652.12 s` total, `221.9 ms/fit`), and SO-TPE **`54.03 ± 3.05 s`** (`1,080.62 s` total, `370.2 ms/fit`, due to adaptive concentration in depths 7–9).
- **Pre-Audit Classification**: **NEEDS QUALIFICATION** (previously stated only aggregate totals without per-method wall-clock and warmup breakdowns).
- **Implemented Correction**: Added Table 13 (`tables/tab_computational_budget.tex`) in `report.tex` and Section 7.4 in `REPORT.md`.

### Claim C-22 — Historical Latency Pinning, Literature Specificity, Immutable Commit SHAs, and AI Disclosure
- **Location**: Main Sections 3 and 10; Supplement latency and code-availability sections; `REPORT.md` Sections 7.3 and 9.
- **Scientific Question**: Were historical `v1.0.0` runs unpinned, which specific chapters of Montgomery (2017) and Myers et al. (2016) ground each statistical method, which Git commit SHAs lock each project phase, and how were AI assistants used and audited?
- **Experimental Evidence & Source Artifact**:
  - `scripts/run_confirmation.py` and `scripts/run_benchmarks.py` both call `SetProcessAffinityMask(GetCurrentProcess(), 1)` (CPU Core 0 pinning), so historical vs. prospective latency differences reflect between-session drift ($\text{SD} = 1.28\,\mu\text{s}$) rather than unpinned execution.
  - Specific textbook citations added: Montgomery (2017) Chapters 4 (RCBD), 5–7 ($2^k$ factorials & blocking), 10 (regression), 11 (RSM & CCD), and 13 (random factors / mixed models); Myers, Montgomery & Anderson-Cook (2016) Chapters 6 (canonical/ridge analysis) and 7 (desirability).
  - Immutable Git commit SHAs documented: `cd63135d2b70a0ee324555c11b67a541b32b98a4` (`v1.0.0`), `421cf24229f1f5fba26cdb3103ec989f9a427c96` (`Revision-v2`, with launch parent `45fe6f7ca8eb66b479edfb68d4b7c0c154a5d624`), and `6b53dac85af261553a7870f4888d5ea7423d4b5e` (Work Package D.2).
  - AI-Assistance and Verification Disclosure added in Section 10 of `report.tex` and Section 9 of `REPORT.md`.
- **Pre-Audit Classification**: **SUPPORTED**.
- **Implemented Correction**: Integrated into `report.tex` and `REPORT.md` and enforced by automated tests in `tests/test_reporting_integrity.py`.
