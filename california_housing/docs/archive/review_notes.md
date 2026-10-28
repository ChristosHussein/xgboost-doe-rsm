# Codex audit findings for Antigravity

Audience: Antigravity. This is an audit handoff, not an implementation of the fixes.

Audited base: `caf3a5bf59f2eae2541d3478d5fc370f0b339717` (`origin/main` fetched on 8 October 2026). All file/line links are pinned to that revision. Recheck the evidence before remediation if the source changes.

Severity: **P1** = high-priority issue affecting a central claim or reproducibility; **P2** = substantive methodological or interpretation weakness; **P3** = lower-impact reporting/maintenance issue. No P0 emergency is alleged.

Finding IDs `CR-001` through `CR-017` are stable. Confirmed discrepancies and methodological limitations are distinguished in the evidence. Record resolution evidence against each ID.

Review date: 8 October 2026. Page numbers below are the printed PDF page numbers.

## Assessment

The project is a useful worked example of blocked response-surface analysis, but its claims about optimizer performance and sequential experimentation are not established by the current implementation. The largest problems are experimental comparisons and reproducibility, rather than arithmetic in the main ANOVA calculations.

I reviewed the current PDF, LaTeX, configuration, analysis and experiment scripts, plotting and report generators, tests, current results, and legacy artifacts. I recalculated selected statistics from the stored run-level data. I did not rerun the complete training/optimization benchmark, so this is not certification that the stored measurements were produced by the current source.

## High-priority findings

### CR-001 [P1] The latency-aware TPE baselines optimize a different objective from DOE

**Severity:** P1

**Status:** Resolved

**File/line:** [`scripts/run_benchmarks.py:254`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L254), [`scripts/run_benchmarks.py:294`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L294)

**Evidence and impact:**
**Pages 17–18; confirmed implementation mismatch.**

Both constrained and multi-objective TPE use `lat_est = 115 + 3*depth + 0.8*depth**2` during search. They do not measure latency and do not use the fitted DOE latency surface. The formula ignores learning rate, subsampling, regularization and actual tree topology. Consequently, the headline comparison confounds optimization method with latency-objective quality.

The constrained method filters configurations using this proxy, so the reported measured threshold violation cannot simply be attributed to hardware variability: proxy error is an unmeasured alternative explanation. The code also has a fallback that can return an infeasible trial if none passes the proxy constraint.

**Suggested fix:** measure the same latency response for every method, or supply the same independently validated proxy to every method and describe that experiment explicitly. Recompute the main comparative claims afterward.

**Resolution:**
- Removed the hardcoded polynomial latency proxy `115 + 3*depth + 0.8*depth**2` entirely from `scripts/run_benchmarks.py`.
- Implemented `measure_trial_latency(model, sample)` which measures genuine online single-sample prediction latency directly on candidate models during search (`model.set_params(n_jobs=1)`, `booster.set_param({"nthread": 1})`, 10 warmup iterations, 30 measured iterations per candidate trial; and 50 warmup, 1000 iterations $\times$ 5 repeats during final dedicated incumbent evaluation).
- Both Constrained TPE and Multi-Objective TPE now optimize genuine single-sample prediction latency under identical single-threaded conditions, with infeasible trial fallback removed to guarantee strict threshold compliance.
- Updated Section 6.4 in `report.tex`, recompiled `report.pdf`, regenerated `tables/tab_benchmarks.tex`, and updated macros in `results/macros.tex`.

Evidence: [run_benchmarks.py](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L254), [MO objective](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L294).

### CR-002 [P1] Reported baseline results are for synthetic median configurations, not the optimizer incumbents

**Severity:** P1

**Status:** Resolved

**File/line:** [`scripts/run_benchmarks.py:399`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L399), [`scripts/run_benchmarks.py:460`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L460)

**Evidence and impact:**
**Pages 17–18, Table 11; confirmed implementation mismatch.**

After the optimizer runs, the code takes a coordinate-wise median across all winning hyperparameter vectors, then evaluates that one vector. A coordinate-wise median need not be a configuration any run selected, and interacting hyperparameters make its performance different from the median performance of the winners.

Thus the confidence intervals and paired tests compare selected fixed configurations across retraining seeds. They do not quantify the variability or expected performance of an optimization algorithm across its independent search runs. Aggregating winners from multiple searches also makes the claimed per-search evaluation budget an incomplete description of how the final reported configuration was constructed.

**Suggested fix:** evaluate each actual incumbent and summarize performance across optimizer repetitions; separate optimizer randomness from retraining/split randomness.

**Resolution:**
- Eliminated coordinate-wise median averaging (`np.median(..., axis=0)`).
- Implemented `select_median_actual_incumbent()`: for all baseline algorithms across the 20 search replicates, the evaluated configuration is the actual winning hyperparameter vector produced by the median-performing search replicate (ranked by validation RMSE for single-objective / random search / constrained TPE, and by desirability for multi-objective TPE).
- Persisted all 20 actual incumbents across all runs to `results/benchmark_optimizer_incumbents.csv` with decoded hyperparameters and objective scores.
- Reran the empirical benchmark across all 20 fresh evaluation seeds (seeds 2001-2020) with pinned CPU core affinity and interleaved repeats.
- Updated `report.tex`, `tables/tab_benchmarks.tex`, and `results/macros.tex`.

Evidence: [median aggregation](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L399), [paired comparisons](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L460).

### CR-003 [P1] The experiment was not executed sequentially as claimed

**Severity:** P1

**Status:** Open

**File/line:** [`pipeline.py:340`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/pipeline.py#L340), [`report.tex:457`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L457)

**Evidence and impact:**
**Pages 1, 5–6, 18–20; confirmed protocol mismatch.**

The entire FCCD is created in advance and shuffled as one batch. The stored execution order has a Phase 2 axial run at evaluation 8, while Phase 1 continues through evaluation 140. Therefore, the experiment does not demonstrate that the curvature test informed the decision to allocate the next phase's evaluations.

**Suggested fix:** describe this as a preplanned FCCD analyzed in stages, or rerun Phase 1, apply a prespecified augmentation rule, then execute Phase 2 with randomization within each phase. The efficiency figure must use that actual sequential history.

Evidence: [global randomization](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/pipeline.py#L340), [stored runs](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/runs.csv).

### CR-004 [P1] The reported multi-objective “optimum” is a manually selected operating point

**Severity:** P1

**Status:** Open

**File/line:** [`report.tex:355`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L355), [`scripts/run_benchmarks.py:354`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L354)

**Evidence and impact:**
**Page 14; confirmed and partially acknowledged in the text.**

The report chooses a point with desirability 0.6782 despite reporting a grid candidate with 0.6952. The confirmation and benchmark scripts hardcode the chosen coordinate. Choosing a conservative interior point can be reasonable, but there is no quantified robustness penalty or selection rule making it the optimum of the stated objective. A boundary point inside the declared domain is not automatically extrapolation.

The report itself supplies the higher grid desirability. This establishes that the selected point is not the maximum of the stated fitted objective, without establishing which point has better measured performance.

**Suggested fix:** call it a selected compromise, specify a reproducible robustness/selection rule, and empirically compare it with the actual desirability maximizer.

Evidence: [hardcoded selection](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L354), [stored fitted surface and coefficients](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/phase3.json).

### CR-005 [P2] The holdout was evaluated during experimental design

**Severity:** P2

**Status:** Open

**File/line:** [`pipeline.py:200`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/pipeline.py#L200), [`report.tex:132`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L132)

**Evidence and impact:**
**Pages 2, 4, 16, 19–20; confirmed wording/protocol discrepancy.**

The PDF says the holdout remains strictly untouched until final testing. `evaluate_model` predicts on it and computes test RMSE for every design run; those values are saved in the design CSV. The holdout rows are excluded from training, which is good, and I found no automatic objective using this test metric. However, the stronger assertion that the test set was never evaluated or exposed during tuning is false. Human selection leakage cannot be ruled out from the code.

**Suggested fix:** separate training/validation evaluation from final holdout evaluation and describe exactly when test results became available. Do not claim demonstrated leakage where the evidence only establishes early exposure.

Evidence: [test evaluation](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/pipeline.py#L201), [stored runs](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/runs.csv).

### CR-006 [P1] Reproduction can produce a mixture of old results and new claims

**Severity:** P1

**Status:** Open

**File/line:** [`scripts/generate_report_artifacts.py:609`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/generate_report_artifacts.py#L609), [`scripts/reproduce_all.py:38`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/reproduce_all.py#L38), [`scripts/run_confirmation.py:54`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_confirmation.py#L54), [`report.tex:469`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L469)

**Evidence and impact:**
**Pages 16 and 19; confirmed reproducibility defects.**

The report generator hardcodes the revised Satterthwaite intervals and confirmation status labels. The confirmation script still calculates intervals with the residual degrees of freedom instead. I independently recomputed the Satterthwaite formulas: the printed RMSE intervals agree for the current dataset. The defect is that they will not update correctly when the experiment changes.

The reproduction script preserves existing design and benchmark files while overwriting environment metadata and rerunning confirmation. There is no source/results provenance manifest. The cited commit `4528425` is the initial commit, not the revision containing the current report's corrections. The legacy top-level runner and verification module also fail on obsolete imports.

**Suggested fix:** calculate all intervals and statuses from current results, preserve per-run environment/source metadata, invalidate dependent artifacts when inputs change, cite the exact release, and remove or clearly retire obsolete entry points.

Evidence: [hardcoded intervals/status](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/generate_report_artifacts.py#L609), [confirmation calculations](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_confirmation.py#L55), [reproduction caching](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/reproduce_all.py#L38).

## Statistical and interpretation weaknesses

### CR-007 [P2] The uncertainty model still assumes away important detected model defects

**Severity:** P2

**Status:** Open

**File/line:** [`analysis.py:507`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/analysis.py#L507), [`scripts/run_confirmation.py:73`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_confirmation.py#L73)

**Evidence and impact:**
**Pages 4, 9–10, 15–16; methodological limitation.**

The report detects significant mean-model lack of fit and treatment-by-block interaction, yet interprets quadratic-model residual MSE as experimental noise in ICC and confirmation calculations. That residual includes structural lack of fit. HC3 addresses heteroscedastic coefficient covariance, but does not remove mean bias or account for all dependence from shared splits/seeds. A nominal interval containing one confirmation mean does not establish its coverage.

The failed depth-7 prediction is correctly disclosed. The remaining claims should be conditional on the approximation rather than treating the ICC or interval coverage as fully established properties of the experiment.

**Suggested fix:** distinguish model discrepancy, block effects, treatment-by-block variation and replicate noise; add local held-out configurations and an uncertainty analysis appropriate to the repeated/block structure. Report uncertainty on the variance components, especially with only five blocks.

### CR-008 [P2] “RMS misfit” is not the typical prediction discrepancy readers may infer

**Severity:** P2

**Status:** Open

**File/line:** [`analysis.py:317`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/analysis.py#L317), [`report.tex:246`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L246)

**Evidence and impact:**
**Pages 1, 9 and 19; misleading scale interpretation.**

The value 0.0297 is `sqrt(SS_LoF / df_LoF)`, a lack-of-fit mean-square scale. It is not the RMS difference between fitted and observed treatment means over the experimental observations. Dividing saved `SS_LoF` by the number of observations before taking its square root gives about 0.00795 for the full design and about 0.00185 for the restricted design. These are recalculations from `results/lof.json` and `results/runs.csv`, not new training results. These alternative calculations also contain sampling variation and are not estimates of pure structural bias.

**Suggested fix:** name the reported statistic precisely, give the denominator, and avoid interpreting it as an average prediction-error magnitude. Use independent prediction errors to quantify practical accuracy.

Evidence: [LoF calculation](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/analysis.py#L317), [stored fitted surface and coefficients](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/phase3.json).

### CR-009 [P2] The canonical eigenvector interpretation is wrong

**Severity:** P2

**Status:** Open

**File/line:** [`report.tex:309`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L309), [`results/phase3.json:41`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/phase3.json#L41)

**Evidence and impact:**
**Page 12; confirmed numerical interpretation error.**

The PDF says the smallest eigenvalue primarily follows `−0.98 x4`. The stored first eigenvector is approximately `(0.023, −0.053, −0.724, 0.687)`, so it mixes subsampling and regularization. The second eigenvector also mixes those factors. Eigenvector sign is arbitrary, but that does not reconcile these different magnitudes/directions. The text appears to confuse matrix rows with eigenvector columns.

The bootstrap supports uncertainty about definiteness; it does not uniquely prove a stationary/rising ridge instead of all alternative surface classifications.

**Suggested fix:** generate the direction descriptions from eigenvector columns and describe the ridge classification as provisional.

Evidence: [eigenvectors](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/phase3.json), [report statement](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L309).

### CR-010 [P2] The low-learning-rate explanation overstates the evidence

**Severity:** P2

**Status:** Open

**File/line:** [`report.tex:248`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L248), [`results/runs.csv:1`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/runs.csv#L1)

**Evidence and impact:**
**Page 9; confirmed contradiction plus unsupported mechanism.**

The report attributes dominant lack of fit to `x1 = −1`, but names `(1, 1, −1, −1)` as an example. That point has the highest learning rate, and recalculation identifies it as the largest individual LoF contribution. Dropping low-learning-rate observations reduces SS, but changes the sample and rank and does not establish the proposed underfitting mechanism.

**Suggested fix:** show a complete contribution plot and learning curves or a controlled tree-count experiment. Present the restricted fit as sensitivity analysis rather than causal confirmation.

Evidence: [stored fitted surface and coefficients](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/phase3.json), PDF Table 6 and accompanying paragraph.

### CR-011 [P2] The DMatrix-overhead explanation is incorrect

**Severity:** P2

**Status:** Open

**File/line:** [`report.tex:262`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L262), [`pipeline.py:238`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/pipeline.py#L238)

**Evidence and impact:**
**Pages 5, 10 and 19; confirmed API interpretation error.**

The report attributes the gap between sklearn `predict` and `Booster.inplace_predict` to DMatrix conversion. Installed XGBoost 3.2.0's sklearn implementation already uses `inplace_predict` by default on supported matching-device arrays, as used here. The comparison therefore does not isolate DMatrix construction; wrapper/configuration/validation overhead is a more plausible category requiring direct measurement.

**Suggested fix:** relabel the measured comparison and separately instrument DMatrix construction if that mechanism matters.

Evidence: inspected installed XGBoost source; [official release 3.2 API](https://xgboost.readthedocs.io/en/release_3.2.0/python/python_api.html).

### CR-012 [P2] Latency evidence is narrower than the performance claims

**Severity:** P2

**Status:** Open

**File/line:** [`scripts/run_benchmarks.py:108`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L108), [`scripts/run_benchmarks.py:432`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L432)

**Evidence and impact:**
**Pages 5, 10, 17–19; methodological limitation.**

The benchmark accuracy uses fresh retraining seeds, but its latency column comes from separately fitted models using seed 42 and repeated prediction of one validation row. The reported latency medians summarize five batches, not a distribution across those fresh models, representative input rows, or independent sessions. The PDF table omits the available timing IQRs, even while discussing small timing differences.

Only depths 3, 6 and 9 support the latency response fit. A quadratic fits three depth means flexibly; this is limited evidence for a general mechanistic growth law or behavior at intermediate depths. Failure to reject a latency interaction is also not proof of invariance across splits.

**Suggested fix:** measure latency for the same models used for accuracy, across representative rows and independent sessions; show uncertainty and validate intermediate depths.

Evidence: [latency harness](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L108), [benchmark summaries](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/results/benchmark.csv).

### CR-013 [P2] The efficiency and Pareto figures select favorable noisy realizations

**Severity:** P2

**Status:** Open

**File/line:** [`plots.py:297`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/plots.py#L297), [`plots.py:366`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/plots.py#L366)

**Evidence and impact:**
**Pages 15 and 18, Figures 4–5; methodological limitation.**

The DOE efficiency curve takes the cumulative minimum of single-run validation RMSE across different blocks. Baselines optimize one fixed split, so these vertical values are not a common evaluation target. The Pareto curve similarly filters individual noisy CCD runs, rather than block-aggregated configuration responses, and overlays confirmation/benchmark summaries obtained under different timing/evaluation conditions.

**Suggested fix:** construct configuration-level estimates, use a shared evaluation protocol, and show uncertainty. Repeat the entire DOE search if claiming method-level convergence variability.

Evidence: [individual-run Pareto filter](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/plots.py#L297), [DOE cumulative minimum](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/plots.py#L366).

### CR-014 [P2] Hypervolume does not compare the algorithms' Pareto fronts

**Severity:** P2

**Status:** Open

**File/line:** [`scripts/run_benchmarks.py:482`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L482), [`report.tex:447`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L447)

**Evidence and impact:**
**Pages 18 and 20; scope limitation.**

The report distinguishes single-point from two-point DOE hypervolume, which is an improvement. However, the comparator is one selected MO-TPE configuration, not its full nondominated set. The point hypervolume arithmetic does not establish superiority of DOE as a multi-objective optimizer. The conclusion also depends on the chosen reference point and uncertain latency measurements.

**Suggested fix:** retain and evaluate nondominated sets under a common protocol, report per-search hypervolume distributions, and check reference-point sensitivity.

Evidence: [hypervolume inputs](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L482).

## Smaller but definite reporting issues

### CR-015 [P2] The stated search-grid resolution does not match the available implementation

**Severity:** P2

**Status:** Open

**File/line:** [`scripts/run_benchmarks.py:537`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L537), [`report.tex:355`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L355)

**Evidence and impact:**
**Page 14.** The PDF describes a 6,174-coordinate grid, while the current sensitivity routine enumerates `21 × 7 × 2 × 2` coordinates and uses only `x3,x4 ∈ {0,1}`. No current routine reproduces the stated finer grid or derives the selected operating point from it. The omitted negative halves of these factors also limit the sensitivity conclusions.

**Suggested fix:** retain the actual optimization routine and results, report its true domain/resolution, and use the same optimizer for each sensitivity scenario.

Evidence: [grid construction](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/run_benchmarks.py#L537).

### CR-016 [P2] The introduction overstates what black-box optimization cannot do

**Severity:** P2

**Status:** Open

**File/line:** [`report.tex:52`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L52)

**Evidence and impact:**
**Page 2.** Claims that these methods cannot supply sensitivity/variance attribution are too categorical. Optuna provides importance analysis, including fANOVA; search algorithms can also use repeated or blocked objectives. Distinguish capabilities of the baselines actually implemented from limitations of the entire method class.

Evidence: [official Optuna fANOVA documentation](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.importance.FanovaImportanceEvaluator.html).

**Suggested fix:** narrow the introduction to the capabilities of the implemented baselines, and acknowledge existing sensitivity-analysis and repeated-evaluation approaches.

### CR-017 [P3] Several labels and legacy artifacts make the project harder to interpret correctly

**Severity:** P3

**Status:** Open

**File/line:** [`report.tex:65`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/report.tex#L65), [`config.yaml:63`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/config.yaml#L63), [`scripts/generate_report_artifacts.py:611`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/scripts/generate_report_artifacts.py#L611), [`REPORT.md:1`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/REPORT.md#L1), [`CHANGELOG.md:1`](https://github.com/ChristosHussein/xgboost-doe-rsm/blob/caf3a5bf59f2eae2541d3478d5fc370f0b339717/CHANGELOG.md#L1)

**Evidence and impact:**

- **Pages 1 and 3:** “four continuous factors” is incorrect: maximum depth is integer-valued.
- **Pages 4–5:** the PDF states 100 warmup iterations, whereas configuration and benchmark code use 50 calls per prediction interface. If 100 counts both interfaces, explain that explicitly.
- **Page 16, Table 10:** label the depth-7 prediction as “not confirmed”; “Empirical Opt” obscures the failed prediction and is not proof of an empirical optimum.
- **Pages 1–2:** the abstract is unusually long and extends onto a second page; shorten it to the question, design, principal findings and most important limitation.
- **Project:** `REPORT.md`, `results_summary.json` and `data/` preserve a materially different analysis; `CHANGELOG.md` contains claims already corrected in the PDF. Clearly archive/label these so a reader cannot mistake them for the current evidence.

**Suggested fix:** correct factor and timing descriptions, label failed confirmation explicitly, shorten the abstract, and clearly archive superseded documents and results.

## Verification record

Checks below were performed during the audit of the same source revision. The current eight-test suite was also rerun for this documentation handoff. No full training/optimizer rerun was performed.

- `python -m pytest tests/ -q`: all eight tests passed.
- The legacy top-level runner fails importing `DataManager`; explicit collection of the legacy verification module fails importing `code_factors`.
- Recomputed saved Phase 1, lack-of-fit and diagnostic numeric fields matched within the tolerances used in this review.
- Independently recomputed Satterthwaite RMSE intervals agree with the current printed values. They are still hardcoded in the generator.
- Provenance check: failed because `results/MANIFEST.json` is absent. This means source-to-result freshness is unverified, not that measurements are necessarily false.
- Lexical number audit of extracted PDF text: 1,383 tokens checked, 1,260 matched values somewhere in results/configuration, 123 unmatched. This is a screening result only: unmatched tokens include bibliographic/protocol/math/PDF-extraction cases, and numeric matches do not prove semantic provenance. It is **not** a count of 123 report errors.
- Full optimizer searches and training experiments were not rerun; raw per-search incumbents and fresh-seed benchmark vectors are not retained by the current benchmark output code, preventing direct reconstruction of the reported paired tests from stored summaries.

The first fixes should be the baseline latency objective, incumbent aggregation, execution/selection claims, and reproducible artifact generation. Those affect the main conclusions more than presentation edits.
