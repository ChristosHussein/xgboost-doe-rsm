# Our experiment: purpose, design, and staged execution

**Review draft — 28 October 2026. No experiment is launched by this document.**

This is the proposed description of the work we will do before committing to a long run. It brings together the original project, the teacher's remarks, and the additional research approach. It separates the intended experiment from limitations of the current implementation.

**Your latest decision is incorporated: run a subset first to save time, review what we learn, and only then consider the full dataset. Full-data execution is a separate decision.**

All budgets and settings below are proposed design parameters or arithmetic, not observed results. Existing historical results, calibration files, and partial campaigns are evidence with their own history; this document does not retroactively preregister them.

## 1. What are we trying to find out?

We are studying whether a structured, explainable experimental design can find useful XGBoost hyperparameters efficiently, explain how those hyperparameters affect prediction quality and latency, and predict the performance of configurations it has not yet evaluated.

The central contribution is **explainable sequential DOE**: start with a designed experiment, analyze effects and interactions, use a response model to choose additional experiments, and independently check whether its predictions and explanations are reliable. Benchmark optimizers tell us whether this structure costs too much or helps us find useful models.

We are not promising that DOE wins. A useful scientific outcome may be that DOE explains effects well but loses on search efficiency, that the quadratic surface is unreliable near a boundary, or that ANOVA and SHAP disagree for understandable reasons.

There are three different models in this study:

| Model | Inputs → output | Role |
|---|---|---|
| XGBoost sentiment classifier | Text features → sentiment probability | The expensive model whose settings we optimize |
| Hyperparameter surrogate: quadratic RSM or Gaussian process (GP) | Hyperparameter configuration → classification loss or latency | Predict performance, explain hyperparameter effects, and guide experiments |
| Local word-level RSM | Selected word-feature weights → a fixed classifier's probability | Explain a particular document locally |

SHAP of the hyperparameter surrogate and SHAP of the sentiment classifier answer different questions. Their results must remain separate.

## 2. How the three source documents fit together

The documents supply project context and requirements to reconcile. Their recommendations do not independently authorize running experiments, changing the scope, or increasing the budget.

| Source | What it contributes | What we retain |
|---|---|---|
| `Original specs.txt` | DOE for efficient hyperparameter selection; factorial designs and response surfaces; comparison with grid/random search; statistical versus ML modeling and XAI | Efficiency remains a research question. We compare a classical response model with a flexible ML surrogate and retain a conventional search comparator. Random Forest is an example in the original text, not a requirement to change away from XGBoost. |
| `Remarks,.txt` | Correct underfitting, timing inconsistency, and exposed holdout; investigate interactions/curvature with ANOVA–SHAP; move to sentiment; include local word explanations and modular outputs | These become explicit experimental checks and deliverables, not simply discussion paragraphs. |
| `Research approach.txt` | Initial CCD → ANOVA/SHAP → adaptive augmentation → independent confirmation; equal-budget DOE/TPE comparison across splits and datasets | This supplies the overall workflow. We distinguish new hyperparameter coordinates from new test documents and measure both kinds of generalization. |
| Your instruction in this chat | Understand the complete experiment before a long session; run a subset before the full dataset | This document is the review point. The subset is a pilot; no automatic promotion to full data. |

Two translations need to be explicit. First, sentiment is classification, so the primary accuracy response becomes **1 − macro F1**, rather than housing-price RMSE. Second, the planned CCD contains factorial and axial points and supports hierarchical analysis, but it is not a demonstrated sequential screening procedure unless a screening decision actually controls what is run next. We will not relabel a preplanned batch as adaptive.

## 3. The questions and the evidence needed to answer them

| ID | Research question | Required evidence |
|---|---|---|
| Q1 — explanation | Do ANOVA and surrogate SHAP agree about important factors, interactions, and curvature? When do they disagree? | Effect tables, interaction matrices, curvature profiles, and agreement summaries on the same responses and declared reference distribution; stability across blocks and search repeats |
| Q2 — prediction | Can the surrogates predict performance at previously unmeasured configurations, including boundaries? | Predictions fixed before confirmation outcomes; observed-minus-predicted bias, prediction RMSE, interval coverage and width, separately for interior and boundary points |
| Q3 — adaptation | Does adding adaptively chosen points improve prediction, explanation stability, or optimization compared with static CCD? | Initial and final explanation snapshots; fixed independent confirmation; equal-budget static/adaptive comparison and progress versus actual XGBoost fits |
| Q4 — efficiency | What trade-off do DOE, TPE, and conventional random search achieve in accuracy, latency, time, and stability? | Comparable development Pareto fronts, hypervolume, observed-reference regret, common-target fit counts, runtime and feasibility; final frozen selections evaluated later |
| Q5 — local explanation | Can a local quadratic model explain a classifier's response to selected word-feature changes, including interactions? | Actual classifier predictions on designed perturbations and unused confirmation perturbations; local RSM fidelity and pairwise SHAP values |

Q1–Q3 are the methodological center. Q4 provides the performance benchmark. Q5 is the teacher's application-level extension and has its own evidence, rather than being conflated with Q1.

## 4. Execution has separate stages

### Stage A — readiness without a long training run

Reconcile this protocol with the code, freeze data identities and split assignments, check the data-exclusion rules, and run the software tests. Synthetic smoke tests can exercise final-evaluation access controls without touching the real final test outcomes. They are not scientific results.

Before any pilot, show its exact configuration, command, output location, fit budget, runtime estimate, and stop/resume behavior. A changed learning-rate range, sample size, endpoint, or comparator is a protocol revision, not an invisible implementation choice.

### Stage B — subset pilot

**Purpose:** establish that the entire development workflow works and that its measurements are useful before paying for full-data execution.

Proposed starting point: a reproducibly selected, stratified **2,000-document IMDb development subset**, with the document IDs and selection seed recorded. This size reflects the existing candidate setup; it is not a claim that this sample is scientifically sufficient. Do not pick documents using model performance. Keep official test documents outside pilot scoring.

Use the same factor definitions, preprocessing rules, timing operation, and method budgets intended for the main comparison. Start with **one search repeat** per method to limit cost. It can test orchestration and yield exploratory observations, but cannot establish between-search stability. Additional pilot repeats require an updated budget.

The pilot includes calibration, blocked search, ANOVA–SHAP analysis, reserved-coordinate confirmation, and the local explanation demonstration. It produces a pilot report, not final population-level conclusions.

**No real final-test evaluation in this stage.** Confirmation measures development performance at unseen hyperparameters; it does not require the official test set. The final-evaluation gate is tested with fixtures. If an internal pilot holdout is later wanted, define it wholly inside development data before scoring it, and label it as pilot-only.

### Stage C — review the subset findings

Review runtime, record completeness, fit failures, learning adequacy, timing behavior, surrogate rank/fit diagnostics, boundary prediction errors, and explanation outputs. Preserve unfavorable results. Any changes motivated by the pilot are documented as pilot-informed choices.

A pilot succeeds operationally when it gives a trustworthy account of what happened, including failures. We do not require favorable rankings, small p-values, or agreement between ANOVA and SHAP to continue. Conversely, a scientifically disappointing but valid result must not be rerun until it looks better.

### Stage D — full-data study, only after a separate decision

Use the full retained IMDb development pool, with an untouched official test partition if the exposure audit supports that description. Include the UCI Sentiment Labelled Sentences Yelp subset as a separately reported second application. It is a small secondary dataset, not the full Yelp review corpus.

Recheck model capacity and timing calibration on full development data: the subset's training cost, vocabulary and adequate tree count need not transfer. Freeze the full protocol after this calibration and before main search. Evaluate every approved method under the same per-dataset conditions.

The present candidate uses **three search repeats and five development folds**. Three repeats support a modest descriptive comparison; they do not provide a strong equivalence study. If a stronger inferential claim is required, justify and budget the repeat count before execution rather than increasing it after inspecting results.

## 5. Data separation and what “generalization” means

1. Record source version, file hashes, normalization, duplicate handling, exclusions, and retained IDs. Detect duplicate documents across development/test partitions without using performance to choose exclusions.
2. Protect final test outcomes before calibration. Data-integrity checks and declared stratification are distinct from inspecting test performance.
3. Within development data, use five stratified folds as nuisance blocks. Each method evaluates each proposed configuration on every fold. Fit TF-IDF vocabulary and weights only on that fold's training documents.
4. Use identical block assignments across methods. Repeating search seeds on those same folds is not repeating independently sampled datasets; overlapping training folds are dependent.
5. Reserve hyperparameter confirmation coordinates before search. Never use their observed responses to fit the surrogate, guide adaptive selection, or choose the final classifier.
6. Freeze selections and the training configuration before final-test scoring. Refit preprocessing on the allowed full development pool and train the frozen classifiers there.
7. After inspecting final-test results, do not tune and score again on the same set while continuing to call it untouched.

**Two separate checks:** reserved hyperparameter coordinates test surrogate generalization within the declared domain; reserved documents test classifier generalization. A hash verifies identity, not historical non-exposure. The old California Housing holdout cannot be made fresh by a new access gate.

## 6. Model, factors, and responses

Use XGBoost binary classification on TF-IDF text features. The candidate preprocessing uses unigrams and bigrams, sublinear term frequency, and a maximum vocabulary of 10,000 features. Keep preprocessing fixed across methods within each dataset and record the actual vocabulary for each fold.

Retain the four factor identities from the existing housing study:

| Factor | Proposed low / center / high | Treatment |
|---|---|---|
| Learning rate | 0.05 / 0.175 / 0.30 | Linear coding; provisional until development-only calibration |
| Maximum depth | 3 / 6 / 9 | Integer; intermediate adaptive/TPE/random settings must be canonicalized to the actually trained integer |
| Row subsampling | 0.50 / 0.75 / 1.00 | Linear coding |
| Leaf-weight regularization, `reg_lambda` | 0.10 / 1.00 / 10.00 | Log coding |

Do not silently substitute `colsample_bytree` for regularization. Tree count is calibrated and then held fixed within a dataset, not introduced as a fifth search factor without a protocol amendment.

The existing calibration candidate compares 100, 300 and 600 trees at the low and central learning rate, with other factors centered. It chooses the smallest tree count whose low-rate macro F1 is within 0.005 of the largest tested count. This is a limited operational check: it cannot prove convergence throughout the domain. If learning is still materially improving at the largest count, or low-rate/high-regularization combinations fail badly, review the capacity/range before a long run. Record the resolution rather than automatically declaring underfitting solved.

Primary responses are validation **1 − macro F1** and warmed **microseconds per document**. Preserve log loss as a secondary classification metric. A probability RMSE may be secondary, clearly named as such; it is not the housing RMSE endpoint. Use the declared classification threshold of 0.5 consistently.

## 7. Methods and fair budgets

For four factors, the face-centered CCD has 16 factorial and 8 axial design entries. Center repetitions are additional fits, not additional unique configurations.

| Method | Proposed schedule per search repeat | XGBoost fit budget |
|---|---|---:|
| Static CCD/RSM | 16 factorial + 8 axial + 4 center entries, each on five blocks | 140 |
| Sequential CCD/RSM | 16 factorial + 8 axial + 2 centers, then two adaptive cycles with one new coordinate per cycle, each on five blocks | 130 + 5 + 5 = 140 |
| Multi-objective TPE | 28 proposed configurations, each evaluated on five blocks; return block-mean loss and latency to TPE | 140 |
| Random search — proposed addition | 28 configurations from the same coded domain, each evaluated on five blocks | 140 |

**Scope decision:** the current runner implements the first three methods. Adding random search is recommended to retain a direct comparator from the original question. Grid search is not also required for this proposed minimal design. If random search is omitted to save time, explicitly narrow the study question to DOE versus TPE; do not claim to have answered the original random-search comparison.

Sequential DOE exchanges two center repetitions for two adaptive coordinates. This preserves the fit budget but reduces replication and gives only a small amount of adaptation. It is a modest sequential extension, not a broad test of all adaptive DOE strategies.

Freeze one acquisition rule. The current candidate is a hybrid of normalized GP predictive uncertainty and distance from existing points: favor informative points while avoiding near-duplicates. It is not expected hypervolume improvement. Alternative criteria are separate experiments, not options to try until one wins.

Use shared folds, consistent training resources, identical timing, and recorded seeds for all methods. Randomize method order across repeats and record evaluation order. All attempted model fits, including failed fits and retries, count in the cost ledger. A method must not obtain unreported extra observations.

## 8. How ANOVA and SHAP will be compared

Fit a full quadratic response surface with block effects separately for classification loss and latency. Retain main effects, pairwise interactions, and quadratic terms. Report fit diagnostics, partial sums of squares, model-conditional tests, and replicated-cell lack of fit where estimable.

Fit a GP to configuration-level mean responses as the flexible alternative. Keep its prediction target distinct from the RSM's block-specific prediction target.

Compute surrogate SHAP and genuine pairwise SHAP interactions on a frozen common background and common explanation coordinates. The current candidate uses the unique initial CCD coordinates as background and the reserved confirmation coordinates as explanation locations. Knowing those coordinates is allowed; using their measured responses during search is not.

Compare factor rankings, the six pairwise interactions, and curvature profiles. Show RSM and GP explanations, initial/final adaptive snapshots, and block-specific variability. Preserve undefined ranks/correlations when effects are constant.

ANOVA significance and SHAP magnitude are different quantities. Total SHAP importance includes interaction contributions, so it is not automatically a pure-main-effect comparator. Label that comparison and also examine the SHAP interaction diagonal/main-effect profiles. Compare curvature through declared low/center/high profiles and second differences; do not equate their numerical units with ANOVA sums of squares. Agreement alone does not prove either surrogate is accurate: Q2 supplies the independent predictive check.

Adaptive sampling and dependent folds limit classical inferential interpretation. Treat post-adaptation significance and small-sample ranking correlations cautiously and descriptively; do not turn a collection of nominal p-values into a universal explanation claim.

## 9. Confirmation and final model selection

Reserve 12 new coordinates before each dataset's searches and use the same set for all methods. Evaluate each on five blocks after fitting the final surrogates. The intended coverage must include interior points, learning-rate boundaries, and intermediate depth settings relevant to the earlier boundary problem. Save the actual coordinates; “random points” is not sufficient assurance that the problematic region was tested.

Freeze surrogate predictions and intervals before observing confirmation responses. For sequential DOE, save these predictions both before and after augmentation on the same reserved coordinates; this allows a direct check of whether adaptation improves predictive accuracy without collecting a second confirmation set. Compute prediction RMSE, signed bias, interval coverage and interval width. Report interior and boundary results separately. RSM fixed-block prediction intervals and GP intervals for block-mean responses need separate tables with their exact targets and assumptions; neither automatically establishes coverage on new datasets.

Confirmation is not additional search budget. Do not use it to pick a more favorable candidate. The current development selection rule chooses the observed configuration with the largest dominated rectangle to a frozen loss/latency reference point. Define a deterministic tie rule before running. This is a multi-objective compromise, not a hard-constrained optimum.

On full data only, freeze one selection per method/search repeat before final test scoring. Report test macro F1, log loss and the declared latency measure for all frozen selections. These selected points are not a fully re-evaluated test Pareto frontier. Claiming test-front superiority would require a separately budgeted, frozen front-evaluation protocol.

## 10. Latency and feasibility

Use the same primary operation and counts in search, confirmation and final verification: **100 warmup calls followed by 500 timed calls**, single-document input, single-thread prediction, and recorded CPU affinity. Retain all raw durations.

The primary operation is fresh `DMatrix` construction plus `Booster.predict`. Record `inplace_predict` separately. Neither includes raw-text TF-IDF transformation, so this is not end-to-end application serving latency. Training threads may differ from inference threads, but must be fixed and recorded across methods.

The current code times the first validation row of each fold. That is a narrow input-specific measure, not evidence for a representative document-latency distribution. Before a stronger deployment claim, define a deterministic multi-document timing panel and repeat-session protocol, then recalibrate and budget it consistently for every method. Do not silently change that operation only at verification.

Freeze each dataset's latency threshold and hypervolume reference before main search. The existing candidate derives them from development calibration: median calibration latency for the threshold; loss reference 1 and twice the maximum calibration median latency for the latency reference. These are design choices, not external service requirements.

Report both the fraction of searched configurations below the threshold and whether each frozen selected model remains below it during verification. Use separate denominators. The existing MO-TPE method is not Constrained TPE; we cannot claim a direct replication or resolution of the historical Constrained-TPE feasibility result without adding that experiment. Matching the measurement protocol addresses a source of inconsistency, not a guarantee of feasibility.

## 11. Local word explanations

Keep this small, reproducible, and separate from hyperparameter analysis. Use a development-trained classifier selected by a fixed rule, then the first ten eligible development-validation documents. Choose each document's four highest-weight present unigram features without looking at its label or explanation quality.

For each document, vary those four TF-IDF weights using a local CCD, fit a quadratic response surface to the classifier's positive-class probability, and measure fidelity on 64 unused perturbation points. Compare its interaction description with exact pairwise SHAP of the fixed classifier under the declared feature-attenuation game.

Keep other feature weights fixed and do not renormalize. This means the inputs are conditional feature-vector perturbations, not necessarily natural rewritten sentences. Bigrams can remain present when a unigram weight changes. The conclusions concern model behavior, not causal effects of words. Repeated deterministic center predictions do not provide independent sampling noise for significance claims.

Save the classifier, selected vectors, feature names, perturbation coordinates, probabilities, coefficients and attribution values so the explanation can be replayed. Count classifier training and explanation prediction calls separately.

## 12. Budget and runtime before we start

These counts follow the proposed schedule; they exclude calibration fits and failed/retried attempts, which must be added separately.

| Scope | Search fits | Confirmation fits | Final test fits | Local-explanation training | Total |
|---|---:|---:|---:|---:|---:|
| IMDb subset pilot, one repeat, current three methods | 3 × 140 = 420 | 3 × 12 × 5 = 180 | 0 | 1 | **601** |
| IMDb subset pilot, one repeat, with random search | 4 × 140 = 560 | 4 × 12 × 5 = 240 | 0 | 1 | **801** |
| Full study per dataset, three repeats, current three methods | 3 × 3 × 140 = 1,260 | 3 × 3 × 12 × 5 = 540 | 9 | 1 | **1,810** |
| Full study per dataset, three repeats, with random search | 4 × 3 × 140 = 1,680 | 4 × 3 × 12 × 5 = 720 | 12 | 1 | **2,413** |

The two-dataset full-study totals would therefore be 3,620 or 4,826 fits respectively. These are alternatives, not two campaigns to execute. The random-search alternative assumes it receives the same surrogate/confirmation analysis; changing that choice changes the budget.

Even a subset run can take substantial time: a smaller dataset does not eliminate the number of fits or the cost of SHAP and GP fitting. Before execution, estimate training/evaluation time across representative shallow and deep configurations, profiling time, surrogate/XAI overhead, preprocessing, final refits, local explanations and report generation. Present a range and its assumptions, not a center-point extrapolation as a guaranteed finish time.

The current calibration summaries explicitly exclude surrogate/explanation overhead and depth variation. They are not yet a defensible total-session estimate. **Runtime ceiling and approved wall-clock range remain unset.** Do not start a long session until these are supplied and accepted.

If the budget would be exceeded, stop at a safe recorded boundary. Keep partial logs and the cause. Do not silently remove methods, blocks, confirmation points or failed trials to finish faster. A safe resume must preserve the original protocol and identify completed fits; current append-only logs alone are not proof of implemented resume support.

## 13. Measurement and reporting contract

Every record needs dataset/cohort hash, protocol version, stage, method, repeat, configuration in coded and natural units, block, model seed, timing settings and execution status. Analysis must filter by those identities, never combine pilot and full results by filename similarity.

The paths below describe the existing output interface or planned additions, not a claim that a completed new study exists. `RUN` means the newly approved output directory.

| Metric ID | Capture/source selector | Derivation and reporting |
|---|---|---|
| `quality.macro_f1` / `quality.logloss` | `RUN/evaluations.jsonl`, labels/probabilities and response fields; filter method/repeat/block/stage | Reconstruct per-fit metrics; average equally across blocks for configuration comparisons. Final test uses `RUN/final_results.json` only in full stage. |
| `timing.classic_us` / `timing.inplace_us` | Per-fit `latency.predict_raw_us` and `latency.inplace_raw_us`, protocol and machine metadata | Per-fit median and distribution summaries; primary configuration response is the block mean of classic medians. No pooling into fictitious independent model replicates. |
| `optimization.hypervolume` | Search-only complete configurations in `evaluations.jsonl`, frozen reference in `measurement_contract.json` | Nondominated development front per method/repeat; minimization of loss and latency; reconstruct area independently. |
| `optimization.regret` | Same development rows | Best method loss minus best observed loss across approved methods in the same dataset/repeat; observed reference, not the unknown optimum. |
| `optimization.fits_to_target` | Recorded fit order and complete-block configuration aggregates | First fit count reaching 95% of the best observed development hypervolume within the comparison; a common retrospective reference with a predeclared formula. Report not reached as such. |
| `feasibility.search` / `feasibility.selected_verification` | Search rows; frozen selections and their verification rows; threshold in contract | Distinct numerator/denominator for searched configurations and frozen selections; retain unfavorable threshold crossings. Add a selected-verification summary if missing. |
| `surrogate.bias` / `rmse` / `coverage` / `width` | `confirmation.json`, `gp_confirmation_*.json`, and raw confirmation records | Signed observed−predicted errors, root mean square error, fraction inside fixed intervals, mean interval width; separate response, target and boundary stratum. Extend capture/summary where absent. |
| `explanation.agreement` / `stability` | `snapshot_*.json`, stored ANOVA, SHAP values/interactions, background and explanation points | Rank correlations and interaction comparisons with defined/undefined counts; before/after and per-block summaries. |
| `local.fidelity` / `interactions` | Word-study contract, model, vectors and `observations.json` | Reconstruct perturbation probabilities, independent-point RMSE/max error, and declared interactions. |
| `cost.fits` / `cost.seconds` | Attempt ledger, `elapsed_seconds`, profiling fields, `overhead_*.json`, programme start/finish and failure records | Separate calibration/search/confirmation/final/explanation fits and all wall-clock components. Current `training_seconds` includes validation prediction/scoring, so do not label it pure training time. |

For full-study repeated-search summaries, show every repeat, mean and sample SD, and paired differences using matching dataset/repeat IDs. If bootstrap intervals are used, resample at search-repeat level with the seed and repetitions frozen; with three repeats they are descriptive and limited. Do not treat folds, timing calls, or the two datasets as independent population replications. Do not infer equivalence from a nonsignificant difference.

Before reporting, bind each table, figure, caption and numerical statement to its metric and exact dataset/method/stage. Planned main outputs are: ANOVA tables; factor/interaction/curvature comparisons; development Pareto and progress plots; independent-confirmation error/coverage plots; runtime and fit ledgers; final selected-model results; local-word fidelity examples; and a limitations/negative-results section. Use one validated reporting layer for Markdown, LaTeX and PDFs.

The repository already has scientific audits and GitHub checks. Apply the scientific-integrity and research GitHub Actions workflow to evidence reconstruction and release; green software tests alone do not certify scientific correctness. Long hardware-sensitive measurements remain separately authorized local runs, not automatic CI work.

## 14. What must be reconciled with the current code

This is a proposal, not a declaration that implementation is ready. The read-only inspection identified these concrete pre-run items:

1. **Protect the full-study test set during the subset pilot.** `run_study` currently invokes the real final evaluator when it is supplied, and research mode currently requires it. A development-only research-pilot mode/gate must exist before using that runner for this plan. Running the existing subset programme unchanged would score the official test set.
2. **Resolve the original comparator.** Random search is proposed here but absent from the current three-method runner. Implement it or explicitly approve the narrower question.
3. **Apply the subset budget.** Existing campaign configurations use three repeats; this draft proposes one initial subset repeat to save time. Do not reuse the existing configuration unmodified.
4. **Audit data exposure and previous execution.** Calibration and partial campaign directories already exist under `results/sentiment_revision_v1/`. Saved `RUNNING` status is not evidence that a process is still active, nor proof the test was never accessed. Inspect final-output/access history before describing any future test as fresh. Preserve old runs and do not backdate this protocol.
5. **Freeze confirmation coverage and interpretation.** Ensure the reserved set includes the scientifically relevant boundaries/intermediate depths; persist initial as well as final surrogate predictions; add interior/boundary coverage and width summaries where the current reporting is narrower. State the GP interval target explicitly.
6. **Set timing scope and feasibility claims.** Decide whether the first-row measurement suffices for the scoped study or whether a representative panel is required. Do not describe MO-TPE as a hard-constrained TPE experiment.
7. **Complete the run ledger and recovery story.** Capture attempts/failures, exact seeds, selected timing input IDs, source/environment identity, wall-clock ceiling and tested resume or restart behavior before launch.
8. **Check the full-data route.** The current programme wrapper assumes an IMDb cohort manifest in its metadata path. Verify the full-dataset input/manifest path before claiming the same command can simply scale up.

No code is changed by this description. These gaps are visible so we do not spend hours running an experiment different from the one reviewed here.

## 15. The agreement before any long session

The first page of the run handoff should answer all of these in plain language:

- **Question:** Which of Q1–Q5 does this run answer, and which answers remain exploratory?
- **Data:** Exact dataset/subset IDs; sample size; test-exposure history; protected final-test policy.
- **Design:** Factor domain and tree count; methods; blocks; repeats; adaptive rule; confirmation coordinates; selection rule.
- **Measurements:** Primary responses and timing operation; feasibility threshold and hypervolume reference; uncertainty and failure rules.
- **Cost:** Exact fits by stage; estimated wall-clock range; maximum authorized time; checkpoint/recovery behavior.
- **Outputs:** Output directory, evidence contract, planned tables/figures and independent verification commands.
- **Decision:** User-approved protocol version and command. No automatic full-data run after the subset.

**Current review choices:** retain or omit random search explicitly; confirm the proposed IMDb subset and one-repeat pilot; choose the latency input scope; set a runtime ceiling after a complete estimate. Full-study repeat count and final dataset scope remain separately reviewable after the pilot.

Any later change to those items gets a dated amendment stating what changed, why, which observations had already been seen, and the cost/conclusion implications. A good-looking outcome is not permission to change the question.

## Appendix — source identity and inspection limits

Source document SHA-256 values at drafting:

- `Original specs.txt`: `143b55642ad25574cb4ecf5633c62022bba6f10ad425c6c398a37ba72e8c9690`
- `Remarks,.txt`: `bd437a6f0dbbc7b0621ede41a202fd1b38fe12a46747277fca91fdac5c9dc015`
- `Research approach.txt`: `88260a24e77d6b0e1e9567888f115aac43978c1c988087675f9ce4fd4e181062`

Code inspected included `scripts/run_research_programme.py` and the `xgb_doe_xai` configuration, partitioning, design, study, profiling, surrogate, explanation, final-evaluation and reporting modules. The checkout reported commit `be481e5cfde6c9fc41a1e59e41e6b5c9a94ab16e` with local changes; this is a working-tree inspection, not an immutable release audit.

Existing calibration summaries and campaign contracts were inspected for design context only. No empirical performance claim is independently validated by this document. No model training, calibration, dataset download, final-test scoring or full campaign was performed to prepare it.
