# Protocol for a new comparative replication

This is a prospective design proposal, not a record of completed experiments. The published campaign remains frozen. The added saved-evidence analyses are retrospective and cannot repair test exposure or create independent optimizer repeats.

## Questions and outcomes

Separate model selection, global response prediction, and explanation accuracy. The selection question is whether a method improves a prespecified meaningful difference in held-out macro F1 under an operational latency requirement and comparable search cost. The response-model question concerns errors and interval coverage at independently reserved coordinates. Explanation agreement is a diagnostic; it is not ground-truth attribution accuracy.

Before running, freeze the primary endpoint, meaningful effect and any equivalence margin, external latency requirement, preference rule, reference point, hardware, software, data identities, exclusion rules, search allocation, and stopping rule. Capacity matching provides a calibration reference; it does not supply an application requirement.

## Data and independent replication

Use an independently controlled test collection that has not informed configuration, preprocessing, thresholds, or report revisions. Freeze every selection before accessing it. Retain text identities and deduplication records. Group by movie where reliable identifiers are available; otherwise explicitly limit the independence claim. Fit vocabulary and TF-IDF weights only within training partitions.

Pair methods on the same outer development partitions, seeds, and test documents. Repeat searches within outer partitions so that optimizer randomness and partition variation can be distinguished. Overlapping inner folds are nuisance blocks, not independent datasets. Choose the number of outer partitions and searches using a development-only pilot or simulation for the prespecified effect and desired precision/power; document assumptions before starting. The completed campaign alone cannot justify a universal replication count.

## Matched methods and budgets

Include static CCD, a sequential design with a meaningful prespecified adaptive allocation, MO-TPE, and random search. Add default XGBoost and a TF-IDF linear classifier as task-level reference models, with their own declared resource accounting. Compare both actual fit budgets and cumulative wall-clock budgets, including preprocessing and method overhead under a declared accounting rule. Report startup allocation and completed unique coordinates explicitly. These additional arms and stopping rules still require implementation before a new campaign.

For global response prediction, compare quadratic and GP surrogates under each sampling method, and include a space-filling design where resources permit. Choose common reserved coordinates without using outcomes and score both model families on the same targets. Report coordinate-level and boundary errors separately from local selection quality. If studying SHAP-guided acquisition, implement a separate arm whose acquisition equation actually uses attribution information; uncertainty-only acquisition cannot answer that question.

## Measurement options now available

The direct Phase B runner supports these opt-in extensions while retaining historical defaults:

- `--threshold-mode capacity_matched` uses calibration records at the selected tree capacity.
- `--latency-panel-size` selects a label-free panel across text-length strata and cycles single-document predictions over it. Repeat independent timing sessions and report session variation; a panel alone does not measure repeatability. Keep model-only prediction separate from end-to-end text vectorization.
- `--consistent-tfidf-perturbations` couples active bigrams to selected unigram attenuations and normalizes the feature vector. This remains a feature-space game. Add matched TreeSHAP/LIME comparisons only after defining compatible targets and perturbations.

Use matching panel settings for calibration and the study. Supplemental contracts and calibration snapshots are written before evaluations. Store each replication in a fresh output directory. These extensions have synthetic checks; no real-data findings from them appear in the current paper.

## Analysis and release

Preserve every observation and unsuccessful search. Report per-search outcomes, uncertainty over independent searches/outer partitions, paired document uncertainty conditional on frozen models, and timing-session variability separately. State how multiplicity and dependence are handled. Do not interpret a nonsignificant difference as equivalence. Evaluate surrogate uncertainty empirically rather than assuming a working quadratic interval is calibrated.

Generate numerical report content from registered evidence and verify method/response/target bindings. Publish the immutable data manifests, protocol, exact code revision, analysis environment, and independently rebuilt report together. A subset pilot checks execution and development calibration; it cannot substitute for the prespecified confirmatory study.
