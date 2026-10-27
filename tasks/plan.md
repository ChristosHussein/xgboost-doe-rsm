# Complete the research programme

Source requirements: the author's Remarks,.txt and Research approach.txt.
These documents are requirements, not scientific evidence. Their historical
description of online timing as unwarmed is contradicted by saved warmup metadata.

## Acceptance criteria and execution order

- Preserve frozen housing evidence and the original experimental factor identities.
- Record real dataset sources, versions, preparation, duplicates, partitions and hashes.
- Calibrate learning-rate adequacy, tree count, latency scale and runtime on development data only.
- Freeze the full measurement contract before confirmatory search and final testing.
- Compare fixed CCD, sequential CCD and TPE at the same original per-search fit budget.
- Use shared development blocks and identical warmed timing; account separately for calibration, confirmation, explanations and final evaluation.
- Analyze ANOVA main effects, interactions and curvature against exact surrogate SHAP; preserve before/after and between-block variability, including undefined results.
- Confirm polynomial and GP predictions on reserved interior and boundary coordinates; report errors, bias and interval coverage with their dependence limitations.
- Compare Pareto quality, observed-reference regret, fit-count progress, total runtime, feasibility and stability across searches and datasets.
- Explain actual sentiment predictions through local RSM word perturbations, genuine pairwise SHAP, independent perturbation checks and explicit non-causal scope.
- Reconstruct all published numbers from saved observations; preserve unfavorable outcomes and uncertainty.
- Integrate real findings into the existing manuscript structure, rebuild and inspect PDFs, and run full scientific/software checks before opening a PR.

## Design decisions

The full fixed CCD uses its original center replication. The sequential variant
starts with fewer center replicates and spends the released fits on adaptive
points. This preserves the total fit budget while retaining an identifiable
quadratic model. This trade-off must be reported, not concealed.

Word-weight perturbations are conditional model explanations, not fluent edited
sentences or causal effects. They are separate from hyperparameter explanations.
Repeated development folds share observations; their variability does not imply
independent dataset replications. Dataset outcomes will be reported separately.

Real datasets approved by the author: IMDb and a second public sentiment dataset,
using local CPU execution after calibration. Selected secondary application:
UCI Sentiment Labelled Sentences, Yelp subset. Calibration and pilot runtime must
be reported before the full execution starts. The pre-run contract must state
sampling, seeds, endpoints, confidence procedures and stopping rules. Historical
housing outputs will never be overwritten.
