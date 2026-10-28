# Explainable sequential DOE

This package studies hyperparameter effects on classification loss and latency.
It complements the historical California Housing study; its experiments and
reports use separate output directories and never replace that frozen evidence.

## Implemented protocol

1. Load a researcher-supplied binary sentiment CSV. Preserve its official test
   split if present; otherwise reserve a stratified test partition. Fit TF-IDF
   separately on each development fold's training documents.
2. Save the configuration, dataset identity/digest, software versions, source
   digests, measurement definitions, fit budgets and reserved confirmation
   coordinates before the first model fit.
3. Start with the four-factor face-centered CCD, with independently seeded
   center repeats in each development block. Factor ranges are explicit in
   `config.py`. The factors retain the historical identities: learning rate,
   depth, row subsampling and leaf-weight regularization. The new application
   calibrates its ranges and capacity on development data.
4. Fit quadratic fixed-block OLS surfaces and Matern GP alternatives. Compute
   exact interventional Shapley values and pair interactions by enumerating the
   four-input coalition game against the declared empirical background. Keep
   the raw per-point interaction tensors and main-effect dependence profiles.
5. Acquire new canonical configurations, evaluate them in every block, and
   refit between cycles. Available rules are uncertainty, discrepancy, hybrid
   uncertainty/space filling and scalarized expected improvement. EI is not
   expected hypervolume improvement. Integer tree depths are never treated as
   fractional model settings.
6. Repeat MO-TPE and optionally the fixed-CCD baseline with the same real model-fit budget, block information, tree
   count and profiling operation. Methods run serially with randomized method
   order across search replicates. Save every evaluation and sampling/surrogate
   overhead. Report validation hypervolume, feasible fraction, observed-reference
   regret, and variability across repeated searches.
7. Evaluate reserved configurations after fitting. Compare predicted and actual
   responses, fixed-block OLS prediction-interval coverage, GP prediction error,
   model-based GP interval coverage and boundary bias. GP confirmation targets
   block means; fixed-block OLS prediction intervals target block observations.
   These observations never
   enter the fitted surfaces. Initial and updated explanation snapshots permit
   before/after comparisons.
8. Freeze one observed selection per method/search replicate before the final
   evaluator fits preprocessing on all development data and accesses test data.
   Gate checks detect changed contracts and observations. This is a Python
   workflow boundary, not protection against a malicious caller inspecting
   private objects; it cannot prove that a dataset was historically unseen.
9. Generate a Markdown report, numerical summary, ANOVA/SHAP snapshots, Pareto
   and fit-progress plots, and a complete per-run artifact registry. The checker
   verifies source and file digests, reconstructs fit accounting, hypervolumes,
   prediction errors, coverage and OLS confirmation predictions, and compares
   the generated report with reconstructed results.

## Run commands

The recorded research revision uses dataset-specific configuration files from
development-only calibration. Generic CLI defaults are not a substitute for
those frozen settings. Its sequential design releases center replication for
adaptive evaluations while preserving the original classifier-fit cap.

Public dataset preparation is explicit, uses pinned sources, and records
normalization, duplicate exclusions and dataset digests. Preparation requires
`requirements-research.txt`; the study itself uses the core requirements.
The orchestration script is `scripts/run_research_programme.py`.

Local feature-weight explanations are in `local_explanations.py` and
`word_study.py`. A separate `scripts/word_masking_study.py` removes literal
words and re-vectorizes their contexts, including related bigrams. It uses
the saved classifier and does not train additional optimization candidates.

Install from the repository with `python -m pip install -e .` (or install
`requirements.txt` and run the script directly). The console entry point is
`xgb-doe-study`.

Preview the planned fit budget without loading data or fitting anything:

```powershell
python run_explainable_doe_demo.py --plan
```

Run the small software demonstration, explicitly marked as synthetic smoke
evidence, with reduced trees, repetitions, blocks and timing calls:

```powershell
python run_explainable_doe_demo.py --smoke --output output/sentiment-smoke
python run_explainable_doe_demo.py --check output/sentiment-smoke
```

For an author-approved real-data campaign, supply a CSV with `text,label`
columns (labels 0 and 1), a versioned dataset identity, and optionally a `split`
column containing `train` and `test`. Duplicate documents are rejected until
the researcher resolves them. No data are downloaded automatically.

```powershell
python run_explainable_doe_demo.py --dataset-csv C:/datasets/sentiment.csv --dataset-id imdb-version-and-preparation-id --output output/sentiment-research
python run_explainable_doe_demo.py --check output/sentiment-research
```

An optional `--config path.json` accepts fields of `BenchmarkConfig`; `--plan`
also accepts it. The research mode requires the declared warmed protocol.
Each run requires a new output directory; partially completed runs are not
silently resumed or overwritten. Failed final access consumes that evaluator
so a caller cannot silently tune against repeatedly exposed test observations.

## Interpretation

ANOVA partial SS, SHAP attributions and SHAP pair interactions are different
estimands, with different units and background-distribution dependence.
Correlation describes agreement, not mathematical equivalence or independent
validation. Undefined quantities are stored as null and reported as undefined.
Partial ANOVA SS rows need not sum to the model SS. Zero estimated error
variance makes F/LoF inference undefined rather than perfectly significant.

OLS intervals target observations at new configurations in the **known fixed
blocks**. With no block supplied, the interval describes the adjusted mean.
The model does not claim Satterthwaite inference or prediction for an unseen
stochastic block. GP intervals are model-based and their observed coverage is
reported separately. Shared cross-validation observations are dependent;
descriptive coverage and between-search variation are not independent-sample
significance tests. GP convergence warnings are retained in snapshot evidence.

The classic latency operation times a fresh DMatrix construction plus
`Booster.predict` on one document, avoiding warmed prediction-cache retrieval.
`inplace_predict` times an existing row without DMatrix construction. Both use
the same configured warmup/timed-call counts and one-core affinity enforcement;
raw durations and environment metadata are saved. Thresholds inherited from a
different dataset/hardware are configuration choices, not demonstrated serving
requirements. Calibrate them, tree count and the learning-rate range using
development data before approving the final measurement contract.

Execution records determine research status. Incomplete campaigns cannot be
published as completed findings. Synthetic software tests are never empirical
substitutes for the real-data campaign or its author review.
