# Correction verification — 28 October 2026

The corrected pipeline passed its bounded operational verification. This does not validate the scientific results of the earlier faulty campaign.

## Corrections

- Canonicalize sparse training/validation features before XGBoost development fits and final refits, and before word-model training. Preserve mathematical feature values and caller-owned inputs.
- Generate and replay word explanations using the CPU prediction path, while retaining CUDA training. This avoids attribution changes caused by CPU/GPU prediction roundoff.
- Export unavailable statistics as `N/A`; do not turn an undefined standard deviation into zero.
- Isolate smoke outputs from published findings.
- Add development-pilot mode that rejects a final-test evaluator; stop claiming that satisfying the relative tree-capacity rule proves underfitting resolved.

## Evidence

The machine-readable record is `correction_verification.json`, which binds the completed checks to source hashes and the verified code commit.

- Eight local tests passed with the CUDA hardware tests enabled, including a check of the actual training device.
- The master smoke command passed across both phases.
- The corrected real-development pilot completed 600 search/confirmation fits and one word-model fit. Study registries, classification reconstruction, confirmation, local-word replay, and all findings exports passed. It scored zero official IMDb test observations.
- All six fresh calibration records were independently reconstructed from saved predictions. Each tree budget produced distinct probabilities; the capacity rule now selects 600 trees.
- GitHub Actions passed on Ubuntu and Windows for commit `3dfb8631dea0b46f93457121576a458f9391fa24`.
- Historical campaign file hashes remain unchanged. The first verification pilot, which exposed the word-replay issue, is preserved separately.

GitHub check: https://github.com/ChristosHussein/xgboost-doe-rsm/actions/runs/40696810685

Draft fix PR: https://github.com/ChristosHussein/xgboost-doe-rsm/pull/6

The fresh pilot is in `results/cuda_fix_verification_v2/`; calibration is in `results/cuda_fixed_calibration/`. The pilot is an operational check with one repeat and a smaller tree budget, rather than a replacement scientific campaign. The original full campaign and the report's old empirical tables remain invalid for final conclusions. Full scientific rerunning must use the corrected calibrated capacity, a recorded budget/runtime description, and an explicit account of previous test exposure. No full corrected campaign was started by this verification.

The existing PDF remains a verification draft documenting the original failure. This note records the subsequent successful software corrections; final performance results must come from the new scientific campaign.
