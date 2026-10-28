# Confirmed training failure: unsorted sparse feature indices

The original development-fold GPU fit reproduced the saved word-model tree dump exactly. Sorting column indices changes storage order only: elementwise matrix equality was asserted. The original training matrix has unsorted indices; validation is already sorted.

| Training input/device | Validation macro F1 | Log loss | Leaf-only trees |
|---|---:|---:|---:|
| cuda_original | 0.448365 | 7.151894 | 98 |
| cuda_sorted | 0.770756 | 0.506193 | 0 |
| cpu_sorted | 0.772668 | 0.506326 | 0 |

These are controlled development diagnostics, not corrected campaign results. No official test rows were evaluated. Both CPU and GPU prediction of each trained model were checked; moving prediction to CPU did not remove the faulty training outcome.

Root cause established at the integration boundary: unsorted CSR TF-IDF training input triggers invalid learning in the installed XGBoost CUDA path. The exact upstream kernel defect and whether other versions are affected have not been established. The hardware itself has not been shown defective.

Required remedy: canonicalize sparse matrices (sum duplicate entries and sort column indices) before development, final-refit and local-word model training; test equivalent sorted/unsorted inputs on CUDA. Preserve old evidence, then rerun calibration and all downstream campaign stages into a new versioned directory. Review prior test exposure before calling any subsequent evaluation untouched. No production code or historical evidence was changed by this diagnosis.

The unconditional underfitting_resolved flag is also insufficient: calibration must verify meaningful learning, not merely equality with a larger tree budget. The separate CI smoke-formatting defect remains outstanding.

Sources: diagnostic/root_cause.json; diagnostic/results.json; confirm_root_cause.py; diagnose_training.py. The original evidence audit is verification.json.