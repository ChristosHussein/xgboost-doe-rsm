# Published experimental evidence

`phase_b.zip` is a lossless snapshot of the sentiment study and its development calibration. `manifest.json` records the archive checksum and each original file's checksum. Observation files retain their original bytes, including line endings.

From the repository root:

```powershell
python report/restore_evidence.py
python report/verify_report.py --check
```

The restoration command verifies checksums, extracts missing observations and refuses to overwrite conflicting local results. It does not train models or modify recorded measurements. The source hashes in the restored measurement contract identify the experimental implementation; the publication manifest identifies the final manuscript and PDF.

Phase A observations remain under `phase_a_rsm_blocking/results/`. The prepared dataset under `phase_b_sequential_xai/data/` and the reference dependency pins accompany the code. Dataset and method references are given in the paper.

`phase_a_sources.zip` preserves the original housing benchmark source files. Their hashes are checked against the recorded benchmark provenance. The verifier accepts documentation-only changes in Python files only when the executable syntax tree is identical; changes to experimental logic fail verification.

Word-classifier replay checks primary probabilities within two float32 rounding units across platforms. It independently refits the saved local surfaces, checks Shapley values against propagated rounding bounds and requires the published RMSE precision to remain unchanged. Original observations and experimental code are preserved; exact saved-result calculations and file checksums use the frozen evidence.
