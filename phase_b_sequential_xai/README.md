# Phase B: explainable sequential design for sentiment classification

IMDb documents are represented by TF-IDF features and classified with XGBoost. Static CCD, sequential CCD and MO-TPE use shared development folds and equal model-fit budgets. The objectives are classification loss and CPU prediction latency.

## Data and results

- `data/`: prepared datasets and source manifests.
- `../results/phase_b_calibration/`: development calibration.
- `../results/phase_b/`: full IMDb experiment.
- `../findings/phase_b/`: scientific tables and figures.
- `../report/report.pdf`: combined paper.

## New study

From the repository root, use a fresh output directory:

```powershell
python phase_b_sequential_xai/run_phase_b.py --stage all --dataset full_imdb --data-dir phase_b_sequential_xai/data --calibration-dir results/imdb_new/calibration --output-dir results/imdb_new --findings-dir findings/imdb_new --device cuda
```

Use `--device cpu` for CPU training. `--training-threads` controls training-side CPU workers. Latency and word-response probabilities use CPU prediction.

Individual stages are `prepare`, `calibrate`, `run` and `generate-findings`. The full study includes local word analysis.

```powershell
python phase_b_sequential_xai/run_phase_b.py --check results/phase_b/imdb
python phase_b_sequential_xai/run_phase_b.py --stage generate-findings --dataset full_imdb --output-dir results/phase_b --findings-dir findings/phase_b
```
