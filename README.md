# Explainable experimental design for XGBoost

Can designed experiments select useful XGBoost configurations, explain hyperparameter effects, and guide further evaluations efficiently?

- **Phase A:** California Housing regression, blocked CCD, response surfaces and optimizer comparison.
- **Phase B:** IMDb sentiment classification, static and sequential CCD versus MO-TPE, ANOVA–SHAP comparison and independent configuration confirmation.

The paper is [report/report.pdf](report/report.pdf), with source in [report/report.tex](report/report.tex).

## Structure

- `phase_a_rsm_blocking/`: housing methods, results and original report.
- `phase_b_sequential_xai/`: sentiment methods and prepared datasets.
- `results/phase_b/`: full sentiment experiment.
- `results/phase_b_calibration/`: development calibration.
- `findings/phase_a/` and `findings/phase_b/`: scientific figures and tables.
- `report/`: the unified paper and its result-driven publication tools.
- `evidence/`: the lossless published sentiment observations and calibration.
- `specifications/`: original project brief and supervisor guidance.

## Run

```powershell
git clone https://github.com/ChristosHussein/xgboost-doe-rsm.git
cd xgboost-doe-rsm
git checkout paper-2026-10-r3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproduction.txt
python run_experiment.py --smoke
python run_experiment.py --phase a --mode full
```

Phase B with CUDA training and a fresh output directory:

```powershell
python phase_b_sequential_xai/run_phase_b.py --stage all --dataset full_imdb --data-dir phase_b_sequential_xai/data --calibration-dir results/imdb_new/calibration --output-dir results/imdb_new --findings-dir findings/imdb_new --device cuda
```

Use `--device cpu` for CPU training. Prediction latency uses CPU in either case. CUDA requires compatible hardware, driver and XGBoost installation.

For a new replication, the direct Phase B command also accepts `--threshold-mode capacity_matched`, `--latency-panel-size 32`, and `--consistent-tfidf-perturbations`. These options define new measurements; they do not alter the published observations. Use matching options for calibration and the main run, with a fresh output directory. The default protocol remains available for historical reproduction. See the paper's execution appendix and [replication protocol](specifications/Replication%20protocol.md) for scope and remaining design requirements.

Check the published results without retraining:

```powershell
python report/restore_evidence.py
python report/verify_report.py --check --self-test
python report/build_evidence.py --check
```

The archive restores the original observation bytes into the ignored `results/` directory and refuses to replace different local results. The prepared IMDb data and the Phase A observations are also included in the repository.

To regenerate the paper, run `python report/build_evidence.py`, then compile `report/report.tex` with pdfLaTeX from its directory. The fixed tag identifies the reference code, evidence and PDF. A new CPU or CUDA run is a separate experimental replication; its timings need not match the reference machine.
