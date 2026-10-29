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
- `specifications/`: original project brief and supervisor guidance.

## Run

```powershell
git clone https://github.com/ChristosHussein/xgboost-doe-rsm.git
cd xgboost-doe-rsm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_experiment.py --smoke
python run_experiment.py --phase a --mode full
```

Phase B with CUDA training and a fresh output directory:

```powershell
python phase_b_sequential_xai/run_phase_b.py --stage all --dataset full_imdb --data-dir phase_b_sequential_xai/data --calibration-dir results/imdb_new/calibration --output-dir results/imdb_new --findings-dir findings/imdb_new --device cuda
```

Use `--device cpu` for CPU training. Prediction latency uses CPU in either case. CUDA requires compatible hardware, driver and XGBoost installation.

```powershell
python report/build_evidence.py
python report/verify_report.py
```

Compile `report/report.tex` from its directory. Reconstructing published numbers requires the raw experiment files; raw results are excluded from Git.
