# XGBoost: planned experiments and explainable sequential optimization

Start with **[the main report](report.pdf)** for the scientific question,
methods, measured housing results, and the explainable sequential extension.
The extension is implemented, but its real sentiment-data study has not run.

- [Technical supplement](technical_supplement.pdf): derivations, complete
  statistical tables, residual diagnostics, and detailed historical protocols.
- [Generated research record](REPORT.md): the comprehensive housing evidence.
- [Sequential study guide](xgb_doe_xai/README.md): inputs, measurement contracts,
  synthetic execution checks, and commands for a separately approved study.
- [Publication integrity audit](docs/report_integrity_audit.md): source bindings,
  verification scope, and remaining scientific limitations.

## Project layout

- `xgb_doe_xai/`: the modular sequential study implementation.
- `scripts/`: reporting, verification, controlled execution, and build commands.
- `tests/`: statistical, provenance, execution, and reporting checks.
- `results/`: frozen empirical evidence and generated numerical macros.
- `figures/` and `tables/`: active publication graphics and generated tables.
- `docs/generated/`: metric, claim-occurrence, and compiled-PDF input bindings.
- `docs/provenance/`: original evidence snapshots; historical paths refer to
  the repository state at capture time.
- `docs/archive/`: superseded prototype outputs and review records, retained
  byte-identically and clearly separated from current evidence.

## Verify and rebuild

Install `requirements.txt`, then run:

```text
python -m pytest -q
python scripts/audit_scientific_consistency.py
python scripts/generate_report_artifacts.py --check
python scripts/generate_research_reporting.py --check
python scripts/build_publication.py --check
```

Edit the manuscript source in `scripts/templates/report.tex`. Numerical
observations belong in the source-backed reporting layer, not in authored
prose. Regenerate the publication inputs and rebuild with a local LaTeX
installation:

```text
python scripts/build_publication.py
python scripts/generate_research_reporting.py --check
```

These commands reconstruct reports from saved observations; they do not rerun
the research campaign. Do not alter frozen inputs to repair a report.
