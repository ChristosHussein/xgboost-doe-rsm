"""Reconstruct and classify the historical desirability selection without changing it."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import build_design_matrix, derringer_suich_desirability, predict_block_averaged
from desirability_provenance import historical_stated_grid, selection_audit


BASELINE_REF = "v1.0.0"


def _git_text(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BASELINE_REF}:{path}"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _surface_desirabilities(fit_y1, fit_y2, grid, specification, latency_offset=0.0):
    predictions = []
    desirabilities = []
    for coordinate in grid:
        y1 = predict_block_averaged(fit_y1, coordinate)
        y2 = predict_block_averaged(fit_y2, coordinate) + latency_offset
        d1, d2, composite = derringer_suich_desirability(
            y1,
            y2,
            specification["L1"],
            specification["U1"],
            specification["L2"],
            specification["U2"],
            w1=specification["w1"],
            w2=specification["w2"],
        )
        predictions.append({
            "validation_rmse": y1,
            "predict_latency_us": y2,
            "rmse_desirability": d1,
            "latency_desirability": d2,
        })
        desirabilities.append(composite)
    return predictions, desirabilities


def build_audit() -> dict:
    runs = pd.read_csv(io.StringIO(_git_text("results/runs.csv")))
    confirmation = json.loads(_git_text("results/confirmation.json"))
    baseline_config_text = _git_text("config.yaml")
    baseline_config = yaml.safe_load(baseline_config_text)
    design = build_design_matrix(runs)
    fit_y1 = sm.OLS(runs["val_rmse"], design).fit()
    fit_y2 = sm.OLS(runs["latency_us_median"], design).fit()
    grid = historical_stated_grid()
    selected = np.asarray(confirmation["x_star_coded"], dtype=float)
    standard = {
        "L1": float(baseline_config["desirability"]["Y1_RMSE"]["L"]),
        "U1": float(baseline_config["desirability"]["Y1_RMSE"]["U"]),
        "L2": float(baseline_config["desirability"]["Y2_Latency"]["L"]),
        "U2": float(baseline_config["desirability"]["Y2_Latency"]["U"]),
        "w1": float(baseline_config["desirability"]["Y1_RMSE"]["weight"]),
        "w2": float(baseline_config["desirability"]["Y2_Latency"]["weight"]),
    }
    predictions, values = _surface_desirabilities(fit_y1, fit_y2, grid, standard)
    audit = selection_audit(selected, grid, values)
    maximum_index = audit["mathematical_grid_maximum"]["candidate_index"]
    audit["mathematical_grid_maximum"].update(predictions[maximum_index])

    selected_y1 = predict_block_averaged(fit_y1, selected)
    selected_y2 = predict_block_averaged(fit_y2, selected)
    selected_d1, selected_d2, selected_composite = derringer_suich_desirability(
        selected_y1,
        selected_y2,
        standard["L1"],
        standard["U1"],
        standard["L2"],
        standard["U2"],
        w1=standard["w1"],
        w2=standard["w2"],
    )

    observed = (
        runs.groupby(["point_id", "x1", "x2", "x3", "x4"], as_index=False)
        .agg(validation_rmse=("val_rmse", "mean"), predict_latency_us=("latency_us_median", "mean"))
    )
    observed["desirability"] = [
        derringer_suich_desirability(
            row.validation_rmse,
            row.predict_latency_us,
            standard["L1"],
            standard["U1"],
            standard["L2"],
            standard["U2"],
            w1=standard["w1"],
            w2=standard["w2"],
        )[2]
        for row in observed.itertuples(index=False)
    ]
    empirical = observed.sort_values(
        ["desirability", "validation_rmse", "predict_latency_us", "point_id"],
        ascending=[False, True, True, True],
    ).iloc[0]

    scenarios = [
        {"name": "standard", **standard},
        {"name": "latency_weight_2", **standard, "w2": 2.0},
        {"name": "rmse_weight_2", **standard, "w1": 2.0},
        {"name": "strict_latency", **standard, "L2": 90.0, "U2": 140.0, "w2": 2.0},
        {"name": "strict_accuracy", **standard, "L1": 0.44, "U1": 0.60, "U2": 200.0, "w1": 2.0},
    ]
    common_latency_shift = float(np.median(runs["latency_us_iqr"]))
    sensitivity = []
    for scenario in scenarios:
        for offset in (-common_latency_shift, 0.0, common_latency_shift):
            _, scenario_values = _surface_desirabilities(
                fit_y1, fit_y2, grid, scenario, latency_offset=offset
            )
            maximum = float(np.max(scenario_values))
            tied = np.flatnonzero(
                np.isclose(scenario_values, maximum, rtol=0.0, atol=1e-12)
            )
            best_index = int(tied[0])
            sensitivity.append({
                "scenario": scenario["name"],
                "latency_offset_us": offset,
                "coordinate": grid[best_index].tolist(),
                "desirability": float(scenario_values[best_index]),
                "tie_count": int(len(tied)),
                "tie_breaking_rule": "lowest deterministic enumeration index",
            })

    baseline_commit = subprocess.run(
        ["git", "rev-parse", f"{BASELINE_REF}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    return {
        "schema_version": 1,
        "classification": "historical_data_reanalysis_no_new_xgboost_evaluations",
        "baseline_ref": BASELINE_REF,
        "baseline_commit": baseline_commit,
        "input_sha256": {
            f"{BASELINE_REF}:config.yaml": _sha256_text(baseline_config_text),
            f"{BASELINE_REF}:results/runs.csv": _sha256_text(
                _git_text("results/runs.csv")
            ),
            f"{BASELINE_REF}:results/confirmation.json": _sha256_text(
                _git_text("results/confirmation.json")
            ),
            "scripts/audit_desirability_selection.py": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "stated_grid": {
            "candidate_count": int(len(grid)),
            "x1_levels": np.linspace(-1.0, 1.0, 21).tolist(),
            "integer_depth_levels": list(range(3, 10)),
            "x3_levels": [0.0, 1.0],
            "x4_levels": [0.0, 1.0],
            "enumeration_order": "x1, x3, x4, integer depth",
        },
        "standard_specification": standard,
        "selection_audit": audit,
        "selected_engineering_compromise": {
            "coordinate": selected.tolist(),
            "predicted_validation_rmse": selected_y1,
            "predicted_latency_us": selected_y2,
            "rmse_desirability": selected_d1,
            "latency_desirability": selected_d2,
            "composite_desirability": selected_composite,
            "source": "hard-coded historical confirmation coordinate",
            "continuous_optimization_or_interpolation_record": (
                "No executable v1 provenance establishes a continuous optimizer, interpolation, "
                "or quantitative robustness rule for this coordinate."
            ),
        },
        "empirically_best_observed_configuration": {
            "point_id": int(empirical["point_id"]),
            "coordinate": [
                float(empirical["x1"]),
                float(empirical["x2"]),
                float(empirical["x3"]),
                float(empirical["x4"]),
            ],
            "mean_validation_rmse": float(empirical["validation_rmse"]),
            "mean_latency_us": float(empirical["predict_latency_us"]),
            "desirability": float(empirical["desirability"]),
        },
        "boundary_statement": (
            "A face-centered CCD boundary is observed within the declared domain; it is not "
            "extrapolation solely because it lies on a face."
        ),
        "common_latency_calibration_shift_sensitivity": {
            "offset_magnitude_us": common_latency_shift,
            "basis": (
                "Common additive shift equal to the median historical within-run latency IQR. "
                "This is a calibration-shift stress test, not a candidate standard error, "
                "prediction interval, or estimate of timing uncertainty."
            ),
            "results": sensitivity,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="docs/provenance/historical_desirability_selection.json",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing audit artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_audit(), indent=2) + "\n", encoding="utf-8")
    print(f"Historical desirability audit written to {output}")


if __name__ == "__main__":
    main()
