"""Fit descriptive latency models within one explicitly identified protocol.

This module refuses to pool timing rows from different protocols. Historical
unversioned measurements can be analyzed only with an explicit CLI flag and are
labelled as such in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def validate_latency_protocol(
    frame: pd.DataFrame,
    *,
    expected_protocol_id: str | None = None,
    allow_historical_unversioned: bool = False,
) -> str:
    if "latency_protocol_id" not in frame:
        if not allow_historical_unversioned:
            raise ValueError(
                "latency_protocol_id is required; pass allow_historical_unversioned=True "
                "only for a separately labelled historical sensitivity analysis"
            )
        protocol_id = "historical_unversioned_latency"
    else:
        values = frame["latency_protocol_id"].dropna().astype(str)
        protocols = sorted(value for value in values.unique() if value.strip())
        if len(values) != len(frame) or len(protocols) != 1:
            raise ValueError(
                f"Latency modelling requires exactly one complete protocol ID; got {protocols}"
            )
        protocol_id = protocols[0]
    if expected_protocol_id is not None and protocol_id != expected_protocol_id:
        raise ValueError(
            f"Expected latency protocol {expected_protocol_id!r}, got {protocol_id!r}"
        )
    return protocol_id


def _normalized_latency_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "latency_us" not in result:
        if "predict_latency_us" in result:
            result["latency_us"] = result["predict_latency_us"]
        elif "latency_us_median" in result:
            result["latency_us"] = result["latency_us_median"]
        else:
            raise ValueError(
                "Expected predict_latency_us or latency_us_median response column"
            )
    if "depth" not in result:
        if "max_depth" in result:
            result["depth"] = result["max_depth"]
        else:
            raise ValueError("Expected depth or max_depth column")
    numeric_columns = ["latency_us", "depth"]
    numeric_columns.extend(
        name
        for name in (
            "learning_rate", "subsample", "reg_lambda", "x1", "x2", "x3", "x4"
        )
        if name in result
    )
    for name in numeric_columns:
        result[name] = pd.to_numeric(result[name], errors="coerce")
    invalid = {
        name: result.index[~np.isfinite(result[name].to_numpy(dtype=float))].tolist()
        for name in numeric_columns
        if not np.all(np.isfinite(result[name].to_numpy(dtype=float)))
    }
    if invalid:
        raise ValueError(f"Latency analysis requires finite complete numeric rows: {invalid}")
    if "block" in result and result["block"].isna().any():
        raise ValueError("Latency analysis requires complete block identifiers")
    if len(result) < 4:
        raise ValueError("At least four complete timing rows are required")
    if (result["latency_us"] <= 0).any():
        raise ValueError("Latency responses must be positive")
    return result


def _fit_checked(formula: str, data: pd.DataFrame):
    fit = smf.ols(formula, data, missing="raise").fit()
    expected_rank = int(fit.model.exog.shape[1])
    if int(fit.nobs) != len(data):
        raise ValueError("statsmodels changed the declared complete-case analysis rows")
    if int(fit.model.rank) != expected_rank:
        raise ValueError(
            f"design is rank deficient ({fit.model.rank} < {expected_rank})"
        )
    if fit.df_resid <= 0:
        raise ValueError("model has no positive residual degrees of freedom")
    if not np.all(np.isfinite(np.asarray(fit.params, dtype=float))):
        raise ValueError("model parameters are not finite")
    return fit


def _unavailable_model(name: str, formula: str, response_scale: str, error: Exception):
    return {
        "model": name,
        "formula": formula,
        "available": False,
        "fit_response_scale": response_scale,
        "reason": f"{type(error).__name__}: {error}",
    }


def _model_record(
    name: str,
    formula: str,
    fit,
    observed,
    predicted,
    *,
    response_scale: str,
    smearing_factor: float | None = None,
) -> dict[str, Any]:
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    if not np.all(np.isfinite(residual)):
        raise ValueError("raw-scale model predictions must be finite")
    fit_statistics = {
        "response_scale": response_scale,
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "comparability": (
            "AIC, BIC, and R-squared are comparable only with models fitted to this same "
            "response scale."
        ),
    }
    if not all(
        np.isfinite(value)
        for key, value in fit_statistics.items()
        if key in {"r_squared", "adjusted_r_squared", "aic", "bic"}
    ):
        raise ValueError("fit statistics must be finite")
    return {
        "model": name,
        "formula": formula,
        "available": True,
        "n": int(fit.nobs),
        "rank": int(fit.model.rank),
        "residual_degrees_of_freedom": float(fit.df_resid),
        "fit_response_scale": response_scale,
        "fit_statistics": fit_statistics,
        "raw_scale_in_sample_rmse": float(np.sqrt(np.mean(residual**2))),
        "raw_scale_prediction_definition": (
            "conditional mean using Duan smearing"
            if smearing_factor is not None
            else "ordinary fitted mean"
        ),
        "duan_smearing_factor": smearing_factor,
        "parameters": {name: float(value) for name, value in fit.params.items()},
    }


def analyze_latency_models(
    frame: pd.DataFrame,
    *,
    expected_protocol_id: str | None = None,
    allow_historical_unversioned: bool = False,
) -> dict[str, Any]:
    protocol_id = validate_latency_protocol(
        frame,
        expected_protocol_id=expected_protocol_id,
        allow_historical_unversioned=allow_historical_unversioned,
    )
    data = _normalized_latency_frame(frame)
    models = []

    formula = "latency_us ~ depth"
    try:
        linear = _fit_checked(formula, data)
        models.append(
            _model_record(
                "linear_depth",
                formula,
                linear,
                data["latency_us"],
                linear.fittedvalues,
                response_scale="latency_us",
            )
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        models.append(_unavailable_model("linear_depth", formula, "latency_us", exc))

    formula = "latency_us ~ depth + I(depth**2)"
    try:
        quadratic = _fit_checked(formula, data)
        models.append(
            _model_record(
                "quadratic_depth",
                "latency_us ~ depth + depth^2",
                quadratic,
                data["latency_us"],
                quadratic.fittedvalues,
                response_scale="latency_us",
            )
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        models.append(_unavailable_model("quadratic_depth", formula, "latency_us", exc))

    formula = "np.log(latency_us) ~ depth"
    try:
        log_linear = _fit_checked(formula, data)
        smearing_factor = float(np.mean(np.exp(log_linear.resid)))
        if not np.isfinite(smearing_factor) or smearing_factor <= 0:
            raise ValueError("Duan smearing factor must be positive and finite")
        models.append(
            _model_record(
                "log_linear_depth",
                "log(latency_us) ~ depth",
                log_linear,
                data["latency_us"],
                np.exp(log_linear.fittedvalues) * smearing_factor,
                response_scale="log_latency_us",
                smearing_factor=smearing_factor,
            )
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        models.append(
            _unavailable_model(
                "log_linear_depth", formula, "log_latency_us", exc
            )
        )

    multifactor_terms = [
        name
        for name in ("learning_rate", "subsample", "reg_lambda")
        if name in data
    ]
    if multifactor_terms:
        formula = "latency_us ~ depth + I(depth**2) + " + " + ".join(multifactor_terms)
        try:
            multifactor = _fit_checked(formula, data)
            models.append(
                _model_record(
                    "multifactor_descriptive",
                    formula,
                    multifactor,
                    data["latency_us"],
                    multifactor.fittedvalues,
                    response_scale="latency_us",
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            models.append(
                _unavailable_model(
                    "multifactor_descriptive", formula, "latency_us", exc
                )
            )

    full_ccd = None
    ccd_columns = {"block", "x1", "x2", "x3", "x4"}
    if ccd_columns <= set(data):
        augmented = data.copy()
        factors = ["x1", "x2", "x3", "x4"]
        for factor in factors:
            augmented[f"{factor}_sq"] = augmented[factor] ** 2
        interactions = []
        for left_index in range(len(factors)):
            for right_index in range(left_index + 1, len(factors)):
                name = f"{factors[left_index]}_{factors[right_index]}"
                augmented[name] = (
                    augmented[factors[left_index]] * augmented[factors[right_index]]
                )
                interactions.append(name)
        formula = (
            "latency_us ~ C(block) + "
            + " + ".join(factors + [f"{factor}_sq" for factor in factors] + interactions)
        )
        try:
            fit = _fit_checked(formula, augmented)
            full_ccd = _model_record(
                "blocked_second_order_ccd",
                formula,
                fit,
                augmented["latency_us"],
                fit.fittedvalues,
                response_scale="latency_us",
            )
            anova = (
                sm.stats.anova_lm(fit, typ=3)
                .reset_index()
                .rename(columns={"index": "term"})
            )
            anova = anova.replace([np.inf, -np.inf], np.nan)
            full_ccd["anova_type_3"] = [
                {
                    key: (None if pd.isna(value) else value)
                    for key, value in row.items()
                }
                for row in anova.to_dict("records")
            ]
            models.append(full_ccd)
        except (ValueError, np.linalg.LinAlgError) as exc:
            models.append(
                _unavailable_model(
                    "blocked_second_order_ccd", formula, "latency_us", exc
                )
            )

    return {
        "schema_version": 2,
        "latency_protocol_id": protocol_id,
        "response": "predict_latency_us",
        "input_rows": int(len(frame)),
        "analysis_rows": int(len(data)),
        "excluded_rows": 0,
        "model_comparison_metric": "raw_scale_in_sample_rmse",
        "interpretation": (
            "Descriptive association within one timing protocol; model form does not establish "
            "a hardware mechanism or a causal source of interface overhead. Likelihood and R-squared "
            "statistics are comparable only among models fitted on the same response scale; the "
            "log-linear model uses Duan smearing for raw-scale mean predictions."
        ),
        "models": models,
    }


def write_analysis(result: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "latency_modeling.json"
    output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    comparison_rows = []
    for model in result["models"]:
        fit_statistics = model.get("fit_statistics", {})
        comparison_rows.append({
            "model": model["model"],
            "available": model["available"],
            "fit_response_scale": model["fit_response_scale"],
            "raw_scale_in_sample_rmse": model.get("raw_scale_in_sample_rmse"),
            "r_squared_within_response_scale": fit_statistics.get("r_squared"),
            "adjusted_r_squared_within_response_scale": fit_statistics.get(
                "adjusted_r_squared"
            ),
            "aic_within_response_scale": fit_statistics.get("aic"),
            "bic_within_response_scale": fit_statistics.get("bic"),
            "reason": model.get("reason"),
        })
    pd.DataFrame(comparison_rows).to_csv(
        destination / "latency_models_comparison.csv", index=False
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit latency models without pooling timing protocols."
    )
    parser.add_argument("--input", required=True, help="CSV containing timing responses")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-protocol-id")
    parser.add_argument(
        "--historical-unversioned",
        action="store_true",
        help="Explicitly label and analyze legacy rows that lack latency_protocol_id.",
    )
    args = parser.parse_args()
    result = analyze_latency_models(
        pd.read_csv(args.input),
        expected_protocol_id=args.expected_protocol_id,
        allow_historical_unversioned=args.historical_unversioned,
    )
    output = write_analysis(result, args.output_dir)
    print(f"Latency model analysis written to {output}")


if __name__ == "__main__":
    main()
