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
    if len(result) < 4:
        raise ValueError("At least four timing rows are required")
    if (result["latency_us"] <= 0).any():
        raise ValueError("Latency responses must be positive")
    return result


def _model_record(name: str, formula: str, fit, observed, predicted) -> dict[str, Any]:
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "model": name,
        "formula": formula,
        "n": int(fit.nobs),
        "rank": int(fit.model.rank),
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "rmse_on_response_scale": float(np.sqrt(np.mean(residual**2))),
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

    linear = smf.ols("latency_us ~ depth", data).fit()
    models.append(
        _model_record(
            "linear_depth", "latency_us ~ depth", linear, data["latency_us"], linear.fittedvalues
        )
    )
    quadratic = smf.ols("latency_us ~ depth + I(depth**2)", data).fit()
    models.append(
        _model_record(
            "quadratic_depth",
            "latency_us ~ depth + depth^2",
            quadratic,
            data["latency_us"],
            quadratic.fittedvalues,
        )
    )
    log_linear = smf.ols("np.log(latency_us) ~ depth", data).fit()
    models.append(
        _model_record(
            "log_linear_depth",
            "log(latency_us) ~ depth",
            log_linear,
            data["latency_us"],
            np.exp(log_linear.fittedvalues),
        )
    )

    multifactor_terms = [
        name
        for name in ("learning_rate", "subsample", "reg_lambda")
        if name in data
    ]
    if multifactor_terms:
        formula = "latency_us ~ depth + I(depth**2) + " + " + ".join(multifactor_terms)
        multifactor = smf.ols(formula, data).fit()
        models.append(
            _model_record(
                "multifactor_descriptive",
                formula,
                multifactor,
                data["latency_us"],
                multifactor.fittedvalues,
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
        fit = smf.ols(formula, augmented).fit()
        full_ccd = _model_record(
            "blocked_second_order_ccd",
            formula,
            fit,
            augmented["latency_us"],
            fit.fittedvalues,
        )
        full_ccd["anova_type_3"] = (
            sm.stats.anova_lm(fit, typ=3).reset_index().rename(columns={"index": "term"}).to_dict("records")
        )
        models.append(full_ccd)

    return {
        "schema_version": 2,
        "latency_protocol_id": protocol_id,
        "response": "predict_latency_us",
        "interpretation": (
            "Descriptive association within one timing protocol; model form does not establish "
            "a hardware mechanism or a causal source of interface overhead."
        ),
        "models": models,
    }


def write_analysis(result: dict[str, Any], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "latency_modeling.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(result["models"]).drop(columns=["parameters", "anova_type_3"], errors="ignore").to_csv(
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
