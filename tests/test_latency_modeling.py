import numpy as np
import pandas as pd
import pytest


def timing_frame(protocol="single_sample_latency_primary_v1"):
    depth = np.tile(np.arange(3, 10), 2)
    generator = np.random.default_rng(17)
    return pd.DataFrame(
        {
            "max_depth": depth,
            "predict_latency_us": 90.0 + 4.0 * depth + 0.7 * depth**2,
            "learning_rate": generator.uniform(0.03, 0.25, len(depth)),
            "subsample": generator.uniform(0.6, 1.0, len(depth)),
            "reg_lambda": generator.uniform(0.2, 2.0, len(depth)),
            "latency_protocol_id": protocol,
        }
    )


def test_latency_models_reject_mixed_protocols():
    from scripts.fit_latency_models import analyze_latency_models

    frame = timing_frame()
    frame.loc[0, "latency_protocol_id"] = "another_protocol"
    with pytest.raises(ValueError, match="exactly one"):
        analyze_latency_models(frame)


def test_latency_models_require_protocol_unless_explicitly_historical():
    from scripts.fit_latency_models import analyze_latency_models

    frame = timing_frame().drop(columns="latency_protocol_id")
    with pytest.raises(ValueError, match="latency_protocol_id"):
        analyze_latency_models(frame)
    result = analyze_latency_models(frame, allow_historical_unversioned=True)
    assert result["latency_protocol_id"] == "historical_unversioned_latency"


def test_latency_models_preserve_protocol_and_response_definition():
    from scripts.fit_latency_models import analyze_latency_models

    result = analyze_latency_models(
        timing_frame(), expected_protocol_id="single_sample_latency_primary_v1"
    )
    assert result["response"] == "predict_latency_us"
    assert result["latency_protocol_id"] == "single_sample_latency_primary_v1"
    assert {item["model"] for item in result["models"]} >= {
        "linear_depth",
        "quadratic_depth",
        "log_linear_depth",
        "multifactor_descriptive",
    }
