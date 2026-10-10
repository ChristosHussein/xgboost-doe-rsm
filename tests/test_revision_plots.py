import json

import pandas as pd
import pytest


def _write_plot_inputs(
    path, *, trial_protocol="online-v1", doe_split=42, trial_split=42
):
    pd.DataFrame(
        [{
            "candidate_id": "doe-1",
            "candidate_type": "historical_doe_evaluated_coordinate",
            "validation_rmse": 0.49,
            "predict_latency_us": 120.0,
            "latency_protocol_id": "online-v1",
            "development_split_seed": doe_split,
        }]
    ).to_csv(path / "doe_matched_candidate_front.csv", index=False)
    pd.DataFrame(
        [{
            "optimizer": "multi_objective_tpe",
            "trial_status": "completed",
            "validation_rmse": 0.48,
            "predict_latency_us": 130.0,
            "latency_protocol_id": trial_protocol,
            "development_split_seed": trial_split,
            "replicate_id": 0,
        }]
    ).to_csv(path / "optimizer_trials.csv", index=False)
    summary = {
        "n": 4,
        "mean": 0.485,
        "median": 0.485,
        "standard_deviation": 0.003,
        "interquartile_range": 0.002,
        "confidence_interval_95": [0.480, 0.490],
    }
    pd.DataFrame(
        [{
            "selection_id": "multi-objective-tpe-rep-0",
            "optimizer": "multi_objective_tpe",
            "validation_rmse": json.dumps(summary),
            "predict_latency_us": 125.0,
            "predict_latency_session_standard_deviation_us": 2.0,
            "latency_protocol_id": "primary-v1",
        }]
    ).to_csv(path / "final_summary.csv", index=False)


def test_revision_pareto_loader_keeps_development_and_independent_estimands_separate(tmp_path):
    from plots import load_revision_pareto_data

    _write_plot_inputs(tmp_path)
    data = load_revision_pareto_data(tmp_path)
    assert set(data) == {
        "doe_development",
        "mo_tpe_development",
        "independent_selected",
    }
    assert data["independent_selected"]["validation_rmse_mean"].iloc[0] == 0.485


def test_revision_pareto_loader_rejects_mixed_development_timing_protocols(tmp_path):
    from plots import load_revision_pareto_data

    _write_plot_inputs(tmp_path, trial_protocol="different-online-v2")
    with pytest.raises(ValueError, match="matching latency protocol"):
        load_revision_pareto_data(tmp_path)


def test_revision_pareto_loader_rejects_missing_protocol_ids(tmp_path):
    from plots import load_revision_pareto_data

    _write_plot_inputs(tmp_path)
    trials = pd.read_csv(tmp_path / "optimizer_trials.csv")
    trials.loc[0, "latency_protocol_id"] = None
    trials.to_csv(tmp_path / "optimizer_trials.csv", index=False)
    with pytest.raises(ValueError, match="complete latency_protocol_id"):
        load_revision_pareto_data(tmp_path)


def test_revision_pareto_loader_rejects_unmatched_development_splits(tmp_path):
    from plots import load_revision_pareto_data

    _write_plot_inputs(tmp_path, doe_split=42, trial_split=43)
    with pytest.raises(ValueError, match="same split"):
        load_revision_pareto_data(tmp_path)
