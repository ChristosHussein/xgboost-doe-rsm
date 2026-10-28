import json
import subprocess
from collections import namedtuple

import numpy as np
import pandas as pd


DevelopmentSplit = namedtuple(
    "DevelopmentSplit", ["X_train", "X_val", "y_train", "y_val"]
)


class SyntheticDevelopmentManager:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_split(self, split_seed):
        del split_seed
        return DevelopmentSplit(
            self.X[:36], self.X[36:48], self.y[:36], self.y[36:48]
        )


class FakeBooster:
    def __init__(self, model):
        self.model = model

    def set_param(self, parameters):
        del parameters

    def inplace_predict(self, X):
        return self.model.predict(X)


class FakeRegressor:
    def __init__(self, **parameters):
        self.parameters = parameters
        self.max_depth = int(parameters["max_depth"])

    def fit(self, X, y):
        del X, y
        return self

    def predict(self, X):
        return np.full(len(X), self.max_depth / 100.0)

    def set_params(self, **parameters):
        self.parameters.update(parameters)
        return self

    def get_booster(self):
        return FakeBooster(self)


def test_synthetic_revision_workflow_writes_gated_versioned_artifacts(
    monkeypatch, tmp_path
):
    import final_evaluation
    from scripts import run_benchmarks as benchmarks

    X = np.arange(480, dtype=float).reshape(60, 8)
    y = np.linspace(0.0, 0.5, 60)
    manager = SyntheticDevelopmentManager(X, y)
    evaluator = final_evaluation.FinalTestEvaluator(X=X, y=y)
    monkeypatch.setattr(benchmarks.xgb, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(final_evaluation.xgb, "XGBRegressor", FakeRegressor)

    destination = benchmarks.run_revision_benchmark(
        "smoke",
        output_dir=str(tmp_path / "revision-run"),
        data_mgr=manager,
        final_evaluator=evaluator,
        settings_override={
            "optimizer_replicates": 1,
            "trials_per_optimizer": 2,
            "evaluation_seeds": [101, 102],
        },
        include_historical_doe=False,
        include_repeated_doe=False,
    )

    expected = {
        "computational_budget.json",
        "optimizer_trials.csv",
        "optimizer_replicates.csv",
        "optimizer_summary.json",
        "doe_selection_summary.json",
        "finalized_selections.json",
        "latency_measurement.json",
        "latency_interface_overhead.json",
        "final_evaluations.csv",
        "final_summary.csv",
        "paired_comparisons.json",
        "hypervolume.json",
        "provenance.json",
        "run_manifest.json",
    }
    assert expected <= {path.name for path in destination.iterdir()}

    trials = pd.read_csv(destination / "optimizer_trials.csv")
    assert len(trials) == 8
    assert trials.groupby(["optimizer", "replicate_id"])["selected"].sum().eq(1).all()
    assert "test_rmse" not in trials.columns

    manifest = json.loads((destination / "finalized_selections.json").read_text())
    assert manifest["status"] == "finalized"
    assert len(manifest["configurations"]) == 4
    assert all(item["config_sha256"] for item in manifest["configurations"])

    final_rows = pd.read_csv(destination / "final_evaluations.csv")
    assert len(final_rows) == 8
    assert set(final_rows["evaluation_seed"]) == {101, 102}

    budget = json.loads((destination / "computational_budget.json").read_text())
    assert budget["search_model_fits"] == 8
    assert budget["matched_historical_doe_candidate_fits"] == 0
    assert budget["primary_latency_refit_fits"] == 8
    assert budget["final_retraining_fits"] == 8
    assert budget["maximum_total_model_fits"] == 24

    latency = json.loads((destination / "latency_measurement.json").read_text())
    assert len(latency["sessions"]) == 2
    assert latency["aggregation"]["session_count"] == 2
    assert set(latency["summaries"]) == {
        "random_search-rep-0",
        "single_objective_tpe-rep-0",
        "constrained_tpe-rep-0",
        "multi_objective_tpe-rep-0",
    }
    assert all(
        "predict_latency_session_standard_deviation_us" in item
        for item in latency["summaries"].values()
    )

    overhead = json.loads(
        (destination / "latency_interface_overhead.json").read_text()
    )
    assert overhead["estimand"] == (
        "predict_latency_us minus inplace_predict_latency_us within timing session"
    )
    assert len(overhead["per_session"]) == 8

    optimizer_summary = json.loads(
        (destination / "optimizer_summary.json").read_text()
    )
    assert optimizer_summary["inference_status"] == "smoke_underpowered"
    assert set(optimizer_summary["optimizers"]) == {
        "random_search",
        "single_objective_tpe",
        "constrained_tpe",
        "multi_objective_tpe",
    }

    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["producer"] == "scripts/run_benchmarks.py"
    peeled_baseline = subprocess.run(
        ["git", "rev-parse", "v1.0.0^{commit}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    assert provenance["historical_baseline_revision"] == peeled_baseline
    assert "optimizer_trials.csv" in provenance["artifact_sha256"]
    assert "scripts/run_benchmarks.py" in provenance["runtime_source_sha256"]
    assert "analysis.py" in provenance["runtime_source_sha256"]

    run_manifest = json.loads((destination / "run_manifest.json").read_text())
    assert run_manifest["classification"] == "new revision_v2 experiment"
    assert run_manifest["historical_artifacts_modified"] is False
    assert run_manifest["inference_status"] == "smoke_underpowered"
    assert expected <= set(run_manifest["artifacts"])


def test_revision_benchmark_resume_loads_checkpoints(monkeypatch, tmp_path):
    import pytest
    import final_evaluation
    from scripts import run_benchmarks as benchmarks

    X = np.arange(480, dtype=float).reshape(60, 8)
    y = np.linspace(0.0, 0.5, 60)
    manager = SyntheticDevelopmentManager(X, y)
    evaluator = final_evaluation.FinalTestEvaluator(X=X, y=y)
    monkeypatch.setattr(benchmarks.xgb, "XGBRegressor", FakeRegressor)
    monkeypatch.setattr(final_evaluation.xgb, "XGBRegressor", FakeRegressor)

    run_dir = tmp_path / "resume-test"
    destination = benchmarks.run_revision_benchmark(
        "smoke",
        output_dir=str(run_dir),
        data_mgr=manager,
        final_evaluator=evaluator,
        settings_override={
            "optimizer_replicates": 1,
            "trials_per_optimizer": 2,
            "evaluation_seeds": [101, 102],
        },
        include_historical_doe=False,
        include_repeated_doe=False,
    )
    ckpt_dir = destination / "checkpoints"
    assert ckpt_dir.exists()
    assert (ckpt_dir / "optimizer_replicate_0.json").exists()
    assert (ckpt_dir / "primary_latency_sessions.json").exists()
    assert (ckpt_dir / "final_evaluations_rows.json").exists()

    with pytest.raises(FileExistsError, match="Refusing to overwrite non-empty"):
        benchmarks.run_revision_benchmark(
            "smoke",
            output_dir=str(run_dir),
            data_mgr=manager,
            final_evaluator=evaluator,
            settings_override={
                "optimizer_replicates": 1,
                "trials_per_optimizer": 2,
                "evaluation_seeds": [101, 102],
            },
            include_historical_doe=False,
            include_repeated_doe=False,
            resume=False,
        )

    resumed_destination = benchmarks.run_revision_benchmark(
        "smoke",
        output_dir=str(run_dir),
        data_mgr=manager,
        final_evaluator=evaluator,
        settings_override={
            "optimizer_replicates": 1,
            "trials_per_optimizer": 2,
            "evaluation_seeds": [101, 102],
        },
        include_historical_doe=False,
        include_repeated_doe=False,
        resume=True,
    )
    assert resumed_destination == destination
    assert (destination / "run_manifest.json").exists()
