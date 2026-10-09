import json
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
    )

    expected = {
        "computational_budget.json",
        "optimizer_trials.csv",
        "optimizer_replicates.csv",
        "finalized_selections.json",
        "latency_measurement.json",
        "final_evaluations.csv",
        "final_summary.csv",
        "paired_comparisons.json",
        "hypervolume.json",
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

    run_manifest = json.loads((destination / "run_manifest.json").read_text())
    assert run_manifest["classification"] == "new revision_v2 experiment"
    assert run_manifest["historical_artifacts_modified"] is False
