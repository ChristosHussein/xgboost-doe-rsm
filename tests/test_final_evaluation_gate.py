import json

import numpy as np
import pytest


def selected_configuration():
    return {
        "selection_id": "tpe-so-rep-0",
        "optimizer": "tpe_single_objective",
        "source_trial_id": 7,
        "selection_metric": {"name": "validation_rmse", "value": 0.47},
        "hyperparameters": {
            "learning_rate": 0.1,
            "max_depth": 4,
            "subsample": 0.8,
            "reg_lambda": 1.0,
        },
        "model": {
            "n_estimators": 100,
            "objective": "reg:squarederror",
            "n_jobs_train": 1,
        },
    }


def test_finalized_manifest_is_tamper_evident(tmp_path):
    from final_evaluation import create_finalized_selection, load_finalized_selection

    path = tmp_path / "selection.json"
    create_finalized_selection(
        path,
        configurations=[selected_configuration()],
        selection_policy="minimum validation RMSE",
        development_seeds=[42],
        source_artifact="results/revision_v2/trials.csv",
        git_revision="abc123",
        created_at_utc="2026-10-09T00:00:00+00:00",
    )
    manifest = load_finalized_selection(path)
    assert manifest["status"] == "finalized"

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["configurations"][0]["hyperparameters"]["max_depth"] = 9
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_finalized_selection(path)


def test_final_evaluation_rejects_nonfinal_manifest(tmp_path):
    from final_evaluation import load_finalized_selection

    path = tmp_path / "draft.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "draft"}), encoding="utf-8")
    with pytest.raises(ValueError, match="finalized"):
        load_finalized_selection(path)


def test_final_evaluator_uses_only_frozen_configuration(monkeypatch, tmp_path):
    import final_evaluation

    seen = []

    class FakeRegressor:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(final_evaluation.xgb, "XGBRegressor", FakeRegressor)
    path = tmp_path / "selection.json"
    final_evaluation.create_finalized_selection(
        path,
        configurations=[selected_configuration()],
        selection_policy="minimum validation RMSE",
        development_seeds=[42],
        source_artifact="results/revision_v2/trials.csv",
        git_revision="abc123",
        created_at_utc="2026-10-09T00:00:00+00:00",
    )

    X = np.arange(320, dtype=float).reshape(40, 8)
    y = np.linspace(0.0, 1.0, 40)
    evaluator = final_evaluation.FinalTestEvaluator(X=X, y=y)
    result = evaluator.evaluate(path, "tpe-so-rep-0", evaluation_seeds=[101])

    assert seen == [{
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "reg_lambda": 1.0,
        "random_state": 101,
        "n_jobs": 1,
        "objective": "reg:squarederror",
    }]
    assert result["selection_id"] == "tpe-so-rep-0"
    assert len(result["per_seed"]) == 1
    assert "test_rmse" in result["per_seed"][0]


def test_final_evaluator_rejects_unknown_selection(tmp_path):
    from final_evaluation import FinalTestEvaluator, create_finalized_selection

    path = tmp_path / "selection.json"
    create_finalized_selection(
        path,
        configurations=[selected_configuration()],
        selection_policy="minimum validation RMSE",
        development_seeds=[42],
        source_artifact="results/revision_v2/trials.csv",
        git_revision="abc123",
    )
    X = np.arange(320, dtype=float).reshape(40, 8)
    y = np.linspace(0.0, 1.0, 40)
    evaluator = FinalTestEvaluator(X=X, y=y)
    with pytest.raises(KeyError, match="not frozen"):
        evaluator.evaluate(path, "ad-hoc-config", evaluation_seeds=[101])
