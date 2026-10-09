from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_benchmarks as benchmarks


DevelopmentSplit = namedtuple(
    "DevelopmentSplit", ["X_train", "X_val", "y_train", "y_val"]
)


class DevelopmentOnlyManager:
    """Small development split with no external-test fields to access."""

    def get_split(self, split_seed):
        del split_seed
        return DevelopmentSplit(
            X_train=np.zeros((4, 4), dtype=float),
            X_val=np.zeros((2, 4), dtype=float),
            y_train=np.zeros(4, dtype=float),
            y_val=np.zeros(2, dtype=float),
        )


class CompatibilitySplit:
    """Supports the revised named interface and lets legacy code reach later assertions."""

    def __init__(self):
        self.X_train = np.zeros((4, 4), dtype=float)
        self.X_val = np.zeros((2, 4), dtype=float)
        self.y_train = np.zeros(4, dtype=float)
        self.y_val = np.zeros(2, dtype=float)

    def __iter__(self):
        # Compatibility only for tests whose focus is unrelated to holdout access.
        yield self.X_train
        yield self.X_val
        yield np.zeros((1, 4), dtype=float)
        yield self.y_train
        yield self.y_val
        yield np.zeros(1, dtype=float)


class CompatibilityManager:
    def get_split(self, split_seed):
        del split_seed
        return CompatibilitySplit()


class FakeBooster:
    def __init__(self, model):
        self.model = model

    def set_param(self, params):
        del params

    def inplace_predict(self, X):
        return self.model.predict(X)


class FakeRegressor:
    prediction_by_depth = {3: 0.10, 4: 0.20, 5: 0.30, 6: 0.40}

    def __init__(self, **kwargs):
        self.params = dict(kwargs)
        self.max_depth = int(kwargs["max_depth"])

    def fit(self, X, y):
        del X, y
        return self

    def predict(self, X):
        value = self.prediction_by_depth.get(self.max_depth, 0.50)
        return np.full(len(X), value, dtype=float)

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def get_booster(self):
        return FakeBooster(self)


@pytest.fixture(autouse=True)
def _reset_fake_predictions():
    FakeRegressor.prediction_by_depth = {3: 0.10, 4: 0.20, 5: 0.30, 6: 0.40}


@dataclass
class PlannedTrial:
    number: int
    planned: dict

    def __post_init__(self):
        self.params = {}
        self.float_calls = []
        self.int_calls = []
        self.user_attrs = {}
        self.value = None
        self.values = None
        self.state = SimpleNamespace(name="COMPLETE")

    def suggest_float(self, name, low, high, **kwargs):
        self.float_calls.append((name, low, high, kwargs))
        value = self.planned.get(name)
        if value is None:
            aliases = {"x1": -0.5, "x2": -1.0, "x3": 0.0, "x4": 0.0}
            value = aliases.get(name, (low + high) / 2.0)
        self.params[name] = float(value)
        return float(value)

    def suggest_int(self, name, low, high, **kwargs):
        self.int_calls.append((name, low, high, kwargs))
        value = int(self.planned.get(name, low))
        self.params[name] = value
        return value

    def set_user_attr(self, name, value):
        self.user_attrs[name] = value


class PlannedStudy:
    def __init__(self, plans):
        self.trials = [PlannedTrial(i, plan) for i, plan in enumerate(plans)]

    def optimize(self, objective, n_trials, **kwargs):
        del kwargs
        assert n_trials == len(self.trials)
        for trial in self.trials:
            outcome = objective(trial)
            if isinstance(outcome, (tuple, list)):
                trial.values = [float(value) for value in outcome]
                trial.value = None
            else:
                trial.value = float(outcome)
                trial.values = [trial.value]

    @property
    def best_trial(self):
        return min(self.trials, key=lambda trial: trial.value)

    @property
    def best_trials(self):
        return self.trials


class RecordingRng:
    def __init__(self):
        self.integer_calls = []

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is not None:
            # Values accepted by the historical coded-space implementation.
            return np.resize(np.array([-0.5, -1.0, 0.0, 0.0]), size)
        return float((low + high) / 2.0)

    def randint(self, low, high=None, size=None, dtype=int):
        self.integer_calls.append(("randint", low, high, size))
        value = int(low if high is None else min(max(4, low), high - 1))
        return np.asarray(value, dtype=dtype) if size is not None else value

    def integers(self, low, high=None, size=None, **kwargs):
        del kwargs
        self.integer_calls.append(("integers", low, high, size))
        value = int(low if high is None else min(max(4, low), high - 1))
        return np.full(size, value, dtype=int) if size is not None else value

    def choice(self, values, size=None, **kwargs):
        del kwargs
        self.integer_calls.append(("choice", tuple(values), None, size))
        value = 4 if 4 in values else values[0]
        return np.full(size, value, dtype=int) if size is not None else value


def _install_fast_model(monkeypatch):
    monkeypatch.setattr(benchmarks.xgb, "XGBRegressor", FakeRegressor)


def _install_planned_study(monkeypatch, plans):
    study = PlannedStudy(plans)
    monkeypatch.setattr(benchmarks.optuna, "create_study", lambda **kwargs: study)
    return study


def _plan(depth, *, x2):
    return {
        "learning_rate": 0.05,
        "max_depth": depth,
        "subsample": 0.75,
        "reg_lambda": 1.0,
        "x1": -0.5,
        "x2": x2,
        "x3": 0.0,
        "x4": 0.0,
    }


def _call_search(name, manager, *, replicate_id=None, max_latency_us=150.0):
    kwargs = {
        "n_trials": 2,
        "sampler_seed": 17,
        "eval_seed": 23,
        "data_mgr": manager,
    }
    if replicate_id is not None:
        kwargs["replicate_id"] = replicate_id
    if name == "constrained_tpe":
        kwargs["max_latency_us"] = max_latency_us
        return benchmarks.run_tpe_constrained(**kwargs)
    if name == "multi_objective_tpe":
        return benchmarks.run_tpe_multi_objective(**kwargs)
    if name == "single_objective_tpe":
        return benchmarks.run_tpe_single_objective(**kwargs)
    if name == "random_search":
        return benchmarks.run_random_search(**kwargs)
    raise AssertionError(f"unknown optimizer {name}")


@pytest.mark.parametrize(
    "optimizer_name",
    ["random_search", "single_objective_tpe", "constrained_tpe", "multi_objective_tpe"],
)
def test_optimizers_accept_a_development_only_split(monkeypatch, optimizer_name):
    _install_fast_model(monkeypatch)
    _install_planned_study(monkeypatch, [_plan(3, x2=-1.0), _plan(4, x2=-2 / 3)])
    monkeypatch.setattr(
        benchmarks,
        "measure_trial_latency",
        lambda model, sample, **kwargs: 100.0 + model.max_depth,
    )

    result = _call_search(optimizer_name, DevelopmentOnlyManager())

    assert result.status == "completed"


def test_tpe_samples_integer_depth_directly(monkeypatch):
    _install_fast_model(monkeypatch)
    study = _install_planned_study(
        monkeypatch, [_plan(3, x2=-1.0), _plan(4, x2=-2 / 3)]
    )

    benchmarks.run_tpe_single_objective(
        n_trials=2,
        sampler_seed=17,
        eval_seed=23,
        data_mgr=CompatibilityManager(),
    )

    for trial in study.trials:
        assert any(
            name == "max_depth" and low == 3 and high == 9
            for name, low, high, _ in trial.int_calls
        )
        assert all(name not in {"max_depth", "depth", "x2"} for name, *_ in trial.float_calls)


def test_random_search_samples_integer_depth_directly(monkeypatch):
    _install_fast_model(monkeypatch)
    rng = RecordingRng()
    monkeypatch.setattr(benchmarks.np.random, "RandomState", lambda seed: rng)
    monkeypatch.setattr(benchmarks.np.random, "default_rng", lambda seed: rng)

    benchmarks.run_random_search(
        n_trials=2,
        sampler_seed=17,
        eval_seed=23,
        data_mgr=CompatibilityManager(),
    )

    assert rng.integer_calls, "max_depth must be sampled from a discrete integer distribution"


def test_all_infeasible_constrained_search_records_no_incumbent(monkeypatch):
    _install_fast_model(monkeypatch)
    _install_planned_study(monkeypatch, [_plan(3, x2=-1.0), _plan(4, x2=-2 / 3)])
    monkeypatch.setattr(
        benchmarks,
        "measure_trial_latency",
        lambda model, sample, **kwargs: 200.0 + model.max_depth,
    )

    result = benchmarks.run_tpe_constrained(
        n_trials=2,
        sampler_seed=17,
        eval_seed=23,
        max_latency_us=150.0,
        data_mgr=CompatibilityManager(),
        replicate_id=8,
    )

    assert result.status == "infeasible"
    assert result.selected_x is None
    assert result.selected_score is None
    assert len(result.trial_records) == 2
    assert not any(record["selected"] for record in result.trial_records)
    assert not any(record["feasibility"] for record in result.trial_records)


REQUIRED_TRIAL_FIELDS = {
    "optimizer",
    "optimizer_version",
    "replicate_id",
    "sampler_seed",
    "development_split_seed",
    "trial_number",
    "x1",
    "x2",
    "x3",
    "x4",
    "learning_rate",
    "max_depth",
    "subsample",
    "reg_lambda",
    "model_config",
    "validation_rmse",
    "predict_latency_us",
    "objective_value",
    "feasibility",
    "constraint_violation_us",
    "training_time_s",
    "evaluation_time_s",
    "trial_status",
    "error",
    "selected",
    "selection_rule",
}


@pytest.mark.parametrize(
    "optimizer_name",
    ["random_search", "single_objective_tpe", "constrained_tpe", "multi_objective_tpe"],
)
def test_trial_records_are_complete_and_selected_incumbent_is_logged(
    monkeypatch, optimizer_name
):
    _install_fast_model(monkeypatch)
    _install_planned_study(monkeypatch, [_plan(3, x2=-1.0), _plan(4, x2=-2 / 3)])
    monkeypatch.setattr(
        benchmarks,
        "measure_trial_latency",
        lambda model, sample, **kwargs: {3: 140.0, 4: 100.0}[model.max_depth],
    )
    if optimizer_name == "random_search":
        rng = RecordingRng()
        monkeypatch.setattr(benchmarks.np.random, "RandomState", lambda seed: rng)
        monkeypatch.setattr(benchmarks.np.random, "default_rng", lambda seed: rng)

    result = _call_search(
        optimizer_name,
        CompatibilityManager(),
        replicate_id=11,
        max_latency_us=150.0,
    )

    assert result.status == "completed"
    assert result.selected_x is not None
    assert len(result.trial_records) == 2
    assert all(REQUIRED_TRIAL_FIELDS <= record.keys() for record in result.trial_records)

    selected_records = [record for record in result.trial_records if record["selected"]]
    assert len(selected_records) == 1
    selected = selected_records[0]
    assert np.allclose(
        result.selected_x,
        [selected["x1"], selected["x2"], selected["x3"], selected["x4"]],
    )
    assert selected["replicate_id"] == 11
    assert selected["sampler_seed"] == 17
    assert selected["development_split_seed"] == 23
    assert selected["selection_rule"] == result.selection_rule


def test_measured_latency_controls_constrained_selection(monkeypatch):
    _install_fast_model(monkeypatch)
    _install_planned_study(monkeypatch, [_plan(3, x2=-1.0), _plan(4, x2=-2 / 3)])
    monkeypatch.setattr(
        benchmarks,
        "measure_trial_latency",
        lambda model, sample, **kwargs: {3: 200.0, 4: 100.0}[model.max_depth],
    )

    result = benchmarks.run_tpe_constrained(
        n_trials=2,
        sampler_seed=17,
        eval_seed=23,
        max_latency_us=150.0,
        data_mgr=CompatibilityManager(),
        replicate_id=12,
    )

    # Depth 3 has the better RMSE but is infeasible; measured latency must select depth 4.
    assert result.status == "completed"
    selected = next(record for record in result.trial_records if record["selected"])
    assert selected["max_depth"] == 4
    assert selected["predict_latency_us"] == pytest.approx(100.0)
    assert selected["feasibility"] is True


def test_measured_latency_controls_multi_objective_selection(monkeypatch):
    _install_fast_model(monkeypatch)
    FakeRegressor.prediction_by_depth = {3: 0.50, 4: 0.50}
    _install_planned_study(monkeypatch, [_plan(3, x2=-1.0), _plan(4, x2=-2 / 3)])
    monkeypatch.setattr(
        benchmarks,
        "measure_trial_latency",
        lambda model, sample, **kwargs: {3: 200.0, 4: 100.0}[model.max_depth],
    )

    result = benchmarks.run_tpe_multi_objective(
        n_trials=2,
        sampler_seed=17,
        eval_seed=23,
        data_mgr=CompatibilityManager(),
        replicate_id=13,
    )

    # Equal-RMSE candidates differ only in measured latency, so the faster one must win.
    selected = next(record for record in result.trial_records if record["selected"])
    assert selected["max_depth"] == 4
    assert selected["predict_latency_us"] == pytest.approx(100.0)
    assert np.allclose(
        result.selected_x,
        [selected["x1"], selected["x2"], selected["x3"], selected["x4"]],
    )
