import itertools

import numpy as np
import pytest

import latency


class FakeBooster:
    def __init__(self):
        self.inplace_calls = 0
        self.parameters = {}

    def set_param(self, parameters):
        self.parameters.update(parameters)

    def inplace_predict(self, sample):
        self.inplace_calls += 1
        return np.zeros(len(sample))


class FakeModel:
    def __init__(self):
        self.predict_calls = 0
        self.parameters = {}
        self.booster = FakeBooster()

    def set_params(self, **parameters):
        self.parameters.update(parameters)
        return self

    def get_booster(self):
        return self.booster

    def predict(self, sample):
        self.predict_calls += 1
        return np.zeros(len(sample))


def fixed_elapsed_clock(elapsed_ns=4_000):
    ticks = itertools.count()

    def clock():
        tick = next(ticks)
        return ((tick + 1) // 2) * elapsed_ns

    return clock


def test_primary_and_online_protocols_have_explicit_measurement_budgets():
    assert latency.PRIMARY_V1.interfaces == ("predict", "inplace_predict")
    assert latency.PRIMARY_V1.warmup_calls_per_interface == 50
    assert latency.PRIMARY_V1.timed_calls_per_interface == 1_000
    assert latency.PRIMARY_V1.primary_metric == "predict_latency_us"

    assert latency.ONLINE_SEARCH_V1.interfaces == ("predict",)
    assert latency.ONLINE_SEARCH_V1.warmup_calls_per_interface == 10
    assert latency.ONLINE_SEARCH_V1.timed_calls_per_interface == 30
    assert latency.ONLINE_SEARCH_V1.primary_metric == "predict_latency_us"


def test_measure_latencies_returns_distinct_metrics_and_auditable_metadata(monkeypatch):
    model = FakeModel()
    protocol = latency.LatencyProtocol(
        protocol_id="unit-test-v1",
        interfaces=("predict", "inplace_predict"),
        warmup_calls_per_interface=1,
        repetitions=2,
        calls_per_repetition=2,
        inference_threads=1,
        cpu_core=3,
        order_seed=7,
        randomize_order=False,
    )
    monkeypatch.setattr(latency, "pin_cpu_affinity", lambda core_id: core_id == 3)
    monkeypatch.setattr(latency.time, "perf_counter_ns", fixed_elapsed_clock())

    result = latency.measure_latencies(
        {"candidate-a": model},
        np.zeros((1, 8), dtype=np.float64),
        session_id="session-001",
        protocol=protocol,
    )

    summary = result["summaries"]["candidate-a"]
    assert summary == {
        "predict_latency_us": 2.0,
        "predict_latency_us_iqr": 0.0,
        "inplace_predict_latency_us": 2.0,
        "inplace_predict_latency_us_iqr": 0.0,
    }
    assert model.predict_calls == 5
    assert model.booster.inplace_calls == 5
    assert model.parameters["n_jobs"] == 1
    assert model.booster.parameters["nthread"] == 1

    metadata = result["metadata"]
    assert metadata["session"]["session_id"] == "session-001"
    assert metadata["protocol"]["protocol_id"] == "unit-test-v1"
    assert metadata["protocol"]["timed_calls_per_interface"] == 4
    assert metadata["protocol"]["api_labels"] == {
        "predict": "XGBRegressor.predict",
        "inplace_predict": "Booster.inplace_predict",
    }
    assert metadata["input"]["batch_rows"] == 1
    assert metadata["input"]["shape"] == [1, 8]
    assert metadata["input"]["dtype"] == "float64"
    assert metadata["input"]["representation"] == "numpy.ndarray"
    assert metadata["input"]["memory_order"] == "C"
    assert metadata["hardware"]["requested_cpu_core"] == 3
    assert metadata["hardware"]["cpu_affinity_applied"] is True
    assert metadata["hardware"]["logical_cpu_count"] is not None

    observations = result["observations"]
    assert len(observations) == 4
    assert {row["interface"] for row in observations} == {"predict", "inplace_predict"}
    assert all(row["calls"] == 2 for row in observations)
    assert all(row["latency_us"] == 2.0 for row in observations)
    assert all(row["session_id"] == "session-001" for row in observations)


def test_seeded_randomized_interleaving_is_repeatable(monkeypatch):
    protocol = latency.LatencyProtocol(
        protocol_id="randomized-unit-test-v1",
        interfaces=("predict", "inplace_predict"),
        warmup_calls_per_interface=0,
        repetitions=3,
        calls_per_repetition=1,
        order_seed=19,
        randomize_order=True,
    )
    monkeypatch.setattr(latency, "pin_cpu_affinity", lambda core_id: True)

    def run_once():
        monkeypatch.setattr(latency.time, "perf_counter_ns", fixed_elapsed_clock())
        result = latency.measure_latencies(
            {name: FakeModel() for name in ("a", "b", "c")},
            np.zeros((1, 2)),
            session_id="repeatable-session",
            protocol=protocol,
        )
        return [
            (row["repetition"], row["order_within_repetition"], row["model_id"], row["interface"])
            for row in result["observations"]
        ]

    assert run_once() == run_once()


def test_online_protocol_marks_unmeasured_inplace_latency_as_missing(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(latency, "pin_cpu_affinity", lambda core_id: True)
    monkeypatch.setattr(latency.time, "perf_counter_ns", fixed_elapsed_clock())

    result = latency.measure_latencies(
        {"trial-7": model},
        np.zeros((1, 3)),
        session_id="optimizer-replicate-4",
        protocol=latency.ONLINE_SEARCH_V1,
    )

    summary = result["summaries"]["trial-7"]
    assert summary["predict_latency_us"] == pytest.approx(0.4)
    assert summary["inplace_predict_latency_us"] is None
    assert summary["inplace_predict_latency_us_iqr"] is None
    assert model.predict_calls == 40
    assert model.booster.inplace_calls == 0


@pytest.mark.parametrize(
    "overrides",
    [
        {"protocol_id": ""},
        {"interfaces": ()},
        {"interfaces": ("predict", "unknown")},
        {"warmup_calls_per_interface": -1},
        {"repetitions": 0},
        {"calls_per_repetition": 0},
        {"inference_threads": 0},
    ],
)
def test_protocol_rejects_invalid_measurement_contracts(overrides):
    values = {
        "protocol_id": "valid-v1",
        "interfaces": ("predict",),
        "warmup_calls_per_interface": 0,
        "repetitions": 1,
        "calls_per_repetition": 1,
        "inference_threads": 1,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        latency.LatencyProtocol(**values)


def test_measure_latencies_requires_a_single_input_row(monkeypatch):
    monkeypatch.setattr(latency, "pin_cpu_affinity", lambda core_id: True)

    with pytest.raises(ValueError, match="exactly one row"):
        latency.measure_latencies(
            {"candidate": FakeModel()},
            np.zeros((2, 8)),
            session_id="invalid-input-session",
        )
