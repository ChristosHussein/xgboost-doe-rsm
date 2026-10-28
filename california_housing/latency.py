"""Shared, auditable single-row inference-latency measurement.

The module measures fitted models only. Training, artifact persistence, and
experiment-specific identifiers remain the responsibility of each caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np


_API_LABELS = {
    "predict": "XGBRegressor.predict",
    "inplace_predict": "Booster.inplace_predict",
}


@dataclass(frozen=True)
class LatencyProtocol:
    """Complete measurement contract for one latency protocol version."""

    protocol_id: str
    interfaces: tuple[str, ...]
    warmup_calls_per_interface: int
    repetitions: int
    calls_per_repetition: int
    inference_threads: int = 1
    cpu_core: int | None = 0
    order_seed: int = 20261005
    randomize_order: bool = True
    primary_metric: str = "predict_latency_us"

    def __post_init__(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty")
        if not self.interfaces:
            raise ValueError("interfaces must be non-empty")
        if len(set(self.interfaces)) != len(self.interfaces):
            raise ValueError("interfaces must not contain duplicates")
        unsupported = set(self.interfaces) - set(_API_LABELS)
        if unsupported:
            raise ValueError(f"unsupported latency interfaces: {sorted(unsupported)}")
        if self.warmup_calls_per_interface < 0:
            raise ValueError("warmup_calls_per_interface must be non-negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.calls_per_repetition <= 0:
            raise ValueError("calls_per_repetition must be positive")
        if self.inference_threads <= 0:
            raise ValueError("inference_threads must be positive")
        if self.cpu_core is not None and self.cpu_core < 0:
            raise ValueError("cpu_core must be non-negative or None")
        expected_primary = {
            "predict": "predict_latency_us",
            "inplace_predict": "inplace_predict_latency_us",
        }
        if self.primary_metric not in {
            expected_primary[interface] for interface in self.interfaces
        }:
            raise ValueError("primary_metric must name one of the measured interfaces")

    @property
    def timed_calls_per_interface(self) -> int:
        return self.repetitions * self.calls_per_repetition

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "primary_metric": self.primary_metric,
            "interfaces": list(self.interfaces),
            "api_labels": {
                interface: _API_LABELS[interface] for interface in self.interfaces
            },
            "warmup_calls_per_interface": self.warmup_calls_per_interface,
            "warmup_schedule": "model_then_interface",
            "repetitions": self.repetitions,
            "calls_per_repetition": self.calls_per_repetition,
            "timed_calls_per_interface": self.timed_calls_per_interface,
            "inference_threads": self.inference_threads,
            "clock": "time.perf_counter_ns",
            "aggregation": "median_of_repetition_mean_microseconds_per_call",
            "dispersion": "interquartile_range_of_repetition_means",
            "execution_order": (
                "seeded_random_permutation_of_model_interface_batches"
                if self.randomize_order
                else "fixed_model_interface_order"
            ),
            "order_seed": self.order_seed,
        }


PRIMARY_V1 = LatencyProtocol(
    protocol_id="single_sample_latency_primary_v1",
    interfaces=("predict", "inplace_predict"),
    warmup_calls_per_interface=50,
    repetitions=5,
    calls_per_repetition=200,
)


ONLINE_SEARCH_V1 = LatencyProtocol(
    protocol_id="single_sample_latency_online_search_v1",
    interfaces=("predict",),
    warmup_calls_per_interface=10,
    repetitions=3,
    calls_per_repetition=10,
    randomize_order=False,
)


def pin_cpu_affinity(core_id: int = 0) -> bool:
    """Pin this process to one CPU core where the platform supports it."""
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core_id})
            return True
        except (OSError, ValueError):
            pass

    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.SetPriorityClass.restype = ctypes.c_bool

            handle = kernel32.GetCurrentProcess()
            affinity_applied = kernel32.SetProcessAffinityMask(handle, 1 << core_id)
            kernel32.SetPriorityClass(handle, 0x00008000)
            return bool(affinity_applied)
        except (AttributeError, OSError, ValueError):
            pass

    return False


def _input_metadata(sample: Any) -> dict[str, Any]:
    shape = getattr(sample, "shape", None)
    if shape is None or len(shape) == 0 or int(shape[0]) != 1:
        raise ValueError("latency measurement requires an input with exactly one row")

    memory_order = None
    if isinstance(sample, np.ndarray):
        if sample.flags.c_contiguous:
            memory_order = "C"
        elif sample.flags.f_contiguous:
            memory_order = "F"
        else:
            memory_order = "non-contiguous"

    dtype = getattr(sample, "dtype", None)
    return {
        "representation": f"{type(sample).__module__}.{type(sample).__qualname__}",
        "shape": [int(size) for size in shape],
        "batch_rows": 1,
        "dtype": str(dtype) if dtype is not None else None,
        "memory_order": memory_order,
    }


def _hardware_metadata(
    requested_cpu_core: int | None,
    affinity_applied: bool | None,
) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpu_count": os.cpu_count(),
        "requested_cpu_core": requested_cpu_core,
        "cpu_affinity_applied": affinity_applied,
    }


def _configured_interfaces(
    models: Mapping[str, Any],
    protocol: LatencyProtocol,
) -> dict[tuple[str, str], Any]:
    callables: dict[tuple[str, str], Any] = {}
    for model_id, model in models.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model identifiers must be non-empty strings")
        try:
            model.set_params(n_jobs=protocol.inference_threads)
            booster = model.get_booster()
            booster.set_param({"nthread": protocol.inference_threads})
        except AttributeError as exc:
            raise TypeError(
                "models must provide set_params(), get_booster(), and predict()"
            ) from exc

        for interface in protocol.interfaces:
            if interface == "predict":
                try:
                    callables[(model_id, interface)] = model.predict
                except AttributeError as exc:
                    raise TypeError("models must provide predict()") from exc
            else:
                try:
                    callables[(model_id, interface)] = booster.inplace_predict
                except AttributeError as exc:
                    raise TypeError("model boosters must provide inplace_predict()") from exc
    return callables


def _summary(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    median = float(np.median(values))
    iqr = float(np.subtract(*np.percentile(values, [75, 25])))
    return median, iqr


def measure_latencies(
    models: Mapping[str, Any],
    sample: Any,
    *,
    session_id: str,
    protocol: LatencyProtocol = PRIMARY_V1,
) -> dict[str, Any]:
    """Measure fitted models under one explicit, serializable protocol.

    The returned dictionary is JSON-ready and contains summary metrics, every
    timed batch observation, and enough metadata to distinguish timing sessions.
    """
    if not models:
        raise ValueError("models must be non-empty")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    started_at_utc = datetime.now(timezone.utc).isoformat()
    input_metadata = _input_metadata(sample)
    affinity_applied = (
        pin_cpu_affinity(protocol.cpu_core) if protocol.cpu_core is not None else None
    )
    callables = _configured_interfaces(models, protocol)

    for measured_call in callables.values():
        for _ in range(protocol.warmup_calls_per_interface):
            measured_call(sample)

    pairs = list(callables)
    random_generator = np.random.default_rng(protocol.order_seed)
    observations = []
    execution_order = 0
    for repetition in range(1, protocol.repetitions + 1):
        if protocol.randomize_order:
            schedule = [pairs[index] for index in random_generator.permutation(len(pairs))]
        else:
            schedule = pairs

        for order_within_repetition, pair in enumerate(schedule, start=1):
            model_id, interface = pair
            measured_call = callables[pair]
            started_ns = time.perf_counter_ns()
            for _ in range(protocol.calls_per_repetition):
                measured_call(sample)
            elapsed_ns = time.perf_counter_ns() - started_ns
            execution_order += 1
            observations.append({
                "session_id": session_id,
                "protocol_id": protocol.protocol_id,
                "model_id": model_id,
                "repetition": repetition,
                "order_within_repetition": order_within_repetition,
                "execution_order": execution_order,
                "interface": interface,
                "api_label": _API_LABELS[interface],
                "calls": protocol.calls_per_repetition,
                "elapsed_ns": int(elapsed_ns),
                "latency_us": float(
                    elapsed_ns / (protocol.calls_per_repetition * 1000.0)
                ),
            })

    summaries = {}
    for model_id in models:
        values = {
            interface: [
                row["latency_us"]
                for row in observations
                if row["model_id"] == model_id and row["interface"] == interface
            ]
            for interface in _API_LABELS
        }
        predict_median, predict_iqr = _summary(values["predict"])
        inplace_median, inplace_iqr = _summary(values["inplace_predict"])
        summaries[model_id] = {
            "predict_latency_us": predict_median,
            "predict_latency_us_iqr": predict_iqr,
            "inplace_predict_latency_us": inplace_median,
            "inplace_predict_latency_us_iqr": inplace_iqr,
        }

    return {
        "schema_version": 1,
        "metadata": {
            "session": {
                "session_id": session_id,
                "started_at_utc": started_at_utc,
                "process_id": os.getpid(),
            },
            "protocol": protocol.metadata(),
            "input": input_metadata,
            "hardware": _hardware_metadata(protocol.cpu_core, affinity_applied),
        },
        "summaries": summaries,
        "observations": observations,
    }
