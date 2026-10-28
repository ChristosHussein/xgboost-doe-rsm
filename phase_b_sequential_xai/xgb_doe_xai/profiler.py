"""xgb_doe_xai.profiler - Unified warmed latency profiling harness.

Addresses Remark 2:
Addresses differences between short online and longer benchmark protocols.
Historical online calls also had warmup. Enforces identical warmed measurement
across all search, surrogate, and benchmark stages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
import os
import platform
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import xgboost as xgb
from scipy import sparse
import sys


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


def get_process_affinity():
    """Capture the caller's CPU allowance before a single-core measurement."""
    if hasattr(os,'sched_getaffinity'):
        return os.sched_getaffinity(0)
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    process_mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(process_mask),ctypes.byref(system_mask)):
        raise OSError('Cannot capture process affinity')
    return process_mask.value


def restore_process_affinity(mask):
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,mask)
        return
    import ctypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),mask):
        raise OSError('Cannot restore process affinity')


@dataclass(frozen=True)
class LatencyProfileResult:
    predict_median_us: float
    predict_mean_us: float
    predict_iqm_us: float
    predict_sd_us: float
    predict_p95_us: float
    inplace_median_us: Optional[float]
    inplace_mean_us: Optional[float]
    inplace_iqm_us: Optional[float]
    n_warmup: int
    n_timed: int
    batch_size: int
    predict_raw_us: list = field(default_factory=list)
    inplace_raw_us: list = field(default_factory=list)
    affinity_applied: bool = False
    cpu_core: Optional[int] = None
    environment: dict = field(default_factory=dict)
    predict_operation: str = 'fresh DMatrix construction plus Booster.predict; uncached'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_iqm(values: np.ndarray) -> float:
    """Compute the interquartile mean (trimmed mean between Q1 and Q3)."""
    q25 = np.percentile(values, 25)
    q75 = np.percentile(values, 75)
    middle = values[(values >= q25) & (values <= q75)]
    if len(middle) == 0:
        return float(np.mean(values))
    return float(np.mean(middle))


class WarmedLatencyProfiler:
    """Standardized latency profiler ensuring identical execution protocol across all phases."""

    def __init__(
        self,
        n_warmup: int = 100,
        n_timed: int = 500,
        measure_inplace: bool = True,
        cpu_core: Optional[int] = None,
        require_affinity: bool = True,
    ) -> None:
        if n_warmup < 1 or n_timed < 2:
            raise ValueError('Profiling requires positive warmup and at least two timed calls')
        self.cpu_core = cpu_core
        self.require_affinity = require_affinity
        self.n_warmup = n_warmup
        self.n_timed = n_timed
        self.measure_inplace = measure_inplace

    def profile_booster(self,booster,sample_input):
        original=get_process_affinity()
        try:
            return self._profile_booster_pinned(booster,sample_input)
        finally:
            restore_process_affinity(original)

    def _profile_booster_pinned(
        self,
        booster: xgb.Booster,
        sample_input: Union[np.ndarray, sparse.csr_matrix],
    ) -> LatencyProfileResult:
        """Profile inference latency of a trained XGBoost booster on sample input."""
        batch_size = sample_input.shape[0] if hasattr(sample_input, "shape") else 1
        if batch_size != 1:
            raise ValueError('Single-document latency requires exactly one row')
        core = self.cpu_core
        if core is None:
            core = min(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 0
        affinity_applied = pin_cpu_affinity(core)
        if self.require_affinity and not affinity_applied:
            raise RuntimeError('Required single-core affinity could not be applied')
        booster.set_param({'nthread': 1, 'device': 'cpu'})
        # A new DMatrix for every call prevents warmed prediction-cache hits.
        # Construction is intentionally included: this is the uncached classic
        # API operation, distinct from inplace_predict on an existing CSR row.
        def predict():
            return booster.predict(xgb.DMatrix(sample_input, nthread=1))

        # 1. Warm-up phase (discards JIT, memory allocation, cache misses)
        for _ in range(self.n_warmup):
            _ = predict()

        # 2. Timed phase: standard predict(DMatrix)
        durations_predict_ns = np.empty(self.n_timed, dtype=np.int64)
        for i in range(self.n_timed):
            t0 = time.perf_counter_ns()
            _ = predict()
            t1 = time.perf_counter_ns()
            durations_predict_ns[i] = t1 - t0

        durations_predict_us = durations_predict_ns / 1000.0

        # 3. Optional timed phase: zero-copy inplace_predict
        durations_inplace_us: Optional[np.ndarray] = None
        if self.measure_inplace:
            try:
                # Warm-up inplace
                for _ in range(self.n_warmup):
                    _ = booster.inplace_predict(sample_input)

                durations_inplace_ns = np.empty(self.n_timed, dtype=np.int64)
                for i in range(self.n_timed):
                    t0 = time.perf_counter_ns()
                    _ = booster.inplace_predict(sample_input)
                    t1 = time.perf_counter_ns()
                    durations_inplace_ns[i] = t1 - t0
                durations_inplace_us = durations_inplace_ns / 1000.0
            except Exception as exc:
                raise RuntimeError('Required inplace prediction measurement failed') from exc

        res = LatencyProfileResult(
            predict_median_us=float(np.median(durations_predict_us)),
            predict_mean_us=float(np.mean(durations_predict_us)),
            predict_iqm_us=_compute_iqm(durations_predict_us),
            predict_sd_us=float(np.std(durations_predict_us, ddof=1)),
            predict_p95_us=float(np.percentile(durations_predict_us, 95)),
            inplace_median_us=float(np.median(durations_inplace_us)) if durations_inplace_us is not None else None,
            inplace_mean_us=float(np.mean(durations_inplace_us)) if durations_inplace_us is not None else None,
            inplace_iqm_us=_compute_iqm(durations_inplace_us) if durations_inplace_us is not None else None,
            n_warmup=self.n_warmup,
            n_timed=self.n_timed,
            batch_size=batch_size,
            predict_raw_us=durations_predict_us.tolist(),
            inplace_raw_us=durations_inplace_us.tolist() if durations_inplace_us is not None else [],
            affinity_applied=affinity_applied, cpu_core=core,
            environment={'platform': platform.platform(), 'processor': platform.processor(),
                         'xgboost': xgb.__version__, 'clock': 'perf_counter_ns'},
        )
        return res
