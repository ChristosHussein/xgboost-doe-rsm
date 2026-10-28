"""
scripts/measure_latency.py - Controlled latency benchmarking across configurations.
Evaluates single-sample inference latency:
  1. Standard predict() with DMatrix overhead.
  2. booster.inplace_predict() without DMatrix overhead.
Conditions:
  - Single-threaded (nthread=1).
  - 50 warmup calls.
  - 5 interleaved repetitions of 200 calls (1000 calls total).
  - Reports median and IQR.
"""

import os
import sys
from typing import Dict, Any
import xgboost as xgb

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from latency import PRIMARY_V1, measure_latencies
from pipeline import CONFIG, CaliforniaHousingDevelopmentDataManager


def measure_config_latency(
    eta: float,
    depth: int,
    subsample: float,
    reg_lambda: float,
    seed: int = 42,
    data_mgr: CaliforniaHousingDevelopmentDataManager = None,
    warmup_calls: int = 50,
    total_calls: int = 1000,
    reps: int = 5,
    session_id: str = "revision-v2-measure-config",
) -> Dict[str, Any]:
    if data_mgr is None:
        data_mgr = CaliforniaHousingDevelopmentDataManager()

    split = data_mgr.get_split(seed)

    model = xgb.XGBRegressor(
        n_estimators=CONFIG["model"]["n_estimators"],
        learning_rate=eta,
        max_depth=depth,
        subsample=subsample,
        reg_lambda=reg_lambda,
        colsample_bytree=CONFIG["model"].get("colsample_bytree", 1.0),
        min_child_weight=CONFIG["model"].get("min_child_weight", 1.0),
        gamma=CONFIG["model"].get("gamma", 0.0),
        tree_method=CONFIG["model"].get("tree_method", "auto"),
        random_state=seed,
        n_jobs=CONFIG["model"]["n_jobs_train"],
        objective=CONFIG["model"]["objective"],
    )
    model.fit(split.X_train, split.y_train)
    protocol = PRIMARY_V1
    if (warmup_calls, total_calls, reps) != (
        PRIMARY_V1.warmup_calls_per_interface,
        PRIMARY_V1.timed_calls_per_interface,
        PRIMARY_V1.repetitions,
    ):
        from latency import LatencyProtocol

        if total_calls % reps:
            raise ValueError("total_calls must be divisible by reps")
        protocol = LatencyProtocol(
            protocol_id="single_sample_latency_measure_config_custom_v1",
            interfaces=PRIMARY_V1.interfaces,
            warmup_calls_per_interface=warmup_calls,
            repetitions=reps,
            calls_per_repetition=total_calls // reps,
            inference_threads=PRIMARY_V1.inference_threads,
            cpu_core=PRIMARY_V1.cpu_core,
            order_seed=PRIMARY_V1.order_seed,
            randomize_order=True,
            primary_metric=PRIMARY_V1.primary_metric,
        )
    measurement = measure_latencies(
        {"candidate": model},
        split.X_val[:1],
        session_id=session_id,
        protocol=protocol,
    )
    summary = measurement["summaries"]["candidate"]

    return {
        "depth": depth,
        "eta": eta,
        "subsample": subsample,
        "reg_lambda": reg_lambda,
        "predict_latency_us": summary["predict_latency_us"],
        "predict_latency_us_iqr": summary["predict_latency_us_iqr"],
        "inplace_predict_latency_us": summary["inplace_predict_latency_us"],
        "inplace_predict_latency_us_iqr": summary["inplace_predict_latency_us_iqr"],
        "predict_latency_us_median": summary["predict_latency_us"],
        "inplace_latency_us_median": summary["inplace_predict_latency_us"],
        "inplace_latency_us_iqr": summary["inplace_predict_latency_us_iqr"],
        "latency_metadata": measurement["metadata"],
        "latency_observations": measurement["observations"],
    }


if __name__ == "__main__":
    data_mgr = CaliforniaHousingDevelopmentDataManager()
    print("Testing latency measurement for depth 4 vs depth 9...")
    res_d4 = measure_config_latency(0.23, 4, 1.0, 0.83, data_mgr=data_mgr)
    res_d9 = measure_config_latency(0.28, 9, 1.0, 0.10, data_mgr=data_mgr)
    print("Depth 4:", res_d4)
    print("Depth 9:", res_d9)
