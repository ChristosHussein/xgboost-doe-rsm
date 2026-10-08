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
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import CONFIG, CaliforniaHousingDataManager


def measure_config_latency(
    eta: float,
    depth: int,
    subsample: float,
    reg_lambda: float,
    seed: int = 42,
    data_mgr: CaliforniaHousingDataManager = None,
    warmup_calls: int = 50,
    total_calls: int = 1000,
    reps: int = 5,
) -> Dict[str, Any]:
    if data_mgr is None:
        data_mgr = CaliforniaHousingDataManager()

    X_train, X_val, X_test, y_train, y_val, y_test = data_mgr.get_split(seed)

    model = xgb.XGBRegressor(
        n_estimators=CONFIG["model"]["n_estimators"],
        learning_rate=eta,
        max_depth=depth,
        subsample=subsample,
        reg_lambda=reg_lambda,
        random_state=seed,
        n_jobs=CONFIG["model"]["n_jobs_train"],
        objective=CONFIG["model"]["objective"],
    )
    model.fit(X_train, y_train)

    # Force single thread for latency testing
    model.set_params(n_jobs=1)
    booster = model.get_booster()
    booster.set_param({"nthread": 1})

    single_row = X_val[:1]
    batch_size = total_calls // reps

    # Warmup
    for _ in range(warmup_calls):
        _ = model.predict(single_row)
        _ = booster.inplace_predict(single_row)

    times_predict = []
    times_inplace = []

    # Interleaved measurement
    for _ in range(reps):
        # 1. Standard predict
        t0 = time.perf_counter_ns()
        for _ in range(batch_size):
            _ = model.predict(single_row)
        t1 = time.perf_counter_ns()
        times_predict.append((t1 - t0) / (batch_size * 1000.0))  # us

        # 2. Inplace predict
        t0 = time.perf_counter_ns()
        for _ in range(batch_size):
            _ = booster.inplace_predict(single_row)
        t1 = time.perf_counter_ns()
        times_inplace.append((t1 - t0) / (batch_size * 1000.0))  # us

    med_pred = float(np.median(times_predict))
    iqr_pred = float(np.subtract(*np.percentile(times_predict, [75, 25])))
    med_inp = float(np.median(times_inplace))
    iqr_inp = float(np.subtract(*np.percentile(times_inplace, [75, 25])))

    return {
        "depth": depth,
        "eta": eta,
        "subsample": subsample,
        "reg_lambda": reg_lambda,
        "predict_latency_us_median": med_pred,
        "predict_latency_us_iqr": iqr_pred,
        "inplace_latency_us_median": med_inp,
        "inplace_latency_us_iqr": iqr_inp,
    }


if __name__ == "__main__":
    data_mgr = CaliforniaHousingDataManager()
    print("Testing latency measurement for depth 4 vs depth 9...")
    res_d4 = measure_config_latency(0.23, 4, 1.0, 0.83, data_mgr=data_mgr)
    res_d9 = measure_config_latency(0.28, 9, 1.0, 0.10, data_mgr=data_mgr)
    print("Depth 4:", res_d4)
    print("Depth 9:", res_d9)
