"""
pipeline.py - Rigorous Experimental Design Pipeline for XGBoost HPO
===================================================================
Ground rules & Architectural specifications:
1. Fixed 20% External Holdout Test Set (4,128 rows) partitioned once with seed 42,
   completely untouched and unseen during all training and hyperparameter search.
2. Development pool (16,512 rows, 80%) deterministically partitioned into
   75% Train (12,384 rows = 60% total) and 25% Validation (4,128 rows = 20% total) per split seed.
3. Block controls data partition split seed: Block b uses base seed S_b.
4. Model random_state:
   - Factorial & Axial runs: split_seed = S_b, model_seed = S_b.
   - Center point replicates r in {1..4}: split_seed = S_b (same block data fold!),
     model_seed = S_b + r * 1000 (isolating internal subsampling stochasticity at subsample=0.75).
5. 25 unique geometric points in CCD (16 factorial, 1 center, 8 axial).
6. Randomized run execution order (tracked via run_order).
7. Dual responses: Validation RMSE (Y1) and microsecond inference latency (Y2).
8. Pinned CPU affinity and single-threaded inference for low-jitter latency measurements.
"""

import os
import sys
import time
from typing import Dict, Any, List, NamedTuple, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import yaml

# Pinned CPU core affinity and elevated process priority
def pin_cpu_affinity():
    """Pins execution to CPU 0 and sets ABOVE_NORMAL_PRIORITY_CLASS on Windows."""
    try:
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_bool

        handle = kernel32.GetCurrentProcess()
        kernel32.SetProcessAffinityMask(handle, 1)
        kernel32.SetPriorityClass(handle, 0x00008000)
    except Exception:
        pass

pin_cpu_affinity()

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

BLOCK_SEEDS = CONFIG["seeds"]["block_seeds"]
CENTER_SEEDS = {int(k): v for k, v in CONFIG["seeds"]["center_replicate_seeds"].items()}
RUN_ORDER_SEED = CONFIG["seeds"]["run_order_seed"]

# Factor Coding definitions
LN_ETA_MIN = np.log(CONFIG["factors"]["x1"]["min"])
LN_ETA_MAX = np.log(CONFIG["factors"]["x1"]["max"])
LN_ETA_MID = (LN_ETA_MAX + LN_ETA_MIN) / 2.0
LN_ETA_HALF = (LN_ETA_MAX - LN_ETA_MIN) / 2.0

DEPTH_MIN = float(CONFIG["factors"]["x2"]["min"])
DEPTH_MAX = float(CONFIG["factors"]["x2"]["max"])
DEPTH_MID = (DEPTH_MAX + DEPTH_MIN) / 2.0
DEPTH_HALF = (DEPTH_MAX - DEPTH_MIN) / 2.0

SUB_MIN = float(CONFIG["factors"]["x3"]["min"])
SUB_MAX = float(CONFIG["factors"]["x3"]["max"])
SUB_MID = (SUB_MAX + SUB_MIN) / 2.0
SUB_HALF = (SUB_MAX - SUB_MIN) / 2.0

LN_LAM_MIN = np.log(CONFIG["factors"]["x4"]["min"])
LN_LAM_MAX = np.log(CONFIG["factors"]["x4"]["max"])
LN_LAM_MID = (LN_LAM_MAX + LN_LAM_MIN) / 2.0
LN_LAM_HALF = (LN_LAM_MAX - LN_LAM_MIN) / 2.0


def encode_factors(eta: float, depth: float, subsample: float, reg_lambda: float) -> np.ndarray:
    """Transforms natural hyperparameters to standard coded space [-1, 1]."""
    x1 = (np.log(eta) - LN_ETA_MID) / LN_ETA_HALF
    x2 = (float(depth) - DEPTH_MID) / DEPTH_HALF
    x3 = (subsample - SUB_MID) / SUB_HALF
    x4 = (np.log(reg_lambda) - LN_LAM_MID) / LN_LAM_HALF
    return np.array([x1, x2, x3, x4], dtype=float)


def decode_factors(x: np.ndarray, clip_domain: bool = True) -> Tuple[float, int, float, float]:
    """
    Transforms coded variables x in [-1, 1] to natural hyperparameters.
    If clip_domain=True, clips to declared physical limits.
    """
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    if clip_domain:
        x1 = float(np.clip(x1, -1.0, 1.0))
        x2 = float(np.clip(x2, -1.0, 1.0))
        x3 = float(np.clip(x3, -1.0, 1.0))
        x4 = float(np.clip(x4, -1.0, 1.0))

    eta = float(np.exp(LN_ETA_MID + x1 * LN_ETA_HALF))
    depth = int(np.round(DEPTH_MID + x2 * DEPTH_HALF))
    depth = max(3, min(9, depth))
    subsample = float(SUB_MID + x3 * SUB_HALF)
    if clip_domain:
        subsample = max(0.50, min(1.00, subsample))
    reg_lambda = float(np.exp(LN_LAM_MID + x4 * LN_LAM_HALF))

    return eta, depth, subsample, reg_lambda


class CaliforniaHousingDataManager:
    """
    Manages California Housing dataset caching:
    - Fixed external holdout test set (20%, 4,128 rows) partitioned once with seed 42.
    - Development pool (80%, 16,512 rows) partitioned per split seed into:
        - 75% Train (12,384 rows = 60% of total 20,640)
        - 25% Validation (4,128 rows = 20% of total 20,640)
    - Zero data leakage: External holdout test set is never seen during training or tuning.
    """

    def __init__(self, external_test_seed: int = 42):
        housing = fetch_california_housing()
        self.X = housing.data
        self.y = housing.target
        self.feature_names = housing.feature_names

        # Fixed external test set carved out once
        self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
            self.X, self.y, test_size=0.20, random_state=external_test_seed
        )
        self._cache = {}

    def get_split(self, split_seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates 75% Train, 25% Validation from development pool based on split_seed.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test.
        """
        if split_seed in self._cache:
            return self._cache[split_seed]

        X_train, X_val, y_train, y_val = train_test_split(
            self.X_dev, self.y_dev, test_size=0.25, random_state=split_seed
        )
        split = (X_train, X_val, self.X_test, y_train, y_val, self.y_test)
        self._cache[split_seed] = split
        return split


class DevelopmentSplit(NamedTuple):
    """Data available to development-stage code; no external holdout fields exist."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray


class CaliforniaHousingDevelopmentDataManager:
    """Expose only train/validation data from the California Housing development pool."""

    def __init__(
        self,
        external_test_seed: int = 42,
        X: np.ndarray = None,
        y: np.ndarray = None,
    ):
        if X is None or y is None:
            housing = fetch_california_housing()
            raw_X = housing.data
            raw_y = housing.target
            self.feature_names = housing.feature_names
        else:
            raw_X = np.asarray(X)
            raw_y = np.asarray(y)
            self.feature_names = None

        self._X_dev, _, self._y_dev, _ = train_test_split(
            raw_X, raw_y, test_size=0.20, random_state=external_test_seed
        )
        self._cache = {}

    def get_split(self, split_seed: int) -> DevelopmentSplit:
        if split_seed not in self._cache:
            X_train, X_val, y_train, y_val = train_test_split(
                self._X_dev, self._y_dev, test_size=0.25, random_state=split_seed
            )
            self._cache[split_seed] = DevelopmentSplit(X_train, X_val, y_train, y_val)
        return self._cache[split_seed]


def evaluate_development_model(
    eta: float,
    depth: int,
    subsample: float,
    reg_lambda: float,
    seed: int = 42,
    data_mgr: CaliforniaHousingDevelopmentDataManager = None,
    measure_latency_details: bool = True,
    split_seed: int = None,
    model_seed: int = None,
) -> Dict[str, float]:
    """Fit and evaluate a model using development data only."""
    if data_mgr is None:
        data_mgr = CaliforniaHousingDevelopmentDataManager()
    if split_seed is None:
        split_seed = seed
    if model_seed is None:
        model_seed = seed

    split = data_mgr.get_split(split_seed)
    t0_fit = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=CONFIG["model"]["n_estimators"],
        learning_rate=eta,
        max_depth=depth,
        subsample=subsample,
        reg_lambda=reg_lambda,
        random_state=model_seed,
        n_jobs=CONFIG["model"]["n_jobs_train"],
        objective=CONFIG["model"]["objective"],
    )
    model.fit(split.X_train, split.y_train)
    fit_duration = time.perf_counter() - t0_fit
    val_pred = model.predict(split.X_val)
    val_rmse = float(np.sqrt(mean_squared_error(split.y_val, val_pred)))

    latency_median = latency_iqr = inplace_median = inplace_iqr = 0.0
    if measure_latency_details:
        pin_cpu_affinity()
        model.set_params(n_jobs=1)
        booster = model.get_booster()
        booster.set_param({"nthread": 1})
        single_sample = split.X_val[:1]
        for _ in range(CONFIG["model"]["latency_warmup"]):
            model.predict(single_sample)
            booster.inplace_predict(single_sample)

        reps = CONFIG["model"]["latency_reps"]
        batch_size = CONFIG["model"]["latency_iters"] // reps
        predict_batches = []
        inplace_batches = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                model.predict(single_sample)
            predict_batches.append((time.perf_counter_ns() - t0) / (batch_size * 1000.0))

            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                booster.inplace_predict(single_sample)
            inplace_batches.append((time.perf_counter_ns() - t0) / (batch_size * 1000.0))

        latency_median = float(np.median(predict_batches))
        latency_iqr = float(np.subtract(*np.percentile(predict_batches, [75, 25])))
        inplace_median = float(np.median(inplace_batches))
        inplace_iqr = float(np.subtract(*np.percentile(inplace_batches, [75, 25])))

    return {
        "val_rmse": val_rmse,
        "latency_us_median": latency_median,
        "latency_us_iqr": latency_iqr,
        "inplace_latency_us_median": inplace_median,
        "inplace_latency_us_iqr": inplace_iqr,
        "fit_time_s": fit_duration,
    }


def evaluate_model(
    eta: float,
    depth: int,
    subsample: float,
    reg_lambda: float,
    seed: int = 42,
    data_mgr: CaliforniaHousingDataManager = None,
    measure_latency_details: bool = True,
    split_seed: int = None,
    model_seed: int = None,
) -> Dict[str, float]:
    """
    Fits XGBoost regressor and evaluates:
      - val_rmse: Validation RMSE (Y1 objective)
      - test_rmse: Test RMSE (Generalization check on fixed external holdout set)
      - latency_us_median, latency_us_iqr: Single-sample latency (Y2 objective)
      - inplace_latency_us_median: Single-sample latency via inplace_predict
      - fit_time_s: Total fitting duration
    """
    if data_mgr is None:
        data_mgr = CaliforniaHousingDataManager()

    if split_seed is None:
        split_seed = seed
    if model_seed is None:
        model_seed = seed

    X_train, X_val, X_test, y_train, y_val, y_test = data_mgr.get_split(split_seed)

    t0_fit = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=CONFIG["model"]["n_estimators"],
        learning_rate=eta,
        max_depth=depth,
        subsample=subsample,
        reg_lambda=reg_lambda,
        random_state=model_seed,
        n_jobs=CONFIG["model"]["n_jobs_train"],
        objective=CONFIG["model"]["objective"],
    )
    model.fit(X_train, y_train)
    fit_duration = time.perf_counter() - t0_fit

    # Y1: Validation RMSE
    val_pred = model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_pred)))

    # Test RMSE (unseen fixed external holdout test set)
    test_pred = model.predict(X_test)
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))

    latency_median = 0.0
    latency_iqr = 0.0
    inplace_median = 0.0
    inplace_iqr = 0.0

    if measure_latency_details:
        pin_cpu_affinity()
        model.set_params(n_jobs=1)
        booster = model.get_booster()
        booster.set_param({"nthread": 1})
        single_sample = X_val[:1]

        # Warmup
        warmup_calls = CONFIG["model"]["latency_warmup"]
        for _ in range(warmup_calls):
            _ = model.predict(single_sample)
            _ = booster.inplace_predict(single_sample)

        # Timed repetitions (5 batches of 200 calls = 1000 calls)
        reps = CONFIG["model"]["latency_reps"]
        batch_size = CONFIG["model"]["latency_iters"] // reps
        batch_times_pred = []
        batch_times_inp = []

        for _ in range(reps):
            # 1. Standard predict
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                _ = model.predict(single_sample)
            t1 = time.perf_counter_ns()
            batch_times_pred.append((t1 - t0) / (batch_size * 1000.0))

            # 2. Inplace predict
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                _ = booster.inplace_predict(single_sample)
            t1 = time.perf_counter_ns()
            batch_times_inp.append((t1 - t0) / (batch_size * 1000.0))

        latency_median = float(np.median(batch_times_pred))
        latency_iqr = float(np.subtract(*np.percentile(batch_times_pred, [75, 25])))
        inplace_median = float(np.median(batch_times_inp))
        inplace_iqr = float(np.subtract(*np.percentile(batch_times_inp, [75, 25])))

    return {
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "latency_us_median": latency_median,
        "latency_us_iqr": latency_iqr,
        "inplace_latency_us_median": inplace_median,
        "inplace_latency_us_iqr": inplace_iqr,
        "fit_time_s": fit_duration,
    }


def generate_design_plan() -> List[Dict[str, Any]]:
    """
    Constructs the 140 design runs:
    - 25 unique geometric points:
      - 1..16: 2^4 factorial points
      - 17: center point (replicated 4 times per block)
      - 18..25: 8 axial (star) points
    - Evaluated across 5 blocks:
      - Block b defines data split seed S_b.
      - Center replicates r in {1..4} share split seed S_b but vary model_seed = S_b + r * 1000,
        isolating within-block stochastic subsampling pure error (15 df) from between-block variance.
    - Run order randomized deterministically using RUN_ORDER_SEED.
    """
    corner_coords = []
    for d in [-1.0, 1.0]:
        for c in [-1.0, 1.0]:
            for b in [-1.0, 1.0]:
                for a in [-1.0, 1.0]:
                    corner_coords.append((a, b, c, d))

    axial_coords = [
        (-1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0), (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, -1.0), (0.0, 0.0, 0.0, 1.0),
    ]

    runs = []
    run_id = 1

    for block_idx in range(1, 6):
        base_seed = BLOCK_SEEDS[block_idx - 1]

        # Phase 1: 16 Factorial Points (point_ids 1 to 16)
        for pt_idx, (a, b, c, d) in enumerate(corner_coords, start=1):
            runs.append({
                "run_id": run_id,
                "phase": "Phase1_Factorial",
                "point_id": pt_idx,
                "block": block_idx,
                "split_seed": base_seed,
                "model_seed": base_seed,
                "seed": base_seed,
                "replicate": 1,
                "x1": a, "x2": b, "x3": c, "x4": d,
            })
            run_id += 1

        # Phase 1: 4 Center Points (point_id 17, replicates 1..4)
        c_seeds = CENTER_SEEDS[block_idx]
        for rep_idx in range(1, 5):
            runs.append({
                "run_id": run_id,
                "phase": "Phase1_Center",
                "point_id": 17,
                "block": block_idx,
                "split_seed": base_seed,
                "model_seed": c_seeds[rep_idx - 1],
                "seed": c_seeds[rep_idx - 1],
                "replicate": rep_idx,
                "x1": 0.0, "x2": 0.0, "x3": 0.0, "x4": 0.0,
            })
            run_id += 1

        # Phase 2: 8 Axial Points (point_ids 18 to 25)
        for pt_idx, (a, b, c, d) in enumerate(axial_coords, start=18):
            runs.append({
                "run_id": run_id,
                "phase": "Phase2_Axial",
                "point_id": pt_idx,
                "block": block_idx,
                "split_seed": base_seed,
                "model_seed": base_seed,
                "seed": base_seed,
                "replicate": 1,
                "x1": a, "x2": b, "x3": c, "x4": d,
            })
            run_id += 1

    # Randomize execution run order
    rng = np.random.RandomState(RUN_ORDER_SEED)
    perm = rng.permutation(len(runs))
    for order_idx, run_idx in enumerate(perm, start=1):
        runs[run_idx]["run_order"] = order_idx

    return runs


def execute_design_pipeline(
    output_csv: str = "results/revision_v2/development_runs.csv",
    runs_plan: List[Dict[str, Any]] = None,
    data_mgr: CaliforniaHousingDevelopmentDataManager = None,
) -> pd.DataFrame:
    """Execute development-only DOE runs without accessing the external holdout."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    if runs_plan is None:
        runs_plan = generate_design_plan()

    # Sort by randomized run_order for real execution sequence
    runs_sorted = sorted(runs_plan, key=lambda r: r["run_order"])

    if data_mgr is None:
        data_mgr = CaliforniaHousingDevelopmentDataManager()
    executed_records = []

    print(f"Starting execution of {len(runs_sorted)} DOE design runs...")
    t_start = time.time()

    for idx, run in enumerate(runs_sorted, start=1):
        x = np.array([run["x1"], run["x2"], run["x3"], run["x4"]])
        eta, depth, subsample, reg_lambda = decode_factors(x)

        eval_res = evaluate_development_model(
            eta=eta,
            depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            split_seed=run["split_seed"],
            model_seed=run["model_seed"],
            data_mgr=data_mgr,
            measure_latency_details=True,
        )

        record = {
            "run_id": run["run_id"],
            "phase": run["phase"],
            "point_id": run["point_id"],
            "block": run["block"],
            "split_seed": run["split_seed"],
            "model_seed": run["model_seed"],
            "seed": run["seed"],
            "replicate": run["replicate"],
            "run_order": run["run_order"],
            "x1": run["x1"],
            "x2": run["x2"],
            "x3": run["x3"],
            "x4": run["x4"],
            "eta": eta,
            "depth": depth,
            "subsample": subsample,
            "reg_lambda": reg_lambda,
            "val_rmse": eval_res["val_rmse"],
            "latency_us_median": eval_res["latency_us_median"],
            "latency_us_iqr": eval_res["latency_us_iqr"],
            "inplace_latency_us_median": eval_res["inplace_latency_us_median"],
            "inplace_latency_us_iqr": eval_res["inplace_latency_us_iqr"],
            "fit_time_s": eval_res["fit_time_s"],
        }
        executed_records.append(record)

        if idx % 28 == 0 or idx == len(runs_sorted):
            elapsed = time.time() - t_start
            print(f"[{idx}/{len(runs_sorted)}] runs completed in {elapsed:.1f}s. Latest Val RMSE: {eval_res['val_rmse']:.4f}")

    # Restore original run_id ordering in DataFrame for clean storage
    df = pd.DataFrame(executed_records).sort_values("run_id").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved complete runs dataset ({len(df)} rows) to: {output_csv}")
    return df


if __name__ == "__main__":
    execute_design_pipeline()
