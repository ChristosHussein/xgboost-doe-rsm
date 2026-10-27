"""xgb_doe_xai.benchmarks - XGBoost training harness, Derringer-Suich desirability, and matched TPE benchmarking.

Addresses Idea 4:
Enforces strictly equal evaluation budget and identical warmed latency profiling
between Sequential DoE and Multi-Objective TPE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import f1_score, log_loss

from xgb_doe_xai.config import DEFAULT_FACTORS, uncode_configuration, canonical_configuration
from xgb_doe_xai.profiler import LatencyProfileResult, WarmedLatencyProfiler


@dataclass(frozen=True)
class EvaluationResult:
    config_id: str
    coded_params: Dict[str, float]
    natural_params: Dict[str, float | int]
    val_macro_f1: float
    val_loss: float  # 1.0 - val_macro_f1
    val_logloss: float
    latency: LatencyProfileResult
    block_id: int = 0
    n_trees: int = 100
    seed: int = 42
    training_seconds: float = 0.
    profiling_seconds: float = 0.
    validation_labels: list = field(default_factory=list)
    validation_probabilities: list = field(default_factory=list)
    validation_probability_dtype: str = 'float64'


def train_and_evaluate_config(
    coded_params: Dict[str, float],
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    profiler: WarmedLatencyProfiler,
    config_id: str = "eval_01",
    block_id: int = 0,
    n_trees: int = 100,
    random_seed: int = 42,
) -> EvaluationResult:
    """Train XGBoost binary classifier and profile warmed inference latency."""
    coded_params = canonical_configuration(coded_params)
    nat = uncode_configuration(coded_params)
    started = time.perf_counter()

    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": float(nat["learning_rate"]),
        "max_depth": int(nat["max_depth"]),
        "subsample": float(nat["subsample"]),
        "colsample_bytree": float(nat["colsample_bytree"]),
        "n_estimators": n_trees,
        "random_state": random_seed,
        "n_jobs": 1,
    }

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train)

    y_val_probs = clf.predict_proba(X_val)[:, 1]
    y_val_preds = (y_val_probs >= 0.5).astype(int)

    f1 = float(f1_score(y_val, y_val_preds, average="macro"))
    ll = float(log_loss(y_val, y_val_probs))
    training_seconds = time.perf_counter() - started

    # Single-sample warmed latency profiling on a representative slice
    sample_slice = X_val[:1]
    started = time.perf_counter()
    latency_res = profiler.profile_booster(clf.get_booster(), sample_slice)

    return EvaluationResult(
        config_id=config_id,
        coded_params=coded_params,
        natural_params=nat,
        val_macro_f1=f1,
        val_loss=1.0 - f1,
        val_logloss=ll,
        latency=latency_res,
        block_id=block_id,
        n_trees=n_trees, seed=random_seed,
        training_seconds=training_seconds,
        profiling_seconds=time.perf_counter()-started,
        validation_labels=np.asarray(y_val, dtype=int).tolist(),
        validation_probabilities=y_val_probs.tolist(),
        validation_probability_dtype=str(y_val_probs.dtype),
    )


def compute_derringer_suich_desirability(
    val_loss: float,
    latency_us: float,
    loss_bounds: Tuple[float, float] = (0.05, 0.50),
    latency_bounds: Tuple[float, float] = (80.0, 200.0),
    weights: Tuple[float, float] = (1.0, 1.0),
) -> float:
    """Compute Derringer-Suich multi-objective desirability index D in [0, 1]."""
    l_min, l_max = loss_bounds
    t_min, t_max = latency_bounds
    w1, w2 = weights

    # Individual desirabilities (smaller is better)
    if val_loss <= l_min:
        d1 = 1.0
    elif val_loss >= l_max:
        d1 = 0.0
    else:
        d1 = ((l_max - val_loss) / (l_max - l_min)) ** w1

    if latency_us <= t_min:
        d2 = 1.0
    elif latency_us >= t_max:
        d2 = 0.0
    else:
        d2 = ((t_max - latency_us) / (t_max - t_min)) ** w2

    # Overall geometric mean
    return float((d1 * d2) ** (1.0 / (w1 + w2)))


def extract_pareto_front(
    evaluations: List[EvaluationResult],
) -> List[EvaluationResult]:
    """Extract non-dominated Pareto frontier minimizing both val_loss and predict_median_us."""
    pareto: List[EvaluationResult] = []
    for cand in evaluations:
        dominated = False
        c_loss = cand.val_loss
        c_lat = cand.latency.predict_median_us
        for other in evaluations:
            o_loss = other.val_loss
            o_lat = other.latency.predict_median_us
            if (o_loss <= c_loss and o_lat <= c_lat) and (o_loss < c_loss or o_lat < c_lat):
                dominated = True
                break
        if not dominated:
            pareto.append(cand)

    # Sort by loss ascending
    pareto.sort(key=lambda e: e.val_loss)
    return pareto


def compute_hypervolume_2d(
    pareto_points: List[Tuple[float, float]],
    reference_point: Tuple[float, float] = (0.50, 250.0),
) -> float:
    """Compute 2D Pareto hypervolume bounded by reference_point (loss, latency)."""
    ref_loss, ref_lat = reference_point
    valid = [p for p in pareto_points if p[0] <= ref_loss and p[1] <= ref_lat]
    if not valid:
        return 0.0

    valid.sort(key=lambda p: p[0])
    hv = 0.0
    curr_lat = ref_lat

    for loss, lat in valid:
        if lat < curr_lat:
            width = ref_loss - loss
            height = curr_lat - lat
            hv += width * height
            curr_lat = lat

    return float(hv)


def run_matched_motpe_benchmark(
    n_trials: int,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
    profiler: WarmedLatencyProfiler,
    random_seed: int = 42,
    n_trees: int = 100,
) -> List[EvaluationResult]:
    """Run Optuna Multi-Objective TPE under identical evaluation budget and warmed latency profiler."""
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    tpe_results: List[EvaluationResult] = []

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        coded = {
            "x1": trial.suggest_float("x1", -1.0, 1.0),
            "x2": (trial.suggest_int("max_depth", 3, 9) - 6) / 3,
            "x3": trial.suggest_float("x3", -1.0, 1.0),
            "x4": trial.suggest_float("x4", -1.0, 1.0),
        }
        coded = canonical_configuration(coded)
        res = train_and_evaluate_config(
            coded_params=coded,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            profiler=profiler,
            config_id=f"TPE_{trial.number:03d}",
            random_seed=random_seed + trial.number,
            n_trees=n_trees,
        )
        tpe_results.append(res)
        return res.val_loss, res.latency.predict_median_us

    # Disable optuna logs to maintain clean stdout
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)
    return tpe_results
