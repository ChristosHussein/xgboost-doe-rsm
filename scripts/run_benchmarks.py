"""Run the versioned revision benchmark and gated final evaluation.

The workflow records development-only optimizer trials, independently repeated
blocked DOE selections, matched timing protocols, frozen-selection evaluation,
and complete provenance. Smoke mode validates the machinery; full mode requires
an explicit computational-budget acknowledgement.
"""

import json
import hashlib
import io
import os
import platform
import sys
import time
import argparse
import subprocess
from importlib import metadata as importlib_metadata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import optuna
import statsmodels.api as sm
import xgboost as xgb
import yaml

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import (
    CaliforniaHousingDataManager,
    CaliforniaHousingDevelopmentDataManager,
    decode_factors,
    encode_factors,
    pin_cpu_affinity,
    CONFIG,
)
from analysis import derringer_suich_desirability
from latency import ONLINE_SEARCH_V1, PRIMARY_V1, measure_latencies
from final_evaluation import FinalTestEvaluator, create_finalized_selection
from scientific_stats import hypervolume_2d_min, paired_difference_summary, paired_tost

optuna.logging.set_verbosity(optuna.logging.WARNING)

FRESH_SEEDS = CONFIG["seeds"]["fresh_eval_seeds"]
OPTIMIZER_SEEDS = CONFIG["seeds"]["optimizer_sampler_seeds"]
HISTORICAL_BASELINE_REF = "v1.0.0"


@dataclass
class SearchResult:
    """One optimizer replicate with its actual selected trial and full trial ledger."""

    optimizer: str
    replicate_id: int
    sampler_seed: int
    development_split_seed: int
    status: str
    selected_x: Optional[np.ndarray]
    selected_score: Optional[float]
    selection_rule: str
    trajectory: List[float]
    trial_records: List[Dict[str, Any]]


def _development_split(data_mgr, split_seed: int):
    """Require the named development-only split interface."""
    split = data_mgr.get_split(split_seed)
    required = ("X_train", "X_val", "y_train", "y_val")
    missing = [name for name in required if not hasattr(split, name)]
    if missing:
        raise TypeError(f"development split is missing fields: {', '.join(missing)}")
    return split


def _model_config(
    learning_rate: float,
    max_depth: int,
    subsample: float,
    reg_lambda: float,
    random_state: int,
) -> Dict[str, Any]:
    """Build the declared common XGBoost configuration for every optimizer."""
    configured = CONFIG["model"]
    return {
        "n_estimators": int(configured["n_estimators"]),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "reg_lambda": float(reg_lambda),
        "colsample_bytree": float(configured.get("colsample_bytree", 1.0)),
        "min_child_weight": float(configured.get("min_child_weight", 1.0)),
        "gamma": float(configured.get("gamma", 0.0)),
        "tree_method": configured.get("tree_method", "auto"),
        "random_state": int(random_state),
        "n_jobs": int(configured["n_jobs_train"]),
        "objective": configured["objective"],
    }


def _natural_from_trial(trial) -> Tuple[np.ndarray, float, int, float, float]:
    """Sample the common natural search space, with depth explicitly integer-valued."""
    factor_cfg = CONFIG["factors"]
    eta = float(
        trial.suggest_float(
            "learning_rate",
            float(factor_cfg["x1"]["min"]),
            float(factor_cfg["x1"]["max"]),
            log=True,
        )
    )
    depth = int(
        trial.suggest_int(
            "max_depth",
            int(factor_cfg["x2"]["min"]),
            int(factor_cfg["x2"]["max"]),
        )
    )
    subsample = float(
        trial.suggest_float(
            "subsample",
            float(factor_cfg["x3"]["min"]),
            float(factor_cfg["x3"]["max"]),
        )
    )
    reg_lambda = float(
        trial.suggest_float(
            "reg_lambda",
            float(factor_cfg["x4"]["min"]),
            float(factor_cfg["x4"]["max"]),
            log=True,
        )
    )
    x = encode_factors(eta, depth, subsample, reg_lambda)
    return x, eta, depth, subsample, reg_lambda


def _trial_record(
    *,
    optimizer: str,
    optimizer_version: str,
    replicate_id: int,
    sampler_seed: int,
    development_split_seed: int,
    trial_number: int,
    x: np.ndarray,
    model_config: Dict[str, Any],
    validation_rmse: Optional[float],
    predict_latency_us: Optional[float],
    objective_value: Optional[Any],
    max_latency_us: Optional[float],
    training_time_s: float,
    evaluation_time_s: float,
    latency_session_id: Optional[str] = None,
    trial_status: str = "completed",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    latency = None if predict_latency_us is None else float(predict_latency_us)
    violation = (
        None
        if latency is None or max_latency_us is None
        else float(max(0.0, latency - max_latency_us))
    )
    feasible = None if max_latency_us is None else bool(violation == 0.0)
    return {
        "optimizer": optimizer,
        "optimizer_version": optimizer_version,
        "replicate_id": int(replicate_id),
        "sampler_seed": int(sampler_seed),
        "development_split_seed": int(development_split_seed),
        "trial_number": int(trial_number),
        "x1": float(x[0]),
        "x2": float(x[1]),
        "x3": float(x[2]),
        "x4": float(x[3]),
        "learning_rate": float(model_config["learning_rate"]),
        "max_depth": int(model_config["max_depth"]),
        "subsample": float(model_config["subsample"]),
        "reg_lambda": float(model_config["reg_lambda"]),
        "model_config": json.dumps(model_config, sort_keys=True),
        "validation_rmse": validation_rmse,
        "predict_latency_us": latency,
        "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id if latency is not None else None,
        "latency_session_id": latency_session_id,
        "objective_value": objective_value,
        "feasibility": feasible,
        "constraint_violation_us": violation,
        "training_time_s": float(training_time_s),
        "evaluation_time_s": float(evaluation_time_s),
        "trial_status": trial_status,
        "error": error,
        "selected": False,
        "selection_rule": None,
    }


def _mark_selected(
    records: List[Dict[str, Any]], selected: Optional[Dict[str, Any]], rule: str
) -> None:
    if selected is not None:
        selected["selected"] = True
        selected["selection_rule"] = rule


def _completed_result(
    optimizer: str,
    replicate_id: int,
    sampler_seed: int,
    eval_seed: int,
    records: List[Dict[str, Any]],
    selected: Optional[Dict[str, Any]],
    selected_score: Optional[float],
    rule: str,
    trajectory: List[float],
    status: str = "completed",
) -> SearchResult:
    _mark_selected(records, selected, rule)
    selected_x = None
    if selected is not None:
        selected_x = np.asarray(
            [selected["x1"], selected["x2"], selected["x3"], selected["x4"]],
            dtype=float,
        )
    return SearchResult(
        optimizer=optimizer,
        replicate_id=int(replicate_id),
        sampler_seed=int(sampler_seed),
        development_split_seed=int(eval_seed),
        status=status,
        selected_x=selected_x,
        selected_score=None if selected_score is None else float(selected_score),
        selection_rule=rule,
        trajectory=trajectory,
        trial_records=records,
    )


def evaluate_config_on_seeds(
    x: np.ndarray,
    seeds: List[int],
    data_mgr: CaliforniaHousingDataManager
) -> Dict[str, Any]:
    """Evaluates a single hyperparameter coordinate x across multiple seeds for Val and Test RMSE."""
    eta, depth, subsample, reg_lambda = decode_factors(x)
    val_rmses = []
    test_rmses = []

    for s in seeds:
        X_tr, X_val, X_te, y_tr, y_val, y_te = data_mgr.get_split(s)
        model = xgb.XGBRegressor(
            n_estimators=CONFIG["model"]["n_estimators"],
            learning_rate=eta,
            max_depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=s,
            n_jobs=CONFIG["model"]["n_jobs_train"],
            objective=CONFIG["model"]["objective"],
        )
        model.fit(X_tr, y_tr)
        val_rmses.append(float(np.sqrt(np.mean((y_val - model.predict(X_val))**2))))
        test_rmses.append(float(np.sqrt(np.mean((y_te - model.predict(X_te))**2))))

    m_val = float(np.mean(val_rmses))
    se_val = float(stats.sem(val_rmses))
    m_test = float(np.mean(test_rmses))
    se_test = float(stats.sem(test_rmses))
    t_crit = float(stats.t.ppf(0.975, len(seeds) - 1))

    return {
        "x": x.tolist(),
        "eta": eta,
        "depth": depth,
        "subsample": subsample,
        "reg_lambda": reg_lambda,
        "val_rmses": val_rmses,
        "test_rmses": test_rmses,
        "val_rmse_mean": m_val,
        "val_rmse_std": float(np.std(val_rmses)),
        "val_rmse_ci95": [m_val - t_crit * se_val, m_val + t_crit * se_val],
        "test_rmse_mean": m_test,
        "test_rmse_std": float(np.std(test_rmses)),
        "test_rmse_ci95": [m_test - t_crit * se_test, m_test + t_crit * se_test],
    }


def measure_interleaved_latencies(
    configs: Dict[str, Tuple[float, int, float, float]],
    data_mgr,
    reps: int = 5,
    batch_size: int = 200,
    warmup: int = 50,
    session_id: str = "revision-v2-benchmark",
) -> Dict[str, Dict[str, float]]:
    """Measure all configurations in one randomized, metadata-rich session."""
    split = _development_split(data_mgr, 42)
    single_sample = split.X_val[:1]

    models = {}
    for name, (eta, depth, subsample, reg_lambda) in configs.items():
        model_config = _model_config(eta, depth, subsample, reg_lambda, 42)
        model_config["n_jobs"] = 1
        model = xgb.XGBRegressor(**model_config)
        model.fit(split.X_train, split.y_train)
        models[name] = model

    protocol = PRIMARY_V1
    if (reps, batch_size, warmup) != (
        PRIMARY_V1.repetitions,
        PRIMARY_V1.calls_per_repetition,
        PRIMARY_V1.warmup_calls_per_interface,
    ):
        from latency import LatencyProtocol

        protocol = LatencyProtocol(
            protocol_id="single_sample_latency_primary_custom_v1",
            interfaces=PRIMARY_V1.interfaces,
            warmup_calls_per_interface=warmup,
            repetitions=reps,
            calls_per_repetition=batch_size,
            inference_threads=PRIMARY_V1.inference_threads,
            cpu_core=PRIMARY_V1.cpu_core,
            order_seed=PRIMARY_V1.order_seed,
            randomize_order=True,
            primary_metric=PRIMARY_V1.primary_metric,
        )
    measurement = measure_latencies(
        models, single_sample, session_id=session_id, protocol=protocol
    )
    results = {}
    for name, summary in measurement["summaries"].items():
        results[name] = {
            "predict_latency_us_median": summary["predict_latency_us"],
            "predict_latency_us_iqr": summary["predict_latency_us_iqr"],
            "inplace_latency_us_median": summary["inplace_predict_latency_us"],
            "inplace_latency_us_iqr": summary["inplace_predict_latency_us_iqr"],
            "latency_protocol_id": measurement["metadata"]["protocol"]["protocol_id"],
            "latency_session_id": session_id,
        }
    return results


def run_random_search(
    n_trials: int,
    sampler_seed: int,
    eval_seed: int,
    data_mgr,
    replicate_id: int = 0,
) -> SearchResult:
    """Uniform random search over the declared natural four-factor space."""
    rng = np.random.RandomState(sampler_seed)
    split = _development_split(data_mgr, eval_seed)
    factor_cfg = CONFIG["factors"]
    records: List[Dict[str, Any]] = []
    trajectory: List[float] = []
    best_record = None
    best_value = float("inf")

    for trial_number in range(n_trials):
        eta = float(
            np.exp(
                rng.uniform(
                    np.log(float(factor_cfg["x1"]["min"])),
                    np.log(float(factor_cfg["x1"]["max"])),
                )
            )
        )
        depth = int(
            rng.randint(
                int(factor_cfg["x2"]["min"]),
                int(factor_cfg["x2"]["max"]) + 1,
            )
        )
        subsample = float(
            rng.uniform(
                float(factor_cfg["x3"]["min"]),
                float(factor_cfg["x3"]["max"]),
            )
        )
        reg_lambda = float(
            np.exp(
                rng.uniform(
                    np.log(float(factor_cfg["x4"]["min"])),
                    np.log(float(factor_cfg["x4"]["max"])),
                )
            )
        )
        x = encode_factors(eta, depth, subsample, reg_lambda)
        model_config = _model_config(eta, depth, subsample, reg_lambda, eval_seed)
        train_start = time.perf_counter()
        try:
            model = xgb.XGBRegressor(**model_config)
            model.fit(split.X_train, split.y_train)
            training_time = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            prediction = model.predict(split.X_val)
            val_rmse = float(np.sqrt(np.mean((split.y_val - prediction) ** 2)))
            latency_session_id = f"random-search-rep-{replicate_id}-trial-{trial_number}"
            latency = measure_trial_latency(
                model,
                split.X_val[:1],
                session_id=latency_session_id,
            )
            evaluation_time = time.perf_counter() - eval_start
            record = _trial_record(
                optimizer="random_search",
                optimizer_version=f"numpy-{np.__version__}",
                replicate_id=replicate_id,
                sampler_seed=sampler_seed,
                development_split_seed=eval_seed,
                trial_number=trial_number,
                x=x,
                model_config=model_config,
                validation_rmse=val_rmse,
                predict_latency_us=latency,
                objective_value=val_rmse,
                max_latency_us=None,
                training_time_s=training_time,
                evaluation_time_s=evaluation_time,
                latency_session_id=latency_session_id,
            )
            if val_rmse < best_value:
                best_value = val_rmse
                best_record = record
        except Exception as exc:
            record = _trial_record(
                optimizer="random_search",
                optimizer_version=f"numpy-{np.__version__}",
                replicate_id=replicate_id,
                sampler_seed=sampler_seed,
                development_split_seed=eval_seed,
                trial_number=trial_number,
                x=x,
                model_config=model_config,
                validation_rmse=None,
                predict_latency_us=None,
                objective_value=None,
                max_latency_us=None,
                training_time_s=time.perf_counter() - train_start,
                evaluation_time_s=0.0,
                trial_status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)
        trajectory.append(best_value)

    rule = "minimum_validation_rmse"
    status = "completed" if best_record is not None else "failed"
    return _completed_result(
        "random_search",
        replicate_id,
        sampler_seed,
        eval_seed,
        records,
        best_record,
        None if best_record is None else best_value,
        rule,
        trajectory,
        status,
    )


def run_tpe_single_objective(
    n_trials: int,
    sampler_seed: int,
    eval_seed: int,
    data_mgr,
    replicate_id: int = 0,
) -> SearchResult:
    """TPE minimizing validation RMSE with an explicitly integer depth dimension."""
    split = _development_split(data_mgr, eval_seed)
    records: List[Dict[str, Any]] = []
    trajectory: List[float] = []
    best_value = float("inf")
    best_record = None

    def objective(trial):
        nonlocal best_value, best_record
        x, eta, depth, subsample, reg_lambda = _natural_from_trial(trial)
        model_config = _model_config(eta, depth, subsample, reg_lambda, eval_seed)
        train_start = time.perf_counter()
        try:
            model = xgb.XGBRegressor(**model_config)
            model.fit(split.X_train, split.y_train)
            training_time = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            prediction = model.predict(split.X_val)
            val_rmse = float(np.sqrt(np.mean((split.y_val - prediction) ** 2)))
            latency_session_id = f"single-tpe-rep-{replicate_id}-trial-{trial.number}"
            latency = measure_trial_latency(
                model,
                split.X_val[:1],
                session_id=latency_session_id,
            )
            evaluation_time = time.perf_counter() - eval_start
            record = _trial_record(
                optimizer="single_objective_tpe",
                optimizer_version=f"optuna-{optuna.__version__}",
                replicate_id=replicate_id,
                sampler_seed=sampler_seed,
                development_split_seed=eval_seed,
                trial_number=trial.number,
                x=x,
                model_config=model_config,
                validation_rmse=val_rmse,
                predict_latency_us=latency,
                objective_value=val_rmse,
                max_latency_us=None,
                training_time_s=training_time,
                evaluation_time_s=evaluation_time,
                latency_session_id=latency_session_id,
            )
            records.append(record)
            if val_rmse < best_value:
                best_value = val_rmse
                best_record = record
            trajectory.append(best_value)
            return val_rmse
        except Exception as exc:
            records.append(
                _trial_record(
                    optimizer="single_objective_tpe",
                    optimizer_version=f"optuna-{optuna.__version__}",
                    replicate_id=replicate_id,
                    sampler_seed=sampler_seed,
                    development_split_seed=eval_seed,
                    trial_number=trial.number,
                    x=x,
                    model_config=model_config,
                    validation_rmse=None,
                    predict_latency_us=None,
                    objective_value=None,
                    max_latency_us=None,
                    training_time_s=time.perf_counter() - train_start,
                    evaluation_time_s=0.0,
                    trial_status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            trajectory.append(best_value)
            return float("inf")

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    rule = "minimum_validation_rmse"
    status = "completed" if best_record is not None else "failed"
    return _completed_result(
        "single_objective_tpe",
        replicate_id,
        sampler_seed,
        eval_seed,
        records,
        best_record,
        None if best_record is None else best_value,
        rule,
        trajectory,
        status,
    )


def measure_trial_latency(
    model,
    sample: np.ndarray,
    warmup: int = 10,
    reps: int = 30,
    *,
    session_id: str = "revision-v2-online-search",
) -> float:
    """Measure online search latency through the centralized protocol."""
    protocol = ONLINE_SEARCH_V1
    if (warmup, reps) != (
        ONLINE_SEARCH_V1.warmup_calls_per_interface,
        ONLINE_SEARCH_V1.timed_calls_per_interface,
    ):
        from latency import LatencyProtocol

        protocol = LatencyProtocol(
            protocol_id="single_sample_latency_online_search_custom_v1",
            interfaces=("predict",),
            warmup_calls_per_interface=warmup,
            repetitions=1,
            calls_per_repetition=reps,
            inference_threads=ONLINE_SEARCH_V1.inference_threads,
            cpu_core=ONLINE_SEARCH_V1.cpu_core,
            order_seed=ONLINE_SEARCH_V1.order_seed,
            randomize_order=False,
            primary_metric=ONLINE_SEARCH_V1.primary_metric,
        )
    result = measure_latencies(
        {"trial": model}, sample, session_id=session_id, protocol=protocol
    )
    return float(result["summaries"]["trial"]["predict_latency_us"])


def run_tpe_constrained(
    n_trials: int,
    sampler_seed: int,
    eval_seed: int,
    max_latency_us: float,
    data_mgr,
    replicate_id: int = 0,
) -> SearchResult:
    """TPE with selection restricted to trials feasible by measured latency."""
    split = _development_split(data_mgr, eval_seed)
    records: List[Dict[str, Any]] = []
    trajectory: List[float] = []
    best_feasible = float("inf")

    def objective(trial):
        nonlocal best_feasible
        x, eta, depth, subsample, reg_lambda = _natural_from_trial(trial)
        model_config = _model_config(eta, depth, subsample, reg_lambda, eval_seed)
        train_start = time.perf_counter()
        try:
            model = xgb.XGBRegressor(**model_config)
            model.fit(split.X_train, split.y_train)
            training_time = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            prediction = model.predict(split.X_val)
            val_rmse = float(np.sqrt(np.mean((split.y_val - prediction) ** 2)))
            latency_session_id = f"constrained-tpe-rep-{replicate_id}-trial-{trial.number}"
            latency = measure_trial_latency(
                model,
                split.X_val[:1],
                session_id=latency_session_id,
            )
            evaluation_time = time.perf_counter() - eval_start
            violation = max(0.0, latency - max_latency_us)
            score = val_rmse + violation * 0.05
            record = _trial_record(
                optimizer="constrained_tpe",
                optimizer_version=f"optuna-{optuna.__version__}",
                replicate_id=replicate_id,
                sampler_seed=sampler_seed,
                development_split_seed=eval_seed,
                trial_number=trial.number,
                x=x,
                model_config=model_config,
                validation_rmse=val_rmse,
                predict_latency_us=latency,
                objective_value=score,
                max_latency_us=max_latency_us,
                training_time_s=training_time,
                evaluation_time_s=evaluation_time,
                latency_session_id=latency_session_id,
            )
            records.append(record)
            if record["feasibility"]:
                best_feasible = min(best_feasible, val_rmse)
            trajectory.append(best_feasible)
            return score
        except Exception as exc:
            records.append(
                _trial_record(
                    optimizer="constrained_tpe",
                    optimizer_version=f"optuna-{optuna.__version__}",
                    replicate_id=replicate_id,
                    sampler_seed=sampler_seed,
                    development_split_seed=eval_seed,
                    trial_number=trial.number,
                    x=x,
                    model_config=model_config,
                    validation_rmse=None,
                    predict_latency_us=None,
                    objective_value=None,
                    max_latency_us=max_latency_us,
                    training_time_s=time.perf_counter() - train_start,
                    evaluation_time_s=0.0,
                    trial_status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            trajectory.append(best_feasible)
            return float("inf")

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    feasible = [
        record
        for record in records
        if record["trial_status"] == "completed" and record["feasibility"] is True
    ]
    selected = min(feasible, key=lambda record: record["validation_rmse"]) if feasible else None
    rule = "minimum_validation_rmse_among_measured_feasible_trials"
    return _completed_result(
        "constrained_tpe",
        replicate_id,
        sampler_seed,
        eval_seed,
        records,
        selected,
        None if selected is None else selected["validation_rmse"],
        rule,
        trajectory,
        "completed" if selected is not None else "infeasible",
    )


def run_tpe_multi_objective(
    n_trials: int,
    sampler_seed: int,
    eval_seed: int,
    data_mgr,
    replicate_id: int = 0,
) -> SearchResult:
    """TPE minimizing validation RMSE and measured prediction latency."""
    split = _development_split(data_mgr, eval_seed)
    records: List[Dict[str, Any]] = []
    trajectory: List[float] = []
    best_desirability = float("-inf")
    L1 = float(CONFIG["desirability"]["Y1_RMSE"]["L"])
    U1 = float(CONFIG["desirability"]["Y1_RMSE"]["U"])
    L2 = float(CONFIG["desirability"]["Y2_Latency"]["L"])
    U2 = float(CONFIG["desirability"]["Y2_Latency"]["U"])

    def objective(trial):
        nonlocal best_desirability
        x, eta, depth, subsample, reg_lambda = _natural_from_trial(trial)
        model_config = _model_config(eta, depth, subsample, reg_lambda, eval_seed)
        train_start = time.perf_counter()
        try:
            model = xgb.XGBRegressor(**model_config)
            model.fit(split.X_train, split.y_train)
            training_time = time.perf_counter() - train_start
            eval_start = time.perf_counter()
            prediction = model.predict(split.X_val)
            val_rmse = float(np.sqrt(np.mean((split.y_val - prediction) ** 2)))
            latency_session_id = f"multi-tpe-rep-{replicate_id}-trial-{trial.number}"
            latency = measure_trial_latency(
                model,
                split.X_val[:1],
                session_id=latency_session_id,
            )
            evaluation_time = time.perf_counter() - eval_start
            _, _, desirability = derringer_suich_desirability(
                val_rmse, latency, L1, U1, L2, U2
            )
            record = _trial_record(
                optimizer="multi_objective_tpe",
                optimizer_version=f"optuna-{optuna.__version__}",
                replicate_id=replicate_id,
                sampler_seed=sampler_seed,
                development_split_seed=eval_seed,
                trial_number=trial.number,
                x=x,
                model_config=model_config,
                validation_rmse=val_rmse,
                predict_latency_us=latency,
                objective_value={
                    "validation_rmse": val_rmse,
                    "predict_latency_us": latency,
                    "desirability": float(desirability),
                },
                max_latency_us=None,
                training_time_s=training_time,
                evaluation_time_s=evaluation_time,
                latency_session_id=latency_session_id,
            )
            records.append(record)
            best_desirability = max(best_desirability, float(desirability))
            trajectory.append(best_desirability)
            return val_rmse, latency
        except Exception as exc:
            records.append(
                _trial_record(
                    optimizer="multi_objective_tpe",
                    optimizer_version=f"optuna-{optuna.__version__}",
                    replicate_id=replicate_id,
                    sampler_seed=sampler_seed,
                    development_split_seed=eval_seed,
                    trial_number=trial.number,
                    x=x,
                    model_config=model_config,
                    validation_rmse=None,
                    predict_latency_us=None,
                    objective_value=None,
                    max_latency_us=None,
                    training_time_s=time.perf_counter() - train_start,
                    evaluation_time_s=0.0,
                    trial_status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            trajectory.append(best_desirability)
            return float("inf"), float("inf")

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    completed = [record for record in records if record["trial_status"] == "completed"]
    rule = "maximum_predeclared_derringer_suich_desirability"
    selected = None
    selected_score = None
    if completed:
        selected = max(
            completed,
            key=lambda record: (
                record["objective_value"]["desirability"],
                -record["validation_rmse"],
                -record["predict_latency_us"],
            ),
        )
        selected_score = selected["objective_value"]["desirability"]
        if selected_score <= 0.0:
            rule = "minimum_normalized_distance_to_predeclared_desirability_ideal"
            selected = min(
                completed,
                key=lambda record: (
                    ((max(0.0, record["validation_rmse"] - L1) / (U1 - L1)) ** 2)
                    + ((max(0.0, record["predict_latency_us"] - L2) / (U2 - L2)) ** 2),
                    record["trial_number"],
                ),
            )
            selected_score = selected["objective_value"]["desirability"]
    return _completed_result(
        "multi_objective_tpe",
        replicate_id,
        sampler_seed,
        eval_seed,
        records,
        selected,
        selected_score,
        rule,
        trajectory,
        "completed" if selected is not None else "failed",
    )


def compute_hypervolume(points: List[Tuple[float, float]], ref_point: Tuple[float, float]) -> float:
    """Computes 2D hypervolume dominated by Pareto points against ref_point."""
    pts = sorted([p for p in points if p[0] <= ref_point[0] and p[1] <= ref_point[1]], key=lambda x: x[0])
    if not pts:
        return 0.0
    hv = 0.0
    cur_lat = ref_point[1]
    for rmse, lat in pts:
        if lat < cur_lat:
            hv += (ref_point[0] - rmse) * (cur_lat - lat)
            cur_lat = lat
    return float(hv)


def run_single_optimizer_replicate(
    sampler_seed: int,
    eval_seed: int = 42,
    *,
    replicate_id: int = 0,
    n_trials: int = 140,
    max_latency_us: float = 145.0,
    data_mgr=None,
) -> Dict[str, Any]:
    """Run all four optimizers against one development-only split."""
    dm = data_mgr or CaliforniaHousingDevelopmentDataManager()
    method_wall_times = {}
    started = time.perf_counter()
    rs = run_random_search(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    method_wall_times[rs.optimizer] = time.perf_counter() - started
    started = time.perf_counter()
    tpe = run_tpe_single_objective(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    method_wall_times[tpe.optimizer] = time.perf_counter() - started
    started = time.perf_counter()
    constrained = run_tpe_constrained(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        max_latency_us=max_latency_us,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    method_wall_times[constrained.optimizer] = time.perf_counter() - started
    started = time.perf_counter()
    multi = run_tpe_multi_objective(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    method_wall_times[multi.optimizer] = time.perf_counter() - started
    return {
        "seed": sampler_seed,
        "replicate_id": replicate_id,
        "search_results": {
            rs.optimizer: rs,
            tpe.optimizer: tpe,
            constrained.optimizer: constrained,
            multi.optimizer: multi,
        },
        "method_wall_times_s": method_wall_times,
        "trial_records": [
            record
            for result in (rs, tpe, constrained, multi)
            for record in result.trial_records
        ],
        "x_rs": rs.selected_x,
        "val_rs": rs.selected_score,
        "traj_rs": rs.trajectory,
        "x_tpe": tpe.selected_x,
        "val_tpe": tpe.selected_score,
        "traj_tpe": tpe.trajectory,
        "x_co": constrained.selected_x,
        "val_co": constrained.selected_score,
        "x_mo": multi.selected_x,
        "des_mo": multi.selected_score,
    }


def select_median_actual_incumbent(incumbents: List[np.ndarray], scores: List[float], higher_is_better: bool = False) -> Tuple[np.ndarray, int]:
    """
    Selects the actual optimizer winner corresponding to the median search performance replicate.
    CR-002: Guarantees the evaluated configuration is an actual incumbent produced by a genuine
    optimizer run, rather than a synthetic coordinate-wise median vector.
    """
    indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    med_idx = indexed[len(indexed) // 2]
    return incumbents[med_idx], med_idx


def _historical_legacy_main():
    print("="*70)
    print("RUNNING FAIR EMPIRICAL BENCHMARKS (20 Replicates x 140 Budget)")
    print("="*70)
    os.makedirs("results", exist_ok=True)
    data_mgr = CaliforniaHousingDataManager()

    # Optima coordinates
    x_doe_mo = np.array([0.8499708, -0.66666667, 1.0, -0.08116946])
    with open("results/phase3.json", "r", encoding="utf-8") as f:
        p3 = json.load(f)
    x_doe_so = np.array(p3["constrained_optimum_cube"]["x"])

    # 1. Run Optimizers across 20 sampler seeds serially on dedicated pinned core
    # Eliminates scheduler contention during online latency timing (Codex Comments 4224787409, 4225043379)
    pinned = pin_cpu_affinity(0)
    if pinned:
        print("Running 20 optimizer replicates serially on dedicated pinned CPU core 0...")
    else:
        print("Running 20 optimizer replicates serially (CPU affinity pinning not supported on this platform)...")
    rep_results = []
    for rep_idx, s in enumerate(OPTIMIZER_SEEDS):
        t0_rep = time.time()
        print(f"[{rep_idx+1}/{len(OPTIMIZER_SEEDS)}] Running replicate seed {s}...")
        r = run_single_optimizer_replicate(s, 42)
        rep_results.append(r)
        print(f"  Completed replicate seed {s} in {time.time()-t0_rep:.1f}s (RS val: {r['val_rs']:.4f}, TPE val: {r['val_tpe']:.4f})")

    rs_incumbents = [r["x_rs"] for r in rep_results]
    rs_vals = [r["val_rs"] for r in rep_results]
    rs_trajectories = [r["traj_rs"] for r in rep_results]

    tpe_so_incumbents = [r["x_tpe"] for r in rep_results]
    tpe_so_vals = [r["val_tpe"] for r in rep_results]
    tpe_so_trajectories = [r["traj_tpe"] for r in rep_results]

    tpe_co_incumbents = [r["x_co"] for r in rep_results]
    tpe_co_vals = [r["val_co"] for r in rep_results]

    tpe_mo_incumbents = [r["x_mo"] for r in rep_results]
    tpe_mo_desirabilities = [r["des_mo"] for r in rep_results]

    # Save trajectories
    df_traj = pd.DataFrame({
        "eval_idx": list(range(1, 141)),
        "rs_median": np.median(rs_trajectories, axis=0),
        "rs_q25": np.percentile(rs_trajectories, 25, axis=0),
        "rs_q75": np.percentile(rs_trajectories, 75, axis=0),
        "tpe_median": np.median(tpe_so_trajectories, axis=0),
        "tpe_q25": np.percentile(tpe_so_trajectories, 25, axis=0),
        "tpe_q75": np.percentile(tpe_so_trajectories, 75, axis=0),
    })
    df_traj.to_csv("results/benchmark_evals_trajectories.csv", index=False)

    # CR-002: Select actual optimizer winners from the median-performing search replicate
    x_rs_winner, idx_rs = select_median_actual_incumbent(rs_incumbents, rs_vals, higher_is_better=False)
    x_tpe_so_winner, idx_tpe = select_median_actual_incumbent(tpe_so_incumbents, tpe_so_vals, higher_is_better=False)
    x_tpe_co_winner, idx_co = select_median_actual_incumbent(tpe_co_incumbents, tpe_co_vals, higher_is_better=False)
    x_tpe_mo_winner, idx_mo = select_median_actual_incumbent(tpe_mo_incumbents, tpe_mo_desirabilities, higher_is_better=True)

    print("\nSelected actual optimizer winners (median search replicate):")
    print(f"  Random Search: rep #{idx_rs} (seed {OPTIMIZER_SEEDS[idx_rs]}), Val RMSE = {rs_vals[idx_rs]:.4f}")
    print(f"  TPE Single-Obj: rep #{idx_tpe} (seed {OPTIMIZER_SEEDS[idx_tpe]}), Val RMSE = {tpe_so_vals[idx_tpe]:.4f}")
    print(f"  Constrained TPE: rep #{idx_co} (seed {OPTIMIZER_SEEDS[idx_co]}), Val RMSE = {tpe_co_vals[idx_co]:.4f}")
    print(f"  Multi-Obj TPE: rep #{idx_mo} (seed {OPTIMIZER_SEEDS[idx_mo]}), Desirability = {tpe_mo_desirabilities[idx_mo]:.4f}")

    # Save all 20 actual incumbents for full transparency and audit trail
    incumbent_records = []
    for s_idx, s in enumerate(OPTIMIZER_SEEDS):
        eta_rs, d_rs, sub_rs, lam_rs = decode_factors(rs_incumbents[s_idx])
        eta_tpe, d_tpe, sub_tpe, lam_tpe = decode_factors(tpe_so_incumbents[s_idx])
        eta_co, d_co, sub_co, lam_co = decode_factors(tpe_co_incumbents[s_idx])
        eta_mo, d_mo, sub_mo, lam_mo = decode_factors(tpe_mo_incumbents[s_idx])
        incumbent_records.append({
            "optimizer_seed": s,
            "rs_val_rmse": rs_vals[s_idx], "rs_depth": d_rs, "rs_eta": eta_rs, "rs_subsample": sub_rs, "rs_lambda": lam_rs,
            "tpe_so_val_rmse": tpe_so_vals[s_idx], "tpe_so_depth": d_tpe, "tpe_so_eta": eta_tpe, "tpe_so_subsample": sub_tpe, "tpe_so_lambda": lam_tpe,
            "tpe_co_val_rmse": tpe_co_vals[s_idx], "tpe_co_depth": d_co, "tpe_co_eta": eta_co, "tpe_co_subsample": sub_co, "tpe_co_lambda": lam_co,
            "tpe_mo_desirability": tpe_mo_desirabilities[s_idx], "tpe_mo_depth": d_mo, "tpe_mo_eta": eta_mo, "tpe_mo_subsample": sub_mo, "tpe_mo_lambda": lam_mo,
        })
    pd.DataFrame(incumbent_records).to_csv("results/benchmark_optimizer_incumbents.csv", index=False)

    # 2. Evaluate all methods across 20 fresh evaluation seeds
    configs_to_eval = {
        "Sequential DOE-CCD (x*, Multi-Objective)": x_doe_mo,
        "Sequential DOE-CCD (Single-Objective)": x_doe_so,
        "Unguided Random Search": x_rs_winner,
        "Bayesian Optimization (Optuna TPE Single-Obj)": x_tpe_so_winner,
        "Constrained TPE (Latency <= 145 us)": x_tpe_co_winner,
        "Multi-Objective TPE (Desirability)": x_tpe_mo_winner,
    }

    eval_results = {}
    print("\nEvaluating all candidate configurations across 20 fresh seeds [2001..2020]...")
    for name, x_coord in configs_to_eval.items():
        res = evaluate_config_on_seeds(x_coord, FRESH_SEEDS, data_mgr)
        eval_results[name] = res
        print(f"{name}: Val RMSE = {res['val_rmse_mean']:.4f}, Test RMSE = {res['test_rmse_mean']:.4f}, Depth = {res['depth']}")

    # 3. Interleaved latency measurements
    configs_natural = {
        name: decode_factors(x_coord) for name, x_coord in configs_to_eval.items()
    }
    print("\nExecuting interleaved latency measurement harness (pinned CPU affinity)...")
    lat_results = measure_interleaved_latencies(configs_natural, data_mgr, reps=5, batch_size=200)

    # Merge results
    summary_rows = []
    for name in configs_to_eval:
        er = eval_results[name]
        lr = lat_results[name]
        summary_rows.append({
            "method": name,
            "val_rmse_mean": er["val_rmse_mean"],
            "val_rmse_std": er["val_rmse_std"],
            "val_rmse_ci95_low": er["val_rmse_ci95"][0],
            "val_rmse_ci95_high": er["val_rmse_ci95"][1],
            "test_rmse_mean": er["test_rmse_mean"],
            "test_rmse_std": er["test_rmse_std"],
            "test_rmse_ci95_low": er["test_rmse_ci95"][0],
            "test_rmse_ci95_high": er["test_rmse_ci95"][1],
            "predict_latency_us_median": lr["predict_latency_us_median"],
            "predict_latency_us_iqr": lr["predict_latency_us_iqr"],
            "inplace_latency_us_median": lr["inplace_latency_us_median"],
            "inplace_latency_us_iqr": lr["inplace_latency_us_iqr"],
            "depth": er["depth"],
            "eta": er["eta"],
            "subsample": er["subsample"],
            "reg_lambda": er["reg_lambda"],
        })

    df_bm = pd.DataFrame(summary_rows)
    df_bm.to_csv("results/benchmark.csv", index=False)
    print("\nBenchmark results saved to results/benchmark.csv:")
    print(df_bm[["method", "val_rmse_mean", "test_rmse_mean", "predict_latency_us_median", "depth"]])

    # 4. Paired comparisons against DOE x*
    doe_val = np.array(eval_results["Sequential DOE-CCD (x*, Multi-Objective)"]["val_rmses"])
    doe_test = np.array(eval_results["Sequential DOE-CCD (x*, Multi-Objective)"]["test_rmses"])

    paired_stats = {}
    for name in configs_to_eval:
        if "DOE-CCD (x*, Multi-Objective)" in name:
            continue
        c_val = np.array(eval_results[name]["val_rmses"])
        c_test = np.array(eval_results[name]["test_rmses"])

        tt_val = stats.ttest_rel(doe_val, c_val)
        tt_test = stats.ttest_rel(doe_test, c_test)
        wx_test = stats.wilcoxon(doe_test, c_test)

        paired_stats[name] = {
            "diff_test_rmse_mean": float(np.mean(doe_test - c_test)),
            "paired_t_stat": float(tt_test.statistic),
            "paired_t_pvalue": float(tt_test.pvalue),
            "wilcoxon_stat": float(wx_test.statistic),
            "wilcoxon_pvalue": float(wx_test.pvalue),
        }

    # 5. Hypervolume calculation (Reference point: [0.60, 250.0])
    # Addresses Codex Comment 4224787423: Compute both single-point (x*_MO alone) and two-point (x*_MO + x*_SO) hypervolume
    ref_point = (0.60, 250.0)

    # Single-point DOE: Depth 4 (x*, Multi-Objective) alone
    pts_doe_single = [
        (eval_results["Sequential DOE-CCD (x*, Multi-Objective)"]["test_rmse_mean"],
         lat_results["Sequential DOE-CCD (x*, Multi-Objective)"]["predict_latency_us_median"])
    ]
    hv_doe_single = compute_hypervolume(pts_doe_single, ref_point)

    # Two-point complementary DOE front: Depth 4 (x*) and Depth 7 (Single-Obj)
    pts_doe_two = [
        (eval_results["Sequential DOE-CCD (Single-Objective)"]["test_rmse_mean"],
         lat_results["Sequential DOE-CCD (Single-Objective)"]["predict_latency_us_median"]),
        (eval_results["Sequential DOE-CCD (x*, Multi-Objective)"]["test_rmse_mean"],
         lat_results["Sequential DOE-CCD (x*, Multi-Objective)"]["predict_latency_us_median"]),
    ]
    hv_doe_two = compute_hypervolume(pts_doe_two, ref_point)

    # MO-TPE selected operating point
    pts_motpe = [
        (eval_results["Multi-Objective TPE (Desirability)"]["test_rmse_mean"],
         lat_results["Multi-Objective TPE (Desirability)"]["predict_latency_us_median"])
    ]
    hv_motpe = compute_hypervolume(pts_motpe, ref_point)

    # Random search
    pts_rs = [
        (eval_results["Unguided Random Search"]["test_rmse_mean"],
         lat_results["Unguided Random Search"]["predict_latency_us_median"])
    ]
    hv_rs = compute_hypervolume(pts_rs, ref_point)

    hv_summary = {
        "reference_point": list(ref_point),
        "hv_doe_single": float(hv_doe_single),
        "hv_doe_two": float(hv_doe_two),
        "hv_doe": float(hv_doe_two),  # backward compatibility alias for two-point
        "hv_motpe": float(hv_motpe),
        "hv_rs": float(hv_rs),
        "doe_single_over_motpe_pct": float((((hv_doe_single - hv_motpe) / hv_motpe) * 100.0) if hv_motpe > 0 else 0.0),
        "doe_over_motpe_pct": float((((hv_doe_two - hv_motpe) / hv_motpe) * 100.0) if hv_motpe > 0 else 0.0),
    }

    # 6. Desirability sensitivity table
    df_runs = pd.read_csv("results/runs.csv")
    Q_cols = ["x1", "x2", "x3", "x4"]
    df_aug = df_runs.copy()
    for q in Q_cols: df_aug[q + "_sq"] = df_aug[q]**2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{Q_cols[i]}_{Q_cols[j]}"] = df_aug[Q_cols[i]] * df_aug[Q_cols[j]]

    fit_y1 = sm.OLS.from_formula("val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                                 "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()
    fit_y2 = sm.OLS.from_formula("latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
                                 "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df_aug).fit()

    scenarios = [
        {"Scenario": "Standard Config", "L1": 0.45, "U1": 0.70, "L2": 100, "U2": 180, "w1": 1, "w2": 1},
        {"Scenario": "Strict Latency", "L1": 0.45, "U1": 0.70, "L2": 90, "U2": 140, "w1": 1, "w2": 2},
        {"Scenario": "Strict Accuracy", "L1": 0.44, "U1": 0.60, "L2": 100, "U2": 200, "w1": 2, "w2": 1},
        {"Scenario": "Narrow Range", "L1": 0.46, "U1": 0.65, "L2": 110, "U2": 160, "w1": 1, "w2": 1},
    ]

    sens_records = []
    grid_pts = []
    for x1_v in np.linspace(-1, 1, 21):
        for x3_v in [0.0, 1.0]:
            for x4_v in [0.0, 1.0]:
                for d_val in range(3, 10):
                    x2_v = (d_val - 6.0) / 3.0
                    grid_pts.append((x1_v, x2_v, x3_v, x4_v, d_val))

    for sc in scenarios:
        L1, U1, L2, U2, w1, w2 = sc["L1"], sc["U1"], sc["L2"], sc["U2"], sc["w1"], sc["w2"]
        best_D = -1.0
        best_pt = None
        for x1_v, x2_v, x3_v, x4_v, d_val in grid_pts:
            row_dict = {
                "Intercept": 1.0,
                "x1": x1_v, "x2": x2_v, "x3": x3_v, "x4": x4_v,
                "x1_sq": x1_v**2, "x2_sq": x2_v**2, "x3_sq": x3_v**2, "x4_sq": x4_v**2,
                "x1_x2": x1_v*x2_v, "x1_x3": x1_v*x3_v, "x1_x4": x1_v*x4_v,
                "x2_x3": x2_v*x3_v, "x2_x4": x2_v*x4_v, "x3_x4": x3_v*x4_v,
            }
            for b_idx in [2, 3, 4, 5]:
                row_dict[f"C(block)[T.{b_idx}]"] = 0.2

            y1_p = sum(fit_y1.params[k] * row_dict.get(k, 0.0) for k in fit_y1.params.index)
            y2_p = sum(fit_y2.params[k] * row_dict.get(k, 0.0) for k in fit_y2.params.index)

            d1, d2, D = derringer_suich_desirability(y1_p, y2_p, L1, U1, L2, U2, w1=w1, w2=w2)
            if D > best_D:
                best_D = D
                best_pt = (x1_v, d_val, x3_v, x4_v)

        opt_eta = float(np.exp(np.log(0.05477) + best_pt[0] * 1.70059869))
        sc_res = sc.copy()
        sc_res["Optimal_Depth"] = best_pt[1]
        sc_res["Optimal_Eta"] = opt_eta
        sc_res["Best_D"] = best_D
        sens_records.append(sc_res)

    df_sens = pd.DataFrame(sens_records)
    df_sens.to_csv("results/desirability_sensitivity.csv", index=False)

    summary_json = {
        "hypervolume": hv_summary,
        "paired_statistics": paired_stats,
    }
    with open("results/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print("\nBenchmark summary & sensitivity analysis completed successfully.")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _git_show_bytes(path: str, ref: str = HISTORICAL_BASELINE_REF) -> bytes:
    """Read exact immutable historical input bytes from the frozen Git ref."""
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def _git_show_text(path: str, ref: str = HISTORICAL_BASELINE_REF) -> str:
    return _git_show_bytes(path, ref).decode("utf-8")


def _default_revision_output(mode: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(CONFIG["revision_v2"]["artifact_root"]) / mode / f"run_{timestamp}_{_git_head()[:8]}"


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def _write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    normalized = [
        {key: _json_cell(value) for key, value in record.items()}
        for record in records
    ]
    pd.DataFrame(normalized).to_csv(path, index=False)


def _selected_record(result: SearchResult) -> Optional[Dict[str, Any]]:
    selected = [record for record in result.trial_records if record["selected"]]
    if result.status == "completed" and len(selected) != 1:
        raise RuntimeError(
            f"{result.optimizer} replicate {result.replicate_id} has {len(selected)} selected trials"
        )
    return selected[0] if selected else None


def _manifest_configuration(
    record: Dict[str, Any], selection_id: str
) -> Dict[str, Any]:
    model_config = json.loads(record["model_config"])
    return {
        "selection_id": selection_id,
        "optimizer": record["optimizer"],
        "optimizer_replicate_id": record["replicate_id"],
        "source_trial_id": record["trial_number"],
        "selection_rule": record["selection_rule"],
        "selection_metric": {
            "name": "validation_rmse",
            "value": record["validation_rmse"],
        },
        "search_time_predict_latency_us": record["predict_latency_us"],
        "search_time_feasible": record["feasibility"],
        "hyperparameters": {
            "learning_rate": record["learning_rate"],
            "max_depth": record["max_depth"],
            "subsample": record["subsample"],
            "reg_lambda": record["reg_lambda"],
        },
        "model": {
            "n_estimators": model_config["n_estimators"],
            "objective": model_config["objective"],
            "n_jobs_train": model_config["n_jobs"],
            "colsample_bytree": model_config["colsample_bytree"],
            "min_child_weight": model_config["min_child_weight"],
            "gamma": model_config["gamma"],
            "tree_method": model_config["tree_method"],
        },
    }


def _historical_doe_configurations() -> List[Dict[str, Any]]:
    confirmation = json.loads(_git_show_text("results/confirmation.json"))
    phase3 = json.loads(_git_show_text("results/phase3.json"))
    model = {
        "n_estimators": CONFIG["model"]["n_estimators"],
        "objective": CONFIG["model"]["objective"],
        "n_jobs_train": CONFIG["model"]["n_jobs_train"],
        "colsample_bytree": CONFIG["model"].get("colsample_bytree", 1.0),
        "min_child_weight": CONFIG["model"].get("min_child_weight", 1.0),
        "gamma": CONFIG["model"].get("gamma", 0.0),
        "tree_method": CONFIG["model"].get("tree_method", "auto"),
    }
    return [
        {
            "selection_id": "doe-mo-historical",
            "optimizer": "historical_preplanned_doe_desirability",
            "optimizer_replicate_id": None,
            "source_trial_id": None,
            "selection_rule": "historical post-search engineering compromise",
            "selection_metric": {
                "name": "surrogate_desirability",
                "value": None,
            },
            "historical_holdout_exposure_caveat": True,
            "hyperparameters": {
                "learning_rate": confirmation["x_star_natural"]["eta"],
                "max_depth": confirmation["x_star_natural"]["depth"],
                "subsample": confirmation["x_star_natural"]["subsample"],
                "reg_lambda": confirmation["x_star_natural"]["reg_lambda"],
            },
            "model": model,
        },
        {
            "selection_id": "doe-so-historical",
            "optimizer": "historical_preplanned_doe_single_objective",
            "optimizer_replicate_id": None,
            "source_trial_id": None,
            "selection_rule": "minimum fitted quadratic response with integer depth",
            "selection_metric": {
                "name": "predicted_validation_rmse",
                "value": phase3["constrained_optimum_cube"]["pred_rmse"],
            },
            "historical_holdout_exposure_caveat": True,
            "hyperparameters": {
                "learning_rate": phase3["constrained_optimum_cube"]["natural"]["eta"],
                "max_depth": phase3["constrained_optimum_cube"]["natural"]["depth"],
                "subsample": phase3["constrained_optimum_cube"]["natural"]["subsample"],
                "reg_lambda": phase3["constrained_optimum_cube"]["natural"]["reg_lambda"],
            },
            "model": model,
        },
    ]


def _fit_manifest_models(configurations, data_mgr, split_seed: int):
    split = _development_split(data_mgr, split_seed)
    models = {}
    for configuration in configurations:
        hp = configuration["hyperparameters"]
        model_cfg = configuration["model"]
        model = xgb.XGBRegressor(
            n_estimators=model_cfg["n_estimators"],
            learning_rate=hp["learning_rate"],
            max_depth=int(hp["max_depth"]),
            subsample=hp["subsample"],
            reg_lambda=hp["reg_lambda"],
            colsample_bytree=model_cfg.get("colsample_bytree", 1.0),
            min_child_weight=model_cfg.get("min_child_weight", 1.0),
            gamma=model_cfg.get("gamma", 0.0),
            tree_method=model_cfg.get("tree_method", "auto"),
            random_state=split_seed,
            n_jobs=model_cfg["n_jobs_train"],
            objective=model_cfg["objective"],
        )
        model.fit(split.X_train, split.y_train)
        models[configuration["selection_id"]] = model
    return split, models


def _measure_frozen_primary_latencies(
    configurations, data_mgr, split_seed: int, session_id: str
) -> Dict[str, Any]:
    split, models = _fit_manifest_models(configurations, data_mgr, split_seed)
    return measure_latencies(
        models,
        split.X_val[:1],
        session_id=session_id,
        protocol=PRIMARY_V1,
    )


def _metric_summary(values: List[float]) -> Dict[str, Any]:
    array = np.asarray(values, dtype=float)
    n = len(array)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "interquartile_range": None,
            "confidence_interval_95": [None, None],
        }
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if n > 1 else None
    interval = [None, None]
    if n > 1:
        half = float(stats.t.ppf(0.975, n - 1) * standard_deviation / np.sqrt(n))
        interval = [mean - half, mean + half]
    return {
        "n": n,
        "mean": mean,
        "median": float(np.median(array)),
        "standard_deviation": standard_deviation,
        "interquartile_range": float(np.subtract(*np.percentile(array, [75, 25]))),
        "confidence_interval_95": interval,
    }


def _conditional_difference_description(a, b) -> Dict[str, Any]:
    """Describe matched retraining-seed differences without inferential tests."""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) == 0 or len(left) != len(right):
        raise ValueError("conditional differences require equal non-empty vectors")
    differences = left - right
    if not np.all(np.isfinite(differences)):
        raise ValueError("conditional differences must be finite")
    return {
        "difference": "anchor_minus_comparator",
        "n_retraining_seed_pairs": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "standard_deviation": (
            float(np.std(differences, ddof=1)) if len(differences) > 1 else None
        ),
        "per_seed_differences": differences.tolist(),
        "inference_performed": False,
        "reason": (
            "Retraining seeds are conditional repeated fits on overlapping partitions and a "
            "common fixed holdout; they are not independent method-level replicates."
        ),
    }


def _aggregate_primary_latency_sessions(
    sessions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not sessions:
        raise ValueError("At least one primary latency session is required")
    protocol_ids = {
        session["metadata"]["protocol"]["protocol_id"] for session in sessions
    }
    if protocol_ids != {PRIMARY_V1.protocol_id}:
        raise ValueError(f"Mixed primary latency protocols: {sorted(protocol_ids)}")
    model_sets = [set(session["summaries"]) for session in sessions]
    if any(model_ids != model_sets[0] for model_ids in model_sets[1:]):
        raise ValueError("Every latency session must measure the same frozen selections")

    summaries = {}
    for model_id in sorted(model_sets[0]):
        predict = [
            float(session["summaries"][model_id]["predict_latency_us"])
            for session in sessions
        ]
        inplace = [
            float(session["summaries"][model_id]["inplace_predict_latency_us"])
            for session in sessions
        ]
        predict_within_iqr = [
            float(session["summaries"][model_id]["predict_latency_us_iqr"])
            for session in sessions
        ]
        inplace_within_iqr = [
            float(session["summaries"][model_id]["inplace_predict_latency_us_iqr"])
            for session in sessions
        ]
        summaries[model_id] = {
            "predict_latency_us": float(np.median(predict)),
            "predict_latency_us_iqr": float(np.median(predict_within_iqr)),
            "predict_latency_session_iqr_us": float(
                np.subtract(*np.percentile(predict, [75, 25]))
            ),
            "predict_latency_session_standard_deviation_us": (
                float(np.std(predict, ddof=1)) if len(predict) > 1 else None
            ),
            "inplace_predict_latency_us": float(np.median(inplace)),
            "inplace_predict_latency_us_iqr": float(np.median(inplace_within_iqr)),
            "inplace_predict_latency_session_iqr_us": float(
                np.subtract(*np.percentile(inplace, [75, 25]))
            ),
            "inplace_predict_latency_session_standard_deviation_us": (
                float(np.std(inplace, ddof=1)) if len(inplace) > 1 else None
            ),
        }
    return {
        "schema_version": 2,
        "protocol_id": PRIMARY_V1.protocol_id,
        "aggregation": {
            "session_count": len(sessions),
            "point_estimate": "median of session medians",
            "within_session_dispersion": "median of session IQRs",
            "between_session_dispersion": "IQR and sample standard deviation of session medians",
            "session_scope": "separate randomized timing sessions in one process",
        },
        "summaries": summaries,
        "sessions": sessions,
    }


def _latency_interface_overhead(
    latency_measurement: Dict[str, Any],
) -> Dict[str, Any]:
    per_session = []
    by_selection: Dict[str, List[float]] = {}
    for session in latency_measurement["sessions"]:
        session_id = session["metadata"]["session"]["session_id"]
        for selection_id, summary in session["summaries"].items():
            overhead = float(
                summary["predict_latency_us"]
                - summary["inplace_predict_latency_us"]
            )
            per_session.append({
                "session_id": session_id,
                "selection_id": selection_id,
                "overhead_us": overhead,
            })
            by_selection.setdefault(selection_id, []).append(overhead)
    return {
        "schema_version": 1,
        "protocol_id": latency_measurement["protocol_id"],
        "estimand": (
            "predict_latency_us minus inplace_predict_latency_us within timing session"
        ),
        "causal_interpretation": (
            "Descriptive interface difference; no single implementation cause is inferred."
        ),
        "per_session": per_session,
        "by_selection": {
            selection_id: _metric_summary(values)
            for selection_id, values in sorted(by_selection.items())
        },
        "pooled_descriptive_summary": _metric_summary(
            [row["overhead_us"] for row in per_session]
        ),
    }


def _evaluate_historical_doe_front(
    data_mgr,
    split_seed: int,
    selected_configurations: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    historical = pd.read_csv(io.StringIO(_git_show_text("results/runs.csv")))
    unique = historical.sort_values("point_id").drop_duplicates("point_id")
    split = _development_split(data_mgr, split_seed)
    rows = []

    def evaluate_candidate(
        candidate_id: str,
        candidate_type: str,
        eta: float,
        depth: int,
        subsample: float,
        reg_lambda: float,
        x: np.ndarray,
    ) -> None:
        model = xgb.XGBRegressor(
            **_model_config(eta, depth, subsample, reg_lambda, split_seed)
        )
        model.fit(split.X_train, split.y_train)
        rmse = float(np.sqrt(np.mean((split.y_val - model.predict(split.X_val)) ** 2)))
        session = f"{candidate_id}-matched-online"
        latency = measure_trial_latency(model, split.X_val[:1], session_id=session)
        rows.append({
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "validation_rmse": rmse,
            "predict_latency_us": latency,
            "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id,
            "development_split_seed": split_seed,
            "learning_rate": float(eta),
            "max_depth": int(depth),
            "subsample": float(subsample),
            "reg_lambda": float(reg_lambda),
            "x1": float(x[0]),
            "x2": float(x[1]),
            "x3": float(x[2]),
            "x4": float(x[3]),
        })

    for row in unique.itertuples(index=False):
        x = np.asarray([row.x1, row.x2, row.x3, row.x4], dtype=float)
        eta, depth, subsample, reg_lambda = decode_factors(x)
        evaluate_candidate(
            f"doe-point-{int(row.point_id)}",
            "historical_doe_evaluated_coordinate",
            eta,
            depth,
            subsample,
            reg_lambda,
            x,
        )

    for configuration in selected_configurations or []:
        hp = configuration["hyperparameters"]
        x = encode_factors(
            hp["learning_rate"], hp["max_depth"], hp["subsample"], hp["reg_lambda"]
        )
        evaluate_candidate(
            configuration["selection_id"],
            "historical_doe_selected_operating_point",
            float(hp["learning_rate"]),
            int(hp["max_depth"]),
            float(hp["subsample"]),
            float(hp["reg_lambda"]),
            x,
        )
    return rows


def _repeated_doe_grid() -> np.ndarray:
    configured = CONFIG["revision_v2"]["repeated_doe_selection_grid"]
    points = []
    for x1 in np.linspace(
        -1.0, 1.0, int(configured["learning_rate_coded_levels"])
    ):
        for depth in configured["integer_depths"]:
            x2 = (int(depth) - 6.0) / 3.0
            for x3 in configured["subsample_coded_levels"]:
                for x4 in configured["reg_lambda_coded_levels"]:
                    points.append((x1, x2, float(x3), float(x4)))
    return np.asarray(points, dtype=float)


def _predict_block_averaged_generic(fit, coordinate: np.ndarray, block_count: int) -> float:
    x1, x2, x3, x4 = (float(value) for value in coordinate)
    values = {
        "const": 1.0,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x1_sq": x1**2,
        "x2_sq": x2**2,
        "x3_sq": x3**2,
        "x4_sq": x4**2,
        "x1_x2": x1 * x2,
        "x1_x3": x1 * x3,
        "x1_x4": x1 * x4,
        "x2_x3": x2 * x3,
        "x2_x4": x2 * x4,
        "x3_x4": x3 * x4,
    }
    for block_index in range(2, block_count + 1):
        values[f"blk_{block_index}"] = 1.0 / block_count
    return float(sum(fit.params[name] * values[name] for name in fit.params.index))


def _repeated_doe_model() -> Dict[str, Any]:
    configured = CONFIG["model"]
    return {
        "n_estimators": int(configured["n_estimators"]),
        "objective": configured["objective"],
        "n_jobs_train": int(configured["n_jobs_train"]),
        "colsample_bytree": float(configured.get("colsample_bytree", 1.0)),
        "min_child_weight": float(configured.get("min_child_weight", 1.0)),
        "gamma": float(configured.get("gamma", 0.0)),
        "tree_method": configured.get("tree_method", "auto"),
    }


def run_repeated_doe_selection(
    data_mgr,
    *,
    replicate_count: int,
    seeds_per_set: int,
    seed_base: int,
    max_latency_us: float,
) -> Dict[str, Any]:
    """Repeat the pre-planned 28-run-per-block design using development data only."""
    if replicate_count <= 0 or seeds_per_set < 2:
        raise ValueError("DOE selection requires positive replicates and at least two blocks")
    historical = pd.read_csv(io.StringIO(_git_show_text("results/runs.csv")))
    template = (
        historical[historical["block"] == historical["block"].min()]
        .sort_values("run_id")
        [["phase", "point_id", "replicate", "x1", "x2", "x3", "x4"]]
        .reset_index(drop=True)
    )
    if len(template) != 28 or template["point_id"].nunique() != 25:
        raise RuntimeError("Frozen historical DOE template is not the expected 28-run/25-point design")

    run_records = []
    candidate_records = []
    replicate_records = []
    configurations = []
    grid = _repeated_doe_grid()
    desirability_cfg = CONFIG["desirability"]
    L1 = float(desirability_cfg["Y1_RMSE"]["L"])
    U1 = float(desirability_cfg["Y1_RMSE"]["U"])
    L2 = float(desirability_cfg["Y2_Latency"]["L"])
    U2 = float(desirability_cfg["Y2_Latency"]["U"])
    w1 = float(desirability_cfg["Y1_RMSE"]["weight"])
    w2 = float(desirability_cfg["Y2_Latency"]["weight"])
    model_manifest = _repeated_doe_model()

    for replicate_id in range(replicate_count):
        block_seeds = [
            int(seed_base + replicate_id * seeds_per_set + offset)
            for offset in range(seeds_per_set)
        ]
        replicate_rows = []
        started_replicate = time.perf_counter()
        for block_index, seed in enumerate(block_seeds, start=1):
            split = _development_split(data_mgr, seed)
            random_generator = np.random.default_rng(seed)
            order = random_generator.permutation(len(template))
            for order_within_block, template_index in enumerate(order, start=1):
                design_row = template.iloc[int(template_index)]
                coordinate = np.asarray(
                    [design_row.x1, design_row.x2, design_row.x3, design_row.x4],
                    dtype=float,
                )
                eta, depth, subsample, reg_lambda = decode_factors(coordinate)
                template_replicate = int(design_row.replicate)
                model_seed = int(seed + 1000 * (template_replicate - 1))
                model_config = _model_config(
                    eta, depth, subsample, reg_lambda, model_seed
                )
                training_started = time.perf_counter()
                session_id = (
                    f"doe-selection-rep-{replicate_id}-block-{block_index}"
                    f"-run-{order_within_block}"
                )
                record = {
                    "doe_replicate_id": replicate_id,
                    "block_index": block_index,
                    "block_seed": seed,
                    "order_within_block": order_within_block,
                    "phase": design_row.phase,
                    "point_id": int(design_row.point_id),
                    "template_replicate": template_replicate,
                    "x1": float(coordinate[0]),
                    "x2": float(coordinate[1]),
                    "x3": float(coordinate[2]),
                    "x4": float(coordinate[3]),
                    "learning_rate": eta,
                    "max_depth": depth,
                    "subsample": subsample,
                    "reg_lambda": reg_lambda,
                    "development_split_seed": seed,
                    "model_seed": model_seed,
                    "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id,
                    "latency_session_id": session_id,
                    "model_config": model_config,
                }
                try:
                    model = xgb.XGBRegressor(**model_config)
                    model.fit(split.X_train, split.y_train)
                    training_time = time.perf_counter() - training_started
                    evaluation_started = time.perf_counter()
                    prediction = model.predict(split.X_val)
                    validation_rmse = float(
                        np.sqrt(np.mean((split.y_val - prediction) ** 2))
                    )
                    latency = measure_trial_latency(
                        model, split.X_val[:1], session_id=session_id
                    )
                    record.update({
                        "validation_rmse": validation_rmse,
                        "predict_latency_us": latency,
                        "training_time_s": training_time,
                        "evaluation_time_s": time.perf_counter() - evaluation_started,
                        "trial_status": "completed",
                        "error": None,
                    })
                except Exception as exc:
                    record.update({
                        "validation_rmse": None,
                        "predict_latency_us": None,
                        "training_time_s": time.perf_counter() - training_started,
                        "evaluation_time_s": 0.0,
                        "trial_status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                run_records.append(record)
                replicate_rows.append(record)

        frame = pd.DataFrame(replicate_rows)
        failed_count = int((frame["trial_status"] != "completed").sum())
        if failed_count:
            replicate_records.append({
                "doe_replicate_id": replicate_id,
                "status": "failed",
                "block_seeds": block_seeds,
                "design_runs": len(frame),
                "completed_design_runs": len(frame) - failed_count,
                "failed_design_runs": failed_count,
                "error": "At least one pre-planned DOE run failed; no surrogate selection was frozen.",
                "replicate_wall_time_s": time.perf_counter() - started_replicate,
            })
            continue
        grouped = (
            frame.groupby(
                [
                    "point_id", "x1", "x2", "x3", "x4",
                    "learning_rate", "max_depth", "subsample", "reg_lambda",
                ],
                as_index=False,
            )
            .agg(
                evaluation_count=("validation_rmse", "size"),
                validation_rmse_mean=("validation_rmse", "mean"),
                validation_rmse_standard_deviation=("validation_rmse", "std"),
                predict_latency_us_mean=("predict_latency_us", "mean"),
                predict_latency_us_standard_deviation=("predict_latency_us", "std"),
            )
        )
        for candidate in grouped.to_dict("records"):
            candidate_records.append({
                "doe_replicate_id": replicate_id,
                "block_seeds": block_seeds,
                "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id,
                **candidate,
            })

        try:
            design = pd.DataFrame(index=frame.index)
            for factor in ("x1", "x2", "x3", "x4"):
                design[factor] = frame[factor]
                design[f"{factor}_sq"] = frame[factor] ** 2
            for left, right in (
                ("x1", "x2"), ("x1", "x3"), ("x1", "x4"),
                ("x2", "x3"), ("x2", "x4"), ("x3", "x4"),
            ):
                design[f"{left}_{right}"] = frame[left] * frame[right]
            block_dummies = pd.get_dummies(
                frame["block_index"], prefix="blk", drop_first=True
            ).astype(float)
            design = sm.add_constant(pd.concat([design, block_dummies], axis=1))
            expected_rank = int(design.shape[1])
            fit_rmse = sm.OLS(frame["validation_rmse"], design).fit()
            fit_latency = sm.OLS(frame["predict_latency_us"], design).fit()
            if int(fit_rmse.model.rank) != expected_rank:
                raise RuntimeError(
                    f"RMSE surrogate rank {fit_rmse.model.rank} is below {expected_rank}"
                )
            if int(fit_latency.model.rank) != expected_rank:
                raise RuntimeError(
                    f"latency surrogate rank {fit_latency.model.rank} is below {expected_rank}"
                )
            if fit_rmse.df_resid <= 0 or fit_latency.df_resid <= 0:
                raise RuntimeError("surrogate residual degrees of freedom must be positive")
            if not np.all(np.isfinite(fit_rmse.params)) or not np.all(
                np.isfinite(fit_latency.params)
            ):
                raise RuntimeError("surrogate coefficients must be finite")
            predicted_rmse = np.asarray([
                _predict_block_averaged_generic(fit_rmse, coordinate, seeds_per_set)
                for coordinate in grid
            ])
            predicted_latency = np.asarray([
                _predict_block_averaged_generic(fit_latency, coordinate, seeds_per_set)
                for coordinate in grid
            ])
            predicted_desirability = np.asarray([
                derringer_suich_desirability(
                    rmse, latency, L1, U1, L2, U2, w1=w1, w2=w2
                )[2]
                for rmse, latency in zip(predicted_rmse, predicted_latency)
            ])
            if not (
                np.all(np.isfinite(predicted_rmse))
                and np.all(np.isfinite(predicted_latency))
                and np.all(np.isfinite(predicted_desirability))
            ):
                raise RuntimeError("surrogate grid predictions must be finite")
            so_index = min(
                range(len(grid)),
                key=lambda index: (
                    predicted_rmse[index], predicted_latency[index], tuple(grid[index])
                ),
            )
            mo_index = min(
                range(len(grid)),
                key=lambda index: (
                    -predicted_desirability[index],
                    predicted_rmse[index],
                    predicted_latency[index],
                    tuple(grid[index]),
                ),
            )

            selections = (
                (
                    "mo",
                    mo_index,
                    "maximum_predeclared_grid_desirability",
                    {
                        "name": "predicted_composite_desirability",
                        "value": float(predicted_desirability[mo_index]),
                    },
                    "repeated_preplanned_doe_multi_objective",
                ),
                (
                    "so",
                    so_index,
                    "minimum_predicted_validation_rmse_on_predeclared_grid",
                    {
                        "name": "predicted_validation_rmse",
                        "value": float(predicted_rmse[so_index]),
                    },
                    "repeated_preplanned_doe_single_objective",
                ),
            )
            selected_payload = {}
            replicate_configurations = []
            for suffix, grid_index, selection_rule, metric, optimizer in selections:
                coordinate = grid[grid_index]
                eta, depth, subsample, reg_lambda = decode_factors(coordinate)
                selection_id = f"doe-{suffix}-rep-{replicate_id}"
                configuration = {
                    "selection_id": selection_id,
                    "optimizer": optimizer,
                    "optimizer_replicate_id": replicate_id,
                    "source_trial_id": None,
                    "source_grid_index": int(grid_index),
                    "selection_rule": selection_rule,
                    "selection_metric": metric,
                    "predicted_validation_rmse": float(predicted_rmse[grid_index]),
                    "search_time_predict_latency_us": float(
                        predicted_latency[grid_index]
                    ),
                    "search_time_feasible": bool(
                        predicted_latency[grid_index] <= max_latency_us
                    ),
                    "development_block_seeds": block_seeds,
                    "hyperparameters": {
                        "learning_rate": eta,
                        "max_depth": depth,
                        "subsample": subsample,
                        "reg_lambda": reg_lambda,
                    },
                    "model": model_manifest,
                }
                replicate_configurations.append(configuration)
                selected_payload[suffix] = {
                    "selection_id": selection_id,
                    "grid_index": int(grid_index),
                    "coordinate": coordinate.tolist(),
                    "predicted_validation_rmse": float(predicted_rmse[grid_index]),
                    "predicted_latency_us": float(predicted_latency[grid_index]),
                    "predicted_desirability": float(predicted_desirability[grid_index]),
                }
            configurations.extend(replicate_configurations)
        except Exception as exc:
            replicate_records.append({
                "doe_replicate_id": replicate_id,
                "status": "failed",
                "block_seeds": block_seeds,
                "design_runs": len(frame),
                "completed_design_runs": len(frame),
                "failed_design_runs": 0,
                "error": f"SurrogateSelectionError: {type(exc).__name__}: {exc}",
                "replicate_wall_time_s": time.perf_counter() - started_replicate,
            })
            continue
        replicate_records.append({
            "doe_replicate_id": replicate_id,
            "status": "completed",
            "block_seeds": block_seeds,
            "design_runs": len(frame),
            "completed_design_runs": len(frame),
            "failed_design_runs": 0,
            "unique_design_coordinates": int(grouped["point_id"].nunique()),
            "surrogate_expected_rank": expected_rank,
            "surrogate_rank_rmse": int(fit_rmse.model.rank),
            "surrogate_rank_latency": int(fit_latency.model.rank),
            "surrogate_residual_df_rmse": int(fit_rmse.df_resid),
            "surrogate_residual_df_latency": int(fit_latency.df_resid),
            "selection_grid_size": int(len(grid)),
            "selected": selected_payload,
            "replicate_wall_time_s": time.perf_counter() - started_replicate,
        })

    return {
        "run_records": run_records,
        "candidate_records": candidate_records,
        "replicate_records": replicate_records,
        "configurations": configurations,
        "grid": {
            "candidate_count": int(len(grid)),
            "definition": CONFIG["revision_v2"]["repeated_doe_selection_grid"],
            "tie_breaking": (
                "desirability/RMSE objective, secondary objective, then coded coordinate"
            ),
        },
    }


def _hypervolume_payload(points, reference) -> Dict[str, Any]:
    result = hypervolume_2d_min(points, reference)
    return {
        "value": result.value,
        "reference": list(result.reference),
        "pareto": asdict(result.pareto),
    }


def _computational_budget(
    mode: str,
    settings: Dict[str, Any],
    *,
    include_historical_doe: bool = True,
    include_repeated_doe: bool = True,
) -> Dict[str, Any]:
    replicates = int(settings["optimizer_replicates"])
    trials = int(settings["trials_per_optimizer"])
    evaluation_seeds = len(settings["evaluation_seeds"])
    doe_replicates = int(settings["doe_block_seed_sets"]) if include_repeated_doe else 0
    doe_seeds_per_set = int(settings["doe_seeds_per_set"]) if include_repeated_doe else 0
    repeated_doe_fits = doe_replicates * doe_seeds_per_set * 28
    frozen_max = 4 * replicates + 2 * doe_replicates + 2
    if not include_historical_doe:
        frozen_max -= 2
    latency_sessions = int(settings["latency_sessions"])
    historical_candidates = 27 if include_historical_doe else 0
    search_fits = 4 * replicates * trials
    primary_latency_refit_fits = frozen_max * latency_sessions
    final_retraining_fits = frozen_max * evaluation_seeds
    online_search_measurements = search_fits
    online_repeated_doe_measurements = repeated_doe_fits
    online_historical_front_measurements = historical_candidates
    online_measurements = (
        online_search_measurements
        + online_repeated_doe_measurements
        + online_historical_front_measurements
    )
    online_timed_calls = (
        online_measurements
        * ONLINE_SEARCH_V1.timed_calls_per_interface
        * len(ONLINE_SEARCH_V1.interfaces)
    )
    online_warmup_calls = (
        online_measurements
        * ONLINE_SEARCH_V1.warmup_calls_per_interface
        * len(ONLINE_SEARCH_V1.interfaces)
    )
    primary_timed_calls = (
        primary_latency_refit_fits
        * PRIMARY_V1.timed_calls_per_interface
        * len(PRIMARY_V1.interfaces)
    )
    primary_warmup_calls = (
        primary_latency_refit_fits
        * PRIMARY_V1.warmup_calls_per_interface
        * len(PRIMARY_V1.interfaces)
    )
    return {
        "mode": mode,
        "optimizer_methods": 4,
        "optimizer_replicates": replicates,
        "trials_per_optimizer": trials,
        "search_model_fits": search_fits,
        "repeated_doe_selection_replicates": doe_replicates,
        "repeated_doe_seeds_per_replicate": doe_seeds_per_set,
        "repeated_doe_design_model_fits": repeated_doe_fits,
        "matched_historical_doe_candidate_fits": historical_candidates,
        "matched_historical_doe_coordinate_fits": 25 if include_historical_doe else 0,
        "matched_historical_doe_selected_point_fits": 2 if include_historical_doe else 0,
        "maximum_frozen_configurations": frozen_max,
        "primary_latency_sessions": latency_sessions,
        "primary_latency_refit_fits": primary_latency_refit_fits,
        "final_retraining_fits": final_retraining_fits,
        "maximum_total_model_fits": (
            search_fits
            + repeated_doe_fits
            + historical_candidates
            + primary_latency_refit_fits
            + final_retraining_fits
        ),
        "online_latency_search_measurements_maximum": online_search_measurements,
        "online_latency_repeated_doe_measurements_maximum": (
            online_repeated_doe_measurements
        ),
        "online_latency_historical_front_measurements_maximum": (
            online_historical_front_measurements
        ),
        "online_latency_measurements_maximum": online_measurements,
        "online_latency_timed_calls_per_measurement": ONLINE_SEARCH_V1.timed_calls_per_interface,
        "online_latency_warmups_per_measurement": ONLINE_SEARCH_V1.warmup_calls_per_interface,
        "online_latency_timed_prediction_calls_maximum": online_timed_calls,
        "online_latency_warmup_prediction_calls_maximum": online_warmup_calls,
        "primary_latency_calls_per_frozen_configuration_per_interface": PRIMARY_V1.timed_calls_per_interface,
        "primary_latency_warmups_per_frozen_configuration_per_interface": PRIMARY_V1.warmup_calls_per_interface,
        "primary_latency_interfaces": len(PRIMARY_V1.interfaces),
        "primary_latency_timed_prediction_calls_maximum": primary_timed_calls,
        "primary_latency_warmup_prediction_calls_maximum": primary_warmup_calls,
        "latency_timed_prediction_calls_maximum": online_timed_calls + primary_timed_calls,
        "latency_warmup_prediction_calls_maximum": online_warmup_calls + primary_warmup_calls,
        "note": "The total is an upper bound; failed or infeasible searches produce no frozen configuration.",
    }


def _optimizer_level_summary(
    mode: str,
    replicate_rows: List[Dict[str, Any]],
    trial_rows: List[Dict[str, Any]],
    configurations: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    final_by_id = {row["selection_id"]: row for row in summary_rows}
    optimizers = {}
    optimizer_names = sorted({row["optimizer"] for row in replicate_rows})
    for optimizer in optimizer_names:
        optimizer_replicates = [
            row for row in replicate_rows if row["optimizer"] == optimizer
        ]
        selected_configs = [
            item
            for item in configurations
            if item["optimizer"] == optimizer
            and item["selection_id"] in final_by_id
        ]
        statuses = {
            status: sum(row["status"] == status for row in optimizer_replicates)
            for status in sorted({row["status"] for row in optimizer_replicates})
        }
        hyperparameter_distributions = {}
        for name in ("learning_rate", "max_depth", "subsample", "reg_lambda"):
            hyperparameter_distributions[name] = _metric_summary([
                float(item["hyperparameters"][name]) for item in selected_configs
            ])

        search_values = []
        independent_validation = []
        final_test = []
        optimism = []
        retraining_variability = {}
        for item in selected_configs:
            selection_id = item["selection_id"]
            final = final_by_id[selection_id]
            metric = item.get("selection_metric", {})
            if metric.get("name") == "validation_rmse" and metric.get("value") is not None:
                search_value = float(metric["value"])
                independent_value = float(final["validation_rmse"]["mean"])
                search_values.append(search_value)
                independent_validation.append(independent_value)
                optimism.append(independent_value - search_value)
            final_test.append(float(final["test_rmse"]["mean"]))
            retraining_variability[selection_id] = {
                "validation_rmse": final["validation_rmse"],
                "test_rmse": final["test_rmse"],
            }

        completed_trials = [
            row
            for row in trial_rows
            if row["optimizer"] == optimizer and row["trial_status"] == "completed"
        ]
        failed_trials = [
            row
            for row in trial_rows
            if row["optimizer"] == optimizer and row["trial_status"] != "completed"
        ]
        success_count = statuses.get("completed", 0)
        replicate_count = len(optimizer_replicates)
        optimizers[optimizer] = {
            "optimizer_replicates": replicate_count,
            "replicate_status_counts": statuses,
            "search_success_rate": success_count / replicate_count,
            "search_failure_rate": (replicate_count - success_count) / replicate_count,
            "completed_trial_count": len(completed_trials),
            "failed_trial_count": len(failed_trials),
            "model_fit_evaluations_attempted": sum(
                int(row["model_fit_evaluations"]) for row in optimizer_replicates
            ),
            "search_wall_time_s": _metric_summary([
                float(row["method_wall_time_s"]) for row in optimizer_replicates
            ]),
            "best_observed_search_validation_rmse": _metric_summary(search_values),
            "independently_retrained_validation_rmse": _metric_summary(
                independent_validation
            ),
            "final_test_rmse": _metric_summary(final_test),
            "selection_optimism_independent_minus_search_rmse": _metric_summary(
                optimism
            ),
            "selected_hyperparameter_distributions": hyperparameter_distributions,
            "between_search_variability": {
                "estimand": "distribution of per-replicate independent validation means",
                "summary": _metric_summary(independent_validation),
            },
            "conditional_retraining_variability": retraining_variability,
        }
    return {
        "schema_version": 1,
        "mode": mode,
        "inference_status": (
            "smoke_underpowered" if mode == "smoke" else "full_protocol"
        ),
        "variance_note": (
            "Between-search summaries use one independent-validation mean per search replicate. "
            "Conditional retraining summaries remain separate and are not pooled into one standard error."
        ),
        "optimizers": optimizers,
    }


def _doe_level_summary(
    mode: str,
    repeated_doe: Optional[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if repeated_doe is None:
        return {
            "schema_version": 1,
            "performed": False,
            "reason": "repeated_doe_disabled",
        }
    final_by_id = {row["selection_id"]: row for row in summary_rows}
    replicate_rows = repeated_doe["replicate_records"]
    run_rows = repeated_doe["run_records"]
    status_counts = {
        status: sum(row["status"] == status for row in replicate_rows)
        for status in sorted({row["status"] for row in replicate_rows})
    }
    best_observed_by_replicate = {}
    for candidate in repeated_doe["candidate_records"]:
        replicate_id = int(candidate["doe_replicate_id"])
        value = float(candidate["validation_rmse_mean"])
        best_observed_by_replicate[replicate_id] = min(
            value, best_observed_by_replicate.get(replicate_id, value)
        )
    methods = {}
    for optimizer in (
        "repeated_preplanned_doe_multi_objective",
        "repeated_preplanned_doe_single_objective",
    ):
        configs = [
            item for item in repeated_doe["configurations"] if item["optimizer"] == optimizer
        ]
        independent_validation = [
            float(final_by_id[item["selection_id"]]["validation_rmse"]["mean"])
            for item in configs
        ]
        predicted_validation = [
            float(item["predicted_validation_rmse"]) for item in configs
        ]
        best_observed_design = [
            best_observed_by_replicate[int(item["optimizer_replicate_id"])]
            for item in configs
        ]
        prediction_error = [
            observed - predicted
            for observed, predicted in zip(
                independent_validation, predicted_validation
            )
        ]
        final_test = [
            float(final_by_id[item["selection_id"]]["test_rmse"]["mean"])
            for item in configs
        ]
        methods[optimizer] = {
            "selection_replicates": len(configs),
            "surrogate_predicted_validation_rmse": _metric_summary(
                predicted_validation
            ),
            "best_observed_design_validation_rmse": _metric_summary(
                best_observed_design
            ),
            "independently_retrained_validation_rmse": _metric_summary(
                independent_validation
            ),
            "selection_prediction_error_independent_minus_surrogate_rmse": (
                _metric_summary(prediction_error)
            ),
            "final_test_rmse": _metric_summary(final_test),
            "selected_hyperparameter_distributions": {
                name: _metric_summary([
                    float(item["hyperparameters"][name]) for item in configs
                ])
                for name in ("learning_rate", "max_depth", "subsample", "reg_lambda")
            },
            "conditional_retraining_variability": {
                item["selection_id"]: {
                    "validation_rmse": final_by_id[item["selection_id"]]["validation_rmse"],
                    "test_rmse": final_by_id[item["selection_id"]]["test_rmse"],
                }
                for item in configs
            },
            "per_selection_replicate": [
                {
                    "doe_replicate_id": int(item["optimizer_replicate_id"]),
                    "selection_id": item["selection_id"],
                    "surrogate_predicted_validation_rmse": float(
                        item["predicted_validation_rmse"]
                    ),
                    "best_observed_design_validation_rmse": (
                        best_observed_by_replicate[
                            int(item["optimizer_replicate_id"])
                        ]
                    ),
                    "independently_retrained_validation_rmse": float(
                        final_by_id[item["selection_id"]]["validation_rmse"]["mean"]
                    ),
                }
                for item in configs
            ],
        }
    replicate_count = len(replicate_rows)
    completed_count = status_counts.get("completed", 0)
    return {
        "schema_version": 1,
        "performed": True,
        "mode": mode,
        "inference_status": (
            "smoke_underpowered" if mode == "smoke" else "full_protocol"
        ),
        "design": (
            "Each replicate reruns the frozen 28-run-per-block factorial-plus-CCD template "
            "on distinct seeded partitions of the same development pool, preserves distinct "
            "model seeds for center replicates, fits RMSE and latency quadratics, and selects "
            "from the predeclared integer-depth grid. Seeded partitions can overlap and are not "
            "independent dataset samples."
        ),
        "selection_replicate_count": replicate_count,
        "replicate_status_counts": status_counts,
        "selection_success_rate": (
            completed_count / replicate_count if replicate_count else None
        ),
        "selection_failure_rate": (
            (replicate_count - completed_count) / replicate_count
            if replicate_count
            else None
        ),
        "model_fit_evaluations_attempted": len(run_rows),
        "completed_design_fit_count": sum(
            row["trial_status"] == "completed" for row in run_rows
        ),
        "failed_design_fit_count": sum(
            row["trial_status"] != "completed" for row in run_rows
        ),
        "selection_replicate_wall_time_s": _metric_summary([
            float(row["replicate_wall_time_s"]) for row in replicate_rows
        ]),
        "total_selection_wall_time_s": float(sum(
            float(row["replicate_wall_time_s"]) for row in replicate_rows
        )),
        "variance_note": (
            "Between-selection summaries use one independent-validation mean per completed "
            "design replicate. Conditional retraining rows share one fixed external holdout and "
            "do not estimate dataset-sampling uncertainty."
        ),
        "replicates": repeated_doe["replicate_records"],
        "methods": methods,
    }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _installed_version(distribution: str) -> Optional[str]:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _write_run_provenance(
    destination: Path,
    *,
    mode: str,
    settings: Dict[str, Any],
) -> None:
    baseline_inputs = (
        "results/runs.csv",
        "results/confirmation.json",
        "results/phase3.json",
    )
    runtime_sources = (
        "config.yaml",
        "pipeline.py",
        "analysis.py",
        "latency.py",
        "final_evaluation.py",
        "scientific_stats.py",
        "scripts/run_benchmarks.py",
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    provenance = {
        "schema_version": 1,
        "producer": "scripts/run_benchmarks.py",
        "classification": "new revision_v2 experiment",
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_head(),
        "git_worktree_status_short": status,
        "historical_baseline_ref": HISTORICAL_BASELINE_REF,
        "historical_baseline_tag_object": subprocess.run(
            ["git", "rev-parse", HISTORICAL_BASELINE_REF],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip(),
        "historical_baseline_revision": subprocess.run(
            ["git", "rev-parse", f"{HISTORICAL_BASELINE_REF}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip(),
        "settings": settings,
        "baseline_input_sha256": {
            f"{HISTORICAL_BASELINE_REF}:{path}": _sha256_bytes(_git_show_bytes(path))
            for path in baseline_inputs
        },
        "runtime_source_sha256": {
            path: _sha256_file(Path(path)) for path in runtime_sources
        },
        "dataset": {
            "name": "scikit-learn California housing",
            "loader": "sklearn.datasets.fetch_california_housing",
            "external_holdout_split_seed": 42,
            "development_split_seed": int(
                CONFIG["revision_v2"]["development_split_seed"]
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER", ""),
            "logical_cpu_count": os.cpu_count(),
            "packages": {
                name: _installed_version(distribution)
                for name, distribution in {
                    "numpy": "numpy",
                    "pandas": "pandas",
                    "scipy": "scipy",
                    "scikit_learn": "scikit-learn",
                    "statsmodels": "statsmodels",
                    "xgboost": "xgboost",
                    "optuna": "optuna",
                }.items()
            },
        },
        "artifact_sha256": {
            path.name: _sha256_file(path)
            for path in sorted(destination.iterdir())
            if path.is_file() and path.name != "provenance.json"
        },
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def run_revision_benchmark(
    mode: str = "smoke",
    *,
    output_dir: Optional[str] = None,
    data_mgr=None,
    final_evaluator=None,
    settings_override: Optional[Dict[str, Any]] = None,
    include_historical_doe: bool = True,
    include_repeated_doe: bool = True,
) -> Path:
    """Run the versioned benchmark workflow without overwriting historical artifacts."""
    modes = CONFIG["revision_v2"]["modes"]
    if mode not in modes:
        raise ValueError(f"Unknown benchmark mode {mode!r}; choose one of {sorted(modes)}")
    settings = dict(modes[mode])
    if settings_override:
        settings.update(settings_override)
    destination = Path(output_dir) if output_dir else _default_revision_output(mode)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    budget = _computational_budget(
        mode,
        settings,
        include_historical_doe=include_historical_doe,
        include_repeated_doe=include_repeated_doe,
    )
    (destination / "computational_budget.json").write_text(
        json.dumps(budget, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(budget, indent=2))

    development_seed = int(CONFIG["revision_v2"]["development_split_seed"])
    manager = data_mgr or CaliforniaHousingDevelopmentDataManager()
    replicate_count = int(settings["optimizer_replicates"])
    sampler_seeds = OPTIMIZER_SEEDS[:replicate_count]
    trials = int(settings["trials_per_optimizer"])
    max_latency = float(CONFIG["revision_v2"]["max_latency_us"])

    replicate_outputs = []
    replicate_rows = []
    trial_rows = []
    for replicate_id, sampler_seed in enumerate(sampler_seeds):
        started = time.perf_counter()
        output = run_single_optimizer_replicate(
            sampler_seed,
            development_seed,
            replicate_id=replicate_id,
            n_trials=trials,
            max_latency_us=max_latency,
            data_mgr=manager,
        )
        wall_time = time.perf_counter() - started
        replicate_outputs.append(output)
        trial_rows.extend(output["trial_records"])
        for result in output["search_results"].values():
            selected = _selected_record(result)
            replicate_rows.append({
                "optimizer": result.optimizer,
                "replicate_id": replicate_id,
                "sampler_seed": sampler_seed,
                "development_split_seed": development_seed,
                "status": result.status,
                "selected_trial_number": None if selected is None else selected["trial_number"],
                "selected_score": result.selected_score,
                "selection_rule": result.selection_rule,
                "model_fit_evaluations": len(result.trial_records),
                "method_wall_time_s": output["method_wall_times_s"][result.optimizer],
                "replicate_wall_time_s_all_four_methods": wall_time,
            })
    _write_csv(trial_rows, destination / "optimizer_trials.csv")
    _write_csv(replicate_rows, destination / "optimizer_replicates.csv")

    configurations = []
    for output in replicate_outputs:
        for result in output["search_results"].values():
            selected = _selected_record(result)
            if selected is not None:
                selection_id = f"{result.optimizer}-rep-{result.replicate_id}"
                configurations.append(_manifest_configuration(selected, selection_id))
    repeated_doe = None
    if include_repeated_doe:
        repeated_doe = run_repeated_doe_selection(
            manager,
            replicate_count=int(settings["doe_block_seed_sets"]),
            seeds_per_set=int(settings["doe_seeds_per_set"]),
            seed_base=int(settings["doe_seed_base"]),
            max_latency_us=max_latency,
        )
        configurations.extend(repeated_doe["configurations"])
        _write_csv(
            repeated_doe["run_records"], destination / "doe_selection_runs.csv"
        )
        _write_csv(
            repeated_doe["candidate_records"],
            destination / "doe_selection_candidates.csv",
        )
        _write_csv(
            repeated_doe["replicate_records"],
            destination / "doe_selection_replicates.csv",
        )
        (destination / "doe_selection_protocol.json").write_text(
            json.dumps(repeated_doe["grid"], indent=2) + "\n", encoding="utf-8"
        )
    if include_historical_doe:
        configurations.extend(_historical_doe_configurations())

    evaluator = final_evaluator or FinalTestEvaluator()
    if not hasattr(evaluator, "evaluation_protocol"):
        raise TypeError("final_evaluator must expose an auditable evaluation_protocol")
    evaluation_seeds = [int(seed) for seed in settings["evaluation_seeds"]]
    manifest_path = destination / "finalized_selections.json"
    finalized_development_seeds = [development_seed]
    if repeated_doe is not None:
        finalized_development_seeds.extend(
            seed
            for replicate in repeated_doe["replicate_records"]
            for seed in replicate["block_seeds"]
        )
    create_finalized_selection(
        manifest_path,
        configurations=configurations,
        selection_policy=(
            "Per-replicate actual winning trial selected on development data only; "
            "repeated DOE configurations selected from a predeclared surrogate grid; "
            "historical DOE selections retain an explicit prior-exposure caveat."
        ),
        development_seeds=sorted(set(finalized_development_seeds)),
        evaluation_seeds=evaluation_seeds,
        evaluation_protocol=evaluator.evaluation_protocol,
        source_artifact=str(destination),
        git_revision=_git_head(),
    )

    latency_session_count = int(settings["latency_sessions"])
    primary_sessions = [
        _measure_frozen_primary_latencies(
            configurations,
            manager,
            development_seed,
            session_id=f"revision-v2-{mode}-frozen-primary-session-{session_index + 1}",
        )
        for session_index in range(latency_session_count)
    ]
    primary_measurement = _aggregate_primary_latency_sessions(primary_sessions)
    (destination / "latency_measurement.json").write_text(
        json.dumps(primary_measurement, indent=2) + "\n", encoding="utf-8"
    )
    latency_overhead = _latency_interface_overhead(primary_measurement)
    (destination / "latency_interface_overhead.json").write_text(
        json.dumps(latency_overhead, indent=2) + "\n", encoding="utf-8"
    )

    final_rows = []
    for configuration in configurations:
        result = evaluator.evaluate(
            manifest_path,
            configuration["selection_id"],
            evaluation_seeds=evaluation_seeds,
        )
        for row in result["per_seed"]:
            final_rows.append({
                "selection_id": configuration["selection_id"],
                "optimizer": configuration["optimizer"],
                "optimizer_replicate_id": configuration.get("optimizer_replicate_id"),
                "config_sha256": result["config_sha256"],
                **row,
            })
    _write_csv(final_rows, destination / "final_evaluations.csv")
    failed_final_rows = [row for row in final_rows if row.get("status") != "completed"]
    if failed_final_rows:
        (destination / "final_evaluation_failures.json").write_text(
            json.dumps(failed_final_rows, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"{len(failed_final_rows)} frozen-selection evaluations failed; "
            "the failure ledger was retained and no aggregate claims were produced"
        )

    config_by_id = {item["selection_id"]: item for item in configurations}
    summary_rows = []
    final_frame = pd.DataFrame(final_rows)
    for selection_id, group in final_frame.groupby("selection_id", sort=True):
        latency_summary = primary_measurement["summaries"][selection_id]
        configuration = config_by_id[selection_id]
        selection_metric = configuration.get("selection_metric", {})
        if configuration.get("predicted_validation_rmse") is not None:
            search_validation_rmse = float(
                configuration["predicted_validation_rmse"]
            )
        elif (
            selection_metric.get("name") == "validation_rmse"
            and selection_metric.get("value") is not None
        ):
            search_validation_rmse = float(selection_metric["value"])
        else:
            search_validation_rmse = None
        independent_validation_mean = _metric_summary(group["val_rmse"].tolist())
        summary_rows.append({
            "selection_id": selection_id,
            "optimizer": configuration["optimizer"],
            "optimizer_replicate_id": configuration.get("optimizer_replicate_id"),
            "search_validation_rmse": search_validation_rmse,
            "validation_rmse": independent_validation_mean,
            "test_rmse": _metric_summary(group["test_rmse"].tolist()),
            "selection_optimism_independent_minus_search_rmse": (
                None
                if search_validation_rmse is None
                else independent_validation_mean["mean"] - search_validation_rmse
            ),
            **latency_summary,
            "latency_protocol_id": PRIMARY_V1.protocol_id,
            "search_time_feasible": configuration.get("search_time_feasible"),
            "benchmark_time_feasible": bool(
                latency_summary["predict_latency_us"] <= max_latency
            ),
        })
    _write_csv(summary_rows, destination / "final_summary.csv")
    optimizer_summary = _optimizer_level_summary(
        mode, replicate_rows, trial_rows, configurations, summary_rows
    )
    (destination / "optimizer_summary.json").write_text(
        json.dumps(optimizer_summary, indent=2) + "\n", encoding="utf-8"
    )
    doe_summary = _doe_level_summary(mode, repeated_doe, summary_rows)
    (destination / "doe_selection_summary.json").write_text(
        json.dumps(doe_summary, indent=2) + "\n", encoding="utf-8"
    )

    paired = []
    pair_specs = [
        ("doe-mo-historical", "multi_objective_tpe", "DOE MO minus MO-TPE"),
        ("doe-mo-historical", "constrained_tpe", "DOE MO minus constrained TPE"),
        ("doe-so-historical", "single_objective_tpe", "DOE SO minus SO-TPE"),
        ("doe-so-historical", "random_search", "DOE SO minus random search"),
    ]
    equivalence_margin = CONFIG["revision_v2"].get("rmse_equivalence_margin")
    if include_historical_doe:
        for anchor_id, comparator_optimizer, label in pair_specs:
            anchor = final_frame[final_frame["selection_id"] == anchor_id].sort_values("evaluation_seed")
            comparators = [
                item for item in configurations if item["optimizer"] == comparator_optimizer
            ]
            for comparator in comparators:
                other = final_frame[
                    final_frame["selection_id"] == comparator["selection_id"]
                ].sort_values("evaluation_seed")
                if anchor["evaluation_seed"].tolist() != other["evaluation_seed"].tolist():
                    raise RuntimeError("paired comparisons require identical evaluation seeds")
                for metric in ("val_rmse", "test_rmse"):
                    paired.append({
                        "comparison": label,
                        "comparison_classification": (
                            "historical_fixed_selection_conditional_descriptive"
                        ),
                        "anchor_selection_id": anchor_id,
                        "comparator_selection_id": comparator["selection_id"],
                        "metric": metric,
                        "replication_unit": "retraining_seed_pair",
                        "scope": (
                            "Conditional descriptive comparison of frozen selections; the common "
                            "fixed holdout and overlapping development partitions are retained."
                        ),
                        "conditional_difference": _conditional_difference_description(
                            anchor[metric].to_numpy(), other[metric].to_numpy()
                        ),
                    })
    if repeated_doe is not None:
        repeated_pairs = (
            ("doe-mo-rep-{replicate}", "multi_objective_tpe", "Repeated DOE MO minus MO-TPE"),
            ("doe-mo-rep-{replicate}", "constrained_tpe", "Repeated DOE MO minus constrained TPE"),
            ("doe-so-rep-{replicate}", "single_objective_tpe", "Repeated DOE SO minus SO-TPE"),
            ("doe-so-rep-{replicate}", "random_search", "Repeated DOE SO minus random search"),
        )
        for anchor_template, comparator_optimizer, label in repeated_pairs:
            for metric in ("val_rmse", "test_rmse"):
                planned_replicates = list(
                    range(int(settings["doe_block_seed_sets"]))
                )
                replicate_differences = []
                omitted_replicates = []
                anchor_means = []
                comparator_means = []
                for replicate_id in planned_replicates:
                    anchor_id = anchor_template.format(replicate=replicate_id)
                    anchor = final_frame[
                        final_frame["selection_id"] == anchor_id
                    ].sort_values("evaluation_seed")
                    if anchor.empty:
                        doe_record = next(
                            (
                                row
                                for row in repeated_doe["replicate_records"]
                                if int(row["doe_replicate_id"]) == replicate_id
                            ),
                            None,
                        )
                        omitted_replicates.append({
                            "selection_replicate_id": replicate_id,
                            "missing_method": "repeated_doe",
                            "reason": (
                                doe_record.get("error")
                                if doe_record is not None
                                else "no repeated DOE replicate record"
                            ),
                        })
                        continue
                    comparators = [
                        item
                        for item in configurations
                        if item["optimizer"] == comparator_optimizer
                        and item.get("optimizer_replicate_id") == replicate_id
                    ]
                    if not comparators:
                        optimizer_record = next(
                            (
                                row
                                for row in replicate_rows
                                if row["optimizer"] == comparator_optimizer
                                and int(row["replicate_id"]) == replicate_id
                            ),
                            None,
                        )
                        omitted_replicates.append({
                            "selection_replicate_id": replicate_id,
                            "missing_method": comparator_optimizer,
                            "reason": (
                                f"search status: {optimizer_record['status']}"
                                if optimizer_record is not None
                                else "no optimizer replicate record"
                            ),
                        })
                        continue
                    comparator = comparators[0]
                    other = final_frame[
                        final_frame["selection_id"] == comparator["selection_id"]
                    ].sort_values("evaluation_seed")
                    if anchor["evaluation_seed"].tolist() != other["evaluation_seed"].tolist():
                        raise RuntimeError("paired comparisons require identical evaluation seeds")
                    anchor_mean = float(anchor[metric].mean())
                    comparator_mean = float(other[metric].mean())
                    anchor_means.append(anchor_mean)
                    comparator_means.append(comparator_mean)
                    replicate_differences.append({
                        "selection_replicate_id": replicate_id,
                        "anchor_selection_id": anchor_id,
                        "comparator_selection_id": comparator["selection_id"],
                        "anchor_retraining_seed_mean": anchor_mean,
                        "comparator_retraining_seed_mean": comparator_mean,
                        "difference": anchor_mean - comparator_mean,
                        "conditional_retraining_seed_description": (
                            _conditional_difference_description(
                                anchor[metric].to_numpy(), other[metric].to_numpy()
                            )
                        ),
                    })
                enough_replicates = len(anchor_means) >= 2
                complete_pair_set = (
                    len(replicate_differences) == len(planned_replicates)
                )
                if mode == "smoke":
                    inference_status = (
                        "smoke_underpowered"
                        if complete_pair_set
                        else "smoke_underpowered_available_case_conditional_on_success"
                    )
                else:
                    inference_status = (
                        "full_protocol"
                        if complete_pair_set and enough_replicates
                        else "available_case_conditional_on_success"
                    )
                paired.append({
                    "comparison": label,
                    "comparison_classification": (
                        "new_end_to_end_selection_replicate_comparison"
                    ),
                    "metric": metric,
                    "replication_unit": (
                        "one retraining-seed mean per matched search/design replicate"
                    ),
                    "estimand": "conditional_on_both_selections_succeeding",
                    "inference_status": inference_status,
                    "scope": (
                        "End-to-end selection variability across matched replicate IDs. Seeded "
                        "development partitions overlap, and every test metric uses the same fixed "
                        "external holdout, so dataset-sampling uncertainty is not estimated."
                    ),
                    "planned_selection_replicate_ids": planned_replicates,
                    "included_selection_replicate_ids": [
                        item["selection_replicate_id"]
                        for item in replicate_differences
                    ],
                    "omitted_selection_replicates": omitted_replicates,
                    "predeclared_pair_set_complete": complete_pair_set,
                    "per_selection_replicate": replicate_differences,
                    "paired_difference": (
                        paired_difference_summary(anchor_means, comparator_means)
                        if enough_replicates
                        else {
                            "performed": False,
                            "reason": "fewer_than_two_completed_selection_replicates",
                        }
                    ),
                    "equivalence": (
                        paired_tost(
                            anchor_means,
                            comparator_means,
                            margin=equivalence_margin,
                        )
                        if enough_replicates
                        else {
                            "performed": False,
                            "equivalent": None,
                            "reason": "fewer_than_two_completed_selection_replicates",
                        }
                    ),
                })
    (destination / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )

    hv_report = {
        "inference_status": (
            "smoke_underpowered" if mode == "smoke" else "full_protocol"
        ),
        "development_domain": {
            "objective_definition": ["validation_rmse", "predict_latency_us"],
            "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id,
            "estimand_note": (
                "Historical matched DOE coordinates and optimizer trials use the same single split. "
                "Repeated DOE fronts average distinct seeded partitions of the same development "
                "pool; those partitions can overlap and are not independent dataset samples. The "
                "repeated-DOE and MO-TPE hypervolumes are reported separately because they use "
                "different RMSE estimands."
            ),
            "reference_points": {},
        },
        "external_test_domain": {
            "interpretation": "retrospective evaluation of frozen selections only",
            "objective_definition": ["mean_test_rmse", "predict_latency_us"],
            "latency_protocol_id": PRIMARY_V1.protocol_id,
            "reference_points": {},
        },
    }
    doe_front_rows = []
    if include_historical_doe:
        historical_selections = [
            item
            for item in configurations
            if item["optimizer"].startswith("historical_preplanned_doe_")
        ]
        doe_front_rows = _evaluate_historical_doe_front(
            manager,
            development_seed,
            selected_configurations=historical_selections,
        )
        _write_csv(doe_front_rows, destination / "doe_matched_candidate_front.csv")
    references = CONFIG["revision_v2"]["hypervolume_reference_points"]
    mo_trials = [
        row for row in trial_rows
        if row["optimizer"] == "multi_objective_tpe" and row["trial_status"] == "completed"
    ]
    for reference in references:
        key = json.dumps(reference)
        development_entry = {
            "mo_tpe_by_replicate": {},
            "repeated_doe_by_replicate": {},
        }
        doe_full_value = None
        if doe_front_rows:
            doe_coordinate_rows = [
                row
                for row in doe_front_rows
                if row["candidate_type"] == "historical_doe_evaluated_coordinate"
            ]
            development_entry["historical_doe_design_coordinate_front"] = _hypervolume_payload(
                [
                    (row["validation_rmse"], row["predict_latency_us"])
                    for row in doe_coordinate_rows
                ],
                reference,
            )
            development_entry["full_doe_evaluated_candidate_front"] = _hypervolume_payload(
                [
                    (row["validation_rmse"], row["predict_latency_us"])
                    for row in doe_front_rows
                ],
                reference,
            )
            doe_full_value = development_entry[
                "full_doe_evaluated_candidate_front"
            ]["value"]
            doe_selected = {
                row["candidate_id"]: row
                for row in doe_front_rows
                if row["candidate_type"] == "historical_doe_selected_operating_point"
            }
            development_entry["doe_selected_point"] = _hypervolume_payload(
                [(
                    doe_selected["doe-mo-historical"]["validation_rmse"],
                    doe_selected["doe-mo-historical"]["predict_latency_us"],
                )],
                reference,
            )
            development_entry["doe_two_point"] = _hypervolume_payload(
                [
                    (
                        doe_selected["doe-mo-historical"]["validation_rmse"],
                        doe_selected["doe-mo-historical"]["predict_latency_us"],
                    ),
                    (
                        doe_selected["doe-so-historical"]["validation_rmse"],
                        doe_selected["doe-so-historical"]["predict_latency_us"],
                    ),
                ],
                reference,
            )
        mo_hypervolumes = []
        for replicate_id in sorted({row["replicate_id"] for row in mo_trials}):
            points = [
                (row["validation_rmse"], row["predict_latency_us"])
                for row in mo_trials if row["replicate_id"] == replicate_id
            ]
            payload = _hypervolume_payload(points, reference)
            development_entry["mo_tpe_by_replicate"][str(replicate_id)] = payload
            mo_hypervolumes.append(float(payload["value"]))
        development_entry["mo_tpe_hypervolume_distribution"] = _metric_summary(
            mo_hypervolumes
        )
        repeated_doe_hypervolumes = []
        if repeated_doe is not None:
            completed_doe_replicates = [
                int(row["doe_replicate_id"])
                for row in repeated_doe["replicate_records"]
                if row["status"] == "completed"
            ]
            for replicate_id in completed_doe_replicates:
                candidates = [
                    row
                    for row in repeated_doe["candidate_records"]
                    if row["doe_replicate_id"] == replicate_id
                ]
                payload = _hypervolume_payload(
                    [
                        (
                            row["validation_rmse_mean"],
                            row["predict_latency_us_mean"],
                        )
                        for row in candidates
                    ],
                    reference,
                )
                development_entry["repeated_doe_by_replicate"][str(replicate_id)] = payload
                repeated_doe_hypervolumes.append(float(payload["value"]))
            development_entry["repeated_doe_hypervolume_distribution"] = _metric_summary(
                repeated_doe_hypervolumes
            )
        if doe_full_value is not None:
            differences = [doe_full_value - value for value in mo_hypervolumes]
            development_entry["full_doe_candidate_front_minus_mo_tpe_hypervolume"] = {
                "sign_convention": "positive values favor the full DOE evaluated front",
                "summary": _metric_summary(differences),
                "per_replicate": differences,
            }
        hv_report["development_domain"]["reference_points"][key] = development_entry

        external_points = {
            row["selection_id"]: (
                row["test_rmse"]["mean"], row["predict_latency_us"]
            )
            for row in summary_rows
        }
        external_entry = {
            "all_frozen_selections": _hypervolume_payload(external_points.values(), reference)
        }
        if include_historical_doe:
            external_entry["doe_selected_point"] = _hypervolume_payload(
                [external_points["doe-mo-historical"]], reference
            )
            external_entry["doe_two_point"] = _hypervolume_payload(
                [external_points["doe-mo-historical"], external_points["doe-so-historical"]],
                reference,
            )
        hv_report["external_test_domain"]["reference_points"][key] = external_entry
    (destination / "hypervolume.json").write_text(
        json.dumps(hv_report, indent=2) + "\n", encoding="utf-8"
    )

    artifact_names = sorted(
        {path.name for path in destination.iterdir()}
        | {"run_manifest.json", "provenance.json"}
    )
    run_manifest = {
        "schema_version": 1,
        "classification": "new revision_v2 experiment",
        "mode": mode,
        "inference_status": (
            "smoke_underpowered" if mode == "smoke" else "full_protocol"
        ),
        "git_revision": _git_head(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "artifacts": artifact_names,
        "historical_artifacts_modified": False,
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    _write_run_provenance(destination, mode=mode, settings=settings)
    return destination


def main():
    parser = argparse.ArgumentParser(
        description="Run versioned development-only optimizer benchmarks and gated final evaluation."
    )
    parser.add_argument("--mode", choices=sorted(CONFIG["revision_v2"]["modes"]), default="smoke")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--confirm-full-budget",
        action="store_true",
        help="Required with --mode full after reviewing the computational budget.",
    )
    args = parser.parse_args()
    if args.mode == "full" and not args.confirm_full_budget:
        budget = _computational_budget("full", CONFIG["revision_v2"]["modes"]["full"])
        raise SystemExit(
            "Full mode requires --confirm-full-budget after reviewing this estimate:\n"
            + json.dumps(budget, indent=2)
        )
    destination = run_revision_benchmark(args.mode, output_dir=args.output_dir)
    print(f"Revision benchmark artifacts written to {destination}")


if __name__ == "__main__":
    main()
