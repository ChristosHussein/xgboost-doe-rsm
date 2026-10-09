"""
scripts/run_benchmarks.py - Task 7: Fair Empirical Benchmarks & Multi-Objective Comparison
========================================================================================
Implements:
1. All methods optimize validation RMSE on the 80% development pool.
2. Baselines (20 independent optimizer replicates, 140 trials each):
   - Unguided Random Search
   - Single-Objective TPE (optimizing Val RMSE)
   - Constrained TPE (Val RMSE s.t. Latency <= 145 us, strictly enforced)
   - Multi-Objective TPE (minimizing Val RMSE and Latency)
3. Evaluates all candidates across 20 fresh evaluation seeds [2001..2020]:
   - Validation RMSE (development fold)
   - Generalization Test RMSE (fixed 20% holdout test set, completely untouched)
4. Includes both Sequential DOE candidates:
   - Sequential DOE (x*, Multi-Objective, Depth 4)
   - Sequential DOE (Single-Objective Optimum, Depth 7)
5. Pinned CPU affinity, interleaved latency measurement across all methods
   (5 randomized blocks of 200 calls = 1000 calls total), reporting both predict() and inplace_predict().
6. Computes Paired t-tests and Wilcoxon signed-rank tests across the 20 shared evaluation seeds.
7. Computes Hypervolume (HV) indicator for multi-objective comparison.
8. Desirability sensitivity grid over (L, U, w).
"""

import json
import os
import sys
import time
import argparse
import subprocess
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
    rs = run_random_search(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    tpe = run_tpe_single_objective(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    constrained = run_tpe_constrained(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        max_latency_us=max_latency_us,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    multi = run_tpe_multi_objective(
        n_trials,
        sampler_seed=sampler_seed,
        eval_seed=eval_seed,
        data_mgr=dm,
        replicate_id=replicate_id,
    )
    return {
        "seed": sampler_seed,
        "replicate_id": replicate_id,
        "search_results": {
            rs.optimizer: rs,
            tpe.optimizer: tpe,
            constrained.optimizer: constrained,
            multi.optimizer: multi,
        },
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
    with open("results/confirmation.json", "r", encoding="utf-8") as handle:
        confirmation = json.load(handle)
    with open("results/phase3.json", "r", encoding="utf-8") as handle:
        phase3 = json.load(handle)
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


def _evaluate_historical_doe_front(data_mgr, split_seed: int) -> List[Dict[str, Any]]:
    historical = pd.read_csv("results/runs.csv")
    unique = historical.sort_values("point_id").drop_duplicates("point_id")
    split = _development_split(data_mgr, split_seed)
    rows = []
    for row in unique.itertuples(index=False):
        x = np.asarray([row.x1, row.x2, row.x3, row.x4], dtype=float)
        eta, depth, subsample, reg_lambda = decode_factors(x)
        model = xgb.XGBRegressor(
            **_model_config(eta, depth, subsample, reg_lambda, split_seed)
        )
        model.fit(split.X_train, split.y_train)
        rmse = float(np.sqrt(np.mean((split.y_val - model.predict(split.X_val)) ** 2)))
        session = f"doe-point-{int(row.point_id)}-matched-online"
        latency = measure_trial_latency(model, split.X_val[:1], session_id=session)
        rows.append({
            "candidate_id": f"doe-point-{int(row.point_id)}",
            "candidate_type": "historical_doe_evaluated_coordinate",
            "validation_rmse": rmse,
            "predict_latency_us": latency,
            "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id,
            "development_split_seed": split_seed,
            "x1": float(row.x1),
            "x2": float(row.x2),
            "x3": float(row.x3),
            "x4": float(row.x4),
        })
    return rows


def _hypervolume_payload(points, reference) -> Dict[str, Any]:
    result = hypervolume_2d_min(points, reference)
    return {
        "value": result.value,
        "reference": list(result.reference),
        "pareto": asdict(result.pareto),
    }


def _computational_budget(mode: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    replicates = int(settings["optimizer_replicates"])
    trials = int(settings["trials_per_optimizer"])
    evaluation_seeds = len(settings["evaluation_seeds"])
    frozen_max = 4 * replicates + 2
    return {
        "mode": mode,
        "optimizer_methods": 4,
        "optimizer_replicates": replicates,
        "trials_per_optimizer": trials,
        "search_model_fits": 4 * replicates * trials,
        "matched_historical_doe_coordinate_fits": 25,
        "maximum_frozen_configurations": frozen_max,
        "final_retraining_fits": frozen_max * evaluation_seeds,
        "maximum_total_model_fits": 4 * replicates * trials + 25 + frozen_max * evaluation_seeds,
        "online_latency_calls_per_search_trial": ONLINE_SEARCH_V1.timed_calls_per_interface,
        "primary_latency_calls_per_frozen_configuration_per_interface": PRIMARY_V1.timed_calls_per_interface,
        "note": "The total is an upper bound; failed or infeasible searches produce no frozen configuration.",
    }


def run_revision_benchmark(
    mode: str = "smoke",
    *,
    output_dir: Optional[str] = None,
    data_mgr=None,
    final_evaluator=None,
    settings_override: Optional[Dict[str, Any]] = None,
    include_historical_doe: bool = True,
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

    budget = _computational_budget(mode, settings)
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
    if include_historical_doe:
        configurations.extend(_historical_doe_configurations())

    manifest_path = destination / "finalized_selections.json"
    create_finalized_selection(
        manifest_path,
        configurations=configurations,
        selection_policy=(
            "Per-replicate actual winning trial selected on development data only; "
            "historical DOE selections retain an explicit prior-exposure caveat."
        ),
        development_seeds=[development_seed],
        source_artifact=str(destination / "optimizer_trials.csv"),
        git_revision=_git_head(),
    )

    primary_measurement = _measure_frozen_primary_latencies(
        configurations,
        manager,
        development_seed,
        session_id=f"revision-v2-{mode}-frozen-primary",
    )
    (destination / "latency_measurement.json").write_text(
        json.dumps(primary_measurement, indent=2) + "\n", encoding="utf-8"
    )

    evaluator = final_evaluator or FinalTestEvaluator()
    evaluation_seeds = [int(seed) for seed in settings["evaluation_seeds"]]
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

    config_by_id = {item["selection_id"]: item for item in configurations}
    summary_rows = []
    final_frame = pd.DataFrame(final_rows)
    for selection_id, group in final_frame.groupby("selection_id", sort=True):
        latency_summary = primary_measurement["summaries"][selection_id]
        configuration = config_by_id[selection_id]
        summary_rows.append({
            "selection_id": selection_id,
            "optimizer": configuration["optimizer"],
            "optimizer_replicate_id": configuration.get("optimizer_replicate_id"),
            "validation_rmse": _metric_summary(group["val_rmse"].tolist()),
            "test_rmse": _metric_summary(group["test_rmse"].tolist()),
            **latency_summary,
            "latency_protocol_id": PRIMARY_V1.protocol_id,
            "search_time_feasible": configuration.get("search_time_feasible"),
            "benchmark_time_feasible": bool(
                latency_summary["predict_latency_us"] <= max_latency
            ),
        })
    _write_csv(summary_rows, destination / "final_summary.csv")

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
                        "anchor_selection_id": anchor_id,
                        "comparator_selection_id": comparator["selection_id"],
                        "metric": metric,
                        "scope": "conditional on frozen selections; common holdout dependence retained",
                        "paired_difference": paired_difference_summary(
                            anchor[metric].to_numpy(), other[metric].to_numpy()
                        ),
                        "equivalence": paired_tost(
                            anchor[metric].to_numpy(),
                            other[metric].to_numpy(),
                            margin=equivalence_margin,
                        ),
                    })
    (destination / "paired_comparisons.json").write_text(
        json.dumps(paired, indent=2) + "\n", encoding="utf-8"
    )

    hv_report = {
        "development_domain": {
            "objective_definition": ["validation_rmse", "predict_latency_us"],
            "latency_protocol_id": ONLINE_SEARCH_V1.protocol_id,
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
        doe_front_rows = _evaluate_historical_doe_front(manager, development_seed)
        _write_csv(doe_front_rows, destination / "doe_matched_candidate_front.csv")
    references = CONFIG["revision_v2"]["hypervolume_reference_points"]
    mo_trials = [
        row for row in trial_rows
        if row["optimizer"] == "multi_objective_tpe" and row["trial_status"] == "completed"
    ]
    for reference in references:
        key = json.dumps(reference)
        development_entry = {"mo_tpe_by_replicate": {}}
        if doe_front_rows:
            development_entry["full_doe_evaluated_front"] = _hypervolume_payload(
                [(row["validation_rmse"], row["predict_latency_us"]) for row in doe_front_rows],
                reference,
            )
        for replicate_id in sorted({row["replicate_id"] for row in mo_trials}):
            points = [
                (row["validation_rmse"], row["predict_latency_us"])
                for row in mo_trials if row["replicate_id"] == replicate_id
            ]
            development_entry["mo_tpe_by_replicate"][str(replicate_id)] = _hypervolume_payload(
                points, reference
            )
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

    run_manifest = {
        "schema_version": 1,
        "classification": "new revision_v2 experiment",
        "mode": mode,
        "git_revision": _git_head(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "artifacts": sorted(path.name for path in destination.iterdir()),
        "historical_artifacts_modified": False,
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
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
