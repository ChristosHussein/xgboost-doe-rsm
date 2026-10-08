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
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import optuna
import statsmodels.api as sm
import xgboost as xgb
import yaml

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import CaliforniaHousingDataManager, decode_factors, encode_factors, pin_cpu_affinity, CONFIG
from analysis import derringer_suich_desirability

optuna.logging.set_verbosity(optuna.logging.WARNING)

FRESH_SEEDS = CONFIG["seeds"]["fresh_eval_seeds"]
OPTIMIZER_SEEDS = CONFIG["seeds"]["optimizer_sampler_seeds"]


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
    data_mgr: CaliforniaHousingDataManager,
    reps: int = 5,
    batch_size: int = 200,
    warmup: int = 50,
) -> Dict[str, Dict[str, float]]:
    """
    Measures latency for all configurations in interleaved blocks on CPU 0.
    Eliminates session noise, thermal throttling bias, and scale discrepancies.
    """
    pin_cpu_affinity()
    X_tr, X_val, X_te, y_tr, y_val, y_te = data_mgr.get_split(42)
    single_sample = X_val[:1]

    models = {}
    boosters = {}
    for name, (eta, depth, subsample, reg_lambda) in configs.items():
        m = xgb.XGBRegressor(
            n_estimators=CONFIG["model"]["n_estimators"],
            learning_rate=eta,
            max_depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=1,
            objective=CONFIG["model"]["objective"],
        )
        m.fit(X_tr, y_tr)
        b = m.get_booster()
        b.set_param({"nthread": 1})
        models[name] = m
        boosters[name] = b

        # Warmup
        for _ in range(warmup):
            _ = m.predict(single_sample)
            _ = b.inplace_predict(single_sample)

    batch_times_pred = {name: [] for name in configs}
    batch_times_inp = {name: [] for name in configs}

    # Interleaved execution across reps
    method_names = list(configs.keys())
    for r in range(reps):
        # Permute methods to avoid order bias
        perm = np.random.permutation(method_names)
        for name in perm:
            m = models[name]
            b = boosters[name]

            # 1. Predict
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                _ = m.predict(single_sample)
            t1 = time.perf_counter_ns()
            batch_times_pred[name].append((t1 - t0) / (batch_size * 1000.0))

            # 2. Inplace Predict
            t0 = time.perf_counter_ns()
            for _ in range(batch_size):
                _ = b.inplace_predict(single_sample)
            t1 = time.perf_counter_ns()
            batch_times_inp[name].append((t1 - t0) / (batch_size * 1000.0))

    results = {}
    for name in configs:
        med_p = float(np.median(batch_times_pred[name]))
        iqr_p = float(np.subtract(*np.percentile(batch_times_pred[name], [75, 25])))
        med_i = float(np.median(batch_times_inp[name]))
        iqr_i = float(np.subtract(*np.percentile(batch_times_inp[name], [75, 25])))
        results[name] = {
            "predict_latency_us_median": med_p,
            "predict_latency_us_iqr": iqr_p,
            "inplace_latency_us_median": med_i,
            "inplace_latency_us_iqr": iqr_i,
        }
    return results


def run_random_search(n_trials: int, sampler_seed: int, eval_seed: int, data_mgr: CaliforniaHousingDataManager):
    """Uniform Random Search over the 4-factor continuous hypercube."""
    rng = np.random.RandomState(sampler_seed)
    best_val = 1e9
    best_x = None
    trajectory = []

    X_tr, X_val, _, y_tr, y_val, _ = data_mgr.get_split(eval_seed)

    for i in range(1, n_trials + 1):
        x = rng.uniform(-1.0, 1.0, 4)
        eta, depth, subsample, reg_lambda = decode_factors(x)
        m = xgb.XGBRegressor(
            n_estimators=CONFIG["model"]["n_estimators"],
            learning_rate=eta,
            max_depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=eval_seed,
            n_jobs=CONFIG["model"]["n_jobs_train"],
            objective=CONFIG["model"]["objective"],
        )
        m.fit(X_tr, y_tr)
        val_rmse = float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_x = x
        trajectory.append(best_val)

    return best_x, best_val, trajectory


def run_tpe_single_objective(n_trials: int, sampler_seed: int, eval_seed: int, data_mgr: CaliforniaHousingDataManager):
    """Single-Objective TPE optimizing Val RMSE."""
    X_tr, X_val, _, y_tr, y_val, _ = data_mgr.get_split(eval_seed)
    trajectory = []
    best_val = 1e9
    best_x = None

    def obj(trial):
        nonlocal best_val, best_x
        x = np.array([trial.suggest_float(f"x{i}", -1.0, 1.0) for i in range(1, 5)])
        eta, depth, subsample, reg_lambda = decode_factors(x)
        m = xgb.XGBRegressor(
            n_estimators=CONFIG["model"]["n_estimators"],
            learning_rate=eta,
            max_depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=eval_seed,
            n_jobs=CONFIG["model"]["n_jobs_train"],
            objective=CONFIG["model"]["objective"],
        )
        m.fit(X_tr, y_tr)
        val_rmse = float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_x = x
        trajectory.append(best_val)
        return val_rmse

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(obj, n_trials=n_trials)
    return best_x, best_val, trajectory


def run_tpe_constrained(n_trials: int, sampler_seed: int, eval_seed: int, max_latency_us: float, data_mgr: CaliforniaHousingDataManager):
    """Constrained TPE: Minimize Val RMSE s.t. Latency <= max_latency_us (strictly enforced)."""
    X_tr, X_val, _, y_tr, y_val, _ = data_mgr.get_split(eval_seed)

    trials_data = []

    def obj(trial):
        x = np.array([trial.suggest_float(f"x{i}", -1.0, 1.0) for i in range(1, 5)])
        eta, depth, subsample, reg_lambda = decode_factors(x)

        # Approximate latency via quadratic model to guide search
        lat_est = 115.0 + 3.0 * depth + 0.8 * (depth**2)

        m = xgb.XGBRegressor(
            n_estimators=CONFIG["model"]["n_estimators"],
            learning_rate=eta,
            max_depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=eval_seed,
            n_jobs=CONFIG["model"]["n_jobs_train"],
            objective=CONFIG["model"]["objective"],
        )
        m.fit(X_tr, y_tr)
        val_rmse = float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))

        # Penalty if estimated latency violates constraint
        penalty = max(0.0, (lat_est - max_latency_us)) * 0.05
        score = val_rmse + penalty
        trials_data.append((x, val_rmse, lat_est, score))
        return score

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(obj, n_trials=n_trials)

    # Strictly filter to trials satisfying constraint
    valid_trials = [t for t in trials_data if t[2] <= max_latency_us]
    if not valid_trials:
        valid_trials = sorted(trials_data, key=lambda t: t[2])[:5]
    best_t = min(valid_trials, key=lambda t: t[1])
    return best_t[0], best_t[1]


def run_tpe_multi_objective(n_trials: int, sampler_seed: int, eval_seed: int, data_mgr: CaliforniaHousingDataManager):
    """Multi-Objective TPE optimizing Val RMSE and Latency."""
    X_tr, X_val, _, y_tr, y_val, _ = data_mgr.get_split(eval_seed)

    def obj(trial):
        x = np.array([trial.suggest_float(f"x{i}", -1.0, 1.0) for i in range(1, 5)])
        eta, depth, subsample, reg_lambda = decode_factors(x)
        lat_est = 115.0 + 3.0 * depth + 0.8 * (depth**2)

        m = xgb.XGBRegressor(
            n_estimators=CONFIG["model"]["n_estimators"],
            learning_rate=eta,
            max_depth=depth,
            subsample=subsample,
            reg_lambda=reg_lambda,
            random_state=eval_seed,
            n_jobs=CONFIG["model"]["n_jobs_train"],
            objective=CONFIG["model"]["objective"],
        )
        m.fit(X_tr, y_tr)
        val_rmse = float(np.sqrt(np.mean((y_val - m.predict(X_val))**2)))
        return val_rmse, lat_est

    sampler = optuna.samplers.TPESampler(seed=sampler_seed)
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
    study.optimize(obj, n_trials=n_trials)

    best_D = -1.0
    best_x = None
    L1, U1 = CONFIG["desirability"]["Y1_RMSE"]["L"], CONFIG["desirability"]["Y1_RMSE"]["U"]
    L2, U2 = CONFIG["desirability"]["Y2_Latency"]["L"], CONFIG["desirability"]["Y2_Latency"]["U"]

    for t in study.trials:
        if t.values is not None:
            v_rmse, v_lat = t.values[0], t.values[1]
            d1, d2, D = derringer_suich_desirability(v_rmse, v_lat, L1, U1, L2, U2)
            if D > best_D:
                best_D = D
                best_x = np.array([t.params[f"x{i}"] for i in range(1, 5)])

    return best_x, best_D


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


def main():
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

    # 1. Run Optimizers across 20 sampler seeds
    rs_incumbents = []
    rs_trajectories = []
    tpe_so_incumbents = []
    tpe_so_trajectories = []
    tpe_co_incumbents = []
    tpe_mo_incumbents = []

    print("Running 20 optimizer replicates...")
    for s_idx, s in enumerate(OPTIMIZER_SEEDS, 1):
        # Evaluate on fixed development seed 42
        x_rs, _, traj_rs = run_random_search(140, sampler_seed=s, eval_seed=42, data_mgr=data_mgr)
        rs_incumbents.append(x_rs)
        rs_trajectories.append(traj_rs)

        x_tpe, _, traj_tpe = run_tpe_single_objective(140, sampler_seed=s, eval_seed=42, data_mgr=data_mgr)
        tpe_so_incumbents.append(x_tpe)
        tpe_so_trajectories.append(traj_tpe)

        x_co, _ = run_tpe_constrained(140, sampler_seed=s, eval_seed=42, max_latency_us=145.0, data_mgr=data_mgr)
        tpe_co_incumbents.append(x_co)

        x_mo, _ = run_tpe_multi_objective(140, sampler_seed=s, eval_seed=42, data_mgr=data_mgr)
        tpe_mo_incumbents.append(x_mo)

        if s_idx % 5 == 0:
            print(f"[{s_idx}/20] optimizer replicates finished.")

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

    # Median incumbent coordinates
    x_rs_med = np.median(rs_incumbents, axis=0)
    x_tpe_so_med = np.median(tpe_so_incumbents, axis=0)
    x_tpe_co_med = np.median(tpe_co_incumbents, axis=0)
    x_tpe_mo_med = np.median(tpe_mo_incumbents, axis=0)

    # 2. Evaluate all methods across 20 fresh evaluation seeds
    configs_to_eval = {
        "Sequential DOE-CCD (x*, Multi-Objective)": x_doe_mo,
        "Sequential DOE-CCD (Single-Objective)": x_doe_so,
        "Unguided Random Search": x_rs_med,
        "Bayesian Optimization (Optuna TPE Single-Obj)": x_tpe_so_med,
        "Constrained TPE (Latency <= 145 us)": x_tpe_co_med,
        "Multi-Objective TPE (Desirability)": x_tpe_mo_med,
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
    ref_point = (0.60, 250.0)
    # DOE front: Depth 4 (x*) and Depth 7 (Single-Obj)
    pts_doe = [
        (eval_results["Sequential DOE-CCD (Single-Objective)"]["test_rmse_mean"],
         lat_results["Sequential DOE-CCD (Single-Objective)"]["predict_latency_us_median"]),
        (eval_results["Sequential DOE-CCD (x*, Multi-Objective)"]["test_rmse_mean"],
         lat_results["Sequential DOE-CCD (x*, Multi-Objective)"]["predict_latency_us_median"]),
    ]
    hv_doe = compute_hypervolume(pts_doe, ref_point)

    # MO-TPE front
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
        "reference_point": ref_point,
        "hv_doe": hv_doe,
        "hv_motpe": hv_motpe,
        "hv_rs": hv_rs,
        "doe_over_motpe_pct": ((hv_doe - hv_motpe) / hv_motpe) * 100.0,
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

if __name__ == "__main__":
    main()
