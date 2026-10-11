"""
scripts/reporting_data.py - Single source of truth for all reported scientific results.

Loads, validates, and cross-reconstructs every canonical experimental artifact from:
- config.yaml
- results/*.json and results/*.csv
- results/revision_v2/full_run_001/*.json and *.csv

Fails closed (raises ReportingDataError) if any required artifact, key, seed list,
reference point, selection count, or row-level statistical reconstruction is missing,
malformed, or inconsistent with saved summary JSONs.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import yaml
from statsmodels.formula.api import ols

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline import decode_factors
from scientific_stats import (
    hypervolume_2d_min,
    paired_difference_summary,
    pareto_front_2d_min,
)


class ReportingDataError(ValueError):
    """Raised when canonical experimental artifacts are missing, malformed, or inconsistent."""


EXPECTED_CONFIRMATION_SEEDS: Tuple[int, ...] = (
    505,
    606,
    707,
    808,
    909,
    1010,
    1111,
    1212,
    1313,
    1414,
)
EXPECTED_BLOCK_SEEDS: Tuple[int, ...] = (42, 101, 202, 303, 404)
EXPECTED_FRESH_EVAL_SEEDS: Tuple[int, ...] = tuple(range(2001, 2021))
EXPECTED_HV_REF_PRIMARY: Tuple[float, float] = (0.60, 250.0)
EXPECTED_HV_REF_SECONDARY: Tuple[float, float] = (0.65, 275.0)

PROSPECTIVE_OPTIMIZER_ORDER: Tuple[str, ...] = (
    "repeated_preplanned_doe_multi_objective",
    "multi_objective_tpe",
    "constrained_tpe",
    "repeated_preplanned_doe_single_objective",
    "single_objective_tpe",
    "random_search",
)

# Trusted LF-normalized SHA-256 baseline digests for config.yaml, all v1.0.0 results/*
# artifacts, and results/revision_v2/full_run_001/provenance.json (which itself locks all
# 18 full_run_001 artifact_sha256 digests).
CANONICAL_EVIDENCE_SHA256: Dict[str, str] = {
    "config.yaml": "72bbd8809d134e02222acca2feb3c59d982f3cc67bb8c0420faaec245a83687a",
    "results/anova_ccd_latency.csv": "94988845e974e9731fc1b15001f33b321ddf34da6c64f4b188aa013cdf7d1363",
    "results/benchmark.csv": "688293629c22378a1d8e196840db986ade0ecb90c71d7a877f299df3b719f978",
    "results/benchmark_evals_trajectories.csv": "6079327e67227f4cb9f52f5afed4227a88a9df9d112f9ab5a5af64783f2f0b83",
    "results/benchmark_optimizer_incumbents.csv": "0862ce271650d44a5503c66376abc8cbc2956bc1968202fdab79ff308d8cbecd",
    "results/benchmark_summary.json": "46684ccc22acbfa8c5c87b85d69a9108d9558d622791c600f199650d7a754c2a",
    "results/confirmation.json": "d8f6b88627a1cc6e3a925ad7af5e0dec6272a44c0df843a8982073ab48139b03",
    "results/confirmation_runs.csv": "ee148833ce2706f7d9c41278814242b1a4b10cca6e6ce3b6c15a99f3454545fd",
    "results/confirmation_runs_single_obj.csv": "11e81023aa3b1c3424e5f611bf9e53ddaf2b29d90055a024348e4b8b23c9c790",
    "results/depth_opt_table.csv": "3459057cf26140d7793d90c1874d8fd23331a5f96666a7aaba84cdad4b1b7e13",
    "results/desirability_sensitivity.csv": "2fbef32f610e08820c222f72b1a1a29168ef1cc5418f77da0152b92beb2dc15b",
    "results/diagnostics.json": "d3dd29d140af85ae117ad8d8c5860cefefc2c96db662e4c397fc62e4bd26a52b",
    "results/env.json": "77186264be0ae86523a81d6d6a674cfe16d58fcacc848a013ecec17bb59d6658",
    "results/icc.json": "39daff0651273fbcbe6c1d02906fadc8b5fbf6dc2ddc0f0a5730b665ed09430f",
    "results/latency_modeling.json": "b8bdd3d508604701d0525aee43d843f207cc850b5c8f700a291d15a8dd13f9f9",
    "results/latency_models_comparison.csv": "d2a3db8dfc389b7c496561ba706863e59074f0b647233b2208801c6a70833529",
    "results/lof.json": "126a39b0bc0771e9ebb116e49b954cbbae5cdcf35b6073942de62a893ff0d037",
    "results/phase1.json": "7434774b16f6d5ad953890cb2dfb34959fea6f3c049296a1f09784dee253df25",
    "results/phase3.json": "0a7d01b228c84d381721497d144120a4d3792205910c50e8f9c489ce8b3e303b",
    "results/ridge_table.csv": "478701c5f5a0e7a59955f544557eb88cf005435ba4172b22fa0ef6a121265e74",
    "results/runs.csv": "87486cb5691c92d3861379f6b45026ecbd1a20897e6bd57ddda5b423e26efeac",
    "results/revision_v2/full_run_001/provenance.json": "7aee3cc1523c2d5ec136fd6f9205eee92365031c7ef214ffe7d8544ab26c4b9b",
}


def _file_matches_sha256(path: Path, expected_hex: str) -> bool:
    if not path.is_file():
        return False
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return expected_hex in (
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(lf).hexdigest(),
        hashlib.sha256(crlf).hexdigest(),
    )


def verify_canonical_evidence_hashes(root_dir: Optional[Path | str] = None) -> int:
    """Verify cryptographic SHA-256 digests of all 40 canonical experimental evidence files.

    Checks both the trusted baseline SHA-256 table (CANONICAL_EVIDENCE_SHA256) and
    every artifact_sha256 / baseline_input_sha256 / runtime_source_sha256 entry in
    results/revision_v2/full_run_001/provenance.json. Raises ReportingDataError on any mismatch.
    Returns the total number of verified canonical evidence files.
    """
    base = Path(root_dir).resolve() if root_dir is not None else ROOT_DIR
    verified_count = 0

    for rel_path, expected_sha in CANONICAL_EVIDENCE_SHA256.items():
        target = base / rel_path
        if not _file_matches_sha256(target, expected_sha):
            raise ReportingDataError(
                f"Cryptographic evidence hash mismatch for {rel_path} (expected {expected_sha})"
            )
        verified_count += 1

    prov_path = base / "results" / "revision_v2" / "full_run_001" / "provenance.json"
    prov = _load_json(prov_path)
    _require_keys(
        prov,
        ["artifact_sha256", "baseline_input_sha256", "runtime_source_sha256"],
        "provenance.json",
    )

    artifact_hashes = prov["artifact_sha256"]
    if len(artifact_hashes) != 18:
        raise ReportingDataError(
            f"Expected 18 artifact_sha256 entries in provenance.json, got {len(artifact_hashes)}"
        )
    for name, expected_sha in artifact_hashes.items():
        target = base / "results" / "revision_v2" / "full_run_001" / name
        if not _file_matches_sha256(target, str(expected_sha)):
            raise ReportingDataError(
                f"Cryptographic evidence hash mismatch for results/revision_v2/full_run_001/{name} "
                f"against provenance.json (expected {expected_sha})"
            )
        verified_count += 1

    for key, expected_sha in prov["baseline_input_sha256"].items():
        rel = key.split(":", 1)[-1]
        if not _file_matches_sha256(base / rel, str(expected_sha)):
            raise ReportingDataError(
                f"Cryptographic baseline_input_sha256 mismatch for {rel} (expected {expected_sha})"
            )

    cfg_sha = prov["runtime_source_sha256"].get("config.yaml")
    if not cfg_sha or not _file_matches_sha256(base / "config.yaml", str(cfg_sha)):
        raise ReportingDataError(
            f"Cryptographic runtime_source_sha256 mismatch for config.yaml (expected {cfg_sha})"
        )

    return verified_count


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise ReportingDataError(f"Required canonical artifact file is missing: {path}")
    return path


def _load_json(path: Path) -> Dict[str, Any]:
    _require_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ReportingDataError(f"Failed to parse JSON artifact {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportingDataError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    _require_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ReportingDataError(f"Failed to parse JSON artifact {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ReportingDataError(f"Expected non-empty JSON list in {path}, got {type(data).__name__}")
    return data


def _load_yaml(path: Path) -> Dict[str, Any]:
    _require_file(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ReportingDataError(f"Failed to parse YAML artifact {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportingDataError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


def _load_csv(path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
    _require_file(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ReportingDataError(f"Failed to parse CSV artifact {path}: {exc}") from exc
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ReportingDataError(
            f"CSV artifact {path} is missing required columns: {missing}"
        )
    if df.empty:
        raise ReportingDataError(f"CSV artifact {path} is empty")
    return df


def _require_keys(mapping: Dict[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ReportingDataError(f"Missing required keys in {context}: {missing}")


def _assert_close(
    actual: float,
    expected: float,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    label: str,
) -> None:
    if not (np.isfinite(actual) and np.isfinite(expected)):
        raise ReportingDataError(
            f"Non-finite value encountered for {label}: actual={actual}, expected={expected}"
        )
    if not np.isclose(actual, expected, atol=atol, rtol=rtol):
        raise ReportingDataError(
            f"Numerical verification failed for {label}: reconstructed={actual:.10g}, "
            f"saved={expected:.10g} (diff={abs(actual - expected):.3e}, atol={atol}, rtol={rtol})"
        )


@dataclass(frozen=True)
class OptimizerBenchmarkMetrics:
    optimizer_key: str
    display_name_md: str
    display_name_tex: str
    search_basis_md: str
    search_basis_tex: str
    n_replicates: int
    val_rmse_mean: float
    val_rmse_sd: float
    val_rmse_ci_low: float
    val_rmse_ci_high: float
    test_rmse_mean: float
    test_rmse_sd: float
    test_rmse_ci_low: float
    test_rmse_ci_high: float
    retrain_sd: float
    predict_latency_mean: float
    predict_latency_sd: float
    predict_latency_ci_low: float
    predict_latency_ci_high: float
    inplace_latency_mean: float
    inplace_latency_sd: float
    search_feasible_count: int
    benchmark_feasible_count: int
    benchmark_feasible_pct: int


@dataclass(frozen=True)
class HistoricalBenchmarkRow:
    raw_method: str
    display_name_md: str
    display_name_tex: str
    search_basis_md: str
    search_basis_tex: str
    val_rmse_mean: float
    test_rmse_mean: float
    predict_latency_us_median: float
    inplace_latency_us_median: Optional[float]
    feasible_145: bool


_HOLD_BOOT_CACHE: Dict[Tuple[bytes, bytes], Dict[str, Any]] = {}


def _compute_holdout_pareto_bootstrap(
    final_evaluations_df: pd.DataFrame,
    final_summary_df: pd.DataFrame,
    latency_measurement: Dict[str, Any],
    hv_ref_primary: Tuple[float, float],
    hv_ref_secondary: Tuple[float, float],
) -> Dict[str, Any]:
    sids = final_summary_df["selection_id"].astype(str).tolist()
    opt_map = dict(zip(sids, final_summary_df["optimizer"].astype(str)))
    piv = final_evaluations_df.pivot(
        index="selection_id", columns="evaluation_seed", values="test_rmse"
    ).loc[sids]
    test_mat = piv.to_numpy(dtype=float)
    lats = final_summary_df["predict_latency_us"].to_numpy(dtype=float)
    lat_sess_mat = np.array(
        [
            [
                float(sess["summaries"][sid]["predict_latency_us"])
                for sess in latency_measurement["sessions"]
            ]
            for sid in sids
        ],
        dtype=float,
    )
    cache_key = (
        hashlib.sha256(test_mat.tobytes()).digest(),
        hashlib.sha256(lat_sess_mat.tobytes()).digest(),
    )
    if cache_key in _HOLD_BOOT_CACHE:
        return _HOLD_BOOT_CACHE[cache_key]

    rng = np.random.default_rng(20261011)
    n_boot = 2000
    hv_060_seed: List[float] = []
    hv_065_seed: List[float] = []
    hv_060_joint: List[float] = []
    hv_065_joint: List[float] = []
    opt_keys = list(PROSPECTIVE_OPTIMIZER_ORDER) + [
        "historical_preplanned_doe_desirability",
        "historical_preplanned_doe_single_objective",
    ]
    opt_any_counts = {opt: 0 for opt in opt_keys}
    opt_pt_counts: Dict[str, List[int]] = {opt: [] for opt in opt_keys}
    sel_pareto_counts = np.zeros(len(sids), dtype=int)

    for _ in range(n_boot):
        seed_idx = rng.integers(0, 20, size=20)
        boot_rmse = test_mat[:, seed_idx].mean(axis=1)
        pts_seed = list(zip(boot_rmse, lats))
        res_060 = hypervolume_2d_min(pts_seed, hv_ref_primary)
        res_065 = hypervolume_2d_min(pts_seed, hv_ref_secondary)
        hv_060_seed.append(res_060.value)
        hv_065_seed.append(res_065.value)

        pareto_set = set(res_060.pareto.points)
        opts_present = set()
        opt_cnt_this = {opt: 0 for opt in opt_keys}
        seen_pts = set()
        for idx_s, (r_v, l_v) in enumerate(pts_seed):
            if (r_v, l_v) in pareto_set:
                sel_pareto_counts[idx_s] += 1
                if (r_v, l_v) not in seen_pts:
                    seen_pts.add((r_v, l_v))
                    opt_k = opt_map[sids[idx_s]]
                    opt_cnt_this[opt_k] = opt_cnt_this.get(opt_k, 0) + 1
                    opts_present.add(opt_k)
        for opt_k in opts_present:
            opt_any_counts[opt_k] = opt_any_counts.get(opt_k, 0) + 1
        for opt_k in opt_keys:
            opt_pt_counts[opt_k].append(opt_cnt_this[opt_k])

        sess_idx = rng.integers(0, 5, size=5)
        boot_lat = lat_sess_mat[:, sess_idx].mean(axis=1)
        pts_joint = list(zip(boot_rmse, boot_lat))
        hv_060_joint.append(hypervolume_2d_min(pts_joint, hv_ref_primary).value)
        hv_065_joint.append(hypervolume_2d_min(pts_joint, hv_ref_secondary).value)

    arr_060_s = np.asarray(hv_060_seed, dtype=float)
    arr_065_s = np.asarray(hv_065_seed, dtype=float)
    arr_060_j = np.asarray(hv_060_joint, dtype=float)
    arr_065_j = np.asarray(hv_065_joint, dtype=float)

    arr_nd_s = np.asarray(
        [sum(opt_pt_counts[opt][i] for opt in opt_keys) for i in range(n_boot)],
        dtype=float,
    )
    by_opt = {}
    for opt_k in PROSPECTIVE_OPTIMIZER_ORDER:
        cnts = np.asarray(opt_pt_counts[opt_k], dtype=float)
        pct_val = float(opt_any_counts[opt_k] / n_boot * 100.0)
        mean_val = float(cnts.mean())
        by_opt[opt_k] = {
            "present_pct": pct_val,
            "seed_only_present_pct": pct_val,
            "mean_pts": mean_val,
            "seed_only_mean_points": mean_val,
            "sd_pts": float(cnts.std(ddof=1)),
            "min_pts": int(cnts.min()),
            "max_pts": int(cnts.max()),
        }

    s060_ci = [float(np.percentile(arr_060_s, 2.5)), float(np.percentile(arr_060_s, 97.5))]
    s065_ci = [float(np.percentile(arr_065_s, 2.5)), float(np.percentile(arr_065_s, 97.5))]
    j060_ci = [float(np.percentile(arr_060_j, 2.5)), float(np.percentile(arr_060_j, 97.5))]
    j065_ci = [float(np.percentile(arr_065_j, 2.5)), float(np.percentile(arr_065_j, 97.5))]

    result = {
        "n_boot": n_boot,
        "seed_only_nd_mean": float(arr_nd_s.mean()),
        "seed_only_nd_sd": float(arr_nd_s.std(ddof=1)),
        "seed_only_060_mean": float(arr_060_s.mean()),
        "seed_only_060_sd": float(arr_060_s.std(ddof=1)),
        "seed_only_060_ci_low": s060_ci[0],
        "seed_only_060_ci_high": s060_ci[1],
        "seed_only_060_ci95": s060_ci,
        "seed_only_065_mean": float(arr_065_s.mean()),
        "seed_only_065_sd": float(arr_065_s.std(ddof=1)),
        "seed_only_065_ci_low": s065_ci[0],
        "seed_only_065_ci_high": s065_ci[1],
        "seed_only_065_ci95": s065_ci,
        "joint_060_mean": float(arr_060_j.mean()),
        "joint_060_sd": float(arr_060_j.std(ddof=1)),
        "joint_060_ci_low": j060_ci[0],
        "joint_060_ci_high": j060_ci[1],
        "joint_060_ci95": j060_ci,
        "joint_065_mean": float(arr_065_j.mean()),
        "joint_065_sd": float(arr_065_j.std(ddof=1)),
        "joint_065_ci_low": j065_ci[0],
        "joint_065_ci_high": j065_ci[1],
        "joint_065_ci95": j065_ci,
        "by_optimizer": by_opt,
        "selections_ever_nondominated": int((sel_pareto_counts > 0).sum()),
        "selections_always_nondominated": int((sel_pareto_counts == n_boot).sum()),
    }
    _HOLD_BOOT_CACHE[cache_key] = result
    return result


@dataclass(frozen=True)
class ReportingDataset:
    root_dir: Path
    config: Dict[str, Any]
    env: Dict[str, Any]
    # Design & seeds
    block_seeds: Tuple[int, ...]
    confirmation_seeds: Tuple[int, ...]
    fresh_eval_seeds: Tuple[int, ...]
    hv_ref_primary: Tuple[float, float]
    hv_ref_secondary: Tuple[float, float]
    # Primary DOE JSONs & DataFrames
    runs_df: pd.DataFrame
    phase1: Dict[str, Any]
    phase3: Dict[str, Any]
    lof: Dict[str, Any]
    diagnostics: Dict[str, Any]
    icc: Dict[str, Any]
    confirmation: Dict[str, Any]
    confirmation_mo_df: pd.DataFrame
    confirmation_so_df: pd.DataFrame
    depth_opt_df: pd.DataFrame
    desirability_sens_df: pd.DataFrame
    latency_models_df: pd.DataFrame
    historical_benchmark_df: pd.DataFrame
    historical_benchmark_summary: Dict[str, Any]
    # Derived ANOVA & LoF decompositions
    anova_phase1_df: pd.DataFrame
    anova_ccd_y1_df: pd.DataFrame
    hc3_ccd_y1_se: pd.Series
    hc3_ccd_y1_p: pd.Series
    ccd_y1_rsq: float
    ccd_y1_adj_rsq: float
    anova_ccd_y2_df: pd.DataFrame
    lof_decomp: Dict[str, float]
    desirability_star: Dict[str, float]
    latency_jacobian_adjustment: float
    # Work Package F: DOE chronology, low-eta allocation & curvature sensitivity
    doe_axial_before_run100: int
    doe_low_eta_run_count: int
    doe_low_eta_run_pct: float
    doe_low_eta_mean_rmse: float
    doe_high_eta_mean_rmse: float
    curvature_sensitivity: Dict[str, float]
    # Work Package F: Historical interface overheads
    hist_runs_interface_overhead_mean: float
    hist_runs_interface_overhead_sd: float
    hist_runs_interface_overhead_min: float
    hist_runs_interface_overhead_max: float
    hist_conf_mo_interface_overhead_mean: float
    hist_conf_mo_interface_overhead_sd: float
    hist_conf_so_interface_overhead_mean: float
    hist_conf_so_interface_overhead_sd: float
    hist_bench_interface_overhead_mean: float
    hist_bench_interface_overhead_sd: float
    hist_bench_interface_overhead_min: float
    hist_bench_interface_overhead_max: float
    # Revision v2 Full Experiment Artifacts
    rev_dir: Path
    run_manifest: Dict[str, Any]
    provenance: Dict[str, Any]
    computational_budget: Dict[str, Any]
    finalized_selections: List[Dict[str, Any]]
    final_evaluations_df: pd.DataFrame
    final_summary_df: pd.DataFrame
    optimizer_summary: Dict[str, Any]
    doe_selection_summary: Dict[str, Any]
    latency_measurement: Dict[str, Any]
    latency_interface_overhead: Dict[str, Any]
    hypervolume: Dict[str, Any]
    paired_comparisons: Dict[str, Any]
    # Reconstructed Revision v2 summaries
    total_model_fits: int
    total_timed_inferences: int
    total_warmup_inferences: int
    total_all_inferences: int
    total_selection_records: int
    distinct_config_hashes: int
    total_evaluation_rows: int
    between_session_latency_sd_mean: float
    interface_overhead_mean: float
    interface_overhead_sd: float
    interface_overhead_min: float
    interface_overhead_max: float
    method_timing_summary: Dict[str, Dict[str, float]]
    prospective_optimizers: Dict[str, OptimizerBenchmarkMetrics]
    historical_benchmarks: Tuple[HistoricalBenchmarkRow, ...]
    ctpe_depth6_count: int
    ctpe_depth6_feasible: int
    ctpe_depth6_latency_mean: float
    ctpe_depth6_latency_sd: float
    ctpe_depth7_count: int
    ctpe_depth7_feasible: int
    ctpe_depth7_latency_mean: float
    ctpe_depth7_latency_sd: float
    doe_so_vs_sotpe_latency_reduction_pct: float
    doe_mo_vs_motpe_diff: float
    doe_mo_vs_motpe_welch_t: float
    doe_mo_vs_motpe_welch_p: float
    doe_mo_vs_motpe_paired_t: float
    doe_mo_vs_motpe_paired_p: float
    hv_summary: Dict[str, Any]
    holdout_nondominated_by_optimizer: Dict[str, int]
    holdout_pareto_bootstrap: Dict[str, Any]



def load_reporting_dataset(root_dir: Optional[Path | str] = None) -> ReportingDataset:
    """Load, validate, and reconstruct all canonical project results."""
    base = Path(root_dir).resolve() if root_dir is not None else ROOT_DIR
    results_dir = base / "results"
    rev_dir = results_dir / "revision_v2" / "full_run_001"

    # 1. Load and validate config.yaml
    config = _load_yaml(base / "config.yaml")
    _require_keys(
        config,
        [
            "project",
            "seeds",
            "factors",
            "model",
            "desirability",
            "budget",
            "revision_v2",
        ],
        "config.yaml",
    )
    seeds_cfg = config["seeds"]
    _require_keys(
        seeds_cfg,
        ["block_seeds", "confirmation_seeds", "fresh_eval_seeds"],
        "config.yaml:seeds",
    )
    block_seeds = tuple(int(s) for s in seeds_cfg["block_seeds"])
    if block_seeds != EXPECTED_BLOCK_SEEDS:
        raise ReportingDataError(
            f"config.yaml block_seeds {block_seeds} do not match expected {EXPECTED_BLOCK_SEEDS}"
        )

    confirmation_seeds = tuple(int(s) for s in seeds_cfg["confirmation_seeds"])
    if confirmation_seeds != EXPECTED_CONFIRMATION_SEEDS:
        raise ReportingDataError(
            f"config.yaml confirmation_seeds {confirmation_seeds} do not match expected {EXPECTED_CONFIRMATION_SEEDS}"
        )

    rev_cfg = config["revision_v2"]
    _require_keys(
        rev_cfg,
        [
            "max_latency_us",
            "hypervolume_reference_points",
            "modes",
        ],
        "config.yaml:revision_v2",
    )
    fresh_eval_seeds = tuple(int(s) for s in seeds_cfg["fresh_eval_seeds"])
    full_mode_eval_seeds = tuple(int(s) for s in rev_cfg["modes"]["full"]["evaluation_seeds"])
    if fresh_eval_seeds != EXPECTED_FRESH_EVAL_SEEDS or full_mode_eval_seeds != EXPECTED_FRESH_EVAL_SEEDS:
        raise ReportingDataError(
            f"config.yaml fresh_eval_seeds {fresh_eval_seeds} / full mode evaluation_seeds {full_mode_eval_seeds} do not match {EXPECTED_FRESH_EVAL_SEEDS}"
        )

    hv_refs_list = rev_cfg["hypervolume_reference_points"]
    if not isinstance(hv_refs_list, list) or len(hv_refs_list) < 2:
        raise ReportingDataError("config.yaml revision_v2.hypervolume_reference_points must contain 2 reference points")
    hv_ref_primary = (
        float(hv_refs_list[0][0]),
        float(hv_refs_list[0][1]),
    )
    hv_ref_secondary = (
        float(hv_refs_list[1][0]),
        float(hv_refs_list[1][1]),
    )
    if hv_ref_primary != EXPECTED_HV_REF_PRIMARY:
        raise ReportingDataError(
            f"Primary hypervolume reference point {hv_ref_primary} != expected {EXPECTED_HV_REF_PRIMARY}"
        )
    if hv_ref_secondary != EXPECTED_HV_REF_SECONDARY:
        raise ReportingDataError(
            f"Secondary hypervolume reference point {hv_ref_secondary} != expected {EXPECTED_HV_REF_SECONDARY}"
        )

    # 2. Load primary DOE artifacts
    env = _load_json(results_dir / "env.json")
    _require_keys(env, ["platform"], "results/env.json")

    phase1 = _load_json(results_dir / "phase1.json")
    _require_keys(
        phase1,
        [
            "yF_bar",
            "yC_bar",
            "diff_F_minus_C",
            "ss_curvature",
            "ms_pe_center",
            "df_pe_center",
            "f_curvature",
        ],
        "results/phase1.json",
    )

    phase3 = _load_json(results_dir / "phase3.json")
    _require_keys(
        phase3,
        [
            "b0",
            "b",
            "B",
            "trace_B",
            "sum_eigenvalues",
            "eigenvalues",
            "fraction_min_eigenvalue_le_zero",
            "stationary_point_coded",
            "y0_hat_unconstrained",
            "distance_coded",
            "stationary_natural_clamped",
            "constrained_optimum_cube",
            "bootstrap_percentiles",
        ],
        "results/phase3.json",
    )

    lof = _load_json(results_dir / "lof.json")
    _require_keys(lof, ["Y1", "Y1_restricted", "Y2"], "results/lof.json")

    diagnostics = _load_json(results_dir / "diagnostics.json")
    _require_keys(
        diagnostics,
        [
            "shapiro_W",
            "shapiro_p",
            "levene_stat",
            "levene_p",
            "brown_forsythe_points_stat",
            "brown_forsythe_points_p",
            "breusch_pagan_stat",
            "breusch_pagan_p",
            "max_cooks_d",
            "thresh_4n",
            "outlier_count",
            "durbin_watson",
            "ljung_box_stat",
            "ljung_box_p",
            "runs_test_p",
        ],
        "results/diagnostics.json",
    )

    icc = _load_json(results_dir / "icc.json")
    _require_keys(icc, ["Y1", "Y2"], "results/icc.json")

    confirmation = _load_json(results_dir / "confirmation.json")
    _require_keys(
        confirmation,
        [
            "x_star_coded",
            "x_star_natural",
            "leverage_h",
            "Y1_Val_RMSE",
            "Y1_Test_RMSE",
            "Y2_Latency",
            "Single_Objective_Optimum",
        ],
        "results/confirmation.json",
    )

    runs_df = _load_csv(
        results_dir / "runs.csv",
        ["block", "phase", "point_id", "x1", "x2", "x3", "x4", "val_rmse", "latency_us_median"],
    )
    if len(runs_df) != 140:
        raise ReportingDataError(f"Expected 140 rows in results/runs.csv, got {len(runs_df)}")

    confirmation_mo_df = _load_csv(
        results_dir / "confirmation_runs.csv",
        ["seed", "val_rmse", "test_rmse", "latency_us_median"],
    )
    confirmation_so_df = _load_csv(
        results_dir / "confirmation_runs_single_obj.csv",
        ["seed", "val_rmse", "test_rmse", "latency_us_median"],
    )

    # Validate confirmation seeds in both CSVs match config.yaml exactly
    mo_seeds = tuple(int(s) for s in confirmation_mo_df["seed"].tolist())
    so_seeds = tuple(int(s) for s in confirmation_so_df["seed"].tolist())
    if mo_seeds != confirmation_seeds:
        raise ReportingDataError(
            f"results/confirmation_runs.csv seeds {mo_seeds} do not match config.yaml {confirmation_seeds}"
        )
    if so_seeds != confirmation_seeds:
        raise ReportingDataError(
            f"results/confirmation_runs_single_obj.csv seeds {so_seeds} do not match config.yaml {confirmation_seeds}"
        )

    # Row-level reconstruction of confirmation statistics
    _assert_close(
        float(confirmation_mo_df["val_rmse"].mean()),
        float(confirmation["Y1_Val_RMSE"]["empirical_mean"]),
        atol=1e-6,
        label="Confirmation MO val_rmse mean",
    )
    _assert_close(
        float(confirmation_mo_df["val_rmse"].std(ddof=1)),
        float(confirmation["Y1_Val_RMSE"]["empirical_std"]),
        atol=1e-6,
        label="Confirmation MO val_rmse std",
    )
    _assert_close(
        float(confirmation_mo_df["test_rmse"].mean()),
        float(confirmation["Y1_Test_RMSE"]["empirical_mean"]),
        atol=1e-6,
        label="Confirmation MO test_rmse mean",
    )
    _assert_close(
        float(confirmation_mo_df["test_rmse"].std(ddof=1)),
        float(confirmation["Y1_Test_RMSE"]["empirical_std"]),
        atol=1e-6,
        label="Confirmation MO test_rmse std",
    )
    _assert_close(
        float(confirmation_mo_df["latency_us_median"].mean()),
        float(confirmation["Y2_Latency"]["empirical_mean"]),
        atol=1e-5,
        label="Confirmation MO latency mean",
    )
    _assert_close(
        float(confirmation_mo_df["latency_us_median"].std(ddof=1)),
        float(confirmation["Y2_Latency"]["empirical_std"]),
        atol=1e-5,
        label="Confirmation MO latency std",
    )

    so_conf = confirmation["Single_Objective_Optimum"]
    _assert_close(
        float(confirmation_so_df["val_rmse"].mean()),
        float(so_conf["empirical_val_rmse"]),
        atol=1e-6,
        label="Confirmation SO val_rmse mean",
    )
    _assert_close(
        float(confirmation_so_df["val_rmse"].std(ddof=1)),
        float(so_conf["empirical_val_std"]),
        atol=1e-6,
        label="Confirmation SO val_rmse std",
    )
    _assert_close(
        float(confirmation_so_df["test_rmse"].mean()),
        float(so_conf["empirical_test_rmse"]),
        atol=1e-6,
        label="Confirmation SO test_rmse mean",
    )
    _assert_close(
        float(confirmation_so_df["test_rmse"].std(ddof=1)),
        float(so_conf["empirical_test_std"]),
        atol=1e-6,
        label="Confirmation SO test_rmse std",
    )
    _assert_close(
        float(confirmation_so_df["latency_us_median"].mean()),
        float(so_conf["empirical_latency"]),
        atol=1e-5,
        label="Confirmation SO latency mean",
    )
    _assert_close(
        float(confirmation_so_df["latency_us_median"].std(ddof=1)),
        float(so_conf["empirical_latency_std"]),
        atol=1e-5,
        label="Confirmation SO latency std",
    )

    depth_opt_df = _load_csv(
        results_dir / "depth_opt_table.csv",
        ["depth", "x1", "x3", "x4", "pred_rmse", "se"],
    )
    desirability_sens_df = _load_csv(
        results_dir / "desirability_sensitivity.csv",
        ["Scenario", "L1", "U1", "L2", "U2", "w1", "w2", "Optimal_Depth", "Optimal_Eta", "Best_D"],
    )
    latency_models_df = _load_csv(
        results_dir / "latency_models_comparison.csv",
        ["Model", "R2", "Adj_R2", "AIC", "BIC", "RMSE"],
    )
    historical_benchmark_df = _load_csv(
        results_dir / "benchmark.csv",
        ["method", "val_rmse_mean", "test_rmse_mean", "predict_latency_us_median", "inplace_latency_us_median"],
    )
    historical_benchmark_summary = _load_json(results_dir / "benchmark_summary.json")

    # 3. Row-level reconstruction of Phase 1 Curvature & ANOVA and Phase 2 CCD ANOVA & LoF
    df_p1_fact = runs_df[runs_df["phase"] == "Phase1_Factorial"]
    df_p1_cent = runs_df[runs_df["phase"] == "Phase1_Center"]
    if len(df_p1_fact) != 80 or len(df_p1_cent) != 20:
        raise ReportingDataError(
            f"Unexpected Phase 1 run counts: n_F={len(df_p1_fact)}, n_C={len(df_p1_cent)}"
        )
    yF_bar = float(df_p1_fact["val_rmse"].mean())
    yC_bar = float(df_p1_cent["val_rmse"].mean())
    diff_fc = yF_bar - yC_bar
    ss_curv = float((80 * 20 / 100.0) * (diff_fc**2))
    ss_pe_center_p1 = float(
        sum(
            np.sum((grp["val_rmse"] - grp["val_rmse"].mean()) ** 2)
            for _, grp in df_p1_cent.groupby("block")
        )
    )
    df_pe_center_p1 = int(sum(len(grp) - 1 for _, grp in df_p1_cent.groupby("block")))
    ms_pe_center_p1 = ss_pe_center_p1 / df_pe_center_p1
    f_curv_p1 = ss_curv / ms_pe_center_p1

    _assert_close(yF_bar, float(phase1["yF_bar"]), atol=1e-6, label="Phase 1 yF_bar")
    _assert_close(yC_bar, float(phase1["yC_bar"]), atol=1e-6, label="Phase 1 yC_bar")
    _assert_close(diff_fc, float(phase1["diff_F_minus_C"]), atol=1e-6, label="Phase 1 diff_F_minus_C")
    _assert_close(ss_curv, float(phase1["ss_curvature"]), atol=1e-6, label="Phase 1 ss_curvature")
    _assert_close(ms_pe_center_p1, float(phase1["ms_pe_center"]), atol=1e-9, label="Phase 1 ms_pe_center")
    _assert_close(f_curv_p1, float(phase1["f_curvature"]), atol=1e-2, label="Phase 1 f_curvature")

    df_p1 = runs_df[runs_df["phase"].isin(["Phase1_Factorial", "Phase1_Center"])].copy()
    fit_p1 = ols("val_rmse ~ C(block) + x1+x2+x3+x4 + x1:x2+x1:x3+x1:x4+x2:x3+x2:x4+x3:x4", df_p1).fit()
    anova_phase1_df = sm.stats.anova_lm(fit_p1, typ=3)

    df_aug = runs_df.copy()
    q_cols = ["x1", "x2", "x3", "x4"]
    for q in q_cols:
        df_aug[q + "_sq"] = df_aug[q] ** 2
    for i in range(4):
        for j in range(i + 1, 4):
            df_aug[f"{q_cols[i]}_{q_cols[j]}"] = df_aug[q_cols[i]] * df_aug[q_cols[j]]

    fit_ccd_y1 = ols(
        "val_rmse ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
        "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4",
        df_aug,
    ).fit()
    fit_ccd_y1_hc3 = fit_ccd_y1.get_robustcov_results("HC3")
    hc3_ccd_y1_se = pd.Series(fit_ccd_y1_hc3.bse, index=fit_ccd_y1_hc3.model.exog_names)
    hc3_ccd_y1_p = pd.Series(fit_ccd_y1_hc3.pvalues, index=fit_ccd_y1_hc3.model.exog_names)
    anova_ccd_y1_df = sm.stats.anova_lm(fit_ccd_y1, typ=3)
    ccd_y1_rsq = float(fit_ccd_y1.rsquared)
    ccd_y1_adj_rsq = float(fit_ccd_y1.rsquared_adj)

    # Verify Block F-statistic against icc.json
    _assert_close(
        float(anova_ccd_y1_df.loc["C(block)", "F"]),
        float(icc["Y1"]["f_block"]),
        atol=1e-4,
        label="Phase 2 CCD Block F-statistic",
    )

    fit_ccd_y2 = ols(
        "latency_us_median ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq + "
        "x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4",
        df_aug,
    ).fit()
    anova_ccd_y2_df = sm.stats.anova_lm(fit_ccd_y2, typ=3)

    # Replicate and interaction variance components for LoF
    ss_pure_center_y1 = float(
        np.sum(
            [
                np.sum((grp["val_rmse"] - grp["val_rmse"].mean()) ** 2)
                for (_, _), grp in runs_df.groupby(["block", "point_id"])
                if len(grp) > 1
            ]
        )
    )
    df_pure_center = int(
        sum(len(grp) - 1 for (_, _), grp in runs_df.groupby(["block", "point_id"]) if len(grp) > 1)
    )
    ms_pure_center_y1 = ss_pure_center_y1 / df_pure_center
    ss_interact_y1 = float(lof["Y1"]["SS_PE"] - ss_pure_center_y1)
    df_interact = int(lof["Y1"]["df_PE"] - df_pure_center)
    ms_interact_y1 = ss_interact_y1 / df_interact
    f_lof_center_y1 = float(lof["Y1"]["MS_LoF"] / ms_pure_center_y1)
    p_lof_center_y1 = float(1.0 - stats.f.cdf(f_lof_center_y1, lof["Y1"]["df_LoF"], df_pure_center))
    f_interact_center_y1 = float(ms_interact_y1 / ms_pure_center_y1)
    p_interact_center_y1 = float(1.0 - stats.f.cdf(f_interact_center_y1, df_interact, df_pure_center))

    ss_pure_center_y2 = float(
        np.sum(
            [
                np.sum((grp["latency_us_median"] - grp["latency_us_median"].mean()) ** 2)
                for (_, _), grp in runs_df.groupby(["block", "point_id"])
                if len(grp) > 1
            ]
        )
    )
    ms_pure_center_y2 = ss_pure_center_y2 / df_pure_center
    ss_interact_y2 = float(lof["Y2"]["SS_PE"] - ss_pure_center_y2)
    ms_interact_y2 = ss_interact_y2 / df_interact
    f_lof_center_y2 = float(lof["Y2"]["MS_LoF"] / ms_pure_center_y2)
    p_lof_center_y2 = float(1.0 - stats.f.cdf(f_lof_center_y2, lof["Y2"]["df_LoF"], df_pure_center))
    f_interact_center_y2 = float(ms_interact_y2 / ms_pure_center_y2)
    p_interact_center_y2 = float(1.0 - stats.f.cdf(f_interact_center_y2, df_interact, df_pure_center))

    lof_decomp = {
        "df_pure_center": float(df_pure_center),
        "df_interact": float(df_interact),
        "ss_pure_center_y1": ss_pure_center_y1,
        "ms_pure_center_y1": ms_pure_center_y1,
        "ss_interact_y1": ss_interact_y1,
        "ms_interact_y1": ms_interact_y1,
        "f_lof_center_y1": f_lof_center_y1,
        "p_lof_center_y1": p_lof_center_y1,
        "f_interact_center_y1": f_interact_center_y1,
        "p_interact_center_y1": p_interact_center_y1,
        "ss_pure_center_y2": ss_pure_center_y2,
        "ms_pure_center_y2": ms_pure_center_y2,
        "ss_interact_y2": ss_interact_y2,
        "ms_interact_y2": ms_interact_y2,
        "f_lof_center_y2": f_lof_center_y2,
        "p_lof_center_y2": p_lof_center_y2,
        "f_interact_center_y2": f_interact_center_y2,
        "p_interact_center_y2": p_interact_center_y2,
    }

    # Desirability at x*_MO
    pred_y1_star = float(confirmation["Y1_Val_RMSE"]["predicted_mean"])
    pred_y2_star = float(confirmation["Y2_Latency"]["predicted_mean"])
    d1_star = max(0.0, min(1.0, (0.700 - pred_y1_star) / (0.700 - 0.450)))
    d2_star = max(0.0, min(1.0, (180.0 - pred_y2_star) / (180.0 - 100.0)))
    D_star = float(np.sqrt(d1_star * d2_star))
    desirability_star = {
        "pred_y1": pred_y1_star,
        "pred_y2": pred_y2_star,
        "d1": d1_star,
        "d2": d2_star,
        "D": D_star,
    }
    latency_jacobian_adjustment = float(2.0 * np.sum(np.log(runs_df["latency_us_median"])))

    # Work Package F: DOE chronology & low-eta allocation
    doe_axial_before_run100 = int(
        (runs_df[runs_df["phase"] == "Phase2_Axial"]["run_order"] <= 100).sum()
    )
    low_eta_mask = runs_df["x1"] == -1.0
    doe_low_eta_run_count = int(low_eta_mask.sum())
    doe_low_eta_run_pct = float(doe_low_eta_run_count / len(runs_df) * 100.0)
    doe_low_eta_mean_rmse = float(runs_df.loc[low_eta_mask, "val_rmse"].mean())
    doe_high_eta_mean_rmse = float(runs_df.loc[~low_eta_mask, "val_rmse"].mean())

    # Work Package F: Curvature sensitivity analyses across alternative error structures
    fit_p1_cell = ols("val_rmse ~ C(block) + C(point_id)", data=df_p1).fit()
    ss_block_adj_pooled = float(fit_p1_cell.ssr)
    df_block_adj_pooled = int(fit_p1_cell.df_resid)  # 79 df = 64 df TxB + 15 df center PE
    ms_block_adj_pooled = ss_block_adj_pooled / df_block_adj_pooled
    f_block_adj_pooled = ss_curv / ms_block_adj_pooled

    ss_txb_p1 = ss_block_adj_pooled - ss_pe_center_p1
    df_txb_p1 = df_block_adj_pooled - df_pe_center_p1  # 64 df
    ms_txb_p1 = ss_txb_p1 / df_txb_p1
    f_txb_p1 = ss_curv / ms_txb_p1

    blk_diffs = [
        float(
            df_p1_fact[df_p1_fact["block"] == b]["val_rmse"].mean()
            - df_p1_cent[df_p1_cent["block"] == b]["val_rmse"].mean()
        )
        for b in sorted(df_p1["block"].unique())
    ]
    df_block_paired = len(blk_diffs) - 1  # 4 df
    se_block_paired = float(np.std(blk_diffs, ddof=1) / np.sqrt(len(blk_diffs)))
    t_block_paired = float(diff_fc / se_block_paired)
    f_block_paired = float(t_block_paired**2)
    p_block_paired = float(2.0 * (1.0 - stats.t.cdf(abs(t_block_paired), df_block_paired)))
    t_crit_blk = float(stats.t.ppf(0.975, df_block_paired))
    ci95_block_paired_low = float(diff_fc - t_crit_blk * se_block_paired)
    ci95_block_paired_high = float(diff_fc + t_crit_blk * se_block_paired)

    ss_pe_coord_unadj = float(
        df_p1.groupby("point_id")["val_rmse"]
        .apply(lambda s: ((s - s.mean()) ** 2).sum())
        .sum()
    )
    df_pe_coord_unadj = int(len(df_p1) - df_p1["point_id"].nunique())  # 83 df
    ms_pe_coord_unadj = ss_pe_coord_unadj / df_pe_coord_unadj
    f_pe_coord_unadj = ss_curv / ms_pe_coord_unadj

    df_p1_ind = df_p1.copy()
    df_p1_ind["is_center"] = (df_p1_ind["phase"] == "Phase1_Center").astype(float)
    fit_p1_2fi_curv = ols(
        "val_rmse ~ C(block) + is_center + x1+x2+x3+x4 + x1:x2+x1:x3+x1:x4+x2:x3+x2:x4+x3:x4",
        data=df_p1_ind,
    ).fit()
    df_2fi_curv_resid = int(fit_p1_2fi_curv.df_resid)  # 84 df
    ms_2fi_curv_resid = float(fit_p1_2fi_curv.ssr / df_2fi_curv_resid)
    f_2fi_curv_resid = ss_curv / ms_2fi_curv_resid

    df_2fi_resid = int(fit_p1.df_resid)  # 85 df
    ms_2fi_resid = float(fit_p1.ssr / df_2fi_resid)
    f_2fi_resid = ss_curv / ms_2fi_resid
    p_2fi_resid = float(1.0 - stats.f.cdf(f_2fi_resid, 1, df_2fi_resid))

    curvature_sensitivity = {
        "diff_rmse": diff_fc,
        "pct_increase_over_center": float(diff_fc / yC_bar * 100.0),
        "ss_curv": ss_curv,
        "df_pe_center": float(df_pe_center_p1),
        "ms_pe_center": ms_pe_center_p1,
        "f_pe_center": f_curv_p1,
        "df_block_adj_pooled": float(df_block_adj_pooled),
        "ss_block_adj_pooled": ss_block_adj_pooled,
        "ms_block_adj_pooled": ms_block_adj_pooled,
        "f_block_adj_pooled": f_block_adj_pooled,
        "df_txb_p1": float(df_txb_p1),
        "ss_txb_p1": ss_txb_p1,
        "ms_txb_p1": ms_txb_p1,
        "f_txb_p1": f_txb_p1,
        "df_block_paired": float(df_block_paired),
        "se_block_paired": se_block_paired,
        "t_block_paired": t_block_paired,
        "f_block_paired": f_block_paired,
        "p_block_paired": p_block_paired,
        "ci95_block_paired_low": ci95_block_paired_low,
        "ci95_block_paired_high": ci95_block_paired_high,
        "ci95_block_paired": [ci95_block_paired_low, ci95_block_paired_high],
        "df_pe_coord_unadj": float(df_pe_coord_unadj),
        "ss_pe_coord_unadj": ss_pe_coord_unadj,
        "ms_pe_coord_unadj": ms_pe_coord_unadj,
        "f_pe_coord_unadj": f_pe_coord_unadj,
        "df_2fi_curv_resid": float(df_2fi_curv_resid),
        "ss_2fi_curv_resid": float(fit_p1_2fi_curv.ssr),
        "ms_2fi_curv_resid": ms_2fi_curv_resid,
        "f_2fi_curv_resid": f_2fi_curv_resid,
        "df_2fi_resid": float(df_2fi_resid),
        "ms_2fi_resid": ms_2fi_resid,
        "f_2fi_resid": f_2fi_resid,
        "p_2fi_resid": p_2fi_resid,
    }

    # Work Package F: Historical predict minus inplace_predict interface overheads
    runs_ov = runs_df["latency_us_median"] - runs_df["inplace_latency_us_median"]
    hist_runs_interface_overhead_mean = float(runs_ov.mean())
    hist_runs_interface_overhead_sd = float(runs_ov.std(ddof=1))
    hist_runs_interface_overhead_min = float(runs_ov.min())
    hist_runs_interface_overhead_max = float(runs_ov.max())

    conf_mo_ov = (
        confirmation_mo_df["latency_us_median"] - confirmation_mo_df["inplace_latency_us_median"]
    )
    hist_conf_mo_interface_overhead_mean = float(conf_mo_ov.mean())
    hist_conf_mo_interface_overhead_sd = float(conf_mo_ov.std(ddof=1))

    conf_so_ov = (
        confirmation_so_df["latency_us_median"] - confirmation_so_df["inplace_latency_us_median"]
    )
    hist_conf_so_interface_overhead_mean = float(conf_so_ov.mean())
    hist_conf_so_interface_overhead_sd = float(conf_so_ov.std(ddof=1))

    bench_ov = (
        historical_benchmark_df["predict_latency_us_median"]
        - historical_benchmark_df["inplace_latency_us_median"]
    )
    hist_bench_interface_overhead_mean = float(bench_ov.mean())
    hist_bench_interface_overhead_sd = float(bench_ov.std(ddof=1))
    hist_bench_interface_overhead_min = float(bench_ov.min())
    hist_bench_interface_overhead_max = float(bench_ov.max())

    # 4. Load and reconstruct Revision-v2 Full Benchmark Campaign artifacts
    run_manifest = _load_json(rev_dir / "run_manifest.json")
    provenance = _load_json(rev_dir / "provenance.json")
    computational_budget = _load_json(rev_dir / "computational_budget.json")
    finalized_selections_obj = _load_json(rev_dir / "finalized_selections.json")
    _require_keys(finalized_selections_obj, ["configurations"], "finalized_selections.json")
    finalized_selections = finalized_selections_obj["configurations"]
    if not isinstance(finalized_selections, list) or len(finalized_selections) != 122:
        raise ReportingDataError(
            f"Expected 122 selection records in finalized_selections.json, got {len(finalized_selections) if isinstance(finalized_selections, list) else type(finalized_selections)}"
        )

    total_selection_records = len(finalized_selections)
    distinct_config_hashes = len({str(x["config_sha256"]) for x in finalized_selections})
    if distinct_config_hashes != 95:
        raise ReportingDataError(
            f"Expected 95 distinct config_sha256 hashes in finalized_selections.json, got {distinct_config_hashes}"
        )

    final_evaluations_df = _load_csv(
        rev_dir / "final_evaluations.csv",
        [
            "selection_id",
            "optimizer",
            "optimizer_replicate_id",
            "config_sha256",
            "evaluation_seed",
            "status",
            "val_rmse",
            "test_rmse",
        ],
    )
    total_evaluation_rows = len(final_evaluations_df)
    if total_evaluation_rows != 2440:
        raise ReportingDataError(
            f"Expected 2440 rows in final_evaluations.csv, got {total_evaluation_rows}"
        )
    if not (final_evaluations_df["status"] == "completed").all():
        raise ReportingDataError("Not all rows in final_evaluations.csv have status 'completed'")

    # Verify evaluation seeds in final_evaluations.csv match fresh_eval_seeds
    eval_seeds_actual = tuple(sorted(int(s) for s in final_evaluations_df["evaluation_seed"].unique()))
    if eval_seeds_actual != fresh_eval_seeds:
        raise ReportingDataError(
            f"final_evaluations.csv seeds {eval_seeds_actual} != config.yaml {fresh_eval_seeds}"
        )

    final_summary_df = _load_csv(
        rev_dir / "final_summary.csv",
        [
            "selection_id",
            "optimizer",
            "optimizer_replicate_id",
            "validation_rmse",
            "test_rmse",
            "predict_latency_us",
            "inplace_predict_latency_us",
            "predict_latency_session_standard_deviation_us",
            "search_time_feasible",
            "benchmark_time_feasible",
        ],
    ).copy()
    if len(final_summary_df) != 122:
        raise ReportingDataError(f"Expected 122 rows in final_summary.csv, got {len(final_summary_df)}")

    for col in ["validation_rmse", "test_rmse"]:
        final_summary_df[col + "_mean"] = final_summary_df[col].apply(
            lambda x: float(ast.literal_eval(x)["mean"])
        )
        final_summary_df[col + "_sd"] = final_summary_df[col].apply(
            lambda x: float(ast.literal_eval(x)["standard_deviation"])
        )

    sel_depth_map = {str(x["selection_id"]): int(x["hyperparameters"]["max_depth"]) for x in finalized_selections}
    final_summary_df["max_depth"] = final_summary_df["selection_id"].map(sel_depth_map)

    # Row-level reconstruction check: final_evaluations.csv -> final_summary.csv per selection_id
    eval_grouped = final_evaluations_df.groupby("selection_id")
    for _, srow in final_summary_df.iterrows():
        sid = str(srow["selection_id"])
        grp = eval_grouped.get_group(sid)
        if len(grp) != 20:
            raise ReportingDataError(f"Selection {sid} has {len(grp)} eval rows, expected 20")
        _assert_close(
            float(grp["val_rmse"].mean()),
            float(srow["validation_rmse_mean"]),
            atol=1e-9,
            label=f"{sid} validation_rmse_mean",
        )
        _assert_close(
            float(grp["val_rmse"].std(ddof=1)),
            float(srow["validation_rmse_sd"]),
            atol=1e-9,
            label=f"{sid} validation_rmse_sd",
        )
        _assert_close(
            float(grp["test_rmse"].mean()),
            float(srow["test_rmse_mean"]),
            atol=1e-9,
            label=f"{sid} test_rmse_mean",
        )
        _assert_close(
            float(grp["test_rmse"].std(ddof=1)),
            float(srow["test_rmse_sd"]),
            atol=1e-9,
            label=f"{sid} test_rmse_sd",
        )

    optimizer_summary = _load_json(rev_dir / "optimizer_summary.json")
    doe_selection_summary = _load_json(rev_dir / "doe_selection_summary.json")
    latency_measurement = _load_json(rev_dir / "latency_measurement.json")
    latency_interface_overhead = _load_json(rev_dir / "latency_interface_overhead.json")
    hypervolume = _load_json(rev_dir / "hypervolume.json")
    paired_comparisons = _load_json_list(rev_dir / "paired_comparisons.json")

    # Reconstruct latency measurements from latency_measurement.json (5 sessions x 122 selections = 610)
    _require_keys(latency_measurement, ["summaries", "sessions"], "latency_measurement.json")
    if len(latency_measurement["sessions"]) != 5:
        raise ReportingDataError(
            f"Expected 5 sessions in latency_measurement.json, got {len(latency_measurement['sessions'])}"
        )
    lat_sum_by_sel = latency_measurement["summaries"]
    if len(lat_sum_by_sel) != 122:
        raise ReportingDataError(
            f"Expected 122 selection summaries in latency_measurement.json, got {len(lat_sum_by_sel)}"
        )
    total_latency_session_models = len(latency_measurement["sessions"]) * len(lat_sum_by_sel)
    for _, srow in final_summary_df.iterrows():
        sid = str(srow["selection_id"])
        if sid not in lat_sum_by_sel:
            raise ReportingDataError(f"Selection {sid} missing from latency_measurement.json")
        lsel = lat_sum_by_sel[sid]
        _assert_close(
            float(lsel["predict_latency_us"]),
            float(srow["predict_latency_us"]),
            atol=1e-9,
            label=f"{sid} predict_latency_us",
        )
        _assert_close(
            float(lsel["inplace_predict_latency_us"]),
            float(srow["inplace_predict_latency_us"]),
            atol=1e-9,
            label=f"{sid} inplace_predict_latency_us",
        )

    between_session_latency_sd_mean = float(
        final_summary_df["predict_latency_session_standard_deviation_us"].mean()
    )
    per_session_overheads = pd.Series(
        [float(x["overhead_us"]) for x in latency_interface_overhead["per_session"]]
    )
    if len(per_session_overheads) != 610:
        raise ReportingDataError(
            f"Expected 610 per_session records in latency_interface_overhead.json, got {len(per_session_overheads)}"
        )
    interface_overhead_mean = float(per_session_overheads.mean())
    interface_overhead_sd = float(per_session_overheads.std(ddof=1))
    interface_overhead_min = float(per_session_overheads.min())
    interface_overhead_max = float(per_session_overheads.max())
    _assert_close(
        interface_overhead_mean,
        float(latency_interface_overhead["pooled_descriptive_summary"]["mean"]),
        atol=1e-9,
        label="Latency interface overhead mean",
    )
    _assert_close(
        interface_overhead_sd,
        float(latency_interface_overhead["pooled_descriptive_summary"]["standard_deviation"]),
        atol=1e-9,
        label="Latency interface overhead SD",
    )

    # Reconstruct computational budget from row-level files
    optimizer_trials_df = _load_csv(
        rev_dir / "optimizer_trials.csv",
        ["optimizer", "replicate_id", "validation_rmse", "predict_latency_us", "trial_status", "training_time_s"],
    )
    doe_selection_runs_df = _load_csv(
        rev_dir / "doe_selection_runs.csv",
        ["doe_replicate_id", "point_id", "validation_rmse", "predict_latency_us", "trial_status", "training_time_s"],
    )
    doe_matched_front_df = _load_csv(
        rev_dir / "doe_matched_candidate_front.csv",
        ["candidate_id", "candidate_type", "validation_rmse", "predict_latency_us"],
    )
    optimizer_replicates_df = _load_csv(
        rev_dir / "optimizer_replicates.csv",
        ["optimizer", "replicate_id", "method_wall_time_s"],
    )
    doe_selection_replicates_df = _load_csv(
        rev_dir / "doe_selection_replicates.csv",
        ["doe_replicate_id", "replicate_wall_time_s"],
    )
    if len(optimizer_trials_df) != 11200:
        raise ReportingDataError(f"Expected 11200 rows in optimizer_trials.csv, got {len(optimizer_trials_df)}")
    if len(doe_selection_runs_df) != 2800:
        raise ReportingDataError(f"Expected 2800 rows in doe_selection_runs.csv, got {len(doe_selection_runs_df)}")
    if len(doe_matched_front_df) != 27:
        raise ReportingDataError(f"Expected 27 rows in doe_matched_candidate_front.csv, got {len(doe_matched_front_df)}")

    reconstructed_total_fits = (
        len(optimizer_trials_df)
        + len(doe_selection_runs_df)
        + len(doe_matched_front_df)
        + len(final_evaluations_df)
        + total_latency_session_models
    )
    total_model_fits = int(computational_budget["maximum_total_model_fits"])
    if reconstructed_total_fits != total_model_fits:
        raise ReportingDataError(
            f"Reconstructed total model fits ({reconstructed_total_fits}) != computational_budget.json ({total_model_fits})"
        )
    total_timed_inferences = int(computational_budget["latency_timed_prediction_calls_maximum"])
    reconstructed_timed_inferences = (
        (len(optimizer_trials_df) + len(doe_selection_runs_df) + len(doe_matched_front_df)) * 30
        + total_latency_session_models * 2000
    )
    if reconstructed_timed_inferences != total_timed_inferences:
        raise ReportingDataError(
            f"Reconstructed timed inferences ({reconstructed_timed_inferences}) != computational_budget.json ({total_timed_inferences})"
        )
    total_warmup_inferences = int(computational_budget["latency_warmup_prediction_calls_maximum"])
    reconstructed_warmup_inferences = (
        (len(optimizer_trials_df) + len(doe_selection_runs_df) + len(doe_matched_front_df)) * 10
        + total_latency_session_models * 100
    )
    if reconstructed_warmup_inferences != total_warmup_inferences:
        raise ReportingDataError(
            f"Reconstructed warmup inferences ({reconstructed_warmup_inferences}) != computational_budget.json ({total_warmup_inferences})"
        )
    total_all_inferences = total_timed_inferences + total_warmup_inferences

    method_timing_summary: Dict[str, Dict[str, float]] = {}
    for opt_k in ["multi_objective_tpe", "constrained_tpe", "random_search", "single_objective_tpe"]:
        sub_rep = optimizer_replicates_df[optimizer_replicates_df["optimizer"] == opt_k]
        sub_tr = optimizer_trials_df[optimizer_trials_df["optimizer"] == opt_k]
        w_mean = float(sub_rep["method_wall_time_s"].mean())
        w_sd = float(sub_rep["method_wall_time_s"].std(ddof=1))
        w_sum = float(sub_rep["method_wall_time_s"].sum())
        f_ms = float(sub_tr["training_time_s"].mean() * 1000.0)
        method_timing_summary[opt_k] = {
            "wall_s_mean": w_mean,
            "wall_clock_s_mean": w_mean,
            "wall_s_sd": w_sd,
            "wall_clock_s_sd": w_sd,
            "wall_s_sum": w_sum,
            "fit_ms_mean": f_ms,
            "train_ms_per_fit_mean": f_ms,
        }
    doe_w_mean = float(doe_selection_replicates_df["replicate_wall_time_s"].mean())
    doe_w_sd = float(doe_selection_replicates_df["replicate_wall_time_s"].std(ddof=1))
    doe_w_sum = float(doe_selection_replicates_df["replicate_wall_time_s"].sum())
    doe_f_ms = float(doe_selection_runs_df["training_time_s"].mean() * 1000.0)
    doe_timing_entry = {
        "wall_s_mean": doe_w_mean,
        "wall_clock_s_mean": doe_w_mean,
        "wall_s_sd": doe_w_sd,
        "wall_clock_s_sd": doe_w_sd,
        "wall_s_sum": doe_w_sum,
        "fit_ms_mean": doe_f_ms,
        "train_ms_per_fit_mean": doe_f_ms,
    }
    method_timing_summary["repeated_doe"] = doe_timing_entry
    method_timing_summary["repeated_preplanned_doe_multi_objective"] = doe_timing_entry

    # Reconstruct prospective optimizer summaries and cross-check against optimizer_summary.json & doe_selection_summary.json
    method_metadata = {
        "repeated_preplanned_doe_multi_objective": (
            r"Repeated DOE MO ($\mathbf{x}^*_{\text{MO}}$)",
            r"\makecell[l]{Repeated DOE MO\\($\mathbf{x}^*_{\text{MO}}$)}",
            "RSM ($N=20$)",
            "RSM ($N{=}20$)",
        ),
        "multi_objective_tpe": (
            "Multi-Objective TPE",
            r"\makecell[l]{Multi-Objective TPE}",
            "Parzen ($N=20$)",
            "Parzen ($N{=}20$)",
        ),
        "constrained_tpe": (
            r"Constrained TPE ($\le 145\,\mu\text{s}$)",
            r"\makecell[l]{Constrained TPE\\($\le 145\,\mu\text{s}$)}",
            "Parzen ($N=20$)",
            "Parzen ($N{=}20$)",
        ),
        "repeated_preplanned_doe_single_objective": (
            "Repeated DOE SO ($d = 7$)",
            r"\makecell[l]{Repeated DOE SO\\($d{=}7$)}",
            "RSM ($N=20$)",
            "RSM ($N{=}20$)",
        ),
        "single_objective_tpe": (
            "Single-Objective TPE",
            r"\makecell[l]{Single-Objective TPE}",
            "Parzen ($N=20$)",
            "Parzen ($N{=}20$)",
        ),
        "random_search": (
            "Unguided Random Search",
            r"\makecell[l]{Unguided Random Search}",
            "Uniform ($N=20$)",
            "Uniform ($N{=}20$)",
        ),
    }

    doe_methods_map = doe_selection_summary.get("methods", {})
    opt_methods_map = optimizer_summary.get("optimizers", {})

    prospective_optimizers: Dict[str, OptimizerBenchmarkMetrics] = {}
    for opt_key in PROSPECTIVE_OPTIMIZER_ORDER:
        g = final_summary_df[final_summary_df["optimizer"] == opt_key]
        if len(g) != 20:
            raise ReportingDataError(f"Expected 20 rows in final_summary.csv for {opt_key}, got {len(g)}")

        saved_source = (
            doe_methods_map[opt_key]
            if opt_key in doe_methods_map
            else opt_methods_map.get(opt_key)
        )
        if saved_source is None:
            raise ReportingDataError(
                f"Optimizer {opt_key} missing from both optimizer_summary.json and doe_selection_summary.json"
            )

        def _ci95(series: pd.Series) -> Tuple[float, float, float, float]:
            m = float(series.mean())
            s = float(series.std(ddof=1))
            if s == 0.0:
                return m, 0.0, m, m
            hw = float(stats.t.ppf(0.975, len(series) - 1) * s / np.sqrt(len(series)))
            return m, s, m - hw, m + hw

        vm, vs, v_lo, v_hi = _ci95(g["validation_rmse_mean"])
        tm, ts, t_lo, t_hi = _ci95(g["test_rmse_mean"])
        lm, ls, l_lo, l_hi = _ci95(g["predict_latency_us"])
        ilm, ils, _, _ = _ci95(g["inplace_predict_latency_us"])

        # Cross-check against saved JSON summary
        saved_val = saved_source["independently_retrained_validation_rmse"]
        saved_test = saved_source["final_test_rmse"]
        _assert_close(vm, float(saved_val["mean"]), atol=1e-9, label=f"{opt_key} val_rmse mean")
        _assert_close(vs, float(saved_val["standard_deviation"]), atol=1e-9, label=f"{opt_key} val_rmse sd")
        _assert_close(v_lo, float(saved_val["confidence_interval_95"][0]), atol=1e-9, label=f"{opt_key} val_rmse ci_low")
        _assert_close(v_hi, float(saved_val["confidence_interval_95"][1]), atol=1e-9, label=f"{opt_key} val_rmse ci_high")
        _assert_close(tm, float(saved_test["mean"]), atol=1e-9, label=f"{opt_key} test_rmse mean")
        _assert_close(ts, float(saved_test["standard_deviation"]), atol=1e-9, label=f"{opt_key} test_rmse sd")
        _assert_close(t_lo, float(saved_test["confidence_interval_95"][0]), atol=1e-9, label=f"{opt_key} test_rmse ci_low")
        _assert_close(t_hi, float(saved_test["confidence_interval_95"][1]), atol=1e-9, label=f"{opt_key} test_rmse ci_high")

        eval_opt = final_evaluations_df[final_evaluations_df["optimizer"] == opt_key]
        retrain_sd = float(eval_opt.groupby("selection_id")["test_rmse"].std(ddof=1).mean())

        s_feas = int(g["search_time_feasible"].sum())
        b_feas = int(g["benchmark_time_feasible"].sum())
        recomp_b_feas = int((g["predict_latency_us"] <= float(rev_cfg["max_latency_us"])).sum())
        if b_feas != recomp_b_feas:
            raise ReportingDataError(
                f"{opt_key} benchmark_time_feasible column ({b_feas}) != predict_latency_us <= 145 ({recomp_b_feas})"
            )

        md_name, tex_name, md_basis, tex_basis = method_metadata[opt_key]
        prospective_optimizers[opt_key] = OptimizerBenchmarkMetrics(
            optimizer_key=opt_key,
            display_name_md=md_name,
            display_name_tex=tex_name,
            search_basis_md=md_basis,
            search_basis_tex=tex_basis,
            n_replicates=len(g),
            val_rmse_mean=vm,
            val_rmse_sd=vs,
            val_rmse_ci_low=v_lo,
            val_rmse_ci_high=v_hi,
            test_rmse_mean=tm,
            test_rmse_sd=ts,
            test_rmse_ci_low=t_lo,
            test_rmse_ci_high=t_hi,
            retrain_sd=retrain_sd,
            predict_latency_mean=lm,
            predict_latency_sd=ls,
            predict_latency_ci_low=l_lo,
            predict_latency_ci_high=l_hi,
            inplace_latency_mean=ilm,
            inplace_latency_sd=ils,
            search_feasible_count=s_feas,
            benchmark_feasible_count=b_feas,
            benchmark_feasible_pct=int(round(b_feas / len(g) * 100)),
        )

    # Constrained TPE depth breakdown
    ctpe_g = final_summary_df[final_summary_df["optimizer"] == "constrained_tpe"]
    ctpe_d6 = ctpe_g[ctpe_g["max_depth"] == 6]
    ctpe_d7 = ctpe_g[ctpe_g["max_depth"] == 7]
    if len(ctpe_d6) + len(ctpe_d7) != 20:
        raise ReportingDataError(
            f"Unexpected depth distribution for constrained_tpe: d6={len(ctpe_d6)}, d7={len(ctpe_d7)}"
        )
    ctpe_depth6_count = len(ctpe_d6)
    ctpe_depth6_feasible = int(ctpe_d6["benchmark_time_feasible"].sum())
    ctpe_depth6_latency_mean = float(ctpe_d6["predict_latency_us"].mean())
    ctpe_depth6_latency_sd = float(ctpe_d6["predict_latency_us"].std(ddof=1))
    ctpe_depth7_count = len(ctpe_d7)
    ctpe_depth7_feasible = int(ctpe_d7["benchmark_time_feasible"].sum())
    ctpe_depth7_latency_mean = float(ctpe_d7["predict_latency_us"].mean())
    ctpe_depth7_latency_sd = float(ctpe_d7["predict_latency_us"].std(ddof=1))

    # Latency reduction of Repeated DOE SO vs SO-TPE
    sotpe_lat = prospective_optimizers["single_objective_tpe"].predict_latency_mean
    doeso_lat = prospective_optimizers["repeated_preplanned_doe_single_objective"].predict_latency_mean
    doe_so_vs_sotpe_latency_reduction_pct = float((sotpe_lat - doeso_lat) / sotpe_lat * 100.0)

    # Statistical comparison: Repeated DOE MO vs MO-TPE
    doe_mo_test = final_summary_df[
        final_summary_df["optimizer"] == "repeated_preplanned_doe_multi_objective"
    ].sort_values("optimizer_replicate_id")["test_rmse_mean"].values
    mo_tpe_test = final_summary_df[
        final_summary_df["optimizer"] == "multi_objective_tpe"
    ].sort_values("optimizer_replicate_id")["test_rmse_mean"].values

    doe_mo_vs_motpe_diff = float(np.mean(doe_mo_test) - np.mean(mo_tpe_test))
    welch_res = stats.ttest_ind(doe_mo_test, mo_tpe_test, equal_var=False)
    doe_mo_vs_motpe_welch_t = float(welch_res.statistic)
    doe_mo_vs_motpe_welch_p = float(welch_res.pvalue)

    paired_mo_res = paired_difference_summary(doe_mo_test, mo_tpe_test)
    doe_mo_vs_motpe_paired_t = float(paired_mo_res["paired_t_statistic"])
    doe_mo_vs_motpe_paired_p = float(paired_mo_res["paired_t_pvalue"])

    matched_paired = [
        c
        for c in paired_comparisons
        if c.get("comparison") == "Repeated DOE MO minus MO-TPE"
        and c.get("metric") == "test_rmse"
    ]
    if len(matched_paired) != 1:
        raise ReportingDataError(
            f"Expected 1 'Repeated DOE MO minus MO-TPE' test_rmse entry in paired_comparisons.json, found {len(matched_paired)}"
        )
    saved_paired_mo = matched_paired[0]["paired_difference"]
    _assert_close(
        doe_mo_vs_motpe_diff,
        float(saved_paired_mo["mean_difference"]),
        atol=1e-9,
        label="DOE MO vs MO-TPE mean difference",
    )
    _assert_close(
        doe_mo_vs_motpe_paired_t,
        float(saved_paired_mo["paired_t_statistic"]),
        atol=1e-9,
        label="DOE MO vs MO-TPE paired t-statistic",
    )
    _assert_close(
        doe_mo_vs_motpe_paired_p,
        float(saved_paired_mo["paired_t_pvalue"]),
        atol=1e-9,
        label="DOE MO vs MO-TPE paired p-value",
    )

    # 5. Row-level reconstruction of Pareto Hypervolumes across both reference points
    _require_keys(hypervolume, ["development_domain", "external_test_domain"], "hypervolume.json")
    dev_refs = hypervolume["development_domain"]["reference_points"]
    test_refs = hypervolume["external_test_domain"]["reference_points"]

    ref1_key = f"[{hv_ref_primary[0]}, {hv_ref_primary[1]}]"
    ref2_key = f"[{hv_ref_secondary[0]}, {hv_ref_secondary[1]}]"
    _require_keys(dev_refs, [ref1_key, ref2_key], "hypervolume.json:development_domain.reference_points")
    _require_keys(test_refs, [ref1_key, ref2_key], "hypervolume.json:external_test_domain.reference_points")

    for ref_tuple, rkey in [(hv_ref_primary, ref1_key), (hv_ref_secondary, ref2_key)]:
        dev_entry = dev_refs[rkey]
        test_entry = test_refs[rkey]

        # Reconstruct MO-TPE per-replicate hypervolumes from optimizer_trials.csv
        motpe_trials = optimizer_trials_df[optimizer_trials_df["optimizer"] == "multi_objective_tpe"]
        motpe_hvs = []
        for rep_id in range(20):
            sub = motpe_trials[motpe_trials["replicate_id"] == rep_id]
            pts = list(zip(sub["validation_rmse"].astype(float), sub["predict_latency_us"].astype(float)))
            hv_calc = hypervolume_2d_min(pts, ref_tuple)
            saved_rep_hv = float(dev_entry["mo_tpe_by_replicate"][str(rep_id)]["value"])
            _assert_close(hv_calc.value, saved_rep_hv, atol=1e-9, label=f"MO-TPE HV rep {rep_id} at {rkey}")
            motpe_hvs.append(hv_calc.value)
        _assert_close(
            float(np.mean(motpe_hvs)),
            float(dev_entry["mo_tpe_hypervolume_distribution"]["mean"]),
            atol=1e-9,
            label=f"MO-TPE HV mean at {rkey}",
        )
        _assert_close(
            float(np.std(motpe_hvs, ddof=1)),
            float(dev_entry["mo_tpe_hypervolume_distribution"]["standard_deviation"]),
            atol=1e-9,
            label=f"MO-TPE HV SD at {rkey}",
        )

        # Reconstruct Repeated DOE candidate fronts (5-block means per point_id) from doe_selection_runs.csv
        rdoe_hvs = []
        for rep_id in range(20):
            sub = doe_selection_runs_df[doe_selection_runs_df["doe_replicate_id"] == rep_id]
            pt_means = sub.groupby("point_id")[["validation_rmse", "predict_latency_us"]].mean()
            pts = list(zip(pt_means["validation_rmse"].astype(float), pt_means["predict_latency_us"].astype(float)))
            hv_calc = hypervolume_2d_min(pts, ref_tuple)
            saved_rep_hv = float(dev_entry["repeated_doe_by_replicate"][str(rep_id)]["value"])
            _assert_close(hv_calc.value, saved_rep_hv, atol=1e-9, label=f"Repeated DOE HV rep {rep_id} at {rkey}")
            rdoe_hvs.append(hv_calc.value)
        _assert_close(
            float(np.mean(rdoe_hvs)),
            float(dev_entry["repeated_doe_hypervolume_distribution"]["mean"]),
            atol=1e-9,
            label=f"Repeated DOE HV mean at {rkey}",
        )
        _assert_close(
            float(np.std(rdoe_hvs, ddof=1)),
            float(dev_entry["repeated_doe_hypervolume_distribution"]["standard_deviation"]),
            atol=1e-9,
            label=f"Repeated DOE HV SD at {rkey}",
        )

        # Reconstruct Fixed Full DOE 27-point front and 2-point historical set from doe_matched_candidate_front.csv
        pts_27 = list(
            zip(
                doe_matched_front_df["validation_rmse"].astype(float),
                doe_matched_front_df["predict_latency_us"].astype(float),
            )
        )
        hv_27 = hypervolume_2d_min(pts_27, ref_tuple)
        _assert_close(
            hv_27.value,
            float(dev_entry["full_doe_evaluated_candidate_front"]["value"]),
            atol=1e-9,
            label=f"Fixed Full DOE 27-point HV at {rkey}",
        )

        sub_2 = doe_matched_front_df[
            doe_matched_front_df["candidate_type"] == "historical_doe_selected_operating_point"
        ]
        pts_2 = list(zip(sub_2["validation_rmse"].astype(float), sub_2["predict_latency_us"].astype(float)))
        hv_2 = hypervolume_2d_min(pts_2, ref_tuple)
        _assert_close(
            hv_2.value,
            float(dev_entry["doe_two_point"]["value"]),
            atol=1e-9,
            label=f"Historical DOE 2-point HV at {rkey}",
        )

        # Reconstruct external holdout test set hypervolume across all 122 frozen selections
        pts_holdout = list(
            zip(
                final_summary_df["test_rmse_mean"].astype(float),
                final_summary_df["predict_latency_us"].astype(float),
            )
        )
        hv_holdout = hypervolume_2d_min(pts_holdout, ref_tuple)
        _assert_close(
            hv_holdout.value,
            float(test_entry["all_frozen_selections"]["value"]),
            atol=1e-9,
            label=f"External holdout 122-selection HV at {rkey}",
        )

    dev_060 = dev_refs[ref1_key]
    dev_065 = dev_refs[ref2_key]
    test_060 = test_refs[ref1_key]
    test_065 = test_refs[ref2_key]

    motpe_nd = pd.Series([len(v["pareto"]["points"]) for v in dev_060["mo_tpe_by_replicate"].values()])
    rdoe_nd = pd.Series([len(v["pareto"]["points"]) for v in dev_060["repeated_doe_by_replicate"].values()])
    fdoe_nd = len(dev_060["full_doe_evaluated_candidate_front"]["pareto"]["points"])
    hdoe_nd = len(dev_060["doe_two_point"]["pareto"]["points"])
    holdout_pts = test_060["all_frozen_selections"]["pareto"]["points"]
    holdout_nd = len(holdout_pts)

    # Reconstruct per-optimizer breakdown of the 12 holdout non-dominated configurations
    holdout_nondominated_by_optimizer: Dict[str, int] = {opt: 0 for opt in PROSPECTIVE_OPTIMIZER_ORDER}
    for pt_rmse, pt_lat in holdout_pts:
        matched = final_summary_df[
            np.isclose(final_summary_df["test_rmse_mean"], float(pt_rmse), atol=1e-12)
            & np.isclose(final_summary_df["predict_latency_us"], float(pt_lat), atol=1e-9)
        ]
        if len(matched) != 1:
            raise ReportingDataError(
                f"Expected exactly 1 frozen selection matching holdout Pareto point ({pt_rmse}, {pt_lat}), found {len(matched)}"
            )
        opt_name = str(matched.iloc[0]["optimizer"])
        holdout_nondominated_by_optimizer[opt_name] = holdout_nondominated_by_optimizer.get(opt_name, 0) + 1

    if sum(holdout_nondominated_by_optimizer.values()) != holdout_nd:
        raise ReportingDataError("Holdout non-dominated point breakdown does not sum to holdout_nd")

    holdout_pareto_bootstrap = _compute_holdout_pareto_bootstrap(
        final_evaluations_df,
        final_summary_df,
        latency_measurement,
        hv_ref_primary,
        hv_ref_secondary,
    )

    hv_summary = {
        "motpe_nd_mean": float(motpe_nd.mean()),
        "motpe_nd_sd": float(motpe_nd.std(ddof=1)),
        "rdoe_nd_mean": float(rdoe_nd.mean()),
        "rdoe_nd_sd": float(rdoe_nd.std(ddof=1)),
        "fdoe_nd": int(fdoe_nd),
        "hdoe_nd": int(hdoe_nd),
        "holdout_nd": int(holdout_nd),
        "motpe_dev_060_mean": float(dev_060["mo_tpe_hypervolume_distribution"]["mean"]),
        "motpe_dev_060_sd": float(dev_060["mo_tpe_hypervolume_distribution"]["standard_deviation"]),
        "motpe_dev_065_mean": float(dev_065["mo_tpe_hypervolume_distribution"]["mean"]),
        "motpe_dev_065_sd": float(dev_065["mo_tpe_hypervolume_distribution"]["standard_deviation"]),
        "rdoe_dev_060_mean": float(dev_060["repeated_doe_hypervolume_distribution"]["mean"]),
        "rdoe_dev_060_sd": float(dev_060["repeated_doe_hypervolume_distribution"]["standard_deviation"]),
        "rdoe_dev_065_mean": float(dev_065["repeated_doe_hypervolume_distribution"]["mean"]),
        "rdoe_dev_065_sd": float(dev_065["repeated_doe_hypervolume_distribution"]["standard_deviation"]),
        "fdoe_dev_060": float(dev_060["full_doe_evaluated_candidate_front"]["value"]),
        "fdoe_dev_065": float(dev_065["full_doe_evaluated_candidate_front"]["value"]),
        "hdoe_dev_060": float(dev_060["doe_two_point"]["value"]),
        "hdoe_dev_065": float(dev_065["doe_two_point"]["value"]),
        "diff_fdoe_minus_motpe_060": float(
            dev_060["full_doe_evaluated_candidate_front"]["value"]
            - dev_060["mo_tpe_hypervolume_distribution"]["mean"]
        ),
        "diff_fdoe_minus_motpe_065": float(
            dev_065["full_doe_evaluated_candidate_front"]["value"]
            - dev_065["mo_tpe_hypervolume_distribution"]["mean"]
        ),
        "holdout_frozen_060": float(test_060["all_frozen_selections"]["value"]),
        "holdout_frozen_065": float(test_065["all_frozen_selections"]["value"]),
    }

    # 6. Historical baseline rows (v1.0.0)
    hist_specs = [
        (
            "Sequential DOE-CCD (x*, Multi-Objective)",
            r"Historical DOE MO ($\mathbf{x}^*_{\text{MO}}$)",
            r"\makecell[l]{Historical DOE MO\\($\mathbf{x}^*_{\text{MO}}$)}",
            "RSM (Single)",
            "RSM (Single)",
        ),
        (
            "Multi-Objective TPE (Desirability)",
            "Historical Multi-Obj TPE",
            r"\makecell[l]{Historical Multi-Obj TPE}",
            "Parzen (Single)",
            "Parzen (Single)",
        ),
        (
            "Constrained TPE (Latency <= 145 us)",
            "Historical Constrained TPE",
            r"\makecell[l]{Historical Constrained TPE}",
            "Parzen (Single)",
            "Parzen (Single)",
        ),
        (
            "Sequential DOE-CCD (Single-Objective)",
            "Historical DOE SO ($d = 7$)",
            r"\makecell[l]{Historical DOE SO\\($d{=}7$)}",
            "RSM (Single)",
            "RSM (Single)",
        ),
        (
            "Bayesian Optimization (Optuna TPE Single-Obj)",
            "Historical Bayesian TPE",
            r"\makecell[l]{Historical Bayesian TPE}",
            "Parzen (Single)",
            "Parzen (Single)",
        ),
        (
            "Unguided Random Search",
            "Historical Random Search",
            r"\makecell[l]{Historical Random Search}",
            "Uniform (Single)",
            "Uniform (Single)",
        ),
    ]
    historical_benchmarks_list: List[HistoricalBenchmarkRow] = []
    for raw_name, md_label, tex_label, md_basis, tex_basis in hist_specs:
        matches = historical_benchmark_df[historical_benchmark_df["method"] == raw_name]
        if len(matches) != 1:
            raise ReportingDataError(f"Expected 1 row in results/benchmark.csv for {raw_name}, got {len(matches)}")
        hrow = matches.iloc[0]
        inp_val = hrow["inplace_latency_us_median"]
        inp_float = None if pd.isna(inp_val) else float(inp_val)
        pred_lat = float(hrow["predict_latency_us_median"])
        historical_benchmarks_list.append(
            HistoricalBenchmarkRow(
                raw_method=raw_name,
                display_name_md=md_label,
                display_name_tex=tex_label,
                search_basis_md=md_basis,
                search_basis_tex=tex_basis,
                val_rmse_mean=float(hrow["val_rmse_mean"]),
                test_rmse_mean=float(hrow["test_rmse_mean"]),
                predict_latency_us_median=pred_lat,
                inplace_latency_us_median=inp_float,
                feasible_145=bool(pred_lat <= 145.0),
            )
        )

    # 7. Verify cryptographic SHA-256 digests of all 40 canonical evidence artifacts
    verify_canonical_evidence_hashes(base)

    return ReportingDataset(
        root_dir=base,
        config=config,
        env=env,
        block_seeds=block_seeds,
        confirmation_seeds=confirmation_seeds,
        fresh_eval_seeds=fresh_eval_seeds,
        hv_ref_primary=hv_ref_primary,
        hv_ref_secondary=hv_ref_secondary,
        runs_df=runs_df,
        phase1=phase1,
        phase3=phase3,
        lof=lof,
        diagnostics=diagnostics,
        icc=icc,
        confirmation=confirmation,
        confirmation_mo_df=confirmation_mo_df,
        confirmation_so_df=confirmation_so_df,
        depth_opt_df=depth_opt_df,
        desirability_sens_df=desirability_sens_df,
        latency_models_df=latency_models_df,
        historical_benchmark_df=historical_benchmark_df,
        historical_benchmark_summary=historical_benchmark_summary,
        anova_phase1_df=anova_phase1_df,
        anova_ccd_y1_df=anova_ccd_y1_df,
        hc3_ccd_y1_se=hc3_ccd_y1_se,
        hc3_ccd_y1_p=hc3_ccd_y1_p,
        ccd_y1_rsq=ccd_y1_rsq,
        ccd_y1_adj_rsq=ccd_y1_adj_rsq,
        anova_ccd_y2_df=anova_ccd_y2_df,
        lof_decomp=lof_decomp,
        desirability_star=desirability_star,
        latency_jacobian_adjustment=latency_jacobian_adjustment,
        doe_axial_before_run100=doe_axial_before_run100,
        doe_low_eta_run_count=doe_low_eta_run_count,
        doe_low_eta_run_pct=doe_low_eta_run_pct,
        doe_low_eta_mean_rmse=doe_low_eta_mean_rmse,
        doe_high_eta_mean_rmse=doe_high_eta_mean_rmse,
        curvature_sensitivity=curvature_sensitivity,
        hist_runs_interface_overhead_mean=hist_runs_interface_overhead_mean,
        hist_runs_interface_overhead_sd=hist_runs_interface_overhead_sd,
        hist_runs_interface_overhead_min=hist_runs_interface_overhead_min,
        hist_runs_interface_overhead_max=hist_runs_interface_overhead_max,
        hist_conf_mo_interface_overhead_mean=hist_conf_mo_interface_overhead_mean,
        hist_conf_mo_interface_overhead_sd=hist_conf_mo_interface_overhead_sd,
        hist_conf_so_interface_overhead_mean=hist_conf_so_interface_overhead_mean,
        hist_conf_so_interface_overhead_sd=hist_conf_so_interface_overhead_sd,
        hist_bench_interface_overhead_mean=hist_bench_interface_overhead_mean,
        hist_bench_interface_overhead_sd=hist_bench_interface_overhead_sd,
        hist_bench_interface_overhead_min=hist_bench_interface_overhead_min,
        hist_bench_interface_overhead_max=hist_bench_interface_overhead_max,
        rev_dir=rev_dir,
        run_manifest=run_manifest,
        provenance=provenance,
        computational_budget=computational_budget,
        finalized_selections=finalized_selections,
        final_evaluations_df=final_evaluations_df,
        final_summary_df=final_summary_df,
        optimizer_summary=optimizer_summary,
        doe_selection_summary=doe_selection_summary,
        latency_measurement=latency_measurement,
        latency_interface_overhead=latency_interface_overhead,
        hypervolume=hypervolume,
        paired_comparisons=paired_comparisons,
        total_model_fits=total_model_fits,
        total_timed_inferences=total_timed_inferences,
        total_warmup_inferences=total_warmup_inferences,
        total_all_inferences=total_all_inferences,
        total_selection_records=total_selection_records,
        distinct_config_hashes=distinct_config_hashes,
        total_evaluation_rows=total_evaluation_rows,
        between_session_latency_sd_mean=between_session_latency_sd_mean,
        interface_overhead_mean=interface_overhead_mean,
        interface_overhead_sd=interface_overhead_sd,
        interface_overhead_min=interface_overhead_min,
        interface_overhead_max=interface_overhead_max,
        method_timing_summary=method_timing_summary,
        prospective_optimizers=prospective_optimizers,
        historical_benchmarks=tuple(historical_benchmarks_list),
        ctpe_depth6_count=ctpe_depth6_count,
        ctpe_depth6_feasible=ctpe_depth6_feasible,
        ctpe_depth6_latency_mean=ctpe_depth6_latency_mean,
        ctpe_depth6_latency_sd=ctpe_depth6_latency_sd,
        ctpe_depth7_count=ctpe_depth7_count,
        ctpe_depth7_feasible=ctpe_depth7_feasible,
        ctpe_depth7_latency_mean=ctpe_depth7_latency_mean,
        ctpe_depth7_latency_sd=ctpe_depth7_latency_sd,
        doe_so_vs_sotpe_latency_reduction_pct=doe_so_vs_sotpe_latency_reduction_pct,
        doe_mo_vs_motpe_diff=doe_mo_vs_motpe_diff,
        doe_mo_vs_motpe_welch_t=doe_mo_vs_motpe_welch_t,
        doe_mo_vs_motpe_welch_p=doe_mo_vs_motpe_welch_p,
        doe_mo_vs_motpe_paired_t=doe_mo_vs_motpe_paired_t,
        doe_mo_vs_motpe_paired_p=doe_mo_vs_motpe_paired_p,
        hv_summary=hv_summary,
        holdout_nondominated_by_optimizer=holdout_nondominated_by_optimizer,
        holdout_pareto_bootstrap=holdout_pareto_bootstrap,
    )


def build_scientific_results_manifest(ds: ReportingDataset) -> Dict[str, Any]:
    """Build the machine-readable scientific results traceability manifest (Task 11)."""
    metrics: List[Dict[str, Any]] = []

    def add_metric(
        metric_id: str,
        value: Any,
        formatted_value: str,
        precision: str,
        input_artifact_path: str,
        source_field_or_reconstruction: str,
        evaluation_domain: str,
        protocol: str,
        n_independent_units: int,
        uncertainty_definition: str,
        verification_status: str,
        target_documents: Sequence[str],
    ) -> None:
        if verification_status not in {"RECONSTRUCTED", "SOURCE-TRACED"}:
            raise ReportingDataError(f"Invalid verification_status: {verification_status}")
        metrics.append(
            {
                "metric_id": metric_id,
                "value": value,
                "formatted_value": formatted_value,
                "precision": precision,
                "input_artifact_path": input_artifact_path,
                "source_field_or_reconstruction": source_field_or_reconstruction,
                "evaluation_domain": evaluation_domain,
                "protocol": protocol,
                "n_independent_units": n_independent_units,
                "uncertainty_definition": uncertainty_definition,
                "verification_status": verification_status,
                "target_documents": list(target_documents),
            }
        )

    docs_all = ["REPORT.md", "report.tex", "results/macros.tex"]
    docs_bm = ["REPORT.md", "report.tex", "results/macros.tex", "tables/tab_benchmarks.tex"]
    docs_hv = ["REPORT.md", "report.tex", "results/macros.tex", "tables/tab_hypervolume_comparison.tex"]
    docs_conf = ["REPORT.md", "report.tex", "results/macros.tex", "tables/tab_confirmation.tex"]

    # Protocol & Seeds
    add_metric(
        "protocol.block_seeds",
        list(ds.block_seeds),
        "42, 101, 202, 303, 404",
        "exact_integer_list",
        "config.yaml",
        "block_seeds (verified against results/runs.csv)",
        "development_split_blocks",
        "RCBD_5_blocks_75_25_split",
        5,
        "fixed_nuisance_blocks",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "protocol.confirmation_seeds",
        list(ds.confirmation_seeds),
        "505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414",
        "exact_integer_list",
        "config.yaml",
        "confirmation_seeds (verified against results/confirmation_runs.csv and results/confirmation_runs_single_obj.csv)",
        "confirmation_splits_and_holdout",
        "10_fresh_confirmation_seeds",
        10,
        "sample_sd_across_10_confirmation_seeds",
        "RECONSTRUCTED",
        docs_conf,
    )
    add_metric(
        "protocol.fresh_evaluation_seeds",
        list(ds.fresh_eval_seeds),
        "2001..2020",
        "exact_integer_list",
        "config.yaml",
        "revision_v2.fresh_evaluation_seeds (verified against results/revision_v2/full_run_001/final_evaluations.csv)",
        "external_holdout_test_and_dev_validation",
        "20_fresh_retraining_seeds",
        20,
        "within_selection_retraining_sd_ddof1",
        "RECONSTRUCTED",
        docs_bm,
    )
    add_metric(
        "protocol.hypervolume_reference_primary",
        list(ds.hv_ref_primary),
        "[0.60, 250.0]",
        "[2dp, 1dp]",
        "config.yaml",
        "revision_v2.hypervolume_reference_point (verified against hypervolume.json)",
        "multi_objective_rmse_latency",
        "2d_minimization_hypervolume",
        1,
        "fixed_reference_point",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "protocol.hypervolume_reference_secondary",
        list(ds.hv_ref_secondary),
        "[0.65, 275.0]",
        "[2dp, 1dp]",
        "config.yaml",
        "revision_v2.hypervolume_secondary_reference_point (verified against hypervolume.json)",
        "multi_objective_rmse_latency",
        "2d_minimization_hypervolume",
        1,
        "fixed_reference_point",
        "RECONSTRUCTED",
        docs_hv,
    )

    # Phase 1, 2, 3, 4 core metrics
    add_metric(
        "phase1.curvature_f_stat",
        float(ds.phase1["f_curvature"]),
        f"{ds.phase1['f_curvature']:,.2f}",
        "2dp",
        "results/runs.csv",
        "SS_curv / MS_pe_center reconstructed from 80 factorial and 20 center runs (verified against results/phase1.json:f_curvature)",
        "development_validation_rmse",
        "1df_curvature_F_test_vs_within_block_center_PE_15df",
        100,
        "F_distribution_1_15_df",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "phase1.factorial_mean_rmse",
        float(ds.phase1["yF_bar"]),
        f"{ds.phase1['yF_bar']:.4f}",
        "4dp",
        "results/runs.csv",
        "mean(val_rmse) for Phase1_Factorial (n=80)",
        "development_validation_rmse",
        "RCBD_phase1",
        80,
        "mean_across_80_factorial_runs",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "phase1.center_mean_rmse",
        float(ds.phase1["yC_bar"]),
        f"{ds.phase1['yC_bar']:.4f}",
        "4dp",
        "results/runs.csv",
        "mean(val_rmse) for Phase1_Center (n=20)",
        "development_validation_rmse",
        "RCBD_phase1",
        20,
        "mean_across_20_center_runs",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "phase2.ccd_rsquared",
        ds.ccd_y1_rsq,
        f"{ds.ccd_y1_rsq:.4f}",
        "4dp",
        "results/runs.csv",
        "OLS second-order response surface R^2 on N=140 runs",
        "development_validation_rmse",
        "FCCD_second_order_OLS",
        140,
        "goodness_of_fit_R2",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "phase2.block_icc_anova",
        float(ds.icc["Y1"]["icc_anova"]),
        f"{ds.icc['Y1']['icc_anova']:.4f}",
        "4dp",
        "results/icc.json",
        "Y1.icc_anova (verified via Phase 2 ANOVA MS_block and MS_E from results/runs.csv)",
        "development_validation_rmse",
        "RCBD_variance_components",
        5,
        "intraclass_correlation_ratio",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "phase2.lof_f_vs_additive",
        float(ds.lof["Y1"]["F_LoF"]),
        f"{ds.lof['Y1']['F_LoF']:.2f}",
        "2dp",
        "results/lof.json",
        "Y1.F_LoF (10 df structural LoF vs 111 df additive residual)",
        "development_validation_rmse",
        "FCCD_lack_of_fit_test",
        140,
        "F_distribution_10_111_df",
        "SOURCE-TRACED",
        ["REPORT.md", "report.tex", "results/macros.tex", "tables/tab_lof.tex"],
    )
    add_metric(
        "phase2.lof_f_vs_center_pe",
        ds.lof_decomp["f_lof_center_y1"],
        f"{ds.lof_decomp['f_lof_center_y1']:.2f}",
        "2dp",
        "results/runs.csv",
        "MS_LoF / MS_PE_center (10 df structural LoF vs 15 df within-block center pure error)",
        "development_validation_rmse",
        "FCCD_lack_of_fit_test_center_pe",
        140,
        "F_distribution_10_15_df",
        "RECONSTRUCTED",
        ["REPORT.md", "report.tex", "results/macros.tex", "tables/tab_lof.tex"],
    )
    add_metric(
        "phase4.composite_desirability",
        ds.desirability_star["D"],
        f"{ds.desirability_star['D']:.4f}",
        "4dp",
        "results/confirmation.json",
        "sqrt(d1 * d2) reconstructed from Y1_Val_RMSE.predicted_mean and Y2_Latency.predicted_mean",
        "development_surrogate_predictions",
        "Derringer_Suich_desirability",
        1,
        "deterministic_surrogate_optimum",
        "RECONSTRUCTED",
        docs_all,
    )

    # Phase 5 Confirmation
    add_metric(
        "phase5.mo_emp_val_rmse_mean",
        float(ds.confirmation["Y1_Val_RMSE"]["empirical_mean"]),
        f"{ds.confirmation['Y1_Val_RMSE']['empirical_mean']:.4f}",
        "4dp",
        "results/confirmation_runs.csv",
        "mean(val_rmse) across 10 confirmation seeds (verified against results/confirmation.json)",
        "development_validation_rmse_fresh_seeds",
        "confirmation_10_seeds",
        10,
        f"mean +/- SD ({ds.confirmation['Y1_Val_RMSE']['empirical_std']:.4f}) across 10 seeds",
        "RECONSTRUCTED",
        docs_conf,
    )
    add_metric(
        "phase5.mo_emp_test_rmse_mean",
        float(ds.confirmation["Y1_Test_RMSE"]["empirical_mean"]),
        f"{ds.confirmation['Y1_Test_RMSE']['empirical_mean']:.4f}",
        "4dp",
        "results/confirmation_runs.csv",
        "mean(test_rmse) across 10 confirmation seeds (verified against results/confirmation.json)",
        "external_holdout_test_rmse",
        "confirmation_10_seeds",
        10,
        f"mean +/- SD ({ds.confirmation['Y1_Test_RMSE']['empirical_std']:.4f}) across 10 seeds",
        "RECONSTRUCTED",
        docs_conf,
    )
    add_metric(
        "phase5.mo_emp_latency_mean",
        float(ds.confirmation["Y2_Latency"]["empirical_mean"]),
        f"{ds.confirmation['Y2_Latency']['empirical_mean']:.2f}",
        "2dp (1dp in compact table)",
        "results/confirmation_runs.csv",
        "mean(latency_us_median) across 10 confirmation seeds (verified against results/confirmation.json)",
        "single_sample_inference_latency_us",
        "confirmation_10_seeds_1000_calls",
        10,
        f"mean +/- SD ({ds.confirmation['Y2_Latency']['empirical_std']:.2f}) across 10 seeds",
        "RECONSTRUCTED",
        docs_conf,
    )
    so_c = ds.confirmation["Single_Objective_Optimum"]
    add_metric(
        "phase5.so_emp_val_rmse_mean",
        float(so_c["empirical_val_rmse"]),
        f"{so_c['empirical_val_rmse']:.4f}",
        "4dp",
        "results/confirmation_runs_single_obj.csv",
        "mean(val_rmse) across 10 confirmation seeds (verified against results/confirmation.json)",
        "development_validation_rmse_fresh_seeds",
        "confirmation_10_seeds",
        10,
        f"mean +/- SD ({so_c['empirical_val_std']:.4f}) across 10 seeds",
        "RECONSTRUCTED",
        docs_conf,
    )
    add_metric(
        "phase5.so_emp_test_rmse_mean",
        float(so_c["empirical_test_rmse"]),
        f"{so_c['empirical_test_rmse']:.4f}",
        "4dp",
        "results/confirmation_runs_single_obj.csv",
        "mean(test_rmse) across 10 confirmation seeds (verified against results/confirmation.json)",
        "external_holdout_test_rmse",
        "confirmation_10_seeds",
        10,
        f"mean +/- SD ({so_c['empirical_test_std']:.4f}) across 10 seeds",
        "RECONSTRUCTED",
        docs_conf,
    )

    # Revision v2 Budget & Counts
    add_metric(
        "rev_v2.total_model_fits",
        ds.total_model_fits,
        f"{ds.total_model_fits:,}",
        "exact_integer",
        "results/revision_v2/full_run_001/computational_budget.json",
        "11200 + 2800 + 27 + 2440 + 610 reconstructed from CSV/JSON row counts",
        "campaign_computational_budget",
        "full_run_001",
        17077,
        "exact_count",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "rev_v2.total_timed_inferences",
        ds.total_timed_inferences,
        f"{ds.total_timed_inferences:,}",
        "exact_integer",
        "results/revision_v2/full_run_001/computational_budget.json",
        "(11200 + 2800 + 27)*30 + 610*2000 reconstructed from trial and latency session records",
        "campaign_computational_budget",
        "full_run_001",
        1640810,
        "exact_count",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "rev_v2.total_selection_records",
        ds.total_selection_records,
        str(ds.total_selection_records),
        "exact_integer",
        "results/revision_v2/full_run_001/finalized_selections.json",
        "len(configurations) == 122 (6 optimizers * 20 reps + 2 historical anchors)",
        "frozen_selection_records",
        "finalized_selections_freeze",
        122,
        "exact_count",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "rev_v2.distinct_config_hashes",
        ds.distinct_config_hashes,
        str(ds.distinct_config_hashes),
        "exact_integer",
        "results/revision_v2/full_run_001/finalized_selections.json",
        "len(set(config_sha256)) == 95 distinct hyperparameter configurations",
        "frozen_selection_records",
        "sha256_deduplicated_configs",
        95,
        "exact_count",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "rev_v2.total_evaluation_rows",
        ds.total_evaluation_rows,
        f"{ds.total_evaluation_rows:,}",
        "exact_integer",
        "results/revision_v2/full_run_001/final_evaluations.csv",
        "len(final_evaluations.csv) == 122 selections * 20 fresh seeds = 2,440",
        "external_holdout_evaluation",
        "20_fresh_seeds_per_selection",
        2440,
        "exact_count",
        "RECONSTRUCTED",
        docs_all,
    )

    # Revision v2 Optimizer Benchmark Metrics
    for opt_key, m in ds.prospective_optimizers.items():
        src_json = (
            "results/revision_v2/full_run_001/doe_selection_summary.json"
            if "doe" in opt_key
            else "results/revision_v2/full_run_001/optimizer_summary.json"
        )
        add_metric(
            f"rev_v2.{opt_key}.val_rmse_mean",
            m.val_rmse_mean,
            f"{m.val_rmse_mean:.5f}",
            "5dp (4dp in compact table)",
            "results/revision_v2/full_run_001/final_evaluations.csv",
            f"Reconstructed from 20 selections x 20 seeds and cross-checked with {src_json}",
            "development_validation_rmse",
            "20_search_replicates_x_20_eval_seeds",
            m.n_replicates,
            f"between-search SD={m.val_rmse_sd:.5f} across N={m.n_replicates} replicates",
            "RECONSTRUCTED",
            docs_bm,
        )
        add_metric(
            f"rev_v2.{opt_key}.test_rmse_mean",
            m.test_rmse_mean,
            f"{m.test_rmse_mean:.5f}",
            "5dp (4dp in compact table)",
            "results/revision_v2/full_run_001/final_evaluations.csv",
            f"Reconstructed from 20 selections x 20 seeds and cross-checked with {src_json}",
            "external_holdout_test_rmse",
            "20_search_replicates_x_20_eval_seeds",
            m.n_replicates,
            f"between-search SD={m.test_rmse_sd:.5f}, 95% t-CI=[{m.test_rmse_ci_low:.5f}, {m.test_rmse_ci_high:.5f}], conditional retrain sigma_eval={m.retrain_sd:.5f}",
            "RECONSTRUCTED",
            docs_bm,
        )
        add_metric(
            f"rev_v2.{opt_key}.predict_latency_mean",
            m.predict_latency_mean,
            f"{m.predict_latency_mean:.2f}",
            "2dp (1dp in compact table)",
            "results/revision_v2/full_run_001/latency_measurement.json",
            f"Reconstructed from 5 sessions x 1,000 calls per selection and cross-checked with {src_json}",
            "single_sample_predict_latency_us",
            "single_sample_latency_primary_v1",
            m.n_replicates,
            f"between-search SD={m.predict_latency_sd:.2f} across N={m.n_replicates} replicates",
            "RECONSTRUCTED",
            docs_bm,
        )
        add_metric(
            f"rev_v2.{opt_key}.inplace_latency_mean",
            m.inplace_latency_mean,
            f"{m.inplace_latency_mean:.2f}",
            "2dp",
            "results/revision_v2/full_run_001/latency_measurement.json",
            f"Reconstructed from 5 sessions x 1,000 calls per selection and cross-checked with {src_json}",
            "single_sample_inplace_predict_latency_us",
            "single_sample_latency_primary_v1",
            m.n_replicates,
            f"between-search SD={m.inplace_latency_sd:.2f} across N={m.n_replicates} replicates",
            "RECONSTRUCTED",
            docs_bm,
        )
        add_metric(
            f"rev_v2.{opt_key}.benchmark_feasible_count",
            m.benchmark_feasible_count,
            f"{m.benchmark_feasible_count}/{m.n_replicates}",
            "exact_ratio",
            "results/revision_v2/full_run_001/final_summary.csv",
            f"sum(predict_latency_us <= 145.0) across N={m.n_replicates} replicates",
            "single_sample_predict_latency_us",
            "latency_constraint_145us_primary_v1",
            m.n_replicates,
            f"{m.benchmark_feasible_pct}% benchmark feasibility (search-time: {m.search_feasible_count}/{m.n_replicates})",
            "RECONSTRUCTED",
            docs_bm,
        )

    # Hypothesis tests & Hypervolume
    add_metric(
        "rev_v2.comparison.doe_mo_vs_motpe_test_rmse_diff",
        ds.doe_mo_vs_motpe_diff,
        f"{ds.doe_mo_vs_motpe_diff:+.5f}",
        "5dp",
        "results/revision_v2/full_run_001/paired_comparisons.json",
        "mean(doe_mo_test_rmse) - mean(mo_tpe_test_rmse) reconstructed from final_summary.csv",
        "external_holdout_test_rmse",
        "Welch_and_paired_t_tests_N20",
        20,
        f"Welch t={ds.doe_mo_vs_motpe_welch_t:.4f} (p={ds.doe_mo_vs_motpe_welch_p:.4f}), paired t={ds.doe_mo_vs_motpe_paired_t:.4f} (p={ds.doe_mo_vs_motpe_paired_p:.4f})",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "rev_v2.hv.motpe_dev_060_mean",
        ds.hv_summary["motpe_dev_060_mean"],
        f"{ds.hv_summary['motpe_dev_060_mean']:.4f}",
        "4dp",
        "results/revision_v2/full_run_001/optimizer_trials.csv",
        "2D hypervolume at [0.60, 250.0] across 20 MO-TPE replicates (140 trials each) on split 42, verified against hypervolume.json",
        "development_split_42_online_30call",
        "single_sample_latency_online_search_v1",
        20,
        f"mean +/- SD ({ds.hv_summary['motpe_dev_060_sd']:.4f}) across 20 search replicates",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "rev_v2.hv.rdoe_dev_060_mean",
        ds.hv_summary["rdoe_dev_060_mean"],
        f"{ds.hv_summary['rdoe_dev_060_mean']:.4f}",
        "4dp",
        "results/revision_v2/full_run_001/doe_selection_runs.csv",
        "2D hypervolume at [0.60, 250.0] across 20 Repeated DOE 5-block-mean candidate fronts, verified against hypervolume.json",
        "development_5block_means_online_30call",
        "single_sample_latency_online_search_v1",
        20,
        f"mean +/- SD ({ds.hv_summary['rdoe_dev_060_sd']:.4f}) across 20 search replicates",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "rev_v2.hv.full_doe_dev_060",
        ds.hv_summary["fdoe_dev_060"],
        f"{ds.hv_summary['fdoe_dev_060']:.4f}",
        "4dp",
        "results/revision_v2/full_run_001/doe_matched_candidate_front.csv",
        "2D hypervolume at [0.60, 250.0] of 27 fixed DOE candidates on split 42, verified against hypervolume.json",
        "development_split_42_online_30call",
        "single_sample_latency_online_search_v1",
        1,
        "fixed_27_point_candidate_set",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "rev_v2.hv.holdout_frozen_060",
        ds.hv_summary["holdout_frozen_060"],
        f"{ds.hv_summary['holdout_frozen_060']:.4f}",
        "4dp",
        "results/revision_v2/full_run_001/final_summary.csv",
        "2D hypervolume at [0.60, 250.0] across 122 frozen selections on external holdout test set, verified against hypervolume.json",
        "external_holdout_test_domain",
        "single_sample_latency_primary_v1",
        122,
        f"{ds.hv_summary['holdout_nd']} non-dominated points (DOE MO={ds.holdout_nondominated_by_optimizer['repeated_preplanned_doe_multi_objective']}, MO-TPE={ds.holdout_nondominated_by_optimizer['multi_objective_tpe']}, cTPE={ds.holdout_nondominated_by_optimizer['constrained_tpe']}, SO-TPE={ds.holdout_nondominated_by_optimizer['single_objective_tpe']})",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "rev_v2.hv.holdout_bootstrap_seed_060_mean",
        ds.holdout_pareto_bootstrap["seed_only_060_mean"],
        f"{ds.holdout_pareto_bootstrap['seed_only_060_mean']:.4f}",
        "4dp",
        "results/revision_v2/full_run_001/final_evaluations.csv",
        "Paired cluster bootstrap (B=2000, seed=20261011) over 20 shared evaluation seeds at [0.60, 250.0]",
        "external_holdout_test_domain",
        "cluster_bootstrap_20_eval_seeds",
        20,
        f"SD={ds.holdout_pareto_bootstrap['seed_only_060_sd']:.4f}, 95% CI=[{ds.holdout_pareto_bootstrap['seed_only_060_ci_low']:.4f}, {ds.holdout_pareto_bootstrap['seed_only_060_ci_high']:.4f}]",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "rev_v2.hv.holdout_bootstrap_joint_060_mean",
        ds.holdout_pareto_bootstrap["joint_060_mean"],
        f"{ds.holdout_pareto_bootstrap['joint_060_mean']:.4f}",
        "4dp",
        "results/revision_v2/full_run_001/final_evaluations.csv",
        "Joint paired bootstrap (B=2000, seed=20261011) over 20 evaluation seeds and 5 latency sessions at [0.60, 250.0]",
        "external_holdout_test_domain",
        "joint_cluster_bootstrap_seeds_and_sessions",
        20,
        f"SD={ds.holdout_pareto_bootstrap['joint_060_sd']:.4f}, 95% CI=[{ds.holdout_pareto_bootstrap['joint_060_ci_low']:.4f}, {ds.holdout_pareto_bootstrap['joint_060_ci_high']:.4f}]",
        "RECONSTRUCTED",
        docs_hv,
    )
    add_metric(
        "phase1.curvature_f_block_adj_pooled",
        ds.curvature_sensitivity["f_block_adj_pooled"],
        f"{ds.curvature_sensitivity['f_block_adj_pooled']:,.2f}",
        "2dp",
        "results/runs.csv",
        "SS_curv / MS_block_adj_pooled (79 df = 64 df TxB interaction + 15 df center PE)",
        "development_validation_rmse",
        "RCBD_phase1_block_adjusted_pooled_error",
        100,
        "F_distribution_1_79_df",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "phase1.curvature_f_block_paired",
        ds.curvature_sensitivity["f_block_paired"],
        f"{ds.curvature_sensitivity['f_block_paired']:,.2f}",
        "2dp",
        "results/runs.csv",
        "Block-paired curvature contrast across b=5 blocks (t^2 with 4 df)",
        "development_validation_rmse",
        "RCBD_phase1_block_paired_contrast",
        5,
        f"95% CI=[+{ds.curvature_sensitivity['ci95_block_paired_low']:.4f}, +{ds.curvature_sensitivity['ci95_block_paired_high']:.4f}] RMSE",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "doe.low_eta_run_count",
        ds.doe_low_eta_run_count,
        f"{ds.doe_low_eta_run_count}/140 ({ds.doe_low_eta_run_pct:.2f}%)",
        "exact_ratio_and_2dp_pct",
        "results/runs.csv",
        "sum(x1 == -1.0) across N=140 pre-planned FCCD runs",
        "development_validation_rmse",
        "preplanned_fccd_budget_allocation",
        140,
        f"mean val_rmse={ds.doe_low_eta_mean_rmse:.4f} vs {ds.doe_high_eta_mean_rmse:.4f} for x1 > -1.0",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "latency.hist_runs_interface_overhead_mean",
        ds.hist_runs_interface_overhead_mean,
        f"{ds.hist_runs_interface_overhead_mean:.2f}",
        "2dp",
        "results/runs.csv",
        "mean(latency_us_median - inplace_latency_us_median) across N=140 historical DOE runs",
        "single_sample_interface_overhead_us",
        "historical_doe_1000_calls",
        140,
        f"SD={ds.hist_runs_interface_overhead_sd:.2f}, range=[{ds.hist_runs_interface_overhead_min:.2f}, {ds.hist_runs_interface_overhead_max:.2f}] us",
        "RECONSTRUCTED",
        docs_all,
    )
    add_metric(
        "rev_v2.total_warmup_inferences",
        ds.total_warmup_inferences,
        f"{ds.total_warmup_inferences:,}",
        "exact_integer",
        "results/revision_v2/full_run_001/computational_budget.json",
        "14027 * 10 + 610 * 100 warmup prediction calls",
        "campaign_computational_budget",
        "full_run_001",
        201270,
        "exact_count",
        "RECONSTRUCTED",
        docs_all,
    )


    return {
        "schema_version": "1.0.0",
        "canonical_experiment_dir": "results/revision_v2/full_run_001",
        "historical_baseline_tag": "v1.0.0",
        "summary_counts": {
            "total_metrics_tracked": len(metrics),
            "reconstructed_count": sum(1 for m in metrics if m["verification_status"] == "RECONSTRUCTED"),
            "source_traced_count": sum(1 for m in metrics if m["verification_status"] == "SOURCE-TRACED"),
        },
        "metrics": metrics,
    }
