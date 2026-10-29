"""run_experiment.py - Master Unified Experiment Runner for Phases A & B.

Project:
Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees
under Stochastic Nuisance Blocking via Response Surface Methodology and Explainable AI.

Procedures:
- Phase A: Continuous Tabular Response Surface Optimization under Stochastic Nuisance Blocking
           (Benchmark: California Housing, 4-factor continuous design, 20% isolated holdout).
- Phase B: High-Dimensional Sequential / Adaptive Central Composite Design with Explainable AI
           (Benchmark: IMDb Sentiment Analysis, sparse TF-IDF, capacity calibration, ANOVA-SHAP).

Usage:
    python run_experiment.py --smoke                   # Run fast end-to-end smoke verification (<25s)
    python run_experiment.py --phase a --mode smoke    # Run Phase A smoke benchmark
    python run_experiment.py --phase a --mode full     # Run Phase A full replication campaign
    python run_experiment.py --phase b --stage smoke   # Run Phase B synthetic smoke study
    python run_experiment.py --phase b --stage all     # Run Phase B complete pipeline on full IMDb
    python run_experiment.py --phase all               # Run both Phase A and Phase B end-to-end
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PHASE_A_DIR = ROOT / "phase_a_rsm_blocking"
PHASE_B_DIR = ROOT / "phase_b_sequential_xai"
FINDINGS_DIR = ROOT / "findings"


def run_phase_a(mode: str = "smoke", output_dir: Path | None = None) -> int:
    """Execute Phase A: Continuous Tabular Response Surface Optimization under Nuisance Blocking."""
    print("\n" + "=" * 80)
    print("PHASE A: CONTINUOUS TABULAR RSM & STOCHASTIC NUISANCE BLOCKING")
    print("=" * 80)
    runner_script = PHASE_A_DIR / "scripts" / "run_benchmarks.py"
    if not runner_script.exists():
        raise FileNotFoundError(f"Phase A runner not found at: {runner_script}")

    out = output_dir or (PHASE_A_DIR / "results" / f"run_{mode}")
    if mode == "smoke" and out.exists():
        import shutil
        shutil.rmtree(out, ignore_errors=True)
    cmd = [
        sys.executable,
        str(runner_script),
        "--mode", mode,
        "--output-dir", str(out),
    ]
    if mode == "full":
        cmd.append("--confirm-full-budget")

    print(f"[Phase A] Executing: {' '.join(cmd)}")
    env = dict(os.environ, PYTHONPATH=str(PHASE_A_DIR))
    ret = subprocess.run(cmd, cwd=str(PHASE_A_DIR), env=env)
    if ret.returncode != 0:
        print(f"[Phase A] Execution failed with return code {ret.returncode}", file=sys.stderr)
        return ret.returncode

    print(f"[Phase A] Successfully completed in {mode} mode. Results saved in: {out}")
    return 0


def run_phase_b(
    stage: str = "all",
    dataset: str = "full_imdb",
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    findings_dir: Path | None = None,
    training_threads: int = 4,
    device: str = "auto",
) -> int:
    """Execute Phase B: Sequential / Adaptive CCD with Explainable AI & Warmed Latency Calibration."""
    print("\n" + "=" * 80)
    print("PHASE B: SEQUENTIAL ADAPTIVE CCD & EXPLAINABLE AI INTEGRATION")
    print("=" * 80)
    runner_script = PHASE_B_DIR / "run_phase_b.py"
    if not runner_script.exists():
        raise FileNotFoundError(f"Phase B runner not found at: {runner_script}")

    d_dir = data_dir or (PHASE_B_DIR / "data")
    out = output_dir or (ROOT / "results" / "phase_b")
    f_dir = findings_dir or (FINDINGS_DIR / "phase_b")

    cmd = [
        sys.executable,
        str(runner_script),
        "--stage", stage,
        "--dataset", dataset,
        "--data-dir", str(d_dir),
        "--output-dir", str(out),
        "--findings-dir", str(f_dir),
        "--training-threads", str(training_threads),
        "--device", str(device),
    ]

    print(f"[Phase B] Executing: {' '.join(cmd)}")
    env = dict(os.environ, PYTHONPATH=str(PHASE_B_DIR))
    ret = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if ret.returncode != 0:
        print(f"[Phase B] Execution failed with return code {ret.returncode}", file=sys.stderr)
        return ret.returncode

    print(f"[Phase B] Successfully completed stage '{stage}'. Findings saved in: {f_dir}")
    return 0


def run_smoke_verification() -> int:
    """Run fast end-to-end smoke verification across both Phase A and Phase B."""
    print("\n" + "=" * 80)
    print("MASTER SMOKE VERIFICATION: TESTING PHASES A & B END-TO-END")
    print("=" * 80)

    # 1. Phase A Smoke Mode
    ret_a = run_phase_a(mode="smoke", output_dir=PHASE_A_DIR / "results" / "smoke_verification")
    if ret_a != 0:
        return ret_a

    # 2. Phase B Smoke Mode
    smoke_dir = ROOT / 'results' / 'verification_smoke'
    ret_b = run_phase_b(stage="smoke", output_dir=smoke_dir,
                        findings_dir=smoke_dir / 'findings', device="cpu")
    if ret_b != 0:
        return ret_b

    print("\n" + "=" * 80)
    print("ALL SMOKE CHECKS PASSED: PHASES A & B VERIFIED END-TO-END")
    print("=" * 80 + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["a", "b", "all"],
        default="all",
        help="Experiment procedure phase to execute (a: Tabular RSM, b: Sequential XAI, all: both).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke verification across both phases without full dataset training.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Phase A execution mode (smoke: fast validation, full: 20-replicate research run).",
    )
    parser.add_argument(
        "--stage",
        choices=["prepare", "calibrate", "run", "words", "generate-findings", "smoke", "all"],
        default="all",
        help="Phase B execution stage (prepare, calibrate, run, generate-findings, smoke, all).",
    )
    parser.add_argument(
        "--dataset",
        choices=["full_imdb", "yelp"],
        default="full_imdb",
        help="Phase B benchmark corpus (full_imdb or yelp).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PHASE_B_DIR / "data",
        help="Path to dataset storage directory.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for Phase B XGBoost training (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--training-threads",
        type=int,
        default=4,
        help="Number of OpenMP worker threads for XGBoost model fitting.",
    )
    args = parser.parse_args()

    if args.smoke:
        return run_smoke_verification()

    if args.phase in ("a", "all"):
        ret = run_phase_a(mode=args.mode)
        if ret != 0:
            return ret

    if args.phase in ("b", "all"):
        ret = run_phase_b(
            stage=args.stage,
            dataset=args.dataset,
            data_dir=args.data_dir,
            training_threads=args.training_threads,
            device=args.device,
        )
        if ret != 0:
            return ret

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
