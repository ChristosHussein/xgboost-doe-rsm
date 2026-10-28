"""Capture a hash-addressed snapshot of published research artifacts.

The snapshot reads tracked files from an explicit Git revision, so dirty working
files can never be mistaken for the historical publication baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATHS = ("results", "data", "tables", "figures")
TOP_LEVEL_ARTIFACTS = (
    "config.yaml",
    "report.tex",
    "report.pdf",
    "REPORT.md",
    "results_summary.json",
)

PRODUCERS = {
    "results/runs.csv": {
        "producer": "pipeline.py:execute_design_pipeline",
        "inputs": ["config.yaml", "sklearn California Housing dataset"],
        "classification": "historical pre-planned DOE raw log",
    },
    "results/phase1.json|phase3.json|lof.json|diagnostics.json|icc.json|depth_opt_table.csv|ridge_table.csv": {
        "producer": "analysis.py",
        "inputs": ["results/runs.csv", "config.yaml"],
        "classification": "historical reanalysis",
    },
    "results/confirmation*.csv|confirmation.json": {
        "producer": "scripts/run_confirmation.py",
        "inputs": ["results/runs.csv", "results/phase3.json", "config.yaml"],
        "classification": "historical confirmation experiment",
    },
    "results/latency_*|anova_ccd_latency.csv": {
        "producer": "scripts/measure_latency.py and scripts/fit_latency_models.py",
        "inputs": ["results/runs.csv", "config.yaml"],
        "classification": "historical latency reanalysis",
    },
    "results/benchmark*|desirability_sensitivity.csv": {
        "producer": "scripts/run_benchmarks.py",
        "inputs": ["results/phase3.json", "results/runs.csv", "config.yaml"],
        "classification": "historical optimizer benchmark",
    },
    "results/macros.tex|tables/tab_*.tex": {
        "producer": "scripts/generate_report_artifacts.py",
        "inputs": ["versioned results artifacts"],
        "classification": "generated manuscript input",
    },
    "figures/*.png": {
        "producer": "plots.py",
        "inputs": ["versioned results artifacts"],
        "classification": "generated manuscript figure",
    },
    "report.pdf": {
        "producer": "LaTeX compilation of report.tex",
        "inputs": ["report.tex", "results/macros.tex", "tables/", "figures/"],
        "classification": "published manuscript artifact",
    },
    "data/*|results_summary.json": {
        "producer": "legacy run_pipeline.py workflow",
        "inputs": ["config.yaml and legacy pipeline interfaces"],
        "classification": "legacy historical output; not authoritative for report.tex",
    },
}


def git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True,
        text=not binary, encoding=None if binary else "utf-8",
    )
    return result.stdout if binary else result.stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def revision_files(revision: str) -> list[str]:
    names = git("ls-tree", "-r", "--name-only", revision).splitlines()
    return sorted(
        path for path in names
        if path in TOP_LEVEL_ARTIFACTS
        or any(path == root or path.startswith(f"{root}/") for root in SNAPSHOT_PATHS)
    )


def installed_versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "scikit-learn": "scikit-learn", "statsmodels": "statsmodels",
        "xgboost": "xgboost", "optuna": "optuna", "PyYAML": "PyYAML",
        "pytest": "pytest",
    }
    versions = {}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "NOT INSTALLED"
    return versions


def build_manifest(revision: str) -> dict:
    commit = git("rev-parse", f"{revision}^{{}}")
    config = yaml.safe_load(git("show", f"{revision}:config.yaml"))
    files = []
    for relative in revision_files(revision):
        committed = git("show", f"{revision}:{relative}", binary=True)
        working_path = ROOT / relative
        working_hash = sha256(working_path.read_bytes()) if working_path.is_file() else None
        committed_hash = sha256(committed)
        files.append({
            "path": relative,
            "bytes": len(committed),
            "sha256": committed_hash,
            "working_tree_sha256": working_hash,
            "working_tree_matches_snapshot": working_hash == committed_hash,
        })

    return {
        "schema_version": 1,
        "snapshot_kind": "historical_release_baseline",
        "revision_requested": revision,
        "commit": commit,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_branch": git("branch", "--show-current"),
        "working_tree_status": git("status", "--porcelain=v1").splitlines(),
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "libraries": installed_versions(),
        },
        "configuration": {
            "sha256": sha256(git("show", f"{revision}:config.yaml", binary=True)),
            "seeds": config.get("seeds", {}),
            "factors": config.get("factors", {}),
            "model": config.get("model", {}),
            "budget": config.get("budget", {}),
        },
        "artifact_producers": PRODUCERS,
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True, help="Immutable tag or commit to snapshot")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.revision)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Captured {len(manifest['files'])} files from {manifest['commit']} -> {output}")


if __name__ == "__main__":
    main()
