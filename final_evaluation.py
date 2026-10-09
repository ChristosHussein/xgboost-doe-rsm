"""Guarded external-holdout evaluation for frozen configurations only."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


REQUIRED_HYPERPARAMETERS = {
    "learning_rate", "max_depth", "subsample", "reg_lambda"
}


def canonical_config_hash(configuration: dict[str, Any]) -> str:
    payload = {
        "hyperparameters": configuration["hyperparameters"],
        "model": configuration["model"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True, encoding="utf-8",
    ).stdout.strip()


def _validate_configuration(configuration: dict[str, Any]) -> None:
    missing = REQUIRED_HYPERPARAMETERS - set(configuration.get("hyperparameters", {}))
    if missing:
        raise ValueError(f"Missing hyperparameters: {sorted(missing)}")
    if not configuration.get("selection_id"):
        raise ValueError("Every frozen configuration requires a selection_id")
    expected = canonical_config_hash(configuration)
    if configuration.get("config_sha256") != expected:
        raise ValueError(
            f"Configuration hash mismatch for {configuration['selection_id']}: "
            f"expected {expected}"
        )


def create_finalized_selection(
    path: str | Path,
    *,
    configurations: list[dict[str, Any]],
    selection_policy: str,
    development_seeds: list[int],
    source_artifact: str,
    git_revision: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Freeze development-selected configurations before any holdout evaluation."""
    frozen = []
    for configuration in configurations:
        item = json.loads(json.dumps(configuration, allow_nan=False))
        item["config_sha256"] = canonical_config_hash(item)
        frozen.append(item)

    manifest = {
        "schema_version": 1,
        "status": "finalized",
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision or _git_revision(),
        "selection_policy": selection_policy,
        "development_seeds": list(development_seeds),
        "source_artifact": source_artifact,
        "configurations": frozen,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_finalized_selection(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("status") != "finalized":
        raise ValueError("External holdout evaluation requires a finalized selection manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported finalized-selection schema")
    configurations = manifest.get("configurations", [])
    if not configurations:
        raise ValueError("Finalized selection contains no configurations")
    identifiers = [item.get("selection_id") for item in configurations]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Finalized selection contains duplicate selection IDs")
    for configuration in configurations:
        _validate_configuration(configuration)
    return manifest


class FinalTestEvaluator:
    """Own the external holdout and evaluate only manifest-frozen selections."""

    def __init__(
        self,
        *,
        external_test_seed: int = 42,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ):
        if X is None or y is None:
            housing = fetch_california_housing()
            X, y = housing.data, housing.target
        self._X_dev, self._X_test, self._y_dev, self._y_test = train_test_split(
            np.asarray(X), np.asarray(y), test_size=0.20, random_state=external_test_seed
        )

    def evaluate(
        self,
        manifest_path: str | Path,
        selection_id: str,
        *,
        evaluation_seeds: list[int],
    ) -> dict[str, Any]:
        manifest = load_finalized_selection(manifest_path)
        matches = [
            item for item in manifest["configurations"]
            if item["selection_id"] == selection_id
        ]
        if not matches:
            raise KeyError(f"Selection {selection_id!r} was not frozen in the manifest")
        configuration = matches[0]
        hp = configuration["hyperparameters"]
        model_config = configuration["model"]

        rows = []
        for seed in evaluation_seeds:
            X_train, X_val, y_train, y_val = train_test_split(
                self._X_dev, self._y_dev, test_size=0.25, random_state=seed
            )
            estimator_args = dict(
                n_estimators=model_config["n_estimators"],
                learning_rate=hp["learning_rate"],
                max_depth=int(hp["max_depth"]),
                subsample=hp["subsample"],
                reg_lambda=hp["reg_lambda"],
                random_state=seed,
                n_jobs=model_config["n_jobs_train"],
                objective=model_config["objective"],
            )
            for optional_name in (
                "colsample_bytree", "min_child_weight", "gamma", "tree_method"
            ):
                if optional_name in model_config:
                    estimator_args[optional_name] = model_config[optional_name]
            model = xgb.XGBRegressor(**estimator_args)
            model.fit(X_train, y_train)
            val_rmse = float(np.sqrt(np.mean((y_val - model.predict(X_val)) ** 2)))
            test_rmse = float(np.sqrt(np.mean((self._y_test - model.predict(self._X_test)) ** 2)))
            rows.append({"evaluation_seed": seed, "val_rmse": val_rmse, "test_rmse": test_rmse})

        return {
            "selection_id": selection_id,
            "config_sha256": configuration["config_sha256"],
            "manifest_git_revision": manifest["git_revision"],
            "per_seed": rows,
        }
