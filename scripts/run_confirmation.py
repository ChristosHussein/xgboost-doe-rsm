"""Gated final evaluation for already-frozen configuration selections.

The historical v1 confirmation implementation remains available at tag
``v1.0.0``. This revision entry point cannot construct or alter selections and
does not expose the external holdout until a tamper-evident finalized-selection
manifest has been supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_evaluation import FinalTestEvaluator, load_finalized_selection


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _validate_evaluation_payload(
    payload: dict[str, Any],
    *,
    selection_id: str,
    expected_config_sha256: str,
    manifest_git_revision: str,
    evaluation_seeds: list[int],
) -> None:
    if payload.get("selection_id") != selection_id:
        raise ValueError("evaluator returned a different selection_id")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("evaluator returned a configuration hash not frozen in the manifest")
    if payload.get("manifest_git_revision") != manifest_git_revision:
        raise ValueError("evaluator returned a different manifest Git revision")
    rows = payload.get("per_seed")
    if not isinstance(rows, list):
        raise ValueError("evaluator payload lacks a per_seed failure ledger")
    returned_seeds = [row.get("evaluation_seed") for row in rows]
    if returned_seeds != evaluation_seeds:
        raise ValueError("evaluator seed coverage differs from the frozen evaluation seeds")
    if any(row.get("status") not in {"completed", "failed"} for row in rows):
        raise ValueError("every evaluator row must have completed or failed status")
    completed = sum(row["status"] == "completed" for row in rows)
    expected_status = (
        "completed"
        if completed == len(rows)
        else ("failed" if completed == 0 else "partial")
    )
    if payload.get("status") != expected_status:
        raise ValueError("evaluator aggregate status disagrees with its per-seed ledger")


def _failed_evaluation(
    *,
    selection_id: str,
    config_sha256: str,
    manifest_git_revision: str,
    evaluation_seeds: list[int],
    error: Exception,
) -> dict[str, Any]:
    message = f"{type(error).__name__}: {error}"
    return {
        "selection_id": selection_id,
        "config_sha256": config_sha256,
        "manifest_git_revision": manifest_git_revision,
        "status": "failed",
        "per_seed": [
            {
                "evaluation_seed": seed,
                "status": "failed",
                "val_rmse": None,
                "test_rmse": None,
                "error": message,
            }
            for seed in evaluation_seeds
        ],
    }


def _write_atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def execute_confirmation(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    evaluation_seeds: Iterable[int] | None = None,
    selection_ids: Iterable[str] | None = None,
    evaluator: Any | None = None,
) -> dict[str, Any]:
    """Evaluate only configurations already frozen in ``manifest_path``."""
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite final-evaluation artifact: {output_path}")

    manifest = load_finalized_selection(manifest_path)
    frozen_ids = [item["selection_id"] for item in manifest["configurations"]]
    requested_ids = list(selection_ids) if selection_ids is not None else frozen_ids
    unknown = sorted(set(requested_ids) - set(frozen_ids))
    if unknown:
        raise KeyError(f"Selections were not frozen in the manifest: {unknown}")
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("selection_ids must not contain duplicates")

    frozen_evaluation = manifest["final_evaluation"]
    frozen_seeds = [int(seed) for seed in frozen_evaluation["evaluation_seeds"]]
    if evaluation_seeds is not None:
        requested_seeds = [int(seed) for seed in evaluation_seeds]
        if requested_seeds != frozen_seeds:
            raise ValueError(
                "Requested evaluation seeds do not exactly match the frozen manifest"
            )
    seeds = frozen_seeds

    final_evaluator = evaluator or FinalTestEvaluator()
    expected_by_id = {
        item["selection_id"]: item for item in manifest["configurations"]
    }
    evaluations = []
    for selection_id in requested_ids:
        configuration = expected_by_id[selection_id]
        try:
            if not hasattr(final_evaluator, "evaluation_protocol"):
                raise TypeError("evaluator does not expose evaluation_protocol")
            if final_evaluator.evaluation_protocol != frozen_evaluation["protocol"]:
                raise ValueError("evaluator protocol does not match the frozen manifest")
            result = final_evaluator.evaluate(
                manifest_path,
                selection_id,
                evaluation_seeds=seeds,
            )
            _validate_evaluation_payload(
                result,
                selection_id=selection_id,
                expected_config_sha256=configuration["config_sha256"],
                manifest_git_revision=manifest["git_revision"],
                evaluation_seeds=seeds,
            )
        except Exception as exc:
            result = _failed_evaluation(
                selection_id=selection_id,
                config_sha256=configuration["config_sha256"],
                manifest_git_revision=manifest["git_revision"],
                evaluation_seeds=seeds,
                error=exc,
            )
        evaluations.append(result)
    completed = sum(item["status"] == "completed" for item in evaluations)
    artifact = {
        "schema_version": 2,
        "classification": "gated_final_evaluation",
        "status": (
            "completed"
            if completed == len(evaluations)
            else ("failed" if completed == 0 else "partial")
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "finalized_selection_manifest": str(manifest_path),
        "finalized_selection_manifest_sha256": _sha256(manifest_path),
        "manifest_git_revision": manifest["git_revision"],
        "evaluation_code_git_revision": _git_revision(),
        "selection_ids": requested_ids,
        "evaluation_seeds": seeds,
        "evaluation_protocol": frozen_evaluation["protocol"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "holdout_scope": (
            "External holdout metrics are emitted only after manifest validation; "
            "they are not returned to development-stage search code."
        ),
        "evaluations": evaluations,
    }
    _write_atomic_json(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a tamper-evident finalized-selection manifest on the holdout."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--selection-id",
        action="append",
        dest="selection_ids",
        help="Frozen selection to evaluate; repeat as needed. Defaults to every frozen selection.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="evaluation_seeds",
        help="Evaluation seed; repeat as needed. Defaults to the configured full seed set.",
    )
    args = parser.parse_args()
    execute_confirmation(
        args.manifest,
        args.output,
        evaluation_seeds=args.evaluation_seeds,
        selection_ids=args.selection_ids,
    )


if __name__ == "__main__":
    main()
