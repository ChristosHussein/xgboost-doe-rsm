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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from final_evaluation import FinalTestEvaluator, load_finalized_selection
from pipeline import CONFIG


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    seeds = [
        int(seed)
        for seed in (
            evaluation_seeds
            if evaluation_seeds is not None
            else CONFIG["revision_v2"]["modes"]["full"]["evaluation_seeds"]
        )
    ]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation_seeds must be a non-empty sequence of unique integers")

    final_evaluator = evaluator or FinalTestEvaluator()
    evaluations = [
        final_evaluator.evaluate(
            manifest_path,
            selection_id,
            evaluation_seeds=seeds,
        )
        for selection_id in requested_ids
    ]
    artifact = {
        "schema_version": 1,
        "classification": "gated_final_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "finalized_selection_manifest": str(manifest_path),
        "finalized_selection_manifest_sha256": _sha256(manifest_path),
        "manifest_git_revision": manifest["git_revision"],
        "selection_ids": requested_ids,
        "evaluation_seeds": seeds,
        "holdout_scope": (
            "External holdout metrics are emitted only after manifest validation; "
            "they are not returned to development-stage search code."
        ),
        "evaluations": evaluations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
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
