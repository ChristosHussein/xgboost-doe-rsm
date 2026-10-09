"""Reproduce the guarded revision-v2 workflow without touching v1 artifacts.

The historical all-in-place pipeline is preserved at tag ``v1.0.0``. It is not
run from this revision because it overwrites the publication snapshot and makes
the holdout available during design evaluation. Every new experiment instead
writes a fresh directory under ``results/revision_v2`` (or ``--output-dir``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


def run_step(command: Sequence[str], description: str) -> None:
    """Run one argv-safe reproduction step and fail on its first error."""
    print(f"\n[Step] {description}...")
    started = time.perf_counter()
    subprocess.run(list(command), check=True)
    print(f"[Done] {description} in {time.perf_counter() - started:.1f}s.")


def build_commands(
    *,
    mode: str,
    output_dir: str | None,
    confirm_full_budget: bool,
    skip_tests: bool,
) -> list[tuple[list[str], str]]:
    if mode == "full" and not confirm_full_budget:
        raise ValueError("full reproduction requires --confirm-full-budget")
    commands: list[tuple[list[str], str]] = []
    if not skip_tests:
        commands.append(
            ([sys.executable, "-m", "pytest", "-q"], "software verification suite")
        )
    benchmark = [sys.executable, "scripts/run_benchmarks.py", "--mode", mode]
    if output_dir is not None:
        benchmark.extend(["--output-dir", output_dir])
    if mode == "full":
        benchmark.append("--confirm-full-budget")
    commands.append((benchmark, f"revision-v2 {mode} experiment"))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tests and a fresh, versioned revision-v2 experiment."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument(
        "--confirm-full-budget",
        action="store_true",
        help="Required for the expensive full protocol.",
    )
    args = parser.parse_args()
    if args.output_dir and Path(args.output_dir).exists() and any(Path(args.output_dir).iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty directory: {args.output_dir}")
    try:
        commands = build_commands(
            mode=args.mode,
            output_dir=args.output_dir,
            confirm_full_budget=args.confirm_full_budget,
            skip_tests=args.skip_tests,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for command, description in commands:
        run_step(command, description)
    print(
        "\nRevision artifacts are complete for this run. Publication tables and PDF remain "
        "separate until a full protocol has completed and passed numerical-consistency review."
    )


if __name__ == "__main__":
    main()
