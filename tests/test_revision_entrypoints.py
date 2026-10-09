import sys
from pathlib import Path

import pytest


def test_reproduction_commands_use_argument_vectors_and_versioned_runner():
    from scripts.reproduce_all import REPOSITORY_ROOT, build_commands

    commands = build_commands(
        mode="smoke",
        output_dir="results/revision_v2/smoke/example",
        confirm_full_budget=False,
        skip_tests=False,
    )
    assert commands[0][0] == [sys.executable, "-m", "pytest", "-q"]
    assert commands[1][0] == [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts" / "run_benchmarks.py"),
        "--mode",
        "smoke",
        "--output-dir",
        str((REPOSITORY_ROOT / "results/revision_v2/smoke/example").resolve()),
    ]
    assert all(isinstance(command, list) for command, _ in commands)


def test_full_reproduction_requires_budget_acknowledgement():
    from scripts.reproduce_all import build_commands

    with pytest.raises(ValueError, match="confirm-full-budget"):
        build_commands(
            mode="full",
            output_dir=None,
            confirm_full_budget=False,
            skip_tests=True,
        )


def test_legacy_top_level_pipeline_delegates_to_guarded_revision_runner():
    source = open("run_pipeline.py", encoding="utf-8").read()
    assert "from scripts.run_benchmarks import main" in source
    assert "evaluate_model" not in source
    assert "DataManager" not in source


def test_reproduction_steps_always_run_from_repository_root(monkeypatch):
    from scripts import reproduce_all

    captured = {}

    def fake_run(command, *, check, cwd):
        captured.update(command=command, check=check, cwd=cwd)

    monkeypatch.setattr(reproduce_all.subprocess, "run", fake_run)
    reproduce_all.run_step([sys.executable, "-c", "pass"], "fixture")
    assert captured["check"] is True
    assert captured["cwd"] == Path(reproduce_all.__file__).resolve().parents[1]
