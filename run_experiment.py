"""run_experiment.py - Root entry point delegating to experiment_code/run_experiment.py.

Usage:
    python run_experiment.py --stage smoke
    python run_experiment.py --stage all --dataset full_imdb
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiment_code"))
from run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main())
