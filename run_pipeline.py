"""Compatibility entry point for the revision-v2 scientific workflow.

The obsolete implementation formerly stored here used removed APIs and exposed
the external holdout during design evaluation. Its exact source and outputs are
preserved at tag ``v1.0.0``. New runs are delegated to the versioned benchmark
workflow, which separates development search from manifest-gated final testing.
"""

from scripts.run_benchmarks import main


if __name__ == "__main__":
    main()
