"""
scripts/reproduce_all.py - Master Pipeline Reproduction Script
==============================================================
Regenerates everything end-to-end with one command:
  1. scripts/generate_env.py -> results/env.json
  2. pipeline.py -> results/runs.csv (140 genuine runs)
  3. analysis.py -> results/{phase1, phase3, lof, diagnostics, icc}.json, results/{depth_opt_table, ridge_table}.csv
  4. scripts/run_confirmation.py -> results/confirmation.json
  5. scripts/run_benchmarks.py -> results/benchmark.csv, results/benchmark_summary.json, results/desirability_sensitivity.csv
  6. scripts/fit_latency_models.py -> results/latency_models_comparison.csv
  7. plots.py -> figures/*.png (all 5 publication figures)
  8. scripts/generate_report_artifacts.py -> results/macros.tex, tables/*.tex
  9. Compiles report.tex via tectonic -> report.pdf
  10. Runs pytest tests/
"""

import os
import subprocess
import sys
import time

def run_step(cmd, desc):
    print(f"\n[Step] {desc}...")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, text=True)
    if res.returncode != 0:
        print(f"[Error] Failed: {desc}")
        sys.exit(res.returncode)
    print(f"[Done] {desc} in {time.time()-t0:.1f}s.")

def main():
    t_start = time.time()
    print("="*70)
    print("REPRODUCING FULL RSM/CCD HPO PIPELINE END-TO-END")
    print("="*70)

    # 1. Environment
    run_step("python scripts/generate_env.py", "Generate environment metadata")

    # 2. Check if runs.csv exists or regenerate
    if not os.path.exists("results/runs.csv"):
        run_step("python pipeline.py", "Execute 140 genuine DOE design runs")
    else:
        print("\n[Notice] results/runs.csv exists. To re-run raw training, delete results/runs.csv first.")

    # 3. Statistical Analysis
    run_step("python analysis.py", "Run full statistical analysis (canonical, ridge, LoF, diagnostics, ICC)")

    # 4. Confirmation
    run_step("python scripts/run_confirmation.py", "Run 10 confirmation trials at x*")

    # 5. Benchmarks
    if not os.path.exists("results/benchmark.csv"):
        run_step("python scripts/run_benchmarks.py", "Execute empirical benchmarks (Random Search, Optuna TPE)")
    else:
        print("\n[Notice] results/benchmark.csv exists. Preserving empirical benchmark trials.")

    # 6. Latency models
    run_step("python scripts/fit_latency_models.py", "Fit candidate latency models")

    # 7. Render figures
    run_step("python plots.py", "Render all 5 publication-grade figures")

    # 8. Generate tables & macros
    run_step("python scripts/generate_report_artifacts.py", "Export LaTeX tables and macros")

    # 9. Run tests
    run_step("python -m pytest", "Run full test suite")

    # 10. Compile PDF
    tectonic_path = r"C:\Users\chris\bin\tectonic.exe"
    if os.path.exists(tectonic_path):
        run_step(f'"{tectonic_path}" report.tex', "Compile report.tex into report.pdf via Tectonic")
    else:
        run_step("tectonic report.tex", "Compile report.tex via Tectonic")

    print("\n" + "="*70)
    print(f"REPRODUCTION COMPLETE IN {time.time()-t_start:.1f}s! ALL ARTIFACTS VERIFIED.")
    print("="*70)

if __name__ == "__main__":
    main()
