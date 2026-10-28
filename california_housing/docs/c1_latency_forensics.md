# Audit C: Forensics of Historical Latency Discrepancy

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Historical Baseline Tag:** `v1.0.0` (`cd63135d2b70a0ee324555c11b67a541b32b98a4`)  
**Audited Dataset:** `results/revision_v2/full_run_001/`  
**Date:** October 10, 2026  

---

## 1. Executive Summary & Definite Forensic Finding

The published historical manuscript (v1.0.0) reported noticeable latency discrepancies for identical hyperparameter configurations between Table 4 (Confirmation) and Table 5 (Benchmarks):
- **DOE Multi-Objective (Depth 4):** Confirmation reported **$142.33\,\mu\text{s}$**, whereas Benchmark reported **$119.84\,\mu\text{s}$** (a difference of $+22.49\,\mu\text{s}$).
- **DOE Single-Objective (Depth 7):** Confirmation reported **$171.12\,\mu\text{s}$**, whereas Benchmark reported **$147.84\,\mu\text{s}$** (a difference of $+23.28\,\mu\text{s}$).

During preliminary revision work, it was conjectured that this discrepancy might have arisen because one script reported `model.predict()` and the other reported `booster.inplace_predict()`, given that `predict` incurs roughly $32.19\,\mu\text{s}$ wrapper overhead over `inplace_predict`.

### Core Forensic Finding:
1. **Source Code Inspection Refutes the Interface Mismatch Hypothesis:** Direct inspection of the authoritative source code at Git tag `v1.0.0` (`pipeline.py`, `scripts/run_confirmation.py`, and `scripts/run_benchmarks.py`) proves that **both experiments measured both APIs, and both reported `predict()` in their published tables**:
   - In `v1.0.0:scripts/run_confirmation.py`, Table 4 reported the empirical mean of `latency_us_median` (which was measured via `model.predict()`).
   - In `v1.0.0:scripts/run_benchmarks.py`, Table 5 reported `predict_latency_us_median` (which was also measured via `model.predict()`).
   - If Confirmation had mistakenly reported `inplace_predict()`, its value would have been $104.15\,\mu\text{s}$, not $142.33\,\mu\text{s}$. If Benchmark had mistakenly reported `inplace_predict()`, its value would have been $87.08\,\mu\text{s}$, not $119.84\,\mu\text{s}$.
2. **Parallel Shift Across Both APIs:**
   Both `predict` and `inplace_predict` were approximately 17–23 µs slower in the confirmation experiment than in the benchmark experiment:
   - For Depth 4 (DOE MO): `predict` shifted by **$+22.49\,\mu\text{s}$** ($142.33$ vs $119.84$); `inplace` shifted by **$+17.07\,\mu\text{s}$** ($104.15$ vs $87.08$).
   - For Depth 7 (DOE SO): `predict` shifted by **$+23.28\,\mu\text{s}$** ($171.12$ vs $147.84$); `inplace` shifted by **$+17.38\,\mu\text{s}$** ($134.33$ vs $116.95$).
3. **Formal Scientific Attribution:**
   **The historical confirmation and benchmark latency discrepancy cannot be conclusively attributed to a specific cause. The revised experiment avoids this inconsistency through a unified timing protocol.**

---

## 2. Comparison of Historical and Revised Latency Implementations

| Dimension | Historical Confirmation (`v1.0.0`) | Historical Benchmark (`v1.0.0`) | Revision v2 Unified Protocol (`latency.py`) |
|---|---|---|---|
| **Primary Script** | `scripts/run_confirmation.py` | `scripts/run_benchmarks.py` | `scripts/measure_latency.py` / `latency.py` |
| **Prediction API** | `model.predict(single_sample)` | `model.predict(single_sample)` | Both measured; `predict` is primary |
| **Secondary API** | `booster.inplace_predict` logged | `booster.inplace_predict` logged | `inplace_predict` explicitly isolated |
| **Input Data Type** | `numpy.ndarray` (shape `(1, 8)`) | `numpy.ndarray` (shape `(1, 8)`) | `numpy.ndarray` (shape `(1, 8)`) |
| **Training Threads** | `n_jobs=4` (OpenMP) during fit | `n_jobs=1` during fit | Configured thread pinning |
| **Inference Threads**| `n_jobs=1`, `nthread=1` set post-fit | `n_jobs=1`, `nthread=1` set post-fit | `nthread=1` strictly pinned |
| **CPU Pinning** | Win32 `kernel32` CPU 0, Above Normal | Win32 `kernel32` CPU 0, Above Normal | Thread pinning + Win32 affinity verified |
| **Warmup Count** | 50 warmup calls | 50 warmup calls | 50 warmup calls |
| **Timed Iterations** | 5 batches $\times$ 200 calls (1,000 calls) | 5 batches $\times$ 200 calls (1,000 calls) | 5 batches $\times$ 200 calls (1,000 calls) |
| **Timing Clock** | `time.perf_counter_ns()` | `time.perf_counter_ns()` | `time.perf_counter_ns()` |
| **Aggregation** | Mean across 10 seeds of medians | Median across 5 repetition batches | Median across 5 repetition batches |
| **Session Control** | Sequential per-seed fitting loop | Interleaved randomized method loop | 5 independent sessions + telemetry |

---

## 3. Forensic Analysis: Evidence vs Unexplained Factors

### What is Supported by Direct Code & Data Evidence:
1. **Interface Identity:** Both experiments called the exact same `XGBRegressor.predict()` wrapper method for their published columns. Neither substituted `inplace_predict()`.
2. **Magnitude Consistency:** The latency gap between confirmation and benchmark is virtually identical across both tree depths:
   - Depth 4: $+22.49\,\mu\text{s}$ gap
   - Depth 7: $+23.28\,\mu\text{s}$ gap
   - Gap difference between depths: only $0.79\,\mu\text{s}$.
3. **Presence of the Gap in Inplace Predict:** The gap is also present when comparing `inplace_predict` across the two experiments ($+17.07\,\mu\text{s}$ at depth 4; $+17.38\,\mu\text{s}$ at depth 7).

### What Remains Unexplained & Confounded:
1. **Operating System P-State / Boost Clock Drift:** Even with process affinity pinned to CPU 0, modern x86-64 processors (such as the AMD Ryzen 5 9600X recorded in `env.json`) dynamically adjust core clock frequencies between $3.9\,\text{GHz}$ and $5.4\,\text{GHz}$ depending on thermal headroom, active thread count, and background Windows system threads. A $15-20\%$ difference in core clock frequency between an idle machine and a loaded machine accounts for a $\sim 20\,\mu\text{s}$ shift on a $120\,\mu\text{s}$ baseline.
2. **OpenMP Thread Pool Teardown Overhead:** In confirmation, models were trained with `n_jobs=4` and dynamically reconfigured to `nthread=1`. In benchmark, models were trained with `n_jobs=1` from initialization. Differences in OpenMP runtime library state or thread pool idling can introduce systematic overhead in wrapper calls.
3. **Interleaved vs Sequential Cache Effects:** The benchmark interleaved execution across 6 models, whereas confirmation ran 10 sequential fits. L1/L2 cache residency and instruction cache warmness differed systematically between the two harness structures.

---

## 4. Assessment of Processor Pinning

The claim that CPU core pinning "eliminates all latency noise" is physically and empirically inaccurate:
- Pinning fixes execution to a single logical processor core, preventing OS thread migration penalties (context switches and cache invalidation across cores).
- However, core pinning **cannot eliminate**:
  - Processor dynamic frequency scaling (boost clock variations).
  - Background operating-system interrupt servicing (DPCs/ISRs).
  - Memory controller latency and NUMA/bus contention from other system processes.
  - Python interpreter runtime garbage collection and memory allocation overhead.
- In Revision v2, across 5 independent timing sessions pinned to Core 0 on an AMD Zen 5 processor, the within-configuration between-session standard deviation averaged **$1.28\,\mu\text{s}$** (maximum: $2.74\,\mu\text{s}$). This demonstrates high repeatability, but confirms that an irreducible $\sim 1-3\,\mu\text{s}$ session noise floor remains.

---

## 5. Mandatory Manuscript Text Guidance

In the revised manuscript and documentation, the study must not retrospectively claim that the historical discrepancy was solved by discovering an interface mismatch.

The approved text formulation is:
> *"The historical confirmation and benchmark latency discrepancy cannot be conclusively attributed to a specific cause. The revised experiment avoids this inconsistency through a unified timing protocol."*
