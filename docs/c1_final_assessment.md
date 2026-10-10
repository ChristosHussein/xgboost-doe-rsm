# Work Package C.1 Final Scientific Assessment and Verification Closeout

**Repository:** `https://github.com/ChristosHussein/xgboost-doe-rsm`  
**Working Branch:** `codex/scientific-revision`  
**Base Revision / Tag:** `v1.0.0`  
**Audited Artifact Root:** `results/revision_v2/full_run_001/`  
**Audit Date:** October 10, 2026  
**Status:** Work Package C.1 Completed and Verified — Ready for Work Package D Authorization  

---

## 1. Executive Summary & Purpose

This assessment provides the final independent synthesis and closeout of **Work Package C.1: Independent Scientific Consistency Audit and Publication Evidence Verification**.

The primary objective of Work Package C.1 was to critically evaluate the empirical evidence, statistical validity, and methodological interpretations arising from the full scientific benchmark experiment ($17,077$ XGBoost model fits, $1,640,810$ timed latency inferences, and $20$ independent search replicates per optimizer) prior to regenerating the publication manuscript (`report.tex` / `report.pdf`).

The audit operated under strict scientific integrity principles:
1. **Evidence-driven neutrality:** Neither Design of Experiments (DOE) nor Tree-structured Parzen Estimator (TPE) was favored; all claims reflect empirical findings and their statistical boundaries.
2. **Raw artifact immutability:** Zero raw results or historical baseline files were modified or overwritten.
3. **Reproducibility to machine precision:** All summary metrics, Pareto frontiers, hypervolumes, and statistical tests were verified directly against trial-level and evaluation-level raw CSV and JSON records.
4. **Distinction between observation and inference:** Descriptive differences on specific splits are explicitly separated from inferential generalizations across unseen data distributions.

---

## 2. Summary of Audit Findings Across Five Core Modules

### Module A: Pareto-Front Hypervolume & Estimand Comparability (`docs/c1_hypervolume_audit.md`)
- **Mathematical Reproducibility:** Every saved hypervolume value in `results/revision_v2/full_run_001/hypervolume.json` was independently recomputed from raw trial coordinates via exact 2D Lebesgue integration. Recomputed values matched saved values to within floating-point precision ($< 1.1 \times 10^{-14}$).
- **Estimand Distinction:** The comparison between MO-TPE candidate fronts (mean $\text{HV} = 18.3834$, $\text{SD} = 0.3301$ at $[0.60, 250.0]$) and Repeated DOE candidate fronts (mean $\text{HV} = 15.6128$, $\text{SD} = 0.7268$) does not evaluate identical estimands. MO-TPE evaluates $N=1$ model fit on development split 42 per trial, whereas Repeated DOE averages validation performance across $N=5$ nuisance block partitions.
- **Matched Development Split Comparison:** Under matched single-split development evaluation on seed 42, the full DOE candidate front (spanning the 25 CCD lattice points and 2 historical optima) yields an observed hypervolume of **$16.8585$** (4 non-dominated candidate points). MO-TPE achieves a mean observed hypervolume of **$18.3834$** ($+1.5249$ units higher).
- **Scope of Conclusion & Geometric Factor Domain:** This empirical finding supports a higher observed candidate-front hypervolume for MO-TPE on this specific development split. It **does not establish universal algorithmic superiority**, as full search frontiers were not independently re-evaluated across fresh holdout partitions. Both methods operated over the exact same 4-dimensional factor hypercube:
  $$\eta \in [0.01, 0.30], \quad \text{max\_depth} \in [3, 9], \quad \text{subsample} \in [0.50, 1.00], \quad \lambda \in [0.10, 10.00]$$
  The observed hypervolume difference reflects the continuous, sequential exploration of 140 points by MO-TPE versus the discrete 25-point geometric lattice of the face-centered Central Composite Design (CCD).
- **Global Holdout Pareto Frontier:** On the external holdout test set (averaged across 20 evaluation seeds), both Repeated DOE MO (3 non-dominated configurations) and MO-TPE (3 non-dominated configurations) contribute to the global Pareto frontier in the low-latency regime ($116 - 126\,\mu\text{s}$).

### Module B: Computational Budget & Sample Efficiency (`docs/c1_computational_efficiency.md`, `budget_reconciliation.json`)
- **Total Fit Ledger:** Reconciled every single model fit across the entire experimental campaign:
  - Optimizer search trials: $4 \text{ optimizers} \times 20 \text{ replicates} \times 140 \text{ trials} = 11,200 \text{ fits}$
  - Repeated DOE search runs: $20 \text{ replicate designs} \times 28 \text{ design runs} \times 5 \text{ block seeds} = 2,800 \text{ fits}$
  - Single-split matched candidate evaluation: $27 \text{ configurations} \times 1 \text{ fit} = 27 \text{ fits}$
  - Final holdout retraining evaluation: $122 \text{ configurations} \times 20 \text{ fresh seeds} = 2,440 \text{ fits}$
  - Benchmark latency profiling refits: $122 \text{ configurations} \times 5 \text{ measurement sessions} = 610 \text{ fits}$
  - **Grand Total Model Fits:** Exactly **$17,077$ fits**.
- **Latency Inferences & Warmup:** Reconciled $1,640,810$ timed inference measurements and $201,270$ un-timed warmup inferences.
- **Historical Narrative Correction:** Refuted the historical "54 fits" claim in the baseline manuscript. The historical code executed 28 runs per block across 5 blocks ($140$ fits). The "54 fits" figure arose from double-counting 4 center points ($25 + 4 = 29$, $29 + 25 = 54$), which had no physical execution basis.
- **Fair Efficiency Comparison:** Under an equal budget of 140 fits per replicate, Repeated DOE allocates fits to 25 unique geometric lattice points across 5 nuisance blocks (enabling formal ANOVA, block effect estimation, lack-of-fit $F$-tests, and curvature modeling), while TPE allocates 140 fits to distinct configurations for global exploratory coverage. Wall-clock search times averaged $26.24\,\text{s}$ for MO-TPE versus $36.49\,\text{s}$ for Repeated DOE.

### Module C: Latency Forensics & Interface Profiling (`docs/c1_latency_forensics.md`)
- **Refutation of Interface Mismatch Hypothesis:** Code inspection of the historical `v1.0.0` repository confirmed that both confirmation and benchmark scripts evaluated both `predict()` and `inplace_predict()`, and both published the standard `predict()` numbers. The historical $+22.5\,\mu\text{s}$ to $+23.3\,\mu\text{s}$ discrepancy was not caused by an API mismatch.
- **Root Cause of Historical Drift:** The shift occurred across all models and both APIs equally ($+22.49\,\mu\text{s}$ at depth 4, $+23.28\,\mu\text{s}$ at depth 7), driven by unpinned, multi-session CPU execution and environment background load.
- **Standardized Benchmarking Stability:** Pinning execution to Core 0 with thread affinity and executing 5 sessions $\times$ 1,000 iterations reduced within-configuration between-session standard deviation to $1.28\,\mu\text{s}$, establishing robust measurement stability.

### Module D: Selection Stability vs Conditional Retraining Variance (`docs/c1_selection_stability.md`)
- **Selection Stability:** Repeated DOE Single-Objective exhibited $100\%$ selection coincidence across all 20 independent nuisance block sets ($20/20$ selected configuration `9d7c4e74...`, $\eta = 0.15195$, $\text{max\_depth} = 7$, $\text{subsample} = 1.0$, $\lambda = 10.0$), yielding zero between-search standard deviation ($\text{SD} = 0.00000$).
- **Conditional Retraining Uncertainty:** This does not mean the selected configuration has zero predictive uncertainty. When retrained across 20 holdout evaluation seeds ($2001 - 2020$), its conditional retraining standard deviation is **$\text{SD} = 0.00304$** ($95\%\text{ CI}: [0.4673, 0.4699]$). The manuscript must strictly distinguish between selection stability and prediction uncertainty.
- **Multi-Objective Knee Sensitivity:** Repeated DOE Multi-Objective selected 12 unique configurations across 20 block sets ($15/20$ selected depth 4, $5/20$ selected depth 5), illustrating curvature sensitivity along the multi-objective desirability boundary.

### Module E: Statistical Inference & Publication Claim Ledger (`docs/c1_statistical_inference.md`)
- **DOE MO vs MO-TPE Test RMSE:** On the holdout test set, Repeated DOE MO achieved a mean Test RMSE of $0.48997 \pm 0.00732$ versus $0.49255 \pm 0.01080$ for MO-TPE. The difference ($-0.00258$) is not statistically significant ($p = 0.3826$ Welch two-sample, $p = 0.3463$ paired).
- **Constrained TPE Benchmark Feasibility:** Constrained TPE registered $100\%$ search-time feasibility ($20/20 \le 145\,\mu\text{s}$) under its 30-call online protocol, but only $45\%$ ($9/20$) feasibility under the primary 1,000-call benchmark protocol.
  - All 9 depth-6 configurations remained fully feasible ($136.62\,\mu\text{s}$ benchmark mean).
  - All 11 depth-7 configurations exceeded the threshold ($150.54\,\mu\text{s}$ benchmark mean).
  - The discrepancy is attributed to possible explanatory factors (online sampling variability in 30-call estimates, protocol differences, and selection effects near the constraint boundary) rather than conclusively asserted root causes.

---

## 3. Final Publication Claim Ledger (10 Audited Claims)

| Claim ID | Candidate Manuscript Claim | Audit Verification Status | Scientific Evidence & Recommended Manuscript Formulation |
|:---:|---|---|---|
| **C1** | *"Repeated DOE Multi-Objective achieves identical or lower Test RMSE compared to MO-TPE under matched latency."* | **Directly Verified (Descriptive)** | Observed holdout Test RMSE is $0.48997$ (DOE MO) vs $0.49255$ (MO-TPE) at matched latency ($122.29\,\mu\text{s}$ vs $121.46\,\mu\text{s}$). Must be reported as a descriptive difference without claiming statistical superiority ($p = 0.38$). |
| **C2** | *"Repeated DOE MO achieves statistically significantly lower Test RMSE than MO-TPE."* | **Refuted / Unsupported** | Difference is not statistically significant ($t = -0.884, p = 0.3826$). Do not claim statistical superiority. |
| **C3** | *"Repeated DOE Single-Objective has zero predictive uncertainty."* | **Refuted / Factually False** | Selection stability was deterministic across the 20 block sets ($20/20$), but conditional retraining standard deviation across holdout evaluation seeds is $\text{SD} = 0.00304$. |
| **C4** | *"Repeated DOE Single-Objective selection is perfectly stable across nuisance block sets."* | **Directly Verified** | All 20 independent nuisance block sets selected the exact same configuration ($20/20$, $\text{SHA-256: } \texttt{9d7c4e74...}$). |
| **C5** | *"MO-TPE achieves higher observed candidate-front hypervolume on the evaluated development split."* | **Directly Verified with Limitations** | MO-TPE achieved mean candidate HV of $18.3834$ vs $16.8585$ for Full DOE on split 42 ($p < 0.0001$). Supports higher observed hypervolume on this development split; does not establish universal algorithmic superiority as complete search frontiers were not re-evaluated across unseen data partitions. |
| **C6** | *"Candidate-level Pareto hypervolume comparisons between Repeated DOE and MO-TPE compare different estimands."* | **Directly Verified** | MO-TPE uses a single split per trial ($N=1$); Repeated DOE candidate fronts average validation performance across 5 nuisance blocks ($N=5$). |
| **C7** | *"Single-Objective TPE discovers the lowest unconstrained Test RMSE in the study."* | **Directly Verified** | SO-TPE achieved Test RMSE $0.46691 \pm 0.00116$, but violates the latency constraint ($188.93\,\mu\text{s}$). |
| **C8** | *"Constrained TPE selections achieved 100% search-time feasibility but only 45% benchmark feasibility."* | **Directly Verified** | Empirical result verified ($20/20$ vs $9/20$). Depth 6 models ($9/9$) remained feasible; depth 7 models ($11/11$) exceeded threshold. Explanatory mechanisms (sampling variability, protocol differences, selection effects) are presented as hypotheses without unsupported causal assertions. |
| **C9** | *"The historical latency discrepancy between confirmation and benchmark was caused by predict vs inplace_predict confusion."* | **Refuted / Unsupported** | Both historical scripts measured both APIs and both published `predict()`. Discrepancy was caused by unpinned execution and environment drift. |
| **C10** | *"Both Repeated DOE MO and MO-TPE contribute non-dominated configurations to the global holdout Pareto front."* | **Directly Verified** | 12 configurations form the global front: 3 from DOE MO ($118.7 - 126.2\,\mu\text{s}$), 3 from MO-TPE ($116.1 - 120.2\,\mu\text{s}$), 5 from Constrained TPE ($136.0 - 150.4\,\mu\text{s}$), and 1 from SO-TPE ($171.3\,\mu\text{s}$). |

---

## 4. Automated Audit Verification Pipeline

To ensure that these audited numbers and claim boundaries cannot be silently regressed or altered during future manuscript revisions, an automated verification script has been implemented:

- **Verification Script:** `scripts/audit_scientific_consistency.py`
  - Validates total model fit count ($17,077$), trial count ($11,200$), and test isolation (zero holdout leaks).
  - Validates full holdout evaluations ($2,440$ rows) and configuration hash integrity (zero mismatches).
  - Validates exact hypervolume values for MO-TPE, Repeated DOE, and Full DOE candidates across reference points $[0.60, 250.0]$ and $[0.65, 275.0]$ within floating-point tolerance ($10^{-9}$).
  - Validates selection stability metrics ($20/20$ for DOE SO) and Constrained TPE benchmark feasibility ($9/20$).
  - Validates Welch two-sample $t$-test results between DOE MO and MO-TPE ($p > 0.30$).
  - Fails with explicit exit code 1 and descriptive assertion messages upon any numerical discrepancy.
- **Test Suite:** `tests/test_scientific_consistency_audit.py`
  - Validates execution against `results/revision_v2/full_run_001/`.
  - Tests failure detection under artificial data corruptions.

---

## 5. Work Package C.1 Closeout Declaration

Work Package C.1 has successfully achieved all required objectives:
1. Every candidate numerical claim for the revised manuscript has been audited against immutable raw artifacts.
2. Estimand differences, geometric domains, and algorithmic boundaries have been rigorously defined.
3. Factual descriptions regarding the CCD factor hypercube and factor bounds ($[0.01, 0.30] \times [3, 9] \times [0.5, 1.0] \times [0.1, 10.0]$) have been corrected across all project documentation.
4. Causal attributions for Constrained TPE latency drift have been appropriately qualified.
5. Automated consistency auditing has been integrated into the repository test suite.

**Work Package C.1 is officially closed and signed off.**  
The repository is fully prepared for **Work Package D: Publication Revision and Manuscript Regeneration** upon explicit user authorization.
