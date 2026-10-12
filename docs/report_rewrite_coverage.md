# Reader-Focused Report Rewrite Coverage Inventory

This inventory records how the complete scientific content at merged baseline
`8da72ce2e48890ff2f5f11079eb5e6691750f7d0` is preserved after the accessibility
rewrite. Empirical values remain bound to `ReportingDataset`, generated LaTeX
macros, and generated tables. This document maps content; it is not a second
source for numerical results.

The main manuscript passed a separate editorial relevance audit. Development
milestones, branch and pull-request history, code-review chronology, integrity
implementation details, configuration hashes, and exhaustive internal execution
ledgers were excluded from the scientific narrative. Scientifically necessary
design controls remain: candidate freezing before holdout evaluation, repeated
searches, matched model-fit budgets, and distinct latency protocols. Detailed
reproducibility records remain in the supplement and repository manifests.

## Publication package

- `report.tex` / `report.pdf`: concise reader-focused manuscript.
- `technical_supplement.tex` / `technical_supplement.pdf`: complete derivations,
  full statistical tables, diagnostics, protocol details, and extended benchmark
  interpretation from the approved scientific manuscript.
- `REPORT.md`: comprehensive generated numerical research record.
- `docs/scientific_claims_audit.md`: authoritative generated register of all
  twenty-two audited claims.
- `docs/generated/scientific_results_manifest.json`: metric-level source and
  reporting-context manifest.

## Audited claim mapping

| Claim | Original topic | Reader-focused destination | Technical destination |
|---|---|---|---|
| C-01 | Pre-planned FCCD chronology | Main Sections 1 and 3; pipeline figure | Supplement Sections 1 and 2 |
| C-02 | Equal fit counts but unequal spatial information | Main Sections 3 and 7; method-comparison figure | Supplement Section 2 and computational-budget table |
| C-03 | Latency protocols and interface overhead | Main Section 3 | Supplement Section 2 |
| C-04 | Factorial screening effects | Main Section 4 | Supplement Section 3 and Phase-1 ANOVA tables |
| C-05 | Curvature and denominator sensitivity | Main Section 4 | Supplement Section 3 |
| C-06 | Blocking, ICC, and variance shielding | Main Section 3; blocking figure | Supplement Sections 2 and 4 |
| C-07 | Second-order fit and HC3 inference | Main Section 4 | Supplement Section 4 and full RMSE ANOVA table |
| C-08 | Lack-of-fit decomposition and restricted domain | Main Section 4 | Supplement Section 4 and lack-of-fit table |
| C-09 | Latency model adequacy | Main Section 4 | Supplement Section 4 and latency tables/figure |
| C-10 | Residual diagnostics and qualified interpretation | Main Section 4 | Supplement Section 5 and diagnostic panel |
| C-11 | Canonical eigenanalysis and ridge behavior | Main Section 5 | Supplement Section 6 and response-surface figure |
| C-12 | Desirability compromise | Main Section 5 | Supplement Section 7 and sensitivity table/figure |
| C-13 | Independent confirmation intervals | Main Section 6 and confirmation table | Supplement Section 8 |
| C-14 | Similar rounded values and configuration provenance | Main Section 6 summary | Supplement Section 8 provenance subsection |
| C-15 | Historical holdout exposure | Main Sections 3, 6, and 8 | Supplement Sections 1 and 8 |
| C-16 | Repeated DOE interpretation and two variance sources | Main Section 6 | Supplement Section 8 |
| C-17 | DOE versus multi-objective TPE inference | Main Section 7 | Supplement Section 8 and benchmark table |
| C-18 | Development hypervolume protocol differences | Main Section 7 | Supplement Section 8 and hypervolume table |
| C-19 | Holdout frontier and bootstrap uncertainty | Main Section 7 and Pareto figure | Supplement Section 8 |
| C-20 | Constrained-TPE feasibility degradation | Main Sections 7 and 8 | Supplement Section 8 |
| C-21 | Full computational budget and wall-clock cost | Main Section 7 | Supplement computational-budget table |
| C-22 | Timing-session differences and reproducibility | Main Sections 3 and 10 | Supplement latency and code-availability sections; repository provenance manifests |

## Results-table coverage

All generated tables remain in the publication package. The main manuscript uses
the confirmation and benchmark tables because they answer the central research
question directly. The supplement retains the full Phase-1 ANOVA, second-order
RMSE ANOVA, latency ANOVA, lack-of-fit decomposition, latency-model comparison,
depth optimization, desirability sensitivity, hypervolume, design-seed, and
computational-budget tables.

## Original empirical-figure coverage

| Original figure | Location before PR #3 | Revised location | Treatment | Scientific purpose |
|---|---|---|---|---|
| `diagnostics_panel_4in1.png` | Approved main manuscript, residual-diagnostics section | Technical supplement, residual-diagnostics section; findings summarized and cross-referenced in main Section 4 | Original reused | Tests normality, residual patterning, execution-order dependence, and influential observations; supports the qualified model-adequacy discussion. |
| `response_surface_rmse_2d_3d.png` | Approved main manuscript, canonical/ridge analysis | Main Section 4 and technical supplement | Original reused | Shows how coded learning rate and depth shape validation RMSE, the in-domain boundary candidate, and the out-of-domain stationary point. |
| `response_surface_latency_2d_3d.png` | Approved main manuscript, latency modeling | Main Section 4 and technical supplement | Original reused | Shows measured prediction latency by depth, uncertainty, and the improvement of the quadratic over the linear depth model. |
| `desirability_pareto_front.png` | Approved main manuscript, desirability optimization | Main Section 5 and technical supplement | Original reused | Shows the observed accuracy--speed Pareto trade-off and why the depth-4 compromise and depth-7 accuracy candidate were both confirmed. |
| `revision_v2_pareto_front.png` | Approved main manuscript, benchmark comparison | Main Section 7 and technical supplement | Original reused under its repository filename | Separates development candidate fronts from holdout results and shows the final non-dominated set with uncertainty. |
| `efficiency_comparison_curve.png` | Approved main manuscript, benchmark comparison | Full original retained in technical supplement; clearer source-derived `explainer_efficiency_comparison.pdf` in main Section 7 | Regenerated for the main narrative from the same canonical trajectory data | Shows how quickly each method improved over the common 140-fit budget and why randomized DOE order is not an adaptive optimization trajectory. |

The main manuscript also uses three conceptual figures generated by
`scripts/generate_explanatory_figures.py`: the research pipeline, FCCD projection,
and blocking example. The method comparison is a native manuscript table. These
elements explain the design and do not replace empirical evidence. The source-derived
efficiency figure is generated by the same script from the canonical trajectory files.
All six original empirical figures remain unchanged in the complete publication package.

## Formal tests and uncertainty coverage

The package retains factorial and second-order ANOVA, multiple curvature
denominators, ANOVA and REML block-variance estimates, classical and HC3
coefficient inference, lack-of-fit decomposition, restricted-domain sensitivity,
Shapiro--Wilk, Levene, Brown--Forsythe, Breusch--Pagan, Durbin--Watson,
Ljung--Box, runs testing, Cook's distance, canonical wild-bootstrap intervals,
Satterthwaite confirmation intervals, unpaired Welch comparison, secondary
replicate-index paired comparison, one-sample development-front comparison,
and paired cluster bootstrap uncertainty for holdout hypervolume.

## Material limitations retained

- The quadratic RMSE surface has significant global lack of fit and localized
  optimism at the accuracy-focused boundary candidate.
- HC3 standard errors address coefficient inference under heteroscedasticity but
  do not repair misspecification.
- Non-significant tests do not establish equivalence or prove the absence of all
  temporal dependence.
- DOE and adaptive search use equal fit counts but different coordinate,
  replication, split, and online-timing structures.
- Search-time latency is not interchangeable with warmed benchmark latency.
- Constrained TPE loses feasibility under the verification protocol.
- The holdout was exposed by historical work before Revision-v2 isolation.
- Fixed-holdout comparisons do not estimate dataset-sampling uncertainty.
- Conclusions are limited to one dataset, one hardware environment, four tuned
  factors, and a fixed one-hundred-tree ensemble.

## Page-count record

- Approved pre-rewrite main report: 27 pages.
- Reader-focused main report: 17 pages, including the dedicated one-page table of contents.
- Technical supplement: 26 pages.

