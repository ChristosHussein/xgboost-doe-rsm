# fix.md: Remediation brief for the RSM/CCD XGBoost HPO report

**Audience:** a coding agent working in the repo that produced `report.pdf`
("Sequential Response Surface Methodology and Central Composite Design for
Multi-Objective Hyperparameter Optimization in Gradient Boosted Trees under
Stochastic Nuisance Blocking").

**Goal:** make every number, table, figure and claim in the report correct,
reproducible and traceable to code. Several reported numbers are internally
inconsistent (see section 1). Fix the pipeline first, regenerate everything,
then rewrite the text to match the results.

---

## 0. Ground rules

1. **No hand-typed results.** Every number in the paper must flow:
   script -> `results/*.csv|json` -> generated `tables/*.tex` / `figures/*` -> `\input`.
2. **Never "fix" a number by editing it to agree with another one.** Re-derive it
   from the fitted model. If a check fails, report the failure; do not mask it.
3. **Do not change scientific claims until Task 11.** The text follows the
   results, not the other way round.
4. **Locate the code first** (`grep -rn "xgboost\|statsmodels\|optuna\|desirab\|eigh" .`).
   If the code that produced the report is not in the repo, say so and stop on
   tasks that need it. Do not reconstruct results from the PDF.
5. **Experiments are cheap** (~1.8 s per run; the full 140-run design is ~5 min),
   so re-running is preferred over patching old CSVs.
6. Keep `CHANGELOG.md` with old -> new for every number listed in Appendix A.
7. Fixed seeds, pinned environment, everything logged (Task 0).
8. Run `scripts/verify_report_numbers.py` (Appendix B) **first** to reproduce the
   inconsistencies below; delete it once the pipeline replaces it.

---

## 1. Verified problems (reproduce each before fixing)

Priority: **P0** = invalidates a headline conclusion, **P1** = serious,
**P2** = presentation/accuracy.

| ID | Pri | Where | Problem | Evidence |
|----|-----|-------|---------|----------|
| P1 | P0 | Eq. 22-23 vs Table 3 | `b` and `B` do not match the ANOVA sums of squares | SS(x1)=1.372904 with sum(x1^2)=90 gives beta1=0.1235; printed b1=0.1070 (all four are x0.866). Off-diagonals of `B` equal beta_ij/sqrt(2), not beta_ij/2 (e.g. beta12=sqrt(0.080835/80)=0.03179, half = 0.01589, printed 0.02248). Diagonals ~9% low (from SS: 0.0987, 0.0245, ..., 0.00189 vs printed 0.0899, 0.0223, ..., 0.00171). |
| P2 | P0 | Eq. 23 vs 27-30 | Eigenvalues are not eigenvalues of the printed `B` | trace(B)=0.11412 but sum of reported eigenvalues = 0.12540 (ratio 1.0988, same as the diagonal discrepancy). |
| P3 | P0 | Eq. 23 | Printed `B` is **not positive definite** | eig(B_printed) = (-2.5e-5, 0.0012, 0.0156, 0.0973). |
| P4 | P0 | Eq. 25 | Reported `x0` does not solve `b + 2B x0 = 0` | B x0 = (0.209, 0.111, 0.0124, 0.0191) vs -b/2 = (0.0535, 0.0159, 0.00204, 0.00077). Solving the printed system gives roughly (0.81, 1.81, -23.8, +3.2). |
| P5 | P0 | Eq. 25 | `y0_hat` inconsistent with printed values | b0 + 0.5 x0'b = 0.372, reported 0.3428. |
| P6 | P0 | Sec 6.2 | Natural-unit conversion wrong | x3=28.44 -> subsample = 0.75+0.25*28.44 = 7.86 (not 4.3; also impossible, a fraction). x4=-1.78 -> lambda = 0.0165. |
| P7 | P0 | Abstract, 6.2 | "Just outside the operational perimeter" is false | x3,0 = 28 coded units, i.e. ~28x beyond the region. Predicted RMSE 0.34 is far below the best observed (0.4645) and implausible for XGBoost on this data: extrapolation artifact. |
| P8 | P0 | 6.2 | "All eigenvalues > 0 -> unique local minimum" unsupported | SE of a quadratic coefficient ~0.0023 > lambda3 (0.0017), lambda4 (0.000126). Table 3: p(x3^2)=0.91, p(x4^2)=0.42. Signs of lambda3, lambda4 are indistinguishable from zero. |
| P9 | P0 | Eq. 32 | Clipping x0 coordinates is **not** the constrained optimum | With non-diagonal B it is generally wrong. "Ridge analysis" in the section title is never performed. Fig 2(a) shows an apparent interior closed contour around x1~0.5-0.8, x2~0-0.8. |
| P10 | P0 | 4.3, Table 5 | Lack-of-fit df wrong | 121-115=6 mixes a model *with* block terms and a pure-error calc *without* them. Correct: m=25 points, 15 treatment params -> LoF df = 10; block-adjusted PE df = 111 (10+111=121). |
| P11 | P0 | 4.3, Table 5 | Significant LoF (F=122.6; 86% of residual SS) dismissed | "As noted by Montgomery" attribution unverified. No transformation tried despite BP failure. RMS misfit ~0.027 > the 0.0097 optimizer gap the paper argues over. |
| P12 | P0 | 8.1, Table 6 | Confirmation is not at x* and not independent | Surrogate predicted at depth 4.36 (x2=-0.5475), runs executed at depth 4. Same 5 seeds (same splits) as the design. |
| P13 | P1 | Table 6 | Intervals inconsistent | Hat value at x* must be identical for Y1 and Y2 if both models have all terms. Y1 CI implies h=0.144, Y2 CI implies h=0.748. Also, a CI for the mean is compared with a mean of 5 runs; the correct object is a prediction interval for the mean of m runs (Y1: ~[0.471, 0.490] passes; Y2 with h=0.144: ~[107, 129], observed 130.97 fails). |
| P14 | P0 | Table 7, 8.2 | Benchmark misreports DOE | DOE "best 0.4645" is the best raw design row, not the recommended x* (confirmed 0.4880 -> 7.3% worse than TPE, 6.4% worse than RS). Abstract says "comparable". |
| P15 | P0 | 8.2 | Selection on the test set | All methods minimise holdout *test* RMSE and report the min of it: optimistic bias; no validation split. (Intro also mentions cross-validation; the method is a single 80/20 split.) |
| P16 | P1 | 8.2 | "Equal budget" is not equal | DOE = 25 unique configs x replicates; baselines = 140 unique configs, one evaluation each. |
| P17 | P1 | 8.2 | Unfair latency comparison | TPE was single-objective. "62.2% slower" uses *predicted* latency 117.99; against empirical 130.97 it is 46.1%. TPE depth-9 = 191.4 us but DOE depth-9 runs in Fig 4 top out ~170 us: likely different measurement conditions. |
| P18 | P1 | 8.2 | Single optimizer trajectories | No replication of RS/TPE runs; no uncertainty on the comparison. |
| P19 | P1 | 8.2 | "Depth 4 achieves 95.4% of maximal accuracy" misreads d1 | d1 is a linear score between the best and worst *observed* RMSE (0.4645 and 0.8173, the latter an underfit corner), not a percent of accuracy. |
| P20 | P0 | 2.2, 9 | ICC = 21.6% contradicts own ANOVA | Block F-tests non-significant (p=0.9995, 0.4635). MS_block/MSE = 0.907 -> ANOVA sigma2_block <= 0 (RMSE); latency ICC ~0.2%. Model/response for the REML ICC unspecified. |
| P21 | P1 | 5(b) | Levene result is statistically implausible | W=0.000332, F(4,135): P(F<W) ~ 2e-7. Likely computed on degenerate input. Also tests blocks, not design points (BP across design failed: p<1e-7). |
| P22 | P1 | 5(a) | Shapiro-Wilk statement wrong | p=0.0226 is *not* rejected at alpha=0.01 (it is at 0.05). |
| P23 | P1 | 5, Fig 1 | Outlier ignored | Studentized residual ~4.2 near run 84; Cook's D 0.16 is >5x the plotted 4/n line. Abstract claims diagnostics "confirm variance homogeneity" despite BP. |
| P24 | P2 | 5(c) | Rolling-mean "proves no drift" | 10-run mean of unit-variance residuals has SE ~0.32, so +-0.3 is just noise. Run-order randomization never described. |
| P25 | P1 | Design | n_estimators fixed at 100 | eta effect is largely underfitting at eta=0.01; optimum at the upper eta bound. eta and n_estimators are coupled. min_child_weight, gamma, colsample, alpha absent; two of four factors are inert. |
| P26 | P1 | 4.2, Table 4 | Latency model weak and under-reported | Approx R^2 ~0.68 from listed terms; table omits other terms; PE is 98% of residual. "2^depth" explanation is wrong (single-sample prediction cost ~ linear in depth; fixed Python/DMatrix overhead likely dominates 100-196 us). Hardware/threads/versions unstated. |
| P27 | P1 | 7.1 | Desirability limits from observed extremes | U2=196.48 us does not appear in the CCD scatter of Fig 4 (max ~170). Limits arbitrary; no sensitivity analysis. |
| P28 | P1 | 9 | Over-generalization | Four general principles drawn from one dataset, one algorithm, 100 trees. Principle 4 (DOE superior for D<=6) contradicted by own benchmark. |
| P29 | P2 | Eq 3/6/21 | Notation | Blocks random in Eq 3, fixed in ANOVA. Eq 6: 5 dummies + intercept is collinear. B is not the Hessian (that is 2B). |
| P30 | P2 | Refs/format | Misc | Verify Montgomery chapter/section numbers against the edition used (RCBD, RSM/CCD, lack of fit; Ch. 14 appears unrelated). Refs [2],[8] uncited. Table 7 overflows the page; TPE median (0.4645) equals DOE best (0.4645): check for copy error. Figure labels show raw LaTeX (`$\ln(\eta)$`). Typo "geometric geometry". Overclaiming language ("proving", "perfect", "definitively"). No code/data/version statement. |
| P31 | P1 | Design | Replicate structure undocumented | If the 4 center replicates per block share the same seed, a deterministic XGBoost gives identical results, so pure error would be zero or reflect only subsample randomness. Document exactly how replicates differ. |

---

## 2. Tasks (execute in order; each has acceptance criteria)

### Task 0: Reproducible environment and tidy data

- Write `results/env.json`: Python, xgboost, sklearn, statsmodels, scipy, numpy,
  pandas, optuna versions; CPU model (`lscpu`), core count, OS, `nthread` used.
- Define seed semantics in one place (`config.yaml`): a seed controls (a) the
  train/val/test split and (b) XGBoost `random_state`. Use a **60/20/20
  train/validation/test** split per seed.
- Tidy results table `results/runs.csv` with columns:
  `run_id, phase, point_id, block, seed, replicate, run_order, x1..x4, eta, depth,
  subsample, reg_lambda, val_rmse, test_rmse, latency_us_median, latency_us_iqr, fit_time_s`.
- **Randomize** run order (record the RNG seed) and store `run_order`.
- Assertions: no duplicate `(point_id, seed)` rows unless explicitly marked as a
  latency-repeat; coded <-> natural round-trip test passes; all natural values
  inside declared ranges.

**Acceptance:** `pytest tests/test_data.py` passes; `runs.csv` regenerated from scratch by one command.

### Task 1: Rebuild the second-order fit and canonical analysis from the fitted model (P1-P8)

Build matrices programmatically. Never type them.

```python
import numpy as np, pandas as pd, statsmodels.api as sm
Q = ["x1", "x2", "x3", "x4"]

def design(df):
    X = pd.DataFrame(index=df.index)
    for q in Q: X[q] = df[q]
    for q in Q: X[q + "_sq"] = df[q] ** 2
    for i in range(4):
        for j in range(i + 1, 4):
            X[f"{Q[i]}_{Q[j]}"] = df[Q[i]] * df[Q[j]]
    blk = pd.get_dummies(df["block"], prefix="blk", drop_first=True).astype(float)
    return sm.add_constant(pd.concat([X, blk], axis=1))

def canonical(fit):
    p = fit.params
    b = p[Q].values
    B = np.diag(p[[q + "_sq" for q in Q]].values)
    for i in range(4):
        for j in range(i + 1, 4):
            B[i, j] = B[j, i] = p[f"{Q[i]}_{Q[j]}"] / 2          # NOT /sqrt(2)
    blk = [c for c in p.index if c.startswith("blk_")]
    b0 = p["const"] + p[blk].sum() / 5                            # average over 5 blocks
    return b0, b, B
```

Assertions that must be in code (and in `tests/test_phase3.py`):

```python
lam, M = np.linalg.eigh(B)
assert np.isclose(np.trace(B), lam.sum())
x0 = -0.5 * np.linalg.solve(B, b)
assert np.allclose(b + 2 * B @ x0, 0, atol=1e-10)
assert np.isclose(b0 + 0.5 * b @ x0, predict_block_avg(fit, x0))

# Type III SS cross-check for every single-df term
XtXi = np.linalg.inv(X.T @ X)
ss3 = fit.params ** 2 / np.diag(XtXi)
anova = sm.stats.anova_lm(fit, typ=3)      # compare row by row, rtol=1e-6
```

Also required:

- **Eigenvalue uncertainty:** wild (Rademacher) bootstrap, >= 2000 reps, because
  of heteroscedasticity. Report 2.5/50/97.5 percentiles of every eigenvalue and
  the fraction of reps with min eigenvalue <= 0.

```python
def wild_boot_eigs(X, y, fit, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    res, fv = fit.resid.values, fit.fittedvalues.values
    out = []
    for _ in range(n):
        f = sm.OLS(fv + res * rng.choice([-1, 1], size=len(res)), X).fit()
        out.append(np.linalg.eigvalsh(canonical(f)[2]))
    out = np.array(out)
    return np.percentile(out, [2.5, 50, 97.5], axis=0), (out[:, 0] <= 0).mean()
```

- **Classification rule:** call the surface "convex / minimum" only if the 2.5th
  percentile of the smallest eigenvalue is > 0. Otherwise report a *stationary
  or rising ridge; curvature in those directions not distinguishable from zero*.
- **Extrapolation flag:** if `max|x0| > 1`, mark x0 as outside the region, report
  its distance in coded units, convert to natural units with the correct formula
  (subsample must stay in [0.5, 1]), and **do not present y0_hat as attainable**.
- Report eigenvectors (`M`). The claim "ridge along subsampling/regularization"
  is only allowed if the eigenvectors actually load on x3/x4.

**Acceptance:** all assertions pass; `results/phase3.json` holds b0, b, B,
eigenvalues + CIs, eigenvectors, x0, y0_hat, extrapolation flag.

### Task 2: Real constrained optimum / ridge analysis (P9)

Remove coordinate clipping everywhere. Implement:

```python
from scipy.optimize import minimize

def box_opt(f, depths=range(3, 10), n_starts=30, seed=0):
    """Min of surrogate f(x) over the cube, depth treated as integer."""
    rng = np.random.default_rng(seed); best = None
    for d in depths:
        x2 = (d - 6) / 3
        g = lambda z: f(np.array([z[0], x2, z[1], z[2]]))
        for _ in range(n_starts):
            r = minimize(g, rng.uniform(-1, 1, 3), bounds=[(-1, 1)] * 3, method="L-BFGS-B")
            if best is None or r.fun < best[0]: best = (r.fun, d, r.x)
    return best
```

- Table: best predicted Y1 **at each integer depth 3..9**, other three factors
  optimized, with prediction SE.
- Ridge trace: for R in `linspace(0, 2, 21)` minimize y_hat s.t. `||x||^2 <= R^2`
  (SLSQP); report x(R), y_hat(R), SE(y_hat(R)). State clearly that the region is a
  cube; report the cube-constrained result as primary.
- Update Fig 2 so slices pass through the *constrained* optimum, not the clipped x0.

**Acceptance:** no code path clips x0; ridge table and integer-depth table generated.

### Task 3: Correct lack-of-fit test and model adequacy (P10, P11)

Use nested-model F-test (it gets df right automatically):

```python
import statsmodels.formula.api as smf
reduced   = smf.ols("y ~ C(block) + x1+x2+x3+x4 + x1_sq+x2_sq+x3_sq+x4_sq"
                    " + x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4", df).fit()
saturated = smf.ols("y ~ C(block) + C(point_id)", df).fit()
print(sm.stats.anova_lm(reduced, saturated))   # expect df_diff = 10, df_resid = 111
```

- Do this for Y1 and Y2. Add a unit test asserting df (10, 111) for the 25-point,
  5-block design.
- Box-Cox for both responses (`scipy.stats.boxcox`, report lambda and its CI);
  refit on log or Box-Cox scale if justified and re-run LoF.
- If LoF remains significant, report **practical** size: `sqrt(MS_LoF)` vs
  `sqrt(MS_PE)` and vs the RMSE differences that matter (~0.01). Do **not** write
  "fully adequate" / "zero detectable deficiency"; write "no significant LoF detected".
- **Recommended augmentation:** the low-eta region is just underfit and drives the
  curvature. Either (a) add runs at two intermediate ln(eta) levels (e.g. x1 = +-0.5)
  so a cubic in x1 is estimable, or (b) steepest-descent move and a *new* CCD
  centered on eta in ~[0.1, 0.3], depth 4-9. Re-run Tasks 1-3 on the new region.

**Acceptance:** `results/lof.json` with correct df; adequacy statement generated from p-values and effect sizes.

### Task 4: Diagnostics done right (P21-P24)

```python
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
infl  = fit.get_influence()
r     = infl.resid_studentized_external
W, p  = stats.shapiro(r)
groups = [fit.resid[df.block == b] for b in sorted(df.block.unique())]
lev   = stats.levene(*groups, center="median")
bp    = sm.stats.diagnostic.het_breuschpagan(fit.resid, X)
cooks = infl.cooks_distance[0]
dw    = durbin_watson(fit.resid[np.argsort(df.run_order.values)])
```

- Print group sizes and group variances next to Levene. Assert each block has 28
  residuals and that W is not degenerate. If W is still ~0, find the bug.
- Also test heteroscedasticity **across design points** (Brown-Forsythe on
  `point_id`, BP). Add HC3 robust inference (`fit.get_robustcov_results("HC3")`)
  as a sensitivity table.
- **Outliers:** list every row with |r|>3 and Cook's D > 4/n (config, seed, raw
  metrics, XGBoost log). Inspect the run near old index 84. Refit with and without,
  report the coefficient/optimum change. Do not delete silently.
- Independence: DW and a runs test in *actual execution order* (needs Task 0
  randomization); drop the rolling-mean "proof".
- Normality wording must be generated from the p-value and alpha, not typed.

**Acceptance:** `results/diagnostics.json` + regenerated Fig 1; text uses computed verdicts.

### Task 5: ICC and blocking (P20, P31)

```python
m = smf.mixedlm("val_rmse ~ x1+x2+x3+x4+x1_sq+x2_sq+x3_sq+x4_sq"
                "+x1_x2+x1_x3+x1_x4+x2_x3+x2_x4+x3_x4",
                df, groups=df["block"]).fit(reml=True)
s2b, s2e = float(m.cov_re.iloc[0, 0]), m.scale
icc_reml = s2b / (s2b + s2e)
icc_anova = max(0, (MS_block - MSE) / n_per_block) / (max(0, (MS_block - MSE) / n_per_block) + MSE)
```

- Report both estimators for Y1 and Y2, with a parametric-bootstrap CI (note the
  boundary at 0). Unit test: the two estimators agree to within sampling error.
- Use **one** convention for blocks (random or fixed) in Eq 3, the ANOVA and the text.
- If ICC ~ 0, the report must say seeds contributed negligible *additive* block
  variance here (seeds also change the split, so seed x config interaction may
  exist; check residual spread by seed x point_id).
- Document the replicate structure (P31): how the 4 center replicates per block
  differ. If they share a seed they are duplicates; fix.

**Acceptance:** one documented model produces the ICC; no ICC appears in the text that is not in `results/icc.json`.

### Task 6: Proper confirmation experiment (P12, P13)

1. Take the optimum from Task 2 with **integer depth** and predict **at exactly that point**.
2. Run confirmation on **new seeds disjoint from the design seeds** (e.g. 505, 606,
   707, 808, 909, ...), **m >= 10**. Record val RMSE, test RMSE, latency (same
   procedure as Task 7).
3. Pre-register pass criteria in `config.yaml` before running: the observed mean
   must lie inside the 95% prediction interval for the mean of m runs.

```python
def pi_mean_of_m(fit, X, x_row, m, level=0.95):
    # x_row: const=1, terms at x*, block dummies set to 1/5 each
    h  = float(x_row @ np.linalg.inv(X.T @ X) @ x_row)
    t  = stats.t.ppf(1 - (1 - level) / 2, fit.df_resid)
    yh = float(x_row @ fit.params.values)
    half = t * np.sqrt(fit.mse_resid * (1 / m + h))
    return yh, yh - half, yh + half, h
```

- Assert h is identical for Y1 and Y2 when both models contain the same terms;
  if Y2 uses a reduced model, state it and report both h values.
- Compare surrogate predictions of **validation** RMSE to confirmation validation
  RMSE; report **test** RMSE separately as the generalization estimate.
- Report pass/fail as computed, whichever way it falls.

**Acceptance:** `results/confirmation.json`; Table 6 regenerated.

### Task 7: Fair benchmark (P14-P19, P27)

Protocol:

- All methods optimise **validation** RMSE (DOE responses are validation RMSE).
  Test RMSE is computed **only** for each method's final selected config.
- Budget variants: **(A)** 140 total training runs per method; **(B)** equal number
  of *unique configs* (baselines also evaluate each config on the same 5 seeds and use
  the mean). Report both.
- Each method's final incumbent is re-evaluated on **20 fresh seeds**: report mean,
  95% CI for val RMSE, test RMSE, and latency.
- Compare DOE's **recommended x\*** (Task 6), not its best raw design row.
- **Replicate** the stochastic optimizers: >= 20 independent RS and TPE runs
  (different sampler seeds). Report median and IQR of the final incumbent. For DOE,
  bootstrap over seed sets.
- Baselines to include:
  1. Random search (single-objective RMSE).
  2. TPE single-objective RMSE.
  3. **Multi-objective TPE/NSGA-II** (`directions=["minimize","minimize"]`) with
     hypervolume vs the DOE Pareto set (`pymoo.indicators.hv.HV` or a 2-D helper).
  4. **Constrained:** best RMSE subject to latency <= DOE's measured latency (use
     `TPESampler(constraints_func=...)` if the installed Optuna supports it, else a
     documented penalty).
  5. Scalarized TPE maximising the same desirability D.
- **Latency measured in one controlled script** (`scripts/measure_latency.py`) for
  every final config of every method: same process, `nthread=1`, warmup, >= 1000
  single-row calls, >= 5 interleaved repetitions, report median and IQR. Measure both
  `predict` with DMatrix creation and `booster.inplace_predict` (no DMatrix). Compute
  all latency ratios from these **empirical** values.
- **Desirability:** move L, U, s to `config.yaml`, justified by stated engineering
  requirements rather than observed extremes. Add a sensitivity grid over (L, U, s)
  and show how x\* moves. Remove the "95.4% of maximal accuracy" statement; report
  relative RMSE gap to the best method instead.
- Regenerate Table 7 and Fig 4/5 from CSV. Investigate why TPE median RMSE equals
  DOE best (0.4645). Fix table overflow (`tabularx`, `adjustbox`, or drop columns).

**Acceptance:** `results/benchmark.csv`, `results/benchmark_summary.json`, Table 7, Fig 4/5.

### Task 8: Latency modelling (P26)

- Document the measurement procedure and hardware in the Methods section.
- Show complete Table 4 (all terms), plus R^2 and adjusted R^2 for Y2.
- Fit candidate forms: latency vs depth linear, linear+quadratic, and log-latency.
  Compare against a mechanistic expectation (~ n_trees x mean depth, plus fixed
  overhead). Replace the "2^depth" explanation with what the data support.
- Replace Fig 3 (subsample x depth slice; subsample is nearly inert) with a
  latency-vs-depth plot with 95% CI band (optionally depth x ln eta).

**Acceptance:** Table 4 complete; latency claims in text generated from fitted comparisons.

### Task 9 (recommended, if compute allows): Design scope (P25, P28)

- Treat the number of trees properly: tune it with early stopping on the validation
  set (report the best iteration), or add it as a factor.
- Add a short screening design (e.g. 2^(6-2)) including `min_child_weight` and
  `colsample_bytree` to justify the 4-factor choice.
- Repeat the key analysis on at least one more regression dataset, or restrict all
  claims to "California Housing, XGBoost, this configuration".

### Task 10: Figures, tables, formatting (P30)

- Fix raw-LaTeX axis labels: do not nest `$` inside `$`. Use
  `r"Factor A: $\ln(\eta)$"` and set `text.usetex` or mathtext consistently.
- Fig 4: add the confirmation point and baseline incumbents. Fig 5: median and IQR
  over replicate optimizer runs; mark DOE runs in execution order.
- Cite [2] and [8] in the text or remove them. Verify Montgomery chapter/section
  numbers against the edition in the bibliography. Fix "geometric geometry".
- Add a Reproducibility section: versions, hardware, seeds, split, run order, repo
  and commit hash, data availability.

### Task 11: Rewrite claims to match regenerated results

Do this **last**. For each item, change the wording to what the new results support.

| Location | Current claim | Required change |
|----------|---------------|-----------------|
| Abstract | stationary point is a local minimum "just outside the perimeter" | Report real position, the extrapolation flag and the constrained optimum from Task 2. |
| Abstract / 5 | diagnostics "confirm variance homogeneity" | State BP result and the remedy used (transform / robust SEs). |
| Abstract / 8.2 | "comparable predictive power" | Give the measured gap between DOE x\* and baselines with CIs. |
| 2.2, 9(2) | ICC 21.6% "validates RCBD" | Use the Task 5 result; drop causal claim if ICC ~ 0. |
| 4.3 | "As noted by Montgomery" | Remove or cite exact edition and page. |
| 4.3 | Y2 "fully adequate / zero deficiency" | "No significant lack of fit detected (power limited by large pure error)". |
| 5(a) | normality "rejected at alpha=0.01" | Generate from the p-value. |
| 5(b), 5(c) | "perfect variance equality", "proving total absence of drift" | Replace with test statistics and neutral wording. |
| 6.2 | "unique local minimum" | Per Task 1 classification rule. |
| 7.1 | "Pareto-optimal" | "Maximizes D under the surrogate and stated limits"; add sensitivity. |
| 8.1 | "squarely within", "exceptional precision" | Report PI result as computed. |
| 8.2 | "62.2% slower", "95.4% of maximal accuracy" | Use empirical ratios from Task 7; drop the 95.4% sentence. |
| 9 | Four general principles | Scope to this study; remove principle 4 or support it with evidence. |
| Throughout | "proving", "definitively", "perfect", "publication-grade" | Neutral, quantified language. |

### Task 12: Tests and CI

Add `tests/`:

- `test_data.py`: no duplicate (point_id, seed); round-trip coding; ranges.
- `test_phase3.py`: trace == sum(eig); `b + 2B x0 == 0`; SS cross-check; y0_hat consistency; subsample in [0.5,1] check on any reported natural-unit value.
- `test_lof.py`: LoF df == (10, 111) for the 25-point, 5-block design.
- `test_intervals.py`: hat value identical across responses with identical terms.
- `test_no_hardcoded_numbers.py`: fail if result sections of the `.tex` contain numeric literals not produced by an `\input`ed generated file (allow a whitelist for constants like 0.05).
- CI runs the full pipeline from `runs.csv` to `report.pdf`.

---

## 3. Definition of done

- All tests pass; the pipeline regenerates `report.pdf` end-to-end with one command.
- Every number in the report is traceable to a script output.
- `CHANGELOG.md` lists old -> new for every Appendix A value, with a one-line reason.
- No claim in the abstract or conclusion is stronger than the regenerated evidence.

---

## Appendix A: Old reported values (for the changelog)

| Quantity | Old value |
|----------|-----------|
| ICC | 0.2160 |
| F_curvature | 57,468.86 (MS_PE center 4.3855e-6, df 19) |
| R^2 / adj R^2 (Y1, phase 2) | 0.9954 / 0.9947 |
| LoF Y1 | SS_LoF 0.007243, SS_PE 0.001132, F=122.63 (df 6/115) |
| LoF Y2 | SS_LoF 258.86, SS_PE 11029.25, F=0.450, p=0.8438 |
| Shapiro-Wilk | W=0.9779, p=0.0226 |
| Levene | W=0.000332, p=1.0000 |
| Breusch-Pagan | LM=69.90 |
| max Cook's D | 0.16 |
| b | (-0.107040, -0.031773, -0.004079, -0.001536) |
| B diag | (0.089855, 0.022312, 0.000241, 0.001712) |
| b0 | 0.506544 |
| Eigenvalues | 0.102302, 0.021290, 0.001677, 0.000126 |
| x0 | (0.9689, 1.6419, 28.4377, -1.7819), y0_hat=0.3428 |
| x\* | (0.8575, -0.5475, 1.0, -1.0) -> eta 0.2354, depth 4, subsample 1.0, lambda 0.1 |
| Desirability | d1=0.9539, d2=0.8213, D=0.8851; L1=0.4645, U1=0.8173, L2=100.86, U2=196.48 |
| Predictions at x\* | Y1 0.4808 [0.4745, 0.4870]; Y2 117.99 [101.45, 134.52] |
| Confirmation | Y1 0.4880 +- 0.0036; Y2 130.97 +- 4.63 (n=5) |
| Table 7 best RMSE | DOE 0.4645, RS 0.4585, TPE 0.4548 (medians 0.5023, 0.5095, 0.4645) |
| TPE latency | 191.4 us at depth 9 |

## Appendix B: `scripts/verify_report_numbers.py`

Reproduces P1-P6 from the values printed in the PDF. Run first; delete after the
pipeline replaces it.

```python
import numpy as np
b = np.array([-0.107040, -0.031773, -0.004079, -0.001536])
B = np.array([[0.089855, 0.022477, 0.003411, 0.006809],
              [0.022477, 0.022312, 0.001911, 0.000894],
              [0.003411, 0.001911, 0.000241, 0.000494],
              [0.006809, 0.000894, 0.000494, 0.001712]])
b0 = 0.506544
x0_rep = np.array([0.9689, 1.6419, 28.4377, -1.7819])
lam_rep = np.array([0.102302, 0.021290, 0.001677, 0.000126])

print("trace(B) =", np.trace(B), " sum(reported eigs) =", lam_rep.sum())          # P2
print("eig(printed B) =", np.linalg.eigvalsh(B))                                   # P3
print("solve printed system:", np.linalg.solve(B, -b / 2))                         # P4
print("B@x0_rep =", B @ x0_rep, " target:", -b / 2)                                # P4
print("y0_hat from printed values =", b0 + 0.5 * x0_rep @ b)                       # P5
print("subsample at x3=28.4377:", 0.75 + 0.25 * 28.4377,
      " lambda at x4=-1.7819:", np.exp(2.302585 * -1.7819))                        # P6

# P1: SS-implied coefficients (orthogonal terms: SS_lin = 90*beta^2, SS_int = 80*beta^2)
for s in [1.372904, 0.121139, 0.001997, 0.000283]:
    print("|beta_lin| from SS:", np.sqrt(s / 90))
for k, s, printed in [("12", 0.080835, 0.022477), ("13", 0.001861, 0.003411),
                      ("14", 0.007417, 0.006809), ("23", 0.000584, 0.001911),
                      ("24", 0.000128, 0.000894), ("34", 0.000039, 0.000494)]:
    bij = np.sqrt(s / 80)
    print(k, "beta/2 =", round(bij / 2, 6), " beta/sqrt2 =", round(bij / np.sqrt(2), 6), " printed =", printed)
```
