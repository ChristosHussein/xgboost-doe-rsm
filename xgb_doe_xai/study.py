"""Logged sequential experiments with reserved configurations and matched fit budgets."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import warnings
from importlib.metadata import version
from datetime import datetime, timezone
import subprocess

import numpy as np
import optuna
from scipy.stats import norm

from .config import FACTOR_NAMES, canonical_configuration
from .design import DesignPoint, generate_fccd_points, select_adaptive_infill_points
from .surrogates import PolynomialRSMSurrogate, GaussianProcessSurrogate
from .xai import extract_anova_attributions, compute_surrogate_shap_attributions, compare_anova_vs_shap


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()),
                          encoding='utf-8')


def key(coded):
    return tuple(coded[name] for name in FACTOR_NAMES)


def git_provenance():
    """A wheel can run without Git; record absence rather than invent a revision."""
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent.parent,
                                  text=True, capture_output=True, check=False)
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=Path(__file__).parent.parent,
                                text=True, capture_output=True, check=False)
    except OSError:
        return {'git_revision': 'UNAVAILABLE', 'git_worktree_dirty': None}
    return {'git_revision': revision.stdout.strip() if revision.returncode == 0 else 'UNAVAILABLE',
            'git_worktree_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None}


def source_hashes():
    root = Path(__file__).parent
    files = list(root.glob('*.py')) + [root.parent/'latency.py', root.parent/'run_explainable_doe_demo.py',
                                     root.parent/'requirements.txt']
    return {p.relative_to(root.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def fit_surrogates(rows):
    x = np.array([list(key(r['coded_params'])) for r in rows])
    blocks = np.array([r['block_id'] for r in rows])
    unique, ids = np.unique(x, axis=0, return_inverse=True)
    surfaces, gps = {}, {}
    for response in ('val_loss', 'latency_us'):
        y = np.array([r[response] for r in rows])
        surfaces[response] = PolynomialRSMSurrogate(True).fit(x, y, blocks)
        means = np.array([y[ids == i].mean() for i in range(len(unique))])
        gps[response] = GaussianProcessSurrogate().fit(unique, means)
    return surfaces, gps


def explanation_snapshot(rows, background, explain_at):
    """Compare both surrogate explanations on a common predeclared background."""
    with warnings.catch_warnings(record=True) as fitting_warnings:
        warnings.simplefilter('always')
        surfaces, gps = fit_surrogates(rows)
    output = {}
    for response in surfaces:
        rsm, gp = surfaces[response], gps[response]
        anova = extract_anova_attributions(rsm)
        comparisons, explanations, curvature = {}, {}, {}
        for label, model in [('rsm', rsm), ('gp', gp)]:
            shap = compute_surrogate_shap_attributions(model, background, explain_at)
            comparisons[label] = asdict(compare_anova_vs_shap(anova, shap))
            explanations[label] = asdict(shap)
            # SHAP main-effect dependence at three fixed anchors, averaged over
            # the SAME background. The second difference isolates curvature
            # along this slice, without asserting equality to ANOVA partial SS.
            profiles = {}
            for j, name in enumerate(FACTOR_NAMES):
                anchors = []
                for value in (-1., 0., 1.):
                    points = background.copy()
                    points[:, j] = value
                    attrs = compute_surrogate_shap_attributions(model, background, points)
                    anchors.append(float(attrs.raw_interaction_values[:, j, j].mean()))
                profiles[name] = dict(anchors=anchors,
                                      second_difference=anchors[0]-2*anchors[1]+anchors[2])
            curvature[label] = profiles
        output[response] = dict(anova=asdict(rsm.anova_), attribution=asdict(anova),
                                comparisons=comparisons, shap=explanations,
                                curvature=curvature, gp_kernel=str(gp.gp.kernel_),
                                rsm_coefficients=rsm.beta_.tolist(),
                                fitting_warnings=[str(w.message) for w in fitting_warnings])
        # Split-specific fits distinguish stability across development partitions
        # from optimizer-seed variability in the combined fixed-block analysis.
        output[response]['by_block'] = []
        for block in sorted({r['block_id'] for r in rows}):
            subset = [r for r in rows if r['block_id'] == block]
            xb = np.array([list(key(r['coded_params'])) for r in subset])
            yb = np.array([r[response] for r in subset])
            unique, ids = np.unique(xb, axis=0, return_inverse=True)
            with warnings.catch_warnings(record=True) as block_warnings:
                warnings.simplefilter('always')
                rb = PolynomialRSMSurrogate(False).fit(xb, yb)
                gb = GaussianProcessSurrogate().fit(unique,
                    np.array([yb[ids == i].mean() for i in range(len(unique))]))
            ab = extract_anova_attributions(rb)
            sb = compute_surrogate_shap_attributions(gb, background, explain_at)
            output[response]['by_block'].append(dict(block_id=block,
                anova=asdict(ab), shap=asdict(sb), comparison=asdict(compare_anova_vs_shap(ab, sb)),
                fitting_warnings=[str(w.message) for w in block_warnings]))
    return output, surfaces, gps


def run_study(config, evaluator, output_dir, *, dataset_id, dataset_sha256,
              evidence_kind='research', final_evaluator=None):
    """Run a predeclared budget; evaluator(point, block, seed, config) fits once.

    Each method evaluates each proposed coordinate on every development block.
    Confirmation coordinates are reserved before search and never used for fitting.
    A callable fixture tests orchestration without pretending to be real experiments.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if evidence_kind not in ('research', 'synthetic_fixture', 'smoke'):
        raise ValueError('Unknown evidence kind')
    if evidence_kind == 'research' and (config.mode != 'research' or final_evaluator is None):
        raise ValueError('Research requires the full protocol and a protected final evaluator')
    initial = generate_fccd_points(n_center=config.center_replicates)
    background = np.unique(np.array([p.vector for p in initial]), axis=0)
    rng = np.random.default_rng(config.random_seed)
    reserved = []
    excluded = {key(p.coded) for p in initial}
    while len(reserved) < config.confirmation_points:
        coded = canonical_configuration(dict(zip(FACTOR_NAMES, rng.uniform(-1, 1, 4))))
        if len(reserved) % 2 == 0:
            coded['x1'] = -1. if len(reserved) % 4 == 0 else 1.
        if key(coded) not in excluded:
            reserved.append(DesignPoint(f'CONF_{len(reserved)}', 'confirmation', coded, 0))
            excluded.add(key(coded))
    explain_at = np.array([p.vector for p in reserved])
    coordinates = len(initial) + config.adaptive_cycles*config.infill_per_cycle
    contract = dict(config=asdict(config), dataset_id=dataset_id, dataset_sha256=dataset_sha256,
                    evidence_kind=evidence_kind, python=sys.version, platform=platform.platform(),
                    declared_at_utc=datetime.now(timezone.utc).isoformat(),
                    **git_provenance(),
                    dependencies={name: version(name) for name in ('numpy', 'scipy', 'pandas',
                        'scikit-learn', 'xgboost', 'optuna', 'matplotlib')},
                    source_hashes=source_hashes(), confirmation=[asdict(p) for p in reserved],
                    background=background.tolist(),
                    search_fits_per_method=coordinates*config.n_nuisance_blocks*config.search_replicates,
                    confirmation_fits=2*len(reserved)*config.n_nuisance_blocks*config.search_replicates,
                    final_fits=2*config.search_replicates if final_evaluator else 0,
                    selection_rule='largest dominated rectangle to predeclared reference; observed coordinate',
                    acquisition_interpretation='EI uses an equally normalized loss/latency scalarization with independent GP predictive variances; not EHVI',
                    metrics={
                        'loss': {'source': 'evaluations.jsonl:val_loss', 'unit': '1 - macro F1', 'aggregation': 'mean across blocks'},
                        'latency': {'source': 'evaluations.jsonl:latency_us', 'unit': 'us/document', 'aggregation': 'mean across blocks'},
                        'hypervolume': {'source': 'search rows, block-averaged by coordinate', 'reference': [config.hv_ref_loss, config.hv_ref_latency_us]},
                        'feasibility': {'source': 'same block-averaged search rows', 'limit_us': config.latency_threshold_us},
                        'coverage': {'source': 'confirmation.json', 'definition': 'fraction of reserved observations inside fixed-block OLS PI'},
                        'explanations': {'source': 'snapshot_*.json', 'definition': 'exact interventional Shapley values/interactions on declared background'},
                        'final_test': {'source': 'final_results.json:results', 'definition': 'test macro F1 and log loss reconstructed from final labels/probabilities; no post-test selection'},
                    })
    write_json(output/'measurement_contract.json', contract)  # BEFORE first model fit.
    write_json(output/'status.json', {'status': 'RUNNING'})
    rows, confirmations, selections = [], [], []
    log_path = output/'evaluations.jsonl'

    def evaluate(points, method, replicate, stage, ordinal_start=0):
        schedule = [(i, b) for i in range(len(points)) for b in range(config.n_nuisance_blocks)]
        rng.shuffle(schedule)
        added = []
        for i, b in schedule:
            point = points[i]
            seed = config.random_seed + replicate*100000 + (ordinal_start+i)*config.n_nuisance_blocks + b
            started = time.perf_counter()
            result = evaluator(point, b, seed, config)
            record = asdict(result)
            if key(record['coded_params']) != key(point.coded):
                raise ValueError('Evaluator trained a configuration different from its acquisition')
            record.update(method=method, replicate=replicate, stage=stage,
                          latency_us=result.latency.predict_median_us,
                          elapsed_seconds=time.perf_counter()-started)
            rows.append(record)
            added.append(record)
            with log_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        return added

    for replicate in range(config.search_replicates):
        methods = ['doe', 'tpe']
        rng.shuffle(methods)  # Serial profiling, randomized method order across repeats.
        for method in methods:
            search = []
            overhead = time.perf_counter()
            if method == 'doe':
                points = list(initial)
                search += evaluate(initial, method, replicate, 'initial')
                for cycle in range(config.adaptive_cycles + 1):
                    snapshot, surfaces, gps = explanation_snapshot(search, background, explain_at)
                    write_json(output/f'snapshot_{replicate}_{method}_{cycle}.json', snapshot)
                    if cycle == config.adaptive_cycles:
                        break
                    # Pareto-oriented scalarization with predeclared scales.
                    # EI is scalarized loss/latency EI, not expected HV improvement.
                    gp_loss, gp_lat = gps['val_loss'], gps['latency_us']
                    def uncertainty(x):
                        return np.sqrt(gp_loss.predict(x, True)[1]**2/config.hv_ref_loss**2 +
                                       gp_lat.predict(x, True)[1]**2/config.hv_ref_latency_us**2)
                    def mean(x):
                        return gp_loss.predict(x)/config.hv_ref_loss + gp_lat.predict(x)/config.hv_ref_latency_us
                    def discrepancy(x):
                        return (np.abs(surfaces['val_loss'].predict(x)-gp_loss.predict(x))/config.hv_ref_loss +
                                np.abs(surfaces['latency_us'].predict(x)-gp_lat.predict(x))/config.hv_ref_latency_us)
                    new = select_adaptive_infill_points(
                        points+reserved, uncertainty, n_infill=config.infill_per_cycle,
                        random_seed=config.random_seed+replicate*100+cycle,
                        criterion=config.acquisition, surrogate_discrepancy_fn=discrepancy,
                        surrogate_mean_fn=mean,
                        best_observed=min(r['loss']/config.hv_ref_loss+r['latency']/config.hv_ref_latency_us
                                          for r in group_search(search)),
                        cycle_id=cycle+1)
                    write_json(output/f'acquisition_{replicate}_{cycle+1}.json',
                               dict(criterion=config.acquisition, points=[asdict(p) for p in new],
                                    source_snapshot=f'snapshot_{replicate}_{method}_{cycle}.json'))
                    search += evaluate(new, method, replicate, f'adaptive_{cycle+1}', len(points))
                    points += new
            else:
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                optimizer = optuna.create_study(directions=['minimize', 'minimize'],
                                                sampler=optuna.samplers.TPESampler(seed=config.random_seed+replicate))
                for trial_index in range(coordinates):
                    trial = optimizer.ask()
                    coded = canonical_configuration(dict(
                        x1=trial.suggest_float('x1', -1, 1),
                        x2=(trial.suggest_int('max_depth', 3, 9)-6)/3,
                        x3=trial.suggest_float('x3', -1, 1), x4=trial.suggest_float('x4', -1, 1)))
                    if key(coded) in {key(p.coded) for p in reserved}:
                        raise ValueError('TPE proposed a reserved confirmation coordinate')
                    trial_rows = evaluate([DesignPoint(f'TPE_{trial_index}', 'tpe', coded, 0)],
                                          method, replicate, 'search', trial_index)
                    search += trial_rows
                    optimizer.tell(trial, [np.mean([r['val_loss'] for r in trial_rows]),
                                           np.mean([r['latency_us'] for r in trial_rows])])
                snapshot, surfaces, gps = explanation_snapshot(search, background, explain_at)
                write_json(output/f'snapshot_{replicate}_{method}_final.json', snapshot)
            write_json(output/f'overhead_{replicate}_{method}.json',
                       dict(seconds=max(0., time.perf_counter()-overhead-sum(r['elapsed_seconds'] for r in search)),
                            includes='design, surrogate fits, explanation and sampling; excludes model evaluations'))
            confirm = evaluate(reserved, method, replicate, 'confirmation', coordinates)
            for response, rsm in surfaces.items():
                x = np.array([list(key(r['coded_params'])) for r in confirm])
                block_ids = np.array([r['block_id'] for r in confirm])
                pred, low, high = rsm.predict_interval(x, block_ids=block_ids)
                means, sd = gps[response].predict(explain_at, return_std=True)
                for i, record in enumerate(confirm):
                    confirmations.append(dict(method=method, replicate=replicate, response=response,
                        config_id=record['config_id'], block_id=record['block_id'],
                        observed=record[response], predicted=float(pred[i]),
                        lower=float(low[i]), upper=float(high[i]),
                        boundary=max(abs(v) for v in record['coded_params'].values()) >= .9))
                observed_mean = [np.mean([r[response] for r in confirm if r['config_id'] == p.point_id]) for p in reserved]
                write_json(output/f'gp_confirmation_{replicate}_{method}_{response}.json',
                           dict(observed=observed_mean, predicted=means, lower=means-1.96*sd,
                                upper=means+1.96*sd, target='block-averaged response at reserved configurations'))
            grouped = group_search(search)
            selected = max(grouped, key=lambda r: max(0, config.hv_ref_loss-r['loss'])*max(0, config.hv_ref_latency_us-r['latency']))
            selections.append(dict(method=method, replicate=replicate, coded_params=selected['coded_params']))
    write_json(output/'confirmation.json', confirmations)
    write_json(output/'selected.json', selections)
    if final_evaluator is not None:
        from .execution import final_classification
        final_evaluator.freeze(output/'finalized_selections.json', selections, dataset_id, asdict(config))
        final_evaluator.evaluate(output/'finalized_selections.json', output/'final_results.json', final_classification)
    write_json(output/'status.json', {'status': 'COMPLETED'})
    from .study_reporting import publish_study
    summary = publish_study(output)
    return summary


def group_search(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(key(row['coded_params']), []).append(row)
    return [dict(coded_params=records[0]['coded_params'],
                 loss=float(np.mean([r['val_loss'] for r in records])),
                 latency=float(np.mean([r['latency_us'] for r in records])))
            for records in grouped.values()]


def check_study(output_dir):
    from .study_reporting import check_study as check
    return check(output_dir)
