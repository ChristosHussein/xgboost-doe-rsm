"""Retrospective diagnostics of frozen observations; no classifier is trained.

All added analyses are post-hoc. Document-bootstrap uncertainty conditions on
the nine frozen classifiers; it is not uncertainty over fresh optimizer runs.
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import f, pearsonr, spearmanr, t

METHODS = ('static_doe', 'doe', 'tpe')
RESPONSES = ('val_loss', 'latency_us')
FACTORS = ('x1', 'x2', 'x3', 'x4')


def quadratic(x, blocks=None):
    x = np.asarray(x, dtype=float)
    columns = [np.ones(len(x)), *x.T, *(x*x).T]
    columns += [x[:, i]*x[:, j] for i, j in combinations(range(4), 2)]
    if blocks is not None:
        columns += [(np.asarray(blocks) == b).astype(float) for b in sorted(set(blocks))[1:]]
    return np.column_stack(columns)


def coordinate(row):
    return tuple(row['coded_params'][name] for name in FACTORS)


def macro_f1_from_counts(counts):
    """Last axis contains true negatives, false positives, false negatives, true positives."""
    tn, fp, fn, tp = np.moveaxis(np.asarray(counts, dtype=float), -1, 0)
    positive = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp), where=(2*tp+fp+fn) != 0)
    negative = np.divide(2*tn, 2*tn+fp+fn, out=np.zeros_like(tn), where=(2*tn+fp+fn) != 0)
    return .5*(positive+negative)


def paired_bootstrap(final, *, repetitions=2000, seed=20261030):
    ordered = [next(r for r in final if r['method'] == method and r['replicate'] == repeat)
               for method in METHODS for repeat in range(3)]
    labels = np.asarray(ordered[0]['validation_labels'], dtype=int)
    if not all(np.array_equal(labels, r['validation_labels']) for r in ordered):
        raise ValueError('A paired bootstrap requires identical ordered test labels')
    predictions = np.column_stack([np.asarray(r['validation_probabilities']) >= .5 for r in ordered])
    # Joint pattern counts yield exactly the ordinary paired nonparametric
    # document bootstrap, without allocating thousands of full resampled texts.
    patterns, counts = np.unique(np.column_stack([labels, predictions]), axis=0, return_counts=True)
    classifications = np.empty((len(patterns), len(ordered), 4), dtype=int)
    for j in range(len(ordered)):
        y, p = patterns[:, 0], patterns[:, j+1]
        classifications[:, j] = np.column_stack([(y == 0)&(p == 0), (y == 0)&(p == 1),
                                                 (y == 1)&(p == 0), (y == 1)&(p == 1)])
    confusion = np.tensordot(counts, classifications, axes=1)
    point = macro_f1_from_counts(confusion).reshape(len(METHODS), 3)
    rng = np.random.default_rng(seed)
    boot_counts = rng.multinomial(len(labels), counts/counts.sum(), size=repetitions)
    boot_scores = macro_f1_from_counts(np.tensordot(boot_counts, classifications, axes=1))
    boot_scores = boot_scores.reshape(repetitions, len(METHODS), 3).mean(axis=2)
    comparisons = []
    for i, j in combinations(range(len(METHODS)), 2):
        delta = boot_scores[:, i]-boot_scores[:, j]
        comparisons.append(dict(first=METHODS[i], second=METHODS[j],
            difference=float(point[i].mean()-point[j].mean()),
            interval_95=np.quantile(delta, [.025, .975]).tolist(),
            bootstrap_sd=float(delta.std(ddof=1)),
            per_search_differences=(point[i]-point[j]).tolist()))
    return dict(status='POST_HOC', seed=seed, repetitions=repetitions,
        test_documents=len(labels), fixed_search_repeats=3, frozen_models=len(ordered),
        sampling='Ordinary paired document resampling; identical sampled documents across all nine classifiers.',
        estimand='Difference in mean macro F1 across each method\'s three frozen classifiers.',
        limitation='Intervals condition on trained models and observed document population. They do not estimate optimizer-run uncertainty or repair prior test exposure; no equivalence margin is tested.',
        per_search_f1={m: point[i].tolist() for i, m in enumerate(METHODS)}, comparisons=comparisons)


def grouped_points(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[coordinate(row)][row['block_id']].append(row)
    return [dict(coded=list(key), loss=float(np.mean([np.mean([r['val_loss'] for r in records])
                       for records in blocks.values()])),
                 latency=float(np.mean([np.mean([r['latency_us'] for r in records])
                       for records in blocks.values()]))) for key, blocks in groups.items()]


def hypervolume(points, reference):
    points = np.asarray([[p['loss'], p['latency']] for p in points])
    points = points[(points[:, 0] < reference[0])&(points[:, 1] < reference[1])]
    if not len(points):
        return 0.
    xs = sorted(set(points[:, 0]))+[reference[0]]
    return float(sum((b-a)*(reference[1]-points[points[:, 0] <= a, 1].min()) for a, b in zip(xs[:-1], xs[1:])))


def fit_summary(frame, response):
    x = frame[list(FACTORS)].to_numpy()
    y = np.asarray(response, dtype=float)
    Q = quadratic(x, frame.block.to_numpy())
    _, treatment = np.unique(x, axis=0, return_inverse=True)
    A = np.column_stack([np.ones(len(y)), pd.get_dummies(frame.block, drop_first=True, dtype=float),
                         pd.get_dummies(treatment, drop_first=True, dtype=float)])
    beta = np.linalg.lstsq(Q, y, rcond=None)[0]
    residual = y-Q@beta
    additive_error = y-A@np.linalg.lstsq(A, y, rcond=None)[0]
    rank_q, rank_a = np.linalg.matrix_rank(Q), np.linalg.matrix_rank(A)
    ss_residual, ss_additive = float(residual@residual), float(additive_error@additive_error)
    df_lof, df_pooled = int(rank_a-rank_q), int(len(y)-rank_a)
    statistic = ((ss_residual-ss_additive)/df_lof)/(ss_additive/df_pooled)
    return dict(n=len(y), quadratic_columns=Q.shape[1], quadratic_rank=int(rank_q),
        additive_rank=int(rank_a), rank_deficient=bool(rank_q < Q.shape[1]),
        r_squared=float(1-ss_residual/np.sum((y-y.mean())**2)),
        lack_of_fit_f=float(statistic), lack_of_fit_p=float(f.sf(statistic, df_lof, df_pooled)),
        df_lack_of_fit=df_lof, df_pooled_additive=df_pooled,
        ss_lack_of_fit=float(ss_residual-ss_additive), ss_pooled_additive=ss_additive)


def prediction_diagnostics(records):
    errors = np.asarray([r['observed']-r['predicted'] for r in records])
    boundary = np.asarray([r['boundary'] for r in records])
    return dict(n=len(records), unique_coordinates=len({r['config_id'] for r in records}),
        rmse=float(np.sqrt(np.mean(errors**2))), mean_bias=float(errors.mean()),
        coverage=float(np.mean([r['lower'] <= r['observed'] <= r['upper'] for r in records])),
        boundary_n=int(boundary.sum()), boundary_bias=float(errors[boundary].mean()),
        interior_bias=float(errors[~boundary].mean()))


def build(root: Path):
    root = Path(root)
    source_hashes = {}
    def source(name):
        path = root/name
        with path.open('rb') as stream:
            source_hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return path
    def read(name):
        return json.loads(source(name).read_text(encoding='utf-8'))
    study = 'results/phase_b/imdb/'
    contract = read(study+'measurement_contract.json'); config = contract['config']
    final = read(study+'final_results.json')['results']
    calibration = read('results/phase_b_calibration/imdb_pilot.json')['observations']
    rows = []
    for line in source(study+'evaluations.jsonl').open(encoding='utf-8'):
        record = json.loads(line)
        rows.append({key: record[key] for key in ['config_id', 'coded_params', 'block_id',
                     'val_loss', 'latency_us', 'method', 'replicate', 'stage', 'elapsed_seconds',
                     'training_seconds', 'profiling_seconds']})
    search = {(method, repeat): [r for r in rows if r['method'] == method and r['replicate'] == repeat
                                and r['stage'] != 'confirmation'] for method in METHODS for repeat in range(3)}
    points = {key: grouped_points(value) for key, value in search.items()}
    reserved = contract['confirmation']
    coords = np.asarray([[p['coded'][k] for k in FACTORS] for p in reserved])
    boundary = np.max(np.abs(coords), axis=1) >= .9
    confirmation = defaultdict(list)
    cost = {}
    explanation = {}
    for method in METHODS:
        cost[method], explanation[method] = [], []
        for repeat in range(3):
            sub = search[method, repeat]
            X = quadratic([coordinate(r) for r in sub], [r['block_id'] for r in sub])
            if np.linalg.matrix_rank(X) != X.shape[1]:
                raise ValueError('Main-study quadratic surface must have full column rank')
            Z = np.column_stack([quadratic(coords), np.full((len(coords), config['n_nuisance_blocks']-1), 1/config['n_nuisance_blocks'])])
            overhead = read(study+f'overhead_{repeat}_{method}.json')['seconds']
            cost[method].append(dict(repeat=repeat, search_fits=len(sub),
                evaluation_seconds=float(sum(r['elapsed_seconds'] for r in sub)),
                training_seconds=float(sum(r['training_seconds'] for r in sub)),
                profiling_seconds=float(sum(r['profiling_seconds'] for r in sub)),
                overhead_seconds=overhead, total_seconds=float(sum(r['elapsed_seconds'] for r in sub)+overhead)))
            for response in RESPONSES:
                y = np.asarray([r[response] for r in sub]);beta = np.linalg.lstsq(X, y, rcond=None)[0]
                error = y-X@beta;degrees = len(y)-X.shape[1];mse = error@error/degrees
                prediction = Z@beta
                se = np.sqrt(mse*(1/config['n_nuisance_blocks']+np.einsum('ij,jk,ik->i', Z, np.linalg.inv(X.T@X), Z)))
                half = t.ppf(.975, degrees)*se
                gp = read(study+f'gp_confirmation_{repeat}_{method}_{response}.json')
                observed = [np.mean([r[response] for r in rows if r['method'] == method and r['replicate'] == repeat
                                    and r['stage'] == 'confirmation' and r['config_id'] == p['point_id']]) for p in reserved]
                if not np.allclose(observed, gp['observed'], atol=1e-12, rtol=1e-12):
                    raise ValueError('GP confirmation means do not match raw fold outcomes')
                for i, p in enumerate(reserved):
                    common = dict(repeat=repeat, config_id=p['point_id'], boundary=bool(boundary[i]), observed=float(observed[i]))
                    confirmation[method+':RSM:'+response].append(dict(common, predicted=float(prediction[i]), lower=float(prediction[i]-half[i]), upper=float(prediction[i]+half[i])))
                    confirmation[method+':GP:'+response].append(dict(common, predicted=gp['predicted'][i], lower=gp['lower'][i], upper=gp['upper'][i]))
            suffix = str(config['adaptive_cycles']) if method == 'doe' else 'final'
            snapshot = read(study+f'snapshot_{repeat}_{method}_{suffix}.json')['val_loss']
            anova = np.asarray(snapshot['attribution']['interaction_matrix_ss'])[np.triu_indices(4, 1)]
            shap = np.asarray(snapshot['shap']['gp']['interaction_matrix_shap'])[np.triu_indices(4, 1)]
            omit = int(np.argmax(anova));keep = np.arange(len(anova)) != omit
            keep_two = ~np.isin(np.arange(len(anova)), np.argsort(anova)[-2:])
            explanation[method].append(dict(repeat=repeat, pair_count=len(anova),
                pearson=float(pearsonr(anova, shap).statistic), spearman=float(spearmanr(anova, shap).statistic),
                largest_anova_pair_index=omit, largest_anova_share=float(anova.max()/anova.sum()),
                pearson_without_largest_anova_pair=float(pearsonr(anova[keep], shap[keep]).statistic),
                two_largest_anova_share=float(np.sort(anova)[-2:].sum()/anova.sum()),
                pearson_without_two_largest_anova_pairs=float(pearsonr(anova[keep_two], shap[keep_two]).statistic)))
    capacity = [r for r in calibration if r['n_trees'] == config['n_trees']]
    matched = float(np.median([r['latency']['predict_median_us'] for r in capacity]))
    feasibility = {}
    for method in METHODS:
        selected = [r for r in final if r['method'] == method]
        latencies = [r['latency']['predict_median_us'] for r in selected]
        feasibility[method] = dict(selected_latencies_us=latencies, selected_count=len(selected),
            selected_under_pooled=int(sum(value <= config['latency_threshold_us'] for value in latencies)),
            selected_under_capacity_matched=int(sum(value <= matched for value in latencies)),
            searched_under_pooled=[sum(p['latency'] <= config['latency_threshold_us'] for p in points[method, repeat]) for repeat in range(3)],
            searched_under_capacity_matched=[sum(p['latency'] <= matched for p in points[method, repeat]) for repeat in range(3)])
    sensitivities = []
    selections = {(r['method'], r['replicate']): coordinate(r) for r in read(study+'selected.json')}
    for loss_ref in [.15, .20, .30, config['hv_ref_loss']]:
        for latency_ref in [200., 250., config['hv_ref_latency_us'], 500.]:
            methods = {}
            for method in METHODS:
                selected = [max(points[method, repeat], key=lambda p: max(0., loss_ref-p['loss'])*max(0., latency_ref-p['latency'])) for repeat in range(3)]
                hv = [hypervolume(points[method, repeat], (loss_ref, latency_ref)) for repeat in range(3)]
                methods[method] = dict(mean_hypervolume=float(np.mean(hv)), sd_hypervolume=float(np.std(hv, ddof=1)),
                    mean_selected_validation_f1=float(np.mean([1-p['loss'] for p in selected])),
                    mean_selected_latency_us=float(np.mean([p['latency'] for p in selected])),
                    selected_depths=[int(round(6+3*p['coded'][1])) for p in selected],
                    selected_coded=[p['coded'] for p in selected],
                    changed_selections=sum(tuple(p['coded']) != selections[method, repeat] for repeat, p in enumerate(selected)))
            sensitivities.append(dict(loss_reference=loss_ref, latency_reference_us=latency_ref, methods=methods))
    housing = pd.read_csv(source('phase_a_rsm_blocking/results/runs.csv'))
    low = housing.eta.min();restricted = housing[~np.isclose(housing.eta, low)].copy()
    phase_a = dict(raw=fit_summary(housing, housing.val_rmse), log_response=fit_summary(housing, np.log(housing.val_rmse)),
        restricted_domain=fit_summary(restricted, restricted.val_rmse), excluded_runs=len(housing)-len(restricted),
        retained_learning_rates=sorted(float(x) for x in restricted.eta.unique()),
        interpretation='Post-hoc sensitivity analyses. The restricted learning-rate design aliases the linear and squared learning-rate columns; rank-aware fits do not identify these effects separately. A changed lack-of-fit statistic is not a causal attribution to the excluded boundary.',
        denominator='Structural lack of fit versus pooled additive treatment-plus-block residual, matching the historical decomposition; the denominator is not center pure error.')
    evaluations = pd.read_csv(source('phase_a_rsm_blocking/results/revision_v2/full_run_001/final_evaluations.csv'))
    summary = pd.read_csv(source('phase_a_rsm_blocking/results/revision_v2/full_run_001/final_summary.csv'))
    within = evaluations.groupby('selection_id').test_rmse.agg(['mean', 'std', 'count'])
    retraining = {}
    for method, selected in summary.groupby('optimizer'):
        values = within.loc[selected.selection_id]
        retraining[method] = dict(selections=len(selected), evaluations_per_selection=sorted(int(v) for v in values['count'].unique()),
            mean_test_rmse=float(values['mean'].mean()), between_selection_sd=float(values['mean'].std(ddof=1)) if len(values)>1 else None,
            mean_within_selection_sd=float(values['std'].mean()),
            root_mean_within_selection_variance=float(np.sqrt(np.mean(values['std']**2))))
    return dict(status='POST_HOC', protocol='Saved-evidence diagnostics; no retraining, no new test selection.',
        paired_document_bootstrap=paired_bootstrap(final),
        symmetric_confirmation=dict(target='Mean response across the same five existing development folds.',
            nominal_level=.95, unique_coordinates=len(reserved), repeats=config['search_replicates'],
            interval_interpretation={'RSM':'Working-model PI for an average of five new independent residuals in known fixed blocks: t * sqrt(MSE*(1/5+leverage)).',
                                     'GP':'Saved GP model-based predictive interval for the block-averaged response; includes fitted GP white-noise variance.'},
            limitation='Intervals have the same response target but different variance models. Overlapping training folds and repeated use of the same coordinates preclude treating these records as independent datasets.',
            summaries={name: prediction_diagnostics(value) for name, value in confirmation.items()}, records=dict(confirmation)),
        capacity_matching=dict(original_pooled_threshold_us=config['latency_threshold_us'], selected_tree_budget=config['n_trees'],
            calibration_latencies_us=[r['latency']['predict_median_us'] for r in capacity],
            capacity_matched_median_us=matched, calibration_learning_rates=[r['natural_params']['learning_rate'] for r in capacity],
            interpretation='Post-hoc descriptive comparison across different fixed input documents and configurations, not a new deployment requirement or a proof that any threshold is structurally unreachable.', methods=feasibility),
        reference_sensitivity=dict(status='POST_HOC', grid_rule='Fixed diagnostic loss references 0.15, 0.20, 0.30 and original 1.0; latency references 200, 250, original 358.3 and 500 microseconds. No application requirement is inferred.',
            target='Selection and hypervolume recomputed on observed development configurations; no test-set evaluation of newly selected points.', grid=sensitivities),
        search_cost=dict(per_repeat=cost, summaries={method:dict(mean_total_seconds=float(np.mean([r['total_seconds'] for r in values])),
            sd_total_seconds=float(np.std([r['total_seconds'] for r in values], ddof=1))) for method, values in cost.items()},
            scope='Search evaluation elapsed time plus recorded design/surrogate/explanation overhead; excludes confirmation, final refits and local word study. First-use GPU initialization and randomized serial method order affect these descriptive costs.'),
        interaction_leverage=dict(pair_order=[list(p) for p in combinations(FACTORS, 2)], per_repeat=explanation,
            summaries={method: {key: float(np.mean([r[key] for r in values])) for key in
                ['pearson', 'spearman', 'largest_anova_share', 'pearson_without_largest_anova_pair',
                 'two_largest_anova_share', 'pearson_without_two_largest_anova_pairs']}
                for method, values in explanation.items()},
            limitation='Descriptive correlations of six strengths fitted to shared observations, not independent validation of explanation correctness.'),
        phase_a_diagnostics=phase_a, phase_a_retraining=retraining, source_sha256=source_hashes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('posthoc_results.json'))
    args = parser.parse_args()
    result = build(Path(__file__).resolve().parents[1])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print('Saved post-hoc diagnostics:', args.output)
