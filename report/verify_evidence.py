"""Independent reconstruction of the paper's principal experimental quantities.

This module does not import the experiment's metric or reporting implementations.
It checks primary predictions, timings and independently fitted response models.
"""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json
import re
import ast
import hashlib
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import f as f_distribution, pearsonr, spearmanr, t


def audit(root: Path, registry):
    metrics = registry['metrics']
    reconstructed = set()

    def read(name):
        return json.loads((root / name).read_text(encoding='utf-8'))

    def compare(name, value, tolerance=1e-9):
        expected = metrics[name]
        if not np.isclose(value, expected['value'], rtol=1e-8, atol=tolerance):
            raise ValueError(f'{name}: reconstructed {value}, registered {expected["value"]}')
        reconstructed.add(name)

    def scores(row):
        labels = np.asarray(row['validation_labels'])
        probabilities = np.asarray(row['validation_probabilities'], dtype=np.float32)
        prediction = probabilities >= .5
        tp = np.sum((labels == 1) & prediction)
        tn = np.sum((labels == 0) & ~prediction)
        fp = np.sum((labels == 0) & prediction)
        fn = np.sum((labels == 1) & ~prediction)
        f1 = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
        epsilon = np.finfo(np.float32).eps
        p = np.clip(probabilities.astype(float), epsilon, 1 - epsilon)
        loss = -np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))
        return f1, loss

    def coordinate(row):
        return tuple(row['coded_params'][f'x{i}'] for i in range(1, 5))

    def design(x, blocks=None):
        x = np.asarray(x)
        columns = [np.ones(len(x)), *x.T, *(x * x).T]
        columns += [x[:, i] * x[:, j] for i, j in combinations(range(4), 2)]
        if blocks is not None:
            columns += [(np.asarray(blocks) == b).astype(float) for b in sorted(set(blocks))[1:]]
        return np.column_stack(columns)

    contract = read('results/phase_b/imdb/measurement_contract.json')
    config = contract['config']
    rows = []
    for line in (root / 'results/phase_b/imdb/evaluations.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        f1, loss = scores(row)
        if abs(f1 - row['val_macro_f1']) > 1e-12 or abs(loss - row['val_logloss']) > 1e-7:
            raise ValueError('Development metrics differ from saved probabilities')
        timing = row['latency']
        if (np.median(timing['predict_raw_us']) != row['latency_us']
                or len(timing['predict_raw_us']) != config['timed_inferences']
                or timing['n_warmup'] != config['warmup_inferences']
                or not timing['affinity_applied']):
            raise ValueError('Development timing protocol or median differs')
        for key in ['validation_labels', 'validation_probabilities', 'latency']:
            row.pop(key)
        rows.append(row)

    methods = {'static_doe': 'A', 'doe': 'B', 'tpe': 'C'}
    final = read('results/phase_b/imdb/final_results.json')['results']
    selected_depths = {r['natural_params']['max_depth'] for r in final}
    if len(selected_depths) != 1:
        raise ValueError('Selected-depth claim is contradicted')
    compare('BSelectedDepth', next(iter(selected_depths)))
    for method, letter in methods.items():
        values = []
        for row in [r for r in final if r['method'] == method]:
            f1, loss = scores(row)
            latency = np.median(row['latency']['predict_raw_us'])
            if abs(f1 - row['val_macro_f1']) > 1e-12 or abs(loss - row['val_logloss']) > 1e-7:
                raise ValueError('Final test metrics differ from saved probabilities')
            values.append((f1, loss, latency))
            prefix = 'Selection' + method.replace('_', '') + chr(65 + row['replicate'])
            compare(prefix + 'Repeat', row['replicate'] + 1)
            for key, suffix in [('learning_rate', 'Rate'), ('max_depth', 'Depth'),
                                ('subsample', 'Sample'), ('reg_lambda', 'Lambda')]:
                compare(prefix + suffix, row['natural_params'][key])
        values = np.asarray(values)
        for column, suffix in enumerate(['Fone', 'Logloss', 'Latency']):
            compare('B' + letter + suffix, values[:, column].mean())
        compare('B' + letter + 'FoneSd', values[:, 0].std(ddof=1))
        if np.any(values[:, 2] <= config['latency_threshold_us']):
            raise ValueError('Final feasibility claim is contradicted')

    confirmations = read('results/phase_b/imdb/confirmation.json')
    for method, letter in methods.items():
        hypervolumes, agreements = [], []
        for repeat in range(config['search_replicates']):
            search = [r for r in rows if r['method'] == method and r['replicate'] == repeat
                      and r['stage'] != 'confirmation']
            reserved = [r for r in rows if r['method'] == method and r['replicate'] == repeat
                        and r['stage'] == 'confirmation']
            groups = defaultdict(lambda: defaultdict(list))
            for row in search:
                groups[coordinate(row)][row['block_id']].append(row)
            points = np.asarray([
                [np.mean([np.mean([r[field] for r in records]) for records in blocks.values()])
                 for field in ['val_loss', 'latency_us']] for blocks in groups.values()])
            if points[:, 1].min() <= config['latency_threshold_us']:
                raise ValueError('Search feasibility claim is contradicted')
            points = points[(points[:, 0] < config['hv_ref_loss'])
                            & (points[:, 1] < config['hv_ref_latency_us'])]
            limits = sorted(set(points[:, 0])) + [config['hv_ref_loss']]
            volume = sum((b - a) * (config['hv_ref_latency_us'] - min(points[points[:, 0] <= a, 1]))
                         for a, b in zip(limits[:-1], limits[1:]))
            hypervolumes.append(volume)
            X = design([coordinate(r) for r in search], [r['block_id'] for r in search])
            Z = design([coordinate(r) for r in reserved], [r['block_id'] for r in reserved])
            inverse = np.linalg.inv(X.T @ X)
            degrees = len(X) - np.linalg.matrix_rank(X)
            suffix = str(config['adaptive_cycles']) if method == 'doe' else 'final'
            snapshot = read(f'results/phase_b/imdb/snapshot_{repeat}_{method}_{suffix}.json')
            for response in ['val_loss', 'latency_us']:
                y = np.asarray([r[response] for r in search])
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                residual = y - X @ beta
                mse = residual @ residual / degrees
                predicted = Z @ beta
                margin = t.ppf(.975, degrees) * np.sqrt(mse * (1 + np.einsum('ij,jk,ik->i', Z, inverse, Z)))
                records = {(r['config_id'], r['block_id']): r for r in confirmations
                           if r['method'] == method and r['replicate'] == repeat and r['response'] == response}
                for row, prediction, width in zip(reserved, predicted, margin):
                    record = records[row['config_id'], row['block_id']]
                    if not np.allclose([prediction, prediction-width, prediction+width],
                                       [record['predicted'], record['lower'], record['upper']],
                                       rtol=1e-8, atol=1e-9):
                        raise ValueError('Confirmation interval differs from independent OLS fit')
                if response == 'val_loss':
                    main = beta[1:5] ** 2 / np.diag(inverse)[1:5]
                    interactions = beta[9:15] ** 2 / np.diag(inverse)[9:15]
                    shap = snapshot[response]['shap']['gp']
                    main_shap = np.abs(np.asarray(shap['raw_shap_values'])).mean(axis=0)
                    pair_shap = np.abs(np.asarray(shap['raw_interaction_values'])).mean(axis=0)
                    agreements.append((spearmanr(main, main_shap).statistic,
                                       pearsonr(interactions, pair_shap[np.triu_indices(4, 1)]).statistic))
                if method == 'doe' and repeat == 0:
                    for index, dropped in enumerate([range(1, 5), range(9, 15), range(5, 9), range(15, 19)], 1):
                        reduced = np.delete(X, list(dropped), axis=1)
                        error = y - reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
                        statistic = ((error @ error - residual @ residual) / len(dropped)) / mse
                        prefix = 'Anova' + response.replace('_', '') + str(index)
                        compare(prefix + 'F', statistic)
                        compare(prefix + 'P', f_distribution.sf(statistic, len(dropped), degrees), 1e-40)
            if repeat == 0:
                initial = sum(r['stage'] == 'initial' for r in search) if method != 'tpe' else len(search)
                for name, value in [('Initial', initial), ('Infill', len(search)-initial), ('Total', len(search))]:
                    compare('Budget' + method.replace('_', '') + name, value)
        compare('B' + letter + 'Hv', np.mean(hypervolumes))
        compare('B' + letter + 'HvSd', np.std(hypervolumes, ddof=1))
        compare('B' + letter + 'Rank', np.mean(agreements, axis=0)[0])
        compare('B' + letter + 'Interaction', np.mean(agreements, axis=0)[1])
        for response in ['val_loss', 'latency_us']:
            group = [r for r in confirmations if r['method'] == method and r['response'] == response]
            errors = np.asarray([r['observed'] - r['predicted'] for r in group])
            boundary = [r['observed']-r['predicted'] for r in group if r['boundary']]
            values = [np.sqrt(np.mean(errors**2)), errors.mean(),
                      100*np.mean([r['lower'] <= r['observed'] <= r['upper'] for r in group]), np.mean(boundary)]
            for suffix, value in zip(['Error', 'Bias', 'Coverage', 'Boundary'], values):
                compare('Confirm' + letter + response.replace('_', '') + suffix, value)

    housing = root / 'phase_a_rsm_blocking/results'
    provenance = read('phase_a_rsm_blocking/results/revision_v2/full_run_001/provenance.json')
    documentation_only = []
    def runtime_tree(source):
        tree = ast.parse(source.decode('utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                    node.body = node.body[1:]
        return ast.dump(tree, include_attributes=False)
    with zipfile.ZipFile(root/'evidence/phase_a_sources.zip') as sources:
        if set(sources.namelist()) != set(provenance['runtime_source_sha256']):
            raise ValueError('Historical source snapshot inventory differs')
        for name, expected in provenance['runtime_source_sha256'].items():
            original = sources.read(name)
            if hashlib.sha256(original).hexdigest() != expected:
                raise ValueError('Historical source snapshot differs: '+name)
            current = (housing.parent/name).read_bytes().replace(b'\r\n', b'\n')
            if hashlib.sha256(current).hexdigest() != expected:
                if not name.endswith('.py') or runtime_tree(current) != runtime_tree(original):
                    raise ValueError('Historical runtime implementation differs: '+name)
                documentation_only.append(name)
    for selector, expected in provenance['baseline_input_sha256'].items():
        current = (housing.parent/selector.split(':', 1)[1]).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(current).hexdigest() != expected:
            raise ValueError('Historical baseline observation differs: '+selector)
    data = pd.read_csv(housing / 'runs.csv')
    x = data[[f'x{i}' for i in range(1, 5)]].to_numpy()
    y = data.val_rmse.to_numpy()
    X = design(x, data.block)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    _, cells = np.unique(x, axis=0, return_inverse=True)
    additive = np.column_stack([np.ones(len(y)), pd.get_dummies(data.block, drop_first=True),
                               pd.get_dummies(cells, drop_first=True)])
    residual = y - X @ beta
    error = y - additive @ np.linalg.lstsq(additive, y, rcond=None)[0]
    statistic = ((residual @ residual - error @ error) / (np.linalg.matrix_rank(additive)-X.shape[1])) / (error @ error/(len(y)-np.linalg.matrix_rank(additive)))
    compare('ALof', statistic)
    factorial = data[data.phase == 'Phase1_Factorial']
    center = data[data.phase == 'Phase1_Center']
    pure = sum(((g.val_rmse-g.val_rmse.mean())**2).sum() for _, g in center.groupby('block'))
    curvature = len(factorial)*len(center)/(len(factorial)+len(center))*(factorial.val_rmse.mean()-center.val_rmse.mean())**2
    compare('ACurvature', curvature/(pure/(len(center)-center.block.nunique())))
    compare('ADesign', len(data)); compare('ABlocks', data.block.nunique())
    compare('ALowRate', data.eta.min()); compare('ALowCount', (data.eta == data.eta.min()).sum())
    compare('ALowPct', 100*(data.eta == data.eta.min()).mean())
    # Profile the random-intercept REML likelihood without the project's mixed-model implementation.
    fixed = design(x)
    block_matrix = pd.get_dummies(data.block, dtype=float).to_numpy()
    sizes = block_matrix.sum(axis=0)
    def reml(log_ratio):
        ratio = np.exp(log_ratio)
        inverse_cov = np.eye(len(y)) - (block_matrix*(ratio/(1+ratio*sizes))) @ block_matrix.T
        information = fixed.T @ inverse_cov @ fixed
        coefficient = np.linalg.solve(information, fixed.T @ inverse_cov @ y)
        residual = y-fixed @ coefficient
        return ((len(y)-fixed.shape[1])*np.log(residual @ inverse_cov @ residual/(len(y)-fixed.shape[1]))
                + np.log1p(ratio*sizes).sum() + np.linalg.slogdet(information)[1])
    fitted_ratio = np.exp(minimize_scalar(reml, bounds=(-20, 10), method='bounded').x)
    # Historical REML optimizer tolerances differ; verify the published rounded ICC.
    if format(100*fitted_ratio/(1+fitted_ratio), metrics['AIcc']['format']) != format(metrics['AIcc']['value'], metrics['AIcc']['format']):
        raise ValueError('Published block ICC differs from independently optimized REML')
    reconstructed.add('AIcc')
    confirmation = read('phase_a_rsm_blocking/results/confirmation.json')
    for label, filename, point in [('Balanced', 'confirmation_runs.csv', confirmation['x_star_coded']),
                                   ('Accuracy', 'confirmation_runs_single_obj.csv', confirmation['Single_Objective_Optimum']['x_coded'])]:
        prediction = np.r_[design([point])[0], [.2]*4] @ beta
        observed = pd.read_csv(housing / filename).val_rmse.mean()
        compare('A'+label+'Pred', prediction); compare('A'+label+'Obs', observed)
        compare('A'+label+'Bias', observed-prediction)
    benchmark = housing / 'revision_v2/full_run_001'
    evaluations = pd.read_csv(benchmark / 'final_evaluations.csv')
    selections = pd.read_csv(benchmark / 'final_summary.csv')
    means = evaluations.groupby('selection_id').test_rmse.mean()
    names = ['repeated_preplanned_doe_multi_objective', 'repeated_preplanned_doe_single_objective',
             'multi_objective_tpe', 'constrained_tpe', 'single_objective_tpe', 'random_search']
    for letter, method in zip('ABCDEF', names):
        selected = selections[selections.optimizer == method]
        values = means.loc[selected.selection_id]
        for suffix, value in [('Error', values.mean()), ('Sd', values.std(ddof=1)),
                              ('Latency', selected.predict_latency_us.mean())]:
            compare('ABench'+letter+suffix, value)
    constrained = selections[selections.optimizer == 'constrained_tpe']
    compare('ASearchFeasible', constrained.search_time_feasible.sum())
    compare('AVerifiedFeasible', constrained.benchmark_time_feasible.sum())
    compare('ARepeats', len(constrained))
    words = read('results/phase_b/imdb_words/observations.json')['documents']
    errors = []
    for document in words:
        observations = document['confirmation']
        error = np.sqrt(np.mean((np.asarray(observations['observed']) - observations['predicted'])**2))
        if not np.isclose(error, observations['rmse'], atol=1e-12):
            raise ValueError('Local explanation error differs from perturbation observations')
        errors.append(error)
    compare('BWordDocs', len(words)); compare('BWordMaxError', max(errors))
    compare('WordExampleIndex', np.argmax(errors)+1)
    compare('BSearch', sum(r['stage'] != 'confirmation' for r in rows))
    compare('BConfirmationFits', sum(r['stage'] == 'confirmation' for r in rows))
    compare('BFinalFits', len(final)); compare('BTotal', len(rows)+len(final)+1)
    documents = pd.read_csv(root / 'phase_b_sequential_xai/data/imdb.csv')
    compare('BDev', (documents.split == 'train').sum()); compare('BTest', (documents.split == 'test').sum())
    calibration = read('results/phase_b_calibration/imdb_pilot.json')['observations']
    tolerance = float(re.search(r'within\s+([.\d]+)', read('results/phase_b_calibration/calibration_contract.json')['selection_rule']).group(1))
    low_rate = [r for r in calibration if r['coded_params']['x1'] == -1]
    reference_f1 = max(low_rate, key=lambda r: r['n_trees'])['val_macro_f1']
    capacity = min(r['n_trees'] for r in low_rate if r['val_macro_f1'] >= reference_f1-tolerance)
    compare('BTrees', capacity)
    compare('BCap', np.median([r['latency']['predict_median_us'] for r in calibration]))
    compare('BReference', 2*max(r['latency']['predict_median_us'] for r in calibration))
    for index, row in enumerate(calibration):
        f1, loss = scores(row)
        latency = np.median(row['latency']['predict_raw_us'])
        for suffix, value in [('Trees', row['n_trees']), ('Rate', row['natural_params']['learning_rate']),
                              ('Fone', f1), ('Latency', latency)]:
            compare('Calibration'+chr(65+index)+suffix, value)
    for name, key in [('BBlocks', 'n_nuisance_blocks'), ('BRepeats', 'search_replicates'),
                      ('BWarmups', 'warmup_inferences'), ('BCalls', 'timed_inferences'),
                      ('BConfirm', 'confirmation_points'), ('BThreads', 'training_threads'),
                      ('BVocab', 'max_features'), ('BReferenceLoss', 'hv_ref_loss')]:
        compare(name, config[key])
    return dict(reconstructed_metrics=sorted(reconstructed), development_and_confirmation_fits=len(rows),
                final_models=len(final), word_documents=len(words),
                historical_source_checks=dict(runtime_files=len(provenance['runtime_source_sha256']),
                    baseline_inputs=len(provenance['baseline_input_sha256']), documentation_only_differences=documentation_only),
                source_traced_metrics=sorted(set(metrics)-reconstructed),
                limitations=['Historical figures retain their original graphics; underlying saved summaries are source-traced.',
                             'GP Shapley arrays are source-traced; their agreement statistics are independently recomputed.',
                             'Independent reconstruction of saved evidence is not a new training experiment.'])
