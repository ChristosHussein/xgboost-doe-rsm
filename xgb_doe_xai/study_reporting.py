"""Reconstruct study summaries and verify the complete per-run artifact registry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .benchmarks import compute_hypervolume_2d


def load_rows(output):
    return [json.loads(line) for line in (output/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]


def reconstruct_summary(output):
    from .study import group_search, key
    contract = json.loads((output/'measurement_contract.json').read_text())
    config = contract['config']
    if json.loads((output/'status.json').read_text()) != {'status': 'COMPLETED'}:
        raise ValueError('Study is incomplete')
    rows = load_rows(output)
    confirmations = json.loads((output/'confirmation.json').read_text())
    final_path = output/'final_results.json'
    final_rows = json.loads(final_path.read_text())['results'] if final_path.exists() else []
    search = [r for r in rows if r['stage'] != 'confirmation']
    confirm_rows = [r for r in rows if r['stage'] == 'confirmation']
    if {key(r['coded_params']) for r in search} & {key(r['coded_params']) for r in confirm_rows}:
        raise ValueError('Confirmation configurations were used in search')
    expected = contract['search_fits_per_method']
    counts = {method: sum(r['method'] == method for r in search) for method in ('doe', 'tpe')}
    if any(n != expected for n in counts.values()) or len(confirm_rows) != contract['confirmation_fits']:
        raise ValueError('Fit accounting does not match the frozen measurement contract')
    if any(r['n_trees'] != config['n_trees'] for r in rows):
        raise ValueError('Training budgets differ from the frozen configuration')
    from sklearn.metrics import f1_score, log_loss
    for row in rows+final_rows:
        truth, probabilities = row['validation_labels'], row['validation_probabilities']
        if not truth and contract['evidence_kind'] == 'synthetic_fixture':
            continue
        if not truth or len(truth) != len(probabilities):
            raise ValueError('Missing primary classification observations')
        probabilities = np.asarray(probabilities, dtype=row['validation_probability_dtype'])
        expected_f1 = f1_score(truth, probabilities >= .5, average='macro')
        expected_logloss = log_loss(truth, probabilities, labels=[0, 1])
        if not np.allclose([row['val_macro_f1'], row['val_loss'], row['val_logloss']],
                           [expected_f1, 1-expected_f1, expected_logloss], rtol=1e-12, atol=1e-12):
            raise ValueError('Classification metrics differ from primary prediction observations')
    if any((r['latency']['n_warmup'], r['latency']['n_timed']) !=
           (config['warmup_inferences'], config['timed_inferences']) for r in rows):
        raise ValueError('Latency protocols differ from the frozen measurement contract')
    if contract['evidence_kind'] == 'research':
        for row in rows:
            latency = row['latency']
            if (not latency['affinity_applied'] or len(latency['predict_raw_us']) != config['timed_inferences']
                    or len(latency['inplace_raw_us']) != config['timed_inferences']):
                raise ValueError('Research latency needs affinity and both complete raw timing records')
    if len(confirmations) != 2*len(confirm_rows):
        raise ValueError('Missing confirmation predictions')
    methods = {}
    for method in counts:
        repeats = []
        for rep in range(config['search_replicates']):
            subset = [r for r in search if r['method'] == method and r['replicate'] == rep]
            if len(subset) != expected//config['search_replicates']:
                raise ValueError('Per-replicate fit budget mismatch')
            grouped = group_search(subset)
            if not grouped:
                raise ValueError('Missing method replicate')
            for point in grouped:
                block_ids = {r['block_id'] for r in subset if key(r['coded_params']) == key(point['coded_params'])}
                if block_ids != set(range(config['n_nuisance_blocks'])):
                    raise ValueError('Incomplete nuisance-block coverage')
            pts = [(p['loss'], p['latency']) for p in grouped]
            hv = compute_hypervolume_2d(pts, (config['hv_ref_loss'], config['hv_ref_latency_us']))
            overhead = json.loads((output/f'overhead_{rep}_{method}.json').read_text())['seconds']
            repeats.append(dict(replicate=rep, hypervolume=hv,
                best_observed_loss=min(p['loss'] for p in grouped),
                feasibility_rate=float(np.mean([p['latency'] <= config['latency_threshold_us'] for p in grouped])),
                overhead_seconds=overhead))
            # Independently reconstruct fixed-block predictions from saved search rows.
            from .surrogates import PolynomialRSMSurrogate
            confirm = [r for r in confirm_rows if r['method'] == method and r['replicate'] == rep]
            x = np.array([list(key(r['coded_params'])) for r in subset])
            blocks = np.array([r['block_id'] for r in subset])
            xc = np.array([list(key(r['coded_params'])) for r in confirm])
            bc = np.array([r['block_id'] for r in confirm])
            for response in ('val_loss', 'latency_us'):
                fitted = PolynomialRSMSurrogate().fit(x, np.array([r[response] for r in subset]), blocks)
                pred, low, high = fitted.predict_interval(xc, block_ids=bc)
                stored = [r for r in confirmations if r['method'] == method and r['replicate'] == rep and r['response'] == response]
                if len(stored) != len(confirm):
                    raise ValueError('Missing response confirmation rows')
                for i, row in enumerate(stored):
                    expected_values = [confirm[i][response], pred[i], low[i], high[i]]
                    if not np.allclose([row[k] for k in ('observed', 'predicted', 'lower', 'upper')],
                                       expected_values, rtol=1e-10, atol=1e-10):
                        raise ValueError('Confirmation predictions differ from search evidence')
        hvs = [r['hypervolume'] for r in repeats]
        methods[method] = dict(replicates=repeats, hypervolume_mean=float(np.mean(hvs)),
                               hypervolume_sd=float(np.std(hvs, ddof=1)) if len(hvs) > 1 else None)
    for rep in range(config['search_replicates']):
        best = min(methods[m]['replicates'][rep]['best_observed_loss'] for m in methods)
        for method in methods:
            methods[method]['replicates'][rep]['observed_reference_regret'] = (
                methods[method]['replicates'][rep]['best_observed_loss'] - best)
    coverage = {}
    for method in counts:
        for response in ('val_loss', 'latency_us'):
            group = [r for r in confirmations if r['method'] == method and r['response'] == response]
            errors = np.array([r['observed']-r['predicted'] for r in group])
            boundary = [r for r in group if r['boundary']]
            coverage[f'{method}:{response}'] = dict(
                n=len(group), rmse=float(np.sqrt(np.mean(errors**2))),
                observed_minus_predicted_bias=float(errors.mean()),
                pi_coverage=float(np.mean([r['lower'] <= r['observed'] <= r['upper'] for r in group])),
                boundary_n=len(boundary), boundary_bias=(float(np.mean([r['observed']-r['predicted'] for r in boundary]))
                                                       if boundary else None))
    final_fits = len(final_rows)
    if final_fits != contract['final_fits']:
        raise ValueError('Final fit accounting mismatch')
    gp_metrics, agreement, adaptive_change = {}, {}, {}
    for method in counts:
        for response in ('val_loss', 'latency_us'):
            observed, predicted, covered, correlations = [], [], [], []
            for rep in range(config['search_replicates']):
                gp = json.loads((output/f'gp_confirmation_{rep}_{method}_{response}.json').read_text())
                observed.extend(gp['observed'])
                predicted.extend(gp['predicted'])
                covered.extend(lo <= y <= hi for y, lo, hi in zip(gp['observed'], gp['lower'], gp['upper']))
                suffix = str(config['adaptive_cycles']) if method == 'doe' else 'final'
                snapshot = json.loads((output/f'snapshot_{rep}_{method}_{suffix}.json').read_text())
                correlations.append(snapshot[response]['comparisons']['gp'])
                if method == 'doe':
                    initial = json.loads((output/f'snapshot_{rep}_doe_0.json').read_text())
                    adaptive_change[f'{rep}:{response}'] = dict(
                        initial=initial[response]['comparisons']['gp'],
                        final=snapshot[response]['comparisons']['gp'],
                        initial_by_block=[r['comparison'] for r in initial[response]['by_block']],
                        final_by_block=[r['comparison'] for r in snapshot[response]['by_block']])
            errors = np.array(observed)-predicted
            gp_metrics[f'{method}:{response}'] = dict(n=len(observed), rmse=float(np.sqrt(np.mean(errors**2))),
                observed_minus_predicted_bias=float(errors.mean()), coverage=float(np.mean(covered)))
            agreement[f'{method}:{response}'] = dict(per_search=correlations)
            for metric in ('spearman_rho', 'kendall_tau', 'interaction_correlation'):
                values = [r[metric] for r in correlations if r[metric] is not None]
                agreement[f'{method}:{response}'][metric] = dict(
                    defined_searches=len(values), mean=float(np.mean(values)) if values else None,
                    sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)
    return dict(evidence_kind=contract['evidence_kind'], search_fits=counts,
                confirmation_fits=len(confirm_rows), final_fits=final_fits,
                total_fits=len(rows)+final_fits, methods=methods, confirmation=coverage,
                final_evaluation_status='COMPLETED' if final_path.exists() else 'NOT RUN',
                gp_confirmation=gp_metrics, explanation_stability=agreement,
                adaptive_explanation_change=adaptive_change,
                test_results=[dict(method=r['method'], replicate=r['replicate'],
                    test_macro_f1=r['val_macro_f1'], test_logloss=r['val_logloss'],
                    test_latency_us=r['latency']['predict_median_us']) for r in final_rows],
                source_selectors=contract['metrics'])


def render_report(summary):
    lines = ['# Explainable sequential DOE study', '',
             f"Evidence classification: **{summary['evidence_kind']}**.",
             'Synthetic/smoke results validate execution only; they support no real-data scientific conclusion.',
             '', '| Method | Search fits | Mean validation hypervolume | SD across searches |',
             '| --- | ---: | ---: | ---: |']
    for method, results in summary['methods'].items():
        sd = results['hypervolume_sd']
        lines.append(f"| {method} | {summary['search_fits'][method]} | {results['hypervolume_mean']:.6f} | "
                     + (f'{sd:.6f}' if sd is not None else 'UNDEFINED (one search)')+' |')
    lines += ['', f"Confirmation fits: {summary['confirmation_fits']}; final fits: {summary['final_fits']}; total: {summary['total_fits']}.",
              f"Final protected evaluation: **{summary['final_evaluation_status']}**.", '',
              '| Method/response | Independent-point observations | Prediction RMSE | Observed - predicted bias | Fixed-block PI coverage |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name, row in summary['confirmation'].items():
        lines.append(f"| {name} | {row['n']} | {row['rmse']:.6f} | {row['observed_minus_predicted_bias']:.6f} | {row['pi_coverage']:.4f} |")
    lines += ['', '## Flexible-surrogate confirmation', '',
              '| Method/response | Reserved configurations | GP prediction RMSE | GP model-based interval coverage |',
              '| --- | ---: | ---: | ---: |']
    for name, row in summary['gp_confirmation'].items():
        lines.append(f"| {name} | {row['n']} | {row['rmse']:.6f} | {row['coverage']:.4f} |")
    lines += ['', '## Frozen-selection final test evaluation', '',
              '| Method | Search replicate | Test macro F1 | Test log loss | Classic latency (us/document) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in summary['test_results']:
        lines.append(f"| {row['method']} | {row['replicate']} | {row['test_macro_f1']:.6f} | "
                     f"{row['test_logloss']:.6f} | {row['test_latency_us']:.6f} |")
    lines += ['', '## Explanation agreement before and after augmentation', '',
              '| Search/response | Initial Spearman rho | Final Spearman rho | Initial interaction correlation | Final interaction correlation |',
              '| --- | ---: | ---: | ---: | ---: |']
    def display(value):
        return f'{value:.4f}' if value is not None else 'UNDEFINED'
    for name, row in summary['adaptive_explanation_change'].items():
        lines.append(f"| {name} | {display(row['initial']['spearman_rho'])} | {display(row['final']['spearman_rho'])} | "
                     f"{display(row['initial']['interaction_correlation'])} | {display(row['final']['interaction_correlation'])} |")
    lines += ['', '## Interpretation and limits', '',
              'Responses are validation 1 - macro F1 and uncached classic-API latency in microseconds per document. '
              'The classic operation includes fresh DMatrix construction; inplace timing excludes this conversion. '
              'Hypervolumes and feasible fractions use means across the same development blocks. '
              'The regret comparator is the best observed loss among these methods, not the unknown true optimum.',
              '', 'ANOVA partial sums of squares and interventional Shapley values are different estimands. '
              'Snapshots compare RSM and GP on the same empirical background and reserved coordinates. '
              'Curvature profiles record SHAP main-effect dependence at three anchors; they are not significance tests. '
              'Undefined correlations remain undefined. Coverage is descriptive for observations at new coordinates '
              'within the observed fixed blocks; it does not validate unseen-block generalization. '
              'Repeated fold observations are dependent, so no independent-sample inference is claimed.',
              '', 'All numerical claims above are reconstructed from `evaluations.jsonl`, `confirmation.json`, '
              'the frozen measurement contract, and per-method overhead observations. '
              'The registry includes every run file; `--check` rejects missing, additional, modified, or stale-source artifacts.',
              '', '## Scientific graphics', '',
              '![Validation Pareto evidence, SHAP interactions, curvature and fit efficiency](study_figures.png)', '']
    return '\n'.join(lines)


def plot_study(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .study import group_search
    rows = load_rows(output)
    search = [r for r in rows if r['stage'] != 'confirmation']
    cfg = json.loads((output/'measurement_contract.json').read_text())['config']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for method, color in [('doe', '#286b94'), ('tpe', '#b15a27')]:
        subset = [r for r in search if r['method'] == method and r['replicate'] == 0]
        grouped = group_search(subset)
        axes[0, 0].scatter([r['latency'] for r in grouped], [1-r['loss'] for r in grouped],
                            label=method, color=color, s=20, alpha=.8)
        frontier = [p for p in grouped if not any(
            q['loss'] <= p['loss'] and q['latency'] <= p['latency']
            and (q['loss'] < p['loss'] or q['latency'] < p['latency']) for q in grouped)]
        frontier.sort(key=lambda p: p['latency'])
        axes[0, 0].plot([r['latency'] for r in frontier], [1-r['loss'] for r in frontier],
                        color=color, linewidth=1.5)
        # Evaluate progress only at prefixes with complete block coverage.
        xs, hs = [], []
        for end in range(cfg['n_nuisance_blocks'], len(subset)+1):
            prefix = subset[:end]
            keys = {}
            for row in prefix:
                keys.setdefault(tuple(row['coded_params'].values()), []).append(row)
            complete = [r for cell in keys.values() if len({r['block_id'] for r in cell}) == cfg['n_nuisance_blocks'] for r in cell]
            if complete:
                pts = group_search(complete)
                xs.append(end)
                hs.append(compute_hypervolume_2d([(p['loss'], p['latency']) for p in pts],
                                                (cfg['hv_ref_loss'], cfg['hv_ref_latency_us'])))
        axes[1, 1].plot(xs, hs, label=method, color=color)
    axes[0, 0].set(xlabel='Block-mean uncached classic latency (us/document)', ylabel='Validation macro F1',
                   title='Observed candidates, search replicate 0')
    axes[0, 0].legend()
    snap = json.loads((output/f"snapshot_0_doe_{cfg['adaptive_cycles']}.json").read_text())['val_loss']
    matrix = np.array(snap['shap']['gp']['interaction_matrix_shap'])
    np.fill_diagonal(matrix, 0)
    im = axes[0, 1].imshow(matrix, cmap='Blues')
    axes[0, 1].set(xticks=range(4), yticks=range(4), xticklabels=['eta', 'depth', 'subsample', 'colsample'],
                   yticklabels=['eta', 'depth', 'subsample', 'colsample'], title='Mean absolute GP SHAP pair interactions (loss units)')
    fig.colorbar(im, ax=axes[0, 1], shrink=.8)
    for label in ('rsm', 'gp'):
        for name, values in snap['curvature'][label].items():
            axes[1, 0].plot([-1, 0, 1], values['anchors'], marker='o',
                            linestyle='-' if label == 'gp' else '--', label=f'{label}: {name}')
    axes[1, 0].set(xlabel='Coded factor coordinate', ylabel='Mean SHAP main effect (loss units)',
                   title='Main-effect dependence: curvature and direction')
    axes[1, 0].legend(fontsize=7, ncol=2)
    axes[1, 1].set(xlabel='Actual search model fits', ylabel='Block-mean validation hypervolume',
                   title='Progress using completed block groups (replicate 0)')
    axes[1, 1].legend()
    fig.savefig(output/'study_figures.png', dpi=160)
    plt.close(fig)


def publish_study(output):
    from .study import write_json, source_hashes
    summary = reconstruct_summary(output)
    write_json(output/'summary.json', summary)
    (output/'report.md').write_text(render_report(summary), encoding='utf-8')
    from .surrogates import AnovaTableResult
    from .reporting import render_latex_anova_table, render_markdown_anova_table
    config = json.loads((output/'measurement_contract.json').read_text())['config']
    for method in ('doe', 'tpe'):
        for rep in range(config['search_replicates']):
            suffix = str(config['adaptive_cycles']) if method == 'doe' else 'final'
            snapshot = json.loads((output/f'snapshot_{rep}_{method}_{suffix}.json').read_text())
            for response in ('val_loss', 'latency_us'):
                anova = AnovaTableResult(**snapshot[response]['anova'])
                stem = f'anova_{rep}_{method}_{response}'
                (output/f'{stem}.md').write_text(render_markdown_anova_table(anova), encoding='utf-8')
                table = render_latex_anova_table(anova, caption=f"Fixed-block partial-SS ANOVA ({method})")
                (output/f'{stem}.tex').write_text(table.replace('tab:anova_rsm', f'tab:{stem}'), encoding='utf-8')
    plot_study(output)
    files = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in output.rglob('*') if p.is_file()}
    write_json(output/'artifact_registry.json', dict(source_hashes=source_hashes(), files=files))
    check_study(output)
    return summary


def check_study(output_dir):
    from .study import source_hashes
    output = Path(output_dir)
    registry = json.loads((output/'artifact_registry.json').read_text())
    actual = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob('*') if p.is_file() and p != output/'artifact_registry.json'}
    if actual != registry['files'] or source_hashes() != registry['source_hashes']:
        raise ValueError('Study integrity mismatch: missing, altered, extra, or stale-source artifact')
    summary = reconstruct_summary(output)
    if json.loads((output/'summary.json').read_text()) != summary:
        raise ValueError('Summary differs from reconstructed evidence')
    if (output/'report.md').read_text(encoding='utf-8') != render_report(summary):
        raise ValueError('Report differs from reconstructed evidence')
    return summary
