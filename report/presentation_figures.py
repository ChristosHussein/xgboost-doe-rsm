"""Readable scientific graphics from frozen observations and the executed protocol."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


METHODS = {
    'historical_preplanned_doe_desirability': ('Historical DOE: compromise', '#4c78a8', 's'),
    'historical_preplanned_doe_single_objective': ('Historical DOE: accuracy', '#83b4df', 'D'),
    'repeated_preplanned_doe_multi_objective': ('Repeated DOE: compromise', '#f28e2b', 'o'),
    'repeated_preplanned_doe_single_objective': ('Repeated DOE: accuracy', '#e6b45a', '^'),
    'multi_objective_tpe': ('MO-TPE', '#59a14f', 'P'),
    'constrained_tpe': ('Constrained TPE', '#b279a2', 'v'),
    'single_objective_tpe': ('Accuracy-focused TPE', '#e15759', 'X'),
    'random_search': ('Random search', '#79706e', 'h'),
}


def _save(fig, destination):
    if destination.suffix == '.pdf':
        fig.savefig(destination, metadata={'CreationDate': None, 'ModDate': None})
    else:
        fig.savefig(destination, dpi=160, metadata={'Software': 'Matplotlib'},
                    pil_kwargs={'compress_level': 0})
    plt.close(fig)


def _front(x, y):
    points = np.column_stack([x, y])
    points = points[np.isfinite(points).all(axis=1)]
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    kept, best = [], np.inf
    for a, b in points:
        if b < best:
            kept.append((a, b)); best = b
    return np.asarray(kept)


def _housing(root, out):
    folder = root/'phase_a_rsm_blocking/results/revision_v2/full_run_001'
    sources = [folder/name for name in ['doe_matched_candidate_front.csv', 'optimizer_trials.csv',
                                       'final_summary.csv', 'final_evaluations.csv']]
    planned, trials, final, evaluations = [pd.read_csv(p) for p in sources]
    evaluations = evaluations[evaluations.status == 'completed']
    moments = evaluations.groupby('selection_id').test_rmse.agg(['mean', 'std'])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.8))
    for method, (label, color, marker) in METHODS.items():
        if method in set(trials.optimizer):
            records = trials[(trials.optimizer == method) & (trials.trial_status == 'completed')]
            front = _front(records.validation_rmse, records.predict_latency_us)
            axes[0].plot(front[:, 0], front[:, 1], color=color, lw=1.5)
            axes[0].scatter(front[:, 0], front[:, 1], color=color, marker=marker, s=15, alpha=.7)
        selections = final[final.optimizer == method]
        for _, row in selections.iterrows():
            moment = moments.loc[row.selection_id]
            axes[1].errorbar(moment['mean'], row.predict_latency_us, xerr=moment['std'],
                yerr=row.predict_latency_session_standard_deviation_us,
                fmt=marker, ms=4, color=color, alpha=.55, lw=.7, capsize=1.5)
    front = _front(planned.validation_rmse, planned.predict_latency_us)
    axes[0].plot(front[:, 0], front[:, 1], color=METHODS['historical_preplanned_doe_desirability'][1], lw=2)
    axes[0].scatter(front[:, 0], front[:, 1], s=30, marker='s', color=METHODS['historical_preplanned_doe_desirability'][1])
    axes[0].set(title='Development candidate fronts', xlabel='Validation RMSE (fixed development split)',
                ylabel='Search CPU latency (microseconds)')
    axes[1].set(title='Frozen selections on housing test partition',
                xlabel='Mean test RMSE (bars: retraining SD)',
                ylabel='CPU latency (microseconds; bars: session SD)')
    for ax in axes:
        ax.grid(alpha=.18); ax.tick_params(labelsize=10)
    handles = [Line2D([], [], color=color, marker=marker, lw=0, markersize=6, label=label)
               for label, color, marker in METHODS.values()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=2, frameon=False, fontsize=11, columnspacing=2)
    fig.tight_layout(rect=(0, .24, 1, 1), w_pad=2.5)
    _save(fig, out/'phase_a_pareto.png')
    return sources


def _words(root, out):
    source = root/'results/phase_b/imdb_words/observations.json'
    documents = json.loads(source.read_text(encoding='utf-8'))['documents']
    errors = np.asarray([d['confirmation']['rmse'] for d in documents])
    index = int(np.argmax(errors)); document = documents[index]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    axes[0].bar(np.arange(1, len(errors)+1), errors, color='#3566a1')
    axes[0].set(xlabel='Document', ylabel='Probability RMSE', title='Local surface fidelity')
    predicted = np.asarray(document['confirmation']['predicted'])
    observed = np.asarray(document['confirmation']['observed'])
    axes[1].scatter(predicted, observed, color='#d47b26', s=16)
    low, high = min(predicted.min(), observed.min()), max(predicted.max(), observed.max())
    axes[1].plot([low, high], [low, high], 'k--', lw=1)
    axes[1].set(xlabel='Quadratic-surface probability', ylabel='Classifier probability',
                title=f'Document {index+1}: reserved perturbations')
    interaction_index = int(np.argmax([np.max(np.abs(np.asarray(d['shap']['raw_interaction_values'])[0][np.triu_indices(4,1)])) for d in documents]))
    interaction_document = documents[interaction_index]
    matrix = np.asarray(interaction_document['shap']['raw_interaction_values'], dtype=float)
    if matrix.shape != (1, 4, 4):
        raise ValueError('Word illustration requires its recorded single Shapley evaluation')
    matrix = matrix[0].copy()
    off_diagonal = ~np.eye(len(matrix), dtype=bool)
    largest = float(np.max(np.abs(matrix[off_diagonal])))
    limit = max(largest, 1e-4)
    np.fill_diagonal(matrix, np.nan)
    image = axes[2].imshow(np.ma.masked_invalid(matrix), cmap='RdBu_r', vmin=-limit, vmax=limit,
                           interpolation='nearest')
    for i, j in zip(*np.where(off_diagonal)):
        value = matrix[i, j]
        label = f'{value:.1e}' if 0 < abs(value) < 1e-4 else f'{value:.4f}'
        axes[2].text(j, i, label, ha='center', va='center', fontsize=8,
                     color='white' if abs(value) > .65*limit else 'black')
    axes[2].set_xticks(range(4), interaction_document['words'], rotation=35, ha='right')
    axes[2].set_yticks(range(4), interaction_document['words'])
    subtitle = 'All recorded pair interactions are zero' if largest == 0 else 'Signed pair effects; diagonal omitted'
    axes[2].set_title(f'Document {interaction_index+1}: pair interactions\n'+subtitle, fontsize=10)
    colorbar = fig.colorbar(image, ax=axes[2], shrink=.75, fraction=.045, pad=.04)
    colorbar.set_label('Probability contribution', fontsize=8)
    colorbar.formatter.set_powerlimits((0, 0)); colorbar.update_ticks()
    fig.tight_layout(w_pad=1.4)
    _save(fig, out/'phase_b_words.pdf')
    return [source]


def _workflow(root, out):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    boxes = {
        'planned': (.16, .82, 'Planned or augmented\nconfigurations', '#e8f0fa'),
        'measure': (.16, .49, 'Measure development\nloss and latency', '#e8f0fa'),
        'fit': (.50, .49, 'Fit blocked RSM\nand GP responses', '#e5f1e9'),
        'acquire': (.50, .82, 'Acquire: GP uncertainty\nÃ— âˆš(nearest-point distance)', '#e5f1e9'),
        'diagnose': (.84, .49, 'ANOVA / SHAP\ndiagnostics', '#eee7f4'),
        'select': (.16, .16, 'Final compromise from\nobserved search candidates', '#fff0de'),
        'confirm': (.70, .16, 'Assess response predictions at\nreserved configurations', '#fff0de'),
    }
    for x, y, label, color in boxes.values():
        ax.add_patch(FancyBboxPatch((x-.135, y-.09), .27, .18,
                                   boxstyle='round,pad=.008', facecolor=color, edgecolor='#74808b', lw=1))
        ax.text(x, y, label, ha='center', va='center', fontsize=10)
    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#52606d', lw=1.4))
    arrow((.16, .72), (.16, .59)); arrow((.305, .49), (.355, .49))
    arrow((.50, .59), (.50, .72)); arrow((.355, .82), (.305, .82))
    arrow((.645, .49), (.695, .49)); arrow((.16, .39), (.16, .26))
    arrow((.56, .39), (.66, .26))
    ax.text(.5, -.025, 'Explanations are diagnostic; they do not enter the executed acquisition score.',
            ha='center', va='top', fontsize=10)
    fig.tight_layout()
    _save(fig, out/'sequential_design.pdf')
    return [root/'phase_b_sequential_xai/xgb_doe_xai/study.py',
            root/'phase_b_sequential_xai/xgb_doe_xai/design.py']


def build(root, out):
    """Write three publication figures into out; return their exact source inputs."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.compression': 0}):
        return {'phase_a_pareto.png': _housing(root, out),
                'phase_b_words.pdf': _words(root, out),
                'sequential_design.pdf': _workflow(root, out)}
