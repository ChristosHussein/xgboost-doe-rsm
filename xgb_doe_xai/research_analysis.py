"""Predeclared descriptive uncertainty and shared-target efficiency summaries."""
from pathlib import Path
import json
import numpy as np


def bootstrap_mean(values, *, seed=812, repetitions=2000):
    """Resample search-level summaries, never pretend shared folds are independent."""
    values=np.asarray([v for v in values if v is not None],dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Undefined measurements must be null, not infinity or NaN')
    if not len(values):
        return dict(n=0,mean=None,sd=None,interval=None)
    interval=None
    if len(values)>1:
        rng=np.random.default_rng(seed)
        samples=rng.choice(values,(repetitions,len(values)),replace=True).mean(axis=1)
        interval=np.quantile(samples,[.025,.975]).tolist()
    return dict(n=len(values),mean=float(values.mean()),
        sd=float(values.std(ddof=1)) if len(values)>1 else None,interval=interval,
        interpretation='descriptive search-resampling interval; limited repeats, not population equivalence')


def progress_to_target(curves, fraction=.95):
    """Use a common observed HV target within each replicate, not method-specific targets."""
    if not 0<fraction<=1:
        raise ValueError('Target fraction must be in (0,1]')
    best=max(value for curve in curves.values() for _,value in curve)
    target=fraction*best
    return dict(fraction=fraction,target=target,
        fits={method:next((count for count,value in curve if best>0 and value>=target),None)
              for method,curve in curves.items()},
        interpretation='relative to best observed validation HV, not unknown optimal HV')


def completed_block_progress(rows,cfg):
    """Every prefix uses complete configurations and equal weights for split blocks."""
    from .study import key
    from .benchmarks import compute_hypervolume_2d
    cells,curve={},[]
    for end,row in enumerate(rows,1):
        block=cells.setdefault(key(row['coded_params']),{}).setdefault(row['block_id'],[])
        block.append(row)
        points=[]
        for blocks in cells.values():
            if set(blocks)!=set(range(cfg['n_nuisance_blocks'])):
                continue
            loss=np.mean([np.mean([r['val_loss'] for r in records]) for records in blocks.values()])
            latency=np.mean([np.mean([r['latency_us'] for r in records]) for records in blocks.values()])
            points.append((float(loss),float(latency)))
        hv=compute_hypervolume_2d(points,(cfg['hv_ref_loss'],cfg['hv_ref_latency_us'])) if points else 0.
        curve.append((end,hv))
    return curve


def analyze_study(output):
    from .study_reporting import load_rows, reconstruct_summary
    from .study import group_search
    from .benchmarks import compute_hypervolume_2d
    output=Path(output)
    summary=reconstruct_summary(output)
    contract=json.loads((output/'measurement_contract.json').read_text())
    cfg=contract['config']
    rows=[r for r in load_rows(output) if r['stage']!='confirmation']
    final_rows=json.loads((output/'final_results.json').read_text())['results']
    progress={}
    for rep in range(cfg['search_replicates']):
        curves={}
        for method in summary['methods']:
            subset=[r for r in rows if r['method']==method and r['replicate']==rep]
            curves[method]=completed_block_progress(subset,cfg)
        progress[str(rep)]=dict(curves=curves,shared_target=progress_to_target(curves))
    estimates={}
    for method,entry in summary['methods'].items():
        reps=entry['replicates']
        estimates[method]={name:bootstrap_mean([r[name] for r in reps]) for name in
            ('hypervolume','best_observed_loss','feasibility_rate','overhead_seconds','observed_reference_regret')}
        test=[r for r in summary['test_results'] if r['method']==method]
        estimates[method]['test_macro_f1']=bootstrap_mean([r['test_macro_f1'] for r in test])
        estimates[method]['test_logloss']=bootstrap_mean([r['test_logloss'] for r in test])
        estimates[method]['test_latency_us']=bootstrap_mean([r['test_latency_us'] for r in test])
        probabilities=[]
        for row in final_rows:
            if row['method']==method:
                y=np.asarray(row['validation_labels'])
                p=np.asarray(row['validation_probabilities'],dtype=row['validation_probability_dtype'])
                probabilities.append(float(np.sqrt(np.mean((p-y)**2))))
        estimates[method]['test_probability_rmse']=bootstrap_mean(probabilities)
    comparisons={}
    for method in ('static_doe','tpe'):
        if method not in summary['methods']:
            continue
        differences=[a['hypervolume']-b['hypervolume'] for a,b in
            zip(summary['methods']['doe']['replicates'],summary['methods'][method]['replicates'])]
        comparisons['adaptive_minus_'+method]=bootstrap_mean(differences)
    curvature=[]
    from scipy.stats import spearmanr
    for method in summary['methods']:
        for rep in range(cfg['search_replicates']):
            suffix=str(cfg['adaptive_cycles']) if method=='doe' else 'final'
            snapshot=json.loads((output/f'snapshot_{rep}_{method}_{suffix}.json').read_text())
            for response,entry in snapshot.items():
                statistical=list(entry['attribution']['curvature_ss'].values())
                explained=[abs(entry['curvature']['gp'][f'x{i}']['second_difference']) for i in range(1,5)]
                rho=float(spearmanr(statistical,explained).statistic) if np.ptp(statistical)>1e-12 and np.ptp(explained)>1e-12 else None
                curvature.append(dict(method=method,replicate=rep,response=response,
                    anova_partial_ss=statistical,gp_main_effect_second_differences=explained,spearman_rho=rho,
                    interpretation='descriptive ordering agreement between distinct curvature summaries'))
    return dict(evidence_kind=summary['evidence_kind'],estimates=estimates,curvature_agreement=curvature,
        paired_validation_hypervolume=comparisons,progress=progress,
        confirmation=summary['confirmation'],gp_confirmation=summary['gp_confirmation'],
        agreement=summary['explanation_stability'],adaptive_change=summary['adaptive_explanation_change'],
        limitations=['Shared folds are dependent','Few search repeats limit bootstrap resolution',
                     'Multiple datasets are separate applications, not independent draws from all tasks',
                     'Calibration used development data; holdout cannot support new tuning'])
