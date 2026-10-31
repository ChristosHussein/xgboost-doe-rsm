"""Build report tables/figures exclusively from identified experimental records."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import shutil
import sys
import argparse
import subprocess
import tempfile
import inspect
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path(__file__).parent)
parser.add_argument('--check',action='store_true')
args=parser.parse_args()
if args.check:
    with tempfile.TemporaryDirectory() as temporary:
        destination=Path(temporary)
        subprocess.run([sys.executable,__file__,'--output',str(destination)],check=True)
        from posthoc_publication import semantic_equal
        changed=[]
        for p in destination.rglob('*'):
            if not p.is_file(): continue
            existing=args.output/p.relative_to(destination)
            same=existing.exists() and p.read_bytes()==existing.read_bytes()
            if not same and existing.exists() and p.name in ['posthoc_results.json','metric_registry.json']:
                current=json.loads(p.read_text());previous=json.loads(existing.read_text())
                same=semantic_equal(current,previous)
                if same and p.name=='metric_registry.json':
                    same=all(format(r['value'],r['format'])==format(previous['metrics'][key]['value'],r['format']) for key,r in current['metrics'].items())
            if not same: changed.append(str(p.relative_to(destination)))
        if changed:
            raise SystemExit('Stale generated publication files: '+', '.join(changed))
    print('All generated tables, metrics and figures match their source observations.')
    raise SystemExit(0)
OUT=args.output
(OUT/'figures').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'phase_b_sequential_xai'))
from xgb_doe_xai.study import check_study
from verify_words import verify_words
from xgb_doe_xai.surrogates import PolynomialRSMSurrogate
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
A=ROOT/'phase_a_rsm_blocking/results'; V=A/'revision_v2/full_run_001'
B=ROOT/'results/phase_b'; S=B/'imdb'
summary=check_study(S);verify_words(ROOT)
contract=read(S/'measurement_contract.json'); cfg=contract['config']
provenance=read(V/'provenance.json')
for n,h in provenance['artifact_sha256'].items():
    assert hashlib.sha256((V/n).read_bytes()).hexdigest()==h,n
sources={}; registry={}
def source_digest(path):
    data=path.read_bytes()
    return hashlib.sha256(data.replace(b'\r\n',b'\n') if path.suffix=='.py' else data).hexdigest()
def metric(name,value,source,selector,fmt='.4f',kind='empirical'):
    registry[name]=dict(value=float(value),source=source.relative_to(ROOT).as_posix(),selector=selector,format=fmt,kind=kind)
    sources[source.relative_to(ROOT).as_posix()]=source_digest(source)
    return '\\m{'+name+'}'
def show(name):
    r=registry[name];return format(r['value'],r['format'])
def table(name,headers,rows):
    s='\\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'
    s+=''.join(' & '.join(row)+' \\\\\n' for row in rows)+'\\bottomrule\n\\end{tabular}\n'
    (OUT/(name+'.tex')).write_text(s,encoding='utf-8',newline='\n')


figure_sources={}


def save_figure(fig,name,paths):
    fig.tight_layout()
    fig.savefig(OUT/'figures'/name,metadata={'CreationDate':None,'ModDate':None} if name.endswith('.pdf') else {})
    plt.close(fig)
    figure_sources[name]=[str(p.relative_to(ROOT)).replace('\\','/') for p in paths]
labels={'static_doe':'Static CCD','doe':'Sequential CCD','tpe':'MO-TPE'}
runs=pd.read_csv(A/'runs.csv'); conf=read(A/'confirmation.json'); lo=read(A/'lof.json')
metric('ADesign',len(runs),A/'runs.csv','row count','.0f')
metric('ABlocks',runs.block.nunique(),A/'runs.csv','distinct block count','.0f')
metric('ALowRate',runs.eta.min(),A/'runs.csv','eta minimum','.2f')
metric('ALowCount',int((runs.eta==runs.eta.min()).sum()),A/'runs.csv','count eta at minimum','.0f')
metric('ALowPct',100*(runs.eta==runs.eta.min()).mean(),A/'runs.csv','percentage eta at minimum','.2f')
metric('ALof',lo['Y1']['F_LoF'],A/'lof.json','Y1.F_LoF','.2f')
metric('AIcc',100*read(A/'icc.json')['Y1']['icc_reml'],A/'icc.json','100*Y1.icc_reml','.2f')
phase1=read(A/'phase1.json');metric('ACurvature',phase1['f_curvature'],A/'phase1.json','f_curvature','.2f')
for label,n in [('Balanced','confirmation_runs.csv'),('Accuracy','confirmation_runs_single_obj.csv')]:
    frame=pd.read_csv(A/n)
    estimate=conf['Y1_Val_RMSE']['predicted_mean'] if label=='Balanced' else conf['Single_Objective_Optimum']['predicted_val_rmse']
    interval=conf['Y1_Val_RMSE']['prediction_interval_95'] if label=='Balanced' else conf['Single_Objective_Optimum']['prediction_interval_95_val']
    metric('A'+label+'Pred',estimate,A/'confirmation.json',label+' predicted validation RMSE')
    metric('A'+label+'Obs',frame.val_rmse.mean(),A/n,'mean val_rmse')
    metric('A'+label+'Bias',frame.val_rmse.mean()-estimate,A/n,'mean val_rmse minus confirmation.json prediction')
    for i,side in enumerate(['Lower','Upper']):metric('A'+label+side,interval[i],A/'confirmation.json',label+' prediction interval '+side)
table('a_confirmation',['Candidate','Predicted','Interval','Observed'],[
    [label,metric_name, '$['+'\\m{A'+label+'Lower}, \\m{A'+label+'Upper}]$','\\m{A'+label+'Obs}']
    for label,metric_name in [('Balanced','\\m{ABalancedPred}'),('Accuracy','\\m{AAccuracyPred}')]])
evaluations=pd.read_csv(V/'final_evaluations.csv'); finala=pd.read_csv(V/'final_summary.csv')
assert (evaluations.status=='completed').all()
means=evaluations.groupby('selection_id').test_rmse.mean()
for _,r in finala.iterrows():assert np.isclose(means[r.selection_id],json.loads(r.test_rmse)['mean'],atol=1e-12)
alabels={'repeated_preplanned_doe_multi_objective':'Repeated DOE compromise',
         'repeated_preplanned_doe_single_objective':'Repeated DOE accuracy',
         'multi_objective_tpe':'MO-TPE','constrained_tpe':'Constrained TPE',
         'single_objective_tpe':'Accuracy-focused TPE','random_search':'Random search'}
atable=[];apoints=[]
for i,(method,label) in enumerate(alabels.items()):
    f=finala[finala.optimizer==method]; vals=means.loc[f.selection_id].to_numpy()
    for suffix,value,selector in [('Error',vals.mean(),'mean selection-level test RMSE'),('Sd',vals.std(ddof=1),'SD selection-level mean test RMSE'),('Latency',f.predict_latency_us.mean(),'mean selected predict_latency_us')]:
        metric('ABench'+chr(65+i)+suffix,value,V/'final_summary.csv',method+': '+selector,'.5f' if suffix!='Latency' else '.2f')
    atable.append([label,'\\m{ABench'+chr(65+i)+'Error}','\\m{ABench'+chr(65+i)+'Sd}','\\m{ABench'+chr(65+i)+'Latency}'])
    apoints.append((label,float(vals.mean()),float(f.predict_latency_us.mean())))
ct=finala[finala.optimizer=='constrained_tpe']
import yaml
aconfig=yaml.safe_load((ROOT/'phase_a_rsm_blocking/config.yaml').read_text())
metric('ACap',aconfig['revision_v2']['max_latency_us'],ROOT/'phase_a_rsm_blocking/config.yaml','revision_v2.max_latency_us','.0f',kind='protocol')
metric('ASearchFeasible',ct.search_time_feasible.sum(),V/'final_summary.csv','constrained_tpe search_time_feasible count','.0f')
metric('AVerifiedFeasible',ct.benchmark_time_feasible.sum(),V/'final_summary.csv','constrained_tpe benchmark_time_feasible count','.0f')
metric('ARepeats',len(ct),V/'final_summary.csv','constrained_tpe selection count','.0f')
table('a_benchmark',['Method','Test RMSE','Between-search SD','CPU latency ($\\mu$s)'],atable)
for name,key in [('BTrees','n_trees'),('BBlocks','n_nuisance_blocks'),('BRepeats','search_replicates'),('BWarmups','warmup_inferences'),('BCalls','timed_inferences'),('BConfirm','confirmation_points'),('BThreads','training_threads'),('BVocab','max_features')]:
    metric(name,cfg[key],S/'measurement_contract.json','config.'+key,'.0f','protocol')
metric('BCap',cfg['latency_threshold_us'],S/'measurement_contract.json','config.latency_threshold_us','.1f','protocol')
metric('BReference',cfg['hv_ref_latency_us'],S/'measurement_contract.json','config.hv_ref_latency_us','.1f','protocol')
metric('BReferenceLoss',cfg['hv_ref_loss'],S/'measurement_contract.json','config.hv_ref_loss','.1f','protocol')
metric('BBackground',len(contract['background']),S/'measurement_contract.json','background coordinate count','.0f','protocol')
metric('BConfirmPerMethod',len(read(S/'confirmation.json'))/(2*len(labels)),S/'confirmation.json','per-method, per-response observation count','.0f','protocol')
metric('BIntervalLevel',100*(1-inspect.signature(PolynomialRSMSurrogate.predict_interval).parameters['alpha'].default),ROOT/'phase_b_sequential_xai/xgb_doe_xai/surrogates.py','100*(1 - predict_interval default alpha)','.0f','protocol')
boundary_source=ROOT/'phase_b_sequential_xai/xgb_doe_xai/study.py'
boundary_cutoff=float(re.search(r'boundary=max\(.+>=\s*([.\d]+)',boundary_source.read_text()).group(1))
metric('BBoundaryCutoff',boundary_cutoff,boundary_source,'confirmation boundary: max absolute coded coordinate cutoff','.1f','protocol')
metric('BWordConfirm',len(read(B/'imdb_words/observations.json')['documents'][0]['confirmation']['weights']),B/'imdb_words/observations.json','per-document reserved perturbation count','.0f','protocol')
metric('ACapacity',aconfig['model']['n_estimators'],ROOT/'phase_a_rsm_blocking/config.yaml','model.n_estimators','.0f','protocol')
for split in ['train','val','test']:
    metric('ASplit'+split,100*aconfig['project'][split+'_ratio'],ROOT/'phase_a_rsm_blocking/config.yaml','project.'+split+'_ratio *100','.0f','protocol')
metric('AConfirmationCount',len(pd.read_csv(A/'confirmation_runs.csv')),A/'confirmation_runs.csv','confirmation observations per candidate','.0f','protocol')
metric('AIntervalLevel',float(next(k for k in conf['Y1_Val_RMSE'] if k.startswith('prediction_interval_')).rsplit('_',1)[1]),A/'confirmation.json','Y1_Val_RMSE prediction-interval nominal level','.0f','protocol')
metric('AEvaluationCount',evaluations.groupby('selection_id').size().iloc[0],V/'final_evaluations.csv','retraining observations per selection','.0f','protocol')
calibration_contract=ROOT/'results/phase_b_calibration/calibration_contract.json'
capacity_tolerance=float(re.search(r'within\s+([.\d]+)',read(calibration_contract)['selection_rule']).group(1))
metric('BCalTolerance',capacity_tolerance,calibration_contract,'selection_rule: low-rate macro F1 tolerance','.3f','protocol')
metric('BTotal',summary['total_fits']+read(B/'imdb_words/contract.json')['training_fits'],S/'summary.json','total_fits + imdb_words/contract.json:training_fits','.0f')
metric('BSearch',sum(summary['search_fits'].values()),S/'summary.json','sum search_fits','.0f')
metric('BConfirmationFits',summary['confirmation_fits'],S/'summary.json','confirmation_fits','.0f')
metric('BFinalFits',summary['final_fits'],S/'summary.json','final_fits','.0f')
metric('BFitsPerSearch',summary['search_fits']['doe']/cfg['search_replicates'],S/'summary.json','search_fits.doe / config.search_replicates','.0f','protocol')
metric('PostProposalBudget',summary['search_fits']['doe']/cfg['search_replicates']/cfg['n_nuisance_blocks'],S/'summary.json','search_fits.doe / config.search_replicates / config.n_nuisance_blocks','.0f','protocol')
metric('BRateLow',contract['factors']['x1']['low'],S/'measurement_contract.json','factors.x1.low','.2f','protocol')
metric('CalSmallTrees',min(r['n_trees'] for r in read(ROOT/'results/phase_b_calibration/imdb_pilot.json')['observations']),ROOT/'results/phase_b_calibration/imdb_pilot.json','minimum n_trees','.0f','protocol')
metric('BHours',read(B/'experiment_summary.json')['wall_seconds']/3600,B/'experiment_summary.json','wall_seconds/3600','.2f')
df=pd.read_csv(ROOT/'phase_b_sequential_xai/data/imdb.csv')
for split,name in [('train','BDev'),('test','BTest')]:metric(name,int((df.split==split).sum()),ROOT/'phase_b_sequential_xai/data/imdb.csv','count split='+split,'.0f')
btable=[];ctable=[];agreement=[];btest=[]
for i,(method,label) in enumerate(labels.items()):
    rows=[r for r in summary['test_results'] if r['method']==method]
    for suffix,field,fmt in [('Fone','test_macro_f1','.5f'),('Latency','test_latency_us','.2f'),('Logloss','test_logloss','.4f')]:
        metric('B'+chr(65+i)+suffix,np.mean([r[field] for r in rows]),S/'final_results.json',method+': mean '+field,fmt)
    metric('B'+chr(65+i)+'FoneSd',np.std([r['test_macro_f1'] for r in rows],ddof=1),S/'final_results.json',method+': sample SD test_macro_f1','.5f')
    metric('B'+chr(65+i)+'Hv',summary['methods'][method]['hypervolume_mean'],S/'summary.json',method+': hypervolume_mean','.2f')
    metric('B'+chr(65+i)+'HvSd',summary['methods'][method]['hypervolume_sd'],S/'summary.json',method+': hypervolume_sd','.2f')
    btable.append([label,'$\\m{B'+chr(65+i)+'Fone} \\pm \\m{B'+chr(65+i)+'FoneSd}$',
        '\\m{B'+chr(65+i)+'Logloss}','\\m{B'+chr(65+i)+'Latency}',
        '$\\m{B'+chr(65+i)+'Hv} \\pm \\m{B'+chr(65+i)+'HvSd}$'])
    c=summary['confirmation'][method+':val_loss']
    for suffix,key,scale,fmt in [('Rmse','rmse',1,'.5f'),('Coverage','pi_coverage',100,'.1f')]:
        metric('B'+chr(65+i)+suffix,c[key]*scale,S/'confirmation.json',method+': '+key,fmt)
    ctable.append([label,'\\m{B'+chr(65+i)+'Rmse}','\\m{B'+chr(65+i)+'Coverage}\\%'])
    ag=summary['explanation_stability'][method+':val_loss']
    for suffix,key in [('Rank','spearman_rho'),('Interaction','interaction_correlation')]:
        metric('B'+chr(65+i)+suffix,ag[key]['mean'],S/'summary.json',method+':val_loss '+key+' mean','.3f')
    agreement.append([label,'\\m{B'+chr(65+i)+'Rank}','\\m{B'+chr(65+i)+'Interaction}'])
    btest.append((label,np.mean([r['test_macro_f1'] for r in rows]),np.mean([r['test_latency_us'] for r in rows]),np.std([r['test_macro_f1'] for r in rows],ddof=1)))
table('b_benchmark',['Method','Test macro F1','Log loss','CPU latency ($\\mu$s)','Dev. HV'],btable)
selection_rows=[]
selected_depths={r['natural_params']['max_depth'] for r in read(S/'final_results.json')['results']}
if len(selected_depths)!=1:
    raise ValueError('Selected-depth statement requires a common recorded depth')
metric('BSelectedDepth',next(iter(selected_depths)),S/'final_results.json','common natural_params.max_depth across frozen selections','.0f','protocol')
for method,label in labels.items():
    records=[r for r in read(S/'final_results.json')['results'] if r['method']==method]
    for r in sorted(records,key=lambda row:row['replicate']):
        prefix='Selection'+method.replace('_','')+chr(65+r['replicate'])
        row=[label,metric(prefix+'Repeat',r['replicate']+1,S/'final_results.json',method+': repeat label (one-based)','.0f','protocol')]
        for key,title,fmt in [('learning_rate','Rate','.4f'),('max_depth','Depth','.0f'),('subsample','Sample','.4f'),('reg_lambda','Lambda','.4f')]:
            row.append(metric(prefix+title,r['natural_params'][key],S/'final_results.json',method+': repeat '+str(r['replicate'])+' natural_params.'+key,fmt,'protocol'))
        selection_rows.append(row)
table('b_agreement',['Method','Factor rank correlation','Interaction correlation'],agreement)
words=read(B/'imdb_words/observations.json')['documents']
metric('BWordDocs',len(words),B/'imdb_words/observations.json','documents count','.0f')
metric('BWordMaxError',max(r['confirmation']['rmse'] for r in words),B/'imdb_words/observations.json','max document confirmation RMSE','.4f')
cal=read(ROOT/'results/phase_b_calibration/imdb_pilot.json')['observations']
for tree,name in [(100,'CalSmall'),(600,'CalLarge')]:
    r=next(r for r in cal if r['n_trees']==tree and r['coded_params']['x1']==-1.)
    metric(name,r['val_macro_f1'],ROOT/'results/phase_b_calibration/imdb_pilot.json',str(tree)+' trees, low eta, val_macro_f1','.4f')
calibration_rows=[]
for i,row in enumerate(cal):
    prefix='Calibration'+chr(65+i);path=ROOT/'results/phase_b_calibration/imdb_pilot.json'
    calibration_rows.append([metric(prefix+'Trees',row['n_trees'],path,'observations['+str(i)+'].n_trees','.0f','protocol'),
        metric(prefix+'Rate',row['natural_params']['learning_rate'],path,'observations['+str(i)+'].natural_params.learning_rate','.3f','protocol'),
        metric(prefix+'Fone',row['val_macro_f1'],path,'observations['+str(i)+'].val_macro_f1','.4f'),
        metric(prefix+'Latency',row['latency']['predict_median_us'],path,'observations['+str(i)+'].latency.predict_median_us','.1f')])
table('b_calibration',['Trees','Learning rate','Validation macro F1','CPU latency ($\\mu$s)'],calibration_rows)

# Factor ranges and resource allocation are extracted from the saved protocols.
factor_rows=[]
for letter,(code,title) in zip('ABCD',[('x1','Learning rate'),('x2','Tree depth'),('x3','Row subsampling'),('x4','Regularization')]):
    row=[title]
    for phase,definition,path in [('A',aconfig['factors'][code],ROOT/'phase_a_rsm_blocking/config.yaml'),('B',contract['factors'][code],S/'measurement_contract.json')]:
        for side,key in [('Low','min' if phase=='A' else 'low'),('High','max' if phase=='A' else 'high')]:
            metric(phase+'Factor'+letter+side,definition[key],path,'factors.'+code+'.'+key,'.2f' if code!='x2' else '.0f','protocol')
        row+=['[\\m{'+phase+'Factor'+letter+'Low}, \\m{'+phase+'Factor'+letter+'High}]']
    factor_rows.append(row)
table('factors',['Factor','Phase A range','Phase B range'],factor_rows)
observations=[json.loads(line) for line in (S/'evaluations.jsonl').read_text().splitlines()]
budget_rows=[]
for method,label in labels.items():
    search=[r for r in observations if r['method']==method and r['replicate']==0 and r['stage']!='confirmation']
    initial=sum(r['stage']=='initial' for r in search)
    if method=='tpe': initial=len(search)
    infill=len(search)-initial
    for field,value in [('Initial',initial),('Infill',infill),('Total',len(search))]:
        metric('Budget'+method.replace('_','')+field,value,S/'evaluations.jsonl',method+': repeat zero '+field+' fits','.0f','protocol')
    budget_rows.append([label,*['\\m{Budget'+method.replace('_','')+field+'}' for field in ['Initial','Infill','Total']]])
table('b_design',['Method','Initial/search fits','Adaptive fits','Total per search'],budget_rows)

# Final sequential-design ANOVA and confirmation diagnostics.
anova_rows=[]
for response,title in [('val_loss','Classification loss'),('latency_us','CPU latency')]:
    snap=read(S/'snapshot_0_doe_2.json')[response]['anova']
    for group in ['  Main Effects','  2-Factor Interactions','  Curvature (Quadratic)','  Blocks']:
        j=snap['source'].index(group);identifier=response.replace('_','')+str(j)
        f=metric('Anova'+identifier+'F',snap['f_stat'][j],S/'snapshot_0_doe_2.json',response+': '+group+' F','.2f')
        p=metric('Anova'+identifier+'P',snap['p_value'][j],S/'snapshot_0_doe_2.json',response+': '+group+' p','.2g')
        ss=metric('Anova'+identifier+'SS',snap['sum_sq'][j],S/'snapshot_0_doe_2.json',response+': '+group+' partial sum of squares','.4g')
        df=metric('Anova'+identifier+'Df',snap['df'][j],S/'snapshot_0_doe_2.json',response+': '+group+' degrees of freedom','.0f','protocol')
        anova_rows.append([title if group=='  Main Effects' else '',group.strip().replace('2-Factor','Pairwise'),ss,df])
table('b_anova',['Response','Term group','Partial SS','df'],anova_rows)
confirmation_rows=[]
for i,(method,label) in enumerate(labels.items()):
    for response,title in [('val_loss','Loss'),('latency_us','Latency')]:
        d=summary['confirmation'][method+':'+response]
        prefix='Confirm'+chr(65+i)+response.replace('_','')
        row=[label,title]
        for suffix,key,scale,fmt in [('Error','rmse',1,'.5f' if response=='val_loss' else '.2f'),('Bias','observed_minus_predicted_bias',1,'.5f' if response=='val_loss' else '.2f'),('Coverage','pi_coverage',100,'.1f'),('Boundary','boundary_bias',1,'.5f' if response=='val_loss' else '.2f')]:
            row.append(metric(prefix+suffix,d[key]*scale,S/'confirmation.json',method+':'+response+' '+key,fmt)+('\\%' if suffix=='Coverage' else ''))
        confirmation_rows.append(row)
table('b_validation',['Method','Response','Pred. RMSE','Mean bias','Coverage','Boundary bias'],confirmation_rows)

original_figures={
 'phase_a_workflow.pdf':'explainer_research_pipeline.pdf',
 'phase_a_design.pdf':'explainer_design_projection.pdf',
 'phase_a_blocking.pdf':'explainer_blocking.pdf',
 'phase_a_error_surface.png':'response_surface_rmse_2d_3d.png',
 'phase_a_latency.png':'response_surface_latency_2d_3d.png',
 'phase_a_tradeoff.png':'desirability_pareto_front.png',
 'phase_a_efficiency.pdf':'explainer_efficiency_comparison.pdf',
 'phase_a_pareto.png':'revision_v2_pareto_front.png',
 'sequential_design.pdf':'explainer_sequential_extension.pdf'}
for destination,source in original_figures.items():
    original=ROOT/'findings/phase_a/figures'/source
    shutil.copy2(original,OUT/'figures'/destination)
    figure_sources[destination]=[str(original.relative_to(ROOT)).replace('\\','/'),str((A/'runs.csv').relative_to(ROOT)).replace('\\','/')]
original_inputs={
 'phase_a_error_surface.png':[A/'phase3.json',A/'confirmation.json'],
 'phase_a_latency.png':[A/'latency_modeling.json',A/'latency_models_comparison.csv'],
 'phase_a_tradeoff.png':[A/'confirmation.json',A/'desirability_sensitivity.csv'],
 'phase_a_efficiency.pdf':[A/'benchmark_evals_trajectories.csv',V/'optimizer_trials.csv',V/'doe_selection_runs.csv'],
 'phase_a_pareto.png':[V/'doe_matched_candidate_front.csv',V/'final_summary.csv',V/'hypervolume.json']}
for name,paths in original_inputs.items():
    figure_sources[name]+=[str(p.relative_to(ROOT)).replace('\\','/') for p in paths]

# New Phase B figures use observations from the published study only.
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.compression':0})
colors={'static_doe':'#3566a1','doe':'#d47b26','tpe':'#27856c'}
fig,axes=plt.subplots(1,2,figsize=(10,3.8))
for method,label in labels.items():
    for rep in range(cfg['search_replicates']):
        groups=defaultdict(list)
        for r in observations:
            if r['method']==method and r['replicate']==rep and r['stage']!='confirmation':
                groups[tuple(r['coded_params'][f'x{i}'] for i in range(1,5))].append(r)
        points=np.array([(np.mean([r['latency_us'] for r in g]),1-np.mean([r['val_loss'] for r in g])) for g in groups.values()])
        axes[0].scatter(points[:,0],points[:,1],s=11,alpha=.32,color=colors[method],label=label if rep==0 else None)
        ordered=points[np.argsort(points[:,0])];front=[];best=-np.inf
        for x,y in ordered:
            if y>best:front.append((x,y));best=y
        front=np.array(front);axes[0].plot(front[:,0],front[:,1],alpha=.6,color=colors[method])
    results=[r for r in summary['test_results'] if r['method']==method]
    axes[1].errorbar(np.mean([r['test_latency_us'] for r in results]),np.mean([r['test_macro_f1'] for r in results]),
        xerr=np.std([r['test_latency_us'] for r in results],ddof=1),yerr=np.std([r['test_macro_f1'] for r in results],ddof=1),fmt='o',capsize=4,color=colors[method],label=label)
axes[0].set(xlabel='CPU latency (microseconds)',ylabel='Validation macro F1',title='Observed development trade-offs')
axes[1].set(xlabel='CPU latency (microseconds)',ylabel='Test macro F1',title='Selected models on test documents')
for ax in axes:ax.legend(fontsize=8)
save_figure(fig,'phase_b_pareto.pdf',[S/'evaluations.jsonl',S/'final_results.json'])

analysis=read(B/'imdb_analysis.json')
fig,axes=plt.subplots(1,2,figsize=(10,3.6))
for method,label in labels.items():
    curves=np.array([analysis['progress'][str(rep)]['curves'][method] for rep in range(cfg['search_replicates'])])
    x=curves[0,:,0];y=curves[:,:,1]
    axes[0].plot(x,np.median(y,axis=0),label=label,color=colors[method]);axes[0].fill_between(x,np.quantile(y,.25,axis=0),np.quantile(y,.75,axis=0),color=colors[method],alpha=.16)
    axes[1].scatter(np.repeat(label,cfg['search_replicates']),[summary['methods'][method]['replicates'][rep]['hypervolume'] for rep in range(cfg['search_replicates'])],color=colors[method],s=28)
axes[0].set(xlabel='Completed XGBoost fits',ylabel='Development hypervolume',title='Complete-block results available');axes[0].legend(fontsize=8)
axes[1].set(ylabel='Final development hypervolume',title='Variation across search repeats')
save_figure(fig,'phase_b_progress.pdf',[B/'imdb_analysis.json',S/'evaluations.jsonl'])

names=['Rate','Depth','Subsample','Regularization'];snapshot_paths=[]
fig,axes=plt.subplots(2,2,figsize=(10,6))
for ri,(response,title) in enumerate([('val_loss','Classification loss'),('latency_us','CPU latency')]):
    for ci,(kind,method_name) in enumerate([('anova','ANOVA main effects'),('shap','GP SHAP contributions')]):
        ax=axes[ri,ci]
        for mi,(method,label) in enumerate(labels.items()):
            scores=[]
            for rep in range(cfg['search_replicates']):
                suffix=str(cfg['adaptive_cycles']) if method=='doe' else 'final';path=S/f'snapshot_{rep}_{method}_{suffix}.json';snapshot_paths.append(path);snap=read(path)[response]
                values=[snap['attribution']['main_effect_ss'][f'x{i}'] for i in range(1,5)] if kind=='anova' else [snap['shap']['gp']['mean_abs_shap'][f'x{i}'] for i in range(1,5)]
                values=np.array(values);scores.append(values/values.sum())
            ax.bar(np.arange(4)+(mi-1)*.25,np.mean(scores,axis=0),width=.24,label=label,color=colors[method])
        ax.set_xticks(range(4),names,rotation=10);ax.set(title=title+': '+method_name,ylabel='Normalized contribution')
axes[0,0].legend(fontsize=8)
save_figure(fig,'phase_b_importance.pdf',sorted(set(snapshot_paths)))

fig,axes=plt.subplots(2,2,figsize=(8.5,7))
for ri,(response,title) in enumerate([('val_loss','Classification loss'),('latency_us','CPU latency')]):
    for ci,(kind,label) in enumerate([('anova','ANOVA interactions'),('shap','GP SHAP interactions')]):
        matrices=[]
        for rep in range(cfg['search_replicates']):
            snap=read(S/f'snapshot_{rep}_doe_2.json')[response]
            mat=np.array(snap['attribution']['interaction_matrix_ss'] if kind=='anova' else snap['shap']['gp']['interaction_matrix_shap'])
            np.fill_diagonal(mat,0);total=mat[np.triu_indices(4,1)].sum();matrices.append(mat/total if total else mat)
        values=np.mean(matrices,axis=0);ax=axes[ri,ci];im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(4),names,rotation=25,ha='right');ax.set_yticks(range(4),names);ax.set_title(title+'\n'+label,fontsize=10)
        for i in range(4):
            for j in range(4):
                if i!=j:ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if values[i,j]>.5 else 'black')
        fig.colorbar(im,ax=ax,shrink=.65)
save_figure(fig,'phase_b_interactions.pdf',[S/f'snapshot_{rep}_doe_2.json' for rep in range(cfg['search_replicates'])])

fig,axes=plt.subplots(2,2,figsize=(9,5.7))
for j,ax in enumerate(axes.flat):
    for cycle,label,color in [(0,'Initial design','#3566a1'),(cfg['adaptive_cycles'],'After augmentation','#d47b26')]:
        values=np.array([read(S/f'snapshot_{rep}_doe_{cycle}.json')['val_loss']['curvature']['gp'][f'x{j+1}']['anchors'] for rep in range(cfg['search_replicates'])])
        ax.plot([-1,0,1],values.mean(axis=0),'o-',label=label,color=color);ax.fill_between([-1,0,1],values.mean(axis=0)-values.std(axis=0,ddof=1),values.mean(axis=0)+values.std(axis=0,ddof=1),alpha=.15,color=color)
    ax.set_xticks([-1,0,1],['Low','Center','High']);ax.set(title=names[j],ylabel='GP SHAP main contribution')
axes[0,0].legend(fontsize=8)
save_figure(fig,'phase_b_curvature.pdf',[S/f'snapshot_{rep}_doe_{cycle}.json' for rep in range(cfg['search_replicates']) for cycle in [0,cfg['adaptive_cycles']]])

confirmations=read(S/'confirmation.json')
fig,axes=plt.subplots(1,2,figsize=(9,3.7))
for ax,response,title in zip(axes,['val_loss','latency_us'],['Classification loss','CPU latency (microseconds)']):
    allvalues=[]
    for method,label in labels.items():
        data=[r for r in confirmations if r['method']==method and r['response']==response]
        x=[r['predicted'] for r in data];y=[r['observed'] for r in data];allvalues+=x+y
        ax.scatter(x,y,s=10,alpha=.35,label=label,color=colors[method])
    lo,hi=min(allvalues),max(allvalues);ax.plot([lo,hi],[lo,hi],'k--',lw=1)
    ax.set(xlabel='Predicted response',ylabel='Observed response',title=title)
axes[0].legend(fontsize=8)
save_figure(fig,'phase_b_confirmation.pdf',[S/'confirmation.json'])

word_errors=np.array([d['confirmation']['rmse'] for d in words]);chosen=int(np.argmax(word_errors));document=words[chosen]
metric('WordExampleIndex',chosen+1,B/'imdb_words/observations.json','document with largest reserved-perturbation RMSE, one-based index','.0f')
fig,axes=plt.subplots(1,3,figsize=(12,3.7))
axes[0].bar(np.arange(1,len(words)+1),word_errors,color='#3566a1');axes[0].set(xlabel='Document',ylabel='Probability RMSE',title='Local surface fidelity')
x=np.array(document['confirmation']['predicted']);y=np.array(document['confirmation']['observed']);axes[1].scatter(x,y,s=15,color='#d47b26');lo=min(x.min(),y.min());hi=max(x.max(),y.max());axes[1].plot([lo,hi],[lo,hi],'k--');axes[1].set(xlabel='RSM probability',ylabel='Classifier probability',title='Most difficult local example')
mat=np.array(document['shap']['interaction_matrix_shap']);np.fill_diagonal(mat,0);im=axes[2].imshow(mat,cmap='Oranges');axes[2].set_xticks(range(4),document['words'],rotation=25,ha='right');axes[2].set_yticks(range(4),document['words']);axes[2].set_title('Word-pair SHAP strength');fig.colorbar(im,ax=axes[2],shrink=.7)
save_figure(fig,'phase_b_words.pdf',[B/'imdb_words/observations.json'])
from posthoc_publication import emit, residual_diagnostics
posthoc=emit(ROOT,OUT,metric,table)
figure_sources['phase_a_diagnostics.pdf']=[p.relative_to(ROOT).as_posix() for p in residual_diagnostics(ROOT,OUT)]
sources.update(posthoc['source_sha256'])
sources['report/posthoc_analysis.py']=source_digest(ROOT/'report/posthoc_analysis.py')
figure_sources['phase_b_sensitivity.pdf']=['report/posthoc_results.json','results/phase_b/imdb/evaluations.jsonl']
figure_sources['phase_b_confirmation.pdf']=['report/posthoc_results.json','results/phase_b/imdb/evaluations.jsonl']
from presentation_figures import build as presentation_figures
for name,paths in presentation_figures(ROOT,OUT/'figures').items():
    figure_sources[name]=[p.relative_to(ROOT).as_posix() for p in paths]
metric('PostWordInteractionMax',np.max(np.abs(np.array(document['shap']['raw_interaction_values'])[0][np.triu_indices(4,1)])),B/'imdb_words/observations.json','illustrated highest-RMSE document: maximum absolute endpoint-game pair interaction','.4f')
metric('PostAdaptiveCoordinates',cfg['adaptive_cycles']*cfg['infill_per_cycle'],S/'measurement_contract.json','config.adaptive_cycles * config.infill_per_cycle','.0f','protocol')
for name,definition in [('Error',aconfig['desirability']['Y1_RMSE']),('Latency',aconfig['desirability']['Y2_Latency'])]:
    for side,key in [('Low','L'),('High','U')]:metric('Desirability'+name+side,definition[key],ROOT/'phase_a_rsm_blocking/config.yaml','desirability.'+name+'.'+key,'.3f','protocol')
import ast
tree=ast.parse((ROOT/'phase_a_rsm_blocking/analysis.py').read_text())
definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='derringer_suich_desirability')
defaults={arg.arg:ast.literal_eval(v) for arg,v in zip(definition.args.args[-len(definition.args.defaults):],definition.args.defaults)}
if len({defaults[k] for k in ['s','t','w1','w2']})!=1:raise ValueError('Desirability statement requires equal recorded default exponents and weights')
metric('DesirabilityExponent',defaults['s'],ROOT/'phase_a_rsm_blocking/analysis.py','derringer_suich_desirability: common default s,t,w1,w2','.0f','protocol')

# A classifier-level TreeSHAP additivity sanity check; no classifier is refitted.
import xgboost as xgb
from scipy import sparse
word_model=B/'imdb_words/classifier.ubj'; word_inputs=B/'imdb_words/inputs.npz'
booster=xgb.Booster();booster.load_model(word_model);booster.set_param({'device':'cpu','nthread':1})
dmatrix=xgb.DMatrix(sparse.load_npz(word_inputs),nthread=1)
contributions=booster.predict(dmatrix,pred_contribs=True)
margin=booster.predict(dmatrix,output_margin=True)
metric('TreeShapDocuments',len(margin),word_inputs,'saved development documents for classifier TreeSHAP check','.0f')
metric('TreeShapAdditivity',np.max(np.abs(contributions.astype(float).sum(axis=1)-margin)),word_model,'max abs(sum exact tree-path-dependent TreeSHAP contributions including bias minus raw classifier margin)','.3e')
sources[word_inputs.relative_to(ROOT).as_posix()]=source_digest(word_inputs)

# Source-bound configuration and same-surrogate explanation disclosures.
selected=read(S/'selected.json'); rows=[]
from xgb_doe_xai.config import uncode_configuration
for i,r in enumerate(sorted(selected,key=lambda r:(r['method'],r['replicate']))):
    natural=uncode_configuration(r['coded_params']); prefix='Selected'+chr(65+i)
    cells=[labels[r['method']],metric(prefix+'Repeat',r['replicate'],S/'selected.json',r['method']+': saved replicate index '+str(r['replicate']),'.0f','protocol')]
    for j,(key,value) in enumerate(natural.items()):cells.append(metric(prefix+chr(65+j),value,S/'selected.json',r['method']+':'+str(r['replicate'])+':'+key,'.5g'))
    rows.append(cells)
table('b_selected',['Method','Repeat index','Learning rate','Depth','Subsample','Regularization'],rows)
confirmation=read(A/'confirmation.json'); rows=[]
for i,(label,values) in enumerate([('Compromise',confirmation['x_star_coded']),('Accuracy',confirmation['Single_Objective_Optimum']['x_coded'])]):
    rows.append([label]+[metric('ACoord'+chr(65+i)+chr(65+j),v,A/'confirmation.json',label+': coded factor '+str(j+1),'.5f') for j,v in enumerate(values)])
table('a_coordinates',['Candidate','Rate','Depth','Subsample','Regularization'],rows)
rows=[]
for i,(method,label) in enumerate(labels.items()):
    cells=[label]
    for model in ['rsm','gp']:
        for field in ['spearman_rho','interaction_correlation']:
            paths=[S/f"snapshot_{rep}_{method}_{cfg['adaptive_cycles'] if method=='doe' else 'final'}.json" for rep in range(cfg['search_replicates'])]
            value=np.mean([read(p)['val_loss']['comparisons'][model][field] for p in paths])
            for p in paths:sources[p.relative_to(ROOT).as_posix()]=source_digest(p)
            cells.append(metric('Matched'+chr(65+i)+model+field.replace('_',''),value,paths[0],'mean across repeats: val_loss.comparisons.'+model+'.'+field,'.3f'))
    rows.append(cells)
table('b_same_model',['Sampling','RSM rank','RSM pairs','GP rank','GP pairs'],rows)
tex='\n'.join('\\expandafter\\def\\csname metric'+k+'\\endcsname{'+show(k)+'}' for k in registry)
(OUT/'numbers.tex').write_text(tex,encoding='utf-8',newline='\n')
accelerator=contract['accelerator'].split(',')[0]
timing_environment=observations[0]['latency']['environment']
text_metrics={
 'AcceleratorName':dict(value=accelerator,source='results/phase_b/imdb/measurement_contract.json',selector='accelerator name'),
 'InferenceProcessor':dict(value=timing_environment['processor'],source='results/phase_b/imdb/evaluations.jsonl',selector='latency.environment.processor'),
 'XGBoostVersion':dict(value=contract['dependencies']['xgboost'],source='results/phase_b/imdb/measurement_contract.json',selector='dependencies.xgboost'),
 'PythonVersion':dict(value=contract['python'].split()[0],source='results/phase_b/imdb/measurement_contract.json',selector='python version')}
with (OUT/'numbers.tex').open('a',encoding='utf-8',newline='\n') as stream:
    for key,definition in text_metrics.items():
        stream.write('\n\\newcommand{\\'+key+'}{'+definition['value']+'}\n')

for paths in figure_sources.values():
 for path in paths:sources[path]=source_digest(ROOT/path)
(OUT/'metric_registry.json').write_text(json.dumps(dict(metrics=registry,source_sha256=sources,
    text_metrics=text_metrics,
    figures=figure_sources),indent=2),encoding='utf-8',newline='\n')
print('Evidence prepared:',len(registry),'bound metrics;',len(sources),'source files; Phase A artifact hashes and Phase B reconstruction passed.')
