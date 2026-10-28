"""Reconstruct publication metrics from the completed, frozen sentiment campaign."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REL='results/sentiment_revision_v1/confirmed_campaign'
WORD_REL='results/sentiment_revision_v1/word_masking'
DATASETS={'imdb':'Imdb','yelp':'Yelp'}
METHODS={'static_doe':'Static','doe':'Adaptive','tpe':'Tpe'}
_CACHE={}


def common_mean_confirmation(root,dataset,method,response):
    """Compare polynomial and GP forecasts against the same block-mean outcome."""
    base=Path(root)/REL/dataset
    contract=json.loads((base/'measurement_contract.json').read_text())
    rows=json.loads((base/'confirmation.json').read_text())
    observed,quadratic,gp_predictions=[],[],[]
    for replicate in range(contract['config']['search_replicates']):
        forecasts=json.loads((base/f'gp_confirmation_{replicate}_{method}_{response}.json').read_text())
        for i,point in enumerate(contract['confirmation']):
            group=[r for r in rows if r['method']==method and r['replicate']==replicate
                   and r['response']==response and r['config_id']==point['point_id']]
            if len(group)!=contract['config']['n_nuisance_blocks']:
                raise ValueError('Incomplete block-mean confirmation')
            measured=float(np.mean([r['observed'] for r in group]))
            if not np.isclose(measured,forecasts['observed'][i],rtol=1e-12,atol=1e-12):
                raise ValueError('GP confirmation observation differs from primary block responses')
            observed.append(measured)
            quadratic.append(float(np.mean([r['predicted'] for r in group])))
            gp_predictions.append(forecasts['predicted'][i])
    return np.array(observed),np.array(quadratic),np.array(gp_predictions)


def verified_campaign(root=ROOT):
    """Cache only after hashing every byte; changed evidence always invalidates it."""
    root=Path(root)
    base=root/REL
    registry=base/'programme_registry.json'
    if not registry.is_file() or not (base/'completion.json').is_file():
        raise ValueError('Sentiment campaign is incomplete; new empirical claims cannot be published')
    expected=json.loads(registry.read_text())
    actual={p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in base.rglob('*') if p.is_file() and p!=registry}
    if actual!=expected:
        raise ValueError('Sentiment evidence registry mismatch')
    word_base=root/WORD_REL
    if not (word_base/'registry.json').is_file():
        raise ValueError('Literal word-deletion study is incomplete')
    word_expected=json.loads((word_base/'registry.json').read_text())
    word_actual={p.relative_to(word_base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in word_base.rglob('*') if p.is_file() and p!=word_base/'registry.json'}
    if word_actual!=word_expected:
        raise ValueError('Word-deletion evidence registry mismatch')
    from xgb_doe_xai.study import source_hashes,check_study
    source=source_hashes()
    contract=json.loads((base/'programme_contract.json').read_text())
    if source!=contract['source_hashes']:
        raise ValueError('Sentiment source code differs from the frozen campaign')
    fingerprint=(str(root.resolve()),json.dumps(actual,sort_keys=True),json.dumps(word_actual,sort_keys=True),json.dumps(source,sort_keys=True))
    if fingerprint in _CACHE:
        return _CACHE[fingerprint]
    from xgb_doe_xai.research_analysis import analyze_study
    from xgb_doe_xai.word_study import check_word_study
    from scripts.word_masking_study import check as check_masking
    check_masking(base,word_base)
    from scipy import sparse
    result={'contract':contract}
    for dataset in DATASETS:
        check_study(base/dataset)
        check_word_study(base/(dataset+'_words'))
        original_vectors=sparse.load_npz(base/(dataset+'_words')/'inputs.npz')
        literal=json.loads((word_base/(dataset+'_observations.json')).read_text())
        for ordinal,entry in enumerate(literal['documents']):
            variants=sparse.load_npz(word_base/f'{dataset}_{ordinal}_vectors.npz')
            if variants.shape[1]!=original_vectors.shape[1]:
                raise ValueError('Word-deletion vocabulary changed')
            difference=variants[15]-original_vectors[ordinal]
            if difference.nnz and np.max(np.abs(difference.data))>1e-12:
                raise ValueError('Full-word context differs between word explanation analyses')
        analysis=analyze_study(base/dataset)
        if analysis!=json.loads((base/(dataset+'_analysis.json')).read_text()):
            raise ValueError('Research analysis differs from reconstructed observations')
        result[dataset]=dict(analysis=analysis,
            summary=json.loads((base/dataset/'summary.json').read_text()),
            words=literal,
            feature_weight_words=json.loads((base/(dataset+'_words')/'observations.json').read_text()),
            measurement=json.loads((base/dataset/'measurement_contract.json').read_text()))
    _CACHE.clear()
    _CACHE[fingerprint]=result
    return result


def metric_records(root=ROOT):
    bundle=verified_campaign(root)
    records={}
    def add(name,value,source,selector,precision=5,status='RECONSTRUCTED'):
        records['numText'+name]=dict(value=value,
            formatted=(value if isinstance(value,str) else r'\text{undefined}' if value is None else f'{value:.{precision}f}' if precision else f'{value:,.0f}'),
            source=source,selector=selector,verification_status=status)
    import inspect
    from xgb_doe_xai.surrogates import PolynomialRSMSurrogate
    alpha=inspect.signature(PolynomialRSMSurrogate.predict_interval).parameters['alpha'].default
    add('NominalCoveragePct',(1-alpha)*100,'xgb_doe_xai/surrogates.py',
        '(1 - PolynomialRSMSurrogate.predict_interval default alpha) * 100',0,'REGISTERED PROTOCOL')
    add('TargetPercent',bundle['contract']['analysis']['efficiency_target_fraction']*100,
        f'{REL}/programme_contract.json','analysis.efficiency_target_fraction * 100',0,'REGISTERED PROTOCOL')
    for dataset,label in DATASETS.items():
        item=bundle[dataset]
        source=f'{REL}/{dataset}_analysis.json'
        cfg=bundle['contract']['configs'][dataset]
        for name,field in [('Trees','n_trees'),('Blocks','n_nuisance_blocks'),
                           ('Repeats','search_replicates'),('Warmups','warmup_inferences'),
                           ('TimedCalls','timed_inferences'),('Threads','training_threads')]:
            add(label+name,cfg[field],f'{REL}/programme_contract.json',f'configs.{dataset}.{field}',0,'REGISTERED PROTOCOL')
        add(label+'LatencyLimit',cfg['latency_threshold_us'],f'{REL}/programme_contract.json',
            f'configs.{dataset}.latency_threshold_us',2,'REGISTERED PROTOCOL')
        add(label+'Fits',item['measurement']['search_fits_per_method']/cfg['search_replicates'],
            f'{REL}/{dataset}/measurement_contract.json','search_fits_per_method / config.search_replicates',0)
        dev = (bundle['contract']['imdb_cohort'].get('development_rows') or
               bundle['contract']['imdb_cohort'].get('split_counts', {}).get('train')
               if dataset == 'imdb' else
               bundle['contract']['source_provenance'][dataset]['split_counts']['train'])
        test = bundle['contract']['source_provenance'][dataset]['split_counts']['test']
        dev_selector = ('imdb_cohort.development_rows'
                        if (dataset == 'imdb' and 'development_rows' in bundle['contract']['imdb_cohort'])
                        else f'source_provenance.{dataset}.split_counts.train')
        for name, value, selector in [('DevelopmentRows', dev, dev_selector),
                                      ('TestRows', test, f'source_provenance.{dataset}.split_counts.test')]:
            add(label+name, value, f'{REL}/programme_contract.json', selector, 0, 'SOURCE-TRACED')
        fields={'FOne':'test_macro_f1','LogLoss':'test_logloss','ProbabilityRmse':'test_probability_rmse',
                'Latency':'test_latency_us','Hypervolume':'hypervolume','Regret':'observed_reference_regret',
                'Feasibility':'feasibility_rate','Overhead':'overhead_seconds'}
        for method,method_label in METHODS.items():
            from xgb_doe_xai.research_analysis import bootstrap_mean
            target=[entry['shared_target']['fits'][method] for entry in item['analysis']['progress'].values()]
            target_stats=bootstrap_mean(target)
            add(label+method_label+'TargetReached',target_stats['n'],source,
                f'count(progress.*.shared_target.fits.{method} != null)',0)
            add(label+method_label+'TargetFitsMean',target_stats['mean'],source,
                f'mean(defined progress.*.shared_target.fits.{method})',1)
            selected=[r for r in item['summary']['test_results'] if r['method']==method]
            add(label+method_label+'FinalSelections',len(selected),f'{REL}/{dataset}/summary.json',
                f'count(test_results[method={method}])',0)
            add(label+method_label+'FinalFeasible',sum(r['test_latency_us']<=cfg['latency_threshold_us'] for r in selected),
                f'{REL}/{dataset}/summary.json',
                f'count(test_results[method={method}].test_latency_us <= configs.{dataset}.latency_threshold_us)',0)
            for suffix,field in fields.items():
                for stat in ('mean','sd'):
                    add(label+method_label+suffix+stat.title(),item['analysis']['estimates'][method][field][stat],
                        source,f'estimates.{method}.{field}.{stat}',2 if suffix in ('Latency','Overhead') else 5)
            for response,response_label in [('val_loss','Loss'),('latency_us','Time')]:
                actual,quadratic,_=common_mean_confirmation(root,dataset,method,response)
                add(label+method_label+response_label+'OlsMeanRmse',float(np.sqrt(np.mean((actual-quadratic)**2))),
                    f'{REL}/{dataset}/confirmation.json',
                    f'RMSE(equal-block means of observed and predicted, method={method}, response={response})',
                    5 if response=='val_loss' else 2)
                agreement=item['analysis']['agreement'][method+':'+response]
                for role,field in [('Anova','top_factor_anova'),('Shap','top_factor_shap')]:
                    from collections import Counter
                    votes=Counter(r[field] for r in agreement['per_search'] if r[field] is not None)
                    common=votes.most_common()
                    winner=common[0][0] if common and (len(common)==1 or common[0][1]>common[1][1]) else None
                    title={'x1':'learning rate','x2':'tree depth','x3':'row subsampling','x4':'regularization'}.get(winner,'no unique most frequent factor')
                    add(label+method_label+response_label+role+'TopFactor',title,source,
                        f'unique mode(agreement.{method}:{response}.per_search[*].{field}); report ties')
                for role,field in [('Anova','top_interaction_anova'),('Shap','top_interaction_shap')]:
                    pairs=Counter(tuple(r[field]) for r in agreement['per_search'] if r[field] is not None)
                    ordered=pairs.most_common()
                    pair=ordered[0][0] if ordered and (len(ordered)==1 or ordered[0][1]>ordered[1][1]) else None
                    factor_labels={'x1':'learning rate','x2':'tree depth','x3':'row subsampling','x4':'regularization'}
                    title=' and '.join(factor_labels[f] for f in pair) if pair else 'no unique most frequent pair'
                    add(label+method_label+response_label+role+'TopPair',title,source,
                        f'unique mode(agreement.{method}:{response}.per_search[*].{field}); report ties')
                if method=='doe':
                    blocks=[record['spearman_rho'] for key,entry in item['analysis']['adaptive_change'].items()
                            if key.endswith(':'+response) for record in entry['final_by_block']]
                    spread=bootstrap_mean(blocks)
                    add(label+method_label+response_label+'BlockRhoSd',spread['sd'],source,
                        f'SD(defined adaptive_change.*:{response}.final_by_block[*].spearman_rho)',3)
                curv=[r['spearman_rho'] for r in item['analysis']['curvature_agreement']
                      if r['method']==method and r['response']==response]
                curv_stats=bootstrap_mean(curv)
                for stat in ('mean','sd'):
                    add(label+method_label+response_label+'CurvatureRho'+stat.title(),curv_stats[stat],source,
                        f'{stat}(defined curvature_agreement[method={method},response={response}].spearman_rho)',3)
                for suffix,field in [('FactorRho','spearman_rho'),('InteractionR','interaction_correlation')]:
                    for stat in ('mean','sd'):
                        add(label+method_label+response_label+suffix+stat.title(),agreement[field][stat],
                            source,f'agreement.{method}:{response}.{field}.{stat}',3)
                for model,section in [('Ols','confirmation'),('Gp','gp_confirmation')]:
                    entry=item['analysis'][section][method+':'+response]
                    for suffix,field in [('Rmse','rmse'),('Bias','observed_minus_predicted_bias'),
                                         ('Coverage','pi_coverage' if model=='Ols' else 'coverage')]:
                        value=entry[field]
                        if suffix=='Coverage':value*=100
                        add(label+method_label+response_label+model+suffix,value,source,
                            f'{section}.{method}:{response}.{field}'+(' * 100' if suffix=='Coverage' else ''),
                            1 if suffix=='Coverage' else 5 if response=='val_loss' else 2)
        word_source=f'{WORD_REL}/{dataset}_observations.json'
        documents=item['words']['documents']
        add(label+'WordDocuments',len(documents),word_source,'len(documents)',0)
        add(label+'WordRmse',float(np.mean([d['confirmation']['rmse'] for d in documents])),word_source,
            'mean(documents[*].confirmation.rmse)',5)
        add(label+'WordUndefinedRanks',sum(not d['rsm']['rank_defined'] for d in documents),word_source,
            'count(documents[*].rsm.rank_defined == false)',0)
    return records


def render_macros(root=ROOT):
    return '\n'.join('\\newcommand{\\'+name+'}{'+entry['formatted']+'}'
                      for name,entry in metric_records(root).items())+'\n'


def macro(name):
    return r'\numText'+name+'{}'


def render_benchmark_table(root=ROOT):
    metric_records(root)
    lines=[r'\begin{tabular}{lccccc}',r'\toprule',
        r'Method & Test macro F-score & Probability RMSE & Latency ($\mu$s) & Validation HV & Feasible \\',r'\midrule']
    for dataset,label in DATASETS.items():
        lines.append(r'\multicolumn{6}{l}{\textbf{'+('IMDb cohort' if dataset=='imdb' else 'Yelp sentences')+r'}} \\')
        for method,title in [('Static','Fixed CCD'),('Adaptive','Sequential CCD'),('Tpe','MO-TPE')]:
            prefix=label+method
            cells=[title]+[r'$'+macro(prefix+field+'Mean')+r'\,\pm\,'+macro(prefix+field+'Sd')+'$'
                            for field in ('FOne','ProbabilityRmse','Latency','Hypervolume')]
            cells.append(macro(prefix+'FinalFeasible')+'/'+macro(prefix+'FinalSelections'))
            lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule'
    return '\n'.join(lines+[r'\end{tabular}',''])


def render_confirmation_table(root=ROOT):
    metric_records(root)
    lines=[r'\begin{tabular}{lllrrr}',r'\toprule',
        r'Dataset & Model / target & Response & Prediction RMSE & Bias & Coverage (\%) \\',r'\midrule']
    for dataset,label in DATASETS.items():
        for response,title in [('Loss','F-score loss'),('Time',r'Latency ($\mu$s)')]:
            for model in ('Ols','Gp'):
                prefix=label+'Adaptive'+response+model
                cells=[dataset.upper(),r'\makecell[l]{OLS\\block observation}' if model=='Ols' else r'\makecell[l]{GP\\block mean}',title]+[
                    '$'+macro(prefix+field)+'$' for field in ('Rmse','Bias','Coverage')]
                lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',''])


def render_agreement_table(root=ROOT):
    metric_records(root)
    lines=[r'\begin{tabular}{llcc}',r'\toprule',
        r'Dataset & Response & Factor-order Spearman $\rho$ & Pair-effect Pearson $r$ \\',r'\midrule']
    for dataset,label in DATASETS.items():
        for response,title in [('Loss','F-score loss'),('Time','Latency')]:
            prefix=label+'Adaptive'+response
            cells=[dataset.upper(),title]+['$'+macro(prefix+field+'Mean')+r'\,\pm\,'+
                        macro(prefix+field+'Sd')+'$' for field in ('FactorRho','InteractionR')]
            lines.append(' & '.join(cells)+r' \\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',''])


def plot_research(root=ROOT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bundle=verified_campaign(root)
    colors={'static_doe':'#72833b','doe':'#286b94','tpe':'#b15a27'}
    labels={'static_doe':'Fixed CCD','doe':'Sequential CCD','tpe':'MO-TPE'}
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'legend.fontsize':9,
                         'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for column,(dataset,title) in enumerate([('imdb','IMDb development cohort'),('yelp','Yelp sentences')]):
        item=bundle[dataset]
        rows=item['summary']['test_results']
        for method in METHODS:
            subset=[r for r in rows if r['method']==method]
            x=np.array([r['test_latency_us'] for r in subset])
            y=np.array([r['test_macro_f1'] for r in subset])
            axes[0,column].scatter(x,y,color=colors[method],alpha=.45,s=22)
            axes[0,column].errorbar(x.mean(),y.mean(),xerr=x.std(ddof=1),yerr=y.std(ddof=1),
                fmt='o',capsize=3,color=colors[method],label=labels[method])
            curves=[entry['curves'][method] for entry in item['analysis']['progress'].values()]
            values=np.array([[p[1] for p in curve] for curve in curves])
            counts=np.array([p[0] for p in curves[0]])
            axes[1,column].plot(counts,np.median(values,axis=0),color=colors[method],label=labels[method])
            lower,upper=np.quantile(values,[.25,.75],axis=0)
            axes[1,column].fill_between(counts,lower,upper,color=colors[method],alpha=.15)
        axes[0,column].set(title=title,xlabel='Verified classic latency (microseconds/document)',ylabel='Holdout macro F-score')
        axes[0,column].axvline(bundle['contract']['configs'][dataset]['latency_threshold_us'],
                              color='#58636c',linestyle='--',linewidth=1,label='Declared design limit')
        axes[1,column].set(xlabel='Actual search classifier fits',ylabel='Validation hypervolume',
                          title='Progress from completed, equally weighted split blocks')
        axes[0,column].legend()
        axes[1,column].legend()
    figure.savefig(Path(root)/'figures/sentiment_comparison.png',dpi=180)
    plt.close(figure)
    figure,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for column,dataset in enumerate(DATASETS):
        entry=bundle[dataset]['words']['documents'][0]
        matrices=[np.array(entry['rsm']['interaction_matrix_ss']),
                  np.array(entry['shap']['raw_interaction_values'][0])]
        for row,(matrix,title) in enumerate(zip(matrices,['RSM partial interaction SS','Exact classifier SHAP interactions'])):
            np.fill_diagonal(matrix,np.nan)
            axis=axes[row,column]
            limit=float(np.nanmax(np.abs(matrix)))
            if row==0:
                image=axis.imshow(matrix,cmap='Blues',vmin=0,vmax=max(limit,1e-12))
            else:
                image=axis.imshow(matrix,cmap='RdBu_r',vmin=-max(limit,1e-12),vmax=max(limit,1e-12))
            axis.set(xticks=range(len(entry['words'])),yticks=range(len(entry['words'])),
                     xticklabels=entry['words'],yticklabels=entry['words'],title=dataset.upper()+': '+title)
            axis.tick_params(axis='x',labelrotation=30)
            if limit<1e-12:
                axis.text(.5,.5,'No detectable pair response',transform=axis.transAxes,
                          ha='center',va='center',bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
            figure.colorbar(image,ax=axis,shrink=.75,
                label='Partial SS (squared probability units)' if row==0 else 'Probability contribution')
    figure.savefig(Path(root)/'figures/sentiment_word_interactions.png',dpi=180)
    plt.close(figure)
    figure,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for column,dataset in enumerate(DATASETS):
        for row,(response,title) in enumerate([('val_loss','F-score loss'),('latency_us','Latency (microseconds)')]):
            actual,quadratic,gp=common_mean_confirmation(root,dataset,'doe',response)
            axis=axes[row,column]
            axis.scatter(quadratic,actual,label='Quadratic',color='#286b94',s=22,alpha=.7)
            axis.scatter(gp,actual,label='Gaussian process',color='#b15a27',s=22,alpha=.7,marker='x')
            all_values=np.concatenate([actual,quadratic,gp])
            low,high=float(all_values.min()),float(all_values.max())
            pad=max((high-low)*.04,1e-4)
            axis.plot([low-pad,high+pad],[low-pad,high+pad],color='#58636c',linestyle='--',linewidth=1)
            axis.set(xlim=(low-pad,high+pad),ylim=(low-pad,high+pad),
                title=dataset.upper()+': '+title,xlabel='Predicted block mean',ylabel='Observed block mean')
            axis.legend()
    figure.savefig(Path(root)/'figures/sentiment_confirmation.png',dpi=180)
    plt.close(figure)


def write_artifacts(root=ROOT):
    root=Path(root)
    verified_campaign(root)
    for name,renderer in [('tab_sentiment_benchmarks.tex',render_benchmark_table),
                          ('tab_sentiment_confirmation.tex',render_confirmation_table),
                          ('tab_sentiment_agreement.tex',render_agreement_table)]:
        (root/'tables'/name).write_text(renderer(root),encoding='utf-8')
    plot_research(root)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    if args.write:write_artifacts()
    else:
        verified_campaign()
        print('Both sentiment applications reconstruct from frozen observations.')
