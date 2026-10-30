"""Publication tables and figure for explicitly post-hoc saved-evidence analyses."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from posthoc_analysis import build, quadratic
import pandas as pd
from scipy.stats import probplot


def residual_diagnostics(root, out):
    """Descriptive raw/log diagnostics from the same blocked quadratic fits."""
    source = root/'phase_a_rsm_blocking/results/runs.csv'
    observations = pd.read_csv(source)
    X = quadratic(observations[['x1','x2','x3','x4']].to_numpy(), observations.block.to_numpy())
    rank = np.linalg.matrix_rank(X)
    inverse = np.linalg.pinv(X.T@X)
    leverage = np.einsum('ij,jk,ik->i', X, inverse, X)
    blocks = sorted(observations.block.unique())
    with plt.rc_context({'font.size':10, 'pdf.compression':0}):
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
        for row, (transform, title, unit) in enumerate([
                (lambda values: values, 'Raw validation RMSE', 'RMSE'),
                (np.log, 'Log-transformed validation RMSE', 'log(RMSE)')]):
            y = transform(observations.val_rmse.to_numpy())
            fitted = X@np.linalg.lstsq(X, y, rcond=None)[0]
            residual = y-fitted
            sigma = np.sqrt(residual@residual/(len(y)-rank))
            studentized = residual/(sigma*np.sqrt(1-leverage))
            for index, block in enumerate(blocks):
                selected = observations.block.to_numpy() == block
                axes[row,0].scatter(fitted[selected], residual[selected], s=16, alpha=.65,
                    color=plt.get_cmap('tab10')(index), label=f'Block {block}')
            axes[row,0].axhline(0, color='black', linewidth=.8, linestyle='--')
            axes[row,0].set(title=title, xlabel='Fitted '+unit, ylabel='Residual ('+unit+')')
            (theoretical, ordered), (slope, intercept, _) = probplot(studentized, fit=True)
            axes[row,1].scatter(theoretical, ordered, s=16, alpha=.7, color='#3566a1')
            axes[row,1].plot(theoretical, intercept+slope*theoretical, 'k--', linewidth=.8)
            axes[row,1].set(title='Normal Q–Q: '+title, xlabel='Normal theoretical quantile',
                            ylabel='Internally studentized residual')
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(blocks), frameon=False)
        fig.tight_layout(rect=(0,.05,1,1))
        fig.savefig(out/'figures/phase_a_diagnostics.pdf', metadata={'CreationDate':None,'ModDate':None})
        plt.close(fig)
    return [source]


def emit(root, out, metric, table):
    data = build(root)
    (out/'posthoc_results.json').write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    source = root/'report/posthoc_results.json'
    def value(name, path, fmt='.5f', scale=1):
        current = data
        for key in path:
            current = current[key]
        return metric(name, current*scale, source, 'POST_HOC:'+json.dumps(path)+':scale='+str(scale), fmt)
    for name, path, fmt in [
        ('PostCapacity', ['capacity_matching','capacity_matched_median_us'], '.3f'),
        ('PostLogLof', ['phase_a_diagnostics','log_response','lack_of_fit_f'], '.2f'),
        ('PostRestrictedLof', ['phase_a_diagnostics','restricted_domain','lack_of_fit_f'], '.2f'),
        ('PostRestrictedRtwo', ['phase_a_diagnostics','restricted_domain','r_squared'], '.4f'),
        ('PostRestrictedRank', ['phase_a_diagnostics','restricted_domain','quadratic_rank'], '.0f'),
        ('PostRestrictedColumns', ['phase_a_diagnostics','restricted_domain','quadratic_columns'], '.0f'),
        ('PostBootstrapCount', ['paired_document_bootstrap','repetitions'], '.0f'),
        ('PostPairCount', ['interaction_leverage','per_repeat','static_doe',0,'pair_count'], '.0f'),
        ('PostMeanConfirmationN', ['symmetric_confirmation','summaries','static_doe:RSM:val_loss','n'], '.0f')]:
        value(name, path, fmt)
    methods={'static_doe':'Static CCD','doe':'Sequential CCD','tpe':'MO-TPE'}
    rows=[]
    for index, comparison in enumerate(data['paired_document_bootstrap']['comparisons']):
        prefix='PostPair'+chr(65+index);path=['paired_document_bootstrap','comparisons',index]
        rows.append([methods[comparison['first']]+' -- '+methods[comparison['second']],
            value(prefix+'Difference',path+['difference'],'.6f'),
            value(prefix+'Lower',path+['interval_95',0],'.6f'),
            value(prefix+'Upper',path+['interval_95',1],'.6f')])
    table('b_bootstrap',['Frozen-model comparison','F1 difference','Lower bound','Upper bound'],rows)
    rows=[]
    for mi,(method,label) in enumerate(methods.items()):
        for ri,(response,title) in enumerate([('val_loss','Loss'),('latency_us','Latency')]):
            for model in ['RSM','GP']:
                path=['symmetric_confirmation','summaries',method+':'+model+':'+response]
                prefix='PostConfirm'+chr(65+mi)+str(ri)+model
                row=[label,title,model]
                for suffix, field, scale in [('Error','rmse',1),('Bias','mean_bias',1),('Coverage','coverage',100),('Boundary','boundary_bias',1)]:
                    row.append(value(prefix+suffix,path+[field],'.1f' if suffix=='Coverage' else ('.5f' if response=='val_loss' else '.2f'),scale))
                rows.append(row)
    table('b_validation',['Sampling','Response','Model','Pred. RMSE','Bias','Coverage (\%)','Boundary bias'],rows)
    rows=[]
    for index,(method,label) in enumerate(methods.items()):
        prefix='PostOperational'+chr(65+index)
        rows.append([label,
            value(prefix+'Pooled',['capacity_matching','methods',method,'selected_under_pooled'],'.0f'),
            value(prefix+'Matched',['capacity_matching','methods',method,'selected_under_capacity_matched'],'.0f'),
            '$'+value(prefix+'Seconds',['search_cost','summaries',method,'mean_total_seconds'],'.1f')+' \\pm '+value(prefix+'SecondsSd',['search_cost','summaries',method,'sd_total_seconds'],'.1f')+'$'])
    table('b_operational',['Method','Below original cap','Below matched reference','Search seconds'],rows)
    names=['repeated_preplanned_doe_multi_objective','repeated_preplanned_doe_single_objective',
           'multi_objective_tpe','constrained_tpe','single_objective_tpe','random_search']
    labels=['Repeated DOE compromise','Repeated DOE accuracy','MO-TPE','Constrained TPE','Accuracy-focused TPE','Random search']
    rows=[]
    for letter, method, label in zip('ABCDEF',names,labels):
        within=value('PostWithin'+letter,['phase_a_retraining',method,'mean_within_selection_sd'],'.5f')
        rows.append([label,'\\m{ABench'+letter+'Error}','\\m{ABench'+letter+'Sd}',within,'\\m{ABench'+letter+'Latency}'])
    table('a_benchmark',['Method','Test RMSE','Between-search SD','Within-selection SD','CPU latency ($\\mu$s)'],rows)
    final=json.loads((root/'results/phase_b/imdb/final_results.json').read_text())['results']
    rows=[]
    for letter,(method,label) in zip('ABC',methods.items()):
        selected=[r for r in final if r['method']==method]
        ll=metric('PostLoglossSd'+letter,np.std([r['val_logloss'] for r in selected],ddof=1),root/'results/phase_b/imdb/final_results.json',method+': sample SD final test log loss','.4f')
        lat=metric('PostLatencySd'+letter,np.std([r['latency']['predict_median_us'] for r in selected],ddof=1),root/'results/phase_b/imdb/final_results.json',method+': sample SD final CPU latency','.2f')
        rows.append([label,'$\\m{B'+letter+'Fone} \\pm \\m{B'+letter+'FoneSd}$',
            '$\\m{B'+letter+'Logloss} \\pm '+ll+'$','$\\m{B'+letter+'Latency} \\pm '+lat+'$',
            '$\\m{B'+letter+'Hv} \\pm \\m{B'+letter+'HvSd}$'])
    table('b_benchmark',['Method','Test macro F1','Log loss','CPU latency ($\\mu$s)','Dev. HV'],rows)
    rows=[]
    for index,(method,label) in enumerate(methods.items()):
        path=['interaction_leverage','summaries',method];prefix='PostLeverage'+chr(65+index)
        rows.append([label,value(prefix+'Rank',path+['spearman'],'.3f'),
            value(prefix+'Omit',path+['pearson_without_two_largest_anova_pairs'],'.3f'),
            value(prefix+'Share',path+['two_largest_anova_share'],'.3f')])
    table('b_interaction_checks',['Sampling','Pair-rank correlation','Pearson after omission','Two-pair ANOVA share'],rows)
    # The grid describes new development-only choices, never newly evaluated test models.
    grid=data['reference_sensitivity']['grid'];losses=sorted({g['loss_reference'] for g in grid});latencies=sorted({g['latency_reference_us'] for g in grid})
    fig,axes=plt.subplots(2,3,figsize=(11,6.5))
    for column,(method,label) in enumerate(methods.items()):
        for row,(field,title) in enumerate([('selected_depths','Mean selected depth'),('changed_selections','Changed selections')]):
            array=np.asarray([[np.mean(next(g for g in grid if g['loss_reference']==loss and g['latency_reference_us']==lat)['methods'][method][field]) for lat in latencies] for loss in losses])
            ax=axes[row,column];im=ax.imshow(array,cmap='Blues',vmin=0 if row else 3,vmax=3 if row else 9)
            ax.set_xticks(range(len(latencies)),[f'{v:g}' for v in latencies]);ax.set_yticks(range(len(losses)),[f'{v:g}' for v in losses]);ax.set(xlabel=r'Latency reference ($\mu$s)',ylabel='Loss reference',title=label+'\n'+title)
            for i,j in np.ndindex(array.shape):ax.text(j,i,f'{array[i,j]:.1f}' if row==0 else f'{array[i,j]:.0f}',ha='center',va='center',color='white' if array[i,j]>(1.5 if row else 6) else 'black')
    fig.tight_layout();fig.savefig(out/'figures/phase_b_sensitivity.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    colors={'static_doe':'#3566a1','doe':'#d47b26','tpe':'#27856c'}
    fig,axes=plt.subplots(2,2,figsize=(9.5,7))
    for row,(response,title) in enumerate([('val_loss','Classification loss'),('latency_us','Latency (microseconds)')]):
        for column,model in enumerate(['RSM','GP']):
            ax=axes[row,column];limits=[]
            for method,label in methods.items():
                records=data['symmetric_confirmation']['records'][method+':'+model+':'+response]
                x=[r['predicted'] for r in records];y=[r['observed'] for r in records];limits+=x+y
                ax.scatter(x,y,s=13,alpha=.55,color=colors[method],label=label)
            low,high=min(limits),max(limits);ax.plot([low,high],[low,high],'k--',lw=1)
            ax.set(title=model+': '+title,xlabel='Predicted block mean',ylabel='Observed block mean')
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'figures/phase_b_confirmation.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    return data


def semantic_equal(first, second):
    """Permit only tiny calculation roundoff; identities and hashes remain exact."""
    if isinstance(first,dict) and isinstance(second,dict):
        return first.keys()==second.keys() and all(semantic_equal(first[k],second[k]) for k in first)
    if isinstance(first,list) and isinstance(second,list):
        return len(first)==len(second) and all(semantic_equal(a,b) for a,b in zip(first,second))
    if isinstance(first,float) and isinstance(second,(int,float)):
        return bool(np.isclose(first,second,atol=1e-12,rtol=1e-10))
    return first==second
