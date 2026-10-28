"""Read-only campaign audit; writes only this report's derived evidence and tables."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'phase_b_sequential_xai'))
from xgb_doe_xai.study import check_study
from xgb_doe_xai.word_study import check_word_study
from xgb_doe_xai.final_evaluation import digest
import xgboost as xgb

run = ROOT / 'results/phase_b_run'
study = run / 'imdb'
summary = check_study(study)
check_word_study(run / 'imdb_words')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
contract = read(study / 'measurement_contract.json')
final = read(study / 'final_results.json')
frozen = read(study / 'finalized_selections.json')
assert digest(frozen) == final['contract_sha256']
data = ROOT / 'phase_b_sequential_xai/data/imdb.csv'
assert hashlib.sha256(data.read_bytes()).hexdigest() == contract['dataset_sha256']
with data.open(encoding='utf-8', newline='') as f:
    docs = list(csv.DictReader(f))
for split, field in [('train', 'development_sha256'), ('test', 'holdout_sha256')]:
    selected = [r for r in docs if r['split'] == split]
    assert digest([[r['text'].strip() for r in selected], [int(r['label']) for r in selected]]) == frozen[field]
rows = [json.loads(line) for line in (study / 'evaluations.jsonl').read_text().splitlines()]
for r in rows + final['results']:
    lat = r['latency']
    assert (lat['n_warmup'], lat['n_timed']) == (100, 500)
    assert lat['affinity_applied']
    for prefix in ['predict', 'inplace']:
        assert len(lat[prefix + '_raw_us']) == 500
        assert np.isclose(np.median(lat[prefix + '_raw_us']), lat[prefix + '_median_us'], rtol=0, atol=1e-10)
published = read(ROOT / 'findings/phase_b/values/metrics.json')
for key in ['methods', 'confirmation', 'search_fits', 'confirmation_fits', 'total_fits']:
    assert published[key] == summary[key], key
pilot = read(ROOT / 'phase_b_sequential_xai/results/calibration/imdb_pilot.json')['observations']
calibration = []
for eta in [0., -1.]:
    group = [r for r in pilot if r['coded_params']['x1'] == eta]
    calibration.append(dict(coded_eta=eta, tree_counts=[r['n_trees'] for r in group],
        macro_f1=[r['val_macro_f1'] for r in group],
        identical_probabilities=all(r['validation_probabilities'] == group[0]['validation_probabilities'] for r in group),
        probability_min=min(group[0]['validation_probabilities']), probability_max=max(group[0]['validation_probabilities'])))
booster = xgb.Booster()
booster.load_model(run / 'imdb_words/classifier.ubj')
trees = [json.loads(t) for t in booster.get_dump(dump_format='json')]
methods = {}
for method in contract['methods']:
    tests = [r for r in summary['test_results'] if r['method'] == method]
    methods[method] = dict(hv_mean=summary['methods'][method]['hypervolume_mean'],
        hv_sd=summary['methods'][method]['hypervolume_sd'],
        test_f1_mean=float(np.mean([r['test_macro_f1'] for r in tests])),
        test_logloss_mean=float(np.mean([r['test_logloss'] for r in tests])),
        test_latency_mean=float(np.mean([r['test_latency_us'] for r in tests])),
        selected_feasible=sum(r['test_latency_us'] <= contract['config']['latency_threshold_us'] for r in tests),
        selected_count=len(tests))
evidence = dict(status='BLOCKED_FOR_FINAL_SCIENTIFIC_SIGNOFF',
    checks=['Existing study hash/source/metric reconstruction passed', 'Local word model replay passed',
            'Dataset hash and development/test identities match frozen contract',
            'Final-selection contract digest matches final output',
            'All search, confirmation and final raw timing medians and counts checked',
            'Published main method summaries match reconstructed local evidence'],
    limitations=['No model fitting rerun', 'No causal diagnosis of GPU anomalies',
        'No matched CPU/GPU speedup measurement', 'No independent audit of historical Phase A results',
        'Historical test non-exposure is not proven by hashes', 'GitHub CI fails; raw run evidence is git-ignored'],
    split_counts=dict(collections.Counter(r['split'] for r in docs)),
    search_fits=summary['search_fits'], confirmation_fits=summary['confirmation_fits'],
    final_fits=len(final['results']), study_fits=summary['total_fits'],
    calibration_fits=len(pilot), local_model_fits=1,
    all_fit_count=summary['total_fits'] + len(pilot) + 1,
    elapsed_seconds=read(run / 'completion.json')['wall_seconds'],
    calibration=calibration, word_model=dict(trees=len(trees), leaf_only=sum('leaf' in t for t in trees)),
    methods=methods, confirmation=summary['confirmation'],
    source_bindings={'methods':'results/phase_b_run/imdb/summary.json reconstructed from evaluations.jsonl and final_results.json',
        'calibration':'phase_b_sequential_xai/results/calibration/imdb_pilot.json:observations',
        'word_model':'results/phase_b_run/imdb_words/classifier.ubj:get_dump',
        'split_counts':'phase_b_sequential_xai/data/imdb.csv:split',
        'elapsed_seconds':'results/phase_b_run/completion.json:wall_seconds'},
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [study/'artifact_registry.json', study/'measurement_contract.json', study/'final_results.json',
         ROOT/'phase_b_sequential_xai/results/calibration/imdb_pilot.json', run/'imdb_words/classifier.ubj', data]})
(OUT / 'verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
labels = {'static_doe':'Static CCD', 'doe':'Sequential CCD', 'tpe':'MO-TPE'}
def table(name, headers, records):
    content = '\\begin{tabular}{' + 'l' + 'r'*(len(headers)-1) + '}\n\\toprule\n'
    content += ' & '.join(headers) + ' \\\\\n\\midrule\n'
    content += ''.join(' & '.join(str(v) for v in row) + ' \\\\\n' for row in records)
    content += '\\bottomrule\n\\end{tabular}\n'
    (OUT / (name+'.tex')).write_text(content, encoding='utf-8')
table('benchmark', ['Method', 'HV mean', 'HV SD', 'Test F1', 'Log loss'],
      [[labels[m], *[f'{v[k]:.4f}' for k in ['hv_mean','hv_sd','test_f1_mean','test_logloss_mean']]] for m,v in methods.items()])
table('calibration', ['Trees', 'Coded rate', 'Macro F1', 'Min. probability', 'Max. probability'],
      [[r['n_trees'], r['coded_params']['x1'], f"{r['val_macro_f1']:.6f}",
        f"{min(r['validation_probabilities']):.3g}", f"{max(r['validation_probabilities']):.3g}"] for r in pilot])
table('confirmation', ['Method', 'Loss RMSE', 'Loss bias', 'PI coverage'],
      [[labels[m], f"{summary['confirmation'][m+':val_loss']['rmse']:.4f}",
        f"{summary['confirmation'][m+':val_loss']['observed_minus_predicted_bias']:.4f}",
        f"{100*summary['confirmation'][m+':val_loss']['pi_coverage']:.1f}\\%"] for m in methods])
macros = {'DevCount':evidence['split_counts']['train'], 'TestCount':evidence['split_counts']['test'],
          'StudyFits':evidence['study_fits'], 'AllFits':evidence['all_fit_count'],
          'ConfirmFits':evidence['confirmation_fits'], 'FinalFits':evidence['final_fits'],
          'LeafTrees':evidence['word_model']['leaf_only'], 'TreeCount':evidence['word_model']['trees'],
          'Elapsed':f"{evidence['elapsed_seconds']:.2f}"}
(OUT/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items()),encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k not in ['source_sha256','confirmation']}, indent=2))
