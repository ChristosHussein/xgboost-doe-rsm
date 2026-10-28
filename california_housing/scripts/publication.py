"""Source/context bindings for the reader manuscript; no result-value allowlist."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def sha(path):
    path=Path(path)
    raw=path.read_bytes()
    if path.suffix in ('.tex','.py','.md','.json','.yaml','.yml','.csv'):
        raw=raw.replace(b'\r\n',b'\n')
    return hashlib.sha256(raw).hexdigest()


def protocol_observations(ds):
    """Read saved protocol/observations rather than transcribing manuscript counts."""
    frozen = json.loads((ds.rev_dir/'finalized_selections.json').read_text())
    protocol = frozen['final_evaluation']['protocol']
    total = protocol['dataset']['n_rows']
    test = math.ceil(total*protocol['external_test_fraction'])
    dev = total-test
    valid = math.ceil(dev*protocol['development_validation_fraction'])
    runs, cfg = ds.runs_df, ds.config
    blocks = runs['block'].nunique()
    result = {}
    def add(name, value, source, selector, category='RECONSTRUCTED', fmt=None):
        result['numPub'+name] = dict(value=value, formatted=fmt(value) if fmt else str(value),
            source=source, selector=selector, verification_status=category)
    raw = 'results/runs.csv'
    spec = 'results/revision_v2/full_run_001/finalized_selections.json'
    for name, value, selector in [
        ('Rows', total, 'final_evaluation.protocol.dataset.n_rows'),
        ('TestRows', test, 'ceil(n_rows * external_test_fraction)'),
        ('DevRows', dev, 'n_rows - test_rows'),
        ('TrainRows', dev-valid, 'development_rows - validation_rows'),
        ('ValRows', valid, 'ceil(development_rows * development_validation_fraction)'),
        ('HoldoutPct', protocol['external_test_fraction']*100, 'external_test_fraction * 100')]:
        add(name, value, spec, selector, 'SOURCE-TRACED', lambda v: f'{v:,.0f}')
    add('DesignRuns', len(runs), raw, 'row_count')
    add('Blocks', blocks, raw, 'nunique(block)')
    add('Factors', len(cfg['factors']), 'config.yaml', 'count(factors)', 'REGISTERED PROTOCOL')
    add('DepthMiddle', sorted(runs['depth'].unique())[len(runs['depth'].unique())//2],
        raw, 'middle observed depth level')
    add('Coordinates', len(runs[['x1','x2','x3','x4']].drop_duplicates()), raw, 'distinct coded configurations')
    for name, phase in [('Corners','Phase1_Factorial'),('Centers','Phase1_Center'),('Axial','Phase2_Axial')]:
        add(name, int(runs['phase'].eq(phase).sum()//blocks), raw, f'count(phase={phase}) / blocks')
    add('AxialTotal', int(runs['phase'].eq('Phase2_Axial').sum()), raw, 'count(phase=Phase2_Axial)')
    add('ScreeningRuns', int(runs['phase'].str.startswith('Phase1').sum()), raw, 'count(phase starts Phase1)')
    add('Trees', cfg['model']['n_estimators'], 'config.yaml', 'model.n_estimators', 'REGISTERED PROTOCOL')
    for label, factor in [('Eta','x1'),('Depth','x2'),('Sub','x3'),('Lambda','x4')]:
        for suffix, bound in [('Low','min'),('High','max')]:
            add(label+suffix, cfg['factors'][factor][bound], 'config.yaml', f'factors.{factor}.{bound}',
                'REGISTERED PROTOCOL', lambda v: f'{v:g}')
    add('SearchReplicates', ds.final_summary_df.loc[ds.final_summary_df['optimizer'].eq('multi_objective_tpe'),
        'optimizer_replicate_id'].nunique(), 'results/revision_v2/full_run_001/final_summary.csv',
        'count distinct optimizer_replicate_id where optimizer=multi_objective_tpe')
    for name, value, selector in [('EvalSeeds', ds.final_evaluations_df.evaluation_seed.nunique(), 'nunique'),
        ('FirstEvalSeed', ds.final_evaluations_df.evaluation_seed.min(), 'min'),
        ('LastEvalSeed', ds.final_evaluations_df.evaluation_seed.max(), 'max')]:
        add(name, int(value), 'results/revision_v2/full_run_001/final_evaluations.csv', f'{selector}(evaluation_seed)')
    budget_source = 'results/revision_v2/full_run_001/computational_budget.json'
    for name, field in [('OptimizerTrials','trials_per_optimizer'),
                        ('RepeatedBlocks','repeated_doe_seeds_per_replicate')]:
        add(name, ds.computational_budget[field], budget_source, field, 'SOURCE-TRACED')
    repeated = ds.computational_budget['repeated_doe_design_model_fits'] / ds.computational_budget['repeated_doe_selection_replicates']
    if repeated != ds.computational_budget['trials_per_optimizer']:
        raise ValueError('Shared search-fit budget claim is unsupported')
    add('SearchFits', int(repeated), budget_source,
        'repeated_doe_design_model_fits / repeated_doe_selection_replicates == trials_per_optimizer', 'SOURCE-TRACED')
    for name, field in [('OnlineCalls','online_latency_timed_calls_per_measurement'),
                         ('PrimaryCalls','primary_latency_calls_per_frozen_configuration_per_interface')]:
        add(name, ds.computational_budget[field], 'results/revision_v2/full_run_001/computational_budget.json', field,
            'SOURCE-TRACED', lambda v: f'{v:,}')
    for name, field in [('PrimarySessions','primary_latency_sessions'),
                         ('OnlineWarmups','online_latency_warmups_per_measurement')]:
        add(name, ds.computational_budget[field], 'results/revision_v2/full_run_001/computational_budget.json',
            field, 'SOURCE-TRACED')
    add('LatencyLimit', cfg['revision_v2']['max_latency_us'], 'config.yaml', 'revision_v2.max_latency_us',
        'REGISTERED PROTOCOL', lambda v: f'{v:g}')
    add('DepthMo', ds.confirmation['x_star_natural']['depth'], 'results/confirmation.json', 'x_star_natural.depth')
    add('DepthSo', ds.confirmation['Single_Objective_Optimum']['x_natural']['depth'],
        'results/confirmation.json', 'Single_Objective_Optimum.x_natural.depth')
    depths = sorted({x['hyperparameters']['max_depth'] for x in ds.finalized_selections if x['optimizer'] == 'constrained_tpe'})
    for name, value in [('CtpeShallow', depths[0]), ('CtpeDeep', depths[-1])]:
        add(name, value, spec, 'ordered distinct constrained_tpe params.max_depth')
    add('ConfirmationSeeds', list(ds.confirmation_seeds), 'config.yaml', 'seeds.confirmation_seeds',
        'REGISTERED PROTOCOL', lambda v: ', '.join(map(str,v)))
    coverage = next(int(re.search(r'prediction_interval_(\d+)', k)[1])
                    for k in ds.confirmation['Y1_Val_RMSE'] if k.startswith('prediction_interval_'))
    add('CoveragePct', coverage, 'results/confirmation.json', 'Y1_Val_RMSE prediction_interval confidence label', 'REGISTERED PROTOCOL')
    add('DevelopmentSeed', ds.provenance['dataset']['development_split_seed'],
        'results/revision_v2/full_run_001/provenance.json', 'dataset.development_split_seed', 'REGISTERED PROTOCOL')
    for label, point in [('Primary', ds.hv_ref_primary), ('Secondary', ds.hv_ref_secondary)]:
        for suffix, value in [('Loss', point[0]), ('Time', point[1])]:
            add('Hv'+label+suffix, value, 'config.yaml', f'revision_v2.hypervolume_reference_points.{label}.{suffix}',
                'REGISTERED PROTOCOL', lambda v, suffix=suffix: f'{v:.2f}' if suffix == 'Loss' else f'{v:.0f}')
    add('FullFrontCount', ds.computational_budget['matched_historical_doe_candidate_fits'],
        'results/revision_v2/full_run_001/computational_budget.json', 'matched_historical_doe_candidate_fits', 'SOURCE-TRACED')
    return result


def render_protocol_macros(ds):
    return '\n'.join('\\newcommand{\\'+name+'}{'+str(spec['formatted'])+'}'
                     for name,spec in protocol_observations(ds).items())+'\n'


def render_manuscript(ds):
    return (ds.root_dir/'scripts/templates/report.tex').read_text(encoding='utf-8')


def check_literal_numbers(tex):
    """Reject naked empirical digits; explicitly type layout, citation and math literals."""
    body = tex.split(r'\begin{document}',1)[-1].split(r'\begin{thebibliography}',1)[0]
    body = re.sub(r'\\(?:cite|label|ref|url|includegraphics|input)(?:\[[^]]*\])?\{[^}]*\}', '', body)
    body = re.sub(r'\\(?:setlength|setcounter|renewcommand)\{[^}]*\}\{[^}]*\}', '', body)
    body = re.sub(r'\d+(?:\.\d+)?(?:em|pt|in|cm|mm|ex)|\d+(?:\.\d+)?\\(?:textwidth|linewidth)', '', body)
    body = re.sub(r'R\^2|L_2|Phase [12]', '', body)
    body = re.sub(r'\\num[A-Za-z]+(?:\{\})?', '', body)
    return re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?', body)


def bindings(ds):
    from scripts.generate_report_artifacts import render_macros
    tex = render_manuscript(ds)
    naked = check_literal_numbers(tex)
    if naked:
        raise ValueError(f'Unbound numeric literals in reader manuscript: {naked}')
    macros = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^\n]*)\}', render_macros(ds)))
    origins = {}
    source = ds.root_dir/'scripts/generate_report_artifacts.py'
    for node in ast.walk(ast.parse(source.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'add_macro':
            if node.args and isinstance(node.args[0], ast.Constant):
                origins[node.args[0].value] = dict(source=f'scripts/generate_report_artifacts.py:{node.lineno}',
                    selector=ast.unparse(node.args[1]), verification_status='RECONSTRUCTED')
    origins.update(protocol_observations(ds))
    counts, occurrences = Counter(), []
    section = 'abstract'
    for line in tex.splitlines():
        heading = re.search(r'\\section\{([^}]+)\}', line)
        if heading:
            section = heading[1]
        for match in re.finditer(r'\\(num[A-Za-z]+)\b', line):
            name = match[1]
            if name not in macros:
                raise ValueError(f'Undefined result binding: {name}')
            counts[(section,name)] += 1
            origin = origins.get(name, dict(source='scripts/generate_report_artifacts.py:render_macros',
                selector=f'dynamic add_macro name={name}; reconstructed by ReportingDataset', verification_status='RECONSTRUCTED'))
            occurrences.append(dict(claim_id=f'{section}/{name}/{counts[(section,name)]}',
                metric_id=name, document='report.tex', section=section, context=line.strip(),
                formatted_value=macros[name], **{k:v for k,v in origin.items() if k not in ('value','formatted')}))
    return dict(evidence_scope='Housing empirical claims; extension methodology only; sentiment research NOT RUN',
                declaration='RETROSPECTIVE map of existing housing evidence',
                occurrences=occurrences, occurrence_count=len(occurrences))


def write_publication(ds):
    (ds.root_dir/'report.tex').write_text(render_manuscript(ds), encoding='utf-8')
    path = ds.root_dir/'docs/generated/publication_claim_bindings.json'
    path.write_text(json.dumps(bindings(ds), indent=2)+'\n', encoding='utf-8')
    (ds.root_dir/'README.md').write_text(render_readme(ds),encoding='utf-8')
    (ds.root_dir/'docs/report_integrity_audit.md').write_text(render_integrity_audit(ds),encoding='utf-8')
    (ds.root_dir/'docs/report_rewrite_coverage.md').write_text(render_coverage(ds),encoding='utf-8')


def render_readme(ds):
    return (ds.root_dir/'scripts/templates/README.md').read_text(encoding='utf-8')


def render_coverage(ds):
    from pypdf import PdfReader
    text=(ds.root_dir/'scripts/templates/report_coverage.md').read_text(encoding='utf-8')
    return text.replace('{{MAIN_PAGES}}',str(len(PdfReader(ds.root_dir/'report.pdf').pages))).replace(
        '{{SUPP_PAGES}}',str(len(PdfReader(ds.root_dir/'technical_supplement.pdf').pages)))


def render_integrity_audit(ds):
    registry=bindings(ds)
    statuses=Counter(r['verification_status'] for r in registry['occurrences'])
    pages=build_record(ds.root_dir)['pages']
    return '\n'.join([
        '# Publication integrity audit', '', '**Verdict: VERIFIED WITH DECLARED LIMITATIONS.**', '',
        '## Measurement plan and scope', '',
        'The housing evidence map is RETROSPECTIVE. The sequential text extension is an implemented method; '
        'its real sentiment-data campaign is NOT RUN. No new real-data performance, superiority, coverage or '
        'explanation-stability result is asserted. Synthetic execution checks are not research evidence.', '',
        '## Evidence and occurrence coverage', '',
        f"The main manuscript contains {registry['occurrence_count']} explicitly bound numerical macro occurrences. "
        'Each occurrence has its document section, exact narrative context, metric identity, formatted value, '
        'source selector and reconstruction expression in `docs/generated/publication_claim_bindings.json`.', '',
        '| Occurrence category | Count |', '| --- | ---: |',
        *[f'| {key} | {value} |' for key,value in sorted(statuses.items())], '',
        'Registered protocol values describe design choices, not observations. SOURCE-TRACED includes saved '
        'dataset dimensions, split proportions and timing ledgers; it does not claim a new independent reload '
        'of the original dataset. Housing results are reconstructed through `ReportingDataset` and the existing '
        'row-level verification suite. Bibliographic years, mathematical notation and layout dimensions are '
        'separately typed nonempirical constants. No numerical-vocabulary allowlist validates the reader manuscript.', '',
        '## Cross-surface and mutation checks', '',
        'The authored template contains no unbound empirical digit literals. The generated LaTeX and occurrence '
        'manifest must match it exactly. A valid value swapped into a different method context, a duplicate or '
        'missing claim, a changed unit or caveat, or a stale source fails comparison. The build manifest binds '
        'the actual compiled PDF to every imported table, macro and graphic. Missing or additional scientific '
        'documents fail the repository registry. Mutation tests run in disposable copies.', '',
        '## Material wording corrections', '',
        'Historical search timing included warmup. Primary benchmark verification used the recorded randomized '
        'session protocol; the earlier reader prose incorrectly described a different final-timing design. '
        'The main report now extracts both counts from the saved computational ledger. Archived DOE running '
        'minima mix validation blocks and cannot establish a common-estimand search-efficiency ranking; this '
        'limitation is now explicit beside the retained figure. Neither correction changes frozen observations.', '',
        '## Rendering and build', '',
        f'The current main PDF has {pages} pages. Its input and PDF digests and page count are recorded in '
        '`docs/generated/publication_build.json`. Compilation and text-layer context checks are automated; '
        'page-by-page visual inspection is a release activity and is not inferred from a checksum.', '',
        '## Remaining limitations', '',
        'The historical holdout was exposed before revised isolation. Dataset-sampling uncertainty and new '
        'sentiment-data generalization are not established. The extension uses fixed-block OLS inference, '
        'not unseen-block Satterthwaite inference. GP intervals and attribution agreement need empirical '
        'confirmation. Archived milestone reports are historical supporting records, not a substitute for '
        'the claim-bound manuscript. A green software workflow is not scientific validation of the new method.', '',
        '## Cleanup and preservation', '',
        'Superseded prototype outputs and unused presentation variants were relocated byte-identically to '
        '`docs/archive/`; the locked catalog records original and archive paths and digests. Frozen canonical '
        'evidence, statistical tests, publication sources and empirical figures remain available. Disposable '
        'build products and preview caches are excluded from the publication package.', ''])


def check_publication_source(ds):
    errors=[]
    for rel, expected in [('report.tex',render_manuscript(ds)),
        ('docs/generated/publication_claim_bindings.json',json.dumps(bindings(ds),indent=2)+'\n')]:
        if not (ds.root_dir/rel).is_file() or (ds.root_dir/rel).read_text(encoding='utf-8') != expected:
            errors.append(f'Publication occurrence/context mismatch: {rel}')
    return errors


def input_hashes(root, manuscript='report.tex'):
    tex=(root/manuscript).read_text(encoding='utf-8')
    paths={manuscript}
    for rel in re.findall(r'\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}',tex):
        paths.add(rel if Path(rel).suffix else rel+'.tex')
    return {rel:sha(root/rel) for rel in sorted(paths)}


def build_record(root):
    from pypdf import PdfReader
    return dict(inputs=input_hashes(root), pdf_sha256=sha(root/'report.pdf'),
                pages=len(PdfReader(root/'report.pdf').pages),
                supplement=dict(inputs=input_hashes(root,'technical_supplement.tex'),
                    pdf_sha256=sha(root/'technical_supplement.pdf'),
                    pages=len(PdfReader(root/'technical_supplement.pdf').pages)))


def check_pdf(ds):
    path=ds.root_dir/'docs/generated/publication_build.json'
    if not path.is_file() or json.loads(path.read_text()) != build_record(ds.root_dir):
        return ['Publication PDF is stale or unbound to its LaTeX sources; rebuild required']
    from pypdf import PdfReader
    text=' '.join(p.extract_text() or '' for p in PdfReader(ds.root_dir/'report.pdf').pages)
    if 'Shapley' not in text or 'sentiment' not in text or 'California Housing' not in text:
        return ['Publication PDF lacks required historical/extension scientific context']
    return []
