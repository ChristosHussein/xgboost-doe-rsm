"""Check publication contexts, source evidence and independent calculations.

Use --bind only after reviewing changed claim contexts. Ordinary generation and
CI never update those bindings. --write --seal records an already checked PDF.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile

from verify_evidence import audit
from verify_words import verify_words

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def digest(path, source=False):
    data = path.read_bytes()
    if source and path.suffix == '.py':
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_sources(root, sources):
    for name, expected in sources.items():
        require(digest(root/name, source=True) == expected, 'Changed evidence: '+name)


def claim_contexts(directory, registry):
    manuscript = (directory/'report.tex').read_text(encoding='utf-8')
    inputs = re.findall(r'\\input\{([^}]+)\}', manuscript)
    documents = ['report.tex'] + [p for p in inputs if p not in ['numbers.tex', 'references.tex']]
    require(set(p.name for p in directory.glob('*.tex')) == set(documents+['numbers.tex', 'references.tex']),
            'Unregistered or missing publication source')
    require(not list(directory.glob('*.md')), 'Unregistered publication document')
    require({p.name for p in directory.glob('*.pdf')} == {'report.pdf'}, 'Unregistered or missing publication PDF')
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{figures/([^}]+)\}', manuscript)
    require(set(figures) == set(registry['figures']), 'Figure source inventory differs from manuscript')
    require({p.name for p in (directory/'figures').iterdir() if p.is_file()} == set(figures),
            'Unregistered or missing figure')
    contexts = []
    for name in documents:
        body = (directory/name).read_text(encoding='utf-8')
        if name == 'report.tex':
            body = body.split('\\begin{document}', 1)[1]
        section = 'abstract' if name == 'report.tex' else 'table'
        for line in body.splitlines():
            heading = re.search(r'\\(?:sub)?section\{([^}]+)\}', line)
            if heading:
                section = heading[1]
            metrics = re.findall(r'\\m\{([^}]+)\}', line)
            text_metrics = [key for key in registry['text_metrics'] if '\\'+key in line]
            if metrics or text_metrics:
                require(all(key in registry['metrics'] for key in metrics), 'Unknown scientific metric')
                contexts.append(dict(document=name, section=section,
                    context=' '.join(line.split()), metrics=metrics, text_metrics=text_metrics))
        if name == 'report.tex':
            body = re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}', '', body, flags=re.S)
            body = re.sub(r'\\\[.*?\\\]', '', body, flags=re.S)
            body = re.sub(r'\\(?:texttt|url)\{[^}]*\}', '', body)
            body = re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]*\}', '', body)
            body = re.sub(r'\\set(?:counter|length)\{[^}]*\}\{[^}]*\}', '', body)
        body = re.sub(r'\\m\{[^}]+\}', '', body)
        body = re.sub(r'\bF1\b', 'F-score', body)
        require(not re.findall(r'\d+(?:\.\d+)?', body), 'Unbound numerical observation in '+name)
    references = (directory/'references.tex').read_text(encoding='utf-8')
    citations = {item.strip() for group in re.findall(r'\\cite\{([^}]+)\}', manuscript) for item in group.split(',')}
    bibliography = set(re.findall(r'\\bibitem\{([^}]+)\}', references))
    require(citations == bibliography, 'Missing or unused scholarly reference')
    return dict(contexts=contexts, figures=figures, documents=documents,
                bibliography_sha256=digest(directory/'references.tex'),
                exclusions={'references.tex':'Bibliographic metadata; no project observations.',
                            'numbers.tex':'Generated definitions checked against the source-backed metric registry.',
                            'equations_and_listings':'Mathematical definitions, executable commands and revision identifiers.'})


def verify_contexts(directory, registry):
    actual = claim_contexts(directory, registry)
    require(actual == read(directory/'claim_bindings.json'),
            'A scientific claim has changed context, method or metric; review its explicit binding')
    definitions = (directory/'numbers.tex').read_text(encoding='utf-8')
    for context in actual['contexts']:
        for key in context['metrics']:
            metric = registry['metrics'][key]
            require('\\csname metric'+key+'\\endcsname{'+format(metric['value'],metric['format'])+'}' in definitions,
                    'Stale numeric definition: '+key)
        for key in context['text_metrics']:
            require('\\newcommand{\\'+key+'}{'+registry['text_metrics'][key]['value']+'}' in definitions,
                    'Stale protocol definition: '+key)
    return actual


def publication_files():
    return {p.relative_to(OUT).as_posix(): digest(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.suffix in ['.tex','.pdf','.png','.py','.json']
            and p.name != 'publication.json' and '__pycache__' not in p.parts}


def self_test(registry):
    results = []
    def rejects(label, function):
        try:
            function()
        except (ValueError, FileNotFoundError):
            results.append(label)
        else:
            raise ValueError('Required negative test was accepted: '+label)
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary)/'report'
        shutil.copytree(OUT, target, ignore=shutil.ignore_patterns('__pycache__'))
        verify_contexts(target, registry)
        table = target/'b_validation.tex'
        original = table.read_text()
        table.write_text(original.replace('ConfirmAvallossError', 'ConfirmCvallossError'))
        rejects('valid metric under wrong method', lambda: verify_contexts(target, registry))
        table.write_text(original.replace('Static CCD', 'MO-TPE', 1))
        rejects('method label changed', lambda: verify_contexts(target, registry))
        table.write_text(original.replace('\\m{ConfirmAvallossError}', '0.54321'))
        rejects('unbound observation', lambda: verify_contexts(target, registry))
        table.write_text(original)
        added = target/'additional_results.md'
        added.write_text('Additional experimental result: 0.54321')
        rejects('unregistered scientific artifact', lambda: verify_contexts(target, registry))
        added.unlink()
        image = target/'figures/phase_b_progress.pdf'
        image.unlink()
        rejects('missing required figure', lambda: verify_contexts(target, registry))
        evidence = Path(temporary)/'observation.json'
        evidence.write_bytes((ROOT/'phase_a_rsm_blocking/results/confirmation.json').read_bytes())
        expected = digest(evidence)
        evidence.write_bytes(evidence.read_bytes()+b' ')
        rejects('modified primary evidence', lambda: verify_sources(Path(temporary), {'observation.json':expected}))
    changed = copy.deepcopy(registry)
    changed['metrics']['ConfirmAvallossError']['value'] = registry['metrics']['ConfirmCvallossError']['value']
    rejects('incorrect registry value despite valid source hashes', lambda: audit(ROOT, changed))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--bind', action='store_true', help='Record reviewed claim contexts; never used by CI')
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--seal', action='store_true', help='Hash the checked publication after its PDF is compiled')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    registry = read(OUT/'metric_registry.json')
    if args.bind:
        (OUT/'claim_bindings.json').write_text(json.dumps(claim_contexts(OUT, registry), indent=2)+'\n', encoding='utf-8')
        print('Reviewed scientific claim contexts recorded.')
        return
    verify_sources(ROOT, registry['source_sha256'])
    contexts = verify_contexts(OUT, registry)
    reconstruction = audit(ROOT, registry)
    word_replay = verify_words(ROOT)
    counts = Counter(key for context in contexts['contexts'] for key in context['metrics'])
    result = dict(verdict='VERIFIED WITH DECLARED LIMITATIONS',
        bound_occurrences=sum(counts.values()), used_metrics=len(counts),
        unbound_empirical_literals=0, evidence_hashes_checked=len(registry['source_sha256']),
        metrics_by_occurrence=dict(counts), reconstruction=reconstruction, word_model_replay=word_replay,
        limitations=['Historical evaluation-set exposure limits external generalization.',
                     'Shared folds and few search repeats limit method-level inference.',
                     'Single-input CPU timings do not estimate deployment latency.',
                     'The calibration-derived threshold is not an application requirement.'])
    if args.self_test:
        print('Negative tests rejected:', ', '.join(self_test(registry)))
    if args.write:
        (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if args.seal:
        require(args.write, '--seal requires --write')
        manifest = dict(paper='report.pdf', revision_tag='paper-2026-10-r2', files=publication_files(),
                        evidence_archive_sha256=digest(ROOT/'evidence/phase_b.zip'),
                        reference_source_archive_sha256=digest(ROOT/'evidence/phase_a_sources.zip'),
                        reference_environment_sha256=digest(ROOT/'requirements-reproduction.txt'))
        (OUT/'publication.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if args.check:
        require(result == read(OUT/'verification.json'), 'Saved verification summary is stale')
        published = read(OUT/'publication.json')
        require(published['files'] == publication_files(), 'Publication files differ from the compiled, checked release')
        require(published['evidence_archive_sha256'] == digest(ROOT/'evidence/phase_b.zip'), 'Published archive differs')
        require(published['reference_source_archive_sha256'] == digest(ROOT/'evidence/phase_a_sources.zip'), 'Historical source snapshot differs')
        require(published['reference_environment_sha256'] == digest(ROOT/'requirements-reproduction.txt'), 'Reference environment differs')
    print(f"{result['verdict']}: {sum(counts.values())} bound occurrences; "
          f"{len(reconstruction['reconstructed_metrics'])} metrics independently reconstructed; "
          f"{len(reconstruction['source_traced_metrics'])} additional metrics source-traced.")


if __name__ == '__main__':
    main()
