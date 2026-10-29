"""Verify semantic numeric bindings, evidence hashes and generated table values."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).parent
registry=json.loads((OUT/'metric_registry.json').read_text())
for name,h in registry['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'Changed evidence: '+name
main_source=OUT/'report.tex'
sources=[main_source]+[OUT/name for name in re.findall(r'\\input\{([^}]+)\}',main_source.read_text(encoding='utf-8')) if name!='numbers.tex']
counts={};tex=(OUT/'numbers.tex').read_text()
for p in sources:
    body=p.read_text(encoding='utf-8')
    for key in re.findall(r'\\m\{([^}]+)\}',body):
        r=registry['metrics'][key];expected=format(r['value'],r['format'])
        assert '\\csname metric'+key+'\\endcsname{'+expected+'}' in tex,key
        counts[key]=counts.get(key,0)+1
    # Numerical observations must be macro-bound. Permit only structural
    # preamble settings, equations and executable/path syntax outside bindings.
    if p.name=='report.tex':
        body=body.split('\\begin{document}',1)[1]
        body=re.sub(r'\\begin\{lstlisting\}.*?\\end\{lstlisting\}','',body,flags=re.S)
        body=re.sub(r'\\\[.*?\\\]','',body,flags=re.S)
        body=re.sub(r'\\(?:texttt|url)\{[^}]*\}','',body)
        body=re.sub(r'\\includegraphics(?:\[[^]]*\])?\{[^}]*\}','',body)
        body=re.sub(r'\\m\{[^}]*\}','',body)
        body=re.sub(r'\\set(?:counter|length)\{[^}]*\}\{[^}]*\}','',body)
        body=re.sub(r'\bF1\b','F-score',body)  # Metric name, not an observed numeric value.
        assert not re.findall(r'\d+(?:\.\d+)?',body),'Unbound numerical literal in manuscript'
    else:
        body=re.sub(r'\bF1\b','F-score',body)
        assert not re.findall(r'\d+(?:\.\d+)?',re.sub(r'\\m\{[^}]+\}','',body)),p.name
for f in registry['figures']:assert (OUT/'figures'/f).exists(),f
result=dict(verdict='VERIFIED WITH DECLARED LIMITATIONS',bound_occurrences=sum(counts.values()),
    used_metrics=len(counts),unbound_empirical_literals=0,evidence_hashes_checked=len(registry['source_sha256']),
    metrics_by_occurrence=counts,limitations=['Historical Phase A inferential summaries are source-traced, not all independently refitted',
    'Historical test exposure remains','No Phase B configuration satisfies the recorded latency cap',
    'No universal superiority or equivalence claim','Raw campaign evidence is local and git-ignored'])
(OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='metrics_by_occurrence'},indent=2))
