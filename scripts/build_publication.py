"""Build the manuscript in temporary storage, then bind the actual PDF to its inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.publication import build_record, check_pdf
from scripts.reporting_data import load_reporting_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:
        errors=check_pdf(load_reporting_dataset())
        if errors:
            raise SystemExit('\n'.join(errors))
        print('Publication PDF and LaTeX input bindings verified.')
        return
    compiler=shutil.which('pdflatex')
    if not compiler:
        raise SystemExit('pdflatex is required to rebuild the publication; existing deliverables preserved')
    from scripts.generate_research_reporting import write_research_reporting, check_research_reporting
    ds=load_reporting_dataset()
    write_research_reporting(ds)
    with tempfile.TemporaryDirectory(prefix='xgb-publication-') as tmp:
        for stem in ('report','technical_supplement'):
            for _ in range(3):
                result=subprocess.run([compiler,'-halt-on-error','-interaction=nonstopmode',
                    '-file-line-error','-output-directory',tmp,stem+'.tex'],cwd=ROOT,
                    text=True,capture_output=True,encoding='utf-8',errors='replace')
                if result.returncode:
                    raise SystemExit(result.stdout[-10000:]+result.stderr[-1000:])
            log=(Path(tmp)/(stem+'.log')).read_text(errors='replace')
            if 'undefined references' in log or 'undefined citations' in log:
                raise SystemExit('Unresolved references/citations after compilation')
            (ROOT/'output').mkdir(exist_ok=True)
            (ROOT/('output/'+stem+'_build.log')).write_text(log,encoding='utf-8')
            shutil.copy2(Path(tmp)/(stem+'.pdf'),ROOT/(stem+'.pdf'))
    record=build_record(ROOT)
    (ROOT/'docs/generated/publication_build.json').write_text(json.dumps(record,indent=2)+'\n')
    write_research_reporting(ds)  # Update PDF-derived page records after the actual build.
    ok,message=check_research_reporting(ds)
    if not ok:
        raise SystemExit(message)
    print(f"Compiled and source-bound publication: {record['pages']} pages")


if __name__ == '__main__':
    main()
