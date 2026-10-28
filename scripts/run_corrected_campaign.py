"""Run the approved corrected IMDb campaign without overwriting previous evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/corrected_cuda_campaign_v1'
FINDINGS = ROOT / 'findings/corrected_cuda_campaign_v1'
CALIBRATION = ROOT / 'results/cuda_fixed_calibration'
sys.path.insert(0, str(ROOT / 'phase_b_sequential_xai'))
from xgb_doe_xai.study import check_study, source_hashes
from xgb_doe_xai.word_study import check_word_study


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    OUTPUT.mkdir(exist_ok=False)
    config = json.loads((CALIBRATION / 'imdb_config.json').read_text())
    meta = json.loads((CALIBRATION / 'imdb_calibration.json').read_text())
    assert config['n_trees'] == 600 and config['device'] == 'cuda'
    dataset = Path(meta['dataset_path'])
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == meta['dataset_sha256']
    pilot = json.loads((ROOT / 'results/cuda_fix_verification_v2/completion.json').read_text())
    assert pilot['status'] == 'PASSED'
    hashes = source_hashes()
    write(OUTPUT / 'launch_contract.json', dict(
        authorized='User: lets go with the corrected campaign',
        declared_at_utc=datetime.now(timezone.utc).isoformat(), config=config,
        source_hashes=hashes, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dataset_sha256=meta['dataset_sha256'], dataset_path=str(dataset),
        calibration_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CALIBRATION.iterdir() if p.is_file()},
        budget=dict(search_fits=1260, confirmation_fits=540, final_fits=9, word_model_fits=1, total_new_fits=1810),
        test_history='Previously scored in the faulty campaign; corrected replication, not historically untouched test data',
        stopping='Complete frozen budgets; preserve failures; no test-driven tuning or automatic extra campaigns',
        runtime_estimate='Calibration extrapolation about 6-7 hours; depth variation and overhead may materially change runtime',
        output=str(OUTPUT), findings=str(FINDINGS)))
    started = time.perf_counter()
    runner = ROOT / 'phase_b_sequential_xai/run_phase_b.py'
    base = [sys.executable, '-u', str(runner)]
    write(OUTPUT / 'campaign_status.json', dict(status='RUNNING', started_at_utc=datetime.now(timezone.utc).isoformat()))
    # Keep the machine awake while this explicitly authorized long run is active;
    # restore the default execution state in all exit paths.
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        subprocess.run(base + ['--stage','run','--dataset','full_imdb',
            '--data-dir',str(ROOT/'phase_b_sequential_xai/data'),
            '--calibration-dir',str(CALIBRATION),'--output-dir',str(OUTPUT),
            '--device','cuda','--training-threads','4'], cwd=ROOT, check=True)
        assert source_hashes() == hashes, 'Source changed during campaign'
        summary = check_study(OUTPUT/'imdb')
        check_word_study(OUTPUT/'imdb_words')
        assert summary['total_fits'] == 1809 and summary['final_fits'] == 9
        subprocess.run(base + ['--stage','generate-findings','--dataset','full_imdb',
            '--output-dir',str(OUTPUT),'--findings-dir',str(FINDINGS)], cwd=ROOT, check=True)
        write(OUTPUT/'campaign_status.json',dict(status='COMPLETED_AND_VERIFIED',
            completed_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
            study_fits=1809,word_model_fits=1,findings=str(FINDINGS)))
    except BaseException as exc:
        write(OUTPUT/'campaign_status.json',dict(status='FAILED',error=repr(exc),
            recorded_at_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started))
        raise
    finally:
        if sys.platform == 'win32':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
