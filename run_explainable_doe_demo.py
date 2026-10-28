"""Run a logged smoke demonstration or an explicitly configured sentiment study.

No dataset is downloaded. --check is non-mutating. A smoke run is a software
verification fixture and must never be cited as real-data scientific evidence.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from xgb_doe_xai import BenchmarkConfig, create_strictly_isolated_partitions, generate_synthetic_sentiment_corpus
from xgb_doe_xai.execution import load_sentiment_csv, make_development_evaluator
from xgb_doe_xai.final_evaluation import digest
from xgb_doe_xai.study import run_study, check_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', metavar='RUN_DIRECTORY')
    parser.add_argument('--plan', action='store_true', help='Print the fit budget without loading data or fitting models')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--dataset-csv')
    parser.add_argument('--dataset-id')
    parser.add_argument('--config', help='JSON BenchmarkConfig; immutable copy saved before fitting')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.check:
        check_study(args.check)
        print('Study evidence, reconstructed summary, report and artifact registry verified.')
        return 0
    if args.plan:
        cfg = BenchmarkConfig(**json.loads(Path(args.config).read_text())) if args.config else BenchmarkConfig()
        from xgb_doe_xai.design import search_budget
        coordinates = search_budget(cfg)['total_coordinates']
        methods = 3 if cfg.include_static_baseline else 2
        search = methods*coordinates*cfg.n_nuisance_blocks*cfg.search_replicates
        confirmation = methods*cfg.confirmation_points*cfg.n_nuisance_blocks*cfg.search_replicates
        final = methods*cfg.search_replicates
        print(json.dumps(dict(config=asdict(cfg), planned_search_fits=search,
            planned_confirmation_fits=confirmation, planned_final_fits=final,
            planned_total_fits=search+confirmation+final), indent=2))
        return 0
    if not args.output:
        parser.error('--output is required for a new run')
    if args.smoke:
        if args.dataset_csv or args.config:
            parser.error('--smoke uses its declared synthetic fixture and reduced configuration')
        cfg = BenchmarkConfig(mode='smoke', n_trees=5, n_nuisance_blocks=2,
            adaptive_cycles=1, infill_per_cycle=2, confirmation_points=3, search_replicates=1,
            warmup_inferences=3, timed_inferences=5)
        texts, labels = generate_synthetic_sentiment_corpus(240, cfg.random_seed)
        official, dataset_id = None, 'synthetic-sentiment-smoke'
        dataset_hash = digest([texts, labels.tolist()])
        evidence_kind = 'smoke'
    else:
        if not args.dataset_csv or not args.dataset_id:
            parser.error('Research runs require --dataset-csv and --dataset-id; use --smoke for a fixture')
        cfg = BenchmarkConfig(**json.loads(Path(args.config).read_text())) if args.config else BenchmarkConfig()
        if cfg.mode != 'research':
            parser.error('Real-data research requires mode=research')
        texts, labels, official, dataset_hash = load_sentiment_csv(args.dataset_csv)
        dataset_id, evidence_kind = args.dataset_id, 'research'
    parts = create_strictly_isolated_partitions(texts, labels, cfg.holdout_ratio,
        cfg.n_nuisance_blocks, cfg.random_seed, max_features=cfg.max_features,
        official_holdout_mask=official)
    result = run_study(cfg, make_development_evaluator(parts, cfg), args.output,
        dataset_id=dataset_id, dataset_sha256=dataset_hash, evidence_kind=evidence_kind,
        final_evaluator=parts.final_evaluator)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
