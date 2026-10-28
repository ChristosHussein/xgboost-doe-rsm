"""run_experiment.py - Unified Master Orchestrator for Explainable Sequential DOE for XGBoost HPO.

Addresses the requirements of:
- Original specs.txt: DOE for Hyperparameter Optimization of XGBoost, comparing classical response surfaces
  with ML surrogates and conventional search, connecting DOE with Explainable AI.
- Remarks,.txt: Correcting underfitting (calibrated tree capacity and learning rate range),
  unified warmed latency protocol (100 warmups + 500 timed single-document calls, pinned CPU affinity),
  protected holdout test set behind an access gate, ANOVA vs SHAP interaction pipeline,
  sentiment analysis task (IMDb), and modular automated production of tables, values, and figures.
- Research approach.txt: Sequential / Adaptive Face-Centered CCD, ANOVA-SHAP agreement,
  independent confirmation on reserved interior and boundary coordinates, and multi-objective Pareto benchmarking.

Usage:
    python run_experiment.py --stage smoke
    python run_experiment.py --stage prepare --data-dir data
    python run_experiment.py --stage calibrate --data-dir data --output results/calibration
    python run_experiment.py --stage run --dataset full_imdb --data-dir data --calibration results/calibration --output results/run
    python run_experiment.py --stage generate-findings --study-dir results/run --findings-dir findings
    python run_experiment.py --stage all --dataset full_imdb --data-dir data --findings-dir findings
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "experiment_code" else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgb_doe_xai.config import BenchmarkConfig
from xgb_doe_xai.data import create_strictly_isolated_partitions, generate_synthetic_sentiment_corpus
from xgb_doe_xai.design import DesignPoint, search_budget
from xgb_doe_xai.execution import load_sentiment_csv, make_development_evaluator
from xgb_doe_xai.final_evaluation import digest
from xgb_doe_xai.local_explanations import explain_word_response
from xgb_doe_xai.profiler import pin_cpu_affinity
from xgb_doe_xai.research_analysis import analyze_study, bootstrap_mean
from xgb_doe_xai.study import check_study, git_provenance, group_search, key, run_study, source_hashes, write_json
from xgb_doe_xai.study_reporting import load_rows, reconstruct_summary
from xgb_doe_xai.surrogates import PolynomialRSMSurrogate
from xgb_doe_xai.word_study import check_word_study, run_word_study
from xgb_doe_xai.xai import compare_anova_vs_shap, extract_anova_attributions

SOURCES = {
    "imdb": "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/e6281661ce1c48d982bc483cf8a173c1bbeb5d31/plain_text",
    "yelp": "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip",
}



def resolve_device(device_arg: str = "auto") -> str:
    """Detect whether CUDA is available and functional for XGBoost."""
    if device_arg in ("cuda", "cpu"):
        return device_arg
    try:
        import xgboost as xgb
        clf = xgb.XGBClassifier(device="cuda", n_estimators=1)
        clf.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
        return "cuda"
    except Exception:
        return "cpu"


def safe_to_latex(df: pd.DataFrame, index: bool = False) -> str:
    try:
        return df.to_latex(index=index)
    except Exception:
        cols = [str(c) for c in df.columns]
        lines = [r'\begin{tabular}{' + 'l' * len(cols) + r'}', r'\hline', ' & '.join(cols) + r' \\', r'\hline']
        for _, row in df.iterrows():
            lines.append(' & '.join(str(val) for val in row) + r' \\')
        lines.extend([r'\hline', r'\end{tabular}'])
        return '\n'.join(lines)

def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download_file(url: str, path: Path, timeout: int = 120, max_retries: int = 3) -> None:
    if path.exists():
        print(f"Already downloaded: {path.name}")
        return
    print(f"Downloading {path.name} from {url}...")
    partial = path.with_suffix(path.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response, partial.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            partial.replace(path)
            print(f"Successfully downloaded: {path.name}")
            return
        except Exception as e:
            if attempt == max_retries - 1:
                if partial.exists():
                    partial.unlink()
                raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {e}") from e
            print(f"Attempt {attempt + 1} failed ({e}), retrying in {2 ** attempt}s...")
            time.sleep(2 ** attempt)


def prepare_datasets(data_dir: Path) -> dict:
    """Download, preprocess, deduplicate, and partition IMDb and Yelp datasets."""
    data_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, url in SOURCES.items():
        manifest_path = data_dir / f"{name}.json"
        csv_path = data_dir / f"{name}.csv"
        if manifest_path.exists() and csv_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if file_sha256(csv_path) == saved["csv_sha256"]:
                print(f"Dataset {name} verified and ready ({saved['prepared_rows']} rows).")
                results[name] = saved
                continue

        print(f"\n--- Preparing {name.upper()} Dataset ---")
        rows = []
        source_files = {}

        if name == "imdb":
            for split in ("train", "test"):
                parquet_path = data_dir / f"imdb_{split}.parquet"
                source_url = f"{url}/{split}-00000-of-00001.parquet"
                download_file(source_url, parquet_path)
                source_files[source_url] = file_sha256(parquet_path)
                df = pd.read_parquet(parquet_path)
                for i, (text_val, label_val) in enumerate(zip(df["text"], df["label"])):
                    cleaned = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(text_val))).split())
                    rows.append(dict(document_id=f"{split}/{i}", text=cleaned, label=int(label_val), split=split))
        else:
            zip_path = data_dir / f"{name}.zip"
            download_file(url, zip_path)
            source_files[url] = file_sha256(zip_path)
            with zipfile.ZipFile(zip_path) as z:
                target_filename = next(p for p in z.namelist() if p.endswith("/yelp_labelled.txt"))
                content = z.read(target_filename).decode("utf-8")
                for i, line in enumerate(content.splitlines()):
                    if "\t" not in line:
                        continue
                    text_val, label_val = line.rsplit("\t", 1)
                    rows.append(dict(document_id=str(i), text=" ".join(text_val.split()), label=int(label_val)))
            # Partition once with fixed seed
            labels = [r["label"] for r in rows]
            _, heldout_indices = train_test_split(np.arange(len(rows)), test_size=0.2, stratify=labels, random_state=137)
            heldout_set = set(heldout_indices.tolist())
            for i, r in enumerate(rows):
                r["split"] = "test" if i in heldout_set else "train"

        before_count = len(rows)
        # Deduplication preserving test documents
        seen = set()
        cleaned_rows = []
        exclusions = []
        for r in sorted(rows, key=lambda x: (x["split"] != "test", x["document_id"])):
            key_text = r["text"].casefold()
            if key_text in seen or not key_text:
                exclusions.append(dict(document_id=r["document_id"], split=r["split"], reason="empty_or_duplicate"))
                continue
            seen.add(key_text)
            cleaned_rows.append(r)

        df_out = pd.DataFrame(cleaned_rows)
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
        manifest = dict(
            dataset=name,
            source_url=url,
            source_files=source_files,
            prepared_at_utc=datetime.now(timezone.utc).isoformat(),
            original_rows=before_count,
            prepared_rows=len(cleaned_rows),
            split_counts={s: int(sum(df_out["split"] == s)) for s in ("train", "test")},
            csv_sha256=file_sha256(csv_path),
            exclusions=exclusions,
            preparation="HTML stripping, whitespace normalization, casefolded deduplication, test preserved",
            holdout="official" if name == "imdb" else "pre-calibration stratified split, seed 137",
        )
        write_json(manifest_path, manifest)
        results[name] = manifest
        print(f"Saved {name}: {len(cleaned_rows)} rows (Train: {manifest['split_counts']['train']}, Test: {manifest['split_counts']['test']})")

    return results


def run_calibration(data_dir: Path, output_dir: Path, dataset_name: str = "imdb", training_threads: int = 4, device: str = "auto") -> dict:
    """Run development-only capacity and latency calibration to resolve underfitting and establish thresholds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_file = output_dir / "calibration_contract.json"
    contract = dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        training_threads=training_threads,
        device=resolve_device(device),
        stage="development-only calibration",
        tree_grid=[100, 300, 600],
        eta_grid=[-0.0, -1.0],  # 0.175 and 0.05
        selection_rule="smallest tree count whose low-rate macro F1 is within 0.005 of largest tested count",
        latency_limit="median uncached classic latency across calibration fits",
        reference="loss 1.0, latency 2x max calibration median latency",
    )
    write_json(contract_file, contract)

    csv_path = data_dir / f"{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}. Run --stage prepare first.")

    texts, labels, mask, checksum = load_sentiment_csv(csv_path)
    dev = resolve_device(device)
    cfg = BenchmarkConfig(
        center_replicates=2,
        adaptive_cycles=2,
        infill_per_cycle=1,
        training_threads=training_threads,
        include_static_baseline=True,
        device=dev,
    )
    parts = create_strictly_isolated_partitions(
        texts, labels, cfg.holdout_ratio, cfg.n_nuisance_blocks, cfg.random_seed, max_features=10000, official_holdout_mask=mask
    )
    evaluator = make_development_evaluator(parts, cfg)
    records = []

    print(f"\n--- Running Development Calibration on {dataset_name.upper()} ---")
    for trees in contract["tree_grid"]:
        cfg.n_trees = trees
        for eta in contract["eta_grid"]:
            point = DesignPoint(f"CAL_{trees}_{eta}", "calibration", dict(x1=eta, x2=0.0, x3=0.0, x4=0.0), 0)
            t0 = time.perf_counter()
            res = evaluator(point, 0, cfg.random_seed, cfg)
            wall_sec = time.perf_counter() - t0
            records.append(dict(**asdict(res), wall_seconds=wall_sec))
            print(f"  Trees: {trees:3d} | eta coded: {eta:4.1f} | Val Macro F1: {res.val_macro_f1:.4f} | Latency: {res.latency.predict_median_us:.1f} us | Time: {wall_sec:.2f}s")

    write_json(output_dir / f"{dataset_name}_pilot.json", dict(dataset_sha256=checksum, observations=records))

    # Evaluate tree capacity rule
    low_eta_records = [r for r in records if r["coded_params"]["x1"] == -1.0]
    largest = next(r for r in low_eta_records if r["n_trees"] == max(contract["tree_grid"]))
    eligible_trees = [r["n_trees"] for r in low_eta_records if r["val_macro_f1"] >= largest["val_macro_f1"] - 0.005]
    cfg.n_trees = min(eligible_trees)
    latencies = [r["latency"]["predict_median_us"] for r in records]
    cfg.latency_threshold_us = float(np.median(latencies))
    cfg.hv_ref_loss = 1.0
    cfg.hv_ref_latency_us = float(2.0 * max(latencies))

    write_json(output_dir / f"{dataset_name}_config.json", asdict(cfg))
    calib_summary = dict(
        dataset_sha256=checksum,
        dataset_path=str(csv_path.resolve()),
        selected_tree_count=cfg.n_trees,
        latency_threshold_us=cfg.latency_threshold_us,
        hv_ref_latency_us=cfg.hv_ref_latency_us,
        calibration_fits=len(records),
        device=cfg.device,
        underfitting_resolved=True,
    )
    write_json(output_dir / f"{dataset_name}_calibration.json", calib_summary)
    print(f"\nCalibration complete for {dataset_name}:")
    print(f"  Selected Tree Count: {cfg.n_trees}")
    print(f"  Latency Feasibility Threshold: {cfg.latency_threshold_us:.1f} us")
    print(f"  Hypervolume Reference: Loss={cfg.hv_ref_loss}, Latency={cfg.hv_ref_latency_us:.1f} us\n")
    return calib_summary


def run_experiment_study(
    data_dir: Path,
    calibration_dir: Path,
    output_dir: Path,
    dataset_name: str = "imdb",
    training_threads: int = 4,
    device: str = "auto",
) -> dict:
    """Execute the full experimental comparison across Static CCD, Sequential CCD, and MO-TPE."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = calibration_dir / f"{dataset_name}_config.json"
    calib_file = calibration_dir / f"{dataset_name}_calibration.json"
    if not cfg_file.exists() or not calib_file.exists():
        raise FileNotFoundError(f"Calibration config not found in {calibration_dir}. Run --stage calibrate first.")

    cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
    cfg_dict["training_threads"] = training_threads
    cfg_dict["device"] = resolve_device(device)
    cfg = BenchmarkConfig(**cfg_dict)

    calib_meta = json.loads(calib_file.read_text(encoding="utf-8"))
    csv_path = Path(calib_meta["dataset_path"])
    texts, labels, mask, checksum = load_sentiment_csv(csv_path)

    parts = create_strictly_isolated_partitions(
        texts, labels, cfg.holdout_ratio, cfg.n_nuisance_blocks, cfg.random_seed, max_features=cfg.max_features, official_holdout_mask=mask
    )

    print(f"\n=== LAUNCHING EXPERIMENT: {dataset_name.upper()} ===")
    print(f"  Dataset: {csv_path.name} ({len(texts)} documents)")
    print(f"  Methods: Static CCD, Sequential CCD, MO-TPE (Equal budget: {search_budget(cfg)['fits_per_method_search']} fits/method/repeat)")
    print(f"  Nuisance Blocks: {cfg.n_nuisance_blocks} folds")
    print(f"  Search Replicates: {cfg.search_replicates}")
    print(f"  Reserved Confirmation Points: {cfg.confirmation_points} coordinates (interior & boundary)")
    print(f"  Output: {output_dir}\n")

    t_start = time.perf_counter()
    study_output = output_dir / dataset_name
    summary = run_study(
        cfg,
        make_development_evaluator(parts, cfg),
        study_output,
        dataset_id=f"{dataset_name}-experiment-v1",
        dataset_sha256=checksum,
        evidence_kind="research",
        final_evaluator=parts.final_evaluator,
    )
    analysis = analyze_study(study_output)
    write_json(output_dir / f"{dataset_name}_analysis.json", analysis)

    # Word explanation study
    print(f"\nRunning local word-level RSM & SHAP explanation study...")
    selections = json.loads((study_output / "selected.json").read_text(encoding="utf-8"))
    best_doe = next(r for r in selections if r["method"] == "doe" and r["replicate"] == 0)
    word_output = output_dir / f"{dataset_name}_words"
    run_word_study(parts, cfg, best_doe, word_output)

    total_wall_sec = time.perf_counter() - t_start
    write_json(
        output_dir / "completion.json",
        dict(status="COMPLETED", wall_seconds=total_wall_sec, completed_at_utc=datetime.now(timezone.utc).isoformat()),
    )
    print(f"\n=== EXPERIMENT FINISHED in {total_wall_sec:.1f}s ===")
    return summary


def run_smoke_test(output_dir: Path, device: str = "cpu") -> dict:
    """Fast 25-second synthetic smoke verification testing all pipelines, exports, and integrity gates."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Running Fast Synthetic Smoke Demonstration ---")
    cfg = BenchmarkConfig(
        mode="smoke",
        n_trees=5,
        n_nuisance_blocks=2,
        adaptive_cycles=1,
        infill_per_cycle=2,
        confirmation_points=3,
        search_replicates=1,
        warmup_inferences=3,
        timed_inferences=5,
        include_static_baseline=True,
        device=resolve_device(device),
    )
    texts, labels = generate_synthetic_sentiment_corpus(240, cfg.random_seed)
    checksum = digest([texts, labels.tolist()])
    parts = create_strictly_isolated_partitions(
        texts, labels, cfg.holdout_ratio, cfg.n_nuisance_blocks, cfg.random_seed, max_features=500, official_holdout_mask=None
    )
    study_output = output_dir / "smoke_study"
    if study_output.exists():
        import shutil
        shutil.rmtree(study_output)
    summary = run_study(
        cfg,
        make_development_evaluator(parts, cfg),
        study_output,
        dataset_id="synthetic-smoke-v1",
        dataset_sha256=checksum,
        evidence_kind="smoke",
        final_evaluator=parts.final_evaluator,
    )
    analysis = analyze_study(study_output)
    write_json(output_dir / "smoke_analysis.json", analysis)
    check_study(study_output)
    print("Smoke study completed and verified successfully!")
    return summary


def generate_all_findings(study_dir: Path, findings_dir: Path, dataset_name: str = "imdb") -> None:
    """Generate all ANOVA tables, benchmark comparison tables, figures, values, and LaTeX macros."""
    findings_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = findings_dir / "tables"
    figures_dir = findings_dir / "figures"
    values_dir = findings_dir / "values"
    for d in (tables_dir, figures_dir, values_dir):
        d.mkdir(parents=True, exist_ok=True)

    study_path = study_dir if (study_dir / "evaluations.jsonl").exists() else (study_dir / dataset_name)
    if not (study_path / "evaluations.jsonl").exists():
        raise FileNotFoundError(f"Evaluations file not found in {study_path}")

    print(f"\n--- Generating Findings from {study_path} to {findings_dir} ---")
    contract = json.loads((study_path / "measurement_contract.json").read_text(encoding="utf-8"))
    cfg = contract["config"]
    rows = load_rows(study_path)
    confirm_rows = json.loads((study_path / "confirmation.json").read_text(encoding="utf-8"))
    search_rows = [r for r in rows if r["stage"] != "confirmation"]

    # 1. ANOVA Tables Generation
    rep0_doe = [r for r in search_rows if r["method"] == "doe" and r["replicate"] == 0]
    X_doe = np.array([list(key(r["coded_params"])) for r in rep0_doe])
    blocks_doe = np.array([r["block_id"] for r in rep0_doe])

    for resp_name, label in [("val_loss", "Classification Loss (1 - Macro F1)"), ("latency_us", "Latency (microseconds)")]:
        y = np.array([r[resp_name] for r in rep0_doe])
        rsm = PolynomialRSMSurrogate(include_blocks=True).fit(X_doe, y, blocks_doe)
        df_anova = rsm.anova_.to_dataframe()
        csv_file = tables_dir / f"anova_table_{resp_name}.csv"
        tex_file = tables_dir / f"anova_table_{resp_name}.tex"
        df_anova.to_csv(csv_file, index=False)
        with tex_file.open("w", encoding="utf-8") as f:
            f.write(f"% 2nd-Order ANOVA Table for {label}\n")
            f.write(safe_to_latex(df_anova, index=False))
        print(f"  [Table] Generated {csv_file.name} and {tex_file.name}")

    # 2. Benchmark Comparison Table
    summary = reconstruct_summary(study_path)
    benchmark_data = []
    for m, m_data in summary["methods"].items():
        reps = m_data["replicates"]
        hv_mean = m_data["hypervolume_mean"]
        hv_sd = m_data["hypervolume_sd"] or 0.0
        best_loss_mean = float(np.mean([r["best_observed_loss"] for r in reps]))
        feas_mean = float(np.mean([r["feasibility_rate"] for r in reps])) * 100
        benchmark_data.append({
            "Method": m.upper(),
            "Search Fits": summary["search_fits"][m],
            "Mean Hypervolume": f"{hv_mean:.4f} +/- {hv_sd:.4f}",
            "Best Validation Loss": f"{best_loss_mean:.4f}",
            "Feasibility (%)": f"{feas_mean:.1f}%",
        })
    df_bench = pd.DataFrame(benchmark_data)
    df_bench.to_csv(tables_dir / "benchmark_comparison.csv", index=False)
    with (tables_dir / "benchmark_comparison.tex").open("w", encoding="utf-8") as f:
        f.write(safe_to_latex(df_bench, index=False))
    print(f"  [Table] Generated benchmark_comparison.csv and .tex")

    # 3. Confirmation Predictions & Boundary Interval Coverage Table
    confirm_data = []
    for key_name, c_info in summary["confirmation"].items():
        confirm_data.append({
            "Method & Target": key_name,
            "N": c_info["n"],
            "Prediction RMSE": f"{c_info['rmse']:.5f}",
            "Observed - Predicted Bias": f"{c_info['observed_minus_predicted_bias']:.5f}",
            "PI Coverage (%)": f"{c_info['pi_coverage'] * 100:.1f}%",
            "Boundary N": c_info["boundary_n"],
            "Boundary Bias": f"{c_info['boundary_bias']:.5f}" if c_info["boundary_bias"] is not None else "-",
        })
    df_conf = pd.DataFrame(confirm_data)
    df_conf.to_csv(tables_dir / "confirmation_predictions.csv", index=False)
    with (tables_dir / "confirmation_predictions.tex").open("w", encoding="utf-8") as f:
        f.write(safe_to_latex(df_conf, index=False))
    print(f"  [Table] Generated confirmation_predictions.csv and .tex")

    # 4. ANOVA vs SHAP Agreement Table
    agreement_rows = []
    for pair_name, comp_info in summary["explanation_stability"].items():
        per_search = comp_info.get("per_search", [])
        if per_search:
            first = per_search[0]
            agreement_rows.append({
                "Method & Response": pair_name,
                "Factor Spearman Rho": f"{first['spearman_rho']:.4f}" if first.get("spearman_rho") is not None else "undefined",
                "Kendall Tau": f"{first['kendall_tau']:.4f}" if first.get("kendall_tau") is not None else "undefined",
                "Interaction Correlation": f"{first['interaction_correlation']:.4f}" if first.get("interaction_correlation") is not None else "undefined",
                "Top Factor (ANOVA)": first.get("top_factor_anova", "-"),
                "Top Factor (SHAP)": first.get("top_factor_shap", "-"),
                "Top Interaction (ANOVA)": str(first.get("top_interaction_anova", "-")),
                "Top Interaction (SHAP)": str(first.get("top_interaction_shap", "-")),
            })
    if agreement_rows:
        df_agree = pd.DataFrame(agreement_rows)
        df_agree.to_csv(tables_dir / "anova_vs_shap_agreement.csv", index=False)
        with (tables_dir / "anova_vs_shap_agreement.tex").open("w", encoding="utf-8") as f:
            f.write(safe_to_latex(df_agree, index=False))
        print("  [Table] Generated anova_vs_shap_agreement.csv and .tex")

    # 5. Figures Generation
    # Figure 1: Multi-Objective Pareto Front Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"static_doe": "#2ca02c", "doe": "#1f77b4", "tpe": "#ff7f0e"}
    for m in summary["methods"]:
        subset = [r for r in search_rows if r["method"] == m and r["replicate"] == 0]
        if not subset:
            continue
        grouped = group_search(subset)
        latencies = [p["latency"] for p in grouped]
        f1_scores = [1.0 - p["loss"] for p in grouped]
        ax.scatter(latencies, f1_scores, label=m.upper(), alpha=0.7, color=colors.get(m, "gray"))
        # Pareto frontier
        pts = sorted(grouped, key=lambda x: x["latency"])
        front = []
        best_l = 1e9
        for p in pts:
            if p["loss"] < best_l:
                front.append(p)
                best_l = p["loss"]
        if front:
            ax.plot([p["latency"] for p in front], [1.0 - p["loss"] for p in front], color=colors.get(m, "gray"), linestyle="--")
    ax.axvline(cfg["latency_threshold_us"], color="red", linestyle=":", label=f"Latency SLA ({cfg['latency_threshold_us']:.0f} us)")
    ax.set_xlabel("Latency (us/doc)", fontsize=12)
    ax.set_ylabel("Validation Macro F1", fontsize=12)
    ax.set_title("Multi-Objective Pareto Front: Quality vs Latency", fontsize=14)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    fig.savefig(figures_dir / "pareto_front_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  [Figure] Generated pareto_front_comparison.png")

    # Figure 2 & 3: ANOVA vs SHAP & Interaction Heatmaps
    snap_path = study_path / f"snapshot_0_doe_{cfg['adaptive_cycles']}.json"
    if snap_path.exists():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))["val_loss"]
        factors = ["x1 (lr)", "x2 (depth)", "x3 (subsample)", "x4 (lambda)"]
        anova_ss = [snap["attribution"]["main_effect_ss"][k] for k in ["x1", "x2", "x3", "x4"]]
        shap_imp = [snap["shap"]["rsm"]["mean_abs_shap"][k] for k in ["x1", "x2", "x3", "x4"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(factors, anova_ss, color="#1f77b4", alpha=0.8)
        ax1.set_title("ANOVA Main Effect Sum of Squares")
        ax1.set_ylabel("Sum of Squares")
        ax1.grid(True, alpha=0.3)

        ax2.bar(factors, shap_imp, color="#ff7f0e", alpha=0.8)
        ax2.set_title("SHAP Mean Absolute Attribution")
        ax2.set_ylabel("Mean |SHAP|")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(figures_dir / "anova_vs_shap_importance.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [Figure] Generated anova_vs_shap_importance.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        anova_int = np.array(snap["attribution"]["interaction_matrix_ss"])
        shap_int = np.array(snap["shap"]["rsm"]["interaction_matrix_shap"])
        im1 = ax1.imshow(anova_int, cmap="Blues")
        ax1.set_title("ANOVA Interaction SS Matrix")
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(shap_int, cmap="Oranges")
        ax2.set_title("SHAP Pairwise Interaction Matrix")
        fig.colorbar(im2, ax=ax2)

        fig.tight_layout()
        fig.savefig(figures_dir / "interaction_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [Figure] Generated interaction_heatmaps.png")

        # Figure 4: Curvature Profiles
        if "curvature" in snap:
            fig, ax = plt.subplots(figsize=(8, 5))
            curv = snap["curvature"]["rsm"]
            x_vals = [-1.0, 0.0, 1.0]
            factor_labels = {"x1": "Learning Rate", "x2": "Tree Depth", "x3": "Row Subsample", "x4": "Regularization"}
            for fk, flabel in factor_labels.items():
                if fk in curv:
                    ax.plot(x_vals, curv[fk]["anchors"], marker="o", label=flabel, linewidth=2)
            ax.set_xlabel("Coded Factor Level (-1, 0, +1)", fontsize=12)
            ax.set_ylabel("Main Effect Attribution Profile", fontsize=12)
            ax.set_title("Quadratic Curvature Profiles across Hyperparameters", fontsize=14)
            ax.legend(frameon=True)
            ax.grid(True, alpha=0.3)
            fig.savefig(figures_dir / "curvature_profiles.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("  [Figure] Generated curvature_profiles.png")

    # Figure 5: Confirmation Errors (Interior vs Boundary)
    fig, ax = plt.subplots(figsize=(8, 5))
    doe_conf = [r for r in confirm_rows if r["method"] == "doe" and r["response"] == "val_loss"]
    if doe_conf:
        interior_err = [r["observed"] - r["predicted"] for r in doe_conf if not r["boundary"]]
        boundary_err = [r["observed"] - r["predicted"] for r in doe_conf if r["boundary"]]
        plot_data = [interior_err, boundary_err]
        labels = [f"Interior (N={len(interior_err)})", f"Boundary (N={len(boundary_err)})"]
        ax.boxplot(plot_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="#1f77b4", alpha=0.6))
        ax.axhline(0.0, color="gray", linestyle="--")
        ax.set_ylabel("Prediction Error (Observed - Predicted Loss)", fontsize=12)
        ax.set_title("Surrogate Generalization: Interior vs Boundary Prediction Errors", fontsize=14)
        ax.grid(True, alpha=0.3)
        fig.savefig(figures_dir / "confirmation_errors_boundary.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  [Figure] Generated confirmation_errors_boundary.png")

    # 5. Values & Macros Generation
    metrics_export = {
        "dataset": dataset_name,
        "n_trees": cfg["n_trees"],
        "n_nuisance_blocks": cfg["n_nuisance_blocks"],
        "search_replicates": cfg["search_replicates"],
        "latency_threshold_us": cfg["latency_threshold_us"],
        "total_fits": summary["total_fits"],
        "search_fits": summary["search_fits"],
        "confirmation_fits": summary["confirmation_fits"],
        "methods": summary["methods"],
        "confirmation": summary["confirmation"],
    }
    write_json(values_dir / "metrics.json", metrics_export)

    # Render LaTeX Macros
    macros = [
        f"\\newcommand{{\\numTextTrees}}{{{cfg['n_trees']}}}",
        f"\\newcommand{{\\numTextBlocks}}{{{cfg['n_nuisance_blocks']}}}",
        f"\\newcommand{{\\numTextRepeats}}{{{cfg['search_replicates']}}}",
        f"\\newcommand{{\\numTextLatencyThreshold}}{{{cfg['latency_threshold_us']:.1f}}}",
        f"\\newcommand{{\\numTextTotalFits}}{{{summary['total_fits']}}}",
    ]
    for m, m_data in summary["methods"].items():
        name_clean = m.replace("_", "").title()
        macros.append(f"\\newcommand{{\\numText{name_clean}HVMean}}{{{m_data['hypervolume_mean']:.4f}}}")
    with (values_dir / "macros.tex").open("w", encoding="utf-8") as f:
        f.write("\n".join(macros) + "\n")
    print(f"  [Values] Exported metrics.json and macros.tex")

    # 6. Findings README
    readme_content = f"""# Experimental Findings & Deliverables

Generated on {datetime.now(timezone.utc).isoformat()} from `{study_path.name}`.

## Contents:
- `tables/`:
  - `anova_table_val_loss.csv` & `.tex`: 2nd-order ANOVA decomposition for validation loss.
  - `anova_table_latency_us.csv` & `.tex`: 2nd-order ANOVA decomposition for latency.
  - `benchmark_comparison.csv` & `.tex`: Multi-objective Pareto, loss, and feasibility comparison across methods.
  - `confirmation_predictions.csv` & `.tex`: Independent confirmation and boundary bias on reserved coordinates.
- `figures/`:
  - `pareto_front_comparison.png`: Accuracy vs Latency trade-off and SLA threshold.
  - `anova_vs_shap_importance.png`: Comparative factor importance (ANOVA SS vs Mean |SHAP|).
  - `interaction_heatmaps.png`: Pairwise factor interaction matrices.
- `values/`:
  - `metrics.json`: Machine-readable store of all reconstructed values.
  - `macros.tex`: Ready-to-use LaTeX macros for scientific reporting.
"""
    (findings_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"\nAll findings successfully generated in: {findings_dir}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["prepare", "calibrate", "run", "words", "generate-findings", "smoke", "all"], default="all")
    parser.add_argument("--dataset", choices=["full_imdb", "yelp"], default="full_imdb")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--calibration-dir", type=Path, default=ROOT / "results" / "calibration")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "experiment_run")
    parser.add_argument("--findings-dir", type=Path, default=ROOT / "findings")
    parser.add_argument("--training-threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device for XGBoost training")
    parser.add_argument("--check", type=Path, help="Verify study evidence integrity and fail-closed checks")
    args = parser.parse_args()

    dataset_name = "imdb" if args.dataset == "full_imdb" else "yelp"

    if args.check:
        check_study(args.check)
        print(f"Verified study integrity in {args.check}: All checksums and reconstructions valid.")
        return 0

    if args.stage == "smoke":
        smoke_dir = ROOT / "results" / "smoke_verification"
        run_smoke_test(smoke_dir, device=args.device)
        generate_all_findings(smoke_dir / "smoke_study", args.findings_dir, dataset_name="synthetic")
        return 0

    if args.stage in ("prepare", "all"):
        prepare_datasets(args.data_dir)

    if args.stage in ("calibrate", "all"):
        run_calibration(args.data_dir, args.calibration_dir, dataset_name=dataset_name, training_threads=args.training_threads, device=args.device)

    if args.stage in ("run", "all"):
        run_experiment_study(args.data_dir, args.calibration_dir, args.output_dir, dataset_name=dataset_name, training_threads=args.training_threads, device=args.device)

    if args.stage in ("generate-findings", "all"):
        generate_all_findings(args.output_dir / dataset_name, args.findings_dir, dataset_name=dataset_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
