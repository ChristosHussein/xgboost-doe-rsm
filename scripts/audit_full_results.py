import json
from pathlib import Path
import pandas as pd
import numpy as np
import ast

run_dir = Path("results/revision_v2/full_run_001")

print("================================================================================")
print("             SCIENTIFIC AUDIT & RESULTS: REVISION V2 FULL RUN 001               ")
print("================================================================================")

with open(run_dir / "run_manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)
print(f"Mode: {manifest.get('mode')}")
print(f"Inference status: {manifest.get('inference_status')}")
print(f"Git revision: {manifest.get('git_revision')}")
print(f"Created at: {manifest.get('created_at_utc')}")
print(f"Artifacts generated: {len(manifest.get('artifacts', []))}")

df = pd.read_csv(run_dir / "final_summary.csv")
for col in ["validation_rmse", "test_rmse"]:
    df[col + "_mean"] = df[col].apply(lambda x: ast.literal_eval(x)["mean"])
    df[col + "_sd"] = df[col].apply(lambda x: ast.literal_eval(x)["standard_deviation"])
    df[col + "_median"] = df[col].apply(lambda x: ast.literal_eval(x)["median"])

print("\n" + "="*80)
print("1. SUMMARY OF FINAL TEST EVALUATIONS & RETRAININGS ACROSS ALL METHODS (N=122)")
print("="*80)

summary_table = []
for opt, group in df.groupby("optimizer"):
    summary_table.append({
        "Optimizer": opt,
        "N_Configs": len(group),
        "Val RMSE Mean": group["validation_rmse_mean"].mean(),
        "Val RMSE SD (between-search)": group["validation_rmse_mean"].std() if len(group) > 1 else 0.0,
        "Test RMSE Mean": group["test_rmse_mean"].mean(),
        "Test RMSE SD (between-search)": group["test_rmse_mean"].std() if len(group) > 1 else 0.0,
        "Latency Mean (us)": group["predict_latency_us"].mean(),
        "Latency SD (us)": group["predict_latency_us"].std() if len(group) > 1 else 0.0,
        "Inplace Lat Mean (us)": group["inplace_predict_latency_us"].mean(),
        "Feasible (<=145us)": f"{group['benchmark_time_feasible'].sum()}/{len(group)}",
        "Optimism (Val - Search)": group["selection_optimism_independent_minus_search_rmse"].mean()
    })

sum_df = pd.DataFrame(summary_table).set_index("Optimizer")
print(sum_df.to_string())

print("\n" + "="*80)
print("2. MULTI-OBJECTIVE PARETO HYPERVOLUME COMPARISONS")
print("="*80)

with open(run_dir / "hypervolume.json", encoding="utf-8") as f:
    hv_data = json.load(f)

for domain in ["development_domain", "external_test_domain"]:
    print(f"\n--- Domain: {domain} ---")
    ddata = hv_data.get(domain, {})
    for ref_key, rdata in ddata.get("reference_points", {}).items():
        print(f"\n  [Reference Point: {ref_key}]")
        for k, v in rdata.items():
            if k in ["mo_tpe_by_replicate", "repeated_doe_by_replicate"]:
                vals = [item["value"] for item in v.values()]
                print(f"    {k}: N={len(vals)}, Mean={np.mean(vals):.4f}, SD={np.std(vals):.4f}, Range=[{np.min(vals):.4f}, {np.max(vals):.4f}]")
            elif isinstance(v, dict) and "mean" in v:
                print(f"    {k}: Mean={v['mean']:.4f}, SD={v.get('standard_deviation', 0):.4f}, Median={v.get('median', 0):.4f}")
            elif isinstance(v, dict) and "value" in v:
                pts = len(v.get("pareto", {}).get("points", []))
                print(f"    {k}: Value={v['value']:.4f} ({pts} non-dominated points)")
            elif isinstance(v, dict) and "hypervolume" in v:
                pts = v.get("point_count", len(v.get("pareto_points", [])))
                print(f"    {k}: Hypervolume={v['hypervolume']:.4f} ({pts} points)")

print("\n" + "="*80)
print("3. PAIRED STATISTICAL COMPARISONS")
print("="*80)

with open(run_dir / "paired_comparisons.json", encoding="utf-8") as f:
    paired = json.load(f)

paired_df = pd.DataFrame(paired)
for comp, group in paired_df.groupby("comparison"):
    print(f"\n-- {comp} --")
    first_item = group.iloc[0]
    classification = first_item.get("comparison_classification")
    print(f"   Classification: {classification}")
    
    if classification == "historical_fixed_selection_conditional_descriptive":
        for metric in ["val_rmse", "test_rmse"]:
            mgroup = group[group["metric"] == metric]
            diffs = [row["conditional_difference"]["mean_difference"] for _, row in mgroup.iterrows()]
            print(f"   Metric {metric}: N={len(diffs)} pairs")
            print(f"     Mean difference (Anchor - Comparator): {np.mean(diffs):+.5f} (SD={np.std(diffs):.5f})")
            print(f"     Range: [{np.min(diffs):+.5f}, {np.max(diffs):+.5f}]")
            first_row = mgroup.iloc[0]
            cd = first_row.get("conditional_difference", {})
            print(f"     Sample pair (rep 0): mean diff={cd.get('mean_difference', 0):+.5f}, median diff={cd.get('median_difference', 0):+.5f}, SD={cd.get('standard_deviation', 0):.5f}")
            print(f"     Inference performed: {cd.get('inference_performed')} (Reason: {cd.get('reason', 'N/A')[:60]}...)")
    elif classification == "new_end_to_end_selection_replicate_comparison":
        for metric in ["val_rmse", "test_rmse"]:
            mgroup = group[group["metric"] == metric]
            row = mgroup.iloc[0]
            pd_res = row.get("paired_difference", {})
            equiv = row.get("equivalence", {})
            print(f"   Metric {metric} (N={len(row.get('included_selection_replicate_ids', []))} matched replicates):")
            print(f"     Mean paired diff: {pd_res.get('mean_difference', 0):+.5f} (SD={pd_res.get('standard_deviation', 0):.5f})")
            print(f"     Paired t-test: t={pd_res.get('t_statistic', 0):.3f}, p={pd_res.get('p_value', 0):.4e}, 95% CI=[{pd_res.get('confidence_interval_95', [0,0])[0]:.5f}, {pd_res.get('confidence_interval_95', [0,0])[1]:.5f}]")
            print(f"     TOST Equivalence (margin={equiv.get('margin')}): equivalent={equiv.get('equivalent')} (p1={equiv.get('p_value_greater', 0):.4e}, p2={equiv.get('p_value_less', 0):.4e})")

print("\n" + "="*80)
print("4. LATENCY MEASUREMENT VALIDATION & HARDWARE PROFILING")
print("="*80)

with open(run_dir / "latency_measurement.json", encoding="utf-8") as f:
    lat_meas = json.load(f)

first_session = lat_meas.get("sessions", [{}])[0]
metadata = first_session.get("metadata", {})
hardware = metadata.get("hardware", {})
print(f"Platform: {hardware.get('platform')}")
print(f"Processor: {hardware.get('processor')}")
print(f"Affinity Applied: {hardware.get('cpu_affinity_applied')}")
print(f"Number of timing sessions: {len(lat_meas.get('sessions', []))}")

session_summaries = lat_meas.get("summaries", {})
between_session_sds = [row["predict_latency_session_standard_deviation_us"] for row in session_summaries.values()]
print(f"Within-configuration between-session SD: mean={np.mean(between_session_sds):.2f} us, max={np.max(between_session_sds):.2f} us")

with open(run_dir / "latency_interface_overhead.json", encoding="utf-8") as f:
    overhead = json.load(f)
overhead_means = [row["overhead_us"] for row in overhead.get("per_session", [])]
print(f"Interface overhead (predict - inplace_predict): mean={np.mean(overhead_means):.2f} us, SD={np.std(overhead_means):.2f} us")

print("\n" + "="*80)
print("5. HOLDOUT ISOLATION & SELECTION INTEGRITY AUDIT")
print("="*80)

trials_df = pd.read_csv(run_dir / "optimizer_trials.csv")
print(f"Total optimizer trials logged: {len(trials_df):,}")
print(f"Contains 'test_rmse' column? {'test_rmse' in trials_df.columns}")
print(f"Selected trials marked: {trials_df['selected'].sum()}")
infeasible_count = (trials_df["feasibility"] == False).sum()
constrained_trials = trials_df[trials_df["optimizer"] == "constrained_tpe"]
print(f"Constrained TPE trials: {len(constrained_trials):,}")
print(f"  Feasible (<= 145 us): {(constrained_trials['feasibility'] == True).sum():,} ({(constrained_trials['feasibility'] == True).mean()*100:.1f}%)")
print(f"  Infeasible (> 145 us): {infeasible_count:,} ({(infeasible_count/len(constrained_trials))*100:.1f}%)")

with open(run_dir / "finalized_selections.json", encoding="utf-8") as f:
    fin_sel = json.load(f)
print(f"Status of finalized selections: {fin_sel.get('status')}")
print(f"Total configurations frozen: {len(fin_sel.get('configurations', []))}")
all_hashes_present = all(bool(c.get("config_sha256")) for c in fin_sel.get("configurations", []))
print(f"All configurations have SHA-256 hashes? {all_hashes_present}")

final_eval_df = pd.read_csv(run_dir / "final_evaluations.csv")
print(f"Total final evaluation rows: {len(final_eval_df):,}")
print(f"Unique evaluation seeds: {sorted(final_eval_df['evaluation_seed'].unique().tolist())}")
print(f"Failed evaluations count: {(final_eval_df['status'] != 'completed').sum()}")
print(f"Configuration hash mismatches: {(final_eval_df['config_sha256'].isnull()).sum()}")
print("================================================================================")
