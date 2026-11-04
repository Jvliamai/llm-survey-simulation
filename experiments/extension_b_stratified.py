"""
Extension B subgroup-aware rectification.

    python experiments/extension_b_stratified.py --out-dir outputs/tables/rq2
    python experiments/extension_b_stratified.py --n-trials 100 --only-dataset soep_q1
"""
 
from __future__ import annotations
 
import argparse
import json
import subprocess
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
 
ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"soep_q1": "SOEP Q1 (Oekonomie)", "soep_q2": "SOEP Q2 (Politik)"}
FOCAL_KEY_FOR_DATASET = {"soep_q1": "q1", "soep_q2": "q2"}
METHODS = {"ft": "Domain-FT", "concept_guided": "Persona-guided", "silicon_sampling": "Demo-only"}
TRIAL_NS = [100, 50, 150, 200]
 
# Default labeled-budget sweep for the budget allocation analysis.
DEFAULT_BUDGET_TRIAL_NS = [50, 100, 150, 200]
 
 
def _trial_ns(n_human: int) -> list[int]:
    """TRIAL_NS reordered so ``n_human`` comes first (= headline n)."""
    return [n_human] + [n for n in TRIAL_NS if n != n_human]
 
 
def _results_subdir(n_human: int) -> str:
    """results.json location inside datasets/<ds>/: the default run keeps the
    usual place, a robustness run gets its own (results_n200, ...)."""
    return "outputs/results" if n_human == 100 else f"outputs/results_n{n_human}"
 
 # Run or reuse evaluation
def run_evaluations(datasets, n_trials: int, seed: int, skip: bool,
                     shrinkage: float, min_stratum_n: int, dims: list[str] | None,
                     n_human: int = 100) -> None:
    """Invoke each dataset's evaluate.py with every subgroup dimension swept."""
    for ds in datasets:
        ds_dir = ROOT / "datasets" / ds
        results = ds_dir / _results_subdir(n_human) / "results.json"
        if skip and results.exists():
            print(f"=== {ds}: reusing {results.relative_to(ROOT)} ===")
            continue
        print(f"\n=== {ds}: evaluating (Extension B subgroup sweep) ===")
        cmd = [sys.executable, "evaluate.py",
               "--trial_ns", *[str(n) for n in _trial_ns(n_human)],
               "--n_trials", str(n_trials), "--seed", str(seed),
               "--shrinkage", str(shrinkage), "--min-stratum-n", str(min_stratum_n),
               "--results-dir", _results_subdir(n_human)]
        cmd += (["--subgroup", *dims] if dims else ["--subgroups-all"])
        subprocess.run(cmd, cwd=ds_dir, check=True)
 
 
def _normalize_subgroups(sub: dict) -> dict:
    if "variable" in sub:
        return {sub["variable"]: sub}
    return sub
 
 
def collect_long(datasets, n_human: int = 100) -> pd.DataFrame:
    """Flatten every results.json's per-subgroup rows into one long frame."""
    rows = []
    for ds in datasets:
        path = ROOT / "datasets" / ds / _results_subdir(n_human) / "results.json"
        if not path.exists():
            print(f"  ! {path.relative_to(ROOT)} missing -- skipping {ds}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
 
        for r in payload["results"]:
            method, model = r["method"], r.get("model_key", "?")
            if method not in METHODS or not r.get("subgroups"):
                continue
            for dim_name, block in _normalize_subgroups(r["subgroups"]).items():
                for row in block.get("rows", []):
                    rows.append({
                        "dataset": DATASETS.get(ds, ds),
                        "method": METHODS[method],
                        "model": model,
                        "dimension": dim_name,
                        "max_disparity": block.get("max_disparity"),
                        "max_disparity_stratified": block.get("max_disparity_stratified"),
                        **row,
                    })
    return pd.DataFrame(rows)
 
 
# Fairness summary
def collect_fairness(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-run ``stratified.fairness_metrics`` per (dataset, method, model,
    dimension) group on the rows already collected in ``long_df``.
    """
    from soepsim.stratified import fairness_metrics
 
    rows = []
    key_cols = ["dataset", "method", "model", "dimension"]
    for key, sub in long_df.groupby(key_cols):
        fm = fairness_metrics(sub)
        row = dict(zip(key_cols, key))
        for bias_col in ("bias_synthesis", "bias_global_rect", "bias_stratified_rect"):
            entry = fm.get(bias_col, {})
            row[f"{bias_col}__max_disparity"] = entry.get("max_disparity")
            row[f"{bias_col}__weighted_disparity"] = entry.get("weighted_disparity")
        row["n_worse_than_synthesis_stratified"] = fm.get("bias_stratified_rect", {}).get(
            "n_worse_than_synthesis"
        )
        row["n_worse_than_synthesis_global"] = fm.get("bias_global_rect", {}).get(
            "n_worse_than_synthesis"
        )
        row["n_subgroups"] = fm.get("n_subgroups")
        row["n_reliable"] = fm.get("n_reliable")
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def print_fairness_summary(fair_df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("EXTENSION B -- fairness (max subgroup bias disparity, %), global vs. stratified")
    print("=" * 88)
    if fair_df.empty:
        print("  (no results)")
        return
    for dim, sub in fair_df.groupby("dimension"):
        print(f"\n-- dimension: {dim} --")
        pv = sub.pivot_table(
            index=["method"], columns="dataset",
            values=["bias_global_rect__max_disparity", "bias_stratified_rect__max_disparity"],
            aggfunc="mean",
        )
        print(pv.round(2).to_string(na_rep="  —"))
    print("\nLower is fairer. Stratified < global here means Extension B's rectification")
    print("narrows the worst-subgroup gap that global rectification leaves (or widens);")
    print("check n_trials_fallback_to_global_lambda in the long table before trusting a")
    print("subgroup whose labeled cell was smaller than --min-stratum-n.")
 
 
 # Population level trade-off
def _load_human_and_predictions(dataset_key: str, dim: str):
    from soepsim.preprocessing.variables import FOCAL_QUESTIONS
 
    focal_key = FOCAL_KEY_FOR_DATASET[dataset_key]
    ds_dir = ROOT / "datasets" / dataset_key
    data_path = ds_dir / "data" / "human_test.csv"
    base_dir = ds_dir / "outputs" / "simulated_data"
    if not data_path.exists():
        print(f"  ! {data_path.relative_to(ROOT)} missing -- skipping {dataset_key}")
        return None
 
    human = pd.read_csv(data_path)
    if dim not in human.columns:
        print(f"  ! {dim!r} not in {data_path.name} -- skipping {dataset_key}")
        return None
    human = human.rename(columns={"correct_answer": "answer"})
    human = human[["id", "answer", dim]].dropna(subset=["id", "answer"])
    return FOCAL_QUESTIONS[focal_key], human, base_dir
 
 
def population_tradeoff(dataset_key: str, dim: str, trial_n: int, n_trials: int,
                         seed: int, shrinkage: float, min_stratum_n: int) -> pd.DataFrame:
    """Population-level comparison of stratified and global rectification for one dimension."""
    from soepsim.metrics import is_valid, reset_rng, trials_metrics
    from soepsim.preprocessing.recode import phi
    from soepsim.utils import BASELINE_METHODS, extract_method, find_model_csvs
 
    loaded = _load_human_and_predictions(dataset_key, dim)
    if loaded is None:
        return pd.DataFrame()
    fq, human, base_dir = loaded
 
    rows = []
    for pred_path in find_model_csvs(str(base_dir)):
        model_key, method_key = Path(pred_path).parent.name, Path(pred_path).stem
        if extract_method(f"{model_key}:{method_key}") in BASELINE_METHODS:
            continue  # no rectified variant for the random baseline (see evaluate.eval_file)
        pred = pd.read_csv(pred_path)
        if "id" not in pred.columns or "answer" not in pred.columns:
            continue
        pred = pred[["id", "answer"]].rename(columns={"answer": "answer_pred"})
 
        merged = human.merge(pred, on="id", how="inner")
        y = phi(merged["answer"], fq)
        yhat = phi(merged["answer_pred"], fq)
        groups = merged[dim].to_numpy()
        theta_true = float(np.nanmean(y[pd.notna(groups)]))
 
        valid = ~(np.isnan(y) | np.isnan(yhat)) & pd.notna(groups)
        y, yhat, groups = y[valid], yhat[valid], groups[valid]
        if len(y) <= trial_n:
            continue
 
        reset_rng(seed)
        g_bt = trials_metrics(y, yhat, trial_n, n_trials, theta_true, mode="ppi")
        reset_rng(seed)  # same draws for both, so the comparison isn't confounded by RNG noise
        s_bt = trials_metrics(y, yhat, trial_n, n_trials, theta_true, mode="stratified",
                               groups=groups, shrinkage=shrinkage, min_stratum_n=min_stratum_n)
 
        rows.append({
            "dataset": DATASETS.get(dataset_key, dataset_key), "model": model_key,
            "method": METHODS.get(method_key, method_key), "dimension": dim, "trial_n": trial_n,
            "bias_global": g_bt["bias"], "bias_stratified": s_bt["bias"],
            "ess_global": g_bt["ess"] if is_valid(g_bt["coverage"]) else None,
            "ess_stratified": s_bt["ess"] if is_valid(s_bt["coverage"]) else None,
            "coverage_global": g_bt["coverage"], "coverage_stratified": s_bt["coverage"],
        })
    return pd.DataFrame(rows)
 
 
def print_population_tradeoff(df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("EXTENSION B -- population-level trade-off (global PPI vs. aggregated stratified PPI)")
    print("=" * 88)
    if df.empty:
        print("  (no results -- data/human_test.csv not found here; this needs to run where "
              "the SOEP data actually lives, see DATA_PROTECTION.md)")
        return
    for dim, sub in df.groupby("dimension", sort=False):
        print(f"\n-- dimension: {dim} --")
        print(sub.drop(columns=["dimension"]).round(2).to_string(index=False, na_rep="  —"))
    print("\nStratified should not be systematically worse here (bias up, ESS down) for the")
    print("thesis's efficiency-cost claim to hold; a small cost is expected and is the whole")
    print("point of RQ2 -- the fairness gain (see the table above) is what should justify it.")
 
 
# Budget allocation analysis
def budget_allocation_analysis(dataset_key: str, dims: list[str], trial_ns: list[int],
                                 n_trials: int, seed: int, shrinkage: float,
                                 min_stratum_n: int) -> pd.DataFrame:
    """
    Bias-variance trade-off of the stratified correction as a function of the
    labeled budget n_human (``trial_ns``) and the number of strata |G|
    """
    from soepsim.metrics import is_valid, reset_rng, trials_metrics
    from soepsim.preprocessing.recode import phi
    from soepsim.utils import BASELINE_METHODS, extract_method, find_model_csvs
 
    rows = []
    for dim in dims:
        loaded = _load_human_and_predictions(dataset_key, dim)
        if loaded is None:
            continue
        fq, human, base_dir = loaded
        n_strata = int(human[dim].nunique(dropna=True))
 
        for pred_path in find_model_csvs(str(base_dir)):
            model_key, method_key = Path(pred_path).parent.name, Path(pred_path).stem
            if extract_method(f"{model_key}:{method_key}") in BASELINE_METHODS:
                continue
            pred = pd.read_csv(pred_path)
            if "id" not in pred.columns or "answer" not in pred.columns:
                continue
            pred = pred[["id", "answer"]].rename(columns={"answer": "answer_pred"})
 
            merged = human.merge(pred, on="id", how="inner")
            y = phi(merged["answer"], fq)
            yhat = phi(merged["answer_pred"], fq)
            groups = merged[dim].to_numpy()
            # Same theta* convention as population_tradeoff
            theta_true = float(np.nanmean(y[pd.notna(groups)]))
 
            valid = ~(np.isnan(y) | np.isnan(yhat)) & pd.notna(groups)
            y_v, yhat_v, groups_v = y[valid], yhat[valid], groups[valid]
            # Share of the smallest stratum: trial_n / |G| hides that e.g.
            # migback's smallest group gets ~4 of 100, not 33.
            min_share = float(pd.Series(groups_v).value_counts(normalize=True).min())
 
            for trial_n in trial_ns:
                if len(y_v) <= trial_n:
                    continue
                reset_rng(seed)
                g_bt = trials_metrics(y_v, yhat_v, trial_n, n_trials, theta_true, mode="ppi")
                reset_rng(seed)  # same draws for global and stratified at this trial_n
                s_bt = trials_metrics(y_v, yhat_v, trial_n, n_trials, theta_true, mode="stratified",
                                       groups=groups_v, shrinkage=shrinkage,
                                       min_stratum_n=min_stratum_n)
 
                rows.append({
                    "dataset": DATASETS.get(dataset_key, dataset_key), "model": model_key,
                    "method": METHODS.get(method_key, method_key), "dimension": dim,
                    "n_strata": n_strata, "trial_n": trial_n,
                    "avg_labeled_per_stratum": trial_n / n_strata,
                    "min_labeled_per_stratum_expected": trial_n * min_share,
                    "ci_width_global": g_bt["ci_width"], "ci_width_stratified": s_bt["ci_width"],
                    "ess_global": g_bt["ess"] if is_valid(g_bt["coverage"]) else None,
                    "ess_stratified": s_bt["ess"] if is_valid(s_bt["coverage"]) else None,
                    "coverage_stratified": s_bt["coverage"],
                    "n_trials_used_stratified": s_bt["n_trials_used"],
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ci_width_ratio"] = df["ci_width_stratified"] / df["ci_width_global"]
    return df
 
 
def print_budget_allocation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("EXTENSION B -- budget allocation: variance of the stratified correction")
    print("vs. labeled budget (n_human) and number of strata |G|")
    print("=" * 88)
    if df.empty:
        print("  (no results -- data/human_test.csv not found here; this needs to run where "
              "the SOEP data actually lives, see DATA_PROTECTION.md)")
        return
 
    summary = (
        df.groupby(["dimension", "n_strata", "trial_n"])
        .agg(ci_width_ratio=("ci_width_ratio", "mean"),
             avg_labeled_per_stratum=("avg_labeled_per_stratum", "first"),
             min_labeled=("min_labeled_per_stratum_expected", "mean"),
             n_trials_used_stratified=("n_trials_used_stratified", "mean"))
        .reset_index()
        .sort_values(["n_strata", "trial_n"])
    )
    pv = summary.pivot_table(index=["dimension", "n_strata"], columns="trial_n",
                              values="ci_width_ratio")
    print("\nCI-width ratio, stratified / global (>1 = stratified is noisier here; averaged")
    print("over model x method x dataset). Columns are n_human (labeled budget):")
    print(pv.round(2).to_string(na_rep="  —"))
 
    print("\nAvg. labeled observations per stratum (trial_n / |G|) -- the proposal's own")
    print("mechanism for why variance should grow as the budget is split more ways:")
    avg_tbl = summary.pivot_table(index=["dimension", "n_strata"], columns="trial_n",
                                   values="avg_labeled_per_stratum")
    print(avg_tbl.round(1).to_string(na_rep="  —"))
 
    print("\nExpected labeled observations in the SMALLEST stratum (trial_n x its share)")
    print("-- the number that decides whether that stratum falls back (< --min-stratum-n):")
    min_tbl = summary.pivot_table(index=["dimension", "n_strata"], columns="trial_n",
                                   values="min_labeled")
    print(min_tbl.round(1).to_string(na_rep="  —"))
 
    print("\nReading this: for a fixed n_human, more strata (higher |G|) should mean a higher")
    print("ci_width_ratio (thinner per-stratum data -> noisier stratified correction). For a")
    print("fixed dimension, the ratio should fall as n_human grows (lower is better, closer to")
    print("global's precision). Rows where ci_width_ratio is far above 1 at low n_human/high")
    print("|G| are exactly the practical challenge the proposal names -- consider raising")
    print("n_human for that dimension or leaning on --shrinkage there instead.")
 
 
 # Tables
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-dataset", nargs="*", default=list(DATASETS))
    parser.add_argument("--skip-eval", action="store_true",
                        help="Reuse existing results.json (must already be from an "
                             "Extension-B run -- see run_evaluations docstring).")
    parser.add_argument("--dims", nargs="*", default=None,
                        help="Subgroup dimensions to sweep (pieces 1 and 3). Default: all "
                             "of SUBGROUP_VARS (--subgroups-all).")
    parser.add_argument("--shrinkage", type=float, default=0.0,
                        help="0.0 = fully stratified lambda, 1.0 = global lambda "
                             "everywhere (James-Stein-style blend).")
    parser.add_argument("--min-stratum-n", type=int, default=10)
    parser.add_argument("--skip-population-tradeoff", action="store_true",
                        help="Skip piece 2 (raw-CSV population-level comparison at a single "
                             "trial_n) -- it repeats the full bootstrap a second time.")
    parser.add_argument("--population-tradeoff-dims", nargs="*", default=None,
                        help="Dimensions for piece 2 (population-level trade-off). "
                             "Default: the same as --dims, i.e. all of SUBGROUP_VARS.")
    parser.add_argument("--n-human", type=int, default=100,
                        help="Labeled-set size for pieces 1 + 2 (subgroup tables and "
                             "population trade-off). 100 matches the paper; any other "
                             "value writes to results_n<N>/ and tables_n<N>/.")
    parser.add_argument("--skip-budget-allocation", action="store_true",
                        help="Skip piece 3 (bias-variance trade-off vs. n_human x |G|) -- "
                             "the most expensive part (dimensions x labeled budgets).")
    parser.add_argument("--budget-trial-ns", type=int, nargs="+",
                         default=DEFAULT_BUDGET_TRIAL_NS,
                         help="Labeled-budget (n_human) values to sweep for the budget "
                              "allocation analysis. Default: 50 100 150 200.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: outputs/tables (n_human 100) or "
                             "outputs/tables_n<N> otherwise.")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = ROOT / "outputs" / (
            "tables" if args.n_human == 100 else f"tables_n{args.n_human}")
    print(f"n_human = {args.n_human}  ->  tables in {args.out_dir}")
 
    from soepsim.preprocessing.variables import SUBGROUP_VARS
    all_dims = args.dims or list(SUBGROUP_VARS)
 
    datasets = [d for d in args.only_dataset if d in DATASETS]
    run_evaluations(datasets, args.n_trials, args.seed, args.skip_eval,
                     args.shrinkage, args.min_stratum_n, args.dims, args.n_human)
 
    long_df = collect_long(datasets, args.n_human)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if long_df.empty:
        print("\nNo subgroup results found. Run the simulation stage first "
              "(see README, 'Running Extension A') -- Extension B evaluates the same "
              "prediction CSVs, just with subgroup sweeping turned on.")
    else:
        long_df.to_csv(args.out_dir / "extension_b_subgroup_long.csv", index=False)
        fair_df = collect_fairness(long_df)
        print_fairness_summary(fair_df)
        fair_df.to_csv(args.out_dir / "extension_b_fairness.csv", index=False)
        print(f"\nTables written to {args.out_dir}")
 
    if not args.skip_population_tradeoff:
        pop_dims = args.population_tradeoff_dims or all_dims
        pop_rows = [
            population_tradeoff(ds, dim, args.n_human, args.n_trials,
                                 args.seed, args.shrinkage, args.min_stratum_n)
            for ds in datasets
            for dim in pop_dims
        ]
        pop_df = pd.concat([d for d in pop_rows if not d.empty], ignore_index=True) \
            if any(not d.empty for d in pop_rows) else pd.DataFrame()
        print_population_tradeoff(pop_df)
        if not pop_df.empty:
            pop_df.to_csv(args.out_dir / "extension_b_population_tradeoff.csv", index=False)
 
    if not args.skip_budget_allocation:
        budget_dims = all_dims
        budget_rows = [
            budget_allocation_analysis(ds, budget_dims, args.budget_trial_ns, args.n_trials,
                                        args.seed, args.shrinkage, args.min_stratum_n)
            for ds in datasets
        ]
        budget_df = pd.concat([d for d in budget_rows if not d.empty], ignore_index=True) \
            if any(not d.empty for d in budget_rows) else pd.DataFrame()
        print_budget_allocation(budget_df)
        if not budget_df.empty:
            budget_df.to_csv(args.out_dir / "extension_b_budget_allocation.csv", index=False)
 
 
if __name__ == "__main__":
    main()