"""Extension A replication
  
    python experiments/extension_a_replication.py
    python experiments/extension_a_replication.py --n-trials 100 --only-dataset soep_q1
    python experiments/extension_a_replication.py --skip-eval   # tables only
"""
 
from __future__ import annotations
 
import argparse
import json
import subprocess
import sys
from pathlib import Path
 
import pandas as pd
 
ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"soep_q1": "SOEP Q1 (Oekonomie)", "soep_q2": "SOEP Q2 (Politik)"}
METHODS = {
    "ft": "Domain-FT",
    "concept_guided": "Persona-guided",
    "silicon_sampling": "Demo-only",
}
BASELINES = {"random"}
TRIAL_NS = ["100", "50", "150", "200"]
 
# Blocks whose ESS figure is a genuine efficiency gain. 
RECTIFIED_BLOCKS = {"Rec_lambda_opt", "Rec_lambda=1"}
 
# Coverage at or above this counts as nominal; mirrors metrics.COVERAGE_TOLERANCE,
# which absorbs bootstrap noise around the nominal 0.95.
COVERAGE_TOLERANCE = 0.90
 
# Published ATP results at n_human = 100 (Table 2), for side-by-side reading.
ATP_REFERENCE = {
    ("Domain-FT", "Synthesize only"): {"ATP Q1": 34.27, "ATP Q2": 62.69},
    ("Persona-guided", "Synthesize only"): {"ATP Q1": 50.22, "ATP Q2": 19.99},
    ("Demo-only", "Synthesize only"): {"ATP Q1": 47.53, "ATP Q2": 31.18},
    ("Domain-FT", "Rec_lambda_opt"): {"ATP Q1": 1.73, "ATP Q2": 3.40},
    ("Persona-guided", "Rec_lambda_opt"): {"ATP Q1": 4.42, "ATP Q2": 7.65},
    ("Demo-only", "Rec_lambda_opt"): {"ATP Q1": 6.75, "ATP Q2": 4.52},
    ("Domain-FT", "Rec_lambda=1"): {"ATP Q1": 8.46, "ATP Q2": 4.11},
    ("Persona-guided", "Rec_lambda=1"): {"ATP Q1": 11.10, "ATP Q2": 4.50},
    ("Demo-only", "Rec_lambda=1"): {"ATP Q1": 16.94, "ATP Q2": 3.12},
}
 
BLOCK_ORDER = ["Baseline", "Synthesize only", "Rec_lambda=1", "Rec_lambda_opt"]
 
 
def run_evaluations(datasets, n_trials: int, seed: int, skip: bool) -> None:
    """Invoke each dataset's evaluate.py, unless results already exist."""
    for ds in datasets:
        ds_dir = ROOT / "datasets" / ds
        results = ds_dir / "outputs" / "results" / "results.json"
        if skip and results.exists():
            print(f"=== {ds}: reusing {results.relative_to(ROOT)} ===")
            continue
        print(f"\n=== {ds}: evaluating ===")
        subprocess.run(
            [sys.executable, "evaluate.py", "--trial_ns", *TRIAL_NS,
             "--n_trials", str(n_trials), "--seed", str(seed)],
            cwd=ds_dir, check=True,
        )
 
 
def collect(datasets) -> pd.DataFrame:
    """Flatten every results.json into one long frame."""
    rows = []
    for ds in datasets:
        path = ROOT / "datasets" / ds / "outputs" / "results" / "results.json"
        if not path.exists():
            print(f"  ! {path.relative_to(ROOT)} missing — skipping {ds}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
 
        for r in payload["results"]:
            method, model = r["method"], r.get("model_key", "?")
 
            llm = r.get("llm")
            if llm:
                if method in METHODS:
                    rows.append(_row(ds, "Synthesize only", METHODS[method], model, llm))
                elif method in BASELINES:
                    rows.append(_row(ds, "Baseline", "Baseline", model, llm))
 
            for key, label in [("rec_lam_opt", "Rec_lambda_opt"), ("rec_lam_1", "Rec_lambda=1")]:
                block = r.get(key)
                if method in METHODS and block:
                    rows.append(_row(ds, label, METHODS[method], model, block))
 
    return pd.DataFrame(rows)
 
 
def _row(dataset, block, method, model, block_dict) -> dict:
    """One per-model result cell.

    The ESS gain is suppressed (a) for unrectified blocks, where it is not
    meaningful, and (b) for rectified blocks that miss nominal coverage (< 0.90).
    """
    ess = block_dict.get("ess", {}).get("100")
    rectified = block in RECTIFIED_BLOCKS
    reportable = rectified and bool(block_dict.get("ess_reportable", False))
    return {
        "dataset": DATASETS.get(dataset, dataset),
        "block": block,
        "method": method,
        "model": model,
        "bias": block_dict.get("bias"),
        "bias_ci_lo": (block_dict.get("bias_ci") or [None, None])[0],
        "bias_ci_hi": (block_dict.get("bias_ci") or [None, None])[1],
        "signed_bias": block_dict.get("signed_bias"),
        "signed_bias_ci_lo": (block_dict.get("signed_bias_ci") or [None, None])[0],
        "signed_bias_ci_hi": (block_dict.get("signed_bias_ci") or [None, None])[1],
        "ess": ess if reportable else None,
        "ess_raw": ess,
        "ess_suppressed": (ess is not None) and not reportable,
        "coverage": block_dict.get("coverage", {}).get("100"),
        "lam": block_dict.get("lam"),
    }
 
 
def print_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 layout: methods x datasets, averaged over models."""
    print("\n" + "=" * 78)
    print("EXTENSION A — main table (averaged over the three models, n_human = 100)")
    print("=" * 78)
 
    out_frames = []
    for metric, header in [("bias", "Bias (%)  lower is better"),
                           ("ess", "ESS gain (%)  higher is better")]:
        print(f"\n{header}")
        pieces = []
        for block in BLOCK_ORDER:
            sub = df[df["block"] == block]
            if sub.empty:
                continue
            # dropna=False keeps rows whose cells were all suppressed, so a
            # method never silently disappears from the table.
            pv = sub.pivot_table(index="method", columns="dataset", values=metric,
                                 aggfunc="mean", dropna=False)
            pv = pv.reindex(columns=[v for v in DATASETS.values() if v in pv.columns])
            if pv.empty:
                continue
            pv["Avg"] = pv.mean(axis=1)
            pv.index = [f"{m} | {block}" if block != "Baseline" else m for m in pv.index]
            pieces.append(pv)
        if not pieces:
            print("  (no results)")
            continue
        table = pd.concat(pieces)
        print(table.round(2).to_string(na_rep="  —"))
        table["metric"] = metric
        out_frames.append(table)
 
    print("\n  — in the ESS table: unrectified cells (Baseline, Synthesize only) are")
    print("    suppressed by rule, because the unrectified ESS measures N/n rather")
    print("    than efficiency; rectified cells are suppressed when the 95% CI did")
    print("    not reach nominal coverage. Both correspond to the paper's dagger.")
    return pd.concat(out_frames) if out_frames else pd.DataFrame()
 
 
def print_atp_comparison(df: pd.DataFrame) -> None:
    """Side-by-side bias against the published ATP numbers."""
    print("\n" + "=" * 78)
    print("SOEP vs. ATP — bias (%), averaged over models")
    print("=" * 78)
    rows = []
    for (method, block), ref in ATP_REFERENCE.items():
        sub = df[(df["method"] == method) & (df["block"] == block)]
        row = {"method": method, "block": block}
        for label in DATASETS.values():
            vals = sub[sub["dataset"] == label]["bias"]
            row[label] = vals.mean() if len(vals) else None
        row.update(ref)
        rows.append(row)
    comp = pd.DataFrame(rows).set_index(["block", "method"]).sort_index()
    print(comp.round(2).to_string(na_rep="  —"))
 
 
def print_per_model(df: pd.DataFrame) -> None:
    """Per-model breakdown"""
    print("\n" + "=" * 78)
    print("Per-model bias (%)")
    print("=" * 78)
    sub = df[df["block"].isin(["Synthesize only", "Rec_lambda_opt"])]
    if sub.empty:
        print("  (no results)")
        return
    pv = sub.pivot_table(index=["model", "method"], columns=["dataset", "block"],
                         values="bias", aggfunc="mean")
    print(pv.round(2).to_string(na_rep="  —"))
 
 
def print_coverage(df: pd.DataFrame) -> None:
    """Coverage per block, read this before trusting any ESS figure."""
    sub = df[df["block"].isin(["Rec_lambda_opt", "Rec_lambda=1"])]
    if sub.empty:
        return
    print("\n" + "=" * 78)
    print("CI coverage at n_human = 100 (nominal 0.95)")
    print("=" * 78)
    pv = sub.pivot_table(index=["method", "model"], columns=["dataset", "block"],
                         values="coverage", aggfunc="mean")
    print(pv.round(3).to_string(na_rep="  —"))
    low = sub[sub["coverage"].notna() & (sub["coverage"] < 0.90)]
    if len(low):
        print("\n  Cells below the 0.90 tolerance (ESS suppressed for these):")
        for _, r in low.iterrows():
            print(f"    {r['dataset']:22} {r['method']:16} {r['model']:10} "
                  f"{r['block']:16} coverage={r['coverage']:.3f}")
    else:
        print("\n  All rectified cells reach nominal coverage.")
 
 
def print_lambdas(df: pd.DataFrame) -> None:
    """Power-tuned lambda per dataset — a small number means the model was ignored."""
    sub = df[df["block"] == "Rec_lambda_opt"]
    if sub.empty:
        return
    print("\n" + "=" * 78)
    print("Power-tuned lambda (mean over trials and models)")
    print("=" * 78)
    print(sub.pivot_table(index="method", columns="dataset", values="lam",
                          aggfunc="mean").round(3).to_string(na_rep="  —"))
    print("\nReference (paper): ATP Q1 ~ 0.30, ATP Q2 ~ 0.05, NHANES ~ 0.15.")
    print("A lambda near zero means power tuning discarded the synthetic responses;")
    print("the estimator is then effectively human-only and the ESS gain will be small.")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-dataset", nargs="*", default=list(DATASETS))
    parser.add_argument("--skip-eval", action="store_true",
                        help="Reuse existing results.json instead of re-evaluating.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "tables")
    args = parser.parse_args()
 
    datasets = [d for d in args.only_dataset if d in DATASETS]
    run_evaluations(datasets, args.n_trials, args.seed, args.skip_eval)
 
    df = collect(datasets)
    if df.empty:
        print("\nNo results found. Run the simulation stage first "
              "(see README, 'Running Extension A').")
        return
 
    table = print_main_table(df)
    print_atp_comparison(df)
    print_per_model(df)
    print_coverage(df)
    print_lambdas(df)
 
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "extension_a_long.csv", index=False)
    if not table.empty:
        table.to_csv(args.out_dir / "extension_a_main_table.csv")
    print(f"\nTables written to {args.out_dir}")
 
 
if __name__ == "__main__":
    main()
 