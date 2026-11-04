"""
Extension C budget allocation between adaptation and rectification.

    python experiments/extension_c_allocation.py prepare  --focal q1
    python experiments/extension_c_allocation.py simulate --focal q1 --job-list ft --job-index 0
    python experiments/extension_c_allocation.py evaluate --focal q1 --out-dir outputs/tables/rq3
"""
 
from __future__ import annotations
 
import argparse
import json
import math
import os
import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import yaml
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
 
from soepsim.ppi import ppi_mean_ci_with_var  # noqa: E402
from soepsim.preprocessing.recode import phi  # noqa: E402
from soepsim.preprocessing.variables import FOCAL_QUESTIONS  # noqa: E402
 
DEFAULT_ALPHAS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80]
METHOD_LABELS = {"zeroshot": "No adaptation (alpha = 0)", "ft": "Domain-FT", "icl": "ICL",
                 "human": "Human only (B responses)"}
COVERAGE_TOLERANCE = 0.90
 

def tag(alpha: float) -> str:
    """0.05 -> 'a05', 0.2 -> 'a20', 0 -> 'a00'."""
    return f"a{int(round(alpha * 100)):02d}"
 
 
def dataset_dir(args) -> Path:
    return Path(args.dataset_dir) if args.dataset_dir else ROOT / "datasets" / f"soep_{args.focal}_full"
 
 
def work_rel(seed: int) -> Path:
    """Work directory relative to the dataset directory (configs use relative paths)."""
    return Path("extc") / f"seed{seed}"
 

def _ft_example(prompt_json: str, letter: str) -> dict:
    """Training example in the exact inference format (same chat messages)."""
    msgs = json.loads(prompt_json)
    return {
        "system": msgs[0]["content"],
        "conversations": [
            {"role": "user", "content": msgs[1]["content"]},
            {"role": "assistant", "content": letter},
        ],
    }
 
 
def _finetune_cfg(base: dict, rel: Path, model: str, t: str, n_train: int, args) -> dict:
    cfg = dict(base)
    cfg["datasets"] = [dict(base["datasets"][0], path=str(rel / "data" / f"{t}_train.chat.jsonl"))]
    cfg["dataset_prepared_path"] = str(rel / "models" / model / t / "prepared")
    cfg["output_dir"] = str(rel / "models" / model / t / "lora")
    # no validation split: with 50 examples every labeled answer counts
    cfg["val_set_size"] = 0
    cfg.pop("evals_per_epoch", None)
    cfg.pop("eval_steps", None)
    cfg["saves_per_epoch"] = 1
    # comparable number of optimisation steps across alphas
    cfg["num_epochs"] = int(min(args.max_epochs, max(1, math.ceil(args.min_train_examples / n_train))))
    cfg["micro_batch_size"] = int(min(cfg.get("micro_batch_size", 3), n_train))
    return cfg
 
 
def _simulate_cfg(base: dict, rel: Path, model: str, method: str, t: str, args) -> dict:
    cfg = {k: v for k, v in base.items()
           if k not in {"adapter", "output_file", "method", "persona_file", "persona_input",
                        "icl_k", "icl_index_dir", "icl_embed_model", "icl_pool_file", "icl_strategy"}}
    cfg["model_key"] = model
    cfg["method"] = "silicon_sampling" if method == "zeroshot" else method
    cfg["output_file"] = str(rel / "predictions" / model / f"{method}_{t}.csv")
    if method == "ft":
        cfg["adapter"] = str(rel / "models" / model / t / "lora")
    if method == "icl":
        cfg["icl_pool_file"] = str(rel / "data" / f"{t}_icl_pool.csv")
        cfg["icl_k"] = args.icl_k
        cfg["icl_strategy"] = args.icl_strategy
        cfg["icl_embed_model"] = args.icl_embed_model
        cfg["batch_size"] = min(int(base.get("batch_size", 32)), args.icl_batch_size)
    return cfg
 
 
def _base_simulate_cfg(base_dir: Path, model: str, ft_base: dict) -> dict:
    p = base_dir / f"simulate_{model}_silicon_sampling.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return {"hf_model": ft_base["base_model"], "language": "de", "batch_size": 32,
            "max_new_tokens": 8, "flash_attention": False, "enable_thinking": False,
            "seed": 42, "temperature": 0.7, "do_sample": True}
 
 
def cmd_prepare(args) -> None:
    ddir = dataset_dir(args)
    fq = FOCAL_QUESTIONS[args.focal]
    human = pd.read_csv(ddir / "data" / "human_test.csv")
    needed = {"id", "pid", "correct_answer", "demographic_profile", "prompt"}
    missing = needed - set(human.columns)
    if missing:
        raise SystemExit(f"{ddir}/data/human_test.csv lacks columns {sorted(missing)}")
    human = human.dropna(subset=["correct_answer", "prompt"]).reset_index(drop=True)
 
    alphas = sorted(args.alphas)
    n_a_max = int(round(max(alphas) * args.budget))
    base_dir = Path(args.base_config_dir) if args.base_config_dir else ROOT / "datasets" / f"soep_{args.focal}" / "configs"
 
    for seed in args.seeds:
        rel = work_rel(seed)
        wdir = ddir / rel
        for sub in ("data", "configs", "predictions", "models"):
            (wdir / sub).mkdir(parents=True, exist_ok=True)
 
        perm = np.random.default_rng(seed).permutation(len(human))
        a_rows = human.iloc[perm[:n_a_max]].reset_index(drop=True)
        frame = human.iloc[perm[n_a_max:]].reset_index(drop=True)
        if args.frame_size and len(frame) > args.frame_size:
            frame = frame.iloc[:args.frame_size].reset_index(drop=True)
        if len(frame) < 3 * args.budget:
            print(f"  ! seed {seed}: evaluation frame has only {len(frame)} respondents "
                  f"(< 3 x budget). U becomes small for alpha near 0.")
        if len(frame) <= args.budget:
            raise SystemExit("Evaluation frame not larger than the budget -- use design=full data.")
 
        frame.to_csv(wdir / "data" / "frame.csv", index=False)
        y = phi(frame["correct_answer"], fq)
 
        jobs_ft, jobs_sim_ft, jobs_sim_other = [], [], []
        for model in args.models:
            ft_base_path = base_dir / f"finetune_{model}.yaml"
            if not ft_base_path.exists():
                raise SystemExit(f"Base fine-tuning config {ft_base_path} not found (--base-config-dir).")
            ft_base = yaml.safe_load(ft_base_path.read_text(encoding="utf-8"))
            sim_base = _base_simulate_cfg(base_dir, model, ft_base)
 
            cfg = _simulate_cfg(sim_base, rel, model, "zeroshot", tag(0), args)
            (wdir / "configs" / f"simulate_{model}_zeroshot_{tag(0)}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            jobs_sim_other.append(f"{model} zeroshot {tag(0)}")
 
            for alpha in alphas:
                t = tag(alpha)
                n_a = int(round(alpha * args.budget))
                a_set = a_rows.iloc[:n_a]
                if model == args.models[0]:
                    with (wdir / "data" / f"{t}_train.chat.jsonl").open("w", encoding="utf-8") as f:
                        for _, r in a_set.iterrows():
                            f.write(json.dumps(_ft_example(r["prompt"], r["correct_answer"]),
                                               ensure_ascii=False) + "\n")
                    a_set[["id", "pid", "demographic_profile", "correct_answer"]].rename(
                        columns={"correct_answer": "answer"}
                    ).to_csv(wdir / "data" / f"{t}_icl_pool.csv", index=False)
 
                ft_cfg = _finetune_cfg(ft_base, rel, model, t, n_a, args)
                (wdir / "configs" / f"finetune_{model}_{t}.yaml").write_text(
                    yaml.safe_dump(ft_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                jobs_ft.append(f"{model} {t}")
                for method, bucket in (("ft", jobs_sim_ft), ("icl", jobs_sim_other)):
                    cfg = _simulate_cfg(sim_base, rel, model, method, t, args)
                    (wdir / "configs" / f"simulate_{model}_{method}_{t}.yaml").write_text(
                        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                    bucket.append(f"{model} {method} {t}")
 
        for name, lines in (("jobs_finetune.txt", jobs_ft), ("jobs_simulate_ft.txt", jobs_sim_ft),
                            ("jobs_simulate_other.txt", jobs_sim_other)):
            (wdir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
 
        split = {
            "focal": args.focal, "seed": seed, "budget": args.budget, "alphas": alphas,
            "n_dataset": int(len(human)), "n_adaptation_max": n_a_max,
            "n_frame": int(len(frame)), "theta_star": float(np.nanmean(y)),
            "n_train_by_alpha": {tag(a): int(round(a * args.budget)) for a in alphas},
            "n_human_by_alpha": {tag(a): int(args.budget - round(a * args.budget))
                                 for a in [0.0, *alphas]},
            "epochs_by_alpha": {tag(a): int(min(args.max_epochs, max(1, math.ceil(
                args.min_train_examples / round(a * args.budget))))) for a in alphas},
            "icl_k": args.icl_k, "icl_strategy": args.icl_strategy, "models": args.models,
            "adaptation_ids": a_rows["id"].tolist(),
        }
        (wdir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
 
        print(f"\n=== seed {seed}: {wdir}")
        print(f"  dataset {len(human)} | adaptation pool {n_a_max} | evaluation frame {len(frame)} "
              f"| theta* = {split['theta_star']:.4f}")
        for a in alphas:
            t = tag(a)
            print(f"  {t}: |A| = {split['n_train_by_alpha'][t]:4d}  n_human = "
                  f"{split['n_human_by_alpha'][t]:4d}  FT epochs = {split['epochs_by_alpha'][t]}")
        print(f"  jobs: {len(jobs_ft)} fine-tuning, {len(jobs_sim_ft)} FT simulations, "
              f"{len(jobs_sim_other)} ICL/zero-shot simulations")
 
 

def cmd_simulate(args) -> None:
    ddir = dataset_dir(args)
    rel = work_rel(args.seed)
    if args.job_list:
        lines = (ddir / rel / f"jobs_simulate_{args.job_list}.txt").read_text().split()
        jobs = [lines[i:i + 3] for i in range(0, len(lines), 3)]
        model, method, t = jobs[args.job_index]
    else:
        model, method, t = args.model, args.method, tag(args.alpha)
    cfg_rel = rel / "configs" / f"simulate_{model}_{method}_{t}.yaml"
    if method == "ft" and not (ddir / rel / "models" / model / t / "lora").exists():
        raise SystemExit(f"No LoRA adapter for {model} {t} -- run the fine-tuning job first.")
    print(f"simulate: model={model} method={method} {t}  config={cfg_rel}")
 
    from soepsim.runner import simulate_main
 
    os.chdir(ddir)
    simulate_main(args.focal, ["--config", str(cfg_rel), "--data", str(rel / "data" / "frame.csv")])
 
 
# Evaluation
def _paired_trials(y, yhat, n_h, n_trials, rng, budget, theta):
    """PPI (lambda_opt) over n_trials paired draws of H; returns per-trial records."""
    n = len(y)
    recs = []
    for _ in range(n_trials):
        idx = rng.choice(n, size=n_h, replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        yl, yhl, yhu = y[mask], yhat[mask], yhat[~mask]
        ok = ~(np.isnan(yl) | np.isnan(yhl))
        yl, yhl = yl[ok], yhl[ok]
        yhu = yhu[~np.isnan(yhu)]
        if len(yl) < 2 or len(yhu) == 0:
            continue
        if yhat is y:  # human-only reference: plain mean of H
            est, var, lam = float(np.mean(yl)), float(np.var(yl, ddof=1) / len(yl)), 0.0
            lo, hi = est - 1.96 * math.sqrt(var), est + 1.96 * math.sqrt(var)
        else:
            (lo, hi), var, lam = ppi_mean_ci_with_var(yl, yhl, yhu)
            est = lam * float(np.mean(yhu)) + float(np.mean(yl - lam * yhl))
        s2 = float(np.var(yl, ddof=1))
        recs.append({
            "err": (est - theta) / abs(theta) * 100,
            "covered": lo <= theta <= hi,
            "width": hi - lo,
            "lam": lam,
            "ess_same_n": ((s2 / len(yl)) / var - 1) * 100 if var > 0 else np.nan,
            "ess_budget": ((s2 / budget) / var - 1) * 100 if var > 0 else np.nan,
        })
    return pd.DataFrame(recs)
 
 
def _summarise(tr: pd.DataFrame) -> dict:
    cov = float(tr["covered"].mean())
    return {
        "bias": float(tr["err"].abs().mean()),
        "signed_bias": float(tr["err"].mean()),
        "signed_bias_ci_lo": float(np.percentile(tr["err"], 2.5)),
        "signed_bias_ci_hi": float(np.percentile(tr["err"], 97.5)),
        "coverage": cov,
        "ci_width": float(tr["width"].mean()),
        "lam": float(tr["lam"].mean()),
        "ess_same_n": float(tr["ess_same_n"].mean()),
        "ess_budget": float(tr["ess_budget"].mean()),
        "ess_reportable": cov >= COVERAGE_TOLERANCE,
        "n_trials_used": int(len(tr)),
    }
 
 
def cmd_evaluate(args) -> None:
    ddir = dataset_dir(args)
    fq = FOCAL_QUESTIONS[args.focal]
    rows, trials = [], []
    for seed in args.seeds:
        wdir = ddir / work_rel(seed)
        split = json.loads((wdir / "split.json").read_text())
        frame = pd.read_csv(wdir / "data" / "frame.csv")
        y = phi(frame["correct_answer"], fq)
        theta = float(np.nanmean(y))
        budget = split["budget"]
        # Answers that phi maps to NaN carry no
        # information for the mean, so the human-only benchmark with the full
        # budget uses the expected number of valid answers among B respondents.
        budget_eff = budget * float(np.mean(~np.isnan(y)))
        alphas = [0.0, *split["alphas"]]
 
        for model in split["models"]:
            for alpha in alphas:
                t = tag(alpha)
                n_h = split["n_human_by_alpha"][t]
                methods = ["zeroshot", "human"] if alpha == 0 else ["ft", "icl"]
                for method in methods:
                    if method == "human":
                        yhat, parse_rate = y, 1.0
                    else:
                        pred_path = wdir / "predictions" / model / f"{method}_{t}.csv"
                        if not pred_path.exists():
                            print(f"  ! missing {pred_path.relative_to(ddir)} -- skipped")
                            continue
                        pred = pd.read_csv(pred_path)[["id", "answer"]]
                        merged = frame[["id"]].merge(pred, on="id", how="left")
                        yhat = phi(merged["answer"], fq)
                        parse_rate = float(merged["answer"].notna().mean())
                    # identical draws for every method at a given alpha (paired comparison)
                    rng = np.random.default_rng([args.eval_seed, seed, int(round(alpha * 100))])
                    tr = _paired_trials(y, yhat, n_h, args.n_trials, rng, budget_eff, theta)
                    if tr.empty:
                        continue
                    trials.append(tr.assign(
                        method=method, alpha=alpha,
                        model=model, seed=seed))
                    rows.append({
                        "focal": args.focal, "seed": seed, "model": model, "method": method,
                        "method_label": METHOD_LABELS[method], "alpha": alpha,
                        "n_adaptation": budget - n_h, "n_human": n_h, "n_frame": len(frame),
                        "theta_star": theta, "parse_rate": parse_rate, **_summarise(tr),
                    })
 
    if not rows:
        raise SystemExit("No predictions found -- run the simulation jobs first.")
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "tables" / f"extc_{args.focal}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "extension_c_allocation_long.csv", index=False)
    pd.concat(trials).to_csv(
        out_dir / "extension_c_trials.csv", index=False)
    report(df, out_dir)
    print(f"\nResults written to {out_dir}")
 
 # Report
def _agg(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over models and seeds (sd across seeds if more than one)."""
    keys = ["method", "alpha", "n_adaptation", "n_human"]
    cols = ["bias", "signed_bias", "signed_bias_ci_lo", "signed_bias_ci_hi", "coverage",
            "ci_width", "lam", "ess_same_n", "ess_budget", "parse_rate"]
    return df.groupby(keys, as_index=False)[cols].mean().sort_values(["alpha", "method"])
 
 
def report(df: pd.DataFrame, out_dir: Path) -> None:
    agg = _agg(df)
    pd.set_option("display.width", 200)
    print("\n" + "=" * 96)
    print("EXTENSION C -- budget allocation (Rec_lambda_opt), mean over models and seeds")
    print("=" * 96)
    show = agg[["method", "alpha", "n_adaptation", "n_human", "bias", "coverage",
                "ci_width", "lam", "ess_same_n", "ess_budget", "parse_rate"]]
    print(show.round(3).to_string(index=False))
 
    # regimes as in Figure 2(d) of the paper
    reg = agg[agg["method"].isin(["ft", "icl"])].copy()
    reg["regime"] = pd.cut(reg["alpha"], [0, 0.2001, 0.5, 1.0],
                           labels=["Conservative (<=20%)", "Balanced (40%)", "Aggressive (>=60%)"])
    regimes = reg.groupby(["method", "regime"], observed=True)[["bias", "ess_budget", "ci_width"]].mean()
    print("\nRegimes (cf. paper Fig. 2d):")
    print(regimes.round(3).to_string())
    regimes.to_csv(out_dir / "extension_c_regimes.csv")
    agg.to_csv(out_dir / "extension_c_allocation_table.csv", index=False)
 
 # CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
 
    def common(p):
        p.add_argument("--focal", default="q1", choices=list(FOCAL_QUESTIONS))
        p.add_argument("--dataset-dir", default=None,
                       help="Dataset built with design=full. Default: datasets/soep_<focal>_full")
 
    p = sub.add_parser("prepare", help="Split, training files, ICL pools and configs.")
    common(p)
    p.add_argument("--models", nargs="+", default=["llama"])
    p.add_argument("--alphas", nargs="+", type=float, default=DEFAULT_ALPHAS)
    p.add_argument("--budget", type=int, default=1000)
    p.add_argument("--frame-size", type=int, default=5000,
                   help="Cap of the evaluation frame (0 = no cap).")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--icl-k", type=int, default=20)
    p.add_argument("--icl-strategy", default="similar", choices=["similar", "random"])
    p.add_argument("--icl-embed-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--icl-batch-size", type=int, default=8,
                   help="Generation batch size for ICL (long prompts).")
    p.add_argument("--min-train-examples", type=int, default=500,
                   help="FT epochs = ceil(this / |A|), so small A still gets enough steps.")
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--base-config-dir", default=None,
                   help="Where finetune_<model>.yaml lives. Default: datasets/soep_<focal>/configs")
    p.set_defaults(func=cmd_prepare)
 
    p = sub.add_parser("simulate", help="Run one simulation job.")
    common(p)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--job-list", choices=["ft", "other"], default=None)
    p.add_argument("--job-index", type=int, default=None)
    p.add_argument("--model", default="llama")
    p.add_argument("--method", choices=["ft", "icl", "zeroshot"])
    p.add_argument("--alpha", type=float, default=0.0)
    p.set_defaults(func=cmd_simulate)
 
    p = sub.add_parser("evaluate", help="Paired PPI evaluation, tables and figure.")
    common(p)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_evaluate)
 
    args = parser.parse_args()
    args.func(args)
 
 
if __name__ == "__main__":
    main()