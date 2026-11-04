"""
Dataset-level entry points, shared across datasets instead of duplicated
per dataset directory as in the reference implementation.
"""

from __future__ import annotations
 
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
import yaml
 
from .evaluate import run_evaluation
from .methods.baseline import random_baseline
from .methods.personas import load_or_generate
from .methods.simulate_hf import get_hf_responses
from .metrics import reset_rng
from .icl import select_icl_examples
from .prompts import build_icl_block
from .preprocessing.recode import phi
from .preprocessing.variables import FOCAL_QUESTIONS, SUBGROUP_VARS, letters_for
from .utils import get_multi_choice_info, parse_multi_choice_response, parse_rate
 
log = logging.getLogger(__name__)
 
 
# Simulation
def _init_logging(tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    logs = Path("outputs/logs")
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"log_{stamp}_{tag}.log"
    logging.basicConfig(
        filename=path, level=logging.INFO, filemode="w",
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger().addHandler(logging.StreamHandler())
    return path
 
 
def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse as a mapping.")
    return cfg
 
 
def simulate_main(focal_key: str, argv: list[str] | None = None) -> None:
    """
    Generate synthetic responses for one focal question and config.

    ``method``: ``silicon_sampling`` (demo-only), ``concept_guided``
    (persona-guided), ``ft`` (Domain-FT), ``icl`` (in-context learning)
    or ``random`` (baseline).
    """
    parser = argparse.ArgumentParser(description=f"Simulate responses for {focal_key}.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/human_test.csv"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only simulate the first N respondents (smoke test).")
    args = parser.parse_args(argv)
 
    cfg = load_config(args.config)
    method = cfg.get("method", "silicon_sampling")
    model_key = cfg.get("model_key", "unknown")
    out_path = Path(cfg.get("output_file", f"outputs/simulated_data/{model_key}/{method}.csv"))
 
    _init_logging(f"{focal_key}_{model_key}_{method}")
    fq = FOCAL_QUESTIONS[focal_key]
    letters = letters_for(fq)
 
    df = pd.read_csv(args.data)
    if args.limit:
        df = df.head(args.limit)
    log.info("Loaded %d respondents for %s / %s / %s", len(df), focal_key, model_key, method)
 
    if method == "random":
        out = random_baseline(df, letters, seed=cfg.get("seed", 0))
        meta = {"method": "random", "seed": cfg.get("seed", 0)}
        _write(df, out["answer"].tolist(), out_path, meta, fq)
        return
 
    prompts = [json.loads(p) for p in df["prompt"]]
 
    if method == "concept_guided":
        persona_cfg = {**cfg, **cfg.get("persona_model", {})}
        personas = load_or_generate(
            Path(cfg.get("persona_file", "outputs/personas/personas.csv")),
            pd.read_csv(cfg.get("persona_input", "data/persona_input.csv")),
            persona_cfg,
            language=cfg.get("language", "de"),
        )
        pmap = dict(zip(personas["id"], personas["persona_description"].fillna("")))
        suffix = ("\n\nZusaetzliche Profilinformationen:\n"
                  if cfg.get("language", "de") == "de"
                  else "\n\nAdditional profile information:\n")
        for i, rid in enumerate(df["id"]):
            persona = pmap.get(rid, "")
            if persona:
                prompts[i][0]["content"] += suffix + persona
 
    elif method == "icl":
        # Examples come only from the explicit pool file
        pool_file = cfg.get("icl_pool_file")
        if not pool_file:
            raise ValueError(
                "method 'icl' needs 'icl_pool_file' in the config: a CSV (id, pid, "
                "demographic_profile, answer) of labeled respondents disjoint from the "
                "simulated frame. Create it with experiments/extension_c_allocation.py prepare."
            )
        pool = pd.read_csv(pool_file)
        examples_by_id = select_icl_examples(
            pool=pool,
            targets=df,
            k=int(cfg.get("icl_k", 20)),
            strategy=cfg.get("icl_strategy", "similar"),
            seed=int(cfg.get("seed", 0)),
            embed_model=cfg.get("icl_embed_model", "paraphrase-multilingual-MiniLM-L12-v2"),
        )
        language = cfg.get("language", "de")
        for i, rid in enumerate(df["id"]):
            block = build_icl_block(examples_by_id.get(rid, []), language=language)
            prompts[i][1]["content"] = block + prompts[i][1]["content"]
        log.info("ICL: pool %s (%d respondents), k=%s, strategy=%s",
                 pool_file, len(pool), cfg.get("icl_k", 20), cfg.get("icl_strategy", "similar"))
 
    completions, gen_meta = get_hf_responses(prompts, cfg)
    for i, c in enumerate(completions[:20]):
        log.info("Sample completion %d: %r", i, c)
 
    index2ans, all_choices = get_multi_choice_info(fq.choices_de)
    parsed = [parse_multi_choice_response(c, all_choices, index2ans) for c in completions]
    rate = parse_rate(parsed)
    log.info("Parse rate: %.4f (%d unparsed)", rate, sum(p is None for p in parsed))
    if rate < 0.95:
        log.warning("Parse rate below 0.95 — inspect the raw completions in this log "
                    "before trusting the results.")
 
    meta = {"method": method, "model_key": model_key, "parse_rate": rate, **gen_meta}
    _write(df, parsed, out_path, meta, fq, raw=completions)
 
 
def _write(df, answers, out_path: Path, meta: dict, fq, raw=None) -> None:
    """Write predictions plus a sidecar metadata JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["answer"] = answers
    if raw is not None:
        out["raw_completion"] = raw
    out.to_csv(out_path, index=False)
 
    meta_path = out_path.with_suffix(".meta.json")
    meta = {**meta, "n": int(len(out)), "focal_variable": fq.variable,
            "written_at": datetime.now().isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Wrote %s (%d rows) and %s", out_path, len(out), meta_path)
    print(f"Wrote {out_path} ({len(out)} rows)")
 
 
# Evaluation
def evaluate_main(focal_key: str, argv: list[str] | None = None) -> None:
    """Evaluate every prediction CSV of one focal question.
    """
    parser = argparse.ArgumentParser(description=f"Evaluate {focal_key}.")
    parser.add_argument("--base_dir", default="outputs/simulated_data")
    parser.add_argument("--data", type=Path, default=Path("data/human_test.csv"))
    parser.add_argument("--trial_ns", type=int, nargs="+", default=[100, 50, 150, 200])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subgroup", nargs="*", default=["sex"],
                        help="One or more columns for the subgroup bias diagnostic. "
                             "Pass nothing at all to disable the diagnostic.")
    parser.add_argument("--subgroups-all", action="store_true",
                        help="Sweep every dimension in variables.SUBGROUP_VARS "
                             "(Extension B). Overrides --subgroup.")
    parser.add_argument("--shrinkage", type=float, default=0.0,
                        help="Extension B: 0.0 = per-stratum lambda, 1.0 = global "
                             "lambda everywhere (James-Stein-style blend).")
    parser.add_argument("--min-stratum-n", type=int, default=10,
                        help="Extension B: strata with fewer labeled observations "
                             "than this fall back to the global correction.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"),
                        help="Where results.json is written. A robustness run with a "
                             "different n_human should use its own directory so the "
                             "Extension A results are not overwritten.")
    args = parser.parse_args(argv)
 
    fq = FOCAL_QUESTIONS[focal_key]
    human = pd.read_csv(args.data)
    if "correct_answer" not in human.columns:
        raise ValueError(f"{args.data} has no 'correct_answer' column — rebuild the dataset.")
    human = human.rename(columns={"correct_answer": "answer"})
 
    # Which subgroup dimensions to sweep
    requested = list(SUBGROUP_VARS) if args.subgroups_all else [c for c in (args.subgroup or []) if c]
    subgroup_cols = [c for c in dict.fromkeys(requested) if c in human.columns]
    missing = [c for c in requested if c not in human.columns]
    if missing:
        log.warning("Subgroup column(s) %s not in the human frame; skipping them.", missing)
    if requested and not subgroup_cols:
        log.warning("None of the requested subgroup dimensions exist; "
                    "subgroup metrics will be skipped entirely.")
    if subgroup_cols:
        log.info("Subgroup dimensions: %s (shrinkage=%.2f, min_stratum_n=%d)",
                 subgroup_cols, args.shrinkage, args.min_stratum_n)
 
    keep = list(dict.fromkeys(["id", "answer", *subgroup_cols]))
    human = human[keep].dropna(subset=["id", "answer"])
 
    def prepare(human_df: pd.DataFrame, pred_path: str):
        pred = pd.read_csv(pred_path)
        if "id" not in pred.columns or "answer" not in pred.columns:
            raise ValueError(f"{pred_path}: missing 'id' or 'answer'")
        pred = pred[["id", "answer"]].rename(columns={"answer": "answer_pred"})
 
        merged = human_df.merge(pred, on="id", how="inner")
        y = phi(merged["answer"], fq)
        yhat = phi(merged["answer_pred"], fq)
 
        # theta* is defined on all humans with a valid phi value, independent of
        # whether the model produced a parseable answer for that person.
        theta_true = float(np.nanmean(y))
 
        valid = ~(np.isnan(y) | np.isnan(yhat))
        meta = {
            "n_merged": int(len(merged)),
            "n_valid_pairs": int(valid.sum()),
            "coverage_of_frame": float(valid.sum() / max(len(merged), 1)),
        }
        if subgroup_cols:
            # Always a dict, one Series per dimension -- eval_file accepts both
            # a bare Series (legacy) and this mapping.
            meta["_subgroups"] = {
                col: merged.loc[valid, col].reset_index(drop=True)
                for col in subgroup_cols
            }
        return y[valid], yhat[valid], theta_true, meta
 
    reset_rng(args.seed)
    run_evaluation(
        dataset_name=focal_key,
        prepare_fn=prepare,
        human_df=human,
        base_dir=args.base_dir,
        trial_ns=args.trial_ns,
        n_trials=args.n_trials,
        seed=args.seed,
        only=args.only,
        subgroup_col=subgroup_cols or None,
        shrinkage=args.shrinkage,
        min_stratum_n=args.min_stratum_n,
        results_dir=args.results_dir,
    )