"""
Evaluation loop.
 
For every prediction CSV under ``outputs/simulated_data/<model>/<method>.csv``,
runs the bootstrap metrics in three modes: unrectified synthesis, PPI with
power-tuned lambda, PPI with lambda = 1 — and writes one ``results.json`` per
focal question.
"""
 
from __future__ import annotations
 
import fnmatch
import json
import logging
import math
from pathlib import Path
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
from .metrics import is_valid, subgroup_metrics, trials_metrics
from .utils import BASELINE_METHODS, extract_method, find_model_csvs
 
log = logging.getLogger(__name__)
 
MODES = [
    ("llm", "llm", None),
    ("rec_lam_opt", "ppi", None),
    ("rec_lam_1", "ppi", 1.0),
]
 
 
def eval_file(
    prepare_fn, human_df, pred_path, trial_ns, n_trials=100, subgroup_col=None,
    shrinkage=0.0, min_stratum_n=10,
):
    """Evaluate one prediction CSV in all three modes.

    :param prepare_fn: returns (y, yhat, theta_true, meta) for the given files
    :param trial_ns: labelled-set sizes; the first is used for the main results
    :return: nested result dict
    """
    y, yhat, theta_true, meta = prepare_fn(human_df, pred_path)
 
    model_key = Path(pred_path).parent.name
    method_key = Path(pred_path).stem
    model_name = f"{model_key}:{method_key}"
    is_baseline = extract_method(model_name) in BASELINE_METHODS
    base_trial_n = trial_ns[0]
 
    result = {
        "model": model_name,
        "model_key": model_key,
        "method": extract_method(model_name),
        "n_responses": int(len(y)),
        "theta_true": float(theta_true),
        **meta,
    }
 
    for key, mode, lam in MODES:
        if is_baseline and mode == "ppi":
            result[key] = None
            continue
 
        block, ess_by_n, cov_by_n = {}, {}, {}
        for trial_n in trial_ns:
            bt = trials_metrics(y, yhat, trial_n, n_trials, theta_true, mode=mode, lam=lam)
            ess_by_n[str(trial_n)] = bt["ess"]
            cov_by_n[str(trial_n)] = bt["coverage"]
            if trial_n == base_trial_n:
                block.update({
                    "theta_hat": bt["theta_hat"],
                    "bias": bt["bias"],
                    "bias_ci": [bt["bias_ci_lo"], bt["bias_ci_hi"]],
                    # Signed error and its interval
                    "signed_bias": bt["signed_bias"],
                    "signed_bias_ci": [bt["signed_bias_ci_lo"], bt["signed_bias_ci_hi"]],
                    "ci_width": bt["ci_width"],
                    "n_trials_used": bt["n_trials_used"],
                })
                if mode == "ppi":
                    block["lam"] = bt["lam"]
        block["ess"] = ess_by_n
        block["coverage"] = cov_by_n
        block["ess_reportable"] = is_valid(cov_by_n.get(str(base_trial_n)))
        result[key] = block
 
    # Subgroup diagnostic under global and stratified rectification
    if subgroup_col is not None and not is_baseline:
        sub = meta.get("_subgroups")
        if isinstance(sub, pd.Series):
            dim_name = subgroup_col if isinstance(subgroup_col, str) else str(subgroup_col)
            sub = {dim_name: sub}
 
        if sub:
            subgroups_out = {}
            for dim_name, series in sub.items():
                sg = subgroup_metrics(
                    y, yhat, series, base_trial_n, n_trials,
                    shrinkage=shrinkage, min_stratum_n=min_stratum_n,
                )
                subgroups_out[dim_name] = {
                    "variable": dim_name,
                    "max_disparity": sg.attrs.get("max_disparity"),
                    "max_disparity_stratified": sg.attrs.get("max_disparity_stratified"),
                    "rows": sg.to_dict("records"),
                }
            result["subgroups"] = (
                next(iter(subgroups_out.values()))
                if len(subgroups_out) == 1
                else subgroups_out
            )
    result.pop("_subgroups", None)
    return result
 
 
def run_evaluation(
    dataset_name: str,
    prepare_fn,
    human_df: pd.DataFrame,
    base_dir: str = "outputs/simulated_data",
    trial_ns: list[int] | None = None,
    n_trials: int = 100,
    seed: int = 0,
    only: list[str] | None = None,
    results_dir: Path | None = None,
    subgroup_col: str | list[str] | None = None,
    shrinkage: float = 0.0,
    min_stratum_n: int = 10,
) -> dict:
    """Evaluate all prediction CSVs of one focal question and write results.json.

    :param dataset_name: "soep_q1" or "soep_q2"
    :param n_trials: number of random draws of the labelled set
    :param subgroup_col: subgroup dimension(s)
    :param shrinkage: 0.0 = per-subgroup lambda, 1.0 = global lambda
    :param min_stratum_n: smaller subgroups fall back to the global correction
    """
    trial_ns = trial_ns or [100, 50, 150, 200]
    results_dir = results_dir or Path("outputs/results")
 
    csvs = find_model_csvs(base_dir)
    if only:
        csvs = [p for p in csvs if any(fnmatch.fnmatch(p, pat) for pat in only)]
    if not csvs:
        log.warning("No prediction CSVs found under %s — nothing to evaluate.", base_dir)
 
    results = [
        eval_file(
            prepare_fn, human_df, p, trial_ns, n_trials, subgroup_col,
            shrinkage=shrinkage, min_stratum_n=min_stratum_n,
        )
        for p in tqdm(csvs, desc=f"Evaluating {dataset_name}")
    ]
 
    payload = {
        "dataset": dataset_name,
        "config": {
            "n_trials": n_trials,
            "trial_ns": trial_ns,
            "seed": seed,
            "subgroup_col": subgroup_col,
            "shrinkage": shrinkage,
            "min_stratum_n": min_stratum_n,
        },
        "results": _clean(results),
    }
 
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    log.info("Results written to %s", out_path)
    print(f"\nResults written to {out_path}")
    return payload
 
 
def _clean(obj):
    """Replace NaN with None so the JSON is valid and readable."""
    if isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return _clean(float(obj))
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
 