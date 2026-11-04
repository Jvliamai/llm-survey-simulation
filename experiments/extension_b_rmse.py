"""
Extension b subgroup RMSE and post-stratified benchmark.

Compares global and stratified rectification per subgroup (bias, SE, RMSE over
the draws of H) and the population ESS gain of global PPI, stratified PPI and a
post-stratified human-only estimator.

    python experiments/extension_b_rmse.py --out-dir outputs/tables/rq2 \
        --dims sex sampreg age3 edu3 income_quintile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from ppi_py import ppi_mean_pointestimate  # noqa: E402

from extension_b_stratified import DATASETS, METHODS, _load_human_and_predictions  # noqa: E402
from soepsim.metrics import COVERAGE_TOLERANCE, _ess_gain  # noqa: E402
from soepsim.ppi import human_only_variance, ppi_mean_ci_with_var  # noqa: E402
from soepsim.preprocessing.recode import phi  # noqa: E402
from soepsim.stratified import _aggregate_variance, stratified_ppi  # noqa: E402

DIM_LABELS = {
    "sex": "Sex", "sampreg": "East/West", "age3": "Age", "edu3": "Education",
    "income_quintile": "Income quintile", "migback": "Migration bg.",
}


def post_stratified_variance(y_l, g_l, g_u, min_stratum_n: int) -> float:
    """Variance of the human-only post-stratified mean (stratified_ppi with lambda = 0)."""
    strata = list(pd.unique(g_u))
    N = len(g_u)
    w = {g: float(np.sum(g_u == g)) / N for g in strata}
    fallback = [g for g in strata if int(np.sum(g_l == g)) < min_stratum_n]
    return _aggregate_variance(
        y_l, np.zeros(len(y_l)), np.zeros(N), g_l, g_u,
        strata, fallback, w, {g: 0.0 for g in strata}, 0.0,
    )


def _err(draws: list[float], theta_g: float) -> dict:
    d = np.asarray(draws, dtype=float)
    if len(d) == 0:
        return {"bias": np.nan, "se": np.nan, "rmse": np.nan}
    denom = abs(theta_g) + 1e-10
    e = d - theta_g
    return {
        "bias": abs(float(e.mean())) / denom * 100,
        "se": float(d.std()) / denom * 100,
        "rmse": float(np.sqrt(np.mean(e ** 2))) / denom * 100,
    }


def evaluate_file(y_all, yhat_all, groups_all, n, n_trials, seed, shrinkage, min_stratum_n):
    """Subgroup rows and one population row for one prediction file."""
    has_g = pd.notna(groups_all)
    theta_true = float(np.nanmean(y_all[has_g]))
    labels = list(pd.unique(groups_all[has_g]))
    theta_g = {g: float(np.nanmean(y_all[has_g & (groups_all == g)])) for g in labels}

    valid = ~(np.isnan(y_all) | np.isnan(yhat_all)) & has_g
    y, yhat, groups = y_all[valid], yhat_all[valid], groups_all[valid]
    if len(y) <= n:
        return [], None

    rng = np.random.default_rng(seed)  # same draws as trials_metrics after reset_rng(seed)
    glob_draws = {g: [] for g in labels}
    strat_draws = {g: [] for g in labels}
    ess_glob, ess_strat, ess_ps, cov_glob, cov_strat = [], [], [], [], []

    for _ in range(n_trials):
        idx = rng.choice(len(y), size=n, replace=False)
        mask = np.zeros(len(y), dtype=bool)
        mask[idx] = True
        y_l, yh_l, g_l = y[mask], yhat[mask], groups[mask]
        yh_u, g_u = yhat[~mask], groups[~mask]
        var_h = human_only_variance(y_l)

        # Global rectification
        (lo, hi), var_glob, lam = ppi_mean_ci_with_var(y_l, yh_l, yh_u, alpha=0.05, lam=None)
        theta_glob = float(np.asarray(ppi_mean_pointestimate(y_l, yh_l, yh_u, lam=lam)).squeeze())
        ess_glob.append(_ess_gain(var_h, var_glob))
        cov_glob.append(1.0 if lo <= theta_true <= hi else 0.0)
        shift = theta_glob - float(np.mean(yh_u))
        for g in labels:
            m = g_u == g
            if m.any():
                glob_draws[g].append(float(np.mean(yh_u[m])) + shift)

        # Stratified rectification
        try:
            s = stratified_ppi(y_l, yh_l, yh_u, g_l, g_u, lam=None,
                               shrinkage=shrinkage, min_stratum_n=min_stratum_n)
        except ValueError:
            s = None
        if s is not None:
            lo_s, hi_s = s["ci"]
            ess_strat.append(_ess_gain(var_h, s["var"]))
            cov_strat.append(1.0 if lo_s <= theta_true <= hi_s else 0.0)
            for g, t in s["theta_by_stratum"].items():
                if g in strat_draws and not np.isnan(t):
                    strat_draws[g].append(float(t))

        # Post-stratified human-only benchmark
        ess_ps.append(_ess_gain(var_h, post_stratified_variance(y_l, g_l, g_u, min_stratum_n)))

    rows = []
    for g in labels:
        eg, es = _err(glob_draws[g], theta_g[g]), _err(strat_draws[g], theta_g[g])
        rows.append({
            "subgroup": g, "n_group": int(np.sum(groups == g)), "theta_g": theta_g[g],
            "bias_glob": eg["bias"], "se_glob": eg["se"], "rmse_glob": eg["rmse"],
            "bias_strat": es["bias"], "se_strat": es["se"], "rmse_strat": es["rmse"],
        })

    c_glob = float(np.mean(cov_glob)) if cov_glob else np.nan
    c_strat = float(np.mean(cov_strat)) if cov_strat else np.nan
    pop = {
        "ess_glob": float(np.nanmean(ess_glob)) if c_glob >= COVERAGE_TOLERANCE else np.nan,
        "ess_strat": float(np.nanmean(ess_strat)) if c_strat >= COVERAGE_TOLERANCE else np.nan,
        "ess_ps": float(np.nanmean(ess_ps)),
        "cov_glob": c_glob, "cov_strat": c_strat,
    }
    return rows, pop


def summarise(long_df: pd.DataFrame, pop_df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset x dimension, averaged over subgroups, models and methods."""
    out = []
    for (ds, dim), sub in long_df.groupby(["dataset", "dimension"], sort=False):
        per_g = sub.groupby("subgroup")[["rmse_glob", "rmse_strat"]].mean()
        p = pop_df[(pop_df["dataset"] == ds) & (pop_df["dimension"] == dim)]
        out.append({
            "dataset": ds, "dimension": dim,
            "bias_glob": sub["bias_glob"].mean(),
            "rmse_glob": sub["rmse_glob"].mean(),
            "rmse_strat": sub["rmse_strat"].mean(),
            "better": f"{int((per_g['rmse_strat'] < per_g['rmse_glob']).sum())}/{len(per_g)}",
            "ess_glob": p["ess_glob"].mean(), "ess_strat": p["ess_strat"].mean(),
            "ess_ps": p["ess_ps"].mean(), "cov_strat": p["cov_strat"].mean(),
        })
    return pd.DataFrame(out)


def main() -> None:
    from soepsim.preprocessing.variables import SUBGROUP_VARS

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-human", type=int, default=100)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shrinkage", type=float, default=0.0)
    p.add_argument("--min-stratum-n", type=int, default=10)
    p.add_argument("--dims", nargs="*", default=list(SUBGROUP_VARS))
    p.add_argument("--only-dataset", nargs="*", default=list(DATASETS))
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "tables")
    args = p.parse_args()

    long_rows, pop_rows = [], []
    for ds in args.only_dataset:
        for dim in args.dims:
            loaded = _load_human_and_predictions(ds, dim)
            if loaded is None:
                continue
            fq, human, base_dir = loaded
            for pred_path in sorted(Path(base_dir).glob("*/*.csv")):
                model, method = pred_path.parent.name, pred_path.stem
                if method not in METHODS:
                    continue
                pred = pd.read_csv(pred_path)
                if "id" not in pred.columns or "answer" not in pred.columns:
                    continue
                pred = pred[["id", "answer"]].rename(columns={"answer": "answer_pred"})
                merged = human.merge(pred, on="id", how="inner")
                y = np.asarray(phi(merged["answer"], fq), dtype=float)
                yhat = np.asarray(phi(merged["answer_pred"], fq), dtype=float)
                groups = merged[dim].to_numpy(dtype=object)

                rows, pop = evaluate_file(y, yhat, groups, args.n_human, args.n_trials,
                                          args.seed, args.shrinkage, args.min_stratum_n)
                key = {"dataset": DATASETS[ds], "dimension": dim,
                       "model": model, "method": METHODS[method]}
                long_rows += [{**key, **r} for r in rows]
                if pop is not None:
                    pop_rows.append({**key, **pop})
                print(f"  {ds:8s} {dim:16s} {model:10s} {method}")

    if not long_rows:
        print("No results -- run this where datasets/*/data/human_test.csv and the "
              "simulated outputs are available.")
        return

    long_df, pop_df = pd.DataFrame(long_rows), pd.DataFrame(pop_rows)
    summary = summarise(long_df, pop_df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(args.out_dir / "extension_b_rmse_long.csv", index=False)
    pop_df.to_csv(args.out_dir / "extension_b_rmse_population.csv", index=False)
    summary.to_csv(args.out_dir / "extension_b_rmse_summary.csv", index=False)

    print("\n" + summary.round(2).to_string(index=False))
    print(f"\nWritten to {args.out_dir}")


if __name__ == "__main__":
    main()