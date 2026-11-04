"""
Bootstrap evaluation metrics for PPI survey simulation.
 
Computes bias, ESS gain, and coverage by repeatedly drawing random
labeled subsets from the full test set and evaluating PPI (or LLM-only)
estimates against the known population parameter theta*.
"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
from ppi_py import ppi_mean_pointestimate
 
from .ppi import human_only_variance, ppi_mean_ci_with_var
from .stratified import _as_1d, stratified_ppi
 
RNG = np.random.default_rng(42)
 
# Coverage is considered nominal if it does not fall below this. The reference
# implementation targets 0.95; a small tolerance absorbs bootstrap noise.
COVERAGE_TOLERANCE = 0.90
 
 
def reset_rng(seed: int) -> None:
    """Reseed the module-level generator. Call once per evaluation run."""
    global RNG
    RNG = np.random.default_rng(seed)
 
 
# Helpers
def _split(y: np.ndarray, yhat: np.ndarray, labeled_idx: np.ndarray):
    """
    Split into labeled (y_l, yhat_l) and unlabeled (yhat_u), dropping NaN.
 
    The two sets are disjoint by construction.
    """
    mask = np.zeros(len(y), dtype=bool)
    mask[labeled_idx] = True
    y_l, yh_l = y[mask], yhat[mask]
    yh_u = yhat[~mask]
    valid = ~(np.isnan(y_l) | np.isnan(yh_l))
    return y_l[valid], yh_l[valid], yh_u[~np.isnan(yh_u)]
 
 
def _split_with_groups(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray, labeled_idx: np.ndarray):
    """
    Same split as :func:`_split`, but keeps the stratum label aligned with
    each observation on both sides.
    """
    mask = np.zeros(len(y), dtype=bool)
    mask[labeled_idx] = True
    y_l, yh_l, g_l = y[mask], yhat[mask], groups[mask]
    yh_u, g_u = yhat[~mask], groups[~mask]
    valid_l = ~(np.isnan(y_l) | np.isnan(yh_l))
    valid_u = ~np.isnan(yh_u)
    return y_l[valid_l], yh_l[valid_l], g_l[valid_l], yh_u[valid_u], g_u[valid_u]
 
 
def _ess_gain(human_var: float, method_var: float) -> float:
    if method_var is None or method_var <= 0 or np.isnan(method_var):
        return np.nan
    return ((human_var / method_var) - 1.0) * 100.0
 
 
def analytical_ci_width(data: np.ndarray, alpha: float = 0.05) -> float:
    """Normal-approximation CI width, used for the unrectified LLM estimator."""
    valid = data[~np.isnan(data)]
    if len(valid) <= 1:
        return np.nan
    se = np.std(valid, ddof=1) / np.sqrt(len(valid))
    return 2 * 1.96 * se
 
 
def is_valid(coverage: float | None, tolerance: float = COVERAGE_TOLERANCE) -> bool:
    """Whether an ESS figure may be reported for this cell."""
    return coverage is not None and not np.isnan(coverage) and coverage >= tolerance
 
 
def bias_is_insignificant(ci_lo: float | None, ci_hi: float | None) -> bool:
    """Whether the signed-bias interval brackets zero."""
    if ci_lo is None or ci_hi is None:
        return False
    if np.isnan(ci_lo) or np.isnan(ci_hi):
        return False
    return bool(ci_lo <= 0.0 <= ci_hi)
 
 
# Bootstrap evaluation
def trials_metrics(
    y_true: np.ndarray,
    yhat: np.ndarray,
    trial_n: int,
    n_trials: int,
    theta_true: float,
    mode: str = "ppi",
    lam: float | None = None,
    groups: np.ndarray | None = None,
    shrinkage: float = 0.0,
    min_stratum_n: int = 10,
) -> dict:
    """Average bias, ESS gain and coverage over random H/U splits.

    :param y_true: human answers after mapping, shape (N,)
    :param yhat: predictions after mapping, shape (N,)
    :param trial_n: size of H per draw
    :param n_trials: number of draws
    :param theta_true: true population mean
    :param mode: "ppi" (global), "llm" (unrectified) or "stratified" 
    :param lam: None = power-tuned, 1.0 = lambda fixed at 1
    :param groups: subgroup labels, only for mode="stratified"
    :param shrinkage: 0 = per-subgroup lambda, 1 = global lambda
    :param min_stratum_n: smaller subgroups fall back to the global correction
    :return: averaged metrics with bootstrap CIs for absolute and signed bias
    """
    empty = {
        "bias": np.nan, "bias_ci_lo": np.nan, "bias_ci_hi": np.nan,
        "signed_bias": np.nan, "signed_bias_ci_lo": np.nan, "signed_bias_ci_hi": np.nan,
        "ess": np.nan, "coverage": np.nan, "lam": np.nan,
        "theta_hat": np.nan, "ci_width": np.nan, "n_trials_used": 0,
    }
    if mode == "stratified" and groups is None:
        raise ValueError("mode='stratified' requires groups.")
    if len(y_true) <= trial_n:
        return empty
 
    groups_arr = _as_1d(groups) if groups is not None else None
    biases, signed_biases = [], []
    esses, covs, lams, thetas, widths = [], [], [], [], []
 
    for _ in range(n_trials):
        idx = RNG.choice(len(y_true), size=trial_n, replace=False)
 
        if mode == "stratified":
            y_l, yh_l, g_l, yh_u, g_u = _split_with_groups(y_true, yhat, groups_arr, idx)
            if len(y_l) <= 1 or len(yh_u) == 0:
                continue
            try:
                strat = stratified_ppi(
                    y_l, yh_l, yh_u, g_l, g_u,
                    lam=lam, shrinkage=shrinkage, min_stratum_n=min_stratum_n,
                )
            except ValueError:
                continue
            theta_hat = strat["theta_hat"]
            method_var = strat["var"]
            lo, hi = strat["ci"]
            covs.append(1.0 if lo <= theta_true <= hi else 0.0)
            lams.append(strat["lam_global"])
            widths.append(hi - lo)
        else:
            y_l, yh_l, yh_u = _split(y_true, yhat, idx)
            if len(y_l) <= 1 or len(yh_u) == 0:
                continue
 
            if mode == "ppi":
                (lo, hi), method_var, lam_used = ppi_mean_ci_with_var(
                    y_l, yh_l, yh_u, alpha=0.05, lam=lam
                )
                theta_hat = float(
                    np.asarray(ppi_mean_pointestimate(y_l, yh_l, yh_u, lam=lam_used)).squeeze()
                )
                covs.append(1.0 if lo <= theta_true <= hi else 0.0)
                lams.append(lam_used)
                widths.append(hi - lo)
            elif mode == "llm":
                theta_hat = float(np.mean(yh_u))
                method_var = float(np.var(yh_u, ddof=1) / len(yh_u))
                widths.append(analytical_ci_width(yh_u))
                se = np.sqrt(method_var)
                covs.append(
                    1.0 if (theta_hat - 1.96 * se) <= theta_true <= (theta_hat + 1.96 * se) else 0.0
                )
            else:
                raise ValueError(f"mode must be 'ppi', 'llm' or 'stratified', got {mode!r}")
 
        denom = abs(theta_true) + 1e-10
        biases.append(abs(theta_hat - theta_true) / denom * 100)
        signed_biases.append((theta_hat - theta_true) / denom * 100)
        thetas.append(theta_hat)
        esses.append(_ess_gain(human_only_variance(y_l), method_var))
 
    if not biases:
        return empty
 
    b = np.asarray(biases)
    sb = np.asarray(signed_biases)
    return {
        "bias": float(np.mean(b)),
        "bias_ci_lo": float(np.percentile(b, 2.5)),
        "bias_ci_hi": float(np.percentile(b, 97.5)),
        "signed_bias": float(np.mean(sb)),
        "signed_bias_ci_lo": float(np.percentile(sb, 2.5)),
        "signed_bias_ci_hi": float(np.percentile(sb, 97.5)),
        "ess": float(np.nanmean(esses)),
        "coverage": float(np.mean(covs)) if covs else np.nan,
        "lam": float(np.mean(lams)) if lams else np.nan,
        "theta_hat": float(np.mean(thetas)),
        "ci_width": float(np.nanmean(widths)) if widths else np.nan,
        "n_trials_used": len(biases),
    }
 
 
 # Subgroup metrics
def subgroup_metrics(
    y_true: np.ndarray,
    yhat: np.ndarray,
    groups: pd.Series,
    trial_n: int,
    n_trials: int,
    min_group_size: int = 30,
    shrinkage: float = 0.0,
    min_stratum_n: int = 10,
) -> pd.DataFrame:
    """Subgroup bias without, with global and with stratified rectification.

    :param groups: subgroup label per observation (NaN dropped)
    :param min_group_size: smaller groups are flagged as unreliable, not dropped
    :param shrinkage: 0 = per-subgroup lambda, 1 = global lambda
    :param min_stratum_n: smaller subgroups fall back to the global correction
    :return: one row per subgroup; df.attrs holds the maximum disparities
    """
    y_true = np.asarray(y_true, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    groups_arr = np.asarray(pd.Series(groups).reset_index(drop=True))
    rows = []
 
    # Global rectification once per trial, then re-centre. Stratified rectification runs
    # alongside it on the same trial draw, so both numbers in a row come from comparable data.
    shifts = []
    strat_theta_draws: dict = {}
    fallback_counts: dict = {}
    for _ in range(n_trials):
        idx = RNG.choice(len(y_true), size=trial_n, replace=False)
        mask = np.zeros(len(y_true), dtype=bool)
        mask[idx] = True
        y_l, yh_l, g_l = y_true[mask], yhat[mask], groups_arr[mask]
        yh_u, g_u = yhat[~mask], groups_arr[~mask]
        valid_l = ~(np.isnan(y_l) | np.isnan(yh_l))
        y_l, yh_l, g_l = y_l[valid_l], yh_l[valid_l], g_l[valid_l]
        valid_u = ~np.isnan(yh_u)
        yh_u, g_u = yh_u[valid_u], g_u[valid_u]
        if len(y_l) <= 1 or len(yh_u) == 0:
            continue
 
        _, _, lam_used = ppi_mean_ci_with_var(y_l, yh_l, yh_u)
        theta_rec = float(
            np.asarray(ppi_mean_pointestimate(y_l, yh_l, yh_u, lam=lam_used)).squeeze()
        )
        shifts.append(theta_rec - float(np.nanmean(yhat)))
 
        try:
            strat = stratified_ppi(
                y_l, yh_l, yh_u, g_l, g_u,
                lam=None, shrinkage=shrinkage, min_stratum_n=min_stratum_n,
            )
        except ValueError:
            # Rare case (e.g. empty strata in this draw): skip instead of failing.
            strat = None
        if strat is not None:
            for g, theta_g in strat["theta_by_stratum"].items():
                if np.isnan(theta_g):
                    continue
                strat_theta_draws.setdefault(g, []).append(theta_g)
                if g in strat["fallback_strata"]:
                    fallback_counts[g] = fallback_counts.get(g, 0) + 1
    shift = float(np.mean(shifts)) if shifts else 0.0
 
    groups_series = pd.Series(groups).reset_index(drop=True)
    for g, mask in groups_series.groupby(groups_series).groups.items():
        sel = np.zeros(len(y_true), dtype=bool)
        sel[np.asarray(mask)] = True
        yg, yhg = y_true[sel], yhat[sel]
        valid = ~(np.isnan(yg) | np.isnan(yhg))
        yg, yhg = yg[valid], yhg[valid]
        if len(yg) == 0:
            continue
 
        theta_g = float(np.mean(yg))
        denom = abs(theta_g) + 1e-10
        bias_raw = abs(float(np.mean(yhg)) - theta_g) / denom * 100
        bias_rec = abs(float(np.mean(yhg)) + shift - theta_g) / denom * 100
 
        draws = strat_theta_draws.get(g, [])
        if draws:
            bias_strat = abs(float(np.mean(draws)) - theta_g) / denom * 100
            n_fallback = fallback_counts.get(g, 0)
        else:
            bias_strat = np.nan
            n_fallback = 0
 
        rows.append({
            "subgroup": g,
            "n": int(len(yg)),
            "theta_true_group": theta_g,
            "bias_synthesis": bias_raw,
            "bias_global_rect": bias_rec,
            "bias_stratified_rect": bias_strat,
            "abs_change": bias_raw - bias_rec,
            "rel_change_pct": (bias_raw - bias_rec) / (bias_raw + 1e-10) * 100,
            "improved": bias_rec < bias_raw,
            "abs_change_stratified": (bias_raw - bias_strat) if draws else np.nan,
            "rel_change_stratified_pct": (
                (bias_raw - bias_strat) / (bias_raw + 1e-10) * 100 if draws else np.nan
            ),
            "improved_stratified": (bias_strat < bias_raw) if draws else None,
            "reliable": len(yg) >= min_group_size,
            "n_trials_used_stratified": len(draws),
            "n_trials_fallback_to_global_lambda": n_fallback,
        })
 
    df = pd.DataFrame(rows)
    if not df.empty:
        df.attrs["max_disparity"] = float(
            df["bias_global_rect"].max() - df["bias_global_rect"].min()
        )
        if df["bias_stratified_rect"].notna().any():
            df.attrs["max_disparity_stratified"] = float(
                df["bias_stratified_rect"].max() - df["bias_stratified_rect"].min()
            )
    return df
 