"""Stratified rectification"""
 
from __future__ import annotations
 
import numpy as np
import pandas as pd
from ppi_py import ppi_mean_pointestimate
from scipy.stats import norm
 
from .ppi import ppi_mean_ci_with_var
 
__all__ = ["stratified_ppi", "fairness_metrics"]
 
 
def _as_1d(x) -> np.ndarray:
    return np.asarray(pd.Series(x).reset_index(drop=True))
 
 
def stratified_ppi(
    y_labeled: np.ndarray,
    yhat_labeled: np.ndarray,
    yhat_unlabeled: np.ndarray,
    strata_labeled,
    strata_unlabeled,
    lam: float | None = None,
    shrinkage: float = 0.0,
    min_stratum_n: int = 10,
    alpha: float = 0.05,
) -> dict:
    """
    Stratified PPI: one correction and lambda per stratum, weighted by the
    stratum shares in U.

    :param y_labeled, yhat_labeled: human answers and predictions in H
    :param yhat_unlabeled: predictions in U
    :param strata_labeled, strata_unlabeled: stratum labels in H and U (NaN dropped)
    :param lam: None = power-tuned, otherwise fixed global lambda
    :param shrinkage: 0 = per-stratum lambda, 1 = global lambda
    :param min_stratum_n: smaller strata fall back to the global correction
    :param alpha: significance level (0.05 = 95% interval)
    :return: dict with theta_hat, ci, var, lambdas and per-stratum results
    :raises ValueError: on length mismatch or if no stratum occurs in U
    """
    y_labeled = np.asarray(y_labeled, dtype=float)
    yhat_labeled = np.asarray(yhat_labeled, dtype=float)
    yhat_unlabeled = np.asarray(yhat_unlabeled, dtype=float)
    strata_labeled_full = _as_1d(strata_labeled)
    strata_unlabeled_full = _as_1d(strata_unlabeled)
 
    if len(strata_labeled_full) != len(y_labeled):
        raise ValueError(
            f"strata_labeled has {len(strata_labeled_full)} entries but "
            f"y_labeled has {len(y_labeled)}."
        )
    if len(strata_unlabeled_full) != len(yhat_unlabeled):
        raise ValueError(
            f"strata_unlabeled has {len(strata_unlabeled_full)} entries but "
            f"yhat_unlabeled has {len(yhat_unlabeled)}."
        )
 
    # Drop observations whose stratum is missing
    valid_l = pd.notna(strata_labeled_full)
    valid_u = pd.notna(strata_unlabeled_full)
    y_labeled, yhat_labeled = y_labeled[valid_l], yhat_labeled[valid_l]
    strata_labeled_full = strata_labeled_full[valid_l]
    yhat_unlabeled = yhat_unlabeled[valid_u]
    strata_unlabeled_full = strata_unlabeled_full[valid_u]
    _, _, lam_global = ppi_mean_ci_with_var(
        y_labeled, yhat_labeled, yhat_unlabeled, alpha=alpha, lam=lam
    )
 
    N = len(strata_unlabeled_full)
    strata = list(pd.unique(strata_unlabeled_full))
 
    theta_by_stratum: dict = {}
    var_by_stratum: dict = {}
    lam_by_stratum: dict = {}
    weights: dict = {}
    n_labeled_by_stratum: dict = {}
    n_unlabeled_by_stratum: dict = {}
    fallback_strata: list = []
 
    for g in strata:
        mask_l = strata_labeled_full == g
        mask_u = strata_unlabeled_full == g
        n_g = int(mask_l.sum())
        N_g = int(mask_u.sum())
        weights[g] = (N_g / N) if N else np.nan
        n_labeled_by_stratum[g] = n_g
        n_unlabeled_by_stratum[g] = N_g
 
        if N_g == 0:
            theta_by_stratum[g] = np.nan
            var_by_stratum[g] = np.nan
            lam_by_stratum[g] = np.nan
            continue
 
        yhat_u_g = yhat_unlabeled[mask_u]
 
        if n_g < min_stratum_n:
            # Fallback
            fallback_strata.append(g)
            lam_g = lam_global
            theta_g = float(np.asarray(
                ppi_mean_pointestimate(y_labeled, yhat_labeled, yhat_u_g, lam=lam_g)
            ).squeeze())
            _, var_g, _ = ppi_mean_ci_with_var(
                y_labeled, yhat_labeled, yhat_u_g, alpha=alpha, lam=lam_g
            )
        else:
            y_g = y_labeled[mask_l]
            yhat_g = yhat_labeled[mask_l]
            _, _, lam_g_raw = ppi_mean_ci_with_var(
                y_g, yhat_g, yhat_u_g, alpha=alpha, lam=None
            )
            lam_g = (1.0 - shrinkage) * lam_g_raw + shrinkage * lam_global
            theta_g = float(np.asarray(
                ppi_mean_pointestimate(y_g, yhat_g, yhat_u_g, lam=lam_g)
            ).squeeze())
            _, var_g, _ = ppi_mean_ci_with_var(
                y_g, yhat_g, yhat_u_g, alpha=alpha, lam=lam_g
            )
 
        theta_by_stratum[g] = theta_g
        var_by_stratum[g] = var_g
        lam_by_stratum[g] = lam_g
 
    valid_strata = [g for g in strata if not np.isnan(theta_by_stratum[g])]
    w_sum = sum(weights[g] for g in valid_strata)
    if w_sum <= 0:
        raise ValueError(
            "No stratum in the unlabeled frame carries positive weight -- "
            "cannot form a stratified estimate from this draw."
        )
 
    theta_hat = sum(weights[g] / w_sum * theta_by_stratum[g] for g in valid_strata)
    var_hat = _aggregate_variance(
        y_labeled, yhat_labeled, yhat_unlabeled,
        strata_labeled_full, strata_unlabeled_full,
        valid_strata, fallback_strata,
        {g: weights[g] / w_sum for g in valid_strata},
        lam_by_stratum, lam_global,
    )
    z = float(norm.ppf(1 - alpha / 2))
    se = float(np.sqrt(var_hat)) if var_hat >= 0 else float("nan")
    ci = (theta_hat - z * se, theta_hat + z * se)
 
    return {
        "theta_hat": theta_hat,
        "ci": ci,
        "var": var_hat,
        "lam_global": lam_global,
        "lam_by_stratum": lam_by_stratum,
        "theta_by_stratum": theta_by_stratum,
        "weights": weights,
        "n_labeled_by_stratum": n_labeled_by_stratum,
        "n_unlabeled_by_stratum": n_unlabeled_by_stratum,
        "fallback_strata": fallback_strata,
        "n_labeled_total": int(len(y_labeled)),
        "n_unlabeled_total": int(N),
    }
 
 
def _aggregate_variance(
    y_labeled, yhat_labeled, yhat_unlabeled,
    strata_labeled, strata_unlabeled,
    valid_strata, fallback_strata, w_norm, lam_by_stratum, lam_global,
) -> float:
    """
    Variance of the stratified estimate, accounting for the fact that all
    fallback strata share the same global rectifier computed on all of H.
    """
    n = len(y_labeled)
    fallback = set(fallback_strata)
    own = [g for g in valid_strata if g not in fallback]
    W_F = float(sum(w_norm[g] for g in valid_strata if g in fallback))
 
    # Unlabeled part: strata are disjoint in U, variances add.
    var_u = 0.0
    for g in valid_strata:
        yh_u_g = yhat_unlabeled[np.asarray(strata_unlabeled == g)]
        if len(yh_u_g) > 0:
            var_u += w_norm[g] ** 2 * lam_by_stratum[g] ** 2 * float(np.var(yh_u_g)) / len(yh_u_g)
 
    # Labeled part: own rectifier + share of the shared global rectifier.
    resid_global = y_labeled - lam_global * yhat_labeled
    var_l = 0.0
    in_own = np.zeros(n, dtype=bool)
    for g in own:
        m = np.asarray(strata_labeled == g)
        in_own |= m
        n_g = int(m.sum())
        d = (w_norm[g] * (y_labeled[m] - lam_by_stratum[g] * yhat_labeled[m]) / n_g
             + W_F * resid_global[m] / n)
        var_l += n_g * float(np.var(d))
 
    n_rest = int((~in_own).sum())
    if W_F > 0 and n_rest > 0:
        # 0 or 1 labeled people give no variance estimate -> borrow all of H's
        s2 = float(np.var(resid_global[~in_own])) if n_rest > 1 else float(np.var(resid_global))
        var_l += n_rest * (W_F / n) ** 2 * s2
 
    return float(var_u + var_l)
 
 
def fairness_metrics(subgroup_table: pd.DataFrame) -> dict:
    """Disparity summary per bias column: max disparity, worst subgroup and number of worsened subgroups."""
    out: dict = {
        "n_subgroups": int(len(subgroup_table)),
        "n_reliable": (
            int(subgroup_table["reliable"].sum())
            if "reliable" in subgroup_table.columns
            else None
        ),
    }
    if subgroup_table.empty:
        return out
 
    bias_cols = [c for c in subgroup_table.columns if c.startswith("bias_")]
    if "n" in subgroup_table.columns:
        weights = subgroup_table["n"].astype(float)
    else:
        weights = pd.Series(1.0, index=subgroup_table.index)
    w_total = weights.sum()
    w_norm = weights / w_total if w_total else weights
 
    for col in bias_cols:
        vals = subgroup_table[col]
        if vals.isna().all():
            continue
        max_disparity = float(vals.max() - vals.min())
        worst_subgroup = (
            subgroup_table.loc[vals.idxmax(), "subgroup"]
            if "subgroup" in subgroup_table.columns
            else None
        )
        weighted_mean = float((vals.fillna(0) * w_norm).sum())
        weighted_disparity = float(
            (w_norm * (vals - weighted_mean).abs()).sum(skipna=True)
        )
        entry = {
            "max_disparity": max_disparity,
            "worst_subgroup": worst_subgroup,
            "weighted_mean_bias": weighted_mean,
            "weighted_disparity": weighted_disparity,
        }
        if "bias_synthesis" in subgroup_table.columns and col != "bias_synthesis":
            entry["n_worse_than_synthesis"] = int(
                (vals > subgroup_table["bias_synthesis"]).sum()
            )
        out[col] = entry
 
    return out