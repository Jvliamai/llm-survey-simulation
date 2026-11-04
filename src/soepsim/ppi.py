"""
Prediction-Powered Inference: point estimate, confidence interval, variance
and the power-tuning parameter lambda.
"""

from __future__ import annotations

import numpy as np
from ppi_py import ppi_mean_ci as _ppi_mean_ci_upstream
from ppi_py import ppi_mean_pointestimate
from ppi_py.ppi import _calc_lam_glm
from ppi_py.utils import construct_weight_vector, reshape_to_2d

__all__ = ["ppi_mean_ci_with_var", "ppi_point_and_ci", "human_only_variance"]


def ppi_mean_ci_with_var(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    alpha: float = 0.05,
    lam: float | None = None,
) -> tuple[tuple[float, float], float, float]:
    """
    PPI confidence interval with estimator variance and lambda.

    :param Y: human answers in H, shape (n,)
    :param Yhat: predictions in H, shape (n,)
    :param Yhat_unlabeled: predictions in U, shape (N,)
    :param alpha: significance level (0.05 = 95% interval)
    :param lam: None = power-tuned, 1.0 = lambda fixed at 1
    :return: ((lo, hi), variance, lam_used)
    """
    Y = np.asarray(Y, dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    Yhat_unlabeled = np.asarray(Yhat_unlabeled, dtype=float)
    n, N = len(Y), len(Yhat_unlabeled)

    Y2 = reshape_to_2d(Y)
    Yhat2 = reshape_to_2d(Yhat)
    Yhat_u2 = reshape_to_2d(Yhat_unlabeled)
    w = construct_weight_vector(n, None, vectorized=True)
    w_u = construct_weight_vector(N, None, vectorized=True)

    if lam is None:
        base = (w_u * Yhat_u2).mean(0) + (w * (Y2 - Yhat2)).mean(0)
        grads = w * (Y2 - base)
        grads_hat = w * (Yhat2 - base)
        grads_hat_u = w_u * (Yhat_u2 - base)
        lam_used = float(
            _calc_lam_glm(
                grads, grads_hat, grads_hat_u, np.eye(1),
                coord=None, clip=True, optim_mode="overall",
            )
        )
    else:
        lam_used = float(lam)

    lo, hi = _ppi_mean_ci_upstream(Y, Yhat, Yhat_unlabeled, alpha=alpha, lam=lam_used)
    ci = (float(np.asarray(lo).squeeze()), float(np.asarray(hi).squeeze()))

    # Variance decomposition, equation (2): synthetic term + rectifier term.
    imputed_std = (w_u * (lam_used * Yhat_u2)).std(0) / np.sqrt(N)
    rectifier_std = (w * (Y2 - lam_used * Yhat2)).std(0) / np.sqrt(n)
    ppi_var = float((imputed_std**2 + rectifier_std**2)[0])

    return ci, ppi_var, lam_used


def ppi_point_and_ci(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
    alpha: float = 0.05,
    lam: float | None = None,
) -> dict:
    """
    Convenience wrapper returning al results for one H/U split.

    :return: dict with ``theta_hat``, ``ci_lo``, ``ci_hi``, ``ci_width``,
        ``var``, ``lam``, ``n_labeled``, ``n_unlabeled``
    """
    (lo, hi), var, lam_used = ppi_mean_ci_with_var(Y, Yhat, Yhat_unlabeled, alpha, lam)
    theta = float(
        np.asarray(ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=lam_used)).squeeze()
    )
    return {
        "theta_hat": theta,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_width": hi - lo,
        "var": var,
        "lam": lam_used,
        "n_labeled": int(len(Y)),
        "n_unlabeled": int(len(Yhat_unlabeled)),
    }


def human_only_variance(Y: np.ndarray) -> float:
    """
    Variance of the human-only mean estimator, the denominator of the ESS gain.
    """
    Y = np.asarray(Y, dtype=float)
    Y = Y[~np.isnan(Y)]
    if len(Y) <= 1:
        return float("nan")
    return float(np.var(Y, ddof=1) / len(Y))
