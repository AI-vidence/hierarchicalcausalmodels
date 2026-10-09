"""Front-door adjustment estimated from samples (Pearl 1995).

    E[Y | do(T=t)] = sum_m P(m | t) * sum_t' E[Y | m, t'] P(t')

T and M are discretized with :func:`quantile_bins` (exact for binary or
few-valued variables, no empty bin otherwise); Y is used as is (cell means),
so no information is lost on the outcome.
"""
from __future__ import annotations

import numpy as np

from hierarchicalcausalmodels.utils.binning import quantile_bins


def frontdoor_effect(t, m, y, max_bins: int = 10, min_cell: int = 1):
    """Return dict with E[Y|do(T=bin k)] for each T bin and the contrast top - bottom.

    Raises ValueError when a (m, t') cell needed by the formula is empty
    (positivity violated): the effect is then not estimable from these data.
    """
    t, m, y = (np.asarray(a, dtype=float).ravel() for a in (t, m, y))
    if not (len(t) == len(m) == len(y)):
        raise ValueError("t, m, y must have the same length")
    tc, _, t_means = quantile_bins(t, max_bins)
    mc, _, _ = quantile_bins(m, max_bins)
    kt, km = tc.max() + 1, mc.max() + 1
    n = len(y)
    p_t = np.bincount(tc, minlength=kt) / n
    count_mt = np.zeros((km, kt))
    sum_mt = np.zeros((km, kt))
    np.add.at(count_mt, (mc, tc), 1.0)
    np.add.at(sum_mt, (mc, tc), y)
    needed = count_mt.sum(axis=1) > 0  # m values that occur
    if np.any(count_mt[needed] < min_cell):
        raise ValueError("positivity violated: some (M, T) cell is empty; front-door effect not estimable")
    ey_mt = np.divide(sum_mt, count_mt, out=np.zeros_like(sum_mt), where=count_mt > 0)
    inner = ey_mt @ p_t  # sum_t' E[Y|m,t'] P(t'), one value per m
    p_m_given_t = count_mt / count_mt.sum(axis=0, keepdims=True)  # columns = t
    ey_do_t = p_m_given_t.T @ inner  # one value per T bin
    return {
        "t_values": t_means,
        "ey_do_t": ey_do_t,
        "effect": float(ey_do_t[-1] - ey_do_t[0]),
    }
