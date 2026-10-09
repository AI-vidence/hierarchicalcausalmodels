"""Quantile binning without empty bins.

A variable with at most ``max_bins`` distinct values is binned exactly (one
code per value, e.g. a binary treatment stays binary). Otherwise the cut
points are empirical quantiles, so every bin is populated. Codes are always
contiguous 0..n_bins-1.

(Independent copy of the binner used in AI-vidence's TreeRank; the two
libraries are deliberately kept separate.)
"""
from __future__ import annotations

import numpy as np


def quantile_bins(x, max_bins: int = 10):
    """Return (codes, cut_points, bin_means).

    codes      : int array, contiguous in 0..k-1, no empty bin
    cut_points : sorted observed values; code = number of cut points < x
    bin_means  : mean of x in each bin (representative value of the bin)
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        raise ValueError("empty input")
    u = np.unique(x)
    if len(u) <= max_bins:
        cuts = u[:-1]
    else:
        cuts = np.unique(np.quantile(x, np.linspace(0, 1, max_bins + 1)[1:-1], method="inverted_cdf"))
    codes = np.searchsorted(cuts, x, side="left")
    # quantiles of heavily tied data can still leave a gap: re-index to contiguous codes
    present, codes = np.unique(codes, return_inverse=True)
    if len(present) < len(cuts) + 1:
        cuts = cuts[present[:-1]]
    means = np.array([x[codes == k].mean() for k in range(len(present))])
    return codes, cuts, means
