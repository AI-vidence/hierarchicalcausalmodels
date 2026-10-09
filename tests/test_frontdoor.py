import numpy as np
import pytest

from hierarchicalcausalmodels.utils.binning import quantile_bins
from hierarchicalcausalmodels.utils.frontdoor import frontdoor_effect


def test_bins_never_empty_and_contiguous_with_outliers():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(size=995), [12, 13, 14, 15, 16]]
    codes, cuts, means = quantile_bins(x, max_bins=5)
    counts = np.bincount(codes)
    assert set(codes) == set(range(len(counts))) and counts.min() > 0
    # the former equal-width scheme left empty bins on exactly this input
    edges = np.linspace(x.min(), x.max(), 6)
    old = np.digitize(x, edges[:-1]) - 1
    assert (np.bincount(old, minlength=5) == 0).any()


def test_binary_variable_binned_exactly():
    x = np.r_[np.zeros(30), np.ones(70)]
    codes, cuts, means = quantile_bins(x, max_bins=10)
    assert np.array_equal(codes, x.astype(int)) and np.allclose(means, [0, 1])


def _confounded(n, rng):
    u = rng.normal(size=n)  # unobserved confounder of T and Y
    t = (u + rng.normal(size=n) > 0).astype(float)
    m = (rng.random(n) < np.where(t == 1, 0.8, 0.2)).astype(float)  # T -> M only
    y = 2.0 * m + 3.0 * u + rng.normal(size=n)  # M -> Y, U -> Y
    return t, m, y


def test_frontdoor_recovers_true_effect_under_confounding():
    rng = np.random.default_rng(1)
    t, m, y = _confounded(200_000, rng)
    true_effect = 2.0 * (0.8 - 0.2)  # = 1.2
    naive = y[t == 1].mean() - y[t == 0].mean()
    fd = frontdoor_effect(t, m, y)["effect"]
    assert abs(fd - true_effect) < 0.05
    assert abs(naive - true_effect) > 1.0  # the naive contrast is badly biased


def test_positivity_violation_is_reported():
    t = np.r_[np.zeros(50), np.ones(50)]
    m = t.copy()  # M is a deterministic copy of T: some (m, t') cells are empty
    y = m + np.random.default_rng(2).normal(size=100)
    with pytest.raises(ValueError, match="positivity"):
        frontdoor_effect(t, m, y)


def test_frontdoor_adjustment_is_a_method():
    from hierarchicalcausalmodels.models import HSCMParametric

    assert callable(getattr(HSCMParametric, "frontdoor_adjustment", None))
