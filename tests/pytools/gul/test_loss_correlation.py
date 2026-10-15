"""Turning a copula correlation into the correlation it induces between two buildings' losses.

The one-factor Gaussian copula correlates the buildings' UNIFORMS. What the variance of their
sum needs is the correlation between their LOSSES, and the damage curve attenuates one into the
other. Expanding the loss in Hermite polynomials, Mehler's formula makes that exact:
``Cov = sum_k a_k^2 rho^k`` against a marginal variance of ``sum_k a_k^2``.

These pin the two properties that matter -- it never exceeds the copula value, and it reproduces
a direct simulation of the copula -- rather than the closed form itself.
"""
import numpy as np
import pytest
from scipy.stats import norm

from oasislmf.pytools.gul.core import (HERMITE_TERMS, accumulate_hermite_coeffs,
                                       loss_correlation)

# the same inverse-normal table the copula itself inverts through
NORM_INV_CDF, X_MIN, X_MAX, N_TABLE = None, 0.0, 1.0, 0


def _lookup():
    """The engine's inverse-normal table and the two constants that index it."""
    global NORM_INV_CDF, X_MIN, X_MAX, N_TABLE
    if NORM_INV_CDF is None:
        N_TABLE = 1_000_000
        X_MIN, X_MAX = 1e-16, 1 - 1e-16
        NORM_INV_CDF = norm.ppf(np.linspace(X_MIN, X_MAX, N_TABLE))
    return NORM_INV_CDF, X_MIN, (N_TABLE - 1) / (X_MAX - X_MIN)


def _coeffs_and_var(prob_to, bin_mean):
    table, x_min, inv_factor = _lookup()
    coeffs = np.zeros(HERMITE_TERMS)
    accumulate_hermite_coeffs(1.0, prob_to, bin_mean, len(prob_to), 1.0,
                              x_min, inv_factor, table, coeffs)
    width = prob_to - np.concatenate([[0.0], prob_to[:-1]])
    var = (width * bin_mean ** 2).sum() - (width * bin_mean).sum() ** 2
    return coeffs, var


def _simulate(prob_to, bin_mean, rho, n=400_000, seed=3):
    """Correlate two uniforms through the copula and measure the losses' correlation."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    pair = []
    for _ in range(2):
        x = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(n)
        pair.append(bin_mean[np.searchsorted(prob_to, norm.cdf(x))])
    return np.corrcoef(pair[0], pair[1])[0, 1]


CURVES = {
    "smooth ramp": (np.linspace(0.02, 1.0, 50), np.linspace(0.0, 1.0, 50)),
    "point mass at zero": (np.concatenate([[0.9], 0.9 + 0.1 * np.linspace(0.05, 1, 20)]),
                           np.concatenate([[0.0], np.linspace(0.05, 1, 20) ** 2])),
}


@pytest.mark.parametrize("curve", list(CURVES))
@pytest.mark.parametrize("rho", [0.2, 0.5, 0.8])
def test_matches_a_direct_simulation_of_the_copula(curve, rho):
    prob_to, bin_mean = CURVES[curve]
    coeffs, var = _coeffs_and_var(prob_to, bin_mean)
    assert loss_correlation(coeffs, var, rho) == pytest.approx(
        _simulate(prob_to, bin_mean, rho), abs=0.02)


@pytest.mark.parametrize("curve", list(CURVES))
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9, 1.0])
def test_never_exceeds_the_copula_correlation(curve, rho):
    """rho^k <= rho for every k >= 1, so the series can only attenuate. Using the copula value
    in its place is what overstated the variance of a sum."""
    prob_to, bin_mean = CURVES[curve]
    coeffs, var = _coeffs_and_var(prob_to, bin_mean)
    r = loss_correlation(coeffs, var, rho)
    assert 0.0 <= r <= rho + 1e-12


def test_a_linear_curve_is_not_attenuated():
    """Attenuation comes from the curve's non-linearity: a loss linear in the normal keeps only
    the k=1 term, where the series is rho exactly."""
    x = np.linspace(-3.5, 3.5, 400)
    prob_to = norm.cdf(x)
    prob_to[-1] = 1.0
    coeffs, var = _coeffs_and_var(prob_to, x)
    for rho in (0.3, 0.7):
        assert loss_correlation(coeffs, var, rho) == pytest.approx(rho, abs=0.02)


def test_degenerate_inputs():
    prob_to, bin_mean = CURVES["smooth ramp"]
    coeffs, var = _coeffs_and_var(prob_to, bin_mean)
    assert loss_correlation(coeffs, var, 0.0) == 0.0        # no copula, no correlation
    assert loss_correlation(coeffs, 0.0, 0.5) == 0.0        # no spread to correlate
    assert loss_correlation(np.zeros(HERMITE_TERMS), var, 0.5) == 0.0


def test_independent_noise_dilutes_it():
    """Where the loss also depends on something drawn independently per building -- the hazard
    intensity in a full Monte Carlo run -- that noise raises the marginal variance without
    adding covariance, so the correlation has to fall."""
    prob_to, bin_mean = CURVES["smooth ramp"]
    coeffs, var = _coeffs_and_var(prob_to, bin_mean)
    correlated = loss_correlation(coeffs, var, 0.6)
    diluted = loss_correlation(coeffs, var * 2.0, 0.6)
    assert diluted == pytest.approx(correlated / 2.0, rel=1e-9)
