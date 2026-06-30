"""Tests for the interpolated lookup in get_corr_rval and get_corr_rval_float."""

import numpy as np
import pytest
from scipy.stats import norm

from oasislmf.pytools.gul.random import (
    _interpolate_lookup,
    get_corr_rval,
    get_corr_rval_float,
    x_min,
    x_max,
    norm_inv_N,
    cdf_min,
    cdf_max,
    inv_factor,
    norm_factor,
    compute_norm_inv_cdf_lookup,
    compute_norm_cdf_lookup,
)

N = norm_inv_N
_norm_inv_cdf = compute_norm_inv_cdf_lookup(x_min, x_max, N)
_norm_cdf = compute_norm_cdf_lookup(cdf_min, cdf_max, N)


@pytest.mark.parametrize("u", [0.001, 0.5, 0.999])
def test_interpolate_lookup_accuracy(u):
    """Lookup should match scipy norm.ppf within 1e-8."""
    result = _interpolate_lookup(u, x_min, inv_factor, _norm_inv_cdf, N)
    exact = norm.ppf(u)
    assert abs(result - exact) < 1e-8


def test_interpolate_lookup_boundary_clamp():
    """Values outside the table range should not crash."""
    assert np.isfinite(_interpolate_lookup(0.0, x_min, inv_factor, _norm_inv_cdf, N))
    assert np.isfinite(_interpolate_lookup(1.0, x_min, inv_factor, _norm_inv_cdf, N))


def test_corr_rval_float_matches_exact():
    """Full pipeline should match exact scipy within 1e-8."""
    x_unif = np.array([0.1, 0.5, 0.999])
    y_unif = np.array([0.9, 0.5, 0.001])
    z_unif = np.zeros(3)
    rho = 0.5

    get_corr_rval_float(x_unif, y_unif, rho, x_min, _norm_inv_cdf, inv_factor,
                        cdf_min, _norm_cdf, norm_factor, 3, z_unif)

    sqrt_rho = np.sqrt(rho)
    sqrt_1_minus_rho = np.sqrt(1.0 - rho)
    z_exact = np.array([
        norm.cdf(sqrt_rho * norm.ppf(x) + sqrt_1_minus_rho * norm.ppf(y))
        for x, y in zip(x_unif, y_unif)
    ])

    np.testing.assert_allclose(z_unif, z_exact, atol=1e-8)


def test_corr_rval_and_float_match():
    """get_corr_rval and get_corr_rval_float should give the same output."""
    rng = np.random.default_rng(42)
    x_unif = rng.uniform(0.001, 0.999, 10)
    y_unif = rng.uniform(0.001, 0.999, 10)
    z1 = np.zeros(10)
    z2 = np.zeros(10)

    get_corr_rval(x_unif, y_unif, 0.5, x_min, x_max, N,
                  _norm_inv_cdf, cdf_min, cdf_max, _norm_cdf, 10, z1)

    get_corr_rval_float(x_unif, y_unif, 0.5, x_min, _norm_inv_cdf, inv_factor,
                        cdf_min, _norm_cdf, norm_factor, 10, z2)

    np.testing.assert_allclose(z1, z2, atol=1e-10)
