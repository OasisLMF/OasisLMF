"""Tests for the interpolated lookup in get_corr_rval (shared by gulpy and gulmc)."""

import os
import subprocess
import sys
from textwrap import dedent

import numpy as np
import pytest
from scipy.stats import norm

from oasislmf.pytools.gul.random import (
    _interpolate_lookup,
    get_corr_rval,
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


def test_corr_rval_tail_accuracy():
    """Tail values should match exact scipy within 1e-8."""
    x_unif = np.array([0.001, 0.999, 0.01])
    y_unif = np.array([0.999, 0.001, 0.99])
    z_unif = np.zeros(3)
    rho = 0.5

    get_corr_rval(x_unif, y_unif, rho, x_min, _norm_inv_cdf, inv_factor,
                  cdf_min, _norm_cdf, norm_factor, 3, z_unif)

    sqrt_rho = np.sqrt(rho)
    sqrt_1_minus_rho = np.sqrt(1.0 - rho)
    z_exact = np.array([
        norm.cdf(sqrt_rho * norm.ppf(x) + sqrt_1_minus_rho * norm.ppf(y))
        for x, y in zip(x_unif, y_unif)
    ])

    np.testing.assert_allclose(z_unif, z_exact, atol=1e-8)


def test_corr_rval_middle_reasonable():
    """Middle values use fast lookup, should still be within 1e-4."""
    x_unif = np.array([0.1, 0.5, 0.9])
    y_unif = np.array([0.9, 0.5, 0.1])
    z_unif = np.zeros(3)
    rho = 0.5

    get_corr_rval(x_unif, y_unif, rho, x_min, _norm_inv_cdf, inv_factor,
                  cdf_min, _norm_cdf, norm_factor, 3, z_unif)

    sqrt_rho = np.sqrt(rho)
    sqrt_1_minus_rho = np.sqrt(1.0 - rho)
    z_exact = np.array([
        norm.cdf(sqrt_rho * norm.ppf(x) + sqrt_1_minus_rho * norm.ppf(y))
        for x, y in zip(x_unif, y_unif)
    ])

    np.testing.assert_allclose(z_unif, z_exact, atol=1e-4)


def test_corr_rval_keeps_sqrt_rho_at_full_width_under_oasis_float_f8(tmp_path):
    """With OASIS_FLOAT=f8, sqrt(rho) must not be rounded to float32 before correlating.

    Runs with its own NUMBA_CACHE_DIR: the on-disk cache does not key on oasis_float, so an f4 build of the kernel would be reused.
    """
    script = dedent("""
        from math import sqrt

        import numpy as np

        from oasislmf.pytools.gul import random as r

        assert r.oasis_float_type is np.float64
        norm_inv_cdf = r.compute_norm_inv_cdf_lookup(r.x_min, r.x_max, r.norm_inv_N)
        norm_cdf = r.compute_norm_cdf_lookup(r.cdf_min, r.cdf_max, r.norm_inv_N)
        N = len(norm_inv_cdf)
        x_unif = np.array([0.0005, 0.9995, 0.3, 0.7])
        y_unif = np.array([0.9995, 0.0005, 0.6, 0.2])
        rho = np.float64(0.3)
        z_unif = np.zeros(len(x_unif))
        r.get_corr_rval(x_unif, y_unif, rho, r.x_min, norm_inv_cdf, r.inv_factor, r.cdf_min,
                        norm_cdf, r.norm_factor, len(x_unif), z_unif)

        def to_norm(u):
            if u < r.epsilon or u > 1.0 - r.epsilon:
                return r._interpolate_lookup(u, r.x_min, r.inv_factor, norm_inv_cdf, N)
            return r._fast_lookup(u, r.x_min, r.inv_factor, norm_inv_cdf, N)

        expected = np.array([
            r._interpolate_lookup(sqrt(rho) * to_norm(x) + sqrt(1.0 - rho) * to_norm(y), r.cdf_min, r.norm_factor, norm_cdf, N)
            for x, y in zip(x_unif, y_unif)
        ])
        np.testing.assert_allclose(z_unif, expected, rtol=1e-12, atol=0)
        """)
    env = {**os.environ, "OASIS_FLOAT": "f8", "NUMBA_CACHE_DIR": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
