"""
This file contains general-purpose utilities.
"""
import numpy as np


def assert_allclose(x, y, rtol=1e-10, atol=1e-8, x_name="x", y_name="y"):
    """
    Drop in replacement for `numpy.testing.assert_allclose` that also shows
    the nonmatching elements in a nice human-readable format.

    Args:
        x (np.array or scalar): first input to compare.
        y (np.array or scalar): second input to compare.
        rtol (float, optional): relative tolreance. Defaults to 1e-10.
        atol (float, optional): absolute tolerance. Defaults to 1e-8.
        x_name (str, optional): header to print for x if x and y do not match. Defaults to "x".
        y_name (str, optional): header to print for y if x and y do not match. Defaults to "y".

    Raises:
        AssertionError: if x and y shapes do not match.
        AssertionError: if x and y data do not match.

    """
    if np.isscalar(x) and np.isscalar(y) == 1:
        return np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

    if x.shape != y.shape:
        raise AssertionError("Shape mismatch: %s vs %s" % (str(x.shape), str(y.shape)))

    d = ~np.isclose(x, y, rtol, atol)

    if np.any(d):
        miss = np.where(d)[0]
        msg = f"Mismatch of {len(miss):d} elements ({len(miss) / x.size * 100:g} %) at the level of rtol={rtol:g}, atol={atol:g},\n" \
            f"{repr(miss)}\n" \
            f"x: {x_name}\n{str(x[d])}\n\n" \
            f"y: {y_name}\n{str(y[d])}"\

        raise AssertionError(msg)
