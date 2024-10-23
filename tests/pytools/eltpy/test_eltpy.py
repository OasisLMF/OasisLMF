import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from oasislmf.pytools.elt.manager import read_quantile_get_intervals, quantile_interval_dtype

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets").joinpath("test_eltpy")


def test_read_quantile_get_intervals():
    fp = Path(TESTS_ASSETS_DIR, "quantile.bin")
    sample_size = 100

    intervals_actual = read_quantile_get_intervals(sample_size, fp)
    intervals_expected = np.zeros(6, dtype=quantile_interval_dtype)
    intervals_expected[0] = (0.0, 1, 0.0)
    intervals_expected[1] = (0.2, 20, 0.8)
    intervals_expected[2] = (0.4, 40, 0.6)
    intervals_expected[3] = (0.5, 50, 0.5)
    intervals_expected[4] = (0.75, 75, 0.25)
    intervals_expected[5] = (1.0, 100, 0.0)

    qs_actual = intervals_actual[:]["q"]
    iparts_actual = intervals_actual[:]["integer_part"]
    fparts_actual = intervals_actual[:]["fractional_part"]

    qs_expected = intervals_expected[:]["q"]
    iparts_expected = intervals_expected[:]["integer_part"]
    fparts_expected = intervals_expected[:]["fractional_part"]

    np.testing.assert_array_almost_equal(qs_actual, qs_expected, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(iparts_actual, iparts_expected, decimal=3, verbose=True)
    np.testing.assert_array_almost_equal(fparts_actual, fparts_expected, decimal=3, verbose=True)
