import numba as nb
import numpy as np


np_oasis_int = np.int32
np_oasis_float = np.float32

nb_oasis_float = nb.float32
nb_oasis_int = nb.int32

float_equal_precision = np.finfo(np_oasis_float).eps


@nb.njit(cache=True)
def almost_equal(a, b):
    return abs(a - b) < float_equal_precision
