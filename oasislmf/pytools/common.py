"""
This file defines quantities reused across the pytools stack.

"""
import os

import numba as nb
import numpy as np

# streams
PIPE_CAPACITY = 65536  # bytes

# data types
oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))
areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
nb_areaperil_int = nb.from_dtype(areaperil_int)
