import numpy as np

allowed_allocation_rule = [0, 1, 2, 3]

need_tiv_policy = (4, 6, 18, 27, 28, 29, 30, 31, 32, 37)
need_extras = (7, 8, 10, 11, 13, 19, 26, 27, 35, 36, 107, 108, 110, 111, 113, 119, 126, 135, 136)
EXTRA_VALUES = 3

compute_idx_dtype = np.dtype([
    ('level_start_compute_i', int),
    ('next_compute_i', int),
    ('compute_i', int),
    ('sidx_i', int),
    ('sidx_ptr_i', int),
    ('loss_ptr_i', int),
    ('extras_ptr_i', int),
])

DEDUCTIBLE = 0
OVERLIMIT = 1
UNDERLIMIT = 2
