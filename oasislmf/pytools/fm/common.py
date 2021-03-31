import numpy as np
import numba as nb

allowed_allocation_rule = [0, 1, 2, 3]

np_oasis_int = np.int32
nb_oasis_int = nb.int32
np_oasis_float = np.float32

float_equal_precision = np.finfo(np_oasis_float).eps


@nb.njit(cache=True)
def almost_equal(a, b):
    return abs(a - b) < float_equal_precision


null_index = np_oasis_int(-1)
need_tiv_policy = (4, 6, 18, 27, 28, 29, 30, 31, 32)

# financial structure static input dtypes
fm_programme_dtype = np.dtype([('from_agg_id', 'i4'), ('level_id', 'i4'), ('to_agg_id', 'i4')])
fm_policytc_dtype = np.dtype([('level_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4'), ('policytc_id', 'i4')])
fm_profile_dtype = np.dtype([('policytc_id', 'i4'),
                             ('calcrule_id', 'i4'),
                             ('deductible_1', 'f4'),
                             ('deductible_2', 'f4'),
                             ('deductible_3', 'f4'),
                             ('attachment_1', 'f4'),
                             ('limit_1', 'f4'),
                             ('share_1', 'f4'),
                             ('share_2', 'f4'),
                             ('share_3', 'f4'),
                             ])
fm_profile_step_dtype = np.dtype([('policytc_id', 'i4'),
                                  ('calcrule_id', 'i4'),
                                  ('deductible_1', 'f4'),
                                  ('deductible_2', 'f4'),
                                  ('deductible_3', 'f4'),
                                  ('attachment_1', 'f4'),
                                  ('limit_1', 'f4'),
                                  ('share_1', 'f4'),
                                  ('share_2', 'f4'),
                                  ('share_3', 'f4'),
                                  ('step_id', 'i4'),
                                  ('trigger_start', 'f4'),
                                  ('trigger_end', 'f4'),
                                  ('payout_start', 'f4'),
                                  ('payout_end', 'f4'),
                                  ('limit_2', 'f4'),
                                  ('scale_1', 'f4'),
                                  ('scale_2', 'f4'),
                                  ])
fm_profile_csv_col_map = {
                         'deductible_1': 'deductible1',
                         'deductible_2': 'deductible2',
                         'deductible_3': 'deductible3',
                         'attachment_1': 'attachment1',
                         'limit_1': 'limit1',
                         'share_1': 'share1',
                         'share_2': 'share2',
                         'share_3': 'share3',
                         'limit_2':' limit2',
                         'scale_1':'scale1',
                         'scale_2': 'scale2',
                        }
fm_xref_dtype = np.dtype([('output_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4')])
fm_xref_csv_col_map = {'output_id':'output'}

coverages_dtype = np.dtype([('coverage_id', 'i4'), ('tiv', 'f4')])

items_dtype = np.dtype([('item_id', 'i4'),
                        ('coverage_id', 'i4'),
                        ('areaperil_id', 'i4'),
                        ('vulnerability_id', 'i4'),
                        ('group_id', 'i4')])

