import numba as nb
import numpy as np

from oasislmf.pytools.common.data import areaperil_int, oasis_float, oasis_int

# gul stream type
# probably need to set this dynamically depending on the stream type
gul_header = np.int32(1 | 2 << 24).tobytes()

# define the damage_bin_dict damage_types
DAMAGE_TYPE_DEFAULT = 0
DAMAGE_TYPE_RELATIVE = 1
DAMAGE_TYPE_ABSOLUTE = 2
DAMAGE_TYPE_DURATION = 3
VALID_DAMAGE_TYPE = {DAMAGE_TYPE_DEFAULT, DAMAGE_TYPE_RELATIVE, DAMAGE_TYPE_ABSOLUTE, DAMAGE_TYPE_DURATION}

ItemAdjustment = nb.from_dtype(np.dtype([('item_id', np.int32),
                                         ('intensity_adjustment', np.int32),
                                         ('return_period', np.int32)
                                         ]))

items_data_type = nb.from_dtype(np.dtype([('item_id', oasis_int),
                                          ('damagecdf_i', oasis_int),
                                          ('rng_index', oasis_int)
                                          ]))

items_MC_data_type = nb.from_dtype(np.dtype([('item_id', oasis_int),
                                             ('item_idx', oasis_int),
                                             ('haz_arr_i', oasis_int),
                                             ('rng_index', oasis_int),
                                             ('hazard_rng_index', oasis_int),
                                             ('intensity_adjustment', oasis_int),
                                             ('return_period', oasis_int)
                                             ]))

VulnCdfLookup = nb.from_dtype(np.dtype([('start', oasis_int),
                                        ('length', oasis_int)]))

coverage_type = nb.from_dtype(np.dtype([('tiv', np.float64),
                                        ('max_items', np.int32),
                                        ('start_items', np.int32),
                                        ('cur_items', np.int32)
                                        ]))

NP_BASE_ARRAY_SIZE = 8


ITEM_MAP_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int32))
# ITEM_MAP_VALUE_TYPE = nb.types.UniTuple(nb.types.int32, 1)
ITEM_MAP_VALUE_TYPE = nb.types.int64
AREAPERIL_TO_EFF_VULN_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int64))
AREAPERIL_TO_EFF_VULN_VALUE_TYPE = nb.types.UniTuple(nb.types.int32, 2)

# compute the relative size of oasis_float and areaperil_int vs int32
oasis_float_to_int32_size = oasis_float.itemsize // np.int32().itemsize
areaperil_int_to_int32_size = areaperil_int.itemsize // np.int32().itemsize

haz_arr_type = nb.from_dtype(np.dtype([('probability', oasis_float),
                                       ('intensity_bin_id', np.int32),
                                       ('intensity', np.int32)]))

ProbMean = nb.from_dtype(np.dtype([('prob_to', oasis_float),
                                   ('bin_mean', oasis_float)
                                   ]))
ProbMean_size = ProbMean.size

damagecdfrec_stream = nb.from_dtype(np.dtype([('event_id', np.int32),
                                              ('areaperil_id', areaperil_int),
                                              ('vulnerability_id', np.int32)
                                              ]))

damagecdfrec = nb.from_dtype(np.dtype([('areaperil_id', areaperil_int),
                                       ('vulnerability_id', np.int32)
                                       ]))


gulSampleslevelHeader = nb.from_dtype(np.dtype([('event_id', 'i4'),
                                                ('item_id', 'i4'),
                                                ]))
gulSampleslevelHeader_size = gulSampleslevelHeader.size

gulSampleslevelRec = nb.from_dtype(np.dtype([('sidx', 'i4'),
                                             ('loss', oasis_float),
                                             ]))
gulSampleslevelRec_size = gulSampleslevelRec.size


Keys = {'LocID': np.int32,
        'PerilID': 'category',
        'CoverageTypeID': np.int32,
        'AreaPerilID': areaperil_int,
        'VulnerabilityID': np.int32}


NormInversionParameters = nb.from_dtype(np.dtype([('x_min', np.float64),
                                                  ('x_max', np.float64),
                                                  ('N', np.int32),
                                                  ('cdf_min', np.float64),
                                                  ('cdf_max', np.float64),
                                                  ('inv_factor', np.float64),
                                                  ('norm_factor', np.float64),
                                                  ]))

gulmc_compute_info_type = nb.from_dtype(np.dtype([
    ('event_id', oasis_int),
    ('cursor', oasis_int),
    ('coverage_i', oasis_int),  # last_processed_coverage_ids_idx
    ('coverage_n', oasis_int),
    ('next_cached_vuln_cdf_i', oasis_int),
    ('max_bytes_per_item', oasis_int),
    ('Ndamage_bins_max', oasis_int),
    ('loss_threshold', oasis_float),
    ('alloc_rule', np.int8),
    ('do_correlation', np.int8),
    ('do_haz_correlation', np.int8),
    ('effective_damageability', np.int8),
    ('debug', np.int8),
]))
