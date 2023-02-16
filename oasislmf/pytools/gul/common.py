"""
This file defines the data types that are loaded from the data files.

"""
import numba as nb
import numpy as np

from oasislmf.pytools.common import areaperil_int, oasis_float

# gul stream type
# probably need to set this dynamically depending on the stream type
gul_header = np.int32(1 | 2 << 24).tobytes()

items_data_type = nb.from_dtype(np.dtype([('item_id', np.int32),
                                          ('damagecdf_i', np.int32),
                                          ('rng_index', np.int32)
                                          ]))

items_MC_data_type = nb.from_dtype(np.dtype([('item_id', np.int32),
                                             ('areaperil_id', areaperil_int),
                                             ('vulnerability_id', np.int32),
                                             ('hazcdf_i', np.int32),
                                             ('rng_index', np.int32),
                                             ('hazard_rng_index', np.int32),
                                             ('eff_vuln_cdf_i', np.int32),
                                             ('eff_vuln_cdf_Ndamage_bins', np.int32)
                                             ]))

VulnCdfLookup = nb.from_dtype(np.dtype([('start', np.int32), ('length', np.int32)]))


coverage_type = nb.from_dtype(np.dtype([('tiv', np.float64),
                                        ('max_items', np.int32),
                                        ('start_items', np.int32),
                                        ('cur_items', np.int32)
                                        ]))

NP_BASE_ARRAY_SIZE = 8

# negative sidx (definition)
MEAN_IDX = -1
STD_DEV_IDX = -2
TIV_IDX = -3
CHANCE_OF_LOSS_IDX = -4
MAX_LOSS_IDX = -5

NUM_IDX = 5

ITEM_MAP_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int32))
ITEM_MAP_VALUE_TYPE = nb.types.UniTuple(nb.types.int32, 3)
AREAPERIL_TO_EFF_VULN_KEY_TYPE = nb.types.Tuple((nb.from_dtype(areaperil_int), nb.types.int64))
AREAPERIL_TO_EFF_VULN_VALUE_TYPE = nb.types.UniTuple(nb.types.int32, 2)

# compute the relative size of oasis_float and areaperil_int vs int32
oasis_float_to_int32_size = oasis_float.itemsize // np.int32().itemsize
areaperil_int_to_int32_size = areaperil_int.itemsize // np.int32().itemsize

haz_cdf_type = nb.from_dtype(np.dtype([('probability', oasis_float),
                                       ('intensity_bin_id', np.int32)]))

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
