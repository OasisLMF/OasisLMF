"""
This file defines the data types that are loaded from the data files.

"""
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import areaperil_int, oasis_int, oasis_float
from oasislmf.pytools.common.event_stream import MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX

items_data_type = nb.from_dtype(np.dtype([('item_id', oasis_int),
                                          ('damagecdf_i', oasis_int),
                                          ('rng_index', oasis_int)
                                          ]))

items_MC_data_type = nb.from_dtype(np.dtype([('item_id', oasis_int),
                                             ('areaperil_id', areaperil_int),
                                             ('vulnerability_id', oasis_int),
                                             ('hazcdf_i', oasis_int),
                                             ('rng_index', oasis_int),
                                             ('hazard_rng_index', oasis_int),
                                             ('eff_vuln_cdf_i', oasis_int),
                                             ('eff_vuln_cdf_Ndamage_bins', oasis_int)
                                             ]))

VulnCdfLookup = nb.from_dtype(np.dtype([('start', oasis_int), ('length', oasis_int)]))


coverage_type = nb.from_dtype(np.dtype([('tiv', np.float64),
                                        ('max_items', oasis_int),
                                        ('start_items', oasis_int),
                                        ('cur_items', oasis_int)
                                        ]))

NP_BASE_ARRAY_SIZE = 8

# Gul stream special sample idx
SPECIAL_SIDX = np.array([MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX], dtype=oasis_int)
NUM_IDX = SPECIAL_SIDX.shape[0]

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
