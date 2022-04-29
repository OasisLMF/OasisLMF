"""
This file defines the data types that are loaded from the data files.

"""
import numpy as np
import numba as nb

from oasislmf.pytools.getmodel.common import oasis_float, areaperil_int

ITEM_ID_TYPE = nb.types.int32
ITEMS_DATA_MAP_TYPE = nb.types.UniTuple(nb.types.int64, 2)
COVERAGE_ID_TYPE = nb.types.int32

# negative sidx (definition)
MEAN_IDX = -1
STD_DEV_IDX = -2
TIV_IDX = -3
CHANCE_OF_LOSS_IDX = -4
MAX_LOSS_IDX = -5

NUM_IDX = 5

# negative sidx + NUM_IDX
SHIFTED_MEAN_IDX = MEAN_IDX + NUM_IDX
SHIFTED_STD_DEV_IDX = STD_DEV_IDX + NUM_IDX
SHIFTED_TIV_IDX = TIV_IDX + NUM_IDX
SHIFTED_CHANCE_OF_LOSS_IDX = CHANCE_OF_LOSS_IDX + NUM_IDX
SHIFTED_MAX_LOSS_IDX = MAX_LOSS_IDX + NUM_IDX


BIN_MAP_KEY_TYPE = nb.types.Tuple((oasis_float, oasis_float))
ITEM_MAP_KEY_TYPE = nb.types.Tuple((nb.types.uint32, nb.types.int32))
ITEM_MAP_VALUE_TYPE = nb.types.UniTuple(nb.types.int32, 3)

# compute the relative size of oasis_float vs int32
oasis_float_to_int32_size = oasis_float.itemsize // np.int32().itemsize

ProbMean = nb.from_dtype(np.dtype([('prob_to', oasis_float),
                                   ('bin_mean', oasis_float)
                                   ]))

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
