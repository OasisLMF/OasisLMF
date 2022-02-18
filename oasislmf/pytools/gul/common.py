"""
This file defines the data types that are loaded from the data files.
"""
import os

import numba as nb
import numpy as np

from oasislmf.pytools.getmodel.common import oasis_float, areaperil_int

ProbMean = nb.from_dtype(np.dtype([('prob_to', oasis_float),
                                   ('bin_mean', oasis_float)
                                   ]))

damagecdfrec = nb.from_dtype(np.dtype([('event_id', np.int32),
                                       ('areaperil_id', areaperil_int),
                                       ('vulnerability_id', np.int32)
                                       ]))

# unused
Item_map_rec = nb.from_dtype(np.dtype([('item_id', np.int32),
                                   ('coverage_id', np.int32),
                                   ('group_id', np.int32)
                                   ]))

gulSampleslevelHeader = nb.from_dtype(np.dtype([('event_id', 'i4'),
                                                ('item_id', 'i4'),
                                                ]))

gulSampleslevelRec = nb.from_dtype(np.dtype([('sidx', 'i4'),
                                             ('loss', oasis_float),
                                             ]))


gulSampleFullRecord = nb.from_dtype(np.dtype([('event_id', 'i4'),
                                              ('item_id', 'i4'),
                                              ('sidx', 'i4'),
                                              ('loss', oasis_float),
                                              ]))
