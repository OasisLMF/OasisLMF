"""
This file defines the data types that are loaded from the data files.
"""
import os

import numba as nb
import numpy as np

# filenames
footprint_filename = 'footprint.bin'
footprint_index_filename = 'footprint.idx'
zfootprint_filename = 'footprint.bin.z'
zfootprint_index_filename = 'footprint.idx.z'
csvfootprint_filename = 'footprint.csv'

areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))

FootprintHeader = nb.from_dtype(np.dtype([('num_intensity_bins', np.int32),
                                          ('has_intensity_uncertainty', np.int32)
                                          ]))

Event = nb.from_dtype(np.dtype([('areaperil_id', areaperil_int),
                                ('intensity_bin_id', np.int32),
                                ('probability', oasis_float)
                                ]))

EventCSV =  nb.from_dtype(np.dtype([('event_id', np.int32),
                                    ('areaperil_id', areaperil_int),
                                    ('intensity_bin_id', np.int32),
                                    ('probability', oasis_float)
                                    ]))

EventIndexBin = nb.from_dtype(np.dtype([('event_id', np.int32),
                                        ('offset', np.int64),
                                        ('size', np.int64)
                                        ]))

EventIndexBinZ = nb.from_dtype(np.dtype([('event_id', np.int32),
                                         ('offset', np.int64),
                                         ('size', np.int64),
                                         ('d_size', np.int64)
                                         ]))

Index_type = nb.from_dtype(np.dtype([('start', np.int64),
                                     ('end', np.int64)
                                     ]))

Vulnerability = nb.from_dtype(np.dtype([('vulnerability_id', np.int32),
                                        ('intensity_bin_id', np.int32),
                                        ('damage_bin_id', np.int32),
                                        ('probability', oasis_float)
                                        ]))
