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
parquetfootprint_filename = "footprint.parquet"
parquetfootprint_meta_filename = "footprint_parquet_meta.json"

areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
nb_areaperil_int = nb.from_dtype(areaperil_int)
oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))

FootprintHeader = nb.from_dtype(np.dtype([('num_intensity_bins', np.int32),
                                          ('has_intensity_uncertainty', np.int32)
                                          ]))

Event = nb.from_dtype(np.dtype([('areaperil_id', areaperil_int),
                                ('intensity_bin_id', np.int32),
                                ('probability', oasis_float)
                                ]))

EventCSV = nb.from_dtype(np.dtype([('event_id', np.int32),
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

Item = nb.from_dtype(np.dtype([('id', np.int32),
                               ('coverage_id', np.int32),
                               ('areaperil_id', areaperil_int),
                               ('vulnerability_id', np.int32),
                               ('group_id', np.int32)
                               ]))

Keys = {'LocID': np.int32,
        'PerilID': 'category',
        'CoverageTypeID': np.int32,
        'AreaPerilID': areaperil_int,
        'VulnerabilityID': np.int32}

Correlation = nb.from_dtype(np.dtype([
    ("peril_correlation_group", np.int32),
    ("correlation_value", oasis_float)
]))
