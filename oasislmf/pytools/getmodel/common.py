"""
This file defines the data types that are loaded from the data files.
"""
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import areaperil_int, oasis_float

# Footprint file formats in order of priority
fp_format_priorities = [
    'parquet_chunk', 'parquet', 'binZ', 'bin', 'csv', 'parquet_dynamic'
]

# filenames
footprint_filename = 'footprint.bin'
footprint_index_filename = 'footprint.idx'
zfootprint_filename = 'footprint.bin.z'
zfootprint_index_filename = 'footprint.idx.z'
csvfootprint_filename = 'footprint.csv'
parquetfootprint_filename = 'footprint.parquet'
parquetfootprint_chunked_dir = 'footprint_chunk'
parquetfootprint_chunked_lookup = 'footprint_lookup.parquet'
footprint_bin_lookup = 'footprint_lookup.bin'
footprint_csv_lookup = 'footprint_lookup.csv'
parquetfootprint_meta_filename = 'footprint_parquet_meta.json'
event_defintion_filename = 'event_definition.parquet'
hazard_case_filename = 'hazard_case.parquet'


FootprintHeader = nb.from_dtype(np.dtype([('num_intensity_bins', np.int32),
                                          ('has_intensity_uncertainty',
                                           np.int32)
                                          ]))

Event_dtype = np.dtype([
    ('areaperil_id', areaperil_int),
    ('intensity_bin_id', np.int32),
    ('probability', oasis_float)
])
Event = nb.from_dtype(Event_dtype)

EventDynamic = nb.from_dtype(np.dtype([('areaperil_id', areaperil_int),
                                       ('intensity_bin_id', np.int32),
                                       ('intensity', np.int32),
                                       ('probability', oasis_float),
                                       ('return_period', np.int32)
                                       ]))

EventIndexBin_dtype = np.dtype([
    ('event_id', np.int32),
    ('offset', np.int64),
    ('size', np.int64)
])
EventIndexBin = nb.from_dtype(EventIndexBin_dtype)

EventIndexBinZ_dtype = np.dtype([
    ('event_id', np.int32),
    ('offset', np.int64),
    ('size', np.int64),
    ('d_size', np.int64)
])
EventIndexBinZ = nb.from_dtype(EventIndexBinZ_dtype)

Index_type = nb.from_dtype(np.dtype([('start', np.int64),
                                     ('end', np.int64)
                                     ]))

Event_defintion = nb.from_dtype(np.dtype([('section_id', np.int32),
                                          ('return_period', np.int32),
                                          ('rp_from', np.int32),
                                          ('rp_to', np.int32),
                                          ('interpolation', np.int32)
                                          ]))

Hazard_case = nb.from_dtype(np.dtype([('section_id', np.int32),
                                      ('areaperil_id', areaperil_int),
                                      ('return_period', np.int32),
                                      ('intensity', np.int32)
                                      ]))

Keys = {'LocID': np.int32,
        'PerilID': 'category',
        'CoverageTypeID': np.int32,
        'AreaPerilID': areaperil_int,
        'VulnerabilityID': np.int32}
