"""Defines the data types that are loaded from the data files."""
import zlib

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
zvulnerability_filename = 'vulnerability.bin.z'
zvulnerability_index_filename = 'vulnerability.idx.z'
event_defintion_filename = 'event_definition.parquet'
hazard_case_filename = 'hazard_case.parquet'


FootprintHeader = np.dtype([('num_intensity_bins', np.int32),
                            ('has_intensity_uncertainty',
                             np.int32)
                            ])

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

VulnerabilityIndex_dtype = np.dtype([
    ('vulnerability_id', np.int32),
    ('offset', np.int64),
    ('size', np.int64),
    ('original_size', np.int64)
])

VulnerabilityRow_dtype = np.dtype([
    ('intensity_bin_id', np.int32),
    ('damage_bin_id', np.int32),
    ('probability', oasis_float)
])
vuln_offset = 4


def incomplete_zvuln_pair_msg(has_zbin, has_zidx):
    """Describe a zipped vulnerability pair with only one of its two files present, or return None."""
    if has_zbin == has_zidx:
        return None
    present, missing = ((zvulnerability_filename, zvulnerability_index_filename) if has_zbin
                        else (zvulnerability_index_filename, zvulnerability_filename))
    return f"found {present} but not {missing}"


def decompress_vuln_block(bin_data, entry, filename):
    """Decompress the zlib block of one vulnerability.idx.z entry into raw VulnerabilityRow bytes.

    Args:
        bin_data (np.ndarray[uint8]): the whole zipped vulnerability bin file
        entry (VulnerabilityIndex_dtype): the index entry pointing to the block
        filename (str): the bin file name, for error messages

    Returns:
        bytes: the decompressed block, original_size bytes long
    """
    vuln_id = int(entry['vulnerability_id'])
    offset = int(entry['offset'])
    size = int(entry['size'])
    original_size = int(entry['original_size'])
    if original_size % VulnerabilityRow_dtype.itemsize:
        raise Exception(
            f"vulnerability_id {vuln_id} has original_size {original_size} in the index of {filename}, which is not a "
            f"whole number of {VulnerabilityRow_dtype.itemsize}-byte vulnerability rows; the file may have been "
            f"written with a different OASIS_FLOAT")
    try:
        data = zlib.decompress(bin_data[offset:offset + size].tobytes())
    except zlib.error as e:
        raise Exception(
            f"vulnerability_id {vuln_id} failed to decompress ({size} bytes at offset {offset}) in {filename}: {e}") from e
    if len(data) != original_size:
        raise Exception(
            f"vulnerability_id {vuln_id} decompressed to {len(data)} bytes in {filename}, "
            f"expected original_size {original_size} from the index file")
    return data


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
