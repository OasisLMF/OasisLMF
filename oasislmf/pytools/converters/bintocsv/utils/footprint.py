import zlib
import numpy as np
import pandas as pd
import re
import sys

from oasislmf.pytools.common.data import resolve_file, write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import Event, EventIndexBin, EventIndexBinZ, FootprintHeader


def _check_event_from_to(event_from_to):
    from_event = -1
    to_event = -1
    if event_from_to == None:
        return True, from_event, to_event

    regex_match = re.fullmatch(r'(\d+)-(\d+)', event_from_to)
    if not regex_match:
        raise ValueError(f"Invalid format for event_from_to string: {event_from_to}. String must be of format \"[int1]-[int2]\"")

    from_event, to_event = map(int, regex_match.groups())
    if from_event > to_event:
        raise ValueError(f"Invalid event range: {from_event} > {to_event}")

    return False, from_event, to_event


def _read_footprint_zips(stack, file_in, idx_file_in):
    footprint_file = resolve_file(file_in, mode="rb", stack=stack)

    if footprint_file == sys.stdin.buffer:
        footprint = np.frombuffer(footprint_file.read(), dtype="u1")
    else:
        footprint = np.fromfile(footprint_file, dtype="u1")

    footprint_header = np.frombuffer(footprint[:FootprintHeader.size].tobytes(), dtype=FootprintHeader)

    uncompressedMask = 1 << 1
    uncompressed_size = int(footprint_header['has_intensity_uncertainty'] & uncompressedMask)

    if uncompressed_size:
        index_dtype = EventIndexBinZ
    else:
        index_dtype = EventIndexBin

    footprint_index_file = resolve_file(idx_file_in, mode="rb", stack=stack)
    footprint_mmap = np.memmap(footprint_index_file, dtype=index_dtype, mode='r')

    footprint_index = pd.DataFrame(footprint_mmap, columns=footprint_mmap.dtype.names).set_index('event_id').to_dict('index')

    return footprint, footprint_index


def _read_footprint_bins(stack, file_in, idx_file_in):
    footprint_file = resolve_file(file_in, mode="rb", stack=stack)

    if footprint_file == sys.stdin.buffer:
        footprint = np.frombuffer(footprint_file.read(), dtype="u1")
    else:
        footprint = np.fromfile(footprint_file, dtype="u1")

    footprint_index_file = resolve_file(idx_file_in, mode="rb", stack=stack)
    footprint_mmap = np.memmap(footprint_index_file, dtype=EventIndexBin, mode='r')

    footprint_index = pd.DataFrame(footprint_mmap, columns=footprint_mmap.dtype.names).set_index('event_id').to_dict('index')

    return footprint, footprint_index


def footprint_tocsv(stack, file_in, file_out, file_type, noheader, idx_file_in, zip_files, event_from_to):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]
    no_event_range, from_event, to_event = _check_event_from_to(event_from_to)

    if zip_files:
        footprint, footprint_index = _read_footprint_zips(stack, file_in, idx_file_in)
    else:
        footprint, footprint_index = _read_footprint_bins(stack, file_in, idx_file_in)

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    event_ids = sorted(footprint_index.keys())
    for event_id in event_ids:
        if no_event_range or (event_id >= from_event and event_id <= to_event):
            offset = footprint_index[event_id]["offset"]
            size = footprint_index[event_id]["size"]
            if zip_files:
                event_data = zlib.decompress(footprint[offset:offset + size].tobytes())
                event_data = np.frombuffer(event_data, dtype=Event)
            else:
                event_data = np.frombuffer(footprint[offset:offset + size].tobytes(), dtype=Event)
            event_csv_data = np.empty(len(event_data), dtype=dtype)
            event_csv_data["event_id"] = event_id
            event_csv_data["areaperil_id"] = event_data["areaperil_id"]
            event_csv_data["intensity_bin_id"] = event_data["intensity_bin_id"]
            event_csv_data["probability"] = event_data["probability"]
            write_ndarray_to_fmt_csv(file_out, event_csv_data, headers, fmt)
