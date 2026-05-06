import zlib
import numba as nb
import numpy as np
import re
import sys

from oasislmf.pytools.common.data import resolve_file, write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import (
    Event_dtype, EventIndexBin_dtype, EventIndexBinZ_dtype, FootprintHeader
)

# Number of output rows to accumulate before flushing to write_ndarray_to_fmt_csv.
# Reduces per-call overhead (format-string construction, np.empty, etc.) from
# O(num_events) down to O(num_events / BATCH_ROWS).
_BATCH_ROWS = 1 << 13  # 8 K rows


@nb.njit(cache=True, error_model="numpy")
def _fill_next_batch(
    out_event_id, out_areaperil_id, out_intensity_bin_id, out_probability,
    index_event_ids, elem_offsets, event_elem_counts,
    fp_areaperil_id, fp_intensity_bin_id, fp_probability,
    event_start,
):
    """Fill output arrays with complete events starting from *event_start*.

    Stops when the batch buffer is full (never splits an event across batches)
    or all events have been consumed.

    Returns (rows_filled, events_consumed).
    events_consumed == 0 means the next event alone exceeds the buffer; the
    caller must handle it directly.
    """
    max_rows = len(out_event_id)
    out_idx = 0
    i = event_start
    while i < len(index_event_ids):
        n = event_elem_counts[i]
        if out_idx + n > max_rows:
            break
        elem_start = elem_offsets[i]
        eid = index_event_ids[i]
        for j in range(n):
            out_event_id[out_idx] = eid
            out_areaperil_id[out_idx] = fp_areaperil_id[elem_start + j]
            out_intensity_bin_id[out_idx] = fp_intensity_bin_id[elem_start + j]
            out_probability[out_idx] = fp_probability[elem_start + j]
            out_idx += 1
        i += 1
    return out_idx, i - event_start


def _check_event_from_to(event_from_to):
    from_event = -1
    to_event = -1
    if event_from_to is None:
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
        footprint = np.memmap(footprint_file, dtype="u1", mode='r')

    footprint_header = np.frombuffer(footprint[:FootprintHeader.size].tobytes(), dtype=FootprintHeader)

    uncompressedMask = 1 << 1
    uncompressed_size = int(footprint_header['has_intensity_uncertainty'].item() & uncompressedMask)

    if uncompressed_size:
        index_dtype = EventIndexBinZ_dtype
    else:
        index_dtype = EventIndexBin_dtype

    footprint_index_file = resolve_file(idx_file_in, mode="rb", stack=stack)
    footprint_index = np.memmap(footprint_index_file, dtype=index_dtype, mode='r')

    return footprint, footprint_index


def _read_footprint_bins(stack, file_in, idx_file_in):
    footprint_file = resolve_file(file_in, mode="rb", stack=stack)

    if footprint_file == sys.stdin.buffer:
        footprint = np.frombuffer(footprint_file.read(), dtype="u1")
    else:
        footprint = np.memmap(footprint_file, dtype="u1", mode='r')

    footprint_index_file = resolve_file(idx_file_in, mode="rb", stack=stack)
    footprint_index = np.memmap(footprint_index_file, dtype=EventIndexBin_dtype, mode='r')

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

    # Use the index as-is if already sorted (the common case — footprinttobin always writes
    # events in ascending order). Only fall back to argsort+copy when out-of-order entries
    # are detected, avoiding an O(n log n) sort and an O(n_events) index copy.
    event_ids = footprint_index['event_id']
    if len(event_ids) > 1 and not np.all(np.diff(event_ids) >= 0):
        sort_order = np.argsort(event_ids, kind='stable')
        sorted_index = footprint_index[sort_order]
    else:
        sorted_index = footprint_index

    # Filter to event range upfront to avoid a per-event branch in the hot loop
    if not no_event_range:
        mask = (sorted_index['event_id'] >= from_event) & (sorted_index['event_id'] <= to_event)
        sorted_index = sorted_index[mask]

    if len(sorted_index) == 0:
        return

    if zip_files:
        _footprint_tocsv_zip(footprint, sorted_index, file_out, dtype, headers, fmt)
    else:
        _footprint_tocsv_bin(footprint, sorted_index, file_out, dtype, headers, fmt)


def _footprint_tocsv_bin(footprint, sorted_index, file_out, dtype, headers, fmt):
    """Non-zip path: batch events through a JIT inner loop; one write per ~_BATCH_ROWS rows."""
    header_size = FootprintHeader.size
    item_size = Event_dtype.itemsize

    # View footprint data (past header) as Event records — zero-copy
    footprint_events = footprint[header_size:].view(Event_dtype)
    fp_areaperil = footprint_events['areaperil_id']
    fp_intensity = footprint_events['intensity_bin_id']
    fp_probability = footprint_events['probability']

    # Pre-compute element-level offsets and counts once (numpy vectorised)
    elem_offsets = (sorted_index['offset'].astype(np.int64) - header_size) // item_size
    event_elem_counts = sorted_index['size'].astype(np.int64) // item_size

    # Cap to actual data size so small files don't over-allocate
    actual_batch_rows = min(_BATCH_ROWS, int(event_elem_counts.sum()))
    batch_data = np.empty(actual_batch_rows, dtype=dtype)
    event_cursor = 0
    n_events = len(sorted_index)

    while event_cursor < n_events:
        n_rows, consumed = _fill_next_batch(
            batch_data['event_id'], batch_data['areaperil_id'],
            batch_data['intensity_bin_id'], batch_data['probability'],
            sorted_index['event_id'], elem_offsets, event_elem_counts,
            fp_areaperil, fp_intensity, fp_probability,
            event_cursor,
        )
        if consumed == 0:
            # Single event exceeds _BATCH_ROWS; allocate exactly and write directly
            n = int(event_elem_counts[event_cursor])
            es = int(elem_offsets[event_cursor])
            large_buf = np.empty(n, dtype=dtype)
            large_buf['event_id'] = int(sorted_index['event_id'][event_cursor])
            large_buf['areaperil_id'] = fp_areaperil[es:es + n]
            large_buf['intensity_bin_id'] = fp_intensity[es:es + n]
            large_buf['probability'] = fp_probability[es:es + n]
            write_ndarray_to_fmt_csv(file_out, large_buf, headers, fmt)
            consumed = 1
        else:
            write_ndarray_to_fmt_csv(file_out, batch_data[:n_rows], headers, fmt)
        event_cursor += consumed


def _footprint_tocsv_zip(footprint, sorted_index, file_out, dtype, headers, fmt):
    """Zip path: decompress per event; numpy field assignment into pre-allocated buffer.

    When the index carries a 'd_size' (decompressed size) field we can pre-allocate
    a single reusable buffer.  Without it the compressed size is smaller than the
    decompressed size, so we allocate per event.
    """
    if 'd_size' in sorted_index.dtype.names:
        max_event_elems = int(sorted_index['d_size'].max()) // Event_dtype.itemsize
        event_csv_data = np.empty(max_event_elems, dtype=dtype)
    else:
        event_csv_data = None

    for row in sorted_index:
        event_id = int(row['event_id'])
        offset = int(row['offset'])
        compressed_size = int(row['size'])
        event_data = np.frombuffer(
            zlib.decompress(footprint[offset:offset + compressed_size].tobytes()),
            dtype=Event_dtype,
        )
        n = len(event_data)
        buf = event_csv_data[:n] if event_csv_data is not None else np.empty(n, dtype=dtype)
        buf['event_id'] = event_id
        buf['areaperil_id'] = event_data['areaperil_id']
        buf['intensity_bin_id'] = event_data['intensity_bin_id']
        buf['probability'] = event_data['probability']
        write_ndarray_to_fmt_csv(file_out, buf, headers, fmt)
