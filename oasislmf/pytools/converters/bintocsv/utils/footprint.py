# Footprint binary → CSV converter.
#
# The binary footprint is memory-mapped (np.memmap) so only pages that are actually
# read are loaded — memory usage is proportional to the events being output, not file size.
# Processing proceeds as follows:
#
#   1. Index loading: the .idx file is memory-mapped as EventIndexBin_dtype or
#      EventIndexBinZ_dtype (when the binary carries a decompressed-size field). Every entry's
#      byte range is checked against the footprint file's actual size before anything reads
#      it: the batched njit loop below indexes without bounds checking, so a truncated or
#      mismatched idx/bin pair would otherwise read past the mapped file — silently wrong
#      data at best, a crash at worst.
#      If the index is out of order it is sorted by event_id once upfront via argsort;
#      the common case (already sorted) skips this entirely.
#
#   2. Event range filtering: when event_from_to is supplied the index is masked to the
#      requested range in one vectorised step before any footprint data is touched.
#
#   3. Output — two paths depending on zip_files:
#
#      Non-zip (_footprint_tocsv_bin): a JIT inner loop (_fill_next_batch) packs complete
#      events into a fixed pre-allocated output buffer (_BATCH_ROWS rows) and flushes it
#      to write_ndarray_to_fmt_csv in one call per batch. Single events that exceed the
#      batch size are written directly with an exact-size allocation.
#
#      Zip (_footprint_tocsv_zip): each event is decompressed individually with zlib. When
#      the index carries a d_size field (decompressed size), a single reusable buffer is
#      pre-allocated at the maximum event size; otherwise a new buffer is allocated per event.

import logging
import zlib
import numba as nb
import numpy as np
import re
import sys

from oasislmf.pytools.common.data import (
    resolve_file, write_ndarray_to_fmt_csv, memmap_or_empty, is_first_entry_zlib_compressed
)
from oasislmf.pytools.converters.bintocsv.utils.common import check_index_bounds
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import (
    Event_dtype, EventIndexBin_dtype, EventIndexBinZ_dtype, FootprintHeader
)
from oasislmf.utils.exceptions import OasisException

# Named explicitly (not __name__) to share bintocsv.manager's logger -- the one cli.py configures
# -- without importing manager.py here, which would be circular (manager.py imports this module).
logger = logging.getLogger("oasislmf.pytools.converters.bintocsv.manager")

# Number of output rows to accumulate before flushing to write_ndarray_to_fmt_csv.
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


def _read_file_or_empty(file_obj, dtype):
    """Read an already-resolved file as `dtype`: fully into memory for stdin (which has no
    fileno to memory-map), or via memmap_or_empty otherwise.
    """
    if file_obj == sys.stdin.buffer:
        return np.frombuffer(file_obj.read(), dtype=dtype)
    return memmap_or_empty(file_obj, dtype)


def _read_footprint_files(stack, file_in, idx_file_in):
    """Read the footprint bin and idx files, zipped or not: the idx entry layout (and so how to
    read the bin's bytes afterwards) only depends on the header bit, not on the caller's own
    zip_files flag -- a zipped and non-zipped footprint are read identically at this stage.
    """
    footprint_file = resolve_file(file_in, mode="rb", stack=stack)
    footprint = _read_file_or_empty(footprint_file, "u1")

    idx_dtype = _get_index_dtype(footprint)
    footprint_index_file = resolve_file(idx_file_in, mode="rb", stack=stack)
    footprint_index = _read_file_or_empty(footprint_index_file, idx_dtype)

    return footprint, footprint_index


def _get_index_dtype(footprint):
    """Index entries carry the decompressed size when the footprint header says so"""
    if len(footprint) < FootprintHeader.itemsize:
        raise OasisException(
            f"Error: footprint file is only {len(footprint)} bytes, too short to hold its "
            f"{FootprintHeader.itemsize}-byte header; it is empty or truncated."
        )
    footprint_header = np.frombuffer(footprint[:FootprintHeader.itemsize].tobytes(), dtype=FootprintHeader)
    uncompressedMask = 1 << 1
    if footprint_header['has_intensity_uncertainty'].item() & uncompressedMask:
        return EventIndexBinZ_dtype
    return EventIndexBin_dtype


def footprint_tocsv(stack, file_in, file_out, file_type, noheader, idx_file_in, zip_files, event_from_to):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]
    no_event_range, from_event, to_event = _check_event_from_to(event_from_to)

    footprint, footprint_index = _read_footprint_files(stack, file_in, idx_file_in)

    if not zip_files and is_first_entry_zlib_compressed(footprint, footprint_index):
        logger.warning(
            "WARNING: footprint holds zlib compressed events despite zip_files=False (saved under "
            "uncompressed names); reading it as a compressed footprint."
        )
        zip_files = True

    # Compressed event blocks are arbitrary zlib byte lengths, so only the uncompressed path has
    # a fixed per-record size to check entries' alignment against.
    check_index_bounds(footprint, footprint_index, FootprintHeader.itemsize, 'event_id', 'footprint',
                       record_size=(1 if zip_files else Event_dtype.itemsize))

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
    header_size = FootprintHeader.itemsize
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
        try:
            decompressed = zlib.decompress(footprint[offset:offset + compressed_size].tobytes())
        except zlib.error as e:
            raise OasisException(
                f"Error: failed to decompress event_id {event_id} with zip_files=True; "
                f"the file may not actually be zipped: {e}"
            ) from e
        if event_csv_data is not None and len(decompressed) != int(row['d_size']):
            raise OasisException(
                f"Error: event_id {event_id} decompressed to {len(decompressed)} bytes, expected "
                f"d_size {int(row['d_size'])} from the index file; the index and footprint files "
                "do not match, or the footprint file is corrupt."
            )
        event_data = np.frombuffer(decompressed, dtype=Event_dtype)
        n = len(event_data)
        buf = event_csv_data[:n] if event_csv_data is not None else np.empty(n, dtype=dtype)
        buf['event_id'] = event_id
        buf['areaperil_id'] = event_data['areaperil_id']
        buf['intensity_bin_id'] = event_data['intensity_bin_id']
        buf['probability'] = event_data['probability']
        write_ndarray_to_fmt_csv(file_out, buf, headers, fmt)
