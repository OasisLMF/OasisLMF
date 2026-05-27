# Footprint CSV → binary converter.
#
# The CSV is read in fixed-size chunks (via iter_csv_as_ndarray) so memory usage is O(chunk)
# regardless of file size. Each chunk is processed as follows:
#
#   1. Validation (unless no_validation=True): three Numba JIT checks run over the chunk,
#      carrying state across chunk boundaries via scalar "prev_*" variables —
#      sort order, probability sums per (event_id, areaperil_id) group, and
#      duplicate intensity_bin_id detection.
#
#   2. Event boundary detection: a single np.diff pass finds all event-id change points,
#      producing start/end slices for every event in the chunk in one vectorised step.
#
#   3. Writing: complete events (all rows present in this chunk) are batch-converted and
#      written in one tobytes()/write() call (non-zip), or compressed individually per
#      event (zip). Events that span a chunk boundary are buffered in partial_chunks and
#      flushed once their final rows arrive in the next chunk.
#
#   4. Index: one (event_id, offset, size[, decompressed_size]) entry per event is
#      accumulated and written to the .idx file after all chunks are processed.
#
# no_validation=True skips step 1 and writes events in whatever order they appear in the
# CSV — the caller is responsible for ensuring the input is sorted by (event_id, areaperil_id).

import zlib
import numba as nb
import numpy as np

from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.getmodel.common import Event_dtype, EventIndexBin_dtype, EventIndexBinZ_dtype
from oasislmf.utils.exceptions import OasisException


@nb.njit(cache=True, error_model="numpy")
def _check_sorted(event_ids, areaperil_ids, prev_event_id, prev_areaperil_id, first_chunk):
    """Single-pass sort check. Returns (bad_idx, last_event_id, last_areaperil_id)."""
    if len(event_ids) == 0:
        return np.int64(-1), prev_event_id, prev_areaperil_id
    if not first_chunk:
        if event_ids[0] < prev_event_id or (
                event_ids[0] == prev_event_id and areaperil_ids[0] < prev_areaperil_id):
            return np.int64(0), event_ids[0], areaperil_ids[0]
    for i in range(1, len(event_ids)):
        if event_ids[i] < event_ids[i - 1] or (
                event_ids[i] == event_ids[i - 1] and areaperil_ids[i] < areaperil_ids[i - 1]):
            return np.int64(i), event_ids[i], areaperil_ids[i]
    return np.int64(-1), event_ids[-1], areaperil_ids[-1]


@nb.njit(cache=True, error_model="numpy")
def _check_prob_sums(event_ids, areaperil_ids, probs,
                     prev_event_id, prev_areaperil_id, running_sum, first_chunk,
                     atol=1e-6):
    """Incremental probability sum check assuming sorted data.
    Returns (bad_idx, last_event_id, last_areaperil_id, running_sum).
    bad_idx=-1 means valid; the final group is not finalised here — check after last chunk."""
    if len(event_ids) == 0:
        return np.int64(-1), prev_event_id, prev_areaperil_id, running_sum
    if first_chunk:
        prev_event_id = event_ids[0]
        prev_areaperil_id = areaperil_ids[0]
        running_sum = np.float64(probs[0])
        i_start = 1
    else:
        i_start = 0
    for i in range(i_start, len(event_ids)):
        if event_ids[i] != prev_event_id or areaperil_ids[i] != prev_areaperil_id:
            if abs(running_sum - 1.0) > atol:
                return np.int64(i - 1), prev_event_id, prev_areaperil_id, running_sum
            running_sum = np.float64(probs[i])
            prev_event_id = event_ids[i]
            prev_areaperil_id = areaperil_ids[i]
        else:
            running_sum += probs[i]
    return np.int64(-1), prev_event_id, prev_areaperil_id, running_sum


@nb.njit(cache=True, error_model="numpy")
def _check_duplicates(event_ids, areaperil_ids, intensity_bin_ids,
                      prev_event_id, prev_areaperil_id, prev_intensity_bin_id, first_chunk):
    """Single-pass duplicate intensity_bin_id check assuming sorted data.
    Returns (bad_idx, last_event_id, last_areaperil_id, last_intensity_bin_id)."""
    if len(event_ids) == 0:
        return np.int64(-1), prev_event_id, prev_areaperil_id, prev_intensity_bin_id
    if not first_chunk:
        if (event_ids[0] == prev_event_id and areaperil_ids[0] == prev_areaperil_id
                and intensity_bin_ids[0] == prev_intensity_bin_id):
            return np.int64(0), event_ids[0], areaperil_ids[0], intensity_bin_ids[0]
    for i in range(1, len(event_ids)):
        if (event_ids[i] == event_ids[i - 1] and areaperil_ids[i] == areaperil_ids[i - 1]
                and intensity_bin_ids[i] == intensity_bin_ids[i - 1]):
            return np.int64(i), event_ids[i], areaperil_ids[i], intensity_bin_ids[i]
    return np.int64(-1), event_ids[-1], areaperil_ids[-1], intensity_bin_ids[-1]


@nb.njit(cache=True, error_model="numpy")
def _exceeds_max_intensity(intensity_bin_ids, max_val):
    """Early-exit check for any intensity_bin_id exceeding max_val."""
    for v in intensity_bin_ids:
        if v > max_val:
            return True
    return False


def _validate_chunk(chunk, event_ids, areaperil_ids, first_chunk,
                    prev_sort_event, prev_sort_areaperil,
                    prev_prob_event, prev_prob_areaperil, running_sum,
                    prev_dup_event, prev_dup_areaperil, prev_dup_intensity):
    """Run all three streaming validation checks for one chunk. Returns updated carry state."""
    # Check sorted by (event_id, areaperil_id)
    bad_idx, prev_sort_event, prev_sort_areaperil = _check_sorted(
        event_ids, areaperil_ids,
        prev_sort_event, prev_sort_areaperil, first_chunk,
    )
    if bad_idx != -1:
        raise OasisException(
            f"IDs not in ascending order at row {bad_idx}: {chunk[bad_idx]}"
        )

    # Check probability sums to 1 per (event_id, areaperil_id) group
    bad_idx, prev_prob_event, prev_prob_areaperil, running_sum = _check_prob_sums(
        event_ids, areaperil_ids, np.ascontiguousarray(chunk["probability"]),
        prev_prob_event, prev_prob_areaperil, running_sum, first_chunk,
    )
    if bad_idx != -1:
        raise OasisException(
            f"Probabilities do not sum to 1 for group ending at row {bad_idx}: "
            f"event_id={prev_prob_event}, areaperil_id={prev_prob_areaperil}"
        )

    # Check no duplicate intensity_bin_id within a group
    bad_idx, prev_dup_event, prev_dup_areaperil, prev_dup_intensity = _check_duplicates(
        event_ids, areaperil_ids, np.ascontiguousarray(chunk["intensity_bin_id"]),
        prev_dup_event, prev_dup_areaperil, prev_dup_intensity, first_chunk,
    )
    if bad_idx != -1:
        raise OasisException(
            f"Duplicate intensity_bin_id at row {bad_idx}: "
            f"event_id={chunk['event_id'][bad_idx]}, areaperil_id={chunk['areaperil_id'][bad_idx]}, "
            f"intensity_bin_id={chunk['intensity_bin_id'][bad_idx]}"
        )

    return (prev_sort_event, prev_sort_areaperil,
            prev_prob_event, prev_prob_areaperil, running_sum,
            prev_dup_event, prev_dup_areaperil, prev_dup_intensity)


def _flush_event(event_id, rows, file_out, idx_entries,
                 max_intensity_bin_idx, zip_files, decompressed_size, offset):
    """Convert, optionally compress, and write a single event. Used for partial events
    (spanning chunk boundaries) and for the zip path where per-event compression is required."""
    bin_data = np.empty(len(rows), dtype=Event_dtype)
    bin_data["areaperil_id"] = rows["areaperil_id"]
    bin_data["intensity_bin_id"] = rows["intensity_bin_id"]
    bin_data["probability"] = rows["probability"]

    if _exceeds_max_intensity(bin_data["intensity_bin_id"], max_intensity_bin_idx):
        raise OasisException(
            f"Error: Found intensity_bin_idx in data larger than max_intensity_bin_idx: {max_intensity_bin_idx}"
        )

    bin_bytes = bin_data.tobytes()
    dsize = len(bin_bytes)
    if zip_files:
        bin_bytes = zlib.compress(bin_bytes)
    file_out.write(bin_bytes)
    size = len(bin_bytes)

    if decompressed_size:
        idx_entries.append((event_id, offset, size, dsize))
    else:
        idx_entries.append((event_id, offset, size))

    return offset + size


def footprint_tobin(
    stack, file_in, file_out, file_type,
    idx_file_out,
    zip_files,
    max_intensity_bin_idx,
    no_intensity_uncertainty,
    decompressed_size,
    no_validation
):
    dtype = TOOL_INFO[file_type]["dtype"]
    idx_file_out = resolve_file(idx_file_out, "wb", stack)

    # Write bin file header
    np.array([max_intensity_bin_idx], dtype=np.int32).tofile(file_out)
    zip_opts = decompressed_size << 1 | (not no_intensity_uncertainty)
    np.array([zip_opts], dtype=np.int32).tofile(file_out)
    offset = np.dtype(np.int32).itemsize * 2

    idx_entries = []
    idx_dtype = EventIndexBinZ_dtype if decompressed_size else EventIndexBin_dtype

    first_chunk = True
    any_data = False

    # Validation carry state (dummy initial values; first_chunk=True prevents their use)
    prev_sort_event = np.int32(0)
    prev_sort_areaperil = np.uint32(0)
    prev_prob_event = np.int32(0)
    prev_prob_areaperil = np.uint32(0)
    running_sum = np.float64(0.0)
    prev_dup_event = np.int32(0)
    prev_dup_areaperil = np.uint32(0)
    prev_dup_intensity = np.int32(0)

    # Partial-event buffer for events that span chunk boundaries
    partial_event_id = None
    partial_chunks = []

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        if len(chunk) == 0:
            continue

        event_ids = np.ascontiguousarray(chunk["event_id"])
        areaperil_ids = np.ascontiguousarray(chunk["areaperil_id"])

        if not no_validation:
            (prev_sort_event, prev_sort_areaperil,
             prev_prob_event, prev_prob_areaperil, running_sum,
             prev_dup_event, prev_dup_areaperil, prev_dup_intensity) = _validate_chunk(
                chunk, event_ids, areaperil_ids, first_chunk,
                prev_sort_event, prev_sort_areaperil,
                prev_prob_event, prev_prob_areaperil, running_sum,
                prev_dup_event, prev_dup_areaperil, prev_dup_intensity,
            )

        first_chunk = False
        any_data = True

        pos = 0

        # Continue partial event from previous chunk boundary
        if partial_event_id is not None:
            end = int(np.searchsorted(event_ids, partial_event_id, side='right'))
            partial_chunks.append(chunk[:end])
            pos = end
            if pos == len(chunk):
                continue
            offset = _flush_event(
                partial_event_id, np.concatenate(partial_chunks), file_out, idx_entries,
                max_intensity_bin_idx, zip_files, decompressed_size, offset,
            )
            partial_event_id = None
            partial_chunks = []

        # Find all event boundaries from pos to end of chunk in one vectorised pass
        remaining_ids = event_ids[pos:]
        if len(remaining_ids) == 0:
            continue

        changes = np.flatnonzero(np.diff(remaining_ids)) + 1 if len(remaining_ids) > 1 \
            else np.empty(0, dtype=np.intp)
        rel_starts = np.concatenate([[np.intp(0)], changes])
        rel_ends = np.append(changes, [np.intp(len(remaining_ids))])
        n_complete = len(rel_starts) - 1

        if n_complete > 0:
            if zip_files:
                # Zip path: each event must be compressed separately
                for i in range(n_complete):
                    s = pos + int(rel_starts[i])
                    e = pos + int(rel_ends[i])
                    offset = _flush_event(
                        int(event_ids[s]), chunk[s:e], file_out, idx_entries,
                        max_intensity_bin_idx, zip_files, decompressed_size, offset,
                    )
            else:
                # Non-zip path: batch convert and write all complete events in one shot
                complete_end = pos + int(rel_ends[n_complete - 1])
                complete_rows = chunk[pos:complete_end]
                bin_data = np.empty(len(complete_rows), dtype=Event_dtype)
                bin_data["areaperil_id"] = complete_rows["areaperil_id"]
                bin_data["intensity_bin_id"] = complete_rows["intensity_bin_id"]
                bin_data["probability"] = complete_rows["probability"]

                if _exceeds_max_intensity(bin_data["intensity_bin_id"], max_intensity_bin_idx):
                    raise OasisException(
                        f"Error: Found intensity_bin_idx in data larger than max_intensity_bin_idx: {max_intensity_bin_idx}"
                    )

                file_out.write(bin_data.tobytes())

                row_size = Event_dtype.itemsize
                for i in range(n_complete):
                    event_id = int(event_ids[pos + int(rel_starts[i])])
                    size = int(rel_ends[i] - rel_starts[i]) * row_size
                    if decompressed_size:
                        idx_entries.append((event_id, offset, size, size))
                    else:
                        idx_entries.append((event_id, offset, size))
                    offset += size

        # Buffer last group — unknown whether it's complete until next chunk arrives
        last_start = pos + int(rel_starts[-1])
        partial_event_id = int(event_ids[last_start])
        partial_chunks = [chunk[last_start:]]

    # Flush final event (held in partial buffer through the last chunk)
    if partial_event_id is not None:
        offset = _flush_event(
            partial_event_id, np.concatenate(partial_chunks), file_out, idx_entries,
            max_intensity_bin_idx, zip_files, decompressed_size, offset,
        )

    # Finalise last probability group (not checked inside the loop)
    if not no_validation and any_data and abs(running_sum - 1.0) > 1e-6:
        raise OasisException(
            f"Probabilities do not sum to 1 for final group: "
            f"event_id={prev_prob_event}, areaperil_id={prev_prob_areaperil}"
        )

    np.array(idx_entries, dtype=idx_dtype).tofile(idx_file_out)
