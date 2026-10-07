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
#   4. Index: one (event_id, offset, size[, decompressed_size]) entry per event is written to
#      the .idx file as each event is flushed — never accumulated in memory for the whole file.
#      The non-zip batch path builds one chunk's worth of entries as a single vectorised array.
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
                     atol=1e-6, rtol=1e-5):
    """Incremental probability sum check assuming sorted data.
    Returns (bad_idx, last_event_id, last_areaperil_id, running_sum).
    bad_idx=-1 means valid; the final group is not finalised here — check after last chunk.

    atol/rtol match np.isclose's defaults (target is always 1.0, so the combined tolerance is
    just atol + rtol): this is the tolerance #1693 used before #1947's streaming rewrite dropped
    the rtol term, leaving a check over 10x stricter than vulnerability's equivalent check.
    """
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
            # NaN comparisons are always False, so "> atol" alone would silently accept a NaN
            # probability (e.g. from a blank CSV field) instead of flagging it as unresolved.
            if not (abs(running_sum - 1.0) <= atol + rtol):
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
    Returns (bad_idx, last_event_id, last_areaperil_id, last_intensity_bin_id).
    """
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
def _intensity_out_of_range(intensity_bin_ids, max_val):
    """Early-exit check for any intensity_bin_id outside [1, max_val]. 0 or below would wrap to
    the last intensity column when gulmc indexes vuln_array with intensity_bin_id - 1.
    """
    for v in intensity_bin_ids:
        if v > max_val or v < 1:
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


def _flush_event(event_id, rows, file_out, idx_file_out, idx_dtype,
                 max_intensity_bin_idx, zip_files, decompressed_size, offset):
    """Convert, optionally compress, and write a single event. Used for partial events
    (spanning chunk boundaries) and for the zip path where per-event compression is required.
    The index entry is written straight to idx_file_out, not accumulated in memory.
    """
    bin_data = np.empty(len(rows), dtype=Event_dtype)
    bin_data["areaperil_id"] = rows["areaperil_id"]
    bin_data["intensity_bin_id"] = rows["intensity_bin_id"]
    bin_data["probability"] = rows["probability"]

    if _intensity_out_of_range(bin_data["intensity_bin_id"], max_intensity_bin_idx):
        raise OasisException(
            f"Error: Found intensity_bin_idx in data outside the valid range [1, {max_intensity_bin_idx}]"
        )

    bin_bytes = bin_data.tobytes()
    dsize = len(bin_bytes)
    if zip_files:
        bin_bytes = zlib.compress(bin_bytes)
    file_out.write(bin_bytes)
    size = len(bin_bytes)

    entry = (event_id, offset, size, dsize) if decompressed_size else (event_id, offset, size)
    idx_file_out.write(np.array([entry], dtype=idx_dtype).tobytes())

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
    from oasislmf.pytools.converters.csvtobin.manager import logger

    dtype = TOOL_INFO[file_type]["dtype"]

    # The runtime looks for zipped footprints as footprint.bin.z / footprint.idx.z
    out_names = [str(idx_file_out), getattr(file_out, "name", None)]
    if zip_files and any(isinstance(name, str) and name not in ("-", "<stdout>") and not name.endswith(".z")
                         for name in out_names):
        logger.warning("WARNING: zipped footprint files should be named with a .z extension (footprint.bin.z / footprint.idx.z)")

    idx_file_out = resolve_file(idx_file_out, "wb", stack)

    # The decompressed size only applies to zipped footprints (as in ktools footprinttobin)
    if decompressed_size and not zip_files:
        logger.warning("WARNING: decompressed_size only applies to zipped footprints, ignoring it as zip_files is not set")
        decompressed_size = False

    # Write bin file header
    file_out.write(np.array([max_intensity_bin_idx], dtype=np.int32).tobytes())
    zip_opts = decompressed_size << 1 | (not no_intensity_uncertainty)
    file_out.write(np.array([zip_opts], dtype=np.int32).tobytes())
    offset = np.dtype(np.int32).itemsize * 2

    idx_dtype = EventIndexBinZ_dtype if decompressed_size else EventIndexBin_dtype

    first_chunk = True
    any_data = False

    # Validation carry state (dummy initial values; first_chunk=True prevents their use)
    prev_sort_event = np.int32(0)
    prev_sort_areaperil = dtype['areaperil_id'].type(0)
    prev_prob_event = np.int32(0)
    prev_prob_areaperil = dtype['areaperil_id'].type(0)
    running_sum = np.float64(0.0)
    prev_dup_event = np.int32(0)
    prev_dup_areaperil = dtype['areaperil_id'].type(0)
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
                partial_event_id, np.concatenate(partial_chunks), file_out, idx_file_out, idx_dtype,
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
                        int(event_ids[s]), chunk[s:e], file_out, idx_file_out, idx_dtype,
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

                if _intensity_out_of_range(bin_data["intensity_bin_id"], max_intensity_bin_idx):
                    raise OasisException(
                        f"Error: Found intensity_bin_idx in data outside the valid range [1, {max_intensity_bin_idx}]"
                    )

                file_out.write(bin_data.tobytes())

                # Vectorised: build and write this chunk's idx entries in one shot, rather than
                # accumulating every event's entry in memory for the whole file.
                row_size = Event_dtype.itemsize
                batch_event_ids = event_ids[pos + rel_starts[:n_complete]]
                batch_sizes = (rel_ends[:n_complete] - rel_starts[:n_complete]).astype(np.int64) * row_size
                batch_offsets = offset + np.concatenate(([0], np.cumsum(batch_sizes)[:-1]))
                batch_idx = np.empty(n_complete, dtype=idx_dtype)
                batch_idx["event_id"] = batch_event_ids
                batch_idx["offset"] = batch_offsets
                batch_idx["size"] = batch_sizes
                if decompressed_size:
                    batch_idx["d_size"] = batch_sizes
                idx_file_out.write(batch_idx.tobytes())
                offset += int(batch_sizes.sum())

        # Buffer last group — unknown whether it's complete until next chunk arrives
        last_start = pos + int(rel_starts[-1])
        partial_event_id = int(event_ids[last_start])
        partial_chunks = [chunk[last_start:]]

    # Flush final event (held in partial buffer through the last chunk)
    if partial_event_id is not None:
        offset = _flush_event(
            partial_event_id, np.concatenate(partial_chunks), file_out, idx_file_out, idx_dtype,
            max_intensity_bin_idx, zip_files, decompressed_size, offset,
        )

    # Finalise last probability group (not checked inside the loop)
    if not no_validation and any_data and not (abs(running_sum - 1.0) <= 1e-6 + 1e-5):
        raise OasisException(
            f"Probabilities do not sum to 1 for final group: "
            f"event_id={prev_prob_event}, areaperil_id={prev_prob_areaperil}"
        )
