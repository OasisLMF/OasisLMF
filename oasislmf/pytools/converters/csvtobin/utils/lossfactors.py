# Loss-factors CSV → binary converter.
#
# The CSV is read in fixed-size chunks (iter_csv_as_ndarray), so memory usage is
# O(chunk_size + max_rows_per_event) regardless of file size. Events spanning chunk
# boundaries are buffered in partial_chunks and flushed once their final rows arrive.
# Input must be sorted by event_id.
# For each complete event an 8-byte header (event_id, count) is written followed by
# the amp_factor body as a contiguous structured-array block.

import numpy as np

from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.pla.common import amp_factor_dtype


def _flush_event(event_id, rows, file_out):
    count = len(rows)
    np.array([event_id, count], dtype=np.int32).tofile(file_out)
    body = np.empty(count, dtype=amp_factor_dtype)
    body['amplification_id'] = rows['amplification_id']
    body['factor'] = rows['factor']
    body.tofile(file_out)


def lossfactors_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]

    np.array([0], dtype=np.int32).tofile(file_out)

    partial_event_id = None
    partial_chunks = []

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        if len(chunk) == 0:
            continue

        event_ids = np.ascontiguousarray(chunk['event_id'])
        pos = 0

        if partial_event_id is not None:
            end = int(np.searchsorted(event_ids, partial_event_id, side='right'))
            partial_chunks.append(chunk[:end])
            pos = end
            if pos == len(chunk):
                continue
            _flush_event(partial_event_id, np.concatenate(partial_chunks), file_out)
            partial_event_id = None
            partial_chunks = []

        remaining_ids = event_ids[pos:]
        if len(remaining_ids) == 0:
            continue

        changes = np.flatnonzero(np.diff(remaining_ids)) + 1 if len(remaining_ids) > 1 \
            else np.empty(0, dtype=np.intp)
        rel_starts = np.concatenate([[np.intp(0)], changes])
        rel_ends = np.append(changes, [np.intp(len(remaining_ids))])
        n_complete = len(rel_starts) - 1

        for i in range(n_complete):
            s = pos + int(rel_starts[i])
            e = pos + int(rel_ends[i])
            _flush_event(int(event_ids[s]), chunk[s:e], file_out)

        last_start = pos + int(rel_starts[-1])
        partial_event_id = int(event_ids[last_start])
        partial_chunks = [chunk[last_start:]]

    if partial_event_id is not None:
        _flush_event(partial_event_id, np.concatenate(partial_chunks), file_out)
