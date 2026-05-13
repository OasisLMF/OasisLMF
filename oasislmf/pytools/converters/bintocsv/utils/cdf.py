# CDF binary → CSV converter.
#
# Input is a getmodel binary stream (CDF_STREAM_ID) piped or read from file.
# read_getmodel_stream (Numba JIT in gul/io.py) parses the stream and yields
# one tuple per event: (event_id, damagecdfrecs, recs, rec_idx_ptr).
# damagecdfrecs is a structured array of (areaperil_id, vulnerability_id) per
# CDF group; recs is a flat ProbMean array; rec_idx_ptr is a numba.typed.List
# of start indices into recs for each group (length = n_groups + 1).
#
# Output paths:
#
#   Normal events (n_rows <= _BATCH_ROWS):
#     A JIT inner loop (_fill_cdf_batch) writes one event's rows directly into
#     a pre-allocated output buffer (batch_data, _BATCH_ROWS rows), starting at
#     the current write position.  The buffer is flushed to
#     write_ndarray_to_fmt_csv in one call per _BATCH_ROWS rows, amortising
#     per-call overhead across multiple events.  rec_idx_ptr is passed directly
#     as a numba.typed.List without conversion.
#
#   Oversized events (n_rows > _BATCH_ROWS):
#     An exact-size buffer is allocated and written in one call, then freed.

import logging
import numba as nb
import numpy as np
from pathlib import Path

from oasislmf.pytools.common.data import resolve_file, write_ndarray_to_fmt_csv, items_dtype
from oasislmf.pytools.common.input_files import read_coverages
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.gul.common import coverage_type
from oasislmf.pytools.gul.manager import generate_item_map, gul_get_items, read_getmodel_stream

logger = logging.getLogger(__name__)

_BATCH_ROWS = 1 << 13  # 8 K rows


@nb.njit(cache=True, error_model="numpy")
def _fill_cdf_batch(
    out_eid, out_apid, out_vid, out_binidx, out_prob_to, out_bin_mean,
    event_id, damagecdfrecs, recs, rec_idx_ptr, out_start,
):
    """Fill output column arrays with one event's rows starting at *out_start*.

    Returns the new write position (out_start + rows_written).
    """
    n_groups = len(rec_idx_ptr) - 1
    out_idx = out_start
    for g in range(n_groups):
        ap_id = damagecdfrecs[g]['areaperil_id']
        vuln_id = damagecdfrecs[g]['vulnerability_id']
        s = rec_idx_ptr[g]
        e = rec_idx_ptr[g + 1]
        for j in range(s, e):
            out_eid[out_idx] = event_id
            out_apid[out_idx] = ap_id
            out_vid[out_idx] = vuln_id
            out_binidx[out_idx] = j - s + 1
            out_prob_to[out_idx] = recs[j]['prob_to']
            out_bin_mean[out_idx] = recs[j]['bin_mean']
            out_idx += 1
    return out_idx


def cdf_tocsv(stack, file_in, file_out, file_type, noheader, run_dir):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    input_path = Path(run_dir, 'input')
    file_in = resolve_file(file_in, "rb", stack)

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    items = gul_get_items(input_path)
    coverages_tiv = read_coverages(input_path)
    coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv
    item_map = generate_item_map(items, coverages)
    del coverages_tiv

    compute = np.zeros(coverages.shape[0] + 1, items.dtype['coverage_id'])
    seeds = np.zeros(len(np.unique(items['group_id'])), dtype=items_dtype['group_id'])
    valid_area_peril_id = None

    batch_data = np.empty(_BATCH_ROWS, dtype=dtype)
    batch_pos = 0

    for event_data in read_getmodel_stream(file_in, item_map, coverages, compute, seeds, valid_area_peril_id):
        event_id, compute_i, items_data, damagecdfrecs, recs, rec_idx_ptr, rng_index = event_data
        n_rows = len(recs)

        if batch_pos + n_rows > _BATCH_ROWS:
            if batch_pos > 0:
                write_ndarray_to_fmt_csv(file_out, batch_data[:batch_pos], headers, fmt)
                batch_pos = 0
            if n_rows > _BATCH_ROWS:
                # Single event exceeds batch; allocate exactly and write directly
                large_buf = np.empty(n_rows, dtype=dtype)
                _fill_cdf_batch(
                    large_buf['event_id'], large_buf['areaperil_id'],
                    large_buf['vulnerability_id'], large_buf['bin_index'],
                    large_buf['prob_to'], large_buf['bin_mean'],
                    event_id, damagecdfrecs, recs, rec_idx_ptr, 0,
                )
                write_ndarray_to_fmt_csv(file_out, large_buf, headers, fmt)
                continue

        batch_pos = _fill_cdf_batch(
            batch_data['event_id'], batch_data['areaperil_id'],
            batch_data['vulnerability_id'], batch_data['bin_index'],
            batch_data['prob_to'], batch_data['bin_mean'],
            event_id, damagecdfrecs, recs, rec_idx_ptr, batch_pos,
        )

    if batch_pos > 0:
        write_ndarray_to_fmt_csv(file_out, batch_data[:batch_pos], headers, fmt)
