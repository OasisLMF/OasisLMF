#!/usr/bin/env python

import logging
import numpy as np
from pathlib import Path

from oasislmf.pytools.common.data import resolve_file, write_ndarray_to_fmt_csv, items_dtype
from oasislmf.pytools.common.input_files import read_coverages
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.gul.common import coverage_type
from oasislmf.pytools.gul.manager import generate_item_map, gul_get_items, read_getmodel_stream

logger = logging.getLogger(__name__)


def get_cdf_data(event_id, damagecdfrecs, recs, rec_idx_ptr, cdf_dtype):
    """Get the cdf data produced by getmodel.
    Note that the input arrays are lists of cdf entries, namely
    the shape on axis=0 is the number of entries.
    Args:
        event_id (int): event_id
        damagecdfrecs (ndarray[damagecdfrec]): cdf record keys
        recs (ndarray[ProbMean]): cdf record values
        rec_idx_ptr (ndarray[int]): array with the indices of `rec` where each cdf record starts.
        cdf_dtype (np.dtype): cdf numpy dtype.
    Returns:
        data (ndarray[cdf_dtype]): cdf data extracted from recs/getmodel.
    """
    assert len(damagecdfrecs) == len(rec_idx_ptr) - 1, "Number of cdfrecs groups does not match number of cdf keys found"

    data = np.zeros(len(recs), dtype=cdf_dtype)
    Nbins = len(rec_idx_ptr) - 1
    idx = 0
    for group_idx in range(Nbins):
        areaperil_id, vulnerability_id = damagecdfrecs[group_idx]
        for bin_index, rec in enumerate(recs[rec_idx_ptr[group_idx]:rec_idx_ptr[group_idx + 1]]):
            data[idx]["event_id"] = event_id
            data[idx]["areaperil_id"] = areaperil_id
            data[idx]["vulnerability_id"] = vulnerability_id
            data[idx]["bin_index"] = bin_index + 1
            data[idx]["prob_to"] = rec["prob_to"]
            data[idx]["bin_mean"] = rec["bin_mean"]
            idx += 1
    return data


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

    for event_data in read_getmodel_stream(file_in, item_map, coverages, compute, seeds, valid_area_peril_id):
        event_id, compute_i, items_data, damagecdfrecs, recs, rec_idx_ptr, rng_index = event_data
        data = get_cdf_data(event_id, damagecdfrecs, recs, rec_idx_ptr, dtype)
        write_ndarray_to_fmt_csv(file_out, data, headers, fmt)
