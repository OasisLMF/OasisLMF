import numpy as np
from oasislmf.pytools.common.data import oasis_int, oasis_float
from oasislmf.pytools.common.event_stream import SUMMARY_STREAM_ID
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


def summarycalc_tobin(stack, file_in, file_out, file_type, max_sample_index, summary_set_id):
    dtype = TOOL_INFO[file_type]["dtype"]

    stream_agg_type = 1
    stream_info = (SUMMARY_STREAM_ID << 24 | stream_agg_type)
    np.array([stream_info], dtype="i4").tofile(file_out)
    np.array([max_sample_index], dtype="i4").tofile(file_out)
    np.array([summary_set_id], dtype="i4").tofile(file_out)

    curr_event_id = -1
    curr_summary_id = -1
    curr_expval = -1
    sidx_losses = []
    sidx_loss_dtype = np.dtype([("sidx", oasis_int), ("loss", oasis_float)])

    for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
        for row in chunk:
            event_id = row["EventId"]
            summary_id = row["SummaryId"]
            sidx = row["SampleId"]
            loss = row["Loss"]
            expval = row["ImpactedExposure"]
            if (event_id != curr_event_id) or (summary_id != curr_summary_id) or (expval != curr_expval):
                if curr_event_id != -1:
                    sidx_losses.append((0, 0))
                    np.array([curr_event_id, curr_summary_id], dtype=oasis_int).tofile(file_out)
                    np.array([curr_expval], dtype=oasis_float).tofile(file_out)
                    np.array(sidx_losses, dtype=sidx_loss_dtype).tofile(file_out)
                curr_event_id = event_id
                curr_summary_id = summary_id
                curr_expval = expval
                sidx_losses = []
            if sidx <= max_sample_index:
                sidx_losses.append((sidx, loss))
    if curr_event_id != -1:
        sidx_losses.append((0, 0))
        np.array([curr_event_id, curr_summary_id], dtype=oasis_int).tofile(file_out)
        np.array([curr_expval], dtype=oasis_float).tofile(file_out)
        np.array(sidx_losses, dtype=sidx_loss_dtype).tofile(file_out)
