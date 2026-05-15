import numpy as np
from oasislmf.pytools.common.data import generate_output_metadata, occurrence_dtype, occurrence_granular_dtype
from oasislmf.pytools.converters.csvtobin.utils.common import iter_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


def occurrence_tobin(stack, file_in, file_out, file_type, no_of_periods, no_date_alg=False, granular=False):
    if no_date_alg and granular:
        raise RuntimeError("Cannot have an occurrence file with granular dates and no date algorithm. Use at most one of -D, -G, but not both")

    date_opts = granular << 1 | (not no_date_alg)
    np.array([date_opts], dtype="i4").tofile(file_out)
    np.array([no_of_periods], dtype="i4").tofile(file_out)

    if no_date_alg:
        dtype = TOOL_INFO[file_type]["dtype"]
        for chunk in iter_csv_as_ndarray(stack, file_in, dtype):
            if np.any(chunk["period_no"] > no_of_periods):
                raise RuntimeError("FATAL: Period number exceeds maximum supplied")
            chunk.tofile(file_out)
    else:
        occ_csv_output = [
            ("event_id", 'i4', "%d"),
            ("period_no", 'i4', "%d"),
            ("occ_year", 'i4', "%d"),
            ("occ_month", 'i4', "%d"),
            ("occ_day", 'i4', "%d"),
        ]
        if granular:
            occ_csv_output += [
                ("occ_hour", 'i4', "%d"),
                ("occ_minute", 'i4', "%d"),
            ]
        _headers, csv_dtype, _fmt = generate_output_metadata(occ_csv_output)
        out_dtype = occurrence_granular_dtype if granular else occurrence_dtype

        for chunk in iter_csv_as_ndarray(stack, file_in, csv_dtype):
            out = np.empty(len(chunk), dtype=out_dtype)
            out["event_id"] = chunk["event_id"]
            out["period_no"] = chunk["period_no"]

            m = (chunk["occ_month"].astype(np.int64) + 9) % 12
            y = chunk["occ_year"].astype(np.int64) - m // 10
            date_id = 365 * y + y // 4 - y // 100 + y // 400 + (m * 306 + 5) // 10 + (chunk["occ_day"].astype(np.int64) - 1)
            if granular:
                date_id = date_id * 1440 + 60 * chunk["occ_hour"].astype(np.int64) + chunk["occ_minute"].astype(np.int64)
            out["occ_date_id"] = date_id

            if np.any(out["period_no"] > no_of_periods):
                raise RuntimeError("FATAL: Period number exceeds maximum supplied")
            out.tofile(file_out)
