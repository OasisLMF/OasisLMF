import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, generate_output_metadata, occurrence_dtype, occurrence_granular_dtype
from oasislmf.pytools.common.input_files import occ_get_date_id
from oasislmf.pytools.converters.csvtobin.utils.common import read_csv_as_ndarray
from oasislmf.pytools.converters.data import TOOL_INFO


def occurrence_tobin(stack, file_in, file_out, file_type, no_of_periods, no_date_alg=False, granular=False):
    @nb.njit(cache=True, error_model="numpy")
    def _get_occ_data_with_date_ids(occ_csv, occ_dtype):
        buffer_size = DEFAULT_BUFFER_SIZE
        buffer = np.zeros(buffer_size, dtype=occ_dtype)

        idx = 0
        for row in occ_csv:
            if idx >= buffer_size:
                yield buffer[:idx]
                buffer = np.zeros(buffer_size, dtype=occ_dtype)
                idx = 0
            occ_date_id = occ_get_date_id(
                False,
                row["occ_year"],
                row["occ_month"],
                row["occ_day"],
            )
            buffer[idx]["event_id"] = row["event_id"]
            buffer[idx]["period_no"] = row["period_no"]
            buffer[idx]["occ_date_id"] = occ_date_id
            idx += 1
        yield buffer[:idx]

    @nb.njit(cache=True, error_model="numpy")
    def _get_occ_data_with_date_ids_gran(occ_csv, occ_dtype):
        buffer_size = DEFAULT_BUFFER_SIZE
        buffer = np.zeros(buffer_size, dtype=occ_dtype)

        idx = 0
        for row in occ_csv:
            if idx >= buffer_size:
                yield buffer[:idx]
                buffer = np.zeros(buffer_size, dtype=occ_dtype)
                idx = 0
            occ_date_id = occ_get_date_id(
                True,
                row["occ_year"],
                row["occ_month"],
                row["occ_day"],
                row["occ_hour"],
                row["occ_minute"],
            )
            buffer[idx]["event_id"] = row["event_id"]
            buffer[idx]["period_no"] = row["period_no"]
            buffer[idx]["occ_date_id"] = occ_date_id
            idx += 1
        yield buffer[:idx]

    if no_date_alg and granular:
        raise RuntimeError("Cannot have an occurrence file with granular dates and no date algorithm. Use at most one of -D, -G, but not both")

    # Write date opts
    date_opts = granular << 1 | (not no_date_alg)
    np.array([date_opts], dtype="i4").tofile(file_out)
    np.array([no_of_periods], dtype="i4").tofile(file_out)

    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    if no_date_alg:
        csv_data = read_csv_as_ndarray(stack, file_in, headers, dtype)
        csv_data.tofile(file_out)
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
        headers, dtype, fmt = generate_output_metadata(occ_csv_output)
        csv_data = read_csv_as_ndarray(stack, file_in, headers, dtype)
        gen = _get_occ_data_with_date_ids(csv_data, occurrence_dtype)
        if granular:
            gen = _get_occ_data_with_date_ids_gran(csv_data, occurrence_granular_dtype)

        for data in gen:
            if any(data["period_no"] > no_of_periods):
                raise RuntimeError("FATAL: Period number exceeds maximum supplied")
            data.tofile(file_out)
