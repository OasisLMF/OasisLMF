from pathlib import Path
import numba as nb
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, generate_output_metadata, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import occ_get_date, read_occurrence_bin
from oasislmf.pytools.converters.data import TOOL_INFO


def occurrence_tocsv(stack, file_in, file_out, file_type, noheader):
    @nb.njit(cache=True, error_model="numpy")
    def _get_occ_data_with_dates(occ_arr, occ_csv_dtype):
        buffer_size = DEFAULT_BUFFER_SIZE
        buffer = np.zeros(buffer_size, dtype=occ_csv_dtype)

        idx = 0
        for row in occ_arr:
            if idx >= buffer_size:
                yield buffer[:idx]
                buffer = np.zeros(buffer_size, dtype=occ_csv_dtype)
                idx = 0
            year, month, day, hour, minute = occ_get_date(row["occ_date_id"], granular_date)
            buffer[idx]["event_id"] = row["event_id"]
            buffer[idx]["period_no"] = row["period_no"]
            buffer[idx]["occ_year"] = year
            buffer[idx]["occ_month"] = month
            buffer[idx]["occ_day"] = day
            idx += 1
        yield buffer[:idx]

    @nb.njit(cache=True, error_model="numpy")
    def _get_occ_data_with_dates_gran(occ_arr, occ_csv_dtype):
        buffer_size = DEFAULT_BUFFER_SIZE
        buffer = np.zeros(buffer_size, dtype=occ_csv_dtype)

        idx = 0
        for row in occ_arr:
            if idx >= buffer_size:
                yield buffer[:idx]
                buffer = np.zeros(buffer_size, dtype=occ_csv_dtype)
                idx = 0
            year, month, day, hour, minute = occ_get_date(row["occ_date_id"], granular_date)
            buffer[idx]["event_id"] = row["event_id"]
            buffer[idx]["period_no"] = row["period_no"]
            buffer[idx]["occ_year"] = year
            buffer[idx]["occ_month"] = month
            buffer[idx]["occ_day"] = day
            buffer[idx]["occ_hour"] = hour
            buffer[idx]["occ_minute"] = minute
            idx += 1
        yield buffer[:idx]

    run_dir = Path(file_in).parent
    filename = Path(file_in).name
    if str(file_in) == "-":
        occ_arr, date_algorithm, granular_date, no_of_periods = read_occurrence_bin(
            use_stdin=True
        )
    else:
        occ_arr, date_algorithm, granular_date, no_of_periods = read_occurrence_bin(
            run_dir=run_dir,
            filename=filename
        )
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    if date_algorithm:
        occ_csv_output = [
            ("event_id", 'i4', "%d"),
            ("period_no", 'i4', "%d"),
            ("occ_year", 'i4', "%d"),
            ("occ_month", 'i4', "%d"),
            ("occ_day", 'i4', "%d"),
        ]
        if granular_date:
            occ_csv_output += [
                ("occ_hour", 'i4', "%d"),
                ("occ_minute", 'i4', "%d"),
            ]
        headers, dtype, fmt = generate_output_metadata(occ_csv_output)
        gen = _get_occ_data_with_dates(occ_arr, dtype)
        if granular_date:
            gen = _get_occ_data_with_dates_gran(occ_arr, dtype)

        if not noheader:
            file_out.write(",".join(headers) + "\n")
        for data in gen:
            write_ndarray_to_fmt_csv(file_out, data, headers, fmt)
    else:
        if not noheader:
            file_out.write(",".join(headers) + "\n")
        write_ndarray_to_fmt_csv(file_out, occ_arr, headers, fmt)
