from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, generate_output_metadata, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_occurrence_bin
from oasislmf.pytools.converters.data import TOOL_INFO


def _date_id_to_out(chunk, dtype, granular_date):
    date_id = chunk["occ_date_id"].astype(np.int64)
    days = date_id // 1440 if granular_date else date_id

    y = (10000 * days + 14780) // 3652425
    ddd = days - (365 * y + y // 4 - y // 100 + y // 400)
    mask = ddd < 0
    if mask.any():
        y[mask] -= 1
        y_m = y[mask]
        ddd[mask] = days[mask] - (365 * y_m + y_m // 4 - y_m // 100 + y_m // 400)
    mi = (100 * ddd + 52) // 3060
    mm = (mi + 2) % 12 + 1
    y = y + (mi + 2) // 12
    dd = ddd - (mi * 306 + 5) // 10 + 1

    out = np.empty(len(chunk), dtype=dtype)
    out["event_id"] = chunk["event_id"]
    out["period_no"] = chunk["period_no"]
    out["occ_year"] = y
    out["occ_month"] = mm
    out["occ_day"] = dd
    if granular_date:
        minutes = date_id % 1440
        out["occ_hour"] = minutes // 60
        out["occ_minute"] = minutes % 60
    return out


def occurrence_tocsv(stack, file_in, file_out, file_type, noheader):
    if str(file_in) == "-":
        occ_arr, date_algorithm, granular_date, no_of_periods = read_occurrence_bin(use_stdin=True)
    else:
        occ_arr, date_algorithm, granular_date, no_of_periods = read_occurrence_bin(
            run_dir=Path(file_in).parent,
            filename=Path(file_in).name,
        )

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

        if not noheader:
            file_out.write(",".join(headers) + "\n")

        for start in range(0, len(occ_arr), DEFAULT_BUFFER_SIZE):
            chunk = occ_arr[start:start + DEFAULT_BUFFER_SIZE]
            write_ndarray_to_fmt_csv(file_out, _date_id_to_out(chunk, dtype, granular_date), headers, fmt)
    else:
        headers = TOOL_INFO[file_type]["headers"]
        dtype = TOOL_INFO[file_type]["dtype"]
        fmt = TOOL_INFO[file_type]["fmt"]

        if not noheader:
            file_out.write(",".join(headers) + "\n")
        write_ndarray_to_fmt_csv(file_out, occ_arr, headers, fmt)
