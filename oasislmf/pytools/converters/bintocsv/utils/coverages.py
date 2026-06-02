from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_coverages
from oasislmf.pytools.converters.data import TOOL_INFO


def coverages_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    if str(file_in) == "-":
        coverages = read_coverages(
            ignore_file_type=set(["csv"]),
            use_stdin=True
        )
    else:
        cov_fp = Path(file_in)
        coverages = read_coverages(
            run_dir=cov_fp.parent,
            ignore_file_type=set(["csv"]),
            filename=cov_fp.name
        )

    n = len(coverages)

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    buf = np.empty(min(DEFAULT_BUFFER_SIZE, n), dtype=dtype)
    for start in range(0, n, DEFAULT_BUFFER_SIZE):
        end = min(start + DEFAULT_BUFFER_SIZE, n)
        batch = buf[:end - start]
        batch["coverage_id"] = np.arange(start + 1, end + 1)
        batch["tiv"] = coverages[start:end]
        write_ndarray_to_fmt_csv(file_out, batch, headers, fmt)
