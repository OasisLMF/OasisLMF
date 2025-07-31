from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
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
    data = np.zeros(len(coverages), dtype=dtype)
    data["coverage_id"] = np.arange(1, len(coverages) + 1)
    data["tiv"] = coverages

    if not noheader:
        file_out.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(file_out, data, headers, fmt)
