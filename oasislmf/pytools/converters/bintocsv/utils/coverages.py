from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_coverages
from oasislmf.pytools.converters.data import TYPE_MAP


def coverages_tocsv(file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    cov_fp = Path(file_in)
    coverages = read_coverages(cov_fp.parent, set(["csv"]), filename=cov_fp.name)
    data = np.zeros(len(coverages), dtype=dtype)
    data["coverage_id"] = np.arange(1, len(coverages) + 1)
    data["tiv"] = coverages

    csv_out_file = open(file_out, "w")
    if not noheader:
        csv_out_file.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(csv_out_file, data, headers, fmt)
    csv_out_file.close()
