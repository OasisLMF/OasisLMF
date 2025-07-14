from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_amplifications
from oasislmf.pytools.converters.data import TYPE_MAP


def amplifications_tocsv(file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    amps_fp = Path(file_in)
    items_amps = read_amplifications(amps_fp.parent, amps_fp.name)
    items_amps = items_amps[1:]
    data = np.zeros(len(items_amps), dtype=dtype)
    data["item_id"] = np.arange(1, len(items_amps) + 1)
    data["amplification_id"] = items_amps

    csv_out_file = open(file_out, "w")
    if not noheader:
        csv_out_file.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(csv_out_file, data, headers, fmt)
    csv_out_file.close()
