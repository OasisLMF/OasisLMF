from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_amplifications
from oasislmf.pytools.converters.data import TOOL_INFO


def amplifications_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    if str(file_in) == "-":
        items_amps = read_amplifications(use_stdin=True)
    else:
        amps_fp = Path(file_in)
        items_amps = read_amplifications(run_dir=amps_fp.parent, filename=amps_fp.name)
    items_amps = items_amps[1:]
    data = np.zeros(len(items_amps), dtype=dtype)
    data["item_id"] = np.arange(1, len(items_amps) + 1)
    data["amplification_id"] = items_amps

    if not noheader:
        file_out.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(file_out, data, headers, fmt)
