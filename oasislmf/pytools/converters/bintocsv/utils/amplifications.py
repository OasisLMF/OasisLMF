from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, write_ndarray_to_fmt_csv
from oasislmf.pytools.common.input_files import read_amplifications
from oasislmf.pytools.converters.data import TOOL_INFO


def amplifications_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    if str(file_in) == "-":
        raw = read_amplifications(use_stdin=True, raw=True)
    else:
        amps_fp = Path(file_in)
        raw = read_amplifications(run_dir=amps_fp.parent, filename=amps_fp.name, raw=True)

    data = raw.view(dtype)

    if not noheader:
        file_out.write(",".join(headers) + "\n")

    for start in range(0, len(data), DEFAULT_BUFFER_SIZE):
        write_ndarray_to_fmt_csv(file_out, data[start:start + DEFAULT_BUFFER_SIZE], headers, fmt)
