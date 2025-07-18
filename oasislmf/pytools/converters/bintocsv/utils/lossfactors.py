from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.pytools.pla.structure import read_lossfactors


def lossfactors_tocsv(stack, file_in, file_out, file_type, noheader):
    headers = TOOL_INFO[file_type]["headers"]
    dtype = TOOL_INFO[file_type]["dtype"]
    fmt = TOOL_INFO[file_type]["fmt"]

    if str(file_in) == "-":
        plafactors = read_lossfactors(
            ignore_file_type=set(["csv"]),
            use_stdin=True
        )
    else:
        lossfactors_fp = Path(file_in)
        plafactors = read_lossfactors(
            run_dir=lossfactors_fp.parent,
            ignore_file_type=set(["csv"]),
            filename=lossfactors_fp.name
        )

    data = np.empty(len(plafactors), dtype=dtype)
    for i, (k, v) in enumerate(plafactors.items()):
        data[i] = (k[0], k[1], v)

    if not noheader:
        file_out.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(file_out, data, headers, fmt)
