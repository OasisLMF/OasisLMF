from pathlib import Path
import numpy as np
from oasislmf.pytools.common.data import write_ndarray_to_fmt_csv
from oasislmf.pytools.converters.data import TYPE_MAP
from oasislmf.pytools.pla.structure import read_lossfactors


def lossfactors_tocsv(file_in, file_out, file_type, noheader):
    headers = TYPE_MAP[file_type]["headers"]
    dtype = TYPE_MAP[file_type]["dtype"]
    fmt = TYPE_MAP[file_type]["fmt"]

    lossfactors_fp = Path(file_in)
    plafactors = read_lossfactors(lossfactors_fp.parent, set(["csv"]), filename=lossfactors_fp.name)

    data = np.empty(len(plafactors), dtype=dtype)
    for i, (k, v) in enumerate(plafactors.items()):
        data[i] = (k[0], k[1], v)

    csv_out_file = open(file_out, "w")
    if not noheader:
        csv_out_file.write(",".join(headers) + "\n")
    write_ndarray_to_fmt_csv(csv_out_file, data, headers, fmt)
    csv_out_file.close()
