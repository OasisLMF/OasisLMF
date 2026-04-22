import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO


def coverages_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]
    tiv_dtype = dtype.fields["tiv"][0]
    f = resolve_file(file_in, "r", stack)
    try:
        tiv = pd.read_csv(f, usecols=["tiv"], dtype={"tiv": tiv_dtype})["tiv"].to_numpy(dtype=tiv_dtype)
    except pd.errors.EmptyDataError:
        tiv = np.empty(0, dtype=tiv_dtype)
    tiv.tofile(file_out)
