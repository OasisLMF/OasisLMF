import pandas as pd
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO


def coverages_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]
    tiv_dtype = dtype.fields["tiv"][0]
    f = resolve_file(file_in, "r", stack)
    try:
        for chunk in pd.read_csv(f, usecols=["tiv"], dtype={"tiv": tiv_dtype}, chunksize=DEFAULT_BUFFER_SIZE):
            chunk["tiv"].to_numpy(dtype=tiv_dtype).tofile(file_out)
    except pd.errors.EmptyDataError:
        pass
