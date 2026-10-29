import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.pytools.converters.data import TOOL_INFO
from oasislmf.utils.exceptions import OasisException


def coverages_tobin(stack, file_in, file_out, file_type):
    dtype = TOOL_INFO[file_type]["dtype"]
    tiv_dtype = dtype.fields["tiv"][0]
    f = resolve_file(file_in, "r", stack)

    # coverages.bin has no id column (coverage n -> index n - 1), so coverage_id must be
    # contiguous from 1 or a TIV silently lands against the wrong coverage.
    last_id = 0
    try:
        for chunk in pd.read_csv(f, usecols=["coverage_id", "tiv"],
                                 dtype={"coverage_id": "int64", "tiv": tiv_dtype},
                                 chunksize=DEFAULT_BUFFER_SIZE):
            coverage_ids = chunk["coverage_id"].to_numpy()
            expected = np.arange(last_id + 1, last_id + 1 + len(coverage_ids))
            bad = np.flatnonzero(coverage_ids != expected)
            if bad.size:
                i = int(bad[0])
                raise OasisException(
                    f"Error: coverage_id {coverage_ids[i]} at row {last_id + i + 1} is not contiguous; "
                    f"expected {expected[i]}. coverage_id must be ascending from 1 with no gaps."
                )
            last_id += len(coverage_ids)
            file_out.write(chunk["tiv"].to_numpy(dtype=tiv_dtype).tobytes())
    except pd.errors.EmptyDataError:
        pass
