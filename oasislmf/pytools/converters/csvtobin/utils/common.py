import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import resolve_file


def read_csv_as_ndarray(stack, file_in, headers, dtype):
    file_in = resolve_file(file_in, "r", stack)

    csv_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    try:
        df = pd.read_csv(file_in, delimiter=',', dtype=csv_dtype, usecols=list(csv_dtype.keys()))
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=dtype)

    data = np.empty(df.shape[0], dtype=dtype)
    for name in dtype.names:
        data[name] = df[name]
    return data
