import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file


def df_to_ndarray(df, dtype):
    """Convert a pandas DataFrame to a numpy structured ndarray.

    Args:
        df (pd.DataFrame): Source DataFrame whose columns match dtype field names.
        dtype (np.dtype): Target numpy structured dtype.

    Returns:
        np.ndarray: Structured array with the given dtype.
    """
    data = np.empty(df.shape[0], dtype=dtype)
    for name in dtype.names:
        data[name] = df[name]
    return data


def iter_csv_as_ndarray(stack, file_in, dtype, chunksize=DEFAULT_BUFFER_SIZE):
    file_in = resolve_file(file_in, "r", stack)
    csv_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    try:
        for df_chunk in pd.read_csv(file_in, delimiter=',', dtype=csv_dtype,
                                    usecols=list(csv_dtype.keys()),
                                    chunksize=chunksize):
            yield df_to_ndarray(df_chunk, dtype)
    except pd.errors.EmptyDataError:
        return


def read_csv_as_ndarray(stack, file_in, headers, dtype):
    file_in = resolve_file(file_in, "r", stack)

    csv_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    try:
        df = pd.read_csv(file_in, delimiter=',', dtype=csv_dtype, usecols=list(csv_dtype.keys()))
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=dtype)

    return df_to_ndarray(df, dtype)
