import numpy as np
import pandas as pd


def read_csv_as_ndarray(file_in, headers, dtype):
    with open(file_in, "r") as fin:
        first_line = fin.readline()
        if not first_line.strip():
            return np.empty(0, dtype=dtype)

    csv_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    try:
        df = pd.read_csv(file_in, delimiter=',', dtype=csv_dtype, usecols=list(csv_dtype.keys()))
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=dtype)

    data = np.empty(df.shape[0], dtype=dtype)
    for name in dtype.names:
        data[name] = df[name]
    return data
