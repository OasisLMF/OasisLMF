import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.int_range import check_int_range, widen_int_dtypes


def _raise_clear_csv_error(e, csv_dtype):
    """Re-raise a pandas 'Usecols do not match' ValueError naming the expected columns, or
    re-raise the original exception unchanged. Must be called from within the `except ValueError
    as e:` block that caught e, so the bare `raise` below re-raises the right exception.
    """
    if "Usecols do not match" in str(e):
        raise OasisException(
            f"Error: CSV header does not match the expected columns {list(csv_dtype.keys())} "
            f"(check header name case/whitespace): {e}"
        ) from e
    raise


def df_to_ndarray(df, dtype):
    """Convert a pandas DataFrame to a numpy structured ndarray.

    Args:
        df (pd.DataFrame): Source DataFrame whose columns match dtype field names.
        dtype (np.dtype): Target numpy structured dtype.

    Returns:
        np.ndarray: Structured array with the given dtype.

    Raises:
        OasisException: if an integer column has a value that does not fit its field, which
            numpy would otherwise wrap to a different value.
    """
    data = np.empty(df.shape[0], dtype=dtype)
    for name in dtype.names:
        check_int_range(df[name], dtype.fields[name][0], name)
        data[name] = df[name]
    return data


def _csv_read_dtypes(dtype):
    """Map a structured dtype to ``read_csv`` dtypes, reading integer fields at 64 bits so that
    ``df_to_ndarray`` can range check them before narrowing.
    """
    return widen_int_dtypes({key: col_dtype for key, (col_dtype, _) in dtype.fields.items()})


def iter_csv_as_ndarray(stack, file_in, dtype, chunksize=DEFAULT_BUFFER_SIZE):
    # Pass path strings directly to pd.read_csv so the C engine can stream
    # lazily. Wrapping in a Python file object forces pandas to buffer the
    # entire file before parsing, defeating chunked reading.
    if str(file_in) == "-":
        source = resolve_file(file_in, "r", stack)  # stdin must be opened
    else:
        source = file_in
    csv_dtype = _csv_read_dtypes(dtype)
    try:
        for df_chunk in pd.read_csv(source, delimiter=',', dtype=csv_dtype,
                                    usecols=list(csv_dtype.keys()),
                                    chunksize=chunksize):
            yield df_to_ndarray(df_chunk, dtype)
    except pd.errors.EmptyDataError:
        return
    except ValueError as e:
        _raise_clear_csv_error(e, csv_dtype)


def read_csv_as_ndarray(stack, file_in, headers, dtype):
    file_in = resolve_file(file_in, "r", stack)

    csv_dtype = _csv_read_dtypes(dtype)
    try:
        df = pd.read_csv(file_in, delimiter=',', dtype=csv_dtype, usecols=list(csv_dtype.keys()))
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=dtype)
    except ValueError as e:
        _raise_clear_csv_error(e, csv_dtype)

    return df_to_ndarray(df, dtype)
