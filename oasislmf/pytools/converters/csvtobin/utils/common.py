import numpy as np
import pandas as pd
from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, resolve_file
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.integer_narrowing import raise_if_out_of_range, widen_int_dtypes


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


def _raise_int_overflow_error(e, file_in, csv_dtype):
    """Re-raise a pandas error for an integer beyond the 64-bit read dtype as an OasisException."""
    int_cols = [col for col, dtype in csv_dtype.items() if np.dtype(dtype).kind in 'iu']
    raise OasisException(
        f"Error: {file_in} has a value in one of the integer columns {int_cols} that is out of range "
        f"for a 64-bit integer: {e}"
    ) from e


def _is_int_overflow_error(e):
    """Whether a pandas read_csv error is about an integer outside the 64-bit read dtype.

    pandas reports values beyond 64 bits as OverflowError, and values that only fit the other
    64-bit integer dtype as "cannot safely convert passed user dtype of int64 for uint64 dtyped
    data" (or the reverse for a negative value read as uint64).
    """
    if isinstance(e, OverflowError):
        return True
    message = str(e)
    return "cannot safely convert passed user dtype" in message and (
        "for uint64 dtyped" in message or "for int64 dtyped" in message)


def df_to_ndarray(df, dtype, source=None):
    """Convert a pandas DataFrame to a numpy structured ndarray.

    Integer values that don't fit their field raise rather than being silently wrapped (#2197);
    columns whose dtype already casts safely to the field are not scanned.

    Args:
        df (pd.DataFrame): Source DataFrame whose columns match dtype field names.
        dtype (np.dtype): Target numpy structured dtype.
        source (str): file or table the data comes from, for error messages.

    Returns:
        np.ndarray: Structured array with the given dtype.

    Raises:
        OasisException: if an integer value is out of range for its field.
    """
    data = np.empty(df.shape[0], dtype=dtype)
    for name in dtype.names:
        raise_if_out_of_range(df[name], dtype[name], name, source)
        data[name] = df[name]
    return data


def iter_csv_as_ndarray(stack, file_in, dtype, chunksize=DEFAULT_BUFFER_SIZE):
    # Pass path strings directly to pd.read_csv so the C engine can stream
    # lazily. Wrapping in a Python file object forces pandas to buffer the
    # entire file before parsing, defeating chunked reading.
    if str(file_in) == "-":
        source = resolve_file(file_in, "r", stack)  # stdin must be opened
    else:
        source = file_in
    csv_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    # integers are read as 64 bits and range-checked by df_to_ndarray: pandas may wrap a value
    # read straight into a narrower dtype
    read_dtype = widen_int_dtypes(csv_dtype)
    try:
        for df_chunk in pd.read_csv(source, delimiter=',', dtype=read_dtype,
                                    usecols=list(csv_dtype.keys()),
                                    chunksize=chunksize):
            yield df_to_ndarray(df_chunk, dtype, source=file_in)
    except pd.errors.EmptyDataError:
        return
    except (ValueError, OverflowError) as e:
        if _is_int_overflow_error(e):
            _raise_int_overflow_error(e, file_in, csv_dtype)
        _raise_clear_csv_error(e, csv_dtype)


def read_csv_as_ndarray(stack, file_in, headers, dtype):
    name = file_in
    file_in = resolve_file(file_in, "r", stack)

    csv_dtype = {key: col_dtype for key, (col_dtype, _) in dtype.fields.items()}
    try:
        df = pd.read_csv(file_in, delimiter=',', dtype=widen_int_dtypes(csv_dtype), usecols=list(csv_dtype.keys()))
    except pd.errors.EmptyDataError:
        return np.empty(0, dtype=dtype)
    except (ValueError, OverflowError) as e:
        if _is_int_overflow_error(e):
            _raise_int_overflow_error(e, name, csv_dtype)
        _raise_clear_csv_error(e, csv_dtype)

    return df_to_ndarray(df, dtype, source=name)
