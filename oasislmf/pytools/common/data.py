import os
import numba as nb
import numpy as np
import pandas as pd


oasis_int = np.dtype(os.environ.get('OASIS_INT', 'i4'))
nb_oasis_int = nb.from_dtype(oasis_int)
oasis_int_size = oasis_int.itemsize

oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))
nb_oasis_float = nb.from_dtype(oasis_float)
oasis_float_size = oasis_float.itemsize

areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
nb_areaperil_int = nb.from_dtype(areaperil_int)
areaperil_int_size = areaperil_int.itemsize

null_index = oasis_int.type(-1)


# Mean type numbers for outputs (SampleType)
MEAN_TYPE_ANALYTICAL = 1
MEAN_TYPE_SAMPLE = 2


def generate_output_metadata(output):
    """Generates *_header, *_dtype and *_fmt items given a list of tuples describing some output description
    output description has type List(Tuple({name: str}, {type: Any}, {format: str}))
    Args:
        output_map (list(tuple(str, Any, str))): Dictionary mapping string name to  {output description}_output list
    Returns:
        result (tuple(list[str], np.dtype, str)): Tuple containing the generated *_header list, *_dtype np.dtype, *_fmt csv format string
    """
    headers = [c[0] for c in output]
    dtype = np.dtype([(c[0], c[1]) for c in output])
    fmt = ','.join([c[2] for c in output])
    result = (headers, dtype, fmt)
    return result


# Types
aggregatevulnerability_output = [
    ("aggregate_vulnerability_id", 'i4', "%d"),
    ("vulnerability_id", 'i4', "%d"),
]
aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt = generate_output_metadata(aggregatevulnerability_output)

amplifications_output = [
    ("item_id", 'i4', "%d"),
    ("amplification_id", 'i4', "%d"),
]
amplifications_headers, amplifications_dtype, amplifications_fmt = generate_output_metadata(amplifications_output)

complex_items_meta_output = [
    ("item_id", 'u4', "%u"),
    ("coverage_id", 'u4', "%u"),
    ("group_id", 'u4', "%u"),
    ("model_data_len", 'u4', "%u"),
]
complex_items_meta_headers, complex_items_meta_dtype, complex_items_meta_fmt = generate_output_metadata(complex_items_meta_output)

coverages_output = [
    ("coverage_id", 'i4', "%d"),
    ("tiv", oasis_float, "%f"),
]
coverages_headers, coverages_dtype, coverages_fmt = generate_output_metadata(coverages_output)

damagebin_output = [
    ("bin_index", 'i4', "%d"),
    ("bin_from", oasis_float, "%f"),
    ("bin_to", oasis_float, "%f"),
    ("interpolation", oasis_float, "%f"),
    ("interval_type", 'i4', "%d"),
]
damagebin_headers, damagebin_dtype, damagebin_fmt = generate_output_metadata(damagebin_output)

eve_output = [
    ("event_id", oasis_int, "%d")
]
eve_headers, eve_dtype, eve_fmt = generate_output_metadata(eve_output)

fm_policytc_output = [
    ("level_id", 'i4', "%d"),
    ("agg_id", 'i4', "%d"),
    ("layer_id", 'i4', "%d"),
    ("profile_id", 'i4', "%d"),
]
fm_policytc_headers, fm_policytc_dtype, fm_policytc_fmt = generate_output_metadata(fm_policytc_output)

fm_profile_output = [
    ("profile_id", 'i4', "%d"),
    ("calcrule_id", 'i4', "%d"),
    ("deductible1", oasis_float, "%f"),
    ("deductible2", oasis_float, "%f"),
    ("deductible3", oasis_float, "%f"),
    ("attachment1", oasis_float, "%f"),
    ("limit1", oasis_float, "%f"),
    ("share1", oasis_float, "%f"),
    ("share2", oasis_float, "%f"),
    ("share3", oasis_float, "%f"),
]
fm_profile_headers, fm_profile_dtype, fm_profile_fmt = generate_output_metadata(fm_profile_output)

fm_profile_step_output = [
    ("profile_id", 'i4', "%d"),
    ("calcrule_id", 'i4', "%d"),
    ("deductible1", oasis_float, "%f"),
    ("deductible2", oasis_float, "%f"),
    ("deductible3", oasis_float, "%f"),
    ("attachment1", oasis_float, "%f"),
    ("limit1", oasis_float, "%f"),
    ("share1", oasis_float, "%f"),
    ("share2", oasis_float, "%f"),
    ("share3", oasis_float, "%f"),
    ("step_id", 'i4', "%d"),
    ("trigger_start", oasis_float, "%f"),
    ("trigger_end", oasis_float, "%f"),
    ("payout_start", oasis_float, "%f"),
    ("payout_end", oasis_float, "%f"),
    ("limit2", oasis_float, "%f"),
    ("scale1", oasis_float, "%f"),
    ("scale2", oasis_float, "%f"),
]
fm_profile_step_headers, fm_profile_step_dtype, fm_profile_step_fmt = generate_output_metadata(fm_profile_step_output)

fm_programme_output = [
    ("from_agg_id", 'i4', "%d"),
    ("level_id", 'i4', "%d"),
    ("to_agg_id", 'i4', "%d"),
]
fm_programme_headers, fm_programme_dtype, fm_programme_fmt = generate_output_metadata(fm_programme_output)

fm_summary_xref_output = [
    ("output", 'i4', "%d"),
    ("summary_id", 'i4', "%d"),
    ("summaryset_id", 'i4', "%d")
]
fm_summary_xref_headers, fm_summary_xref_dtype, fm_summary_xref_fmt = generate_output_metadata(fm_summary_xref_output)

fm_xref_output = [
    ("output", 'i4', "%d"),
    ("agg_id", 'i4', "%d"),
    ("layer_id", 'i4', "%d"),
]
fm_xref_headers, fm_xref_dtype, fm_xref_fmt = generate_output_metadata(fm_xref_output)

gul_summary_xref_output = [
    ("item_id", 'i4', "%d"),
    ("summary_id", 'i4', "%d"),
    ("summaryset_id", 'i4', "%d")
]
gul_summary_xref_headers, gul_summary_xref_dtype, gul_summary_xref_fmt = generate_output_metadata(gul_summary_xref_output)

items_output = [
    ("item_id", 'i4', "%d"),
    ("coverage_id", 'i4', "%d"),
    ("areaperil_id", areaperil_int, "%u"),
    ("vulnerability_id", 'i4', "%d"),
    ("group_id", 'i4', "%d"),
]
items_headers, items_dtype, items_fmt = generate_output_metadata(items_output)


def load_as_ndarray(dir_path, name, _dtype, must_exist=True, col_map=None):
    """
    load a file as a numpy ndarray
    useful for multi-columns files
    Args:
        dir_path: path to the directory where the binary or csv file is stored
        name: name of the file
        _dtype: np.dtype
        must_exist: raise FileNotFoundError if no file is present
        col_map: name re-mapping to change name of csv columns
    Returns:
        numpy ndarray
    """

    if os.path.isfile(os.path.join(dir_path, name + '.bin')):
        return np.fromfile(os.path.join(dir_path, name + '.bin'), dtype=_dtype)
    elif must_exist or os.path.isfile(os.path.join(dir_path, name + '.csv')):
        # in csv column cam be out of order and have different name,
        # we load with pandas and write each column to the ndarray
        if col_map is None:
            col_map = {}
        with open(os.path.join(dir_path, name + '.csv')) as file_in:
            cvs_dtype = {col_map.get(key, key): col_dtype for key, (col_dtype, _) in _dtype.fields.items()}
            df = pd.read_csv(file_in, delimiter=',', dtype=cvs_dtype, usecols=list(cvs_dtype.keys()))
            res = np.empty(df.shape[0], dtype=_dtype)
            for name in _dtype.names:
                res[name] = df[col_map.get(name, name)]
            return res
    else:
        return np.empty(0, dtype=_dtype)


def load_as_array(dir_path, name, _dtype, must_exist=True):
    """
    load file as a single numpy array,
     useful for files with a binary version with only one type of value where their index correspond to an id.
     For example coverage.bin only contains tiv value for each coverage id
     coverage_id n correspond to index n-1
    Args:
        dir_path: path to the directory where the binary or csv file is stored
        name: name of the file
        _dtype: numpy dtype of the required array
        must_exist: raise FileNotFoundError if no file is present
    Returns:
        numpy array of dtype type
    """
    fp = os.path.join(dir_path, name + '.bin')
    if os.path.isfile(fp):
        return np.fromfile(fp, dtype=_dtype)
    elif must_exist or os.path.isfile(os.path.join(dir_path, name + '.csv')):
        fp = os.path.join(dir_path, name + '.csv')
        with open(fp) as file_in:
            return np.loadtxt(file_in, dtype=_dtype, delimiter=',', skiprows=1, usecols=1)
    else:
        return np.empty(0, dtype=_dtype)


def write_ndarray_to_fmt_csv(output_file, data, headers, row_fmt):
    """Writes a custom dtype array with headers to csv with the provided row_fmt str

    This function is a faster replacement for np.savetxt as it formats each row one at a time before writing to csv.
    We create one large string, and formats all the data at once, and writes all the data at once.

    WARNING: untested with string types in custom data.

    Args:
        output_file (io.TextIOWrapper): CSV file
        data (ndarray[<custom dtype>]): Custom dtype ndarray with column names
        headers (list[str]): Column names for custom ndarray
        row_fmt (str): Format for each row in csv
    """
    if len(headers) != len(row_fmt.split(",")):
        raise RuntimeError(f"ERROR: write_ndarray_to_fmt_csv requires row_fmt ({row_fmt}) and headers ({headers}) to have the same length.")

    # Copy data as np.ravel does not work with custom dtype arrays
    # Default type of np.empty is np.float64.
    data_cpy = np.empty((data.shape[0], len(headers)))
    for i in range(len(headers)):
        data_cpy[:, i] = data[headers[i]]

    # Create one large formatted string
    final_fmt = "\n".join([row_fmt] * data_cpy.shape[0])
    str_data = final_fmt % tuple(np.ravel(data_cpy))

    output_file.write(str_data)
    output_file.write("\n")


float_equal_precision = np.finfo(oasis_float).eps


@nb.njit(cache=True)
def almost_equal(a, b):
    return abs(a - b) < float_equal_precision
