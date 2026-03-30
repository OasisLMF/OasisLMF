import os
import sys
import numba as nb
import numpy as np
import pandas as pd


oasis_int = np.dtype(os.environ.get('OASIS_INT', 'i4'))
nb_oasis_int = nb.from_dtype(oasis_int)
oasis_int_size = oasis_int.itemsize

oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))
nb_oasis_float = nb.from_dtype(oasis_float)
oasis_float_size = oasis_float.itemsize

areaperil_int = np.dtype(os.environ.get('OASIS_AREAPERIL_TYPE', 'u4'))
nb_areaperil_int = nb.from_dtype(areaperil_int)
areaperil_int_size = areaperil_int.itemsize

null_index = oasis_int.type(-1)

# A default buffer size for nd arrays to be initialised to
DEFAULT_BUFFER_SIZE = 1_000_000

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


# single type definition index
DTYPE_IDX = 1
NAME_DTYPE_SLICE = slice(2)

# single type definition (alphabetical order)
agg_id = ("agg_id", 'i4', "%d")
aggregate_vulnerability_id = ("aggregate_vulnerability_id", 'i4', "%d")
amplification_id = ("amplification_id", 'i4', "%d")
areaperil_id = ("areaperil_id", areaperil_int, "%d")
attachment1 = ("attachment1", oasis_float, "%f")
bin_from = ("bin_from", oasis_float, "%f")
bin_index = ("bin_index", 'i4', "%d")
bin_mean = ("bin_mean", 'f4', "%f")
bin_to = ("bin_to", oasis_float, "%f")
calcrule_id = ("calcrule_id", 'i4', "%d")
coverage_id = ("coverage_id", 'u4', "%u")
damage_bin_id = ("damage_bin_id", 'i4', "%d")
damage_correlation_value = ("damage_correlation_value", oasis_float, "%f")
damage_type = ("damage_type", 'i4', "%d")
deductible1 = ("deductible1", oasis_float, "%f")
deductible2 = ("deductible2", oasis_float, "%f")
deductible3 = ("deductible3", oasis_float, "%f")
event_id = ("event_id", 'i4', "%d")
factor = ("factor", 'f4', "%.2f")  # Amplification factor
from_agg_id = ("from_agg_id", 'i4', "%d")
group_id = ("group_id", 'u4', "%u")
hazard_correlation_value = ("hazard_correlation_value", oasis_float, "%f")
hazard_group_id = ("hazard_group_id", 'i4', "%d")
intensity_adjustment = ('intensity_adjustment', 'i4', "%d")
intensity_bin_id = ('intensity_bin_id', 'i4', "%d")
interpolation = ("interpolation", oasis_float, "%f")
item_id = ("item_id", 'i4', "%d")
item_return_period = ('return_period', 'i4', "%d")
layer_id = ("layer_id", 'i4', "%d")
level_id = ("level_id", 'i4', "%d")
limit1 = ("limit1", oasis_float, "%f")
limit2 = ("limit2", oasis_float, "%f")
loss = ("loss", oasis_float, "%.2f")
model_data_len = ("model_data_len", 'u4', "%u")
occ_date_id = ("occ_date_id", 'i4', "%d")
occ_date_id_granular = ("occ_date_id", 'i8', "%d")
output_id = ("output_id", 'i4', "%d")
output = ("output", output_id[1], output_id[2])
payout_end = ("payout_end", oasis_float, "%f")
payout_start = ("payout_start", oasis_float, "%f")
peril_correlation_group = ("peril_correlation_group", 'i4', "%d")
period_no = ("period_no", 'i4', "%d")
period_weighting = ("weighting", 'f8', "%0.9lf")
prob_to = ("prob_to", 'f4', "%f")
probability = ('probability', oasis_float, "%.6f")
profile_id = ("profile_id", 'i4', "%d")
quantile = ("quantile", 'f4', "%f")
quantile_fractional_part = ('fractional_part', oasis_float, "%f")
quantile_integer_part = ('integer_part', oasis_int, "%d")
random_no = ("random_no", 'f4', "%f")
return_period = ("return_period", 'i4', "%d")
scale1 = ("scale1", oasis_float, "%f")
scale2 = ("scale2", oasis_float, "%f")
section_id = ("scale2", oasis_int, "%d")
share1 = ("share1", oasis_float, "%f")
share2 = ("share2", oasis_float, "%f")
share3 = ("share3", oasis_float, "%f")
sidx = ("sidx", 'i4', "%d")
step_id = ("step_id", 'i4', "%d")
summary_id = ("summary_id", 'i4', "%d")
summaryset_id = ("summaryset_id", 'i4', "%d")
tiv = ("tiv", oasis_float, "%f")
to_agg_id = ("to_agg_id", 'i4', "%d")
trigger_end = ("trigger_end", oasis_float, "%f")
trigger_start = ("trigger_start", oasis_float, "%f")
vulnerability_id = ("vulnerability_id", 'i4', "%d")
vulnerability_weight = ("weight", oasis_float, "%f")


# Types
aggregatevulnerability_output = [
    aggregate_vulnerability_id,
    vulnerability_id,
]
aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt = generate_output_metadata(aggregatevulnerability_output)

amplifications_output = [
    item_id,
    amplification_id,
]
amplifications_headers, amplifications_dtype, amplifications_fmt = generate_output_metadata(amplifications_output)

cdf_output = [
    event_id,
    areaperil_id,
    vulnerability_id,
    bin_index,
    prob_to,
    bin_mean,
]
cdf_headers, cdf_dtype, cdf_fmt = generate_output_metadata(cdf_output)

complex_items_meta_output = [
    item_id,
    coverage_id,
    group_id,
    model_data_len,
]
complex_items_meta_headers, complex_items_meta_dtype, complex_items_meta_fmt = generate_output_metadata(complex_items_meta_output)

correlations_output = [
    item_id,
    peril_correlation_group,
    damage_correlation_value,
    hazard_group_id,
    hazard_correlation_value,
]
correlations_headers, correlations_dtype, correlations_fmt = generate_output_metadata(correlations_output)

coverages_output = [
    coverage_id,
    tiv,
]
coverages_headers, coverages_dtype, coverages_fmt = generate_output_metadata(coverages_output)

damagebin_output = [
    bin_index,
    bin_from,
    bin_to,
    interpolation,
    damage_type,
]
damagebin_headers, damagebin_dtype, damagebin_fmt = generate_output_metadata(damagebin_output)

eve_output = [
    event_id
]
eve_headers, eve_dtype, eve_fmt = generate_output_metadata(eve_output)

footprint_event_output = [
    event_id,
    areaperil_id,
    intensity_bin_id,
    probability,
]
footprint_event_headers, footprint_event_dtype, footprint_event_fmt = generate_output_metadata(footprint_event_output)

fm_output = [
    event_id,
    output_id,
    sidx,
    loss,
]
fm_headers, fm_dtype, fm_fmt = generate_output_metadata(fm_output)

fm_policytc_output = [
    level_id,
    agg_id,
    layer_id,
    profile_id,
]
fm_policytc_headers, fm_policytc_dtype, fm_policytc_fmt = generate_output_metadata(fm_policytc_output)

fm_profile_output = [
    profile_id,
    calcrule_id,
    deductible1,
    deductible2,
    deductible3,
    attachment1,
    limit1,
    share1,
    share2,
    share3,
]
fm_profile_headers, fm_profile_dtype, fm_profile_fmt = generate_output_metadata(fm_profile_output)

fm_profile_step_output = [
    profile_id,
    calcrule_id,
    deductible1,
    deductible2,
    deductible3,
    attachment1,
    limit1,
    share1,
    share2,
    share3,
    step_id,
    trigger_start,
    trigger_end,
    payout_start,
    payout_end,
    limit2,
    scale1,
    scale2,
]
fm_profile_step_headers, fm_profile_step_dtype, fm_profile_step_fmt = generate_output_metadata(fm_profile_step_output)

fm_programme_output = [
    from_agg_id,
    level_id,
    to_agg_id,
]
fm_programme_headers, fm_programme_dtype, fm_programme_fmt = generate_output_metadata(fm_programme_output)

fm_summary_xref_output = [
    output,
    summary_id,
    summaryset_id
]
fm_summary_xref_headers, fm_summary_xref_dtype, fm_summary_xref_fmt = generate_output_metadata(fm_summary_xref_output)

fm_xref_output = [
    output,
    agg_id,
    layer_id,
]
fm_xref_headers, fm_xref_dtype, fm_xref_fmt = generate_output_metadata(fm_xref_output)

gul_output = [
    event_id,
    item_id,
    sidx,
    loss,
]
gul_headers, gul_dtype, gul_fmt = generate_output_metadata(gul_output)

gul_summary_xref_output = [
    item_id,
    summary_id,
    summaryset_id
]
gul_summary_xref_headers, gul_summary_xref_dtype, gul_summary_xref_fmt = generate_output_metadata(gul_summary_xref_output)

items_output = [
    item_id,
    coverage_id,
    areaperil_id,
    vulnerability_id,
    group_id
]
items_headers, items_dtype, items_fmt = generate_output_metadata(items_output)

item_adjustment = [
    item_id,
    intensity_adjustment,
    item_return_period
]

item_adjustment_headers, item_adjustment_dtype, item_adjustment_fmt = generate_output_metadata(item_adjustment)

lossfactors_output = [
    event_id,
    amplification_id,
    factor,
]
lossfactors_headers, lossfactors_dtype, lossfactors_fmt = generate_output_metadata(lossfactors_output)

occurrence_output = [
    event_id,
    period_no,
    occ_date_id,
]
occurrence_headers, occurrence_dtype, occurrence_fmt = generate_output_metadata(occurrence_output)

occurrence_granular_output = [
    event_id,
    period_no,
    occ_date_id_granular,
]
occurrence_granular_headers, occurrence_granular_dtype, occurrence_granular_fmt = generate_output_metadata(occurrence_granular_output)

periods_output = [
    period_no,
    period_weighting,
]
periods_headers, periods_dtype, periods_fmt = generate_output_metadata(periods_output)

quantile_output = [
    quantile,
]
quantile_headers, quantile_dtype, quantile_fmt = generate_output_metadata(quantile_output)

quantile_interval_output = quantile_output + [
    quantile_integer_part,
    quantile_fractional_part,
]
quantile_interval_headers, quantile_interval_dtype, quantile_interval_fmt = generate_output_metadata(quantile_interval_output)

random_output = [
    random_no,
]
random_headers, random_dtype, random_fmt = generate_output_metadata(random_output)

returnperiods_output = [
    return_period,
]
returnperiods_headers, returnperiods_dtype, returnperiods_fmt = generate_output_metadata(returnperiods_output)

vulnerability_output = [
    vulnerability_id,
    intensity_bin_id,
    damage_bin_id,
    probability,
]
vulnerability_headers, vulnerability_dtype, vulnerability_fmt = generate_output_metadata(vulnerability_output)

vulnerability_weight_output = [
    areaperil_id,
    vulnerability_id,
    vulnerability_weight,
]
vulnerability_weight_headers, vulnerability_weight_dtype, vulnerability_weight_fmt = generate_output_metadata(vulnerability_weight_output)


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
        return np.memmap(os.path.join(dir_path, name + '.bin'), dtype=_dtype, mode='r')
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
        return np.memmap(fp, dtype=_dtype, mode='r')
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


def resolve_file(path, mode, stack):
    """Resolve file path to open file or use sys.stdin

    Args:
        path (str | os.PathLike): File path or "-" indicationg standard input/output.
        mode (str): Mode to open file ("r", "rb, "w", "wb").
        stack (ExitStack): Context manager stack used to manage file lifecycle.

    Returns:
        file (IO): A file-like object opened in the specified mode.
    """
    is_read = "r" in mode
    is_binary = "b" in mode

    if str(path) == "-":
        if is_read:
            return sys.stdin.buffer if is_binary else sys.stdin
        else:
            return sys.stdout.buffer if is_binary else sys.stdout
    else:
        return stack.enter_context(open(path, mode))
