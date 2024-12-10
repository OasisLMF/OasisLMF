import logging
import os
import numba as nb
import numpy as np
import pandas as pd
from pathlib import Path


logger = logging.getLogger(__name__)

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


summary_xref_dtype = np.dtype([('item_id', 'i4'), ('summary_id', 'i4'), ('summary_set_id', 'i4')])

# financial structure static input dtypes
fm_programme_dtype = np.dtype([('from_agg_id', 'i4'), ('level_id', 'i4'), ('to_agg_id', 'i4')])
fm_policytc_dtype = np.dtype([('level_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4'), ('profile_id', 'i4')])
fm_profile_dtype = np.dtype([('profile_id', 'i4'),
                             ('calcrule_id', 'i4'),
                             ('deductible_1', 'f4'),
                             ('deductible_2', 'f4'),
                             ('deductible_3', 'f4'),
                             ('attachment_1', 'f4'),
                             ('limit_1', 'f4'),
                             ('share_1', 'f4'),
                             ('share_2', 'f4'),
                             ('share_3', 'f4'),
                             ])
fm_profile_step_dtype = np.dtype([('profile_id', 'i4'),
                                  ('calcrule_id', 'i4'),
                                  ('deductible_1', 'f4'),
                                  ('deductible_2', 'f4'),
                                  ('deductible_3', 'f4'),
                                  ('attachment_1', 'f4'),
                                  ('limit_1', 'f4'),
                                  ('share_1', 'f4'),
                                  ('share_2', 'f4'),
                                  ('share_3', 'f4'),
                                  ('step_id', 'i4'),
                                  ('trigger_start', 'f4'),
                                  ('trigger_end', 'f4'),
                                  ('payout_start', 'f4'),
                                  ('payout_end', 'f4'),
                                  ('limit_2', 'f4'),
                                  ('scale_1', 'f4'),
                                  ('scale_2', 'f4'),
                                  ])
fm_profile_csv_col_map = {
    'deductible_1': 'deductible1',
    'deductible_2': 'deductible2',
    'deductible_3': 'deductible3',
    'attachment_1': 'attachment1',
    'limit_1': 'limit1',
    'share_1': 'share1',
    'share_2': 'share2',
    'share_3': 'share3',
    'limit_2': ' limit2',
    'scale_1': 'scale1',
    'scale_2': 'scale2',
}
fm_xref_dtype = np.dtype([('output_id', 'i4'), ('agg_id', 'i4'), ('layer_id', 'i4')])
fm_xref_csv_col_map = {'output_id': 'output'}

coverages_dtype = np.dtype([('coverage_id', 'i4'), ('tiv', 'f4')])

items_dtype = np.dtype([('item_id', 'i4'),
                        ('coverage_id', 'i4'),
                        ('areaperil_id', areaperil_int),
                        ('vulnerability_id', 'i4'),
                        ('group_id', 'i4')])

# Mean type numbers for outputs (SampleType)
MEAN_TYPE_ANALYTICAL = 1
MEAN_TYPE_SAMPLE = 2


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


float_equal_precision = np.finfo(oasis_float).eps


@nb.njit(cache=True)
def almost_equal(a, b):
    return abs(a - b) < float_equal_precision


def read_event_rates(run_dir, filename="event_rates.csv"):
    """Reads event rates from a CSV file
    Args:
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): event rates csv file name
    Returns:
        unique_event_ids (ndarray[oasis_int]): unique event ids
        event_rates (ndarray[oasis_float]): event rates
    """
    try:
        event_rate_file = Path(run_dir, filename)
        data = load_as_ndarray(run_dir, filename[:-4], np.dtype([('event_id', oasis_int), ('rate', oasis_float)]),
                               col_map={"event_id": "EventIds", "rate": "Event_rates"})
        if data is None or data.size == 0:
            logger.info(f"Event rate file {event_rate_file} is empty, proceeding without event rates.")
            return np.array([], dtype=oasis_int), np.array([], dtype=oasis_float)
        unique_event_ids = data['event_id']
        event_rates = data['rate']

        # Make sure event_ids are sorted
        sort_idx = np.argsort(unique_event_ids)
        unique_event_ids = unique_event_ids[sort_idx]
        event_rates = event_rates[sort_idx]
        return unique_event_ids, event_rates
    except Exception as e:
        logger.warning(f"An error occurred while reading event rate file: {str(e)}")
        return np.array([], dtype=oasis_int), np.array([], dtype=oasis_float)


def read_quantile(sample_size, run_dir, filename="quantile.bin", return_empty=False):
    """Generate a quantile interval Dictionary based on sample size and quantile binary file
    Args:
        sample_size (int): Sample size
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): quantile binary file name
        return_empty (bool): return an empty intervals array regardless of the existence of the quantile binary
    Returns:
        intervals (quantile_interval_dtype): Numpy array emulating a dictionary for numba
    """
    intervals = []
    quantile_interval_dtype = np.dtype([
        ('q', oasis_float),
        ('integer_part', oasis_int),
        ('fractional_part', oasis_float),
    ])

    if return_empty:
        return np.array([], dtype=quantile_interval_dtype)
    try:
        data = load_as_ndarray(run_dir, filename[:-4], np.float32)
        for q in data:
            # Calculate interval index and fractional part
            pos = (sample_size - 1) * q + 1
            integer_part = int(pos)
            fractional_part = pos - integer_part
            intervals.append((q, integer_part, fractional_part))
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")

    # Convert to numpy array
    intervals = np.array(intervals, dtype=quantile_interval_dtype)
    return intervals


def read_occurrence(run_dir, filename="occurrence.bin"):
    """Read the occurrence binary file and returns an occurrence map
    Args:
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): occurrence binary file name
    Returns:
        occ_map (ndarray[occ_map_dtype]): numpy map of event_id, period_no, occ_date_id from the occurrence file
    """
    from .event_stream import mv_read
    try:
        occurrence_fp = Path(run_dir, filename)
        fin = np.memmap(occurrence_fp, mode="r", dtype="u1")
        cursor = 0
        valid_buff = len(fin)

        if valid_buff - cursor < np.dtype(np.int32).itemsize:
            raise RuntimeError("Error: broken occurrence file, not enough data")
        date_opts, cursor = mv_read(fin, cursor, np.int32, np.dtype(np.int32).itemsize)

        date_algorithm = date_opts & 1
        granular_date = date_opts >> 1

        # (event_id: int, period_no: int, occ_date_id: int)
        record_size = np.dtype(np.int32).itemsize * 3
        # (event_id: int, period_no: int, occ_date_id: long long)
        if granular_date:
            record_size = np.dtype(np.int32).itemsize * 2 + np.dtype(np.int64).itemsize
        # Should not get here
        if not date_algorithm and granular_date:
            raise RuntimeError("FATAL: Unknown date algorithm")

        # Extract no_of_periods
        if valid_buff - cursor < np.dtype(np.int32).itemsize:
            raise RuntimeError("Error: broken occurrence file, not enough data")
        no_of_periods, cursor = mv_read(fin, cursor, np.int32, np.dtype(np.int32).itemsize)

        num_records = (valid_buff - cursor) // record_size
        if (valid_buff - cursor) % record_size != 0:
            logger.warning(
                f"Occurrence File size (num_records: {num_records}) does not align with expected record size (record_size: {record_size})"
            )

        occ_map_dtype = np.dtype([
            ("event_id", np.int32),
            ("period_no", np.int32),
            ("occ_date_id", np.int32),
        ])
        if granular_date:
            occ_map_dtype = np.dtype([
                ("event_id", np.int32),
                ("period_no", np.int32),
                ("occ_date_id", np.int64),
            ])

        occ_map = np.zeros(num_records, dtype=occ_map_dtype)

        for i in range(num_records):
            if valid_buff - cursor < record_size:
                break  # Not enough data left
            event_id, cursor = mv_read(fin, cursor, np.int32, np.dtype(np.int32).itemsize)
            period_no, cursor = mv_read(fin, cursor, np.int32, np.dtype(np.int32).itemsize)
            if granular_date:
                occ_date_id, cursor = mv_read(fin, cursor, np.int64, np.dtype(np.int64).itemsize)
            else:
                occ_date_id, cursor = mv_read(fin, cursor, np.int32, np.dtype(np.int32).itemsize)
            occ_map[i] = (event_id, period_no, occ_date_id)

        return occ_map, date_algorithm, granular_date, no_of_periods
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


def read_periods(no_of_periods, run_dir, filename="periods.bin"):
    """Returns an array of period weights for each period between 1 and no_of_periods inclusive (with no gaps).
    Args:
        no_of_periods (int): Number of periods
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): occurrence binary file name
    Returns:
        period_weights (ndarray[period_weights_dtype]): Returns the period weights
    """
    periods_fp = Path(run_dir, filename)
    period_weights_dtype = np.dtype([
        ("period_no", np.int32),
        ("weighting", np.float64),
    ])

    if not periods_fp.exists():
        # If no periods binary file found, the revert to using period weights reciprocal to no_of_periods
        logger.warning(f"Periods file not found at {periods_fp}, using reciprocal calculated period weights based on no_of_periods {no_of_periods}")
        period_weights = np.array(
            [(i + 1, 1 / no_of_periods) for i in range(no_of_periods)],
            dtype=period_weights_dtype
        )
        return period_weights

    try:
        data = load_as_ndarray(run_dir, filename[:-4], period_weights_dtype)
        # Less data than no_of_periods
        if len(data) != no_of_periods:
            raise RuntimeError(f"ERROR: no_of_periods does not match total period_no in period binary file {periods_fp}.")

        # Sort by period_no
        period_weights = np.sort(data, order="period_no")

        # Identify any missing periods
        expected_periods = np.arange(1, no_of_periods + 1)
        actual_periods = period_weights['period_no']
        missing_periods = np.setdiff1d(expected_periods, actual_periods)
        if len(missing_periods) > 0:
            raise RuntimeError(f"ERROR: Missing period_no in period binary file {periods_fp}.")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")
    return period_weights
