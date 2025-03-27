import logging
import numpy as np
from pathlib import Path

from oasislmf.pytools.common.data import load_as_ndarray
from oasislmf.pytools.common.event_stream import mv_read, oasis_int, oasis_float


logger = logging.getLogger(__name__)

# Input file names (input/<file_name>)
EVENTRATES_FILE = "event_rates.csv"
OCCURRENCE_FILE = "occurrence.bin"
PERIODS_FILE = "periods.bin"
QUANTILE_FILE = "quantile.bin"
RETURNPERIODS_FILE = "returnperiods.bin"


def read_event_rates(run_dir, filename=EVENTRATES_FILE):
    """Reads event rates from a CSV file
    Args:
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): event rates csv file name
    Returns:
        unique_event_ids (ndarray[oasis_int]): unique event ids
        event_rates (ndarray[oasis_float]): event rates
    """
    event_rate_file = Path(run_dir, filename)
    data = load_as_ndarray(
        run_dir,
        filename[:-4],
        np.dtype([('event_id', oasis_int), ('rate', oasis_float)]),
        must_exist=False,
        col_map={"event_id": "EventIds", "rate": "Event_rates"}
    )
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


def read_quantile(sample_size, run_dir, filename=QUANTILE_FILE, return_empty=False):
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
    data = load_as_ndarray(run_dir, filename[:-4], np.float32, must_exist=True)
    for q in data:
        # Calculate interval index and fractional part
        pos = (sample_size - 1) * q + 1
        integer_part = int(pos)
        fractional_part = pos - integer_part
        intervals.append((q, integer_part, fractional_part))

    # Convert to numpy array
    intervals = np.array(intervals, dtype=quantile_interval_dtype)
    return intervals


def read_occurrence(run_dir, filename=OCCURRENCE_FILE):
    """Read the occurrence binary file and returns an occurrence map
    Args:
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): occurrence binary file name
    Returns:
        occ_map (ndarray[occ_map_dtype]): numpy map of event_id, period_no, occ_date_id from the occurrence file
    """
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

    occ_map = np.zeros(0, dtype=occ_map_dtype)

    if num_records > 0:
        occ_map = np.frombuffer(fin[cursor:cursor + num_records * record_size], dtype=occ_map_dtype)

    return occ_map, date_algorithm, granular_date, no_of_periods


def read_periods(no_of_periods, run_dir, filename=PERIODS_FILE):
    """Returns an array of period weights for each period between 1 and no_of_periods inclusive (with no gaps).
    Args:
        no_of_periods (int): Number of periods
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): periods binary file name
    Returns:
        period_weights (ndarray[period_weights_dtype]): Period weights
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

    data = load_as_ndarray(run_dir, filename[:-4], period_weights_dtype, must_exist=True)
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
    return period_weights


def read_return_periods(use_return_period_file, run_dir, filename=RETURNPERIODS_FILE):
    """Returns an array of return periods decreasing order with no duplicates.
    Args:
        use_return_period_file (bool): Bool to use Return Period File
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): return periods binary file name
    Returns:
        return_periods (ndarray[np.int32]): Return Periods
        use_return_period_file (bool): Bool to use Return Period File
    """
    if not use_return_period_file:
        return np.array([], dtype=np.int32), use_return_period_file
    returnperiods_fp = Path(run_dir, filename)

    if not returnperiods_fp.exists():
        raise RuntimeError(f"ERROR: Return Periods file not found at {returnperiods_fp}.")

    returnperiods = load_as_ndarray(
        run_dir,
        filename[:-4],
        np.int32,
        must_exist=False
    )

    if len(returnperiods) == 0:
        logger.warning(f"WARNING: Empty return periods file at {returnperiods_fp}. Running without defined return periods option")
        return None, False

    # Check return periods validity
    # Return periods should be unique and decreasing in order
    if len(returnperiods) != len(np.unique(returnperiods)):
        raise RuntimeError(f"ERROR: Invalid return periods file. Duplicate return periods found: {returnperiods}")
    lastrp = -1
    for rp in returnperiods:
        if lastrp != -1 and lastrp <= rp:
            raise RuntimeError(f"ERROR: Invalid return periods file. Non-decreasing return periods found: {returnperiods}")
        lastrp = rp

    return returnperiods, use_return_period_file
