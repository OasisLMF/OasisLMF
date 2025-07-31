from contextlib import ExitStack
import logging
import sys
import numba as nb
import numpy as np
from pathlib import Path

from oasislmf.pytools.common.data import (
    load_as_ndarray, nb_oasis_int,
    correlations_headers, correlations_dtype, coverages_headers,
    occurrence_dtype, occurrence_granular_dtype, periods_dtype, quantile_dtype,
    quantile_interval_dtype, returnperiods_dtype
)
from oasislmf.pytools.common.event_stream import mv_read, oasis_int, oasis_float


logger = logging.getLogger(__name__)

# Input file names (input/<file_name>)
AMPLIFICATIONS_FILE = "amplifications.bin"
CORRELATIONS_FILENAME = "correlations.bin"
COVERAGES_FILE = "coverages.bin"
EVENTRATES_FILE = "event_rates.csv"
FMPOLICYTC_FILE = "fm_policytc.bin"
FMPROGRAMME_FILE = "fm_programme.bin"
FMPROFILE_FILE = "fm_profile.bin"
FMSUMMARYXREF_FILE = "fmsummaryxref.bin"
FMXREF_FILE = "fmxref.bin"
GULSUMMARYXREF_FILE = "gulsummaryxref.bin"
ITEMS_FILE = "items.bin"
OCCURRENCE_FILE = "occurrence.bin"
PERIODS_FILE = "periods.bin"
QUANTILE_FILE = "quantile.bin"
RETURNPERIODS_FILE = "returnperiods.bin"


def read_amplifications(run_dir="", filename=AMPLIFICATIONS_FILE, use_stdin=False):
    """
    Get array of amplification IDs from amplifications.bin, where index
    corresponds to item ID.

    amplifications.bin is binary file with layout:
        reserved header (4-byte int),
        item ID 1 (4-byte int), amplification ID a_1 (4-byte int),
        ...
        item ID n (4-byte int), amplification ID a_n (4-byte int)

    Args:
        run_dir (str): path to amplifications.bin file
        filename (str | os.PathLike): amplifications file name
        use_stdin (bool): Use standard input for file data, ignores run_dir/filename. Defaults to False.
    Returns:
        items_amps (numpy.ndarray): array of amplification IDs, where index
            corresponds to item ID
    """
    header_size = 4
    if use_stdin:
        items_amps = np.frombuffer(sys.stdin.buffer.read(), dtype=np.int32, offset=header_size)
    else:
        amplification_file = Path(run_dir, filename)
        if not amplification_file.exists():
            raise FileNotFoundError('amplifications file not found.')
        items_amps = np.fromfile(amplification_file, dtype=np.int32, offset=header_size)

    # Check item IDs start from 1 and are contiguous
    if items_amps[0] != 1:
        raise ValueError(f'First item ID is {items_amps[0]}. Expected 1.')
    items_amps = items_amps.reshape(len(items_amps) // 2, 2)
    if not np.all(items_amps[1:, 0] - items_amps[:-1, 0] == 1):
        raise ValueError('Item IDs in amplifications file are not contiguous')

    items_amps = np.concatenate((np.array([0]), items_amps[:, 1]))

    return items_amps


def read_correlations(run_dir, ignore_file_type=set(), filename=CORRELATIONS_FILENAME):
    """Load the correlations from the correlations file.
    Args:
        run_dir (str): path to correlations file
        ignore_file_type (Set[str]): file extension to ignore when loading.
        filename (str | os.PathLike): correlations file name
    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
        vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
        areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """

    for ext in ["bin", "csv"]:
        if ext in ignore_file_type:
            continue

        correlations_file = Path(run_dir, filename).with_suffix("." + ext)
        if correlations_file.exists():
            logger.debug(f"loading {correlations_file}")
            if ext == "bin":
                try:
                    correlations = np.memmap(correlations_file, dtype=correlations_dtype, mode='r')
                except ValueError:
                    logger.debug("binary file is empty, numpy.memmap failed. trying to read correlations.csv.")
                    correlations = read_correlations(run_dir, ignore_file_type={'bin'}, filename=correlations_file.with_suffix(".csv").name)
            elif ext == "csv":
                # Check for header
                with open(correlations_file, "r") as fin:
                    first_line = fin.readline()
                    first_line_elements = [header.strip() for header in first_line.strip().split(',')]
                    has_header = first_line_elements == correlations_headers
                correlations = np.loadtxt(
                    correlations_file,
                    dtype=correlations_dtype,
                    delimiter=",",
                    skiprows=1 if has_header else 0,
                    ndmin=1
                )
            else:
                raise RuntimeError(f"Cannot read correlations file of type {ext}. Not Implemented.")
            return correlations

    raise FileNotFoundError(f'correlations file not found at {run_dir}. Ignoring files with ext {ignore_file_type}.')


def read_coverages(run_dir="", ignore_file_type=set(), filename=COVERAGES_FILE, use_stdin=False):
    """Load the coverages from the coverages file.
    Args:
        run_dir (str): path to coverages file
        ignore_file_type (Set[str]): file extension to ignore when loading.
        filename (str | os.PathLike): coverages file name
        use_stdin (bool): Use standard input for file data, ignores run_dir/filename. Defaults to False.
    Returns:
        numpy.array[oasis_float]: array with the coverage values for each coverage_id.
    """
    for ext in ["bin", "csv"]:
        if ext in ignore_file_type:
            continue

        coverages_file = Path(run_dir, filename).with_suffix("." + ext)
        if coverages_file.exists():
            logger.debug(f"loading {coverages_file}")
            if ext == "bin":
                if use_stdin:
                    coverages = np.frombuffer(sys.stdin.buffer.read(), dtype=oasis_float)
                else:
                    coverages = np.fromfile(coverages_file, dtype=oasis_float)
            elif ext == "csv":
                with ExitStack() as stack:
                    if use_stdin:
                        fin = sys.stdin
                    else:
                        fin = stack.enter_context(open(coverages_file, "r"))

                    lines = fin.readlines()
                    # Check for header
                    first_line_elements = [header.strip() for header in lines[0].strip().split(',')]
                    has_header = first_line_elements == coverages_headers

                    data_lines = lines[1:] if has_header else lines
                    coverages = np.loadtxt(
                        data_lines,
                        dtype=oasis_float,
                        delimiter=",",
                        ndmin=1
                    )[:, 1]
            else:
                raise RuntimeError(f"Cannot read coverages file of type {ext}. Not Implemented.")
            return coverages

    raise FileNotFoundError(f'coverages file not found at {run_dir}. Ignoring files with ext {ignore_file_type}.')


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

    if return_empty:
        return np.array([], dtype=quantile_interval_dtype)
    data = load_as_ndarray(run_dir, filename[:-4], quantile_dtype, must_exist=True)
    for row in data:
        q = row["quantile"]
        # Calculate interval index and fractional part
        pos = (sample_size - 1) * q + 1
        integer_part = int(pos)
        fractional_part = pos - integer_part
        intervals.append((q, integer_part, fractional_part))

    # Convert to numpy array
    intervals = np.array(intervals, dtype=quantile_interval_dtype)
    return intervals


def read_occurrence_bin(run_dir="", filename=OCCURRENCE_FILE, use_stdin=False):
    """Read the occurrence binary file and returns an occurrence map
    Args:
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): occurrence binary file name
        use_stdin (bool): Use standard input for file data, ignores run_dir/filename. Defaults to False.
    Returns:
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
    """
    occurrence_fp = Path(run_dir, filename)
    if use_stdin:
        fin = np.frombuffer(sys.stdin.buffer.read(), dtype="u1")
    else:
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

    occ_dtype = occurrence_dtype
    if granular_date:
        occ_dtype = occurrence_granular_dtype
    occ_arr = np.zeros(0, dtype=occ_dtype)

    if num_records > 0:
        occ_arr = np.frombuffer(fin[cursor:cursor + num_records * record_size], dtype=occ_dtype)

    return occ_arr, date_algorithm, granular_date, no_of_periods


def read_occurrence(run_dir, filename=OCCURRENCE_FILE):
    """Read the occurrence binary file and returns an occurrence map
    Args:
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): occurrence binary file name
    Returns:
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
    """
    occ_arr, date_algorithm, granular_date, no_of_periods = read_occurrence_bin(run_dir, filename=filename)

    occ_dtype = occurrence_dtype
    if granular_date:
        occ_dtype = occurrence_granular_dtype

    occ_map_valtype = occ_dtype[["period_no", "occ_date_id"]]
    NB_occ_map_valtype = nb.types.Array(nb.from_dtype(occ_map_valtype), 1, "C")

    occ_map = _read_occ_arr(occ_arr, occ_map_valtype, NB_occ_map_valtype)

    return occ_map, date_algorithm, granular_date, no_of_periods


@nb.njit(cache=True, error_model="numpy")
def _read_occ_arr(occ_arr, occ_map_valtype, NB_occ_map_valtype):
    """Reads occurrence file array and returns an occurrence map of event_id to list of (period_no, occ_date_id)
    """
    occ_map = nb.typed.Dict.empty(nb_oasis_int, NB_occ_map_valtype)
    occ_map_sizes = nb.typed.Dict.empty(nb_oasis_int, nb.types.int64)
    for row in occ_arr:
        event_id = row["event_id"]
        if event_id not in occ_map:
            occ_map[event_id] = np.zeros(8, dtype=occ_map_valtype)
            occ_map_sizes[event_id] = 0
        array = occ_map[event_id]
        current_size = occ_map_sizes[event_id]

        if current_size >= len(array):  # Resize if the array is full
            new_array = np.empty(len(array) * 2, dtype=occ_map_valtype)
            new_array[:len(array)] = array
            array = new_array

        occ_map_current_size = occ_map_sizes[event_id]
        array[occ_map_current_size]["period_no"] = row["period_no"]
        array[occ_map_current_size]["occ_date_id"] = row["occ_date_id"]
        occ_map[event_id] = array
        occ_map_sizes[event_id] += 1

    for event_id in occ_map:
        occ_map[event_id] = occ_map[event_id][:occ_map_sizes[event_id]]

    return occ_map


@nb.njit(cache=True)
def occ_get_date(occ_date_id, granular_date):
    """Returns date as year, month, day, hour, minute from occ_date_id

    Args:
        occ_date_id (np.int32 | np.int64): occurrence file date id (int64 for granular dates)
        granular_date (bool): boolean for whether granular date should be extracted or not

    Returns:
        (oasis_int, oasis_int, oasis_int, oasis_int, oasis_int): Returns year, month, date, hour, minute
    """
    days = occ_date_id / (1440 - 1439 * (not granular_date))

    # Function void d(long long g, int& y, int& mm, int& dd) taken from pltcalc.cpp
    y = (10000 * days + 14780) // 3652425
    ddd = days - (365 * y + y // 4 - y // 100 + y // 400)
    if ddd < 0:
        y = y - 1
        ddd = days - (365 * y + y // 4 - y // 100 + y // 400)
    mi = (100 * ddd + 52) // 3060
    mm = (mi + 2) % 12 + 1
    y = y + (mi + 2) // 12
    dd = ddd - (mi * 306 + 5) // 10 + 1

    minutes = (occ_date_id % 1440) * granular_date
    occ_hour = minutes // 60
    occ_minutes = minutes % 60

    return y, mm, dd, occ_hour, occ_minutes


@nb.njit(cache=True)
def occ_get_date_id(granular_date, occ_year, occ_month, occ_day, occ_hour=0, occ_minute=0):
    """Returns the occ_date_id from year, month, day, hour, minute and whether it is a granular date
    Args:
        granular_date (bool): boolean for whether granular date should be extracted or not
        occ_year (int): Occurrence Year.
        occ_month (int): Occurrence Month.
        occ_day (int): Occurrence Day.
        occ_hour (int): Occurrence Hour. Defaults to 0.
        occ_minute (int): Occurrence Minute. Defaults to 0.
    Returns:
        occ_date_id (np.int64): occurrence file date id (int64 for granular dates)
    """
    occ_month = (occ_month + 9) % 12
    occ_year = occ_year - occ_month // 10
    occ_date_id = np.int64(
        365 * occ_year + occ_year // 4 - occ_year // 100 + occ_year // 400 + (occ_month * 306 + 5) // 10 + (occ_day - 1)
    )

    occ_date_id *= (1440 // (1440 - 1439 * granular_date))
    occ_date_id += (60 * occ_hour + occ_minute)
    return occ_date_id


def read_periods(no_of_periods, run_dir, filename=PERIODS_FILE):
    """Returns an array of period weights for each period between 1 and no_of_periods inclusive (with no gaps).
    Args:
        no_of_periods (int): Number of periods
        run_dir (str | os.PathLike): Path to input files dir
        filename (str | os.PathLike): periods binary file name
    Returns:
        period_weights (ndarray[periods_dtype]): Period weights
    """
    periods_fp = Path(run_dir, filename)

    if not periods_fp.exists():
        # If no periods binary file found, the revert to using period weights reciprocal to no_of_periods
        logger.warning(f"Periods file not found at {periods_fp}, using reciprocal calculated period weights based on no_of_periods {no_of_periods}")
        period_weights = np.array(
            [(i + 1, 1 / no_of_periods) for i in range(no_of_periods)],
            dtype=periods_dtype
        )
        return period_weights

    data = load_as_ndarray(run_dir, filename[:-4], periods_dtype, must_exist=True)
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


def read_returnperiods(use_return_period_file, run_dir, filename=RETURNPERIODS_FILE):
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
        return np.array([], dtype=returnperiods_dtype)["return_period"], use_return_period_file
    returnperiods_fp = Path(run_dir, filename)

    if not returnperiods_fp.exists():
        raise RuntimeError(f"ERROR: Return Periods file not found at {returnperiods_fp}.")

    returnperiods = load_as_ndarray(
        run_dir,
        filename[:-4],
        returnperiods_dtype,
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
    for rp in returnperiods["return_period"]:
        if lastrp != -1 and lastrp <= rp:
            raise RuntimeError(f"ERROR: Invalid return periods file. Non-decreasing return periods found: {returnperiods}")
        lastrp = rp

    return returnperiods["return_period"], use_return_period_file
