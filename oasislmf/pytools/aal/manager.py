# aal/manager.py

import logging
import os
from sys import byteorder
from line_profiler import profile
import numpy as np
import numba as nb
import struct
from contextlib import ExitStack
from pathlib import Path

from oasislmf.pytools.common.data import (oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, NUM_SPECIAL_SIDX, NUMBER_OF_AFFECTED_RISK_IDX, EventReader, init_streams_in, mv_read)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


_MEAN_TYPE_ANALYTICAL = 1
_MEAN_TYPE_SAMPLE = 2

_AAL_REC_DTYPE = np.dtype([
    ('summary_id', np.int32),  # TODO: remove this as can be infered from max_summary_id
    ('type', np.int32),  # TODO: remove this as can be infered from which array is being used
    ('mean', np.float64),
    ('mean_squared', np.float64),
])

_AAL_REC_PERIOD_DTYPE = np.dtype(
    _AAL_REC_DTYPE.descr + [('mean_period', np.float64)]
)

_VECS_SAMPLE_AAL_DTYPE = np.dtype(
    [('subset_size', np.int32)] + _AAL_REC_PERIOD_DTYPE.descr  # TODO: remove this as can be infered from modulo?
)

_VREC_DTYPE = np.dtype([
    ('sidx', oasis_int),
    ('loss', oasis_float),
])

_SUMMARIES_DTYPE = np.dtype([
    ("summary_id", np.int32),
    ("file_idx", np.int32),
    ("period_no", np.int32),
    ("file_offset", np.int64),
])


def read_occurrence(occurrence_fp):
    """Read the occurrence binary file and returns an occurrence map
    Args:
        occurrence_fp (str | os.PathLike): Path to the occurrence binary file
    Returns:
        occ_map (ndarray[occ_map_dtype]): numpy map of event_id, period_no, occ_date_id from the occurrence file
    """
    try:
        with open(occurrence_fp, "rb") as fin:
            # Extract Date Options
            date_opts = fin.read(4)
            if not date_opts or len(date_opts) < 4:
                raise RuntimeError("Occurrence file is empty or currupted")
            date_opts = int.from_bytes(date_opts, byteorder="little", signed=True)

            date_algorithm = date_opts & 1  # Unused as granular_date not supported
            granular_date = date_opts >> 1

            # (event_id: int, period_no: int, occ_date_id: int)
            record_format = "<iii"
            # (event_id: int, period_no: int, occ_date_id: long long)
            if granular_date:
                record_format = "<iiq"
            record_size = struct.calcsize(record_format)

            # Should not get here
            if not date_algorithm and granular_date:
                raise RuntimeError("FATAL: Unknown date algorithm")

            # Extract no_of_periods
            no_of_periods = fin.read(4)
            if not no_of_periods or len(no_of_periods) < 4:
                raise RuntimeError("Occurrence file is empty or currupted")
            no_of_periods = int.from_bytes(no_of_periods, byteorder="little", signed=True)

            data = fin.read()

        num_records = len(data) // record_size
        if len(data) % record_size != 0:
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
            offset = i * record_size
            curr_data = data[offset:offset + record_size]
            if len(curr_data) < record_size:
                break
            event_id, period_no, occ_date_id = struct.unpack(record_format, curr_data)
            occ_map[i] = (event_id, period_no, occ_date_id)

        return occ_map, date_algorithm, granular_date, no_of_periods
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL: Error opening file {occurrence_fp}")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


def read_periods(periods_fp, no_of_periods):
    """Returns an array of period weights for each period between 1 and no_of_periods inclusive (with no gaps).
    Args:
        periods_fp (str | os.PathLike): Path to periods binary file
        no_of_periods (int): Number of periods
    Returns:
        period_weights (ndarray[period_weights_dtype]): Returns the period weights
    """
    period_weights_dtype = np.dtype([
        ("period_no", np.int32),
        ("weighting", np.float64),
    ])

    period_weights = np.zeros(no_of_periods, dtype=period_weights_dtype)

    try:
        with open(periods_fp, "rb") as fin:
            record_format = "<id"  # int, double
            record_size = struct.calcsize(record_format)
            num_read = 0
            while True:
                record_data = fin.read(record_size)

                if not record_data:
                    break

                period_no, weighting = struct.unpack(record_format, record_data)

                # Checks for gaps in periods
                if num_read + 1 != period_no:
                    raise RuntimeError(f"ERROR: Missing period_no in period binary file {periods_fp}.")
                num_read += 1

                # More data than no_of_periods
                if num_read > no_of_periods:
                    raise RuntimeError(f"ERROR: no_of_periods does not match total period_no in period binary file {periods_fp}.")

                period_weights[period_no - 1] = (period_no, weighting)

            # Less data than no_of_periods
            if num_read != no_of_periods:
                raise RuntimeError(f"ERROR: no_of_periods does not match total period_no in period binary file {periods_fp}.")
    except FileNotFoundError:
        # If no periods binary file found, the revert to using period weights reciprocal to no_of_periods
        logger.warning(f"Periods file not found at {periods_fp}, using reciprocal calculated period weights based on no_of_periods {no_of_periods}")
        period_weights = np.array(
            [(i + 1, 1 / no_of_periods) for i in range(no_of_periods)],
            dtype=period_weights_dtype
        )
        return period_weights
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")

    return period_weights


def read_input_files(run_dir):
    """Reads all input files and returns a dict of relevant data
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
    """
    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(Path(run_dir, "input", "occurrence.bin"))
    period_weights = read_periods(Path(run_dir, "input", "periods.bin"), no_of_periods)

    file_data = {
        "occ_map": occ_map,
        "date_algorithm": date_algorithm,
        "granular_date": granular_date,
        "no_of_periods": no_of_periods,
        "period_weights": period_weights,
    }
    return file_data


def read_max_summary_idx(workspace_folder):
    """Get the max summary id and summary file list from idx files
    Args:
        workspace_folder (str| os.PathLike): location of the workspace folder
    Returns:
        max_summary_id (int): max summary id int
    """
    max_summary_id_file = Path(workspace_folder, "max_summary_id.idx")

    try:
        with open(max_summary_id_file, "r") as fin:
            line = fin.readline()
            if not line:
                raise ValueError("File is empty or missing data")
            try:
                max_summary_id = int(line.strip())
                return max_summary_id
            except ValueError:
                raise ValueError(f"Invalid data in file: {line.strip()}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot open {max_summary_id_file}")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")


def read_filelist_idx(workspace_folder):
    """Get the max summary id and summary file list from idx files
    Args:
        workspace_folder (str| os.PathLike): location of the workspace folder
    Returns:
        filelist (List[str]): list of summary binary files
    """
    filelist_file = Path(workspace_folder, "filelist.idx")
    filelist = []

    try:
        with open(filelist_file, "r") as fin:
            line = fin.readline()
            if not line:
                raise ValueError("File is empty or missing data")
            while line:
                try:
                    filename = str(line.strip())
                    filelist.append(filename)
                    line = fin.readline()
                except ValueError:
                    raise ValueError(f"Invalid data in file: {line.strip()}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot open {filelist_file}")
    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")

    return filelist


def get_sample_sizes(alct, sample_size, max_summary_id):
    """Generates Sample AAL np map for subset sizes up to sample_size
    Args:
        alct (bool): Boolean for ALCT output
        sample_size (int): Sample size
        max_summary_id (int): Max summary ID
    Returns:
        vec_sample_aal (ndarray[vec_sample_aal_dtype]): A numpy dict for sample AAL values per subset size up to sample_size
    """
    entries = []
    if alct and sample_size > 1:
        i = 0
        while ((1 << i) + ((1 << i) - 1)) <= sample_size:
            data = np.zeros(max_summary_id + 1, dtype=_VECS_SAMPLE_AAL_DTYPE)
            data['subset_size'][:] = [1 << i]
            entries.append(data)
            i += 1

    data = np.zeros(max_summary_id + 1, dtype=_VECS_SAMPLE_AAL_DTYPE)
    data['subset_size'][:] = sample_size
    entries.append(data)

    vecs_sample_aal = np.concatenate(entries, axis=0)
    return vecs_sample_aal

@nb.njit(cache=True, error_model="numpy")
def do_calc_end(
        period_no,
        no_of_periods,
        period_weights,
        sample_size,
        curr_summary_id,
        vec_analytical_aal,
        vecs_sample_aal,
        vec_sample_sum_loss,
    ):
    # Get weighting
    weighting = 1
    if no_of_periods > 0:
        # period_no in period_weights
        if period_no > 0 and period_no <= no_of_periods:
            weighting = period_weights[period_no - 1][1] * no_of_periods
        else:
            weighting = 0

    mean = vec_sample_sum_loss[0]
    aa = vec_analytical_aal[curr_summary_id]
    aa["summary_id"] = curr_summary_id
    aa["type"] = _MEAN_TYPE_ANALYTICAL
    aa["mean"] += mean * weighting
    aa["mean_squared"] += mean * mean * weighting

    idxs = np.where(vecs_sample_aal["subset_size"] == sample_size)[0]
    a_total_idx = idxs[curr_summary_id]
    a_total = vecs_sample_aal[a_total_idx]
    a_total["type"] = _MEAN_TYPE_SAMPLE
    a_total["summary_id"] = curr_summary_id if sample_size != 0 else 0

    # TODO: fix this loop, inner loop does not use curr_summary_id
    total_mean_by_period = 0
    for sidx in range(1, sample_size + 1):
        for iter in vecs_sample_aal:
            if iter["subset_size"] == sample_size:
                break
            mean_by_period = 0
            for sidx in range(iter["subset_size"], iter["subset_size"] << 1):
                mean = vec_sample_sum_loss[sidx]
                iter["type"] = _MEAN_TYPE_SAMPLE
                iter["summary_id"] = curr_summary_id
                iter["mean"] += mean * weighting
                iter["mean_squared"] += mean * mean * weighting
                
                a_total["mean"] += mean * weighting
                a_total["mean_squared"] += mean * mean * weighting
                
                mean_by_period += mean * weighting
                total_mean_by_period += mean * weighting
            iter["mean_period"] += mean_by_period * mean_by_period
        
        mean = vec_sample_sum_loss[sidx]
        total_mean_by_period += mean * weighting
        a_total["mean"] += mean * weighting
        a_total["mean_squared"] += mean * mean * weighting
    a_total["mean_period"] += mean_by_period * mean_by_period
    vec_sample_sum_loss.fill(0)


@nb.njit(cache=True, error_model="numpy")
def do_calc_by_period(
        vrec,
        vec_sample_sum_loss,
    ):
    for rec in vrec:
        loss = rec["loss"]
        if loss > 0:
            type_idx = rec["sidx"] != -1
            sidx = rec["sidx"] if type_idx != 0 else 0
            vec_sample_sum_loss[sidx] += loss


@nb.njit(cache=True, error_model="numpy")
def read_losses(summary_fin, cursor, sample_size):
    # Max losses is sample_size + num special sidxs
    vrec = np.zeros(sample_size + NUM_SPECIAL_SIDX, dtype=_VREC_DTYPE)
    valid_buff = len(summary_fin)
    idx = 0
    while True:
        if valid_buff - cursor < oasis_int_size:
            raise RuntimeError("Error: broken summary file, not enough data")
        sidx, cursor = mv_read(summary_fin, cursor, oasis_int, oasis_int_size)
        
        if valid_buff - cursor < oasis_float_size:
            raise RuntimeError("Error: broken summary file, not enough data")
        loss, cursor = mv_read(summary_fin, cursor, oasis_float, oasis_float_size)

        if sidx == 0:
            break
        if sidx == NUMBER_OF_AFFECTED_RISK_IDX or sidx == MAX_LOSS_IDX:
            continue

        vrec[idx]["sidx"] = sidx
        vrec[idx]["loss"] = loss
        idx += 1
    return vrec[:idx]


# @profile
@nb.njit(cache=True, error_model="numpy")
def run_aal(
        summaries, 
        no_of_periods,
        period_weights,
        sample_size,
        filelist,
        files_handles,
        vec_analytical_aal,
        vecs_sample_aal,
        vec_sample_sum_loss,
    ):
    if len(summaries) == 0:
        raise ValueError("File is empty or missing data")

    lineno = 1
    curr_summary_id = 0
    last_summary_id = -1
    last_period_no = -1
    last_file_idx = -1
    summary_fin = None
    for line in summaries:
        summary_id = line["summary_id"]
        file_idx = line["file_idx"]
        period_no = line["period_no"]
        file_offset = line["file_offset"]

        if last_summary_id != summary_id:
            if last_summary_id != -1:
                do_calc_end(
                    last_period_no,
                    no_of_periods,
                    period_weights,
                    sample_size,
                    curr_summary_id,
                    vec_analytical_aal,
                    vecs_sample_aal,
                    vec_sample_sum_loss,
                )
            last_period_no = -1
            curr_summary_id = summary_id
            last_summary_id = summary_id
        if last_period_no != period_no:
            if last_period_no != -1:
                do_calc_end(
                    last_period_no,
                    no_of_periods,
                    period_weights,
                    sample_size,
                    curr_summary_id,
                    vec_analytical_aal,
                    vecs_sample_aal,
                    vec_sample_sum_loss,
                )
            last_period_no = period_no
        if last_file_idx != file_idx:
            last_file_idx - file_idx
            summary_fin = files_handles[file_idx]
                
        # Read summary header values (event_id, summary_id, expval)
        cursor = file_offset + (2 * oasis_int_size) + oasis_float_size

        vrec = read_losses(summary_fin, cursor, sample_size)
        
        do_calc_by_period(
            vrec,
            vec_sample_sum_loss,
        )
        
        lineno += 1

    curr_summary_id = last_summary_id
    if last_summary_id != -1:
        do_calc_end(
            last_period_no,
            no_of_periods,
            period_weights,
            sample_size,
            curr_summary_id,
            vec_analytical_aal,
            vecs_sample_aal,
            vec_sample_sum_loss,
        )

@profile
def run(run_dir, subfolder, aal=None, alct=None, meanonly=False, noheader=False):
    """Runs AAL calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        subfolder (str): Workspace subfolder inside <run_dir>/work/<subfolder>
        aal (str, optional): Path to AAL output file. Defaults to None
        alct (str, optional): Path to ALCT output file. Defaults to None
        meanonly (bool): Boolean value to output AAL with mean only
        noheader (bool): Boolean value to skip header in output file
    """
    with ExitStack() as stack:
        workspace_folder = Path(run_dir, "work", subfolder)
        max_summary_id = read_max_summary_idx(workspace_folder)
        filelist = read_filelist_idx(workspace_folder)
        
        files_in = [Path(workspace_folder, file) for file in filelist]
        files_handles = [np.memmap(file, mode="r", dtype="u1") for file in files_in]
        streams_in, (stream_source_type, stream_agg_type, sample_size) = init_streams_in(files_in, stack)

        file_data = read_input_files(run_dir)
        vecs_sample_aal = get_sample_sizes(alct, sample_size, max_summary_id)
        vec_sample_sum_loss = np.zeros(sample_size + 1, dtype=np.float64)
        vec_analytical_aal = np.zeros(max_summary_id + 1, dtype=_AAL_REC_DTYPE)

        # TODO: remove these
        print(sample_size)
        print(file_data["occ_map"])
        print(max_summary_id)
        print(filelist)
        print(vecs_sample_aal)
        print(vec_sample_sum_loss)
        print(vec_analytical_aal)

        # TODO: read summaries.idx and update above vecs loop
        summaries_file = Path(workspace_folder, "summaries.idx")
        summaries = np.loadtxt(summaries_file, delimiter=",", dtype=_SUMMARIES_DTYPE)
        
        if len(summaries) == 0:
            raise RuntimeError("Error: summaries file empty")
        run_aal(
            summaries,
            file_data["no_of_periods"],
            file_data["period_weights"],
            sample_size,
            filelist,
            files_handles,
            vec_analytical_aal,
            vecs_sample_aal,
            vec_sample_sum_loss,
        )

        # TODO: remove these
        print("#" * 50)
        print(vecs_sample_aal)
        print(vec_sample_sum_loss)
        print(vec_analytical_aal)

        # TODO: output csvs


@redirect_logging(exec_name='aalpy')
def main(run_dir='.', subfolder=None, aal=None, alct=None, meanonly=False, noheader=False, **kwargs):
    run(
        run_dir,
        subfolder,
        aal=aal,
        meanonly=meanonly,
        alct=alct,
        noheader=noheader,
    )
