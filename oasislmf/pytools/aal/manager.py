# aal/manager.py

import logging
import numpy as np
import numba as nb
import os
from contextlib import ExitStack
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from oasislmf.pytools.aal.data import AAL_meanonly_dtype, AAL_meanonly_fmt, AAL_meanonly_headers, AAL_dtype, AAL_fmt, AAL_headers, ALCT_dtype, ALCT_fmt, ALCT_headers
from oasislmf.pytools.common.data import (DEFAULT_BUFFER_SIZE, MEAN_TYPE_ANALYTICAL, MEAN_TYPE_SAMPLE, oasis_int, oasis_float,
                                          oasis_int_size, oasis_float_size, write_ndarray_to_fmt_csv)
from oasislmf.pytools.common.event_stream import (MEAN_IDX, MAX_LOSS_IDX, NUMBER_OF_AFFECTED_RISK_IDX, SUMMARY_STREAM_ID,
                                                  init_streams_in, mv_read)
from oasislmf.pytools.common.input_files import read_occurrence, read_periods
from oasislmf.pytools.common.utils.nb_heapq import heap_pop, heap_push, init_heap
from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)

# Total amount of memory AAL summary index should use before raising an error (GB)
OASIS_AAL_MEMORY = float(os.environ["OASIS_AAL_MEMORY"]) if "OASIS_AAL_MEMORY" in os.environ else 4

# Similar to aal_rec in ktools
# summary_id can be infered from index
# type can be infered from which array is using it
_AAL_REC_DTYPE = np.dtype([
    ('mean', np.float64),
    ('mean_squared', np.float64),
])

# Similar to aal_rec_period
_AAL_REC_PERIOD_DTYPE = np.dtype(
    _AAL_REC_DTYPE.descr + [('mean_period', np.float64)]
)

_SUMMARIES_DTYPE = np.dtype([
    ("summary_id", np.int32),
    ("file_idx", np.int32),
    ("period_no", np.int32),
    ("file_offset", np.int64),
])
_SUMMARIES_DTYPE_size = _SUMMARIES_DTYPE.itemsize


@nb.njit(cache=True, error_model="numpy")
def process_bin_file(
    fbin,
    offset,
    occ_map,
    summaries_data,
    summaries_idx,
    file_index,
):
    """Reads summary<n>.bin file event_ids and summary_ids to populate summaries_data
    Args:
        fbin (np.memmap): summary binary memmap
        offset (int): file offset to read from
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
        summaries_data (ndarray[_SUMMARIES_DTYPE]): Index summary data (summaries.idx data)
        summaries_idx (int): current index reached in summaries_data
        file_index (int): Summary bin file index
    Returns:
        summaries_idx (int): current index reached in summaries_data
        resize_flag (bool): flag to indicate whether to resize summaries_data when full
        offset (int): file offset to read from
    """
    while offset < len(fbin):
        cursor = offset
        event_id, cursor = mv_read(fbin, cursor, oasis_int, oasis_int_size)
        summary_id, cursor = mv_read(fbin, cursor, oasis_int, oasis_int_size)

        if event_id not in occ_map:
            # If the event_id doesn't exist in occ_map, continue with the next
            offset = cursor
            # Skip over Expval and losses
            _, offset = mv_read(fbin, offset, oasis_float, oasis_float_size)
            offset = skip_losses(fbin, offset)
            continue

        # Get the number of rows for the current event_id
        n_rows = occ_map[event_id].shape[0]

        if summaries_idx + n_rows >= len(summaries_data):
            # Resize array if full
            return summaries_idx, True, offset

        # Now fill the summaries_data with the rows that match the current event_id
        current_row = 0
        for row in occ_map[event_id]:
            summaries_data[summaries_idx]["summary_id"] = summary_id
            summaries_data[summaries_idx]["file_idx"] = file_index
            summaries_data[summaries_idx]["period_no"] = row["period_no"]
            summaries_data[summaries_idx]["file_offset"] = offset
            summaries_idx += 1
            current_row += 1
            if current_row >= n_rows:
                break

        offset = cursor
        # Read Expval
        _, offset = mv_read(fbin, offset, oasis_float, oasis_float_size)
        # Skip over losses
        offset = skip_losses(fbin, offset)

    return summaries_idx, False, offset


def sort_and_save_chunk(summaries_data, temp_file_path):
    """Sort a chunk of summaries data and save it to a temporary file.
    Args:
        summaries_data (ndarray[_SUMMARIES_DTYPE]): Indexed summary data
        temp_file_path (str | os.PathLike): Path to temporary file
    """
    sort_columns = ["file_idx", "period_no", "summary_id"]
    sorted_indices = np.lexsort([summaries_data[col] for col in sort_columns])
    sorted_chunk = summaries_data[sorted_indices]
    sorted_chunk.tofile(temp_file_path)


@nb.njit(cache=True, error_model="numpy")
def merge_sorted_chunks(memmaps):
    """
    Merge sorted chunks using a k-way merge algorithm and yield next smallest row
    Args:
        memmaps (List[np.memmap]): List of temporary file memmaps
    Yields:
        smallest_row (ndarray[_SUMMARIES_DTYPE]): yields the next smallest row from sorted summaries partial files
    """
    min_heap = init_heap(num_compare=3)
    size = 0
    # Initialize the min_heap with the first row of each memmap
    for i, mmap in enumerate(memmaps):
        if len(mmap) > 0:
            first_row = mmap[0]
            min_heap, size = heap_push(min_heap, size, np.array(
                [first_row["summary_id"], first_row["period_no"], first_row["file_idx"], i, 0],
                dtype=np.int32
            ))

    # Perform the k-way merge
    while size > 0:
        # The min heap will store the smallest row at the top when popped
        element, min_heap, size = heap_pop(min_heap, size)
        file_idx = element[3]
        row_num = element[4]
        smallest_row = memmaps[file_idx][row_num]
        yield smallest_row

        # Push the next row from the same file into the heap if there are any more rows
        if row_num + 1 < len(memmaps[file_idx]):
            next_row = memmaps[file_idx][row_num + 1]
            min_heap, size = heap_push(min_heap, size, np.array(
                [next_row["summary_id"], next_row["period_no"], next_row["file_idx"], file_idx, row_num + 1],
                dtype=np.int32
            ))


def get_summaries_data(
    path,
    files_handles,
    occ_map,
    aal_max_memory
):
    """Gets the indexed summaries data, ordered with k-way merge if not enough memory
    Args:
        path (os.PathLike): Path to the workspace folder containing summary binaries
        files_handles (List[np.memmap]): List of memmaps for summary files data
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
        aal_max_memory (float): OASIS_AAL_MEMORY value (has to be passed in as numba won't update from environment variable)
    Returns:
        memmaps (List[np.memmap]): List of temporary file memmaps
        max_summary_id (int): Max summary ID
    """
    # Remove existing temp bdat files if exists
    for temp_file in path.glob("indexed_summaries.part*.bdat"):
        os.remove(temp_file)

    buffer_size = int(((aal_max_memory * (1024**3) // _SUMMARIES_DTYPE_size)))
    temp_files = []
    chunk_index = 0

    summaries_data = np.empty(buffer_size, dtype=_SUMMARIES_DTYPE)
    summaries_idx = 0
    max_summary_id = 0
    for file_index, fbin in enumerate(files_handles):
        offset = oasis_int_size * 3  # Summary stream header size

        while True:
            summaries_idx, resize_flag, offset = process_bin_file(
                fbin,
                offset,
                occ_map,
                summaries_data,
                summaries_idx,
                file_index,
            )

            # Write new summaries partial file when buffer size or end of summary file reached
            if resize_flag:
                temp_file_path = Path(path, f"indexed_summaries.part{chunk_index}.bdat")
                summaries_data = summaries_data[:summaries_idx]
                sort_and_save_chunk(summaries_data, temp_file_path)
                temp_files.append(temp_file_path)
                chunk_index += 1
                summaries_idx = 0
                max_summary_id = max(max_summary_id, np.max(summaries_data["summary_id"]))

            # End of file, move to next file
            if offset >= len(fbin):
                break

    # Write remaining summaries data to temporary file
    temp_file_path = Path(path, f"indexed_summaries.part{chunk_index}.bdat")
    summaries_data = summaries_data[:summaries_idx]
    sort_and_save_chunk(summaries_data, temp_file_path)
    max_summary_id = max(max_summary_id, np.max(summaries_data["summary_id"]))
    temp_files.append(temp_file_path)

    memmaps = [np.memmap(temp_file, mode="r", dtype=_SUMMARIES_DTYPE) for temp_file in temp_files]

    return memmaps, max_summary_id


def summary_index(path, occ_map, stack):
    """Index the summary binary outputs
    Args:
        path (os.PathLike): Path to the workspace folder containing summary binaries
        occ_map (nb.typed.Dict): numpy map of event_id, period_no, occ_date_id from the occurrence file
        stack (ExitStack): Exit stack
    Returns:
        files_handles (List[np.memmap]): List of memmaps for summary files data
        sample_size (int): Sample size
        max_summary_id (int): Max summary ID
        memmaps (List[np.memmap]): List of temporary file memmaps
    """
    # work folder for aal files
    aal_files_folder = Path(path, "aal_files")
    aal_files_folder.mkdir(parents=False, exist_ok=True)

    # Find summary binary files
    files = [file for file in path.glob("*.bin")]
    files_handles = [np.memmap(file, mode="r", dtype="u1") for file in files]

    streams_in, (stream_source_type, stream_agg_type, sample_size) = init_streams_in(files, stack)
    if stream_source_type != SUMMARY_STREAM_ID:
        raise RuntimeError(f"Error: Not a summary stream type {stream_source_type}")

    memmaps, max_summary_id = get_summaries_data(
        aal_files_folder,
        files_handles,
        occ_map,
        OASIS_AAL_MEMORY
    )
    return files_handles, sample_size, max_summary_id, memmaps


def read_input_files(run_dir):
    """Reads all input files and returns a dict of relevant data
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
    Returns:
        file_data (Dict[str, Any]): A dict of relevent data extracted from files
    """
    occ_map, date_algorithm, granular_date, no_of_periods = read_occurrence(Path(run_dir, "input"))
    period_weights = read_periods(no_of_periods, Path(run_dir, "input"))

    file_data = {
        "occ_map": occ_map,
        "date_algorithm": date_algorithm,
        "granular_date": granular_date,
        "no_of_periods": no_of_periods,
        "period_weights": period_weights,
    }
    return file_data


def get_num_subsets(alct, sample_size, max_summary_id):
    """Gets the number of subsets required to generates the Sample AAL np map for subset sizes up to sample_size
    Example: sample_size[10], max_summary_id[2] generates following ndarray
    [
        #   subset_size, mean,  mean_squared, mean_period
        [0, 0, 0],  # subset_size = 1 , summary_id = 1
        [0, 0, 0],  # subset_size = 1 , summary_id = 2
        [0, 0, 0],  # subset_size = 2 , summary_id = 1
        [0, 0, 0],  # subset_size = 2 , summary_id = 2
        [0, 0, 0],  # subset_size = 4 , summary_id = 1
        [0, 0, 0],  # subset_size = 4 , summary_id = 2
        [0, 0, 0],  # subset_size = 10 , summary_id = 1, subset_size = sample_size
        [0, 0, 0],  # subset_size = 10 , summary_id = 2, subset_size = sample_size
    ]
    Subset_size is implicit based on position in array, grouped by max_summary_id
    So first two arrays are subset_size 2^0 = 1
    The next two arrays are subset_size 2^1 = 2
    The next two arrays are subset_size 2^2 = 4
    The last two arrays are subset_size = sample_size = 10
    Doesn't generate one with subset_size 8 as double that is larger than sample_size
    Therefore this function returns 4, and the sample aal array is 4 * 2
    Args:
        alct (bool): Boolean for ALCT output
        sample_size (int): Sample size
        max_summary_id (int): Max summary ID
    Returns:
        num_subsets (int): Number of subsets
    """
    i = 0
    if alct and sample_size > 1:
        while ((1 << i) + ((1 << i) - 1)) <= sample_size:
            i += 1
    return i + 1


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def get_weighted_means(
    vec_sample_sum_loss,
    weighting,
    sidx,
    end_sidx,
):
    """Get sum of weighted mean and weighted mean_squared
    Args:
        vec_sample_sum_loss (ndarray[_AAL_REC_DTYPE]): Vector for sample sum losses
        weighting (float): Weighting value
        sidx (int): start index
        end_sidx (int): end index
    Returns:
        weighted_mean (float): Sum weighted mean
        weighted_mean_squared (float): Sum weighted mean squared
    """
    weighted_mean = 0
    weighted_mean_squared = 0
    while sidx < end_sidx:
        sumloss = vec_sample_sum_loss[sidx]
        weighted_mean += sumloss * weighting
        weighted_mean_squared += sumloss * sumloss * weighting
        sidx += 1
    return weighted_mean, weighted_mean_squared


@nb.njit(cache=True, error_model="numpy")
def do_calc_end(
    period_no,
    no_of_periods,
    period_weights,
    sample_size,
    curr_summary_id,
    max_summary_id,
    vec_analytical_aal,
    vecs_sample_aal,
    vec_used_summary_id,
    vec_sample_sum_loss,
):
    """Updates Analytical and Sample AAL vectors from sample sum losses
    Args:
        period_no (int): Period Number
        no_of_periods (int): Number of periods
        period_weights (ndarray[periods_dtype]): Period Weights
        sample_size (int): Sample Size
        curr_summary_id (int): Current summary_id
        max_summary_id (int): Max summary_id
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        vec_used_summary_id (ndarray[bool]): vector to store if summary_id is used
        vec_sample_sum_loss (ndarray[_AAL_REC_DTYPE]): Vector for sample sum losses
    """
    # Get weighting
    weighting = 1
    if no_of_periods > 0:
        # period_no in period_weights
        if period_no > 0 and period_no <= no_of_periods:
            weighting = period_weights[period_no - 1][1] * no_of_periods
        else:
            weighting = 0

    # Update Analytical AAL
    mean = vec_sample_sum_loss[0]  # 0 index is where the analytical mean is stored
    vec_used_summary_id[curr_summary_id - 1] = True

    vec_analytical_aal[curr_summary_id - 1]["mean"] += mean * weighting
    vec_analytical_aal[curr_summary_id - 1]["mean_squared"] += mean * mean * weighting

    # Update Sample AAL
    # Get relevant indexes for curr_summary_id
    len_sample_aal = len(vecs_sample_aal)
    num_subsets = len_sample_aal // max_summary_id
    idxs = [i * max_summary_id + (curr_summary_id - 1) for i in range(num_subsets)]

    # Get sample aal idx for sample_size
    last_sample_aal = vecs_sample_aal[idxs[-1]]

    total_mean_by_period = 0
    sidx = 1
    aal_idx = 0
    while sidx < sample_size + 1:
        # Iterate through aal_idx except the last one which is subset_size == sample_size
        while aal_idx < num_subsets - 1:
            curr_sample_aal = vecs_sample_aal[idxs[aal_idx]]

            # Calculate the subset_size and assign to sidx
            sidx = 1 << aal_idx
            end_sidx = sidx << 1

            # Traverse sidx == subset_size to sidx == subset_size * 2
            weighted_mean, weighted_mean_squared = get_weighted_means(
                vec_sample_sum_loss,
                weighting,
                sidx,
                end_sidx
            )

            # Update sample size Sample AAL
            last_sample_aal["mean"] += weighted_mean
            last_sample_aal["mean_squared"] += weighted_mean_squared
            total_mean_by_period += weighted_mean

            # Update current Sample AAL
            curr_sample_aal["mean"] += weighted_mean
            curr_sample_aal["mean_squared"] += weighted_mean_squared
            # Update current Sample AAL mean_period
            curr_sample_aal["mean_period"] += weighted_mean * weighted_mean

            sidx = end_sidx
            aal_idx += 1
        # Update sample size Sample AAL
        mean = vec_sample_sum_loss[sidx]
        total_mean_by_period += mean * weighting
        last_sample_aal["mean"] += mean * weighting
        last_sample_aal["mean_squared"] += mean * mean * weighting
        sidx += 1
    # Update sample size Sample AAL mean_period
    last_sample_aal["mean_period"] += total_mean_by_period * total_mean_by_period
    vec_sample_sum_loss.fill(0)


@nb.njit(cache=True, error_model="numpy")
def read_losses(summary_fin, cursor, vec_sample_sum_loss):
    """Read losses from summary_fin starting at cursor, populate vec_sample_sum_loss
    Args:
        summary_fin (np.memmap): summary file memmap
        cursor (int): data offset for reading binary files
        (ndarray[_AAL_REC_DTYPE]): Vector for sample sum losses
    Returns:
        cursor (int): data offset for reading binary files
    """
    # Max losses is sample_size + num special sidxs
    valid_buff = len(summary_fin)
    while True:
        if valid_buff - cursor < oasis_int_size + oasis_float_size:
            raise RuntimeError("Error: broken summary file, not enough data")
        sidx, cursor = mv_read(summary_fin, cursor, oasis_int, oasis_int_size)
        loss, cursor = mv_read(summary_fin, cursor, oasis_float, oasis_float_size)

        if sidx == 0:
            break
        if sidx == NUMBER_OF_AFFECTED_RISK_IDX or sidx == MAX_LOSS_IDX:
            continue
        if sidx == MEAN_IDX:
            sidx = 0
        vec_sample_sum_loss[sidx] += loss
    return cursor


@nb.njit(cache=True, error_model="numpy")
def skip_losses(summary_fin, cursor):
    """Skip through losses in summary_fin starting at cursor
    Args:
        summary_fin (np.memmap): summary file memmap
        cursor (int): data offset for reading binary files
    Returns:
        cursor (int): data offset for reading binary files
    """
    valid_buff = len(summary_fin)
    sidx = 1
    while sidx:
        if valid_buff - cursor < oasis_int_size + oasis_float_size:
            raise RuntimeError("Error: broken summary file, not enough data")
        sidx, cursor = mv_read(summary_fin, cursor, oasis_int, oasis_int_size)
        cursor += oasis_float_size
    return cursor


@nb.njit(cache=True, error_model="numpy")
def run_aal(
    memmaps,
    no_of_periods,
    period_weights,
    sample_size,
    max_summary_id,
    files_handles,
    vec_analytical_aal,
    vecs_sample_aal,
    vec_used_summary_id,
):
    """Run AAL calculation loop to populate vec data
    Args:
        memmaps (List[np.memmap]): List of temporary file memmaps
        no_of_periods (int): Number of periods
        period_weights (ndarray[periods_dtype]): Period Weights
        sample_size (int): Sample Size
        max_summary_id (int): Max summary_id
        files_handles (List[np.memmap]): List of memmaps for summary files data
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        vec_used_summary_id (ndarray[bool]): vector to store if summary_id is used
    """
    if len(memmaps) == 0:
        raise ValueError("File is empty or missing data")

    # Index 0 is mean
    vec_sample_sum_loss = np.zeros(sample_size + 1, dtype=np.float64)
    last_summary_id = -1
    last_period_no = -1

    for line in merge_sorted_chunks(memmaps):
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
                    last_summary_id,
                    max_summary_id,
                    vec_analytical_aal,
                    vecs_sample_aal,
                    vec_used_summary_id,
                    vec_sample_sum_loss,
                )
            last_period_no = period_no
            last_summary_id = summary_id
        elif last_period_no != period_no:
            if last_period_no != -1:
                do_calc_end(
                    last_period_no,
                    no_of_periods,
                    period_weights,
                    sample_size,
                    last_summary_id,
                    max_summary_id,
                    vec_analytical_aal,
                    vecs_sample_aal,
                    vec_used_summary_id,
                    vec_sample_sum_loss,
                )
            last_period_no = period_no
        summary_fin = files_handles[file_idx]

        # Read summary header values (event_id, summary_id, expval)
        cursor = file_offset + (2 * oasis_int_size) + oasis_float_size

        read_losses(summary_fin, cursor, vec_sample_sum_loss)

    if last_summary_id != -1:
        do_calc_end(
            last_period_no,
            no_of_periods,
            period_weights,
            sample_size,
            last_summary_id,
            max_summary_id,
            vec_analytical_aal,
            vecs_sample_aal,
            vec_used_summary_id,
            vec_sample_sum_loss,
        )


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def calculate_mean_stddev(
    observable_sum,
    observable_squared_sum,
    number_of_observations
):
    """Compute the mean and standard deviation from the sum and squared sum of an observable
    Args:
        observable_sum (ndarray[oasis_float]): Observable sum
        observable_squared_sum (ndarray[oasis_float]): Observable squared sum
        number_of_observations (int | ndarray[int]): number of observations
    Returns:
        mean (ndarray[oasis_float]): Mean
        std (ndarray[oasis_float]): Standard Deviation
    """
    mean = observable_sum / number_of_observations
    std = np.sqrt(
        (
            observable_squared_sum - (observable_sum * observable_sum)
            / number_of_observations
        ) / (number_of_observations - 1)
    )
    return mean, std


@nb.njit(cache=True, error_model="numpy")
def get_aal_data(
    vec_analytical_aal,
    vecs_sample_aal,
    vec_used_summary_id,
    sample_size,
    no_of_periods
):
    """Generate AAL csv data
    Args:
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        vec_used_summary_id (ndarray[bool]): vector to store if summary_id is used
        sample_size (int): Sample Size
        no_of_periods (int): Number of periods
    Returns:
        aal_data (List[Tuple]): AAL csv data
    """
    aal_data = []
    assert len(vec_analytical_aal) == len(vecs_sample_aal), \
        f"Lengths of analytical ({len(vec_analytical_aal)}) and sample ({len(vecs_sample_aal)}) aal data differ"
    mean_analytical, std_analytical = calculate_mean_stddev(
        vec_analytical_aal["mean"],
        vec_analytical_aal["mean_squared"],
        no_of_periods,
    )

    mean_sample, std_sample = calculate_mean_stddev(
        vecs_sample_aal["mean"],
        vecs_sample_aal["mean_squared"],
        no_of_periods * sample_size,
    )

    for summary_idx in range(len(vec_analytical_aal)):
        if not vec_used_summary_id[summary_idx]:
            continue
        aal_data.append((summary_idx + 1, MEAN_TYPE_ANALYTICAL, mean_analytical[summary_idx], std_analytical[summary_idx]))
    for summary_idx in range(len(vecs_sample_aal)):
        if not vec_used_summary_id[summary_idx]:
            continue
        aal_data.append((summary_idx + 1, MEAN_TYPE_SAMPLE, mean_sample[summary_idx], std_sample[summary_idx]))

    return aal_data


@nb.njit(cache=True, error_model="numpy")
def get_aal_data_meanonly(
    vec_analytical_aal,
    vecs_sample_aal,
    vec_used_summary_id,
    sample_size,
    no_of_periods
):
    """Generate AAL csv data
    Args:
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        vec_used_summary_id (ndarray[bool]): vector to store if summary_id is used
        sample_size (int): Sample Size
        no_of_periods (int): Number of periods
    Returns:
        aal_data (List[Tuple]): AAL csv data
    """
    aal_data = []
    assert len(vec_analytical_aal) == len(vecs_sample_aal), \
        f"Lengths of analytical ({len(vec_analytical_aal)}) and sample ({len(vecs_sample_aal)}) aal data differ"
    mean_analytical, std_analytical = calculate_mean_stddev(
        vec_analytical_aal["mean"],
        vec_analytical_aal["mean_squared"],
        no_of_periods,
    )

    mean_sample, std_sample = calculate_mean_stddev(
        vecs_sample_aal["mean"],
        vecs_sample_aal["mean_squared"],
        no_of_periods * sample_size,
    )

    # aalmeanonlycalc orders output data differently, this if condition is here to match the output to the ktools output
    for summary_idx in range(len(vec_analytical_aal)):
        if not vec_used_summary_id[summary_idx]:
            continue
        aal_data.append((summary_idx + 1, MEAN_TYPE_ANALYTICAL, mean_analytical[summary_idx]))
        aal_data.append((summary_idx + 1, MEAN_TYPE_SAMPLE, mean_sample[summary_idx]))

    return aal_data


@nb.njit(cache=True, fastmath=True, error_model="numpy")
def calculate_confidence_interval(std_err, confidence_level):
    """Calculate the confidence interval based on standard error and confidence level.
    Args:
        std_err (float): The standard error.
        confidence_level (float): The confidence level (e.g., 0.95 for 95%).
    Returns:
        confidence interval (float): The confidence interval.
    """
    # Compute p-value above 0.5
    p_value = (1 + confidence_level) / 2
    p_value = np.sqrt(-2 * np.log(1 - p_value))

    # Approximation formula for z-value from Abramowitz & Stegun, Handbook
    # of Mathematical Functions: with Formulas, Graphs, and Mathematical
    # Tables, Dover Publications (1965), eq. 26.2.23
    # Also see John D. Cook Consulting, https://www.johndcook.com/blog/cpp_phi_inverse/
    c = np.array([2.515517, 0.802853, 0.010328])
    d = np.array([1.432788, 0.189269, 0.001308])
    z_value = p_value - (
        ((c[2] * p_value + c[1]) * p_value + c[0]) /
        (((d[2] * p_value + d[1]) * p_value + d[0]) * p_value + 1)
    )
    # Return the confidence interval
    return std_err * z_value


@nb.njit(cache=True, error_model="numpy")
def get_alct_data(
    vecs_sample_aal,
    max_summary_id,
    sample_size,
    no_of_periods,
    confidence,
):
    """Generate ALCT csv data
    Args:
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        max_summary_id (int): Max summary_id
        sample_size (int): Sample Size
        no_of_periods (int): Number of periods
        confidence (float): Confidence level between 0 and 1, default 0.95
    Returns:
        alct_data (List[List]): ALCT csv data
    """
    alct_data = []

    num_subsets = len(vecs_sample_aal) // max_summary_id
    # Generate the subset sizes (last one is always sample_size)
    subset_sizes = np.array([2 ** i for i in range(num_subsets)])
    subset_sizes[-1] = sample_size

    for summary_id in range(1, max_summary_id + 1):
        # Get idxs for summary_id across all subset_sizes
        idxs = np.array([i * max_summary_id + (summary_id - 1) for i in range(num_subsets)])
        v_curr = vecs_sample_aal[idxs]

        mean, std = calculate_mean_stddev(
            v_curr["mean"],
            v_curr["mean_squared"],
            subset_sizes * no_of_periods,
        )
        mean_period = v_curr["mean_period"] / (subset_sizes * subset_sizes)

        var_vuln = (
            (v_curr["mean_squared"] - subset_sizes * mean_period)
            / (subset_sizes * no_of_periods - subset_sizes)
        ) / (subset_sizes * no_of_periods)
        var_haz = (
            subset_sizes * (mean_period - no_of_periods * mean * mean)
            / (no_of_periods - 1)
        ) / (subset_sizes * no_of_periods)

        std_err = np.sqrt(var_vuln)
        ci = calculate_confidence_interval(std_err, confidence)

        std_err_haz = np.sqrt(var_haz)
        std_err_vuln = np.sqrt(var_vuln)

        lower_ci = np.where(ci > 0, mean - ci, 0)
        upper_ci = np.where(ci > 0, mean + ci, 0)

        curr_data = np.column_stack((
            np.array([summary_id] * num_subsets),
            mean,
            std,
            subset_sizes,
            lower_ci,
            upper_ci,
            std_err, std_err / mean,
            var_haz, std_err_haz, std_err_haz / mean,
            var_vuln, std_err_vuln, std_err_vuln / mean,
        ))
        for row in curr_data:
            alct_data.append(row)
    return alct_data


def run(
    run_dir,
    subfolder,
    aal_output_file=None,
    alct_output_file=None,
    meanonly=False,
    noheader=False,
    confidence=0.95,
    output_format="csv",
):
    """Runs AAL calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        subfolder (str): Workspace subfolder inside <run_dir>/work/<subfolder>
        aal_output_file (str, optional): Path to AAL output file. Defaults to None
        alct_output_file (str, optional): Path to ALCT output file. Defaults to None
        meanonly (bool): Boolean value to output AAL with mean only
        noheader (bool): Boolean value to skip header in output file
        confidence (float): Confidence level between 0 and 1, default 0.95
        output_format (str): Output format extension. Defaults to "csv".
    """
    outmap = {
        "aal": {
            "compute": aal_output_file is not None,
            "file_path": aal_output_file,
            "fmt": AAL_fmt if not meanonly else AAL_meanonly_fmt,
            "headers": AAL_headers if not meanonly else AAL_meanonly_headers,
            "file": None,
            "dtype": AAL_dtype if not meanonly else AAL_meanonly_dtype,
        },
        "alct": {
            "compute": alct_output_file is not None,
            "file_path": alct_output_file,
            "fmt": ALCT_fmt,
            "headers": ALCT_headers,
            "file": None,
            "dtype": ALCT_dtype,
        },
    }

    output_format = "." + output_format
    output_binary = output_format == ".bin"
    output_parquet = output_format == ".parquet"
    # Check for correct suffix
    for path in [v["file_path"] for v in outmap.values()]:
        if path is None:
            continue
        if Path(path).suffix == "":  # Ignore suffix for pipes
            continue
        if (Path(path).suffix != output_format):
            raise ValueError(f"Invalid file extension for {output_format}, got {path},")

    if not all([v["compute"] for v in outmap.values()]):
        logger.warning("No output files specified")

    with ExitStack() as stack:
        workspace_folder = Path(run_dir, "work", subfolder)
        if not workspace_folder.is_dir():
            raise RuntimeError(f"Error: Unable to open directory {workspace_folder}")

        file_data = read_input_files(run_dir)
        files_handles, sample_size, max_summary_id, memmaps = summary_index(
            workspace_folder,
            file_data["occ_map"],
            stack
        )
        # aal vec are Indexed on summary_id - 1
        num_subsets = get_num_subsets(outmap["alct"]["compute"], sample_size, max_summary_id)
        vecs_sample_aal = np.zeros(num_subsets * max_summary_id, dtype=_AAL_REC_PERIOD_DTYPE)
        vec_analytical_aal = np.zeros(max_summary_id, dtype=_AAL_REC_DTYPE)
        vec_used_summary_id = np.zeros(max_summary_id, dtype=np.bool_)

        # Run AAL calculations, populate above vecs
        run_aal(
            memmaps,
            file_data["no_of_periods"],
            file_data["period_weights"],
            sample_size,
            max_summary_id,
            files_handles,
            vec_analytical_aal,  # unique in output_aal
            vecs_sample_aal,
            vec_used_summary_id,
        )

        # Initialise output files AAL
        if output_binary:
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue
                out_file = stack.enter_context(open(outmap[out_type]["file_path"], 'wb'))
                outmap[out_type]["file"] = out_file
        elif output_parquet:
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue
                temp_out_data = np.zeros(DEFAULT_BUFFER_SIZE, dtype=outmap[out_type]["dtype"])
                temp_df = pd.DataFrame(temp_out_data, columns=outmap[out_type]["headers"])
                temp_table = pa.Table.from_pandas(temp_df)
                out_file = pq.ParquetWriter(outmap[out_type]["file_path"], temp_table.schema)
                outmap[out_type]["file"] = out_file
        else:
            for out_type in outmap:
                if not outmap[out_type]["compute"]:
                    continue
                out_file = stack.enter_context(open(outmap[out_type]["file_path"], 'w'))
                if not noheader:
                    csv_headers = ','.join(outmap[out_type]["headers"])
                    out_file.write(csv_headers + '\n')
                outmap[out_type]["file"] = out_file

        def write_output(data, out_type):
            if output_binary:
                data.tofile(outmap[out_type]["file"])
            elif output_parquet:
                data_df = pd.DataFrame(data)
                data_table = pa.Table.from_pandas(data_df)
                outmap[out_type]["file"].write_table(data_table)
            else:
                write_ndarray_to_fmt_csv(
                    outmap[out_type]["file"],
                    data,
                    outmap[out_type]["headers"],
                    outmap[out_type]["fmt"]
                )

        if outmap["aal"]["compute"]:
            # Get Sample AAL data for subset_size == sample_size (last group of arrays)
            start_idx = (num_subsets - 1) * max_summary_id
            aal_data_func = get_aal_data_meanonly if meanonly else get_aal_data
            aal_data = aal_data_func(
                vec_analytical_aal,
                vecs_sample_aal[start_idx:],
                vec_used_summary_id,
                sample_size,
                file_data["no_of_periods"],
            )
            aal_data = np.array(aal_data, dtype=outmap["aal"]["dtype"])

            write_output(aal_data, "aal")
        if outmap["alct"]["compute"]:
            alct_data = get_alct_data(
                vecs_sample_aal,
                max_summary_id,
                sample_size,
                file_data["no_of_periods"],
                confidence
            )
            alct_data = np.array([tuple(arr) for arr in alct_data], dtype=outmap["alct"]["dtype"])

            write_output(alct_data, "alct")


@redirect_logging(exec_name='aalpy')
def main(run_dir='.', subfolder=None, aal=None, alct=None, meanonly=False, noheader=False, confidence=0.95, ext="csv", **kwargs):
    run(
        run_dir,
        subfolder,
        aal_output_file=aal,
        meanonly=meanonly,
        alct_output_file=alct,
        noheader=noheader,
        confidence=confidence,
        output_format=ext,
    )
