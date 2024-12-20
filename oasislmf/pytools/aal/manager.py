# aal/manager.py

import logging
import numpy as np
import numba as nb
import os
from contextlib import ExitStack
from pathlib import Path

from oasislmf.pytools.aal.utils import heap_pop, heap_push, init_heap
from oasislmf.pytools.common.data import (MEAN_TYPE_ANALYTICAL, MEAN_TYPE_SAMPLE, oasis_int, oasis_float, oasis_int_size, oasis_float_size)
from oasislmf.pytools.common.event_stream import (MAX_LOSS_IDX, NUM_SPECIAL_SIDX, NUMBER_OF_AFFECTED_RISK_IDX, SUMMARY_STREAM_ID,
                                                  init_streams_in, mv_read)
from oasislmf.pytools.common.input_files import read_occurrence, read_periods
from oasislmf.pytools.utils import redirect_logging


logger = logging.getLogger(__name__)

# Total amount of memory AAL summary index should use before raising an error (GB)
OASIS_AAL_MEMORY = float(os.environ["OASIS_AAL_MEMORY"]) if "OASIS_AAL_MEMORY" in os.environ else 4

# Similar to aal_rec in ktools
# summary_id can be infered from index
# we have a use_id boolean to store whether this summary_id/index is used
# type can be infered from which array is using it
_AAL_REC_DTYPE = np.dtype([
    ('use_id', np.bool_),
    ('mean', np.float64),
    ('mean_squared', np.float64),
])

# Similar to aal_rec_period
_AAL_REC_PERIOD_DTYPE = np.dtype(
    _AAL_REC_DTYPE.descr + [('mean_period', np.float64)]
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
_SUMMARIES_DTYPE_size = _SUMMARIES_DTYPE.itemsize

AAL_output = [
    ('SummaryId', oasis_int, '%d'),
    ('SampleType', oasis_int, '%d'),
    ('MeanLoss', oasis_float, '%.6f'),
    ('SDLoss', oasis_float, '%.6f'),
]

ALCT_output = [
    ("SummaryId", oasis_float, '%d'),
    ("MeanLoss", oasis_float, '%.6f'),
    ("SDLoss", oasis_float, '%.6f'),
    ("SampleSize", oasis_float, '%d'),
    ("LowerCI", oasis_float, '%.6f'),
    ("UpperCI", oasis_float, '%.6f'),
    ("StandardError", oasis_float, '%.6f'),
    ("RelativeError", oasis_float, '%.6f'),
    ("VarElementHaz", oasis_float, '%.6f'),
    ("StandardErrorHaz", oasis_float, '%.6f'),
    ("RelativeErrorHaz", oasis_float, '%.6f'),
    ("VarElementVuln", oasis_float, '%.6f'),
    ("StandardErrorVuln", oasis_float, '%.6f'),
    ("RelativeErrorVuln", oasis_float, '%.6f'),
]


@nb.njit(cache=True, error_model="numpy")
def process_bin_file(
    fbin,
    offset,
    occ_map,
    summaries_data,
    summaries_idx,
    file_index,
    sample_size
):
    """Reads summary<n>.bin file event_ids and summary_ids to populate summaries_data
    Args:
        fbin (np.memmap): summary binary memmap
        offset (int): file offset to read from
        occ_map (ndarray[occ_map_dtype]): numpy map of event_id, period_no, occ_date_id from the occurrence file
        summaries_data (ndarray[_SUMMARIES_DTYPE]): Index summary data (summaries.idx data)
        summaries_idx (int): current index reached in summaries_data
        file_index (int): Summary bin file index
        sample_size (int): Sample size
    Returns:
        summaries_idx (int): current index reached in summaries_data
        resize_flag (bool): flag to indicate whether to resize summaries_data when full
        offset (int): file offset to read from
    """
    while offset < len(fbin):
        cursor = offset
        event_id, cursor = mv_read(fbin, cursor, oasis_int, oasis_int_size)
        summary_id, cursor = mv_read(fbin, cursor, oasis_int, oasis_int_size)

        for row in occ_map:
            if row["event_id"] == event_id:
                if summaries_idx >= len(summaries_data):
                    # Resize array as full
                    return summaries_idx, True, offset

                summaries_data[summaries_idx]["summary_id"] = summary_id
                summaries_data[summaries_idx]["file_idx"] = file_index
                summaries_data[summaries_idx]["period_no"] = row["period_no"]
                summaries_data[summaries_idx]["file_offset"] = offset
                summaries_idx += 1

        offset = cursor
        # Read Expval
        _, offset = mv_read(fbin, offset, oasis_float, oasis_float_size)
        # Skip over losses
        _, offset = read_losses(fbin, offset, sample_size)

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
def merge_sorted_chunks(output_mmap, buffer_size, memmaps):
    """Merge sorted chunks using a k-way merge algorithm and save to the output file.
    Args:
        output_mmap (np.memmap): Memmap for final sorted output file for indexed summaries
        buffer_size (int): Size of buffer to sort and store before writing to output file
        memmaps (List[np.memmap]): List of temporary file memmaps
    Returns:
        max_summary_id (int): Max summary ID
    """
    max_summary_id = 0
    buffer = np.empty(buffer_size, dtype=_SUMMARIES_DTYPE)
    buffer_index = 0
    output_index = 0

    min_heap = init_heap()
    size = 0
    # Initialize the min_heap with the first row of each memmap
    for i, mmap in enumerate(memmaps):
        if len(mmap) > 0:
            first_row = mmap[0]
            max_summary_id = max(max_summary_id, first_row["summary_id"])
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
        buffer[buffer_index] = smallest_row
        buffer_index += 1

        # If the buffer is full, look for max_summary_id, write to the memmap, and reset
        if buffer_index == buffer_size:
            max_summary_id = max(max_summary_id, np.max(buffer["summary_id"]))
            # Write buffer to the output memmap
            output_mmap[output_index:output_index + buffer_index] = buffer[:buffer_index]
            output_index += buffer_index
            buffer_index = 0

        # Push the next row from the same file into the heap if there are any more rows
        if row_num + 1 < len(memmaps[file_idx]):
            next_row = memmaps[file_idx][row_num + 1]
            min_heap, size = heap_push(min_heap, size, np.array(
                [next_row["summary_id"], next_row["period_no"], next_row["file_idx"], file_idx, row_num + 1],
                dtype=np.int32
            ))

    # Check for max_summary_id in remaining rows and write to memmap
    if buffer_index > 0:
        max_summary_id = max(max_summary_id, np.max(buffer[:buffer_index]["summary_id"]))
        output_mmap[output_index:output_index + buffer_index] = buffer[:buffer_index]

    return max_summary_id


def get_summaries_data(path, files_handles, occ_map, sample_size, aal_max_memory):
    """Gets the indexed summaries data, ordered with k-way merge if not enough memory
    Args:
        path (os.PathLike): Path to the workspace folder containing summary binaries
        files_handles (List[np.memmap]): List of memmaps for summary files data
        occ_map (ndarray[occ_map_dtype]): numpy map of event_id, period_no, occ_date_id from the occurrence file
        sample_size (int): Sample size
        aal_ma_memory (float): OASIS_AAL_MEMORY value (has to be passed in as numba won't update from environment variable)
    Returns:
        summaries_data (np.memmap[_SUMMARIES_DTYPE]): Index summary data (summaries.idx data)
        max_summary_id (int): Max summary ID
    """
    buffer_size = int(((aal_max_memory * (1024**3) // _SUMMARIES_DTYPE_size)))
    temp_files = []
    chunk_index = 0

    summaries_data = np.empty(buffer_size, dtype=_SUMMARIES_DTYPE)
    summaries_idx = 0
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
                sample_size
            )

            # Write new summaries partial file when buffer size or end of summary file reached
            if resize_flag:
                temp_file_path = Path(path, f"indexed_summaries.part{chunk_index}.bdat")
                sort_and_save_chunk(summaries_data[:summaries_idx], temp_file_path)
                temp_files.append(temp_file_path)
                chunk_index += 1
                summaries_idx = 0

            # End of file, move to next file
            if offset >= len(fbin):
                break

    # Write remaining summaries data to temporary file
    temp_file_path = Path(path, f"indexed_summaries.part{chunk_index}.bdat")
    summaries_data = summaries_data[:summaries_idx]
    sort_and_save_chunk(summaries_data, temp_file_path)
    temp_files.append(temp_file_path)

    # Final summaries_index file
    summaries_file = Path(path, "indexed_summaries.bdat")

    if len(temp_files) == 1:
        # If only one file, rename and return data
        os.rename(temp_files[0], summaries_file)
        max_summary_id = np.max(summaries_data["summary_id"])
        return summaries_data, max_summary_id
    else:
        # If multiple files, merge chunks into one file
        memmaps = [np.memmap(temp_file, mode="r", dtype=_SUMMARIES_DTYPE) for temp_file in temp_files]

        # Create a memmap for the output file with enough space for the merged data
        total_size = sum([len(mmap) for mmap in memmaps])
        output_mmap = np.memmap(summaries_file, mode="w+", dtype=_SUMMARIES_DTYPE, shape=(total_size,))
        max_summary_id = merge_sorted_chunks(output_mmap, buffer_size, memmaps)

        # Clean up temporary files
        for temp_file in temp_files:
            os.remove(temp_file)

    summaries_data = np.memmap(summaries_file, mode="r", dtype=_SUMMARIES_DTYPE)
    return summaries_data, max_summary_id


def summary_index(path, occ_map, stack):
    """Index the summary binary outputs
    Args:
        path (os.PathLike): Path to the workspace folder containing summary binaries
        occ_map (ndarray[occ_map_dtype]): numpy map of event_id, period_no, occ_date_id from the occurrence file
        stack (ExitStack): Exit stack
    Returns:
        files_handles (List[np.memmap]): List of memmaps for summary files data
        sample_size (int): Sample size
        max_summary_id (int): Max summary ID
        summaries_data (np.memmap[_SUMMARIES_DTYPE]): Index summary data
    """
    # Find summary binary files
    files = [file for file in path.glob("*.bin")]
    files_handles = [np.memmap(file, mode="r", dtype="u1") for file in files]

    streams_in, (stream_source_type, stream_agg_type, sample_size) = init_streams_in(files, stack)
    if stream_source_type != SUMMARY_STREAM_ID:
        raise RuntimeError(f"Error: Not a summary stream type {stream_source_type}")

    summaries_data, max_summary_id = get_summaries_data(path, files_handles, occ_map, sample_size, OASIS_AAL_MEMORY)
    return files_handles, sample_size, max_summary_id, summaries_data


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


def get_sample_sizes(alct, sample_size, max_summary_id):
    """Generates Sample AAL np map for subset sizes up to sample_size
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
            data = np.zeros(max_summary_id, dtype=_AAL_REC_PERIOD_DTYPE)
            entries.append(data)
            i += 1

    data = np.zeros(max_summary_id, dtype=_AAL_REC_PERIOD_DTYPE)
    entries.append(data)

    vecs_sample_aal = np.concatenate(entries, axis=0)
    return vecs_sample_aal


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
    vec_sample_sum_loss,
):
    """Updates Analytical and Sample AAL vectors from sample sum losses
    Args:
        period_no (int): Period Number
        no_of_periods (int): Number of periods
        period_weights (ndarray[period_weights_dtype]): Period Weights
        sample_size (int): Sample Size
        curr_summary_id (int): Current summary_id
        max_summary_id (int): Max summary_id
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
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
    mean = vec_sample_sum_loss[0]
    vec_analytical_aal[curr_summary_id - 1]["use_id"] = True
    vec_analytical_aal[curr_summary_id - 1]["mean"] += mean * weighting
    vec_analytical_aal[curr_summary_id - 1]["mean_squared"] += mean * mean * weighting

    # Update Sample AAL
    # Get relevant indexes for curr_summary_id
    len_sample_aal = len(vecs_sample_aal)
    num_subsets = len_sample_aal // max_summary_id
    idxs = [i * max_summary_id + (curr_summary_id - 1) for i in range(num_subsets)]

    # Get sample aal idx for sample_size
    last_sample_aal = vecs_sample_aal[idxs[-1]]
    last_sample_aal["use_id"] = True

    total_mean_by_period = 0
    sidx = 1
    aal_idx = 0
    while sidx < sample_size + 1:
        # Iterate through aal_idx except the last one which is subset_size == sample_size
        while aal_idx < num_subsets - 1:
            curr_sample_idx = idxs[aal_idx]
            curr_sample_aal = vecs_sample_aal[curr_sample_idx]
            curr_sample_aal["use_id"] = True

            # Calculate the subset_size and assign to sidx
            subset_size = 2 ** (curr_sample_idx // max_summary_id)
            sidx = subset_size
            end_sidx = subset_size << 1

            # Traverse sidx == subset_size to sidx == subset_size * 2
            weighted_mean, weighted_mean_squared = get_weighted_means(
                vec_sample_sum_loss,
                weighting,
                sidx,
                end_sidx
            )
            sidx = end_sidx
            curr_sample_aal["mean"] += weighted_mean
            curr_sample_aal["mean_squared"] += weighted_mean_squared

            last_sample_aal["mean"] += weighted_mean
            last_sample_aal["mean_squared"] += weighted_mean_squared

            mean_by_period = weighted_mean
            total_mean_by_period += weighted_mean
            sidx = end_sidx
            # Update current Sample AAL mean_period
            curr_sample_aal["mean_period"] += mean_by_period * mean_by_period
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
def do_calc_by_period(
    vrec,
    vec_sample_sum_loss,
):
    """Populate vec_sample_sum_loss
    Args:
        vrec (ndarray[_VREC_DTYPE]): array of sidx and losses
        vec_sample_sum_loss (ndarray[_AAL_REC_DTYPE]): Vector for sample sum losses
    """
    for rec in vrec:
        loss = rec["loss"]
        if loss > 0:
            sidx = rec["sidx"]
            if rec["sidx"] == -1:  # MEAN_SIDX
                sidx = 0
            vec_sample_sum_loss[sidx] += loss


@nb.njit(cache=True, error_model="numpy")
def read_losses(summary_fin, cursor, sample_size):
    """Read losses from summary_fin starting at cursor
    Args:
        summary_fin (np.memmap): summary file memmap
        cursor (int): data offset for reading binary files
        sample_size (int): Sample Size
    Returns:
        vrec (ndarray[_VREC_DTYPE]): array of sidx and losses
    """
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
    return vrec[:idx], cursor


@nb.njit(cache=True, error_model="numpy")
def run_aal(
    summaries,
    no_of_periods,
    period_weights,
    sample_size,
    max_summary_id,
    files_handles,
    vec_analytical_aal,
    vecs_sample_aal,
    vec_sample_sum_loss,
):
    """Run AAL calculation loop to populate vec data
    Args:
        summaries (np.memmap[_SUMMARIES_DTYPE]): summaries.idx data
        no_of_periods (int): Number of periods
        period_weights (ndarray[period_weights_dtype]): Period Weights
        sample_size (int): Sample Size
        max_summary_id (int): Max summary_id
        files_handles (List[np.memmap]): List of memmaps for summary files data
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        vec_sample_sum_loss (ndarray[_AAL_REC_DTYPE]): Vector for sample sum losses
    """
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
                    max_summary_id,
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
                    max_summary_id,
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

        vrec, _ = read_losses(summary_fin, cursor, sample_size)

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
            max_summary_id,
            vec_analytical_aal,
            vecs_sample_aal,
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
    meanonly,
    sample_size,
    no_of_periods
):
    """Generate AAL csv data
    Args:
        vec_analytical_aal (ndarray[_AAL_REC_DTYPE]): Vector for Analytical AAL
        vecs_sample_aal (ndarray[_AAL_REC_PERIODS_DTYPE]): Vector for Sample AAL
        meanonly (bool): Boolean value to output AAL with mean only
        sample_size (int): Sample Size
        no_of_periods (int): Number of periods
    Returns:
        aal_data (List[List]): AAL csv data
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

    if not meanonly:
        for i in range(len(vec_analytical_aal)):
            if not vec_analytical_aal[i]["use_id"]:
                continue
            aal_data.append([i + 1, MEAN_TYPE_ANALYTICAL, mean_analytical[i], std_analytical[i]])
        for i in range(len(vecs_sample_aal)):
            if not vecs_sample_aal[i]["use_id"]:
                continue
            aal_data.append([i + 1, MEAN_TYPE_SAMPLE, mean_sample[i], std_sample[i]])
    else:  # For some reason aalmeanonlycalc orders data differently
        for i in range(len(vec_analytical_aal)):
            if not vec_analytical_aal[i]["use_id"]:
                continue
            aal_data.append([i + 1, MEAN_TYPE_ANALYTICAL, mean_analytical[i]])
            if not vecs_sample_aal[i]["use_id"]:
                continue
            aal_data.append([i + 1, MEAN_TYPE_SAMPLE, mean_sample[i]])

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


def run(run_dir, subfolder, aal_output_file=None, alct_output_file=None, meanonly=False, noheader=False, confidence=0.95):
    """Runs AAL calculations
    Args:
        run_dir (str | os.PathLike): Path to directory containing required files structure
        subfolder (str): Workspace subfolder inside <run_dir>/work/<subfolder>
        aal_output_file (str, optional): Path to AAL output file. Defaults to None
        alct_output_file (str, optional): Path to ALCT output file. Defaults to None
        meanonly (bool): Boolean value to output AAL with mean only
        noheader (bool): Boolean value to skip header in output file
        confidence (float): Confidence level between 0 and 1, default 0.95
    """
    output_aal = aal_output_file is not None
    output_alct = alct_output_file is not None

    with ExitStack() as stack:
        workspace_folder = Path(run_dir, "work", subfolder)
        if not workspace_folder.is_dir():
            raise RuntimeError(f"Error: Unable to open directory {workspace_folder}")

        file_data = read_input_files(run_dir)
        files_handles, sample_size, max_summary_id, summaries_data = summary_index(workspace_folder, file_data["occ_map"], stack)

        vecs_sample_aal = get_sample_sizes(output_alct, sample_size, max_summary_id)
        # Index 0 is mean
        vec_sample_sum_loss = np.zeros(sample_size + 1, dtype=np.float64)
        # Indexed on summary_id - 1
        vec_analytical_aal = np.zeros(max_summary_id, dtype=_AAL_REC_DTYPE)

        # Run AAL calculations, populate above vecs
        run_aal(
            summaries_data,
            file_data["no_of_periods"],
            file_data["period_weights"],
            sample_size,
            max_summary_id,
            files_handles,
            vec_analytical_aal,
            vecs_sample_aal,
            vec_sample_sum_loss,
        )

        # Initialise csv column names for output files
        output_files = {}
        if output_aal:
            aal_file = stack.enter_context(open(aal_output_file, "w"))
            if not noheader:
                if not meanonly:
                    AAL_headers = ",".join([c[0] for c in AAL_output])
                    aal_file.write(AAL_headers + "\n")
                else:
                    AAL_headers = ",".join([c[0] for c in AAL_output if c[0] != "SDLoss"])
                    aal_file.write(AAL_headers + "\n")
            output_files["aal"] = aal_file
        if output_alct:
            alct_file = stack.enter_context(open(alct_output_file, "w"))
            if not noheader:
                if not meanonly:
                    ALCT_headers = ",".join([c[0] for c in ALCT_output])
                    alct_file.write(ALCT_headers + "\n")
            output_files["alct"] = alct_file

        # Output file data
        if not meanonly:
            AAL_fmt = ','.join([c[2] for c in AAL_output])
        else:
            AAL_fmt = ','.join([c[2] for c in AAL_output if c[0] != "SDLoss"])
        ALCT_fmt = ','.join([c[2] for c in ALCT_output])

        if output_aal:
            # Get Sample AAL data for subset_size == sample_size (last group of arrays)
            num_groups = len(vecs_sample_aal) // max_summary_id
            start_idx = (num_groups - 1) * max_summary_id
            end_idx = len(vecs_sample_aal)
            aal_data = get_aal_data(
                vec_analytical_aal,
                vecs_sample_aal[start_idx:end_idx],
                meanonly,
                sample_size,
                file_data["no_of_periods"],
            )
            np.savetxt(output_files["aal"], aal_data, delimiter=",", fmt=AAL_fmt)
        if output_alct:
            alct_data = get_alct_data(
                vecs_sample_aal,
                max_summary_id,
                sample_size,
                file_data["no_of_periods"],
                confidence
            )
            np.savetxt(output_files["alct"], alct_data, delimiter=",", fmt=ALCT_fmt)


@redirect_logging(exec_name='aalpy')
def main(run_dir='.', subfolder=None, aal=None, alct=None, meanonly=False, noheader=False, confidence=0.95, **kwargs):
    run(
        run_dir,
        subfolder,
        aal_output_file=aal,
        meanonly=meanonly,
        alct_output_file=alct,
        noheader=noheader,
        confidence=confidence,
    )
