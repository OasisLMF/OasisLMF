"""
This file is the entry point for the gul command for the package.

"""
import sys
import os
from select import select
import logging
from contextlib import ExitStack
import numpy as np
from numba import njit
from numba.typed import Dict, List

from oasislmf.pytools.getmodel.manager import get_damage_bins, Item
from oasislmf.pytools.getmodel.common import oasis_float

from oasislmf.pytools.gul.common import (
    MEAN_IDX, PIPE_CAPACITY, STD_DEV_IDX, TIV_IDX, CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX, NUM_IDX,
    ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE,
    gulSampleslevelRec_size, gulSampleslevelHeader_size, coverage_type, gul_header,
)
from oasislmf.pytools.gul.io import (
    write_negative_sidx, write_sample_header,
    write_sample_rec, read_getmodel_stream
)
from oasislmf.pytools.gul.random import get_random_generator
from oasislmf.pytools.gul.core import split_tiv, get_gul, setmaxloss, compute_mean_loss
from oasislmf.pytools.gul.utils import append_to_dict_value, binary_search


logger = logging.getLogger(__name__)


def get_coverages(input_path, ignore_file_type=set()):
    """Load the coverages from the coverages file.

    Args:
        input_path (str): the path containing the coverage file.
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        numpy.array[oasis_float]: array with the coverage values for each coverage_id.
    """
    input_files = set(os.listdir(input_path))
    # TODO: store default filenames (e.g., coverages.bin) in a parameters file

    if "coverages.bin" in input_files and "bin" not in ignore_file_type:
        coverages_fname = os.path.join(input_path, 'coverages.bin')
        logger.debug(f"loading {coverages_fname}")
        coverages = np.fromfile(coverages_fname, dtype=oasis_float)

    elif "coverages.csv" in input_files and "csv" not in ignore_file_type:
        coverages_fname = os.path.join(input_path, 'coverages.csv')
        logger.debug(f"loading {coverages_fname}")
        coverages = np.genfromtxt(coverages_fname, dtype=oasis_float, delimiter=",")

    else:
        raise FileNotFoundError(f'coverages file not found at {input_path}')

    return coverages


def gul_get_items(input_path, ignore_file_type=set()):
    """Load the items from the items file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
          vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
          areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))
    if "items.bin" in input_files and "bin" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.bin')
        logger.debug(f"loading {items_fname}")
        items = np.memmap(items_fname, dtype=Item, mode='r')
    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.genfromtxt(items_fname, dtype=Item, delimiter=",")
    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


@njit(cache=True, fastmath=True)
def generate_item_map(items, coverages):
    """Generate item_map; requires items to be sorted.

    Args:
        items (numpy.ndarray[int32, int32, int32]): 1-d structured array storing
          `item_id`, `coverage_id`, `group_id` for all items.
          items need to be sorted by increasing areaperil_id, vulnerability_id
          in order to output the items in correct order.

    Returns:
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
    """
    item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
    Nitems = items.shape[0]

    for j in range(Nitems):
        append_to_dict_value(
            item_map,
            tuple((items[j]['areaperil_id'], items[j]['vulnerability_id'])),
            tuple((items[j]['id'], items[j]['coverage_id'], items[j]['group_id'])),
            ITEM_MAP_VALUE_TYPE
        )
        coverages[items[j]['coverage_id']]['max_items'] += 1
    return item_map


def run(run_dir, ignore_file_type, sample_size, loss_threshold, alloc_rule, debug,
        random_generator, file_in=None, file_out=None, **kwargs):
    """Execute the main gulpy worklow.

    Args:
        run_dir: (str) the directory of where the process is running
        ignore_file_type set(str): file extension to ignore when loading
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        alloc_rule (int): back-allocation rule.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        random_generator (int): random generator function id.
        file_in (str, optional): filename of input stream. Defaults to None.
        file_out (str, optional): filename of output stream. Defaults to None.

    Raises:
        ValueError: if alloc_rule is not 0, 1, or 2.

    Returns:
        int: 0 if no errors occurred.
    """
    logger.info("starting gulpy")

    static_path = os.path.join(run_dir, 'static')
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    static_path = 'static/'
    # TODO: store static_path in a paraparameters file
    damage_bins = get_damage_bins(static_path)

    input_path = 'input/'
    # TODO: store input_path in a paraparameters file

    # read coverages from file
    coverages_tiv = get_coverages(input_path)

    # init the structure for computation
    # coverages are numbered from 1, therefore we skip element 0 in `coverages`
    coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv
    del coverages_tiv

    items = gul_get_items(input_path)

    # in-place sort items in order to store them in item_map in the desired order
    # currently numba only supports a simple call to np.sort() with no `order` keyword,
    # so we do the sort here.
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map = generate_item_map(items, coverages)

    # init array to store the coverages to be computed
    # coverages are numebered from 1, therefore skip element 0.
    compute = np.zeros(coverages.shape[0] + 1, items.dtype['coverage_id'])

    with ExitStack() as stack:
        # set up streams
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        select_stream_list = [stream_out]

        # prepare output buffer, write stream header
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2]:
            raise ValueError(f"Expect alloc_rule to be 0 or 1 or 2, got {alloc_rule}")

        cursor = 0
        cursor_bytes = 0

        # create the array to store the seeds
        seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])

        # create buffer to be reused to store all losses for one coverage
        losses_buffer = np.zeros((sample_size + NUM_IDX + 1, np.max(coverages[1:]['max_items'])), dtype=oasis_float)

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size + 2 * gulSampleslevelHeader_size

        for event_data in read_getmodel_stream(streams_in, item_map, coverages, compute, seeds):

            event_id, compute_i, items_data, recs, rec_idx_ptr, rng_index = event_data

            rndms = generate_rndm(seeds[:rng_index], sample_size)

            last_processed_coverage_ids_idx = 0

            # adjust buff size so that the buffer fits the longest coverage
            buff_size = PIPE_CAPACITY
            max_bytes_per_coverage = np.max(coverages['cur_items']) * max_bytes_per_item
            while buff_size < max_bytes_per_coverage:
                buff_size *= 2

            # define the raw memory view and its int32 view
            mv_write = memoryview(bytearray(buff_size))
            int32_mv_write = np.ndarray(buff_size // 4, buffer=mv_write, dtype='i4')

            while last_processed_coverage_ids_idx < compute_i:
                cursor, cursor_bytes, last_processed_coverage_ids_idx = compute_event_losses(
                    event_id, coverages, compute[:compute_i], items_data,
                    last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr,
                    damage_bins, loss_threshold, losses_buffer, alloc_rule, rndms, debug,
                    max_bytes_per_item, buff_size, int32_mv_write, cursor
                )

                # write the losses to the output stream
                write_start = 0
                while write_start < cursor_bytes:
                    select([], select_stream_list, select_stream_list)
                    write_start += stream_out.write(mv_write[write_start:cursor_bytes])

                cursor = 0

            logger.info(f"event {event_id} DONE")

    return 0


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, coverages, coverage_ids, items_data,
                         last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr, damage_bins,
                         loss_threshold, losses, alloc_rule, rndms, debug, max_bytes_per_item, buff_size,
                         int32_mv, cursor):
    """Compute losses for an event.

    Args:
        event_id (int32): event id.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        coverage_ids (numpy.array[: array of **uniques** coverage ids used in this event.
        items_data (numpy.array[items_data_type]): items-related data.
        last_processed_coverage_ids_idx (int): index of the last coverage_id stored in `coverage_ids` that was fully processed
          and printed to the output stream.
        sample_size (int): number of random samples to draw.
        recs (numpy.array[ProbMean]): all the cdfs used in event_id.
        rec_idx_ptr (numpy.array[int]): array with the indices of `rec` where each cdf record starts.
        damage_bins (List[Union[damagebindictionaryCsv, damagebindictionary]]): loaded data from the damage_bin_dict file.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): array (to be re-used) to store losses for all item_ids.
        alloc_rule (int): back-allocation rule.
        rndms (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        max_bytes_per_item (int): maximum bytes to be written in the output stream for an item.
        buff_size (int): size in bytes of the output buffer.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int, int, int: updated value of cursor, updated value of cursor_bytes, last last_processed_coverage_ids_idx
    """
    for coverage_i in range(last_processed_coverage_ids_idx, coverage_ids.shape[0]):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']  # coverages are indexed from 1
        Nitem_ids = coverage['cur_items']
        exposureValue = tiv / Nitem_ids

        # estimate max number of bytes needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        # number of records of type gulSampleslevelRec_size is sample_size + 5 (negative sidx) + 1 (terminator line)
        cursor_bytes = cursor * int32_mv.itemsize
        est_cursor_bytes = Nitem_ids * max_bytes_per_item

        # return before processing this coverage if the number of free bytes left in the buffer
        # is not sufficient to write out the full coverage
        if cursor_bytes + est_cursor_bytes > buff_size:
            return cursor, cursor_bytes, last_processed_coverage_ids_idx

        items = items_data[coverage['start_items']: coverage['start_items'] + coverage['cur_items']]

        for item_i in range(coverage['cur_items']):
            item = items[item_i]
            damagecdf_i = item['damagecdf_i']
            rng_index = item['rng_index']

            rec = recs[rec_idx_ptr[damagecdf_i]:rec_idx_ptr[damagecdf_i + 1]]
            prob_to = rec['prob_to']
            bin_mean = rec['bin_mean']
            Nbins = len(prob_to)

            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                tiv, prob_to, bin_mean, Nbins, damage_bins[Nbins - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_i] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_i] = chance_of_loss
            losses[TIV_IDX, item_i] = exposureValue
            losses[STD_DEV_IDX, item_i] = std_dev
            losses[MEAN_IDX, item_i] = gul_mean

            if sample_size > 0:
                if debug:
                    for sample_idx in range(1, sample_size + 1):
                        rval = rndms[rng_index][sample_idx - 1]
                        losses[sample_idx, item_i] = rval
                else:
                    for sample_idx in range(1, sample_size + 1):
                        # cap `rval` to the maximum `prob_to` value (which should be 1.)
                        rval = rndms[rng_index][sample_idx - 1]

                        if rval >= prob_to[Nbins - 1]:
                            rval = prob_to[Nbins - 1] - 0.00000003
                            bin_idx = Nbins - 1
                        else:
                            # find the bin in which the random value `rval` falls into
                            # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                            # there's a 1:1 mapping between indices of rec and damage_bins
                            bin_idx = binary_search(rval, prob_to, Nbins)

                        # compute ground-up losses
                        gul = get_gul(
                            damage_bins['bin_from'][bin_idx],
                            damage_bins['bin_to'][bin_idx],
                            bin_mean[bin_idx],
                            prob_to[bin_idx - 1] * (bin_idx > 0),
                            prob_to[bin_idx],
                            rval,
                            tiv
                        )

                        if gul >= loss_threshold:
                            losses[sample_idx, item_i] = gul
                        else:
                            losses[sample_idx, item_i] = 0

        cursor = write_losses(event_id, sample_size, loss_threshold, losses[:, :items.shape[0]], items['item_id'], alloc_rule, tiv,
                              int32_mv, cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    return cursor, cursor * int32_mv.itemsize, last_processed_coverage_ids_idx


@njit(cache=True, fastmath=True)
def write_losses(event_id, sample_size, loss_threshold, losses, item_ids, alloc_rule, tiv,
                 int32_mv, cursor):
    """Write the computed losses.

    Args:
        event_id (int32): event id.
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): losses for all item_ids
        item_ids (numpy.array[ITEM_ID_TYPE]): ids of items whose losses are in `losses`.
        alloc_rule (int): back-allocation rule.
        tiv (oasis_float): total insured value.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int: updated values of cursor
    """
    # note that Nsamples = sample_size + NUM_IDX

    if alloc_rule == 2:
        losses[1:] = setmaxloss(losses[1:])

    # split tiv has to be executed after setmaxloss, if alloc_rule==2.
    if alloc_rule >= 1 and tiv > 0:
        # check whether the sum of losses-per-sample exceeds TIV
        # if so, split TIV in proportion to the losses
        for sample_i in range(1, losses.shape[0]):
            split_tiv(losses[sample_i], tiv)

    # output the losses for all the items
    for item_j in range(item_ids.shape[0]):

        # write header
        cursor = write_sample_header(
            event_id, item_ids[item_j], int32_mv, cursor)

        # write negative sidx
        cursor = write_negative_sidx(
            MAX_LOSS_IDX, losses[MAX_LOSS_IDX, item_j],
            CHANCE_OF_LOSS_IDX, losses[CHANCE_OF_LOSS_IDX, item_j],
            TIV_IDX, losses[TIV_IDX, item_j],
            STD_DEV_IDX, losses[STD_DEV_IDX, item_j],
            MEAN_IDX, losses[MEAN_IDX, item_j],
            int32_mv, cursor
        )

        # write the random samples (only those with losses above the threshold)
        for sample_idx in range(1, sample_size + 1):
            if losses[sample_idx, item_j] >= loss_threshold:
                cursor = write_sample_rec(
                    sample_idx, losses[sample_idx, item_j], int32_mv, cursor)

        # write terminator for the samples for this item
        cursor = write_sample_rec(0, 0., int32_mv, cursor)

    return cursor
