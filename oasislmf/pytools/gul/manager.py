"""
This file is the entry point for the gul command for the package.

"""
import sys
import os
import time
import logging
from contextlib import ExitStack
import numpy as np
import numba as nb
from numba import njit
from numba.types import int_
from numba.typed import Dict, List

from oasislmf.pytools.getmodel.manager import get_damage_bins, Item
from oasislmf.pytools.getmodel.common import oasis_float

from oasislmf.pytools.gul.common import (
    MEAN_IDX, STD_DEV_IDX, TIV_IDX, CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX, NUM_IDX,
    ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE, ITEM_ID_TYPE, ITEMS_DATA_MAP_TYPE,
    COVERAGE_ID_TYPE, gulSampleslevelRec_size, gulSampleslevelHeader_size, coverage_type, items_data_type, NP_BASE_ARRAY_SIZE
)
from oasislmf.pytools.gul.io import (
    LossWriter, write_negative_sidx, write_sample_header,
    write_sample_rec, read_getmodel_stream
)
from oasislmf.pytools.gul.random import get_random_generator
from oasislmf.pytools.gul.core import split_tiv, get_gul, setmaxloss, compute_mean_loss
from oasislmf.pytools.gul.utils import find_bin_idx, append_to_dict_value


# gul stream type
# probably need to set this dynamically depending on the stream type
gul_header = np.int32(1 | 2 << 24).tobytes()

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


# TODO probably getmodel get_items can be used as well. double check
def gul_get_items(input_path, ignore_file_type=set()):
    """Load the items from the items file.
    # TODO check return datatype

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
    coverages = np.empty(coverages_tiv.shape[0]+1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv   # MT2SS: 0 is empty to use same number of coverages (which start from 1)
    coverages['cur_items'].fill(0)
    del coverages_tiv

    items = gul_get_items(input_path)

    # in-place sort items in order to store them in item_map in the desired order
    # currently numba only supports a simple call to np.sort() with no `order` keyword,
    # so we do the sort here.
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map = generate_item_map(items, coverages)

    # +1 is because coverages start from 1
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

        # prepare output buffer, write stream header
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())

        # TODO: check if selectors can be used.
        writer = LossWriter(stream_out, sample_size, buff_size=65536)

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2]:
            raise ValueError(f"Expect alloc_rule to be 0 or 1 or 2, got {alloc_rule}")

        cursor = 0
        cursor_bytes = 0
        t_read = []
        t_random = []
        t_compute = []
        t_write = []
        t0 = time.time()

        # MT2SS: 1) these are not seeds, 2) before, we determined them on the fly, but here we rely on assumption of external data
        # MT2SS: 3) I don't understand. seeds is never read but only written in read_getmodel_stream. Why do we set it to unique? Just for the length?
        seeds = np.unique(items['group_id'])
        # create buffer to compute each coverage loss, in the buffer
        losses_buffer = np.zeros((sample_size + NUM_IDX + 1, np.max(coverages['max_items'])), dtype=oasis_float)
        for packed in read_getmodel_stream(run_dir, streams_in, item_map, coverages, compute, seeds):
            #event_id, items_data_by_coverage_id, coverage_ids, Nunique_coverage_ids, recs, rec_idx_ptr, seeds = packed
            event_id, compute_i, items_data, recs, rec_idx_ptr, rng_index = packed

            t1 = time.time()
            t_read.append(t1-t0)
            t0 = t1
            # print(compute[:compute_i])
            # print(items_data[:20])
            # 0/0
            # print(event_id, len(items_data_by_coverage_id), len(coverage_ids),
            #       Nunique_coverage_ids, recs.shape, len(rec_idx_ptr), rec_idx_ptr[-1])
            # # items_data_by_coverage_id, item_ids_by_coverage_id, coverage_ids, Nunique_coverage_ids, seeds = reconstruct_coverages(
            # #     event_id, damagecdfs, item_map)

            rndms = generate_rndm(seeds[:rng_index], sample_size)
            t1 = time.time()
            t_random.append(t1-t0)
            t0 = t1
            last_processed_coverage_ids_idx = 0
            while last_processed_coverage_ids_idx < compute_i:
                cursor, cursor_bytes, last_processed_coverage_ids_idx = compute_event_losses(
                    event_id, coverages, compute[:compute_i], items_data,
                    last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr,
                    damage_bins, loss_threshold, losses_buffer, alloc_rule, rndms, debug, writer.buff_size,
                    writer.int32_mv, cursor
                )
                t1 = time.time()
                t_compute.append(t1 - t0)
                t0 = t1
                # TODO use select
                stream_out.write(writer.mv[:cursor_bytes])
                t1 = time.time()
                t_write.append(t1 - t0)
                t0 = t1

                cursor = 0

    print('t_read', t_read[0], sum(t_read[1:]))
    print('t_random', t_random[0], sum(t_random[1:]))
    print('t_compute', t_compute[0], sum(t_compute[1:]))
    print('t_write', t_write[0], sum(t_write[1:]))

    return 0


# @njit(cache=True, fastmath=True)
# def reconstruct_coverages(event_id, damagecdfs, item_map):
#     """Reconstruct coverages, building a mapping of item ids to coverages.

#     Args:
#         event_id (int32): event id.
#         damagecdfs (numpy.array[damagecdf]):  array of damagecdf entries (areaperil_id, vulnerability_id)
#           for this event.
#         item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
#           the mapping between areaperil_id, vulnerability_id to item.

#     Returns:
#         Dict[int,List[ITEMS_DATA_MAP_TYPE]], Dict[int,List[ITEM_ID_TYPE]], List[COVERAGE_ID_TYPE], int, List[int64]:
#           dictionary of items data (item index and random seed) for each item in each coverage_id,
#           dictionary of item ids of each coverage_id, unique coverage ids, number of unique coverage_ids, seeds_list
#     """
#     items_data_by_coverage_id = Dict.empty(int_, List.empty_list(ITEMS_DATA_MAP_TYPE))
#     item_ids_by_coverage_id = Dict.empty(int_, List.empty_list(ITEM_ID_TYPE))
#     list_coverage_ids = List.empty_list(COVERAGE_ID_TYPE)

#     Ncoverage_ids = 0
#     seeds = set()

#     for damagecdf_i, damagecdf in enumerate(damagecdfs):
#         item_key = tuple((damagecdf['areaperil_id'], damagecdf['vulnerability_id']))

#         for item in item_map[item_key]:
#             item_id, coverage_id, group_id = item

#             rng_seed = generate_hash(group_id, event_id)

#             # append always, will filter the unqiue list_coverage_ids later
#             # for `list_coverage_ids` list is preferable over set because order is important
#             list_coverage_ids.append(coverage_id)
#             Ncoverage_ids += 1

#             append_to_dict_value(items_data_by_coverage_id, coverage_id, (damagecdf_i, rng_seed), ITEMS_DATA_MAP_TYPE)
#             append_to_dict_value(item_ids_by_coverage_id, coverage_id, item_id, nb.types.int32)

#             # using set instead of a typed list is 2x faster
#             seeds.add(rng_seed)

#     # convert list_coverage_ids to np.array to apply np.unique() to it
#     coverage_ids_tmp = np.empty(Ncoverage_ids, dtype=np.int32)
#     for j in range(Ncoverage_ids):
#         coverage_ids_tmp[j] = list_coverage_ids[j]

#     coverage_ids = np.unique(coverage_ids_tmp)
#     Nunique_coverage_ids = coverage_ids.shape[0]

#     # transform the set in a typed list in order to pass it to another jit'd function
#     seeds_list = List(seeds)

#     return items_data_by_coverage_id, item_ids_by_coverage_id, coverage_ids, Nunique_coverage_ids, seeds_list


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, coverages, coverage_ids, items_data,
                         last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr, damage_bins,
                         loss_threshold, losses, alloc_rule, rndms, debug, buff_size,
                         int32_mv, cursor):
    """Compute losses for an event.

    Args:
        event_id (int32): event id.
        items_data_by_coverage_id (Dict[COVERAGE_ID_TYPE,Tuple(int,int64)]): dict storing, for each coverage_id, a list
          of tuples (one tuple for each item containing the item index in Nbinss and recs, and the random seed).
        item_ids_by_coverage_id (Dict[COVERAGE_ID_TYPE,ITEM_ID_TYPE]): dict storing the list of item ids for each coveage_id.
        coverage_ids (numpy.array[COVERAGE_ID_TYPE]): array of **uniques** coverage ids used in this event.
        last_processed_coverage_ids_idx (int): index of the last coverage_id stored in `coverage_ids` that was fully processed
          and printed to the output stream.
        sample_size (int): number of random samples to draw.
        Nbinss (numpy.array[int]):  number of bins in all cdfs of event_id.
        recs (numpy.array[ProbMean]): all the cdfs used in event_id.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        damage_bins (List[Union[damagebindictionaryCsv, damagebindictionary]]): loaded data from the damage_bin_dict file.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        alloc_rule (int): back-allocation rule.
        rndms (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed.
        rndms_idx (Dict[int]): dic storing the map between the `seed` value and the row in the `rndms` array
          containing the drawn random samples.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        buff_size (int): size in bytes of the output buffer.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.
        cursor_bytes (int): number of bytes written in int32_mv.

    Returns:
        int, int, int: updated value of cursor, updated value of cursor_bytes, last last_processed_coverage_ids_idx
    """
    max_size_per_item =  (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size + 2 * gulSampleslevelHeader_size
    for coverage_i in range(last_processed_coverage_ids_idx, coverage_ids.shape[0]):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']  # coverages are indexed from 1
        Nitem_ids = coverage['cur_items']
        exposureValue = tiv / Nitem_ids

        # estimate max number of bytes are needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        # number of records of type gulSampleslevelRec_size is sample_size + 5 (negative sidx) + 1 (terminator line)
        est_cursor_bytes = Nitem_ids * max_size_per_item

        # return before processing this coverage if bytes to be written in mv exceed `buff_size`
        if cursor * int32_mv.itemsize + est_cursor_bytes > buff_size:
            return cursor, cursor * int32_mv.itemsize, last_processed_coverage_ids_idx

        # # sort item_ids (need to convert to np.array beforehand)
        # item_ids = np.empty(Nitem_ids, dtype=ITEM_ID_TYPE)
        # for j in range(Nitem_ids):
        #     item_ids[j] = items_data_by_coverage_id[coverage_id][j][0]

        items = items_data[coverage['start_items']: coverage['start_items'] + coverage['cur_items']]

        for item_i in range(coverage['cur_items']):
            item = items[item_i]
            damagecdf_i = item['damagecdf_i']
            rng_index = item['rng_index']

            rec = recs[rec_idx_ptr[damagecdf_i]:rec_idx_ptr[damagecdf_i + 1]]
            prob_to = rec['prob_to']
            bin_mean = rec['bin_mean']
            Nbins = len(prob_to)

            # print(rec_start, rec_end, Nbins)
            # print(prob_to)
            # print(bin_mean)
            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                # max-damage bin=damage_bins[Nbins-1] maybe is wrong...
                tiv, prob_to, bin_mean, Nbins, damage_bins[Nbins - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_i] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_i] = chance_of_loss
            losses[TIV_IDX, item_i] = exposureValue
            losses[STD_DEV_IDX, item_i] = std_dev
            losses[MEAN_IDX, item_i] = gul_mean

            if sample_size > 0:
                if debug:
                    for sample_idx, rval in enumerate(rndms[rng_index], start=1):  # TO BE CHECKED
                        losses[sample_idx, item_i] = rval
                else:
                    for sample_idx, rval in enumerate(rndms[rng_index], start=1):  # TO BE CHECKED
                        # cap `rval` to the maximum `prob_to` value (which should be 1.)
                        rval = min(rval, prob_to[Nbins - 1] - 0.00000003)

                        # find the bin in which the random value `rval` falls into
                        # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                        # there's a 1:1 mapping between indices of rec and damage_bins
                        bin_idx = find_bin_idx(rval, prob_to, Nbins)

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
        losses (numpy.array[oasis_float]): losses for all item_ids
        item_ids (numpy.array[ITEM_ID_TYPE]): ids of items whose losses are in `losses`.
        alloc_rule (int): back-allocation rule.
        tiv (oasis_float): total insured value.
        sample_idx_to_write (Dict[int,List[int64]]): indices of samples with losses above threshold
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
        for sample_idx in range(1, sample_size+1):
            if losses[sample_idx, item_j] >= loss_threshold:
                cursor = write_sample_rec(
                    sample_idx, losses[sample_idx, item_j], int32_mv, cursor)

        # write terminator for the samples for this item
        cursor = write_sample_rec(0, 0., int32_mv, cursor)

    return cursor
