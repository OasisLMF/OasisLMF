"""
This file is the entry point for the gul command for the package.

"""
import sys
import os
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
    SHIFTED_MEAN_IDX, SHIFTED_STD_DEV_IDX, SHIFTED_TIV_IDX,
    SHIFTED_CHANCE_OF_LOSS_IDX, SHIFTED_MAX_LOSS_IDX,
    ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE, ITEM_ID_TYPE, ITEMS_DATA_MAP_TYPE,
    COVERAGE_ID_TYPE, gulSampleslevelRec_size, gulSampleslevelHeader_size
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

GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)
HASH_MOD_CODE = np.int64(2147483648)


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
def generate_item_map(items):
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

    return item_map


@njit(cache=True, fastmath=True)
def generate_hash(group_id, event_id, base_seed=0):
    """Generate hash for a given `group_id`, `event_id` pair.

    Args:
        group_id (int): group id.
        event_id (int]): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        int64: hash
    """
    hash = (base_seed + (group_id * GROUP_ID_HASH_CODE) % HASH_MOD_CODE +
            (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE) % HASH_MOD_CODE

    return hash


@njit(cache=True, fastmath=True)
def generate_correlated_hash(event_id, base_seed=0):
    """Generate hash for an `event_id`.

    Args:
        event_id (int): event id.
        base_seed (int, optional): base random seed. Defaults to 0.

    Returns:
        int64: hash
    """
    hash = (base_seed + (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE) % HASH_MOD_CODE

    return hash


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

    items = gul_get_items(input_path)

    # in-place sort items in order to store them in item_map in the desired order
    # currently numba only supports a simple call to np.sort() with no `order` keyword,
    # so we do the sort here.
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map = generate_item_map(items)

    # read coverages from file
    coverages = get_coverages(input_path)

    with ExitStack() as stack:
        # set up streams
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None:
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
        # for event_id, damagecdfs, Nbinss, recs in read_getmodel_stream(run_dir, streams_in):
        for event_id, bin_lookup, item_bin_ids in read_getmodel_stream(run_dir, streams_in):

            items_data_by_coverage_id, item_ids_by_coverage_id, coverage_ids, Nunique_coverage_ids, seeds = reconstruct_coverages(
                event_id, item_bin_ids, item_map)

            rndms, rndms_idx = generate_rndm(seeds, sample_size)

            last_processed_coverage_ids_idx = 0
            while last_processed_coverage_ids_idx < Nunique_coverage_ids:
                cursor, cursor_bytes, last_processed_coverage_ids_idx = compute_event_losses(
                    event_id, items_data_by_coverage_id, item_ids_by_coverage_id, coverage_ids,
                    last_processed_coverage_ids_idx, sample_size, 0, bin_lookup, coverages,
                    damage_bins, loss_threshold, alloc_rule, rndms, rndms_idx, debug, writer.buff_size,
                    writer.int32_mv, cursor, cursor_bytes
                )

                stream_out.write(writer.mv[:cursor_bytes])

                cursor = 0
                cursor_bytes = 0

    return 0


@njit(cache=True, fastmath=True)
def reconstruct_coverages(event_id, item_bin_ids, item_map):
    """Reconstruct coverages, building a mapping of item ids to coverages.

    Args:
        event_id (int32): event id.
        damagecdfs (numpy.array[damagecdf]):  array of damagecdf entries (areaperil_id, vulnerability_id)
          for this event.
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.

    Returns:
        Dict[int,List[ITEMS_DATA_MAP_TYPE]], Dict[int,List[ITEM_ID_TYPE]], List[COVERAGE_ID_TYPE], int, List[int64]:
          dictionary of items data (item index and random seed) for each item in each coverage_id,
          dictionary of item ids of each coverage_id, unique coverage ids, number of unique coverage_ids, seeds_list
    """
    items_data_by_coverage_id = Dict.empty(int_, List.empty_list(ITEMS_DATA_MAP_TYPE))
    item_ids_by_coverage_id = Dict.empty(int_, List.empty_list(ITEM_ID_TYPE))

    item_bin_ids_by_coverage_id_type = List.empty_list(List.empty_list())
    item_bin_ids_by_coverage_id = Dict.empty(int_, item_bin_ids_by_coverage_id_type)

    list_coverage_ids = List.empty_list(COVERAGE_ID_TYPE)

    Ncoverage_ids = 0
    seeds = set()

    # for damagecdf_i, damagecdf in enumerate(damagecdfs):
    for item_key in item_bin_ids:
        # item_key = tuple((damagecdf['areaperil_id'], damagecdf['vulnerability_id']))

        for item in item_map[item_key]:
            item_id, coverage_id, group_id = item

            rng_seed = generate_hash(group_id, event_id)

            # append always, will filter the unqiue list_coverage_ids later
            # for `list_coverage_ids` list is preferable over set because order is important
            list_coverage_ids.append(coverage_id)
            Ncoverage_ids += 1

            # append_to_dict_value(items_data_by_coverage_id, coverage_id, (damagecdf_i, rng_seed), ITEMS_DATA_MAP_TYPE)
            append_to_dict_value(items_data_by_coverage_id, coverage_id, (0, rng_seed), ITEMS_DATA_MAP_TYPE)
            append_to_dict_value(item_ids_by_coverage_id, coverage_id, item_id, nb.types.int32)
            append_to_dict_value(item_bin_ids_by_coverage_id, coverage_id,
                                 item_bin_ids[item_key], List.empty_list(ITEM_ID_TYPE))

            # using set instead of a typed list is 2x faster
            seeds.add(rng_seed)

    # convert list_coverage_ids to np.array to apply np.unique() to it
    coverage_ids_tmp = np.empty(Ncoverage_ids, dtype=np.int32)
    for j in range(Ncoverage_ids):
        coverage_ids_tmp[j] = list_coverage_ids[j]

    coverage_ids = np.unique(coverage_ids_tmp)
    Nunique_coverage_ids = coverage_ids.shape[0]

    # transform the set in a typed list in order to pass it to another jit'd function
    seeds_list = List(seeds)

    return items_data_by_coverage_id, item_ids_by_coverage_id, item_bin_ids_by_coverage_id, coverage_ids, Nunique_coverage_ids, seeds_list


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, items_data_by_coverage_id, item_ids_by_coverage_id, item_bin_ids_by_coverage_id, coverage_ids,
                         last_processed_coverage_ids_idx, sample_size, Nbinss, bin_lookup, coverages, damage_bins,
                         loss_threshold, alloc_rule, rndms, rndms_idx, debug, buff_size,
                         int32_mv, cursor, cursor_bytes):
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
    used_bins_prob_to = bin_lookup[:, 0]
    used_bins_bin_mean = bin_lookup[:, 1]

    for coverage_id in coverage_ids[last_processed_coverage_ids_idx:]:
        tiv = coverages[coverage_id - 1]  # coverages are indexed from 1
        Nitems = len(items_data_by_coverage_id[coverage_id])
        exposureValue = tiv / Nitems

        Nitem_ids = len(item_ids_by_coverage_id[coverage_id])

        # estimate max number of bytes are needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        # number of records of type gulSampleslevelRec_size is sample_size + 5 (negative sidx) + 1 (terminator line)
        est_cursor_bytes = Nitem_ids * (
            (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size + 2 * gulSampleslevelHeader_size
        )

        # return before processing this coverage if bytes to be written in mv exceed `buff_size`
        if cursor_bytes + est_cursor_bytes > buff_size:
            return cursor, cursor_bytes, last_processed_coverage_ids_idx

        # sort item_ids (need to convert to np.array beforehand)
        item_ids = np.empty(Nitem_ids, dtype=ITEM_ID_TYPE)
        for j in range(Nitem_ids):
            item_ids[j] = item_ids_by_coverage_id[coverage_id][j]

        item_ids_argsorted = np.argsort(item_ids)
        item_ids_sorted = item_ids[item_ids_argsorted]

        # dict that contains, for each item, the list of all random sidx with losses
        # larger than loss_threshold. caching this allows for smaller loop  when
        # writing the output
        sample_idx_to_write = Dict()

        # nomenclature change in gulcalc `gilv[item_id, losses]` becomes losses in gulpy
        losses = np.zeros((sample_size + NUM_IDX, Nitems), dtype=oasis_float)

        for item_j, item_id_sorted in enumerate(item_ids_argsorted):

            k, seed = items_data_by_coverage_id[coverage_id][item_id_sorted]
            bin_ids = item_bin_ids_by_coverage_id[coverage_id][item_id_sorted]

            # prob_to = recs[k]['prob_to']
            # bin_mean = recs[k]['bin_mean']
            # Nbins = Nbinss[k]
            prob_to = used_bins_prob_to[bin_ids]
            bin_mean = used_bins_bin_mean[bin_ids]
            Nbins = len(prob_to)

            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                tiv, prob_to, bin_mean, Nbins, damage_bins[Nbins - 1]['bin_to'],
            )

            losses[SHIFTED_MAX_LOSS_IDX, item_j] = max_loss
            losses[SHIFTED_CHANCE_OF_LOSS_IDX, item_j] = chance_of_loss
            losses[SHIFTED_TIV_IDX, item_j] = exposureValue
            losses[SHIFTED_STD_DEV_IDX, item_j] = std_dev
            losses[SHIFTED_MEAN_IDX, item_j] = gul_mean

            idx_loss_above_threshold = List.empty_list(nb.types.int64)

            if sample_size > 0:
                if debug:
                    for sample_idx, rval in enumerate(rndms[rndms_idx[seed]]):
                        losses[sample_idx + NUM_IDX, item_j] = rval
                else:
                    for sample_idx, rval in enumerate(rndms[rndms_idx[seed]]):
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
                            losses[sample_idx + NUM_IDX, item_j] = gul
                            idx_loss_above_threshold.append(sample_idx)

            sample_idx_to_write[item_j] = idx_loss_above_threshold

        cursor, cursor_bytes = write_losses(event_id, losses, item_ids_sorted, alloc_rule, tiv,
                                            sample_idx_to_write, int32_mv, cursor, cursor_bytes)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    return cursor, cursor_bytes, last_processed_coverage_ids_idx


@njit(cache=True, fastmath=True)
def write_losses(event_id, losses, item_ids, alloc_rule, tiv, sample_idx_to_write,
                 int32_mv, cursor, cursor_bytes):
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
        cursor_bytes (int): number of bytes written in int32_mv.

    Returns:
        int, int: updated values of cursor and cursor_bytes
    """
    # note that Nsamples = sample_size + NUM_IDX
    Nsamples, Nitems = losses.shape

    if alloc_rule == 2:
        losses = setmaxloss(losses)

    # split tiv has to be executed after setmaxloss, if alloc_rule==2.
    if alloc_rule >= 1:
        # check whether the sum of losses-per-sample exceeds TIV
        # if so, split TIV in proportion to the losses
        for sample_i in range(Nsamples):
            split_tiv(losses[sample_i], tiv)

    # output the losses for all the items
    for item_j in range(Nitems):

        # write header
        cursor, cursor_bytes = write_sample_header(
            event_id, item_ids[item_j], int32_mv, cursor, cursor_bytes)

        # write negative sidx
        cursor, cursor_bytes = write_negative_sidx(
            MAX_LOSS_IDX, losses[SHIFTED_MAX_LOSS_IDX, item_j],
            CHANCE_OF_LOSS_IDX, losses[SHIFTED_CHANCE_OF_LOSS_IDX, item_j],
            TIV_IDX, losses[SHIFTED_TIV_IDX, item_j],
            STD_DEV_IDX, losses[SHIFTED_STD_DEV_IDX, item_j],
            MEAN_IDX, losses[SHIFTED_MEAN_IDX, item_j],
            int32_mv, cursor, cursor_bytes
        )

        # write the random samples (only those with losses above the threshold)
        for sample_idx in sample_idx_to_write[item_j]:
            cursor, cursor_bytes = write_sample_rec(
                sample_idx + 1, losses[sample_idx + NUM_IDX, item_j], int32_mv, cursor, cursor_bytes)

        # write terminator for the samples for this item
        cursor, cursor_bytes = write_sample_rec(0, 0., int32_mv, cursor, cursor_bytes)

    return cursor, cursor_bytes
