"""
This file is the entry point for the gul command for the package.

"""
import time
import sys
import os
import logging
from contextlib import ExitStack
from typing import OrderedDict
import numpy as np
from numpy.random import Generator, MT19937

from numba import njit
import numba as nb
from numba.types import int_
from numba.typed import Dict, List

from math import sqrt  # faster than numpy.sqrt

from oasislmf.pytools.getmodel.manager import get_damage_bins, Item
from oasislmf.pytools.getmodel.common import oasis_float, areaperil_int
from oasislmf.pytools.gul.common import gulSampleslevelHeader, gulSampleslevelRec, oasis_float_to_int_size

from oasislmf.pytools.gul.common import MEAN_IDX, STD_DEV_IDX, TIV_IDX, CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX, NUM_IDX
from oasislmf.pytools.gul.common import SHIFTED_MEAN_IDX, SHIFTED_STD_DEV_IDX, SHIFTED_TIV_IDX, SHIFTED_CHANCE_OF_LOSS_IDX, SHIFTED_MAX_LOSS_IDX
from oasislmf.pytools.gul.common import ProbMean, damagecdfrec, damagecdfrec_stream
from oasislmf.pytools.gul.random import get_random_generator
from oasislmf.pytools.gul.io import read_getmodel_stream
from oasislmf.pytools.gul.io import write_negative_sidx, write_sample_header, write_sample_rec, LossWriter
from oasislmf.pytools.gul.core import split_tiv, get_gul, setmaxloss, compute_mean_loss
from oasislmf.pytools.gul.utils import find_bin_idx, append_to_dict_value

# gul stream type
# probably need to set this dynamically depending on the stream type
gul_header = np.int32(1 | 2 << 24).tobytes()

logger = logging.getLogger(__name__)


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)
HASH_MOD_CODE = np.int64(2147483648)


loss_rel_size = 1
gulSampleslevelRec_size = gulSampleslevelRec.size
gulSampleslevelHeader_size = gulSampleslevelHeader.size


def get_coverages(input_path, ignore_file_type=set()):
    """Loads the coverages from the coverages file.

    Args:
        input_path (str): the path containing the coverage file.
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns: 
        coverages (np.array[oasis_float]): the coverages read from file.
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


# TODO probably I can use getmodel get_items. double check
def gul_get_items(input_path, ignore_file_type=set()):
    """Loads the items from the items file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        items (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
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
def generate_item_map(items, item_key_type, item_value_type):
    """Generate item_map; requires items to be sorted.

    Args:
        items (_type_): _description_
        item_key_type (_type_): _description_
        item_value_type (_type_): _description_

    Returns:
        _type_: _description_
    """
    # # in-place sort items in order to store them in item_map in the desired order
    # items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])

    # item_value_type = Item_map_rec
    item_map = Dict.empty(item_key_type, List.empty_list(item_value_type))
    Nitems = items.shape[0]

    for j in range(Nitems):
        append_to_dict_value(
            item_map,
            tuple((items[j]['areaperil_id'], items[j]['vulnerability_id'])),
            tuple((items[j]['id'], items[j]['coverage_id'], items[j]['group_id'])),
            item_value_type
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
        hash (int64): hash
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
        hash (int64): hash
    """
    hash = (base_seed + (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE) % HASH_MOD_CODE

    return hash


def run(run_dir, ignore_file_type, sample_size, loss_threshold, alloc_rule, debug,
        random_generator, file_in=None, file_out=None, **kwargs):
    """
    Runs the main process of the gul calculation.

    Args:
        run_dir: (str) the directory of where the process is running
        ignore_file_type: set(str) file extension to ignore when loading
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
    # currently numba only supports simple call to np.sort() with no `order` keyword.
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map_key_type = nb.types.Tuple((nb.types.uint32, nb.types.int32))
    item_map_value_type = nb.types.UniTuple(nb.types.int32, 3)
    item_map = generate_item_map(items, item_map_key_type, item_map_value_type)

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

        # TODO: optimize LossWriter: cleanup, and check if selectors can be used.
        writer = LossWriter(stream_out, sample_size, buff_size=65536)

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2]:
            raise ValueError(f"Expect alloc_rule to be 0 or 1 or 2, got {alloc_rule}")

        # TODO: probably here we need a with Losswriter context
        # TODO: rename mode1_stats_2_type and mode1_stats_2 to more sensible names
        mode1_stats_2_type = nb.types.UniTuple(nb.types.int64, 2)
        mode1_item_id_dtype = nb.types.int32

        cursor = 0
        cursor_bytes = 0
        for event_id, damagecdfs, Nbinss, recs in read_getmodel_stream(run_dir, streams_in):

            mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs_arr, Nunique_coverage_ids, seeds = reconstruct_coverages_properties(
                event_id, damagecdfs, item_map, mode1_stats_2_type, mode1_item_id_dtype)

            rndms, rndms_idx = generate_rndm(seeds, sample_size)

            coverage_idx = 0
            while coverage_idx < Nunique_coverage_ids:
                cursor, cursor_bytes, coverage_idx = compute_event_losses(
                    event_id, mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs_arr, sample_size,
                    Nbinss, recs, coverages, damage_bins, loss_threshold, alloc_rule, rndms, rndms_idx,
                    debug, writer.int32_mv, writer.buff_size, cursor, cursor_bytes, coverage_idx
                )

                stream_out.write(writer.mv[:cursor_bytes])

                cursor = 0
                cursor_bytes = 0

    return 0


@njit(cache=True, fastmath=True)
def reconstruct_coverages_properties(event_id, damagecdfs, item_map, mode1_stats_2_type, mode1_item_id_dtype):
    # TODO: write docstring

    mode1_stats_2 = Dict.empty(int_, List.empty_list(mode1_stats_2_type))
    mode1_item_id = Dict.empty(int_, List.empty_list(mode1_item_id_dtype))

    mode1UsedCoverageIDs = List.empty_list(nb.types.int32)
    Ncoverage_ids = 0
    seeds = set()

    for damagecdf_i, damagecdf in enumerate(damagecdfs):
        item_key = tuple((damagecdf['areaperil_id'], damagecdf['vulnerability_id']))

        for item in item_map[item_key]:
            item_id, coverage_id, group_id = item

            rng_seed = generate_hash(group_id, event_id)

            # append always, will filter the unqiue coverage_ids later
            # here list is preferable over set since order is important
            mode1UsedCoverageIDs.append(coverage_id)
            Ncoverage_ids += 1

            append_to_dict_value(mode1_stats_2, coverage_id, (damagecdf_i, rng_seed), mode1_stats_2_type)
            # def_lst = List.empty_list(mode1_stats_2_type)
            # mode1_stats_2.setdefault(coverage_id, def_lst)
            # lst = mode1_stats_2[coverage_id]
            # lst.append((damagecdf_i, rng_seed))
            # mode1_stats_2[coverage_id] = lst

            append_to_dict_value(mode1_item_id, coverage_id, item_id, nb.types.int32)
            # def_lst = List.empty_list(nb.types.int32)
            # mode1_item_id.setdefault(coverage_id, def_lst)
            # lst = mode1_item_id[coverage_id]
            # lst.append(item_id)
            # mode1_item_id[coverage_id] = lst

            # using set instead of a typed list is 2x faster
            seeds.add(rng_seed)

    # convert mode1UsedCoverageIDs to np.array to apply np.unique() to it
    mode1UsedCoverageIDs_arr_tmp = np.empty(Ncoverage_ids, dtype=np.int32)
    for j in range(Ncoverage_ids):
        mode1UsedCoverageIDs_arr_tmp[j] = mode1UsedCoverageIDs[j]

    mode1UsedCoverageIDs_arr = np.unique(mode1UsedCoverageIDs_arr_tmp)
    Nunique_coverage_ids = mode1UsedCoverageIDs_arr.shape[0]

    # transform the set in a typed list in order to pass it to another jit'd function
    lst_seeds = List(seeds)

    return mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs_arr, Nunique_coverage_ids, lst_seeds


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs_arr, sample_size,
                         Nbinss, recs, coverages, damage_bins, loss_threshold, alloc_rule, rndms, rndms_idx,
                         debug, int32_mv, buff_size, cursor, cursor_bytes, coverage_idx):

    # mode1UsedCoverageIDs_arr is already np.unique() applied

    for coverage_id in mode1UsedCoverageIDs_arr[coverage_idx:]:
        tiv = coverages[coverage_id - 1]  # coverages are indexed from 1
        Nitems = len(mode1_stats_2[coverage_id])
        exposureValue = tiv / Nitems

        Nitem_ids = len(mode1_item_id[coverage_id])

        # estimate max number of bytes are needed to output this coverage
        # conservatively assume all random samples are printed (loss>loss_threshold)
        # number of records of type gulSampleslevelRec_size is sample_size + 5 (negative sidx) + 1 (terminator line)
        est_cursor_bytes = Nitem_ids * (
            (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size + 2 * gulSampleslevelHeader_size
        )

        # return before processing this coverage if bytes to be written in mv exceed `buff_size`
        if cursor_bytes + est_cursor_bytes > buff_size:
            return cursor, cursor_bytes, coverage_idx

        # sort item_ids (need to convert to np.array beforehand)
        item_ids_arr = np.empty(Nitem_ids, dtype=np.int32)
        for j in range(Nitem_ids):
            item_ids_arr[j] = mode1_item_id[coverage_id][j]

        item_ids_arr_argsorted = np.argsort(item_ids_arr)
        item_ids_arr_sorted = item_ids_arr[item_ids_arr_argsorted]

        # dict that contains, for each item, the list of all random sidx with loss
        # larger than loss_threshold. caching this allows for smaller loop  when
        # writing the output
        items_loss_above_threshold = Dict()

        # nomenclature change in gulcalc `gilv[item_id, loss]` becomes loss in gulpy
        loss_Nrows = sample_size + NUM_IDX
        loss = np.zeros((loss_Nrows, Nitems), dtype=oasis_float)

        for item_j, item_id_sorted in enumerate(item_ids_arr_argsorted):

            k, seed = mode1_stats_2[coverage_id][item_id_sorted]

            prob_to = recs[k]['prob_to']
            bin_mean = recs[k]['bin_mean']
            Nbins = Nbinss[k]

            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                tiv, prob_to, bin_mean, Nbins, damage_bins[Nbins - 1]['bin_to'],
            )

            # print(tiv, exposureValue)
            loss[SHIFTED_MAX_LOSS_IDX, item_j] = max_loss
            loss[SHIFTED_CHANCE_OF_LOSS_IDX, item_j] = chance_of_loss
            loss[SHIFTED_TIV_IDX, item_j] = exposureValue
            loss[SHIFTED_STD_DEV_IDX, item_j] = std_dev
            loss[SHIFTED_MEAN_IDX, item_j] = gul_mean

            if sample_size > 0:
                if debug:
                    for sample_idx, rval in enumerate(rndms[rndms_idx[seed]]):
                        loss[sample_idx + NUM_IDX, item_j] = rval
                else:
                    # maybe define the list outside and clear it here
                    idx_loss_above_threshold = List.empty_list(nb.types.int64)

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
                            loss[sample_idx + NUM_IDX, item_j] = gul
                            idx_loss_above_threshold.append(sample_idx)

                    items_loss_above_threshold[item_j] = idx_loss_above_threshold

        cursor, cursor_bytes = write_output(loss, item_ids_arr_sorted, alloc_rule, tiv, event_id,
                                            items_loss_above_threshold, int32_mv, cursor, cursor_bytes)

        # register that another `coverage_id` has been processed
        coverage_idx += 1

    return cursor, cursor_bytes, coverage_idx


@njit(cache=True, fastmath=True)
def write_output(loss, item_ids_arr_sorted, alloc_rule, tiv, event_id, items_loss_above_threshold, int32_mv, cursor, cursor_bytes):

    # note that Nsamples = sample_size + NUM_IDX
    Nsamples, Nitems = loss.shape

    if alloc_rule == 2:
        loss = setmaxloss(loss)

    # split tiv has to be executed after setmaxloss, if alloc_rule==2.
    if alloc_rule >= 1:
        # check whether the sum of losses-per-sample exceeds TIV
        # if so, split TIV in proportion to losses
        for sample_i in range(Nsamples):
            split_tiv(loss[sample_i], tiv)

    # output the losses for all the items
    for item_j in range(Nitems):

        # write header
        cursor, cursor_bytes = write_sample_header(
            event_id, item_ids_arr_sorted[item_j], int32_mv, cursor, cursor_bytes)
        # print("header: ", cursor, cursor_bytes)

        # write negative sidx
        cursor, cursor_bytes = write_negative_sidx(
            MAX_LOSS_IDX, loss[SHIFTED_MAX_LOSS_IDX, item_j],
            CHANCE_OF_LOSS_IDX, loss[SHIFTED_CHANCE_OF_LOSS_IDX, item_j],
            TIV_IDX, loss[SHIFTED_TIV_IDX, item_j],
            STD_DEV_IDX, loss[SHIFTED_STD_DEV_IDX, item_j],
            MEAN_IDX, loss[SHIFTED_MEAN_IDX, item_j],
            int32_mv, cursor, cursor_bytes
        )
        # print("neg sidx: ", cursor, cursor_bytes)

        # write the random samples (only those with losses above the threshold)
        for sample_idx in items_loss_above_threshold[item_j]:
            cursor, cursor_bytes = write_sample_rec(
                sample_idx + 1, loss[sample_idx + NUM_IDX, item_j], int32_mv, cursor, cursor_bytes)
        # print("sidx: ", cursor, cursor_bytes)

        # write terminator for the samples for this item
        cursor, cursor_bytes = write_sample_rec(0, 0., int32_mv, cursor, cursor_bytes)
        # print("termin: ", cursor, cursor_bytes)

    return cursor, cursor_bytes
