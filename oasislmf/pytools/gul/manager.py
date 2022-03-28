"""
This file is the entry point for the gul command for the package.

"""
import time
import sys
import os
import logging
from contextlib import ExitStack
import numpy as np
from numpy.random import Generator, MT19937

import numba as nb
from numba.types import int_
from numba.typed import Dict, List

from scipy.stats import qmc

from math import sqrt  # faster than numpy.sqrt

from oasislmf.pytools.getmodel.manager import get_mean_damage_bins, get_damage_bins, get_items, Item
from oasislmf.pytools.getmodel.common import oasis_float, areaperil_int
from oasislmf.pytools.gul.common import gulSampleslevelHeader, gulSampleslevelRec, gulSampleFullRecord
from oasislmf.pytools.gul.common import processrecData, gulItemIDLoss, oasis_float_to_int_size

from oasislmf.pytools.gul.common import ProbMean, damagecdfrec, Item_map_rec, damagecdfrec_stream

# gul stream type
# probably need to set this dynamically depending on the stream type
gul_header = np.int32(1 | 2 << 24).tobytes()

logger = logging.getLogger(__name__)

MEAN_IDX = -1
STD_DEV_IDX = -2
TIV_IDX = -3
CHANCE_OF_LOSS_IDX = -4
MAX_LOSS_IDX = -5

NUM_IDX = 5


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)
HASH_MOD_CODE = np.int64(2147483648)


loss_rel_size = 1
gulSampleslevelRec_size = gulSampleslevelRec.size
gulSampleslevelHeader_size = gulSampleslevelHeader.size


def read_getmodel_stream(run_dir, streams_in, buff_size=65536):
    """Read the getmodel output stream.

    Args:
        run_dir ([str]): Path to the run directory.
        streams_in ([]): input streams

    Raises:
        ValueError: If the stream type is not 1.

    Yields:
        All the cdfs related to 1 event.

        damagecdfs: array-like([damagecdf])
        Nbinss: array-like([int]) 
        recs: array-like([ProbMean])

    """
    # determine stream type
    stream_type = np.frombuffer(streams_in.read(4), dtype='i4')

    # TODO: make sure the bit1 and bit 2-4 compliance is checked
    # see https://github.com/OasisLMF/ktools/blob/master/docs/md/CoreComponents.md
    if stream_type[0] != 1:
        raise ValueError(f"FATAL: Invalid stream type: expect 1, got {stream_type[0]}.")

    # get damage bins from fileFIXME
    static_path = os.path.join(run_dir, 'static')
    damage_bins = get_damage_bins(static_path=static_path)

    # maximum number of damage bins (individual items can have up to `total_bins` bins)
    if damage_bins.shape[0] == 0:
        max_Nbins = 1000
    else:
        max_Nbins = damage_bins.shape[0]

    # maximum number of entries is buff_size divided by the minimum entry size
    # (corresponding to a 1-bin only cdf)
    min_size_cdf_entry = damagecdfrec_stream.size + ProbMean.size + 4

    # each record from getmodel stream is expected to contain:
    # 1 damagecdfrec_stream obj, 1 int (Nbins), a number `Nbins` of ProbMean objects

    # use a memoryview of size twice the buff_size to ensure `read_into1` always reads the maximum amount
    # data possible (the largest between the pipe limit and the remaining memory to fill the memoryview)
    mv = memoryview(bytearray(buff_size * 2))
    int32_mv = np.ndarray(buff_size * 2 // 4, buffer=mv, dtype='i4')

    cursor = 0
    valid_buf = 0
    last_event_id = -1
    damagecdfs, Nbinss, recs = [], [], []
    len_read = 1
    size_cdf_entry = min_size_cdf_entry
    while True:
        if len_read > 0:
            # read stream from valid_buf onwards
            len_read = streams_in.readinto1(mv[valid_buf:])
            # extend the valid_buf by the same amount of data that was read
            valid_buf += len_read

        # read the streamed data into formatted data
        cursor, i, yield_event, event_id, damagecdf, Nbins, rec, last_event_id = stream_to_data(
            int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id)

        if yield_event:
            # event is fully read, append the last chunk of data to the list of this event
            damagecdfs.append(damagecdf[:i])
            Nbinss.append(Nbins[:i])
            recs.append(rec[:i])

            yield last_event_id, np.concatenate(damagecdfs), np.concatenate(Nbinss), np.concatenate(recs)

            # start a new list for the new event, storing the first element
            damagecdfs, Nbinss, recs = [damagecdf[i:i + 1]], [Nbins[i:i + 1]], [rec[i:i + 1]]
            last_event_id = event_id

        else:
            # the current event is not finished, keep appending data about this event
            damagecdfs.append(damagecdf[:i + 1])
            Nbinss.append(Nbins[:i + 1])
            recs.append(rec[:i + 1])

        # convert cursor to bytes
        cursor_buf = cursor * int32_mv.itemsize

        if valid_buf == cursor_buf:
            # this is the last cycle, all data has been read, append the last chunk of data
            damagecdfs.append(damagecdf[:i])
            Nbinss.append(Nbins[:i])
            recs.append(rec[:i])

            yield last_event_id, np.concatenate(damagecdfs), np.concatenate(Nbinss), np.concatenate(recs)
            break

        else:
            # this is not the last cycle
            # move the un-read data to the beginning of the memoryview
            mv[:valid_buf - cursor_buf] = mv[cursor_buf:valid_buf]

            # update the length of the valid data
            valid_buf -= cursor_buf


@nb.njit(cache=True, fastmath=True)
def stream_to_data(int32_mv, valid_buf, size_cdf_entry, max_Nbins, last_event_id):
    """Read streamed data into formatted data.

    Args:
        int32_mv (ndarray): int32 view of the buffer
        valid_buf (int): number of bytes with valid data
        size_cdf_entry (int): size (in bytes) of a single record
        max_Nbins (int): Maximum number of probability bins
        last_event_id (int): event_id of the last event that was completed

    Returns:
        cursor (int): number of int numbers read from the int32_mv ndarray.
        i (int): number of cdf data entries read.
        yield_event (bool): if True, the current event (id=`event_id`) is completed.
        event_id (int): id of the event being read.
        damagecdf (array-like[damagecdf]):
        Nbins (array-like[damagecdf]):
        rec (array-like[damagecdf]):
        last_event_id (int): event_id of the last event that was completed

    # TODO: don't store event_id (it's wasted space). need to change this everywhere `damagecdfrec` is used

    """
    valid_len = valid_buf // size_cdf_entry
    yield_event = False

    Nbins = np.zeros(valid_len, dtype='i4')
    damagecdf = np.zeros(valid_len, dtype=damagecdfrec)
    rec = np.zeros((valid_len, max_Nbins), dtype=ProbMean)

    i = 0
    cursor = 0
    while cursor < valid_len:
        event_id, cursor = int32_mv[cursor], cursor + 1
        damagecdf[i]['areaperil_id'], cursor = int32_mv[cursor:cursor + 1].view(areaperil_int)[0], cursor + 1
        damagecdf[i]['vulnerability_id'], cursor = int32_mv[cursor], cursor + 1
        Nbins[i] = int32_mv[cursor]
        cursor += 1

        for j in range(Nbins[i]):
            rec[i, j]['prob_to'] = int32_mv[cursor: cursor + oasis_float_to_int_size].view(oasis_float)[0]
            cursor += oasis_float_to_int_size
            rec[i, j]['bin_mean'] = int32_mv[cursor: cursor + oasis_float_to_int_size].view(oasis_float)[0]
            cursor += oasis_float_to_int_size

        if event_id != last_event_id:
            # a new event has started
            if last_event_id > 0:
                # this is not the beginning of first event
                yield_event = True

                return cursor, i, yield_event, event_id, damagecdf, Nbins, rec, last_event_id

            last_event_id = event_id

        i += 1

    return cursor, i - 1, yield_event, event_id, damagecdf, Nbins, rec, last_event_id


def get_coverages(input_path, ignore_file_type=set()):
    """
    Loads the coverages from the coverages file.

    Args:
        input_path: (str) the path containing the coverage file
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: np.array[oasis_float]
        coverages array
    """
    input_files = set(os.listdir(input_path))

    # TODO: store default filenames (e.g., coverages.bin) in a parameters file

    if "coverages.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'coverages.csv')}")
        coverages = np.fromfile(os.path.join(
            input_path, "coverages.bin"), dtype=oasis_float)

    elif "coverages.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'coverages.csv')}")
        coverages = np.genfromtxt(os.path.join(
            input_path, "coverages.csv"), dtype=oasis_float, delimiter=",")

    else:
        raise FileNotFoundError(f'coverages file not found at {input_path}')

    return coverages


@nb.njit(cache=True, fastmath=True)
def generate_hash(group_id, event_id, rand_seed=0):
    """
    Generate hash for group_id, event_id

    Args:
        group_id ([type]): [description]
        event_id ([type]): [description]
        rand_seed (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    hashed = rand_seed
    hashed += (group_id * GROUP_ID_HASH_CODE) % HASH_MOD_CODE
    hashed += (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE
    hashed %= HASH_MOD_CODE

    return hashed


@nb.njit(cache=True, fastmath=True)
def generate_correlated_hash(event_id, rand_seed=0):
    """
    Generate hash for group_id, event_id

    Args:
        event_id ([type]): [description]
        rand_seed (int, optional): [description]. Defaults to 0.
    Returns:
        [type]: [description]
    """
    hashed = rand_seed
    hashed += (event_id * EVENT_ID_HASH_CODE) % HASH_MOD_CODE
    hashed %= HASH_MOD_CODE

    return hashed


def generate_rndm_MT19937(seeds, n, rndms_idx, rndms):

    for i, seed in enumerate(seeds):
        rng = Generator(MT19937(seed=123))
        rndms_idx[seed] = i
        rndms[i, :] = rng.uniform(0., 1., size=n)


def generate_rndm_LHS(seeds, n, rndms_idx, rndms):

    for i, seed in enumerate(seeds):
        rng = qmc.LatinHypercube(d=1, seed=seed)
        rndms_idx[seed] = i
        rndms[i, :] = rng.random(n).ravel()


# TODO probably I can use getmodel get_items. double check
def gul_get_items(input_path, ignore_file_type=set()):
    """
    Loads the items from the items file.

    Args:
        input_path: (str) the path pointing to the file
        ignore_file_type: set(str) file extension to ignore when loading

    Returns: (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
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


@nb.njit()
def append_to_dict_entry(d, key, value, value_type):
    # append is done in-place
    def_lst = List.empty_list(value_type)
    d.setdefault(key, def_lst)
    lst = d[key]
    lst.append(value)
    d[key] = lst


@nb.njit(cache=True, fastmath=True)
def generate_item_map(items, item_key_type, item_value_type):
    """ generate item_map; requires items to be sorted"""
    # # in-place sort items in order to store them in item_map in the desired order
    # items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])

    # item_value_type = Item_map_rec
    item_map = Dict.empty(item_key_type, List.empty_list(item_value_type))
    Nitems = items.shape[0]

    for j in range(Nitems):
        append_to_dict_entry(
            item_map,
            tuple((items[j]['areaperil_id'], items[j]['vulnerability_id'])),
            tuple((items[j]['id'], items[j]['coverage_id'], items[j]['group_id'])),
            item_value_type
        )

    return item_map


@nb.njit(cache=True)
def gen_new_dicts(mode1_stats_2_type):
    mode1_stats_2 = Dict.empty(int_, List.empty_list(mode1_stats_2_type))
    rndms_idx = Dict.empty(nb.types.int64, int_)

    mytype = nb.types.int32
    mode1_item_id = Dict.empty(int_, List.empty_list(mytype))

    return mode1_stats_2, mode1_item_id, rndms_idx


def run(run_dir, ignore_file_type, sample_size, loss_threshold, alloc_rule, random_numbers_file, debug,
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
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map_key_type = nb.types.Tuple((nb.types.uint32, nb.types.int32))
    item_map_value_type = nb.types.UniTuple(nb.types.int32, 3)
    item_map = generate_item_map(items, item_map_key_type, item_map_value_type)

    coverages = get_coverages(input_path)

    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None:
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        # prepare output buffer
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())

        writer = LossWriter(stream_out, sample_size, buff_size=65536 * 100)

        if random_numbers_file:
            rndm = np.loadtxt(random_numbers_file)

            assert len(rndm) > sample_size, \
                f"The number of random values in the file must be strictly larger than sample size, " \
                f"but got {len(rndm)} random numbers with sample size={sample_size}."
        else:
            # define random generator
            if random_generator == 0:
                generate_rndm = generate_rndm_MT19937
                logger.info("Random generator: MT19937")

            elif random_generator == 1:
                generate_rndm = generate_rndm_LHS
                logger.info("Random generator: Latin Hypercube")

        # min condition is needed to avoid seg fault if sample_size=0
        # rndms = get_arr_chunks(rndm, len_rndm, max(1, sample_size))

        # TODO: probably here I need a with Losswriter context
        # get the stream, for each entry in the stream:

        assert alloc_rule in [0, 1, 2], f"Expect alloc_rule to be 0 or 1 or 2, got {alloc_rule}"

        # def compute_nitems(damagecdfs, item_map):
        #     nitems = 0
        #     for damagecdf in damagecdfs:
        #         nitems += len(item_map[tuple(damagecdf[['areaperil_id', 'vulnerability_id']])])

        #     return nitems

        def compute_mv_size(damagecdfs, item_map, sample_size):
            # compute number of items in this event
            ntotitems = 0
            for damagecdf in damagecdfs:
                ntotitems += len(item_map[tuple(damagecdf[['areaperil_id', 'vulnerability_id']])])

            mv_size = ntotitems * ((NUM_IDX + sample_size + 1) * gulSampleslevelRec_size + gulSampleslevelHeader_size)

            return mv_size

        if alloc_rule == 0:

            cursor = 0
            cursor_bytes = 0
            for event_id, damagecdfs, Nbinss, recs in read_getmodel_stream(run_dir, streams_in):
                # I don't understand why it doesn't work using mv of size mv_size
                mv_size = compute_mv_size(damagecdfs, item_map, sample_size)
                cursor, cursor_bytes = processrec(damagecdfs, Nbinss, recs, damage_bins, coverages,
                                                  item_map, sample_size, loss_threshold,
                                                  debug, generate_rndm, writer.int32_mv, cursor, cursor_bytes)
                stream_out.write(writer.mv[:cursor_bytes])
                cursor = 0
                cursor_bytes = 0

        else:
            # type of mode1_bin_ids values
            mode1_stats_2_type = nb.types.UniTuple(nb.types.int64, 2)

            for event_id, damagecdfs, Nbinss, recs in read_getmodel_stream(run_dir, streams_in):
                mode1_stats_2, mode1_item_id, rndms_idx = gen_new_dicts(mode1_stats_2_type)
                cursor = 0
                cursor_bytes = 0

                mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs, seeds = processrec_mode1(
                    event_id, damagecdfs, item_map, mode1_stats_2, mode1_stats_2_type, mode1_item_id)

                # bin_lookup_ndarr = np.empty((len(bin_lookup),), dtype=ProbMean)
                # bin_lookup_arr = np.array(bin_lookup)
                # bin_lookup_ndarr[:]['prob_to'] = bin_lookup_arr[:, 0]
                # bin_lookup_ndarr[:]['bin_mean'] = bin_lookup_arr[:, 1]
                rndms = np.zeros((len(seeds), sample_size), dtype=oasis_float)
                generate_rndm(seeds, sample_size, rndms_idx, rndms)

                cursor, cursor_bytes = outputmode1data(event_id, mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs, sample_size,
                                                       Nbinss, recs, coverages, damage_bins, loss_threshold, alloc_rule, rndms_idx, rndms,
                                                       debug, writer.int32_mv, cursor, cursor_bytes)

                stream_out.write(writer.mv[:cursor_bytes])

    return 0


@nb.njit(cache=True, fastmath=True)
def write_sample_header(event_id, item_id, int32_mv, cursor, cursor_bytes):
    int32_mv[cursor], cursor = event_id, cursor + 1
    int32_mv[cursor], cursor = item_id, cursor + 1
    cursor_bytes += gulSampleslevelHeader_size

    return cursor, cursor_bytes


@nb.njit(cache=True, fastmath=True)
def write_sample_rec(sidx, loss, int32_mv, cursor, cursor_bytes):

    int32_mv[cursor], cursor = sidx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = loss
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    return cursor, cursor_bytes


@nb.njit(cache=True, fastmath=True)
def write_negative_sidx(max_loss_idx, max_loss, chance_of_loss_idx, chance_of_loss,
                        tiv_idx, tiv, std_dev_idx, std_dev, mean_idx, gul_mean,
                        int32_mv, cursor, cursor_bytes):

    int32_mv[cursor], cursor = max_loss_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = max_loss
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = chance_of_loss_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = chance_of_loss
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = tiv_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = tiv
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = std_dev_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = std_dev
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    int32_mv[cursor], cursor = mean_idx, cursor + 1
    int32_mv[cursor:cursor + loss_rel_size].view(oasis_float)[:] = gul_mean
    cursor += loss_rel_size
    cursor_bytes += gulSampleslevelRec_size

    return cursor, cursor_bytes


def processrec(event_id, damagecdfs, Nbinss, recs, damage_bins, coverages, item_map, sample_size, loss_threshold, debug,
               generate_rndm, int32_mv, cursor, cursor_bytes):
    # TODO: avoid passing generate_rndm function by splitting the random samples in another function
    # TODO: write docstring
    # TODO: port to numba

    Nitems = damagecdfs.shape[0]
    for k in range(Nitems):
        # print(k, cursor_bytes)
        item_key = tuple(damagecdfs[k][['areaperil_id', 'vulnerability_id']])

        for item in item_map[item_key]:
            coverage_id = item['coverage_id']
            tiv = coverages[coverage_id - 1]  # coverages are indexed from 1

            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = output_mean(
                tiv, recs[k]['prob_to'], recs[k]['bin_mean'], Nbinss[k], damage_bins[Nbinss[k] - 1]['bin_to']
            )

            # write header
            cursor, cursor_bytes = write_sample_header(
                event_id, item['item_id'], int32_mv, cursor, cursor_bytes)

            # write default samples
            cursor, cursor_bytes = write_negative_sidx(
                MAX_LOSS_IDX, max_loss,
                CHANCE_OF_LOSS_IDX, chance_of_loss,
                TIV_IDX, tiv,
                STD_DEV_IDX, std_dev,
                MEAN_IDX, gul_mean,
                int32_mv, cursor, cursor_bytes
            )

            # generate seed
            # problem here: with new hashing perhaps this goes out of numba
            seed = generate_hash(item['group_id'], event_id)

            # problem here is that this goes out of numba
            rndms = generate_rndm([seed], sample_size)

            if debug:
                for sample_idx, rval in enumerate(rndms[seed]):
                    cursor, cursor_bytes = write_sample_rec(sample_idx + 1, rval, int32_mv, cursor, cursor_bytes)
            else:
                for sample_idx, rval in enumerate(rndms[seed]):
                    # print(sample_idx, rval)
                    # take the random sample
                    # rval = rndm[sample_idx]
                    # find the bin in which rval falls into
                    # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                    # there's a 1:1 mapping between indices of rec and damage_bins
                    bin_idx = first_index_numba(rval, recs[k]['prob_to'], Nbinss[k])

                    # compute the loss
                    loss = get_gul(
                        damage_bins['bin_from'][bin_idx],
                        damage_bins['bin_to'][bin_idx],
                        recs[k]['bin_mean'][bin_idx],
                        recs[k]['prob_to'][max(bin_idx - 1, 0)],  # for bin_idx=0 take prob_to in bin_idx=0
                        recs[k]['prob_to'][bin_idx],
                        rval,
                        tiv
                    )

                    if loss >= loss_threshold:
                        cursor, cursor_bytes = write_sample_rec(sample_idx + 1, loss, int32_mv, cursor, cursor_bytes)

            # terminate list of samples for this event-item
            cursor, cursor_bytes = write_sample_rec(0, 0., int32_mv, cursor, cursor_bytes)

            # if cursor_bytes > buff_size:
            # TODO: issue with 12k Piwind is that buffsize of 65k is too small (see printout of cursor-bytes)
            # need to find a way to flush (eg yield if cursor_bytes > buff_size)

    # print(cursor_bytes)

    return cursor, cursor_bytes


@nb.njit(cache=True, fastmath=True)
def processrec_mode1(event_id, damagecdfs, item_map, mode1_stats_2, mode1_stats_2_type, mode1_item_id):
    # TODO: write docstring

    mode1UsedCoverageIDs = List.empty_list(nb.types.int32)
    seeds = set()
    Nitems = damagecdfs.shape[0]
    for k in range(Nitems):
        item_key = tuple((damagecdfs[k]['areaperil_id'], damagecdfs[k]['vulnerability_id']))

        for item in item_map[item_key]:
            item_id, coverage_id, group_id = item

            rng_seed = generate_hash(group_id, event_id)

            # append always, will filter out unqiue coverage_ids later
            mode1UsedCoverageIDs.append(coverage_id)

            append_to_dict_entry(mode1_stats_2, coverage_id, (k, rng_seed), mode1_stats_2_type)

            append_to_dict_entry(mode1_item_id, coverage_id, item_id, nb.types.int32)

            seeds.add(rng_seed)

    return mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs, seeds


@nb.njit(cache=True, fastmath=True)
def outputmode1data(event_id, mode1_stats_2, mode1_item_id, mode1UsedCoverageIDs, sample_size,
                    Nbinss, recs, coverages, damage_bins, loss_threshold, alloc_rule, rndms_idx, rndms,
                    debug, int32_mv, cursor, cursor_bytes):

    # convert mode1UsedCoverageIDs to np.array (needed to apply np.unique to it)
    N_cov_ids = len(mode1UsedCoverageIDs)
    mode1UsedCoverageIDs_arr = np.empty(N_cov_ids, dtype=np.int32)
    for j in range(N_cov_ids):
        mode1UsedCoverageIDs_arr[j] = mode1UsedCoverageIDs[j]

    for coverage_id in np.unique(mode1UsedCoverageIDs_arr):

        tiv = coverages[coverage_id - 1]  # coverages are indexed from 1
        Nitems = len(mode1_stats_2[coverage_id])
        exposureValue = tiv / Nitems

        # gulcalc: gilv[item_id, loss] -> gulpy:[loss]]
        loss_Nrows = sample_size + NUM_IDX
        loss = np.zeros((loss_Nrows, Nitems), dtype=oasis_float)

        # TODO: check again if np.argsort() is needed. If not, we could save the conversion from List to np.array
        # convert mode1UsedCoverageIDs to np.array (needed to apply np.unique to it)
        N_item_ids = len(mode1_item_id[coverage_id])
        item_ids_arr = np.empty(N_item_ids, dtype=np.int32)
        for j in range(N_item_ids):
            item_ids_arr[j] = mode1_item_id[coverage_id][j]

        item_ids_arr_argsorted = np.argsort(item_ids_arr)
        item_ids_arr_sorted = item_ids_arr[item_ids_arr_argsorted]
        items_loss_above_threshold = Dict()

        for i, j_stats in enumerate(item_ids_arr_argsorted):

            k, seed = mode1_stats_2[coverage_id][j_stats]

            prob_to = recs[k]['prob_to']
            bin_mean = recs[k]['bin_mean']
            Nbins = Nbinss[k]

            # compute mean values
            gul_mean, std_dev, chance_of_loss, max_loss = output_mean(
                tiv, prob_to, bin_mean, Nbins, damage_bins[Nbins - 1]['bin_to'],
            )

            loss[MAX_LOSS_IDX + NUM_IDX, i] = max_loss
            loss[CHANCE_OF_LOSS_IDX + NUM_IDX, i] = chance_of_loss
            loss[TIV_IDX + NUM_IDX, i] = exposureValue
            loss[STD_DEV_IDX + NUM_IDX, i] = std_dev
            loss[MEAN_IDX + NUM_IDX, i] = gul_mean

            if sample_size > 0:
                if debug:
                    for sample_idx, rval in enumerate(rndms[rndms_idx[seed], :]):
                        loss[sample_idx + NUM_IDX, i] = rval
                else:
                    idx_loss_above_threshold = List.empty_list(nb.types.int64)

                    for sample_idx, rval in enumerate(rndms[rndms_idx[seed], :]):
                        # find the bin in which the random value `rval` falls into
                        # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                        # there's a 1:1 mapping between indices of rec and damage_bins
                        bin_idx = first_index_numba(rval, prob_to, Nbins)

                        # compute the ground up loss
                        gul = get_gul(
                            # I don't understand why this is bin_idx. Doesn't have it to be remapped?
                            damage_bins['bin_from'][bin_idx],
                            # I don't understand why this is bin_idx. Doesn't have it to be remapped?
                            damage_bins['bin_to'][bin_idx],
                            bin_mean[bin_idx],
                            prob_to[max(bin_idx - 1, 0)],  # for bin_idx=0 take prob_to in bin_idx=0
                            prob_to[bin_idx],
                            rval,
                            tiv
                        )

                        if gul >= loss_threshold:
                            loss[sample_idx + NUM_IDX, i] = gul
                            idx_loss_above_threshold.append(sample_idx)

                    items_loss_above_threshold[i] = idx_loss_above_threshold

        cursor, cursor_bytes = writemode1output(loss, item_ids_arr_sorted, alloc_rule, tiv, event_id,
                                                items_loss_above_threshold, int32_mv, cursor, cursor_bytes)

    return cursor, cursor_bytes


@nb.jit(nopython=True, fastmath=True)
def setmaxloss(loss):
    """Set max loss.
    For each sample, find the maximum loss across all items.

    """
    nrows, ncols = loss.shape

    # the main loop starts from STD_DEV
    for i in range(NUM_IDX + STD_DEV_IDX, nrows, 1):
        loss_max = 0.
        max_loss_count = 0

        # find maximum loss and count occurrences
        for j in range(ncols):
            if loss[i, j] > loss_max:
                loss_max = loss[i, j]
                max_loss_count = 1
            elif loss[i, j] == loss_max:
                max_loss_count += 1

        # distribute maximum losses evenly among highest
        # contributing subperils and set other losses to 0
        loss_max_normed = loss_max / max_loss_count
        for j in range(ncols):
            if loss[i, j] == loss_max:
                loss[i, j] = loss_max_normed
            else:
                loss[i, j] = 0.

    return loss


@nb.jit(nopython=True, fastmath=True)
def split_tiv(gulitems, tiv):
    # if the total loss exceeds the tiv
    # then split tiv in the same proportions to the losses
    if tiv > 0:
        total_loss = np.sum(gulitems)

        nitems = gulitems.shape[0]
        if total_loss > tiv:
            for j in range(nitems):
                # editing in-place the np array
                gulitems[j] *= tiv / total_loss


@nb.njit(cache=True, fastmath=True)
def writemode1output(loss, item_ids_arr_sorted, alloc_rule, tiv, event_id, items_loss_above_threshold, int32_mv, cursor, cursor_bytes):
    if alloc_rule == 2:
        loss = setmaxloss(loss)

    # note that nsamples = sample_size + NUM_IDX
    nsamples, nitems = loss.shape

    # Check whether the sum of losses per sample exceed TIV
    # If so, split TIV in proportion to losses
    for i in range(nsamples):
        split_tiv(loss[i], tiv)

    # output the items
    for j in range(nitems):

        # write header
        cursor, cursor_bytes = write_sample_header(
            event_id, item_ids_arr_sorted[j], int32_mv, cursor, cursor_bytes)

        cursor, cursor_bytes = write_negative_sidx(
            MAX_LOSS_IDX, loss[MAX_LOSS_IDX + NUM_IDX, j],
            CHANCE_OF_LOSS_IDX, loss[CHANCE_OF_LOSS_IDX + NUM_IDX, j],
            TIV_IDX, loss[TIV_IDX + NUM_IDX, j],
            STD_DEV_IDX, loss[STD_DEV_IDX + NUM_IDX, j],
            MEAN_IDX, loss[MEAN_IDX + NUM_IDX, j],
            int32_mv, cursor, cursor_bytes
        )

        for sample_idx in items_loss_above_threshold[j]:
            # optimize this by computing the j values for which loss is > treshold and loop only on them with no ifs
            cursor, cursor_bytes = write_sample_rec(
                sample_idx + 1, loss[sample_idx + NUM_IDX, j], int32_mv, cursor, cursor_bytes)

        # terminate list of samples for this event-item
        cursor, cursor_bytes = write_sample_rec(0, 0., int32_mv, cursor, cursor_bytes)

    return cursor, cursor_bytes


@nb.jit(nopython=True, fastmath=True)
def get_arr_chunks(arr, len_arr, N):
    """Get chunks of length `N` from an array `arr` of length `len_arr`.
    The function yields indefinitely, cycling through the array end.
    N must be strictly smaller than len_arr.

    Args:
        arr (_type_): _description_
        len_arr (_type_): _description_
        N (_type_): _description_

    Yields:
        _type_: _description_
    """
    start = -N
    while True:
        start += N
        end = start + N
        start_mod = start % len_arr
        end_mod = end % len_arr

        if start_mod == end_mod:
            yield arr
        elif end_mod < start_mod:
            yield np.concatenate((arr[start_mod:], arr[:end_mod]))
        else:
            yield arr[start_mod:end_mod]


class LossWriter():

    def __init__(self, lossout, len_sample, buff_size=65536) -> None:

        # number of bytes to read at a given time.
        # number_size = 8 works only if loss in gulSampleslevelRec is float32.
        self.number_size = max(gulSampleslevelHeader.size, gulSampleslevelRec.size)  # bytes

        self.len_sample = len_sample
        self.lossout = lossout
        self.buff_size = buff_size  # bytes

        # compute how many numbers of size `number_size` fit in the buffer
        # for safety, take 1000 less than the compute number to allow flushing the buffer not too often
        # if -1 instead of -1000 is taken, it requires checking whether to flush or not for every write to mv.
        self.buff_safety = self.number_size * 1000
        self.nb_number = (self.buff_size + self.buff_safety) // self.number_size
        self.flush_number = self.nb_number - 4

        # define the raw memory view, the int32 view of it, and their respective cursors
        self.mv = memoryview(bytearray(buff_size * 10))
        self.int32_mv = np.ndarray(buff_size * 10 // self.number_size, buffer=self.mv, dtype='i4')
        # cannot use because the header is int int
        # self.loss_mv = np.ndarray(self.nb_number, buffer=self.mv, dtype=gulSampleslevelRec.dtype)
        # cannot use two views loss_mv and header_mv because it only works if oasis_float is float32.
        # if oasis_float is set to float64, the cursor will not map correctly both mv.
        self.cursor_bytes = 0
        self.cursor = 0

        # size(oasis_float)/size(i4)
        # TODO find a way to do that programmatically and test if this works with oasis_float=float64
        self.loss_rel_size = 1

    def flush(self):
        # print("FLUSHING ", self.cursor, " ", self.cursor_bytes)
        self.lossout.write(self.mv[:self.cursor_bytes])
        self.cursor_bytes = 0
        self.cursor = 0
        # print("FLUSHED ", self.cursor, " ", self.cursor_bytes)

    def write_sample_header(self, event_id, item_id):
        self.int32_mv[self.cursor] = event_id
        self.cursor += 1
        self.int32_mv[self.cursor] = item_id
        self.cursor += 1
        self.cursor_bytes += gulSampleslevelHeader.size

    def write_sample_rec(self, sidx, loss):

        if self.cursor >= self.flush_number:
            self.flush()

        self.int32_mv[self.cursor] = sidx
        self.cursor += 1
        self.int32_mv[self.cursor:self.cursor + self.loss_rel_size].view(oasis_float)[:] = loss
        self.cursor += self.loss_rel_size
        self.cursor_bytes += gulSampleslevelRec.size


@nb.jit(cache=True, nopython=True, fastmath=True)
def generate_random_numbers(sample_size, seed=None):

    rndm = np.random.uniform(0., 1., size=sample_size)

    return rndm


@nb.jit(cache=True, nopython=True, fastmath=True)
def first_index_numba(val, arr, len_arr):
    """
    Find the first element of `arr` larger than `val`, assuming `arr` is sorted. 

    interesting answer on different methods and their performance
    especially, using numba https://stackoverflow.com/a/49927020/3709114

    """
    for idx in range(len_arr):
        if arr[idx] > val:
            return idx
    return -1


# @nb.njit(cache=False, fastmath=True)
def output_mean_mode1(tiv, prob_to, bin_mean, bin_count, max_damage_bin_to, bin_map, bin_lookup):
    """Compute output mean gul.

    Note that this implementation is approx 20x faster than pure numpy/scipy functions,
    and 2x faster than numpy/scipy functions wrapped in numba.jit.

    Args:
        tiv (_type_): _description_
        prob_to (_type_): _description_
        bin_mean (_type_): _description_
        bin_count (_type_): _description_
        max_damage_bin_to (_type_): _description_

    Returns:
        _type_: _description_
    """
    # chance_of_loss = 1. - prob_to[0] if bin_mean[0] == 0. else 1.
    chance_of_loss = 1 - prob_to[0] * (1 - (bin_mean[0] > 0))

    gul_mean = 0.
    ctr_var = 0.
    last_prob_to = 0.
    bin_ids = List()
    for i in range(bin_count):
        prob_from = last_prob_to
        new_gul = (prob_to[i] - prob_from) * bin_mean[i]
        gul_mean += new_gul
        ctr_var += new_gul * bin_mean[i]
        last_prob_to = prob_to[i]

        bin_key = tuple((prob_to[i], bin_mean[i]))
        bin_lst = List([prob_to[i], bin_mean[i]])
        if bin_key not in bin_map.keys():
            bin_map[bin_key] = len(bin_map)
            bin_lookup.append(bin_lst)

        bin_ids.append(bin_map[bin_key])

    gul_mean *= tiv
    ctr_var *= tiv**2.
    std_dev = sqrt(max(ctr_var - gul_mean**2., 0.))
    max_loss = max_damage_bin_to * tiv

    return gul_mean, std_dev, chance_of_loss, max_loss, bin_map, bin_ids, bin_lookup


@nb.jit(cache=False, nopython=True, fastmath=False)
def get_gul(bin_from, bin_to, bin_mean, prob_from, prob_to, rval, tiv):
    # TODO: write docstring
    # the interpolation engine for each bin can be cached since the decision on whether to use
    # point-like/linear/quadratic only depends on bin properties, not on rval.
    # however, if samples are few and do not use all the bins it might not be advantageous

    bin_width = bin_to - bin_from

    # point-like bin
    if bin_width == 0.:
        gul = tiv * bin_to

        return gul

    bin_height = prob_to - prob_from
    rval_bin_offset = rval - prob_from

    # linear interpolation
    x = np.float64((bin_mean - bin_from) / bin_width)
    # if np.int(np.round(x * 100000)) == 50000:
    if np.abs(x - 0.5) <= 5e-6:
        # this condition requires 1 less operation
        gul = tiv * (bin_from + rval_bin_offset * bin_width / bin_height)

        return gul

    # quadratic interpolation
    # MT: I haven't re-derived the algorithm for this case; not sure where the parabola vertex is set
    aa = 3. * bin_height / bin_width**2 * (2. * x - 1.)
    bb = 2. * bin_height / bin_width * (2. - 3. * x)
    cc = - rval_bin_offset

    gul = tiv * (bin_from + (sqrt(bb**2. - 4. * aa * cc) - bb) / (2. * aa))

    return gul


@nb.jit(cache=True, nopython=True, fastmath=True)
def output_mean(tiv, prob_to, bin_mean, bin_count, max_damage_bin_to):
    """Compute output mean gul.

    Note that this implementation is approx 20x faster than pure numpy/scipy functions,
    and 2x faster than numpy/scipy functions wrapped in numba.jit.

    Args:
        tiv (_type_): _description_
        prob_to (_type_): _description_
        bin_mean (_type_): _description_
        bin_count (_type_): _description_
        max_damage_bin_to (_type_): _description_

    Returns:
        _type_: _description_
    """
    # chance_of_loss = 1. - prob_to[0] if bin_mean[0] == 0. else 1.
    chance_of_loss = 1 - prob_to[0] * (1 - (bin_mean[0] > 0))

    gul_mean = 0.
    ctr_var = 0.
    last_prob_to = 0.
    for i in range(bin_count):
        prob_from = last_prob_to
        new_gul = (prob_to[i] - prob_from) * bin_mean[i]
        gul_mean += new_gul
        ctr_var += new_gul * bin_mean[i]
        last_prob_to = prob_to[i]

    gul_mean *= tiv
    ctr_var *= tiv**2.
    std_dev = sqrt(max(ctr_var - gul_mean**2., 0.))
    max_loss = max_damage_bin_to * tiv

    return gul_mean, std_dev, chance_of_loss, max_loss
