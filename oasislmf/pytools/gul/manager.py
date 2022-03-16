"""
This file is the entry point for the gul command for the package.

"""

from ctypes import sizeof
from importlib.metadata import files
from readline import append_history_file
import time
import sys
import os
import logging
from contextlib import ExitStack
import numpy as np
from numpy.random import Generator, MT19937

import numba as nb
from scipy.stats import qmc

from math import sqrt, ceil, log2  # faster than numpy.sqrt

from oasislmf.pytools.getmodel.manager import get_mean_damage_bins, get_damage_bins, get_items, Item
from oasislmf.pytools.getmodel.common import oasis_float
from oasislmf.pytools.gul.common import gulSampleslevelHeader, gulSampleslevelRec, gulSampleFullRecord
from oasislmf.pytools.gul.common import processrecData, gulItemIDLoss

from oasislmf.pytools.gul.common import ProbMean, damagecdfrec, Item_map_rec

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


def read_getmodel_stream(run_dir, streams_in):
    """Read the getmoudel output stream.

    Args:
        run_dir ([str]): Path to the run directory.

    Raises:
        ValueError: If the stream type is not 1.

    """
    # TODO: read in chunks of buffsize and keep yielding record by record

    # get damage bins from file
    static_path = os.path.join(run_dir, 'static')
    damage_bins = get_damage_bins(static_path=static_path)

    # maximum number of damage bins (individual items can have up to `total_bins` bins)
    if damage_bins.shape[0] == 0:
        total_bins = 1000
    else:
        total_bins = damage_bins.shape[0]

    # determine stream type
    stream_type = np.frombuffer(streams_in.read(4), dtype='i4')

    # FIXME: make sure the bit1 and bit 2-4 compliance is checked
    # see https://github.com/OasisLMF/ktools/blob/master/docs/md/CoreComponents.md
    if stream_type[0] != 1:
        raise ValueError(f"FATAL: Invalid stream type: expect 1, got {stream_type[0]}.")

    # prepare all the data buffers
    damagecdf_mv = memoryview(bytearray(damagecdfrec.size))
    damagecdf = np.ndarray(1, buffer=damagecdf_mv, dtype=damagecdfrec.dtype)
    Nbins_mv = memoryview(bytearray(4))
    Nbins = np.ndarray(1, buffer=Nbins_mv, dtype='i4')
    rec_mv = memoryview(bytearray(total_bins * ProbMean.size))
    rec = np.ndarray(total_bins, buffer=rec_mv, dtype=ProbMean.dtype)

    # start reading the stream
    # each record from getmodel is expected to contain:
    # 1 damagecdfrec obj, 1 int (Nbins), a number `Nbins` of ProbMean objects
    while True:
        len_read = streams_in.readinto(damagecdf_mv)
        len_read = streams_in.readinto(Nbins_mv)
        len_read = streams_in.readinto(rec_mv[:Nbins[0] * ProbMean.size])

        # exit if the stream has ended
        if len_read == 0:
            break

        yield damagecdf, Nbins, rec

    return 0


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


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
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


def generate_rndm_MT19937(seeds, n):
    rndm = {}
    for seed in seeds:
        rng = Generator(MT19937(seed=123))
        rndm[seed] = rng.uniform(0., 1., size=n)

    return rndm


def generate_rndm_Sobol(seeds, n):

    # draw a power of 2 number of samples to keep 'Sobol' sequences
    # balance properties (see Sobol docs for more details), then take
    # only the `n` required numbers.
    n_closest_base2 = max(n, 2**ceil(log2(n)))

    rndm = {}
    for seed in seeds:
        rng = qmc.Sobol(d=1, scramble=True, seed=seed)
        rndm[seed] = rng.random(n_closest_base2)[:n].ravel()

    return rndm


def generate_rndm_LHS(seeds, n):
    rndm = {}
    for seed in seeds:
        rng = qmc.LatinHypercube(d=1, seed=seed)
        rndm[seed] = rng.random(n).ravel()

    return rndm


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


def generate_item_map(items):

    # in-place sort items in order to store them in item_map in the desired order
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])

    item_map = {}
    for item in items:
        item_map.setdefault(tuple(item[['areaperil_id', 'vulnerability_id']]), []).append(
            np.array((item['id'], item['coverage_id'], item['group_id']), dtype=Item_map_rec)
        )

    return item_map


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
    item_map = generate_item_map(items)

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

        writer = LossWriter(stream_out, sample_size)

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

            elif random_generator == 2:
                generate_rndm = generate_rndm_Sobol
                logger.info("Random generator: Sobol sequences")

        # min condition is needed to avoid seg fault if sample_size=0
        # rndms = get_arr_chunks(rndm, len_rndm, max(1, sample_size))

        # TODO: probably here I need a with Losswriter context
        # get the stream, for each entry in the stream:

        assert alloc_rule in [0, 1, 2], f"Expect alloc_rule to be 0 or 1 or 2, got {alloc_rule}"

        if alloc_rule == 0:
            for damagecdf, Nbins, rec in read_getmodel_stream(run_dir, streams_in):
                processrec(damagecdf[0], Nbins[0], rec, damage_bins, coverages,
                           item_map, writer, sample_size, loss_threshold,
                           debug, generate_rndm)

            # flush all the remaining data
            writer.flush(force=True)

        else:
            last_event_id = -1
            mode1_stats = {}
            mode1UsedCoverageIDs = set()
            bin_map = {}
            bin_lookup = []
            mode1_stats_item_id = {}
            seeds = set()
            all_seeds = set()
            for damagecdf, Nbins, rec in read_getmodel_stream(run_dir, streams_in):
                # TODO issue re speeding up read_getmodel_stream:
                # if read_getmodel_stream yields chunks, rather than each cdf record,
                # i need to unroll it anyway to process each record.
                # so perhaps read_getmodel_stream should yield each record like now,
                # but inside it the reading from the stream should be done in large chunks
                # rather then number by number like now -> this will speed up

                event_id = damagecdf[0]['event_id']

                # perhaps it'd be faster if read_getmodel_stream yields event by event
                # so we can avoid these ifs inside here, and we always output and clearmode1_data

                # seeds = generate_seeds()
                if event_id != last_event_id:

                    if last_event_id > 0:
                        # this is not the very first event to be processed
                        bin_lookup_ndarr = np.empty((len(bin_lookup),), dtype=ProbMean)
                        bin_lookup_arr = np.array(bin_lookup)

                        bin_lookup_ndarr[:]['prob_to'] = bin_lookup_arr[:, 0]
                        bin_lookup_ndarr[:]['bin_mean'] = bin_lookup_arr[:, 1]

                        rndms = generate_rndm(seeds, sample_size)
                        logger.debug(f"len_sets={len(seeds)}")
                        all_seeds.update(seeds)

                        outputmode1data(last_event_id, mode1_stats, mode1UsedCoverageIDs, sample_size,
                                        coverages, bin_lookup_ndarr, damage_bins, loss_threshold, writer,
                                        alloc_rule, rndms, mode1_stats_item_id, debug)

                        # clearmode1_data
                        mode1_stats = {}
                        mode1UsedCoverageIDs = set()
                        mode1_stats_item_id = {}
                        seeds = set()
                        # probably I can add a nitems counter to avoid running the sum() above before outputmode1data

                    last_event_id = event_id

                mode1_stats, mode1UsedCoverageIDs, bin_map, bin_lookup, mode1_stats_item_id, seeds = processrec_mode1(damagecdf[0], Nbins[0], rec, damage_bins, coverages, item_map, writer, sample_size, loss_threshold,
                                                                                                                      mode1_stats, mode1UsedCoverageIDs, bin_map, bin_lookup, mode1_stats_item_id, seeds)

            # write out the the last event
            bin_lookup_ndarr = np.empty((len(bin_lookup),), dtype=ProbMean)
            bin_lookup_arr = np.array(bin_lookup)

            bin_lookup_ndarr[:]['prob_to'] = bin_lookup_arr[:, 0]
            bin_lookup_ndarr[:]['bin_mean'] = bin_lookup_arr[:, 1]

            rndms = generate_rndm(seeds, sample_size)

            all_seeds.update(seeds)

            outputmode1data(event_id, mode1_stats, mode1UsedCoverageIDs, sample_size,
                            coverages, bin_lookup_ndarr, damage_bins, loss_threshold, writer,
                            alloc_rule, rndms, mode1_stats_item_id, debug)

            # flush all the remaining data
            writer.flush(force=True)

            logger.debug(f"len_sets={len(seeds)}")
            logger.debug(f"all_seeds: {len(all_seeds)}")

    return 0


def outputmode1data(event_id, mode1_stats, mode1UsedCoverageIDs, sample_size, coverages, bin_lookup_ndarr,
                    damage_bins, loss_threshold, loss_writer, alloc_rule, rndms, mode1_stats_item_id, debug):

    for coverage_id in mode1UsedCoverageIDs:

        # [check before deleting] this if is not needed (mode1UsedCoverageIDs will only contain the mode1_stats keys)
        # if mode1_stats[coverage_id]:

        tiv = coverages[coverage_id - 1]  # coverages are indexed from 1
        exposureValue = tiv / len(mode1_stats[coverage_id])

        gilv = np.zeros((sample_size + NUM_IDX, len(mode1_stats[coverage_id])), dtype=gulItemIDLoss)

        # probably this for loop can go inside the if, since if mode1_stats[coverage_id] is None, enumerate will raise error
        # for i, item in enumerate(mode1_stats[coverage_id]):
        for i, j_stats in enumerate(np.argsort(mode1_stats_item_id[coverage_id])):
            item = mode1_stats[coverage_id][j_stats]

            gilv[:, i]['item_id'] = item['recData']['item_id']
            gilv[MAX_LOSS_IDX + NUM_IDX, i]['loss'] = item['recData']['max_loss']
            gilv[CHANCE_OF_LOSS_IDX + NUM_IDX, i]['loss'] = item['recData']['chance_of_loss']
            gilv[TIV_IDX + NUM_IDX, i]['loss'] = exposureValue
            gilv[STD_DEV_IDX + NUM_IDX, i]['loss'] = item['recData']['std_dev']
            gilv[MEAN_IDX + NUM_IDX, i]['loss'] = item['recData']['gul_mean']

            prob_to = bin_lookup_ndarr[item['bin_ids']]['prob_to']
            bin_mean = bin_lookup_ndarr[item['bin_ids']]['bin_mean']
            Nbins_remapped = len(item['bin_ids'])

            if sample_size > 0:
                if debug:
                    for sample_idx, rval in enumerate(rndms[item['seed']]):
                        gilv[sample_idx + NUM_IDX, i]['loss'] = rval
                else:
                    for sample_idx, rval in enumerate(rndms[item['seed']]):
                        # take the random sample
                        # rval = rndm[sample_idx]
                        # find the bin in which rval falls into
                        # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                        # there's a 1:1 mapping between indices of rec and damage_bins
                        bin_idx = first_index_numba(rval, prob_to, Nbins_remapped)

                        # compute the loss
                        loss = get_gul(
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

                        # here store all losses (filter later based on loss_threshold)
                        gilv[sample_idx + NUM_IDX, i]['loss'] = loss

        writemode1output(gilv, alloc_rule, tiv, loss_writer, event_id, loss_threshold)

        # # terminate list of samples for this event-item
        # loss_writer.write_sample_rec(0, 0.)


@nb.jit(nopython=True, fastmath=True)
def setmaxloss(gilv):
    """Set max loss.
    For each sample, find the maximum loss across all items.

    """
    nrows, ncols = gilv.shape

    # the main loop starts from STD_DEV
    for i in range(NUM_IDX + STD_DEV_IDX, nrows, 1):
        gilv_max = 0.
        max_loss_count = 0

        # find maximum loss and count occurrences
        for j in range(ncols):
            if gilv[i, j]['loss'] > gilv_max:
                gilv_max = gilv[i, j]['loss']
                max_loss_count = 1
            elif gilv[i, j]['loss'] == gilv_max:
                max_loss_count += 1

        # distribute maximum losses evenly among highest
        # contributing subperils and set other losses to 0
        gilv_max_normed = gilv_max / max_loss_count
        for j in range(ncols):
            if gilv[i, j]['loss'] == gilv_max:
                gilv[i, j]['loss'] = gilv_max_normed
            else:
                gilv[i, j]['loss'] = 0.

    return gilv


@nb.jit(nopython=True, fastmath=True)
def split_tiv(gulitems, tiv):
    # if the total loss exceeds the tiv
    # then split tiv in the same proportions to the losses
    if tiv > 0:
        total_loss = np.sum(gulitems['loss'])

        nitems = gulitems.shape[0]
        if total_loss > tiv:
            for j in range(nitems):
                # editing in-place the np array
                gulitems[j]['loss'] *= tiv / total_loss


def writemode1output(gilv, alloc_rule, tiv, loss_writer, event_id, loss_threshold):
    if alloc_rule == 2:
        gilv = setmaxloss(gilv)

    # note that nsamples=sample_size + NUM_IDX
    nsamples, nitems = gilv.shape

    # Check whether the sum of losses per sample exceed TIV
    # If so, split TIV in proportion to losses
    for i in range(nsamples):
        split_tiv(gilv[i], tiv)

    # output the items
    for j in range(nitems):
        loss_writer.write_sample_header(event_id, gilv[0, j]['item_id'])
        for i in range(NUM_IDX):
            loss_writer.write_sample_rec(i - NUM_IDX, gilv[i, j]['loss'])

        for i in range(NUM_IDX, nsamples, 1):
            # optimize this by computing the j values for which loss is > treshold and loop only on them with no ifs
            if gilv[i, j]['loss'] > loss_threshold:
                loss_writer.write_sample_rec(i - NUM_IDX + 1, gilv[i, j]['loss'])

        # terminate list of samples for this event-item
        loss_writer.write_sample_rec(0, 0.)

    return


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
        self.mv = memoryview(bytearray(self.nb_number * self.number_size))
        self.int32_mv = np.ndarray(self.nb_number, buffer=self.mv, dtype='i4')
        # cannot use because the header is int int
        # self.loss_mv = np.ndarray(self.nb_number, buffer=self.mv, dtype=gulSampleslevelRec.dtype)
        # cannot use two views loss_mv and header_mv because it only works if oasis_float is float32.
        # if oasis_float is set to float64, the cursor will not map correctly both mv.
        self.cursor_bytes = 0
        self.cursor = 0

        # size(oasis_float)/size(i4)
        # TODO find a way to do that programmatically and test if this works with oasis_float=float64
        self.loss_rel_size = 1

    def flush(self, force=False):
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


# @nb.jit(cache=False, nopython=True, fastmath=True)
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
    bin_ids = []
    for i in range(bin_count):
        prob_from = last_prob_to
        new_gul = (prob_to[i] - prob_from) * bin_mean[i]
        gul_mean += new_gul
        ctr_var += new_gul * bin_mean[i]
        last_prob_to = prob_to[i]

        bin_key = tuple((prob_to[i], bin_mean[i]))
        if bin_key not in bin_map.keys():
            bin_map[bin_key] = len(bin_map)
            bin_lookup.append(list(bin_key))

        bin_ids.append(bin_map[bin_key])

    gul_mean *= tiv
    ctr_var *= tiv**2.
    std_dev = sqrt(max(ctr_var - gul_mean**2., 0.))
    max_loss = max_damage_bin_to * tiv

    return gul_mean, std_dev, chance_of_loss, max_loss, bin_map, bin_ids, bin_lookup


def processrec_mode1(damagecdf, Nbins, rec, damage_bins, coverages, item_map, loss_writer, sample_size, loss_threshold,
                     mode1_stats, mode1UsedCoverageIDs, bin_map, bin_lookup, mode1_stats_item_id, seeds):
    # TODO: write docstring
    # TODO: port to numba
    item_key = tuple(damagecdf[['areaperil_id', 'vulnerability_id']])
    # coverage_id = int(item_map[item_key]['coverage_id'])
    # tiv = coverages[coverage_id - 1]  # coverages are indexed from 1

    # TODO need a for loop on all coverages, and need to change the item map to append_history_file
    for item in item_map[item_key]:
        coverage_id = int(item['coverage_id'])
        tiv = coverages[coverage_id - 1]  # coverages are indexed from 1

        # compute mean values
        gul_mean, std_dev, chance_of_loss, max_loss, bin_map, bin_ids, bin_lookup = output_mean_mode1(
            tiv, rec['prob_to'], rec['bin_mean'], Nbins, damage_bins[Nbins - 1]['bin_to'],
            bin_map, bin_lookup
        )

        recData = np.ndarray(1, dtype=processrecData)
        recData[:] = (gul_mean, std_dev, chance_of_loss, max_loss, item['group_id'], item['item_id'])

        rng_seed = generate_hash(item['group_id'], damagecdf['event_id'])

        logger.debug(f"{damagecdf['event_id']}, {item['group_id']}, {coverage_id}, {item['item_id']}, {rng_seed}")

        mode1_stats.setdefault(coverage_id, []).append(
            {'recData': recData, 'bin_ids': bin_ids, 'seed': rng_seed})
        mode1_stats_item_id.setdefault(coverage_id, []).append(item['item_id'])
        mode1UsedCoverageIDs.add(coverage_id)
        seeds.add(rng_seed)

    return mode1_stats, mode1UsedCoverageIDs, bin_map, bin_lookup, mode1_stats_item_id, seeds


def processrec(damagecdf, Nbins, rec, damage_bins, coverages, item_map, loss_writer, sample_size, loss_threshold, debug,
               generate_rndm):
    # TODO: avoid passing generate_rndm function by splitting the random samples in another function
    # TODO: write docstring
    # TODO: port to numba
    item_key = tuple(damagecdf[['areaperil_id', 'vulnerability_id']])

    for item in item_map[item_key]:
        coverage_id = item['coverage_id']
        tiv = coverages[coverage_id - 1]  # coverages are indexed from 1

        # compute mean values
        gul_mean, std_dev, chance_of_loss, max_loss = output_mean(
            tiv, rec['prob_to'], rec['bin_mean'], Nbins, damage_bins[Nbins - 1]['bin_to']
        )

        # write header
        loss_writer.write_sample_header(damagecdf['event_id'], item['item_id'])

        # write default samples
        # TODO create a function that takes all of them and writes all of them
        loss_writer.write_sample_rec(MAX_LOSS_IDX, max_loss)
        loss_writer.write_sample_rec(CHANCE_OF_LOSS_IDX, chance_of_loss)
        loss_writer.write_sample_rec(TIV_IDX, tiv)
        loss_writer.write_sample_rec(STD_DEV_IDX, std_dev)
        loss_writer.write_sample_rec(MEAN_IDX, gul_mean)

        # generate seed
        seed = set({generate_hash(item['group_id'], damagecdf['event_id'])})

        rndms = generate_rndm(seed, sample_size)

        if debug:
            for sample_idx, rval in enumerate(rndms):
                loss_writer.write_sample_rec(sample_idx + 1, rval)
        else:
            for sample_idx, rval in enumerate(rndms):
                # take the random sample
                # rval = rndm[sample_idx]
                # find the bin in which rval falls into
                # note that rec['bin_mean'] == damage_bins['interpolation'], therefore
                # there's a 1:1 mapping between indices of rec and damage_bins
                bin_idx = first_index_numba(rval, rec['prob_to'], Nbins)

                # compute the loss
                loss = get_gul(
                    damage_bins['bin_from'][bin_idx],
                    damage_bins['bin_to'][bin_idx],
                    rec['bin_mean'][bin_idx],
                    rec['prob_to'][max(bin_idx - 1, 0)],  # for bin_idx=0 take prob_to in bin_idx=0
                    rec['prob_to'][bin_idx],
                    rval,
                    tiv
                )

                if loss >= loss_threshold:
                    loss_writer.write_sample_rec(sample_idx + 1, loss)

        # terminate list of samples for this event-item
        loss_writer.write_sample_rec(0, 0.)

    return 0


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
