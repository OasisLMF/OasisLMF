"""
This file is the entry point for the gul command for the package.

"""

import time
import sys
import os
import logging
from contextlib import ExitStack
import numpy as np
import numba as nb

from oasislmf.pytools.getmodel.manager import get_mean_damage_bins, get_damage_bins, get_items
from oasislmf.pytools.getmodel.common import oasis_float

from oasislmf.pytools.gul.common import ProbMean, damagecdfrec


logger = logging.getLogger(__name__)


def read_stream(run_dir, skip_header):
    """Run cdftocsv command: convert the cdf output from getmodel into csv format.
    The binary data is read from an input stream, and the csv file is streamed to stdout.

    Args:
        run_dir ([str]): Path to the run directory.

        skip_header ([bool]): If True, does not print the csv header.

    Raises:
        ValueError: If the stream type is not 1.

    """
    # set up the streams
    streams_in = sys.stdin.buffer
    stream_out = sys.stdout

    # get damage bins from file
    static_path = os.path.join(run_dir, 'static')
    damage_bins = get_mean_damage_bins(static_path=static_path)

    # maximum number of damage bins (individual items can have up to `total_bins` bins)
    if damage_bins.shape[0] == 0:
        total_bins = 1000
    else:
        total_bins = damage_bins.shape[0]

    # determine stream type
    stream_type = np.frombuffer(streams_in.read(4), dtype='i4')

    if stream_type[0] != 1:
        raise ValueError(f"FATAL: Invalid stream type: expect 1, got {stream_type[0]}.")

    if not skip_header:
        stream_out.write("event_id,areaperil_id,vulnerability_id,bin_index,prob_to,bin_mean\n")

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

        # print(damagecdf, Nbins, rec)
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


GROUP_ID_HASH_CODE = np.int64(1543270363)
EVENT_ID_HASH_CODE = np.int64(1943272559)

HASH_MOD_CODE = np.int64(2147483648)


def generate_hash(group_id, event_id, rand_seed=0, correlated=False):
    """
    Generate hash for group_id, event_id

    Args:
        group_id ([type]): [description]
        event_id ([type]): [description]
        rand_seed (int, optional): [description]. Defaults to 0.
        correlated (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    hashed = rand_seed
    hashed += np.mod(group_id * GROUP_ID_HASH_CODE, HASH_MOD_CODE)

    if correlated:
        return hashed

    hashed += np.mod(event_id, * EVENT_ID_HASH_CODE, HASH_MOD_CODE)

    return hashed


def print_cdftocsv(damagecdf, Nbins, rec):
    # print out in csv format

    csv_line_fixed = "{}".format(damagecdf['event_id'][0]) + ","
    csv_line_fixed += "{}".format(damagecdf['areaperil_id'][0]) + ","
    csv_line_fixed += "{}".format(damagecdf['vulnerability_id'][0]) + ","

    for i in range(Nbins[0]):
        csv_line = csv_line_fixed
        csv_line += "{}".format(i + 1) + ","   # bin index starts from 1
        csv_line += "{:8.6f},{:8.6f}".format(rec[i]["prob_to"], rec[i]["bin_mean"])

        yield csv_line


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

    from oasislmf.pytools.getmodel.manager import Item

    input_files = set(os.listdir(input_path))
    if "items.bin" in input_files and "bin" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'items.csv')}")
        items = np.memmap(os.path.join(input_path, "items.bin"), dtype=Item, mode='r')
    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        logger.debug(f"loading {os.path.join(input_path, 'items.csv')}")
        items = np.genfromtxt(os.path.join(input_path, "items.csv"), dtype=Item, delimiter=",")
    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


# unused
def generate_item_map(items):

    item_map = {}
    for item in items:
        item_map[item[['areaperil_id', 'vulnerability_id']]] = np.array(
            (item['id'], item['coverage_id'], item['group_id']), dtype=Item_map
        )

    return item_map


def run(run_dir, ignore_file_type, sample_size, file_in=None, file_out=None, **kwargs):
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
    # print("damage_bins:")
    # print(damage_bins.dtype)
    # print(damage_bins)

    input_path = 'input/'
    # TODO: store input_path in a paraparameters file
    items = gul_get_items(input_path)
    areaperil_id_vuln_id_to_coverage_id = {}
    for item in items:
        areaperil_id_vuln_id_to_coverage_id[tuple(item[['areaperil_id', 'vulnerability_id']])] = item['coverage_id']

    # TODO NOW: understand how to get tiv from the iterator provably needs the items dict
    # check with Stephane if the architecture is ok or not

    coverages = get_coverages(input_path)
    Ncoverages = coverages

    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None:
            stream_out = sys.stdout
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        # event_id_mv = memoryview(bytearray(4))
        # event_ids = np.ndarray(1, buffer=event_id_mv, dtype='i4')

        # logger.debug('gulcalc starting')
        # while True:
        #     len_read = streams_in.readinto(event_id_mv)
        #     if len_read == 0:
        #         break

        # get random numbers
        # getRands rnd(opt.rndopt, opt.rand_vector_size, opt.rand_seed);
        # getRands rnd0(opt.rndopt, opt.rand_vector_size, opt.rand_seed);
        Nrands = sample_size * Ncoverages

        seed = 123456  # substitute with desired value
        # rands = generate_rands(N=100, method='uniform', d=1, rng=None, seed=seed)
        # run gulcalc

        # get the stream, for each entry in the stream:
        for damagecdf, Nbins, rec in read_stream(run_dir, kwargs.get('skip_header')):
            # uncomment below as a debug
            # it should print out the cdf
            # for line in print_cdftocsv(damagecdf, Nbins, rec):
            #     stream_out.write(line + "\n")
            processrec(damagecdf[0], Nbins[0], rec, damage_bins, coverages, areaperil_id_vuln_id_to_coverage_id)

        logger.info("gulpy is finished")

    return 0


def itemoutputgul(*args):
    pass


def processrec(damagecdf, Nbins, rec, damage_bins, coverages, areaperil_id_vuln_id_to_coverage_id):

    coverage_id_key = tuple(damagecdf[['areaperil_id', 'vulnerability_id']])
    coverage_id = areaperil_id_vuln_id_to_coverage_id[coverage_id_key]
    tiv = coverages[coverage_id - 1]  # coverages are indexed from 1

    # t0 = time.time()
    # oum = output_mean_py(tiv, rec, damage_bins)
    # t1 = time.time()
    # print("time output_mean_py: ", t1 - t0)

    # t0_nb = time.time()
    # oum_nb = output_mean_np_nb(tiv, rec['prob_to'], rec['bin_mean'], damage_bins[-1]['bin_to'])
    # t1_nb = time.time()
    # print("time output_mean_np_nb: ", t1_nb - t0_nb, "speedup:", (t1 - t0) / (t1_nb - t0_nb))

    # t0_nb2 = time.time()
    oum_nb2 = output_mean(tiv, rec['prob_to'], rec['bin_mean'], Nbins, damage_bins[-1]['bin_to'])
    # t1_nb2 = time.time()
    # print("time output_mean: ", t1_nb2 - t0_nb2, "speedup:", (t1 - t0) / (t1_nb2 - t0_nb2))

    # print(oum)
    # print(oum_nb)
    # print(coverage_id, tiv, oum_nb2)  # debug
    # np.testing.assert_allclose(oum, oum_nb)  # raise error if results are different
    # np.testing.assert_allclose(oum, oum_nb2)  # raise error if results are different

    # itemoutputgul(sidx=max_loss_idx, loss=max_loss)


# def output_mean_py(tiv, pp, damage_bins):

#     if pp[0]['bin_mean'] == 0.:
#         chance_of_loss = 1. - pp[0]['prob_to']
#     else:
#         chance_of_loss = 1.

#     diffs = np.diff(np.concatenate((np.zeros(1), pp['prob_to']))) * pp['bin_mean']
#     gul_mean = tiv * np.sum(diffs)
#     ctr_var = tiv**2. * np.sum(diffs * pp['bin_mean'])

#     std_dev = np.sqrt(max(ctr_var - gul_mean ** 2., 0.))

#     max_loss = tiv * damage_bins[-1]['bin_to']

#     return gul_mean, std_dev, chance_of_loss, max_loss


# @nb.jit(cache=True, nopython=True, fastmath=True)
# def output_mean_np_nb(tiv, prob_to, bin_mean, max_damage_bin_to):

#     if bin_mean[0] == 0.:
#         chance_of_loss = 1. - prob_to[0]
#     else:
#         chance_of_loss = 1.

#     diffs = np.multiply(np.diff(np.concatenate((np.zeros(1), prob_to))), bin_mean)
#     gul_mean = tiv * np.sum(diffs)
#     ctr_var = tiv**2. * np.sum(diffs * bin_mean)

#     std_dev = np.sqrt(max(ctr_var - gul_mean ** 2., 0.))

#     max_loss = tiv * max_damage_bin_to

#     return gul_mean, std_dev, chance_of_loss, max_loss


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
    std_dev = np.sqrt(max(ctr_var - gul_mean**2., 0.))
    max_loss = max_damage_bin_to * tiv

    return gul_mean, std_dev, chance_of_loss, max_loss
