"""
This file is the entry point for the gul command for the package.

"""
import logging
import os
import sys
from contextlib import ExitStack
from select import select

import numpy as np
from numba import njit
import time
from oasislmf.utils.ping import oasis_ping

from oasislmf.pytools.common.data import correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import (PIPE_CAPACITY, mv_write_item_header, mv_write_sidx_loss, mv_write_delimiter,
                                                  stream_info_to_bytes, LOSS_STREAM_ID, ITEM_STREAM)
from oasislmf.pytools.getmodel.common import oasis_float
from oasislmf.pytools.common.data import areaperil_int, oasis_int
from oasislmf.pytools.common.hashmap import (
    init_dict, unpack as hm_unpack, rehash as hm_rehash,
    _try_add_key as hm_try_add_key, i_add_key_fail as hm_i_add_key_fail,
)
from oasislmf.pytools.gul.common import (SPECIAL_SIDX, CHANCE_OF_LOSS_IDX,
                                         MAX_LOSS_IDX,
                                         MEAN_IDX, NUM_IDX, STD_DEV_IDX,
                                         TIV_IDX,
                                         gulSampleslevelHeader_size,
                                         gulSampleslevelRec_size)
from oasislmf.pytools.gul.core import (compute_mean_loss, get_gul, setmaxloss,
                                       split_tiv_classic,
                                       split_tiv_multiplicative)
from oasislmf.pytools.gul.io import read_getmodel_stream
from oasislmf.pytools.gul.random import (generate_correlated_hash_vector,
                                         get_corr_rval, get_random_generator)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.defaults import SERVER_UPDATE_TIME

logger = logging.getLogger(__name__)


@njit(cache=True)
def adjust_byte_mv_size(byte_mv, max_bytes_per_coverage):
    """
    adjust buff size so that the buffer fits the longest coverage
    Args:
        byte_mv: numpy byte array
        max_bytes_per_coverage: max size possible to accommodate all the coverage in byte_mv

    Returns:
        byte_mv: numpy byte array
    """
    #
    buff_size = byte_mv.shape[0]
    while buff_size < max_bytes_per_coverage:
        buff_size *= 2

    if byte_mv.shape[0] < buff_size:
        # create a new bigger byte_mv
        byte_mv = np.empty(buff_size, dtype='b')

    return byte_mv


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
        items = np.memmap(items_fname, dtype=items_dtype, mode='r')
    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.loadtxt(items_fname, dtype=items_dtype, delimiter=",", skiprows=1, ndmin=1)
    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


item_map_key_dtype = np.dtype([('areaperil_id', areaperil_int), ('vulnerability_id', np.int32)])


@njit(cache=True, fastmath=True)
def generate_item_map(items, coverages):
    """Generate item_map as a hashmap + jagged array; requires items to be sorted.

    Items must be sorted by (areaperil_id, vulnerability_id). Builds:
        - hashmap: (areaperil_id, vulnerability_id) → pair_index
        - jagged:  pair_index → item indices via ja_offsets / ja_item_idxs

    Args:
        items (numpy.ndarray): 1-d structured array sorted by
            (areaperil_id, vulnerability_id).
        coverages (numpy.ndarray): coverage id to information on items.

    Returns:
        item_map_hm (np.array[uint8]): packed hashmap table.
        item_map_hm_keys (np.array[item_map_key_dtype]): key storage for the hashmap.
        item_map_ja_offsets (np.array[oasis_int]): CSR offsets (N_pairs + 1).
        item_map_ja_item_idxs (np.array[oasis_int]): flat item indices into items array.
    """
    N = items.shape[0]

    # Pass 1: extract unique (areaperil_id, vulnerability_id) pairs and record pair boundaries
    item_map_hm_keys = np.empty(max(N, 1), dtype=item_map_key_dtype)
    item_map_ja_offsets = np.empty(N + 1, dtype=oasis_int)
    item_map_ja_offsets[0] = 0
    pair_idx = np.int32(-1)
    prev_ap = areaperil_int.type(0)
    prev_vuln = np.int32(-1)
    for j in range(N):
        ap = items[j]['areaperil_id']
        vuln = items[j]['vulnerability_id']
        coverages[items[j]['coverage_id']]['max_items'] += 1
        if ap != prev_ap or vuln != prev_vuln:
            if pair_idx >= 0:
                item_map_ja_offsets[pair_idx + 1] = j
            pair_idx += 1
            item_map_hm_keys[pair_idx]['areaperil_id'] = ap
            item_map_hm_keys[pair_idx]['vulnerability_id'] = vuln
            prev_ap = ap
            prev_vuln = vuln
    if pair_idx >= 0:
        item_map_ja_offsets[pair_idx + 1] = N
    n_pairs = pair_idx + 1
    item_map_hm_keys = item_map_hm_keys[:n_pairs]
    item_map_ja_offsets = item_map_ja_offsets[:n_pairs + 1]

    # Pass 2: build hashmap from the extracted keys (by-position mode)
    item_map_hm = init_dict(n_pairs)
    hm_info, hm_lookup, hm_index = hm_unpack(item_map_hm)
    for i in range(n_pairs):
        result = hm_try_add_key(hm_info, hm_lookup, hm_index,
                                item_map_hm_keys, item_map_hm_keys[i], i)
        while result == hm_i_add_key_fail:
            item_map_hm = hm_rehash(item_map_hm, item_map_hm_keys)
            hm_info, hm_lookup, hm_index = hm_unpack(item_map_hm)
            result = hm_try_add_key(hm_info, hm_lookup, hm_index,
                                    item_map_hm_keys, item_map_hm_keys[i], i)

    return (item_map_hm, item_map_hm_keys, item_map_ja_offsets)


@redirect_logging(exec_name='gulpy')
def run(run_dir, ignore_file_type, sample_size, loss_threshold, alloc_rule, debug,
        random_generator, peril_filter=[], file_in=None, file_out=None, ignore_correlation=False, **kwargs):
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
        ignore_correlation (bool): if True, do not compute correlated random samples.

    Raises:
        ValueError: if alloc_rule is not 0, 1, or 2.

    Returns:
        int: 0 if no errors occurred.
    """
    logger.info("starting gulpy")

    # --- load or build read-only structures --------------------------------
    from oasislmf.pytools.gul.structure import (
        gulpy_structure_exists, load_gulpy_structure,
        build_structures as gul_build_structures,
    )
    if gulpy_structure_exists(run_dir):
        logger.info("loading pre-computed gulpy structures (shared memory)")
        structures = load_gulpy_structure(run_dir)
    else:
        logger.info("building gulpy structures from input files")
        structures = gul_build_structures(run_dir, ignore_file_type, peril_filter)

    damage_bins = structures['damage_bins']
    coverages = structures['coverages'].copy()  # writable copy: cur_items/start_items mutated per event
    items = structures['items']
    item_map_hm = structures['item_map_hm']
    item_map_hm_keys = structures['item_map_hm_keys']
    item_map_ja_offsets = structures['item_map_ja_offsets']

    # init array to store the coverages to be computed
    # coverages are numbered from 1, therefore skip element 0.
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
        stream_out.write(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM))
        stream_out.write(np.int32(sample_size).tobytes())

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2, 3]:
            raise ValueError(f"Expect alloc_rule to be 0, 1, 2, or 3, got {alloc_rule}")

        cursor = 0

        # create the array to store the seeds
        seeds = np.zeros(len(np.unique(items['group_id'])), dtype=items_dtype['group_id'])

        # --- correlation setup from pre-computed structures --------------------
        do_correlation = bool(structures['do_correlation'])
        if ignore_correlation:
            do_correlation = False
            logger.info("Correlated random number generation: switched OFF because --ignore-correlation is True.")

        if do_correlation:
            logger.info("Correlated random number generation: switched ON.")
            corr_data_by_item_id = structures['corr_data_by_item_id']
            unique_peril_correlation_groups = structures['unique_peril_correlation_groups']
            norm_inv_cdf = structures['norm_inv_cdf']
            norm_cdf = structures['norm_cdf']

            corr_seeds = np.zeros(np.max(unique_peril_correlation_groups) + 1, dtype='int64')

            arr_min, arr_max, arr_N = 1e-16, 1 - 1e-16, 1000000
            arr_min_cdf, arr_max_cdf = -20., 20.

            # buffer to be re-used to store all the correlated random values
            z_unif = np.zeros(sample_size, dtype='float64')

        else:
            if not ignore_correlation:
                logger.info("Correlated random number generation: switched OFF because 0 peril correlation groups were detected or "
                            "the correlation value is zero for all peril correlation groups.")
            # create dummy data structures with proper dtypes to allow correct numba compilation
            corr_seeds = np.zeros(1, dtype='int64')
            corr_data_by_item_id = np.ndarray(1, dtype=correlations_dtype)
            arr_min, arr_max, arr_N = 0, 0, 0
            arr_min_cdf, arr_max_cdf = 0, 0
            norm_inv_cdf, norm_cdf = np.zeros(1, dtype='float64'), np.zeros(1, dtype='float64')
            z_unif = np.zeros(1, dtype='float64')

        # create buffer to be reused to store all losses for one coverage
        losses_buffer = np.zeros((sample_size + NUM_IDX + 1, np.max(coverages[1:]['max_items'])), dtype=oasis_float)
        byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size

        counter = 0
        timer = time.time()
        socket_server_val = kwargs.get('socket_server', 'False')
        ping = socket_server_val != 'False'
        ping_port = int(socket_server_val) if ping and str(socket_server_val).isdigit() else None
        for event_data in read_getmodel_stream(streams_in, items,
                                               item_map_hm, item_map_hm_keys,
                                               item_map_ja_offsets,
                                               coverages, compute, seeds):
            event_id, compute_i, items_data, damagecdfrecs, recs, rec_idx_ptr, rng_index = event_data

            # generation of "base" random values is done as before
            rndms_base = generate_rndm(seeds[:rng_index], sample_size)

            # to generate the correlated part, we do the hashing here for now (instead of in stream_to_data)
            # generate the correlated samples for the whole event, for all peril correlation groups
            if do_correlation:
                generate_correlated_hash_vector(unique_peril_correlation_groups, event_id, corr_seeds)
                eps_ij = generate_rndm(corr_seeds, sample_size, skip_seeds=1)

            else:
                # create dummy data structures with proper dtypes to allow correct numba compilation
                eps_ij = np.zeros((1, 1), dtype='float64')

            last_processed_coverage_ids_idx = 0

            # adjust buff size so that the buffer fits the longest coverage
            byte_mv = adjust_byte_mv_size(byte_mv, np.max(coverages['cur_items']) * max_bytes_per_item)

            while last_processed_coverage_ids_idx < compute_i:
                cursor, last_processed_coverage_ids_idx = compute_event_losses(
                    event_id, coverages, compute[:compute_i], items_data,
                    last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr,
                    damage_bins, loss_threshold, losses_buffer, alloc_rule, do_correlation, rndms_base, eps_ij, corr_data_by_item_id,
                    arr_min, arr_max, arr_N, norm_inv_cdf, arr_min_cdf, arr_max_cdf, norm_cdf, z_unif, debug,
                    max_bytes_per_item, byte_mv, cursor
                )

                # write the losses to the output stream
                write_start = 0
                while write_start < cursor:
                    select([], select_stream_list, select_stream_list)
                    write_start += stream_out.write(byte_mv[write_start:cursor].tobytes())
                    counter += 1

                cursor = 0

            logger.info(f"event {event_id} DONE")

            if ping and time.time() - timer > SERVER_UPDATE_TIME:
                ping_data = {"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)}
                if ping_port is not None:
                    ping_data['port_override'] = ping_port
                oasis_ping(ping_data)
                counter = 0

        if ping:
            ping_data = {"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)}
            if ping_port is not None:
                ping_data['port_override'] = ping_port
            oasis_ping(ping_data)
    return 0


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, coverages, coverage_ids, items_data,
                         last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr, damage_bins,
                         loss_threshold, losses, alloc_rule, do_correlation, rndms_base, eps_ij, corr_data_by_item_id,
                         arr_min, arr_max, arr_N, norm_inv_cdf, arr_min_cdf, arr_max_cdf, norm_cdf,
                         z_unif, debug, max_bytes_per_item, byte_mv, cursor):
    """Compute losses for an event.

    Args:
        event_id (int32): event id.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        coverage_ids (numpy.array[int]): array of unique coverage ids used in this event.
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
        do_correlation (bool): if True, compute correlated random samples.
        rndms (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        max_bytes_per_item (int): maximum bytes to be written in the output stream for an item.
        byte_mv (numpy.array): byte view of where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int, int: updated value of cursor, last last_processed_coverage_ids_idx
    """
    for coverage_i in range(last_processed_coverage_ids_idx, coverage_ids.shape[0]):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']  # coverages are indexed from 1
        Nitem_ids = coverage['cur_items']
        exposureValue = tiv / Nitem_ids

        # estimate max number of bytes needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        # number of records of type gulSampleslevelRec_size is sample_size + 5 (negative sidx) + 1 (terminator line)
        est_cursor_bytes = Nitem_ids * max_bytes_per_item

        # return before processing this coverage if the number of free bytes left in the buffer
        # is not sufficient to write out the full coverage
        if cursor + est_cursor_bytes > byte_mv.shape[0]:
            return cursor, last_processed_coverage_ids_idx

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
                if do_correlation and corr_data_by_item_id[item['item_id']]['damage_correlation_value'] > 0:
                    item_corr_data = corr_data_by_item_id[item['item_id']]
                    get_corr_rval(
                        eps_ij[item_corr_data['peril_correlation_group']], rndms_base[rng_index],
                        item_corr_data['damage_correlation_value'], arr_min, arr_max, arr_N, norm_inv_cdf,
                        arr_min_cdf, arr_max_cdf, norm_cdf, sample_size, z_unif
                    )
                    rndms = z_unif
                else:
                    rndms = rndms_base[rng_index]

                if debug:
                    for sample_idx in range(1, sample_size + 1):
                        rval = rndms[sample_idx - 1]
                        losses[sample_idx, item_i] = rval
                else:
                    for sample_idx in range(1, sample_size + 1):
                        # cap `rval` to the maximum `prob_to` value (which should be 1.)
                        rval = rndms[sample_idx - 1]

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
                              byte_mv, cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    return cursor, last_processed_coverage_ids_idx


@njit(cache=True, fastmath=True)
def write_losses(event_id, sample_size, loss_threshold, losses, item_ids, alloc_rule, tiv,
                 byte_mv, cursor):
    """Write the computed losses.

    Args:
        event_id (int32): event id.
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): losses for all item_ids
        item_ids (numpy.array[ITEM_ID_TYPE]): ids of items whose losses are in `losses`.
        alloc_rule (int): back-allocation rule.
        tiv (oasis_float): total insured value.
        byte_mv (numpy.ndarray): byte view of where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        int: updated values of cursor
    """
    if alloc_rule == 2:
        setmaxloss(losses)

    if tiv > 0:
        # check whether the sum of losses-per-sample exceeds TIV
        # if so, split TIV in proportion to the losses

        if alloc_rule in [1, 2]:
            split_tiv_classic(losses[TIV_IDX], tiv)
            split_tiv_classic(losses[MAX_LOSS_IDX], tiv)
            split_tiv_classic(losses[MEAN_IDX], tiv)
            for sample_i in range(1, losses.shape[0] - NUM_IDX):
                split_tiv_classic(losses[sample_i], tiv)

        elif alloc_rule == 3:
            split_tiv_multiplicative(losses[TIV_IDX], tiv)
            split_tiv_multiplicative(losses[MAX_LOSS_IDX], tiv)
            split_tiv_multiplicative(losses[MEAN_IDX], tiv)
            for sample_i in range(1, losses.shape[0] - NUM_IDX):
                split_tiv_multiplicative(losses[sample_i], tiv)

    # output the losses for all the items
    for item_j in range(item_ids.shape[0]):

        # write header
        cursor = mv_write_item_header(byte_mv, cursor, event_id, item_ids[item_j])

        # write negative sidx
        for sample_idx in SPECIAL_SIDX:
            cursor = mv_write_sidx_loss(byte_mv, cursor, sample_idx, losses[sample_idx, item_j])

        # write the random samples (only those with losses above the threshold)
        for sample_idx in range(1, sample_size + 1):
            if losses[sample_idx, item_j] >= loss_threshold:
                cursor = mv_write_sidx_loss(byte_mv, cursor, sample_idx, losses[sample_idx, item_j])

        # write terminator for the samples for this item
        cursor = mv_write_delimiter(byte_mv, cursor)

    return cursor
