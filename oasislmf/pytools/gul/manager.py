"""This file is the entry point for the gul command for the package."""
import logging
import os
import sys
from contextlib import ExitStack
from select import select

import numpy as np
from numba import njit
import time
from oasislmf.utils.ping import oasis_ping, oasis_ping_async

from oasislmf.pytools.common.data import correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import (PIPE_CAPACITY, check_packed_sidx_fits, encode_sidx,
                                                  mv_write_item_header,
                                                  mv_write_sidx_loss,
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
from oasislmf.pytools.gul.core import (compute_mean_loss, get_gul,
                                       setmaxloss_items,
                                       split_tiv_classic,
                                       split_tiv_multiplicative)
from oasislmf.pytools.gul.io import read_getmodel_stream
from oasislmf.pytools.gul.random import (build_packed_rndm_offsets, cdf_min,
                                         generate_correlated_hash_vector,
                                         get_corr_rval, get_correlation_generator,
                                         get_sample_generator,
                                         inv_factor, norm_factor, x_min)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.defaults import SERVER_UPDATE_TIME

logger = logging.getLogger(__name__)


@njit(cache=True)
def adjust_byte_mv_size(byte_mv, max_bytes_per_coverage):
    """Adjust buff size so that the buffer fits the longest coverage

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
        ignore_file_type (set(str)): file extension to ignore when loading
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        alloc_rule (int): back-allocation rule.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        random_generator (int): random generator function id.
        peril_filter (list[int], optional): list of perils to include in the computation (all
          included if empty). Defaults to [].
        file_in (str, optional): filename of input stream. Defaults to None.
        file_out (str, optional): filename of output stream. Defaults to None.
        ignore_correlation (bool): if True, do not compute correlated random samples.
        **kwargs: additional keyword arguments, accepted and ignored so that callers can forward a
          wider parameter dict.

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
        generate_rndm = get_correlation_generator(random_generator)

        # Building packing is the N > 1 case of one mechanism, not a second path: an unpacked run
        # is every item carrying one building, and the packed generator's first block per seed is
        # the legacy draw byte-for-byte. So the compute always takes the packed route.
        # Signed: magnitude is the building count, a negative sign means "keep the buildings
        # separate". Unpacked into locals wherever it is consumed -- never used raw as a bound.
        n_buildings_by_item_id = structures['n_buildings_by_item_id']
        max_buildings = int(np.abs(n_buildings_by_item_id).max())
        check_packed_sidx_fits(max_buildings, sample_size, oasis_int)
        generate_rndm_packed = get_sample_generator(random_generator)

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

            arr_min, arr_min_cdf = x_min, cdf_min
            arr_inv_factor, arr_norm_factor = inv_factor, norm_factor

            # buffer to be re-used to store all the correlated random values
            z_unif = np.zeros(sample_size, dtype='float64')

        else:
            if not ignore_correlation:
                logger.info("Correlated random number generation: switched OFF because 0 peril correlation groups were detected or "
                            "the correlation value is zero for all peril correlation groups.")
            # create dummy data structures with proper dtypes to allow correct numba compilation
            corr_seeds = np.zeros(1, dtype='int64')
            corr_data_by_item_id = np.ndarray(1, dtype=correlations_dtype)
            arr_min, arr_min_cdf = 0., 0.
            arr_inv_factor, arr_norm_factor = 0., 0.
            norm_inv_cdf, norm_cdf = np.zeros(1, dtype='float64'), np.zeros(1, dtype='float64')
            z_unif = np.zeros(1, dtype='float64')

        # create buffer to be reused to store all losses for one coverage
        max_items_per_coverage = np.max(coverages[1:]['max_items'])
        losses_buffer = np.zeros((sample_size + NUM_IDX + 1, max_items_per_coverage), dtype=oasis_float)
        # per-building samples: the specials stay on losses_buffer, being building-independent
        building_losses = np.zeros((max(sample_size, 1), max_items_per_coverage, max_buildings),
                                   dtype=oasis_float)
        byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size
        # a packed item writes one block of that per building
        max_bytes_per_item *= max_buildings

        # one entry per seed, holding the largest building count in its group
        n_buildings_by_rng = np.ones(seeds.shape[0] + 1, dtype='i4')

        counter = 0
        timer = time.time()
        socket_server_val = kwargs.get('socket_server', 'False')
        ping = socket_server_val != 'False'
        ping_port = int(socket_server_val) if ping and str(socket_server_val).isdigit() else None
        for event_data in read_getmodel_stream(streams_in, items,
                                               item_map_hm, item_map_hm_keys,
                                               item_map_ja_offsets,
                                               coverages, compute, seeds,
                                               n_buildings_by_item_id, n_buildings_by_rng):
            event_id, compute_i, items_data, damagecdfrecs, recs, rec_idx_ptr, rng_index = event_data

            # flat, ragged: seed i owns n_buildings_by_rng[i] blocks of sample_size, which is
            # one block of the legacy draw when nothing is packed
            rndm_offsets = build_packed_rndm_offsets(n_buildings_by_rng[:rng_index], sample_size)
            rndms_flat = generate_rndm_packed(
                seeds[:rng_index], sample_size, n_buildings_by_rng[:rng_index], rndm_offsets)

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
                    damage_bins, loss_threshold, losses_buffer, alloc_rule, do_correlation, eps_ij, corr_data_by_item_id,
                    arr_min, arr_inv_factor, norm_inv_cdf, arr_min_cdf, arr_norm_factor, norm_cdf, z_unif, debug,
                    building_losses, rndms_flat, rndm_offsets,
                    n_buildings_by_item_id,
                    max_bytes_per_item, byte_mv, cursor
                )

                # write the losses to the output stream
                write_start = 0
                while write_start < cursor:
                    select([], select_stream_list, select_stream_list)
                    write_start += stream_out.write(byte_mv[write_start:cursor].tobytes())

                cursor = 0

            logger.info(f"event {event_id} DONE")
            counter += 1

            if ping and time.time() - timer > SERVER_UPDATE_TIME:
                ping_data = {"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)}
                if ping_port is not None:
                    ping_data['port_override'] = ping_port
                oasis_ping_async(ping_data)
                counter = 0
                timer = time.time()

        if ping:
            ping_data = {"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)}
            if ping_port is not None:
                ping_data['port_override'] = ping_port
            oasis_ping(ping_data)
    return 0


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, coverages, coverage_ids, items_data,
                         last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr, damage_bins,
                         loss_threshold, losses, alloc_rule, do_correlation, eps_ij, corr_data_by_item_id,
                         arr_min, arr_inv_factor, norm_inv_cdf, arr_min_cdf, arr_norm_factor, norm_cdf,
                         z_unif, debug, building_losses, rndms_flat, rndm_offsets,
                         n_buildings_by_item_id,
                         max_bytes_per_item, byte_mv, cursor):
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
        eps_ij (np.array[float]): correlated random values for damage sampling.
        corr_data_by_item_id (np.array[correlations_dtype]): correlation values by item id.
        arr_min (float): minimum value of the inverse Gaussian cdf lookup table.
        arr_inv_factor (float): scaling factor to index the inverse Gaussian cdf lookup table.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf lookup table.
        arr_min_cdf (float): minimum value of the Gaussian cdf lookup table.
        arr_norm_factor (float): scaling factor to index the Gaussian cdf lookup table.
        norm_cdf (np.array[float]): Gaussian cdf lookup table.
        z_unif (np.array[float]): reusable buffer for correlated random values.
        debug (bool): if True, for each random sample, print to the output stream the random value
          instead of the loss.
        building_losses (numpy.array[oasis_float]): 3d (sample_size, max_items, max_buildings)
          reusable buffer for the per-building samples.
        rndms_flat (numpy.array[float64]): flat packed random values, seed-major then building.
        rndm_offsets (numpy.array[int64]): prefix-sum offsets into ``rndms_flat`` per seed.
        n_buildings_by_item_id (numpy.array[int]): per item, the signed building count. The
            magnitude is how many buildings the item carries; a negative sign means those
            buildings must reach the financial module as separate blocks, positive that they are
            summed here. Unpack it before use -- a negative value as a loop bound silently does
            nothing.
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
                # One block per building. An unpacked item is the N == 1 case, whose single block
                # is the legacy draw byte-for-byte. The specials above are building-independent.
                item_n_buildings = abs(n_buildings_by_item_id[item['item_id']])
                base_off = rndm_offsets[rng_index]
                for building_i in range(item_n_buildings):
                    rndms = rndms_flat[base_off + building_i * sample_size:
                                       base_off + (building_i + 1) * sample_size]
                    if do_correlation and corr_data_by_item_id[item['item_id']]['damage_correlation_value'] > 0:
                        item_corr_data = corr_data_by_item_id[item['item_id']]
                        get_corr_rval(
                            eps_ij[item_corr_data['peril_correlation_group']], rndms,
                            item_corr_data['damage_correlation_value'], arr_min, norm_inv_cdf, arr_inv_factor,
                            arr_min_cdf, norm_cdf, arr_norm_factor, sample_size, z_unif
                        )
                        rndms = z_unif

                    if debug:
                        for sample_idx in range(1, sample_size + 1):
                            building_losses[sample_idx - 1, item_i, building_i] = rndms[sample_idx - 1]
                        continue

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
                            building_losses[sample_idx - 1, item_i, building_i] = gul
                        else:
                            building_losses[sample_idx - 1, item_i, building_i] = 0

        cursor = write_losses_packed(
            event_id, sample_size, loss_threshold, losses[:, :items.shape[0]],
            building_losses[:, :items.shape[0], :], items['item_id'],
            n_buildings_by_item_id[items['item_id']],
            alloc_rule, tiv, byte_mv, cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    return cursor, last_processed_coverage_ids_idx


@njit(cache=True, fastmath=True)
def write_losses_packed(event_id, sample_size, loss_threshold, losses, building_losses,
                        item_ids, n_buildings, alloc_rule, tiv,
                        byte_mv, cursor):
    """Write building-packed losses for one coverage to the output byte buffer.

    A single item multiplexes its N buildings into the sample dimension via ``encode_sidx``:
    one stream item (header + delimiter) carries, per building ``b`` (1-based), that building's
    5 special records followed by its random samples. The special statistics (mean, std, tiv,
    chance-of-loss, max) are building-independent (they derive from the CDF, not the random
    draw), so the same ``losses[special, item_j]`` value is emitted for every building, only
    at building-shifted special sidx. The logical ``sample_size`` (S) written in the stream
    header is unchanged; the building index is recovered by consumers from the sidx.

    An item whose buildings nothing downstream can tell apart (a positive ``n_buildings``,
    i.e. the site levels sum them before applying any term) is written **summed** instead, as an
    ordinary unpacked item. Doing it here rather than in the financial module's reader is what
    keeps a packed stream unambiguous: the reader discriminates only on the sidx range, so if both
    kinds were emitted packed it could not tell which to collapse. Summing reproduces what the
    reader used to do on its behalf -- the additive specials (max, tiv, mean, and the std the
    financial module ignores) are scaled by the building count, chance-of-loss is
    building-independent and taken once -- so the result matches the aggregate of the same
    buildings as separate items.

    ``alloc_rule`` caps a coverage's item losses at its TIV. The coverage TIV here is the
    **per-building** share, because generation divides the location TIV by the building count, so
    the cap applies within each building block rather than across the location — which is exactly
    what row disaggregation does, where each building is its own coverage. The building-independent
    specials are capped once across items. Blocks are capped before they are summed, so a
    summed-at-source item comes out capped at ``n_buildings * tiv``, the location's TIV.

    Args:
        event_id (int32): event id.
        sample_size (int): logical number of random samples per building (S).
        loss_threshold (float): threshold above which random samples are written.
        losses (numpy.array[oasis_float]): 2d (S + NUM_IDX + 1, max_items) buffer; only the
          special rows (negative sidx) are read here.
        building_losses (numpy.array[oasis_float]): 3d (S, max_items, max_buildings) buffer of
          per-building random sample losses. Only the first ``n_buildings[item_j]`` slots of each
          item are read as written; the rest are zeroed here before the alloc-rule passes, so a
          caller need not clear the buffer between coverages.
        item_ids (numpy.array): item ids for the coverage being written.
        n_buildings (numpy.array[int]): per item in ``item_ids``, the SIGNED building count. The
            magnitude is how many buildings the item carries; a negative sign means they must be
            emitted as separate blocks, positive that they are summed into one ordinary item. It
            is unpacked into ``nb_item``/``keep_separate`` at the top of the write loop -- never
            use it raw as a bound.
        alloc_rule (int): back-allocation rule, deciding how the per-coverage TIV cap applies.
        tiv (oasis_float): the coverage's total insured value, per building.
        byte_mv (numpy.ndarray): byte view of the output buffer.
        cursor (int): index in byte_mv at which to start writing.

    Returns:
        int: updated cursor.
    """
    # n_buildings is SIGNED. Take the magnitude for anything used as a bound: comparing the raw
    # value would leave max_nb at 0 for the keep-separate items, and ranging over it would index
    # building_losses negatively.
    max_nb = 0
    for item_j in range(item_ids.shape[0]):
        nb = abs(n_buildings[item_j])
        if nb > max_nb:
            max_nb = nb

    # The alloc-rule passes below work across items at a fixed building index, so a slot no item
    # on this coverage wrote must read 0 rather than whatever a previous coverage left in the
    # reused buffer. Done here rather than in each compute loop because both bounds are known
    # here: n_buildings per item, and max_nb, past which nothing is read at all.
    if alloc_rule != 0:
        for item_j in range(item_ids.shape[0]):
            nb = abs(n_buildings[item_j])
            for b in range(nb, max_nb):
                for sample_idx in range(sample_size):
                    building_losses[sample_idx, item_j, b] = 0

    if alloc_rule == 2:
        setmaxloss_items(losses[TIV_IDX])
        setmaxloss_items(losses[MAX_LOSS_IDX])
        setmaxloss_items(losses[MEAN_IDX])
        for b in range(max_nb):
            for sample_idx in range(sample_size):
                setmaxloss_items(building_losses[sample_idx, :, b])

    if tiv > 0:
        if alloc_rule == 1 or alloc_rule == 2:
            split_tiv_classic(losses[TIV_IDX], tiv)
            split_tiv_classic(losses[MAX_LOSS_IDX], tiv)
            split_tiv_classic(losses[MEAN_IDX], tiv)
            for b in range(max_nb):
                for sample_idx in range(sample_size):
                    split_tiv_classic(building_losses[sample_idx, :, b], tiv)
        elif alloc_rule == 3:
            split_tiv_multiplicative(losses[TIV_IDX], tiv)
            split_tiv_multiplicative(losses[MAX_LOSS_IDX], tiv)
            split_tiv_multiplicative(losses[MEAN_IDX], tiv)
            for b in range(max_nb):
                for sample_idx in range(sample_size):
                    split_tiv_multiplicative(building_losses[sample_idx, :, b], tiv)

    for item_j in range(item_ids.shape[0]):
        cursor = mv_write_item_header(byte_mv, cursor, event_id, item_ids[item_j])
        # Unpack the signed count into an unsigned bound and a flag. The raw value must never reach
        # a range(), which would silently iterate zero times and drop the item's buildings.
        packed_item = n_buildings[item_j]
        nb_item = abs(packed_item)
        keep_separate = packed_item < 0

        if keep_separate:
            for b in range(1, nb_item + 1):
                # special (negative) sidx — same value for every building, shifted per building
                for special_idx in SPECIAL_SIDX:
                    cursor = mv_write_sidx_loss(byte_mv, cursor, encode_sidx(b, special_idx, sample_size),
                                                losses[special_idx, item_j])
                # random samples for this building
                for sample_idx in range(1, sample_size + 1):
                    loss = building_losses[sample_idx - 1, item_j, b - 1]
                    if loss >= loss_threshold:
                        cursor = mv_write_sidx_loss(byte_mv, cursor, encode_sidx(b, sample_idx, sample_size), loss)
        else:
            # summed at source: an ordinary unpacked item covering all nb_item buildings
            for special_idx in SPECIAL_SIDX:
                value = losses[special_idx, item_j]
                if special_idx != CHANCE_OF_LOSS_IDX:
                    value = value * nb_item
                cursor = mv_write_sidx_loss(byte_mv, cursor, special_idx, value)
            for sample_idx in range(1, sample_size + 1):
                loss = 0.
                for b in range(nb_item):
                    loss += building_losses[sample_idx - 1, item_j, b]
                if loss >= loss_threshold:
                    cursor = mv_write_sidx_loss(byte_mv, cursor, sample_idx, loss)

        # one delimiter terminates the whole (multi-building) item
        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)  # item delimiter

    return cursor
