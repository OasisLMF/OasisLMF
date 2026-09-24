"""This file is the entry point for the gul command for the package."""
import logging
from math import sqrt
import os
import sys
from contextlib import ExitStack
from select import select

import numpy as np
from numba import njit
import time
from oasislmf.utils.ping import oasis_ping, oasis_ping_async

from oasislmf.pytools.common.data import correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import (PIPE_CAPACITY, check_packed_item_fits, check_packing_supported,
                                                  encode_sidx, max_emitted_blocks,
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
                                       accumulate_hermite_coeffs, loss_correlation, HERMITE_TERMS,
                                       apply_alloc_rule)
from oasislmf.pytools.gul.io import read_getmodel_stream
from oasislmf.pytools.gul.random import (_lh_philox_block, PHILOX_U32_MASK, PHILOX_SHIFT32,
                                         cdf_min,
                                         generate_correlated_hash_vector,
                                         get_corr_rval, get_random_generator,
                                         inv_factor, norm_factor, x_min)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.defaults import SERVER_UPDATE_TIME

logger = logging.getLogger(__name__)


# A fused coverage is flushed between building blocks, so ONE block is all it strictly needs.
# Sizing the buffer to exactly that is correct but wasteful: a 630,510-building item is then
# flushed ~8,000 times per event at S=200, and every resume recomputes that item's CDF and
# analytic moments. Room for many blocks brings that to ~124 flushes -- measured at 5% less
# wall time on a 200,000-building item at S=200, for no extra resident memory. What matters is
# that the bound is this constant and not the building count, which was the point of flushing.
FUSED_FLUSH_TARGET_BYTES = 8 * 1024 * 1024


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
        generate_rndm = get_random_generator(random_generator)

        # Building packing is the N > 1 case of one mechanism, not a second path: an unpacked run
        # is every item carrying one building, and the packed generator's first block per seed is
        # the legacy draw byte-for-byte. So the compute always takes the packed route.
        # Signed: magnitude is the building count, a negative sign means "keep the buildings
        # separate". Unpacked into locals wherever it is consumed -- never used raw as a bound.
        n_buildings_by_item_id = structures['n_buildings_by_item_id']
        damage_correlation_by_item_id = structures['damage_correlation_by_item_id']
        max_buildings = int(np.abs(n_buildings_by_item_id).max())
        check_packing_supported(random_generator, n_buildings_by_item_id)
        # only kept-separate items meet either stream ceiling: a summed one writes a single
        # block at sidx 1..S however many buildings it carries
        check_packed_item_fits(max_emitted_blocks(n_buildings_by_item_id), sample_size)

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
            # The structures were built with the file's correlation, and --ignore-correlation is a
            # RUN-time flag the build never saw. A summed packed item's std_dev is combined with
            # this value, so leaving it would report the spread of a correlation nothing drew --
            # and worse, the dummy norm_inv_cdf substituted below would make the Hermite
            # coefficients meaningless rather than merely stale. gulmc writes the effective value
            # per event for the same reason; gulpy carries it per item, so zero it here.
            damage_correlation_by_item_id = np.zeros_like(damage_correlation_by_item_id)

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
        # Per-building samples: the specials stay on losses_buffer, being building-independent.
        # Sized to the coverages that genuinely have to hold every building at once rather than
        # to the portfolio's largest location -- see buffered_building_width.
        max_buffered_buildings = buffered_building_width(
            alloc_rule, sample_size, items['coverage_id'],
            n_buildings_by_item_id[items['item_id']], coverages['max_items'])
        building_losses = np.zeros((max(sample_size, 1), max_items_per_coverage, max_buffered_buildings),
                                   dtype=oasis_float)
        # Accumulates a summed-at-source item's buildings when it is emitted as it is computed.
        # float64, not oasis_float: write_losses sums into a `loss = 0.` local, which numba types
        # as float64, so accumulating in float32 here would round differently.
        summed_scratch = np.zeros(max(sample_size, 1), dtype=np.float64)
        # per item of the current coverage, the correlation between two of its buildings'
        # LOSSES -- 0 unless the item is summed and correlated, the only case that reads it
        loss_correlation_by_item = np.zeros(max_items_per_coverage, dtype=oasis_float)
        # Generator 2 is counter-based, so a building's block is a pure function of the group
        # key and the building index and can be produced where it is consumed -- which is the
        # reason it is the only generator packing is allowed on. These hold one building's worth;
        # empty_draws stands in for the per-group array that is then never built.
        lazy_draws = np.int8(1 if random_generator == 2 else 0)
        draw_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        perm_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        # 2d like the real array: numba unifies the two branches of the assignment below, and a
        # 1d stand-in would make rndms_base[rng_index] a scalar on one side and a row on the other
        empty_draws = np.empty((1, 1), dtype='float64')
        hermite_coeffs = np.zeros(HERMITE_TERMS, dtype='float64')
        # Resume point WITHIN a coverage, so a flush need not fall on a coverage boundary:
        # [0] is the next item of that coverage to process, [1] how many of its buildings have
        # already been emitted. gulpy signals resumption through its return value, so this is
        # carried in an array the callee mutates rather than on a state struct.
        resume_state = np.zeros(2, dtype=np.int64)
        byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')

        # One block: the item header, a building's NUM_IDX specials and S samples, and the
        # delimiter. Only the first block of an item carries the header and only the last the
        # delimiter, so charging every block for both is a deliberate over-estimate.
        max_bytes_per_block = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size
        max_bytes_per_item = max_bytes_per_block
        # a kept-separate item writes one block of that per building; a summed one writes a single
        # block whatever it carries
        max_bytes_per_item *= max_emitted_blocks(n_buildings_by_item_id)

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

            # One row of sample_size per rng group. Generator 2 draws each building's block
            # where it is used instead, so nothing is built here for it -- see lazy_draws.
            if lazy_draws:
                rndms_base = empty_draws
            else:
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
            # The buffer no longer has to hold a whole coverage. A fused one is flushed between
            # buildings, so one block is enough; only a coverage still emitted whole through
            # write_losses has to fit entire.
            cur_items = coverages['cur_items']
            emitted_whole = ~((sample_size > 0) & ((alloc_rule == 0) | (cur_items == 1)))
            # the extra header is what an item's first block is reserved WITH, so a buffer sized
            # to exactly one block would reject that reservation on an empty buffer, forever
            required_bytes = gulSampleslevelHeader_size + max_bytes_per_block
            if emitted_whole.any():
                required_bytes = max(required_bytes,
                                     int(cur_items[emitted_whole].max()) * max_bytes_per_item)
            # room for many blocks, but never more than the largest item could ever write
            required_bytes = max(required_bytes,
                                 min(FUSED_FLUSH_TARGET_BYTES, max_bytes_per_item))
            byte_mv = adjust_byte_mv_size(byte_mv, required_bytes)
            resume_state[:] = 0

            while last_processed_coverage_ids_idx < compute_i:
                resume_point_before = (last_processed_coverage_ids_idx,
                                       int(resume_state[0]), int(resume_state[1]))
                cursor, last_processed_coverage_ids_idx = compute_event_losses(
                    event_id, coverages, compute[:compute_i], items_data,
                    last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr,
                    damage_bins, loss_threshold, losses_buffer, alloc_rule, do_correlation, eps_ij, corr_data_by_item_id,
                    arr_min, arr_inv_factor, norm_inv_cdf, arr_min_cdf, arr_norm_factor, norm_cdf, z_unif, debug,
                    building_losses, summed_scratch, resume_state, rndms_base,
                    seeds, lazy_draws, draw_scratch, perm_scratch,
                    loss_correlation_by_item, hermite_coeffs,
                    n_buildings_by_item_id, damage_correlation_by_item_id,
                    max_bytes_per_item, max_bytes_per_block, byte_mv, cursor
                )

                # A call that stops short must have advanced the resume point. It only stops
                # because the buffer is full, and the buffer is empty on entry, so if it stops at
                # the same place it will keep stopping there -- an infinite loop with no error.
                # It is the RESUME POINT that has to move, not the cursor: a resume that restarts
                # an item re-emits the same blocks forever and writes plenty while never
                # finishing. The sizing above makes this unreachable today; the check turns a
                # future violation of it into a failure rather than a hang.
                resume_point = (last_processed_coverage_ids_idx, int(resume_state[0]), int(resume_state[1]))
                if last_processed_coverage_ids_idx < compute_i and resume_point <= resume_point_before:
                    raise RuntimeError(
                        f"gulpy made no progress on event {event_id}: it asked to resume at "
                        f"coverage/item/building {resume_point}, no further on than the "
                        f"{resume_point_before} it started from, having written {cursor} bytes "
                        f"into a {byte_mv.shape[0]} byte buffer. The buffer must hold at least "
                        f"one building block plus an item header "
                        f"({gulSampleslevelHeader_size + max_bytes_per_block} bytes) and a whole "
                        f"coverage for any written through write_losses.")

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
                         z_unif, debug, building_losses, summed_scratch, resume_state, rndms_base,
                         seeds, lazy_draws, draw_scratch, perm_scratch,
                         loss_correlation_by_item, hermite_coeffs,
                         n_buildings_by_item_id, damage_correlation_by_item_id,
                         max_bytes_per_item, max_bytes_per_block, byte_mv, cursor):
    """Compute losses for an event.

    Args:
        damage_correlation_by_item_id (numpy.array[oasis_float]): per item_id, the correlation
          actually applied to its damage draws -- 0 when correlation is off. Only a summed
          packed item reads it, to combine its buildings' variances.
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
        resume_state (numpy.array[int64]): length 2, mutated here. [0] is the item of the
          returned coverage to resume at, [1] how many of its buildings were already emitted
          (0 = none, so the item header is still to be written). Both are cleared as each item
          and coverage completes.
        summed_scratch (numpy.array[float64]): length-S accumulator for a summed-at-source item
          emitted as it is computed, since its buildings are added up rather than written.
          float64 to match the precision write_losses accumulates at.
        building_losses (numpy.array[oasis_float]): 3d (sample_size, max_items, W)
          reusable buffer for the per-building samples.
        rndms_base (numpy.array[float64]): 2d (rng groups, sample_size) random values, one row
          per group. Empty when lazy_draws is set, where it is never read.
        seeds (numpy.array[int]): per rng group, the Philox key. Read only when lazy_draws is
          set, where it replaces rndms_base entirely.
        lazy_draws (int8): 1 when the generator is counter-based (generator 2), where a building's
          block is produced on demand. It is also the only generator that packing is allowed on,
          so every other generator reaches the loop below with exactly one building.
        draw_scratch (numpy.array[float64]): length-S buffer for one building's block.
        perm_scratch (numpy.array[float64]): length-S scratch the block generator permutes in.
        n_buildings_by_item_id (numpy.array[int]): per item, the signed building count. The
            magnitude is how many buildings the item carries; a negative sign means those
            buildings must reach the financial module as separate blocks, positive that they are
            summed here. Unpack it before use -- a negative value as a loop bound silently does
            nothing.
        loss_correlation_by_item (numpy.array[oasis_float]): per item of the current coverage,
          the correlation between two of its buildings' losses. 0 for everything that does not
          report the spread of a sum.
        hermite_coeffs (numpy.array[float64]): length HERMITE_TERMS scratch for that.
        max_bytes_per_item (int): maximum bytes to be written in the output stream for an item.
        max_bytes_per_block (int): the same for ONE building's block, which is the unit a fused
          coverage is flushed at.
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

        items = items_data[coverage['start_items']: coverage['start_items'] + coverage['cur_items']]

        # Emit each building as it is computed rather than buffering the coverage. Safe exactly
        # when the alloc-rule cap never has to look across items at a fixed building: with
        # alloc_rule 0 it does not run, and with one item the cross-item reductions act on a
        # length-1 slice, which is element-local. See buffered_building_width, which sizes
        # building_losses on the same rule.
        fuse_emit = sample_size > 0 and (alloc_rule == 0 or coverage['cur_items'] == 1)

        # A coverage emitted whole through write_losses has to fit the buffer before we start,
        # conservatively assuming every random sample is printed. The bound OVER-reserves beyond
        # that: max_bytes_per_item carries max_emitted_blocks over EVERY item, so a coverage of
        # one-building items is charged for the largest packed item anywhere. A fused coverage
        # reserves per BLOCK instead -- one per building where they are kept separate, one for the
        # whole item where they are summed -- which keeps the buffer off the largest building count.
        if not fuse_emit:
            if cursor + Nitem_ids * max_bytes_per_item > byte_mv.shape[0]:
                return cursor, last_processed_coverage_ids_idx

        for item_i in range(resume_state[0], coverage['cur_items']):
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

            # Only a SUMMED item reports the spread of a sum, and only then does the
            # correlation between two buildings' losses matter. gulpy samples no hazard, so
            # this item's own CDF is the whole story.
            item_rho = damage_correlation_by_item_id[item['item_id']]
            if n_buildings_by_item_id[item['item_id']] > 1 and item_rho > 0.:
                for k in range(HERMITE_TERMS):
                    hermite_coeffs[k] = 0.
                accumulate_hermite_coeffs(tiv, prob_to, bin_mean, Nbins, 1.,
                                          arr_min, arr_inv_factor, norm_inv_cdf, hermite_coeffs)
                loss_correlation_by_item[item_i] = loss_correlation(
                    hermite_coeffs, std_dev * std_dev, item_rho)
            else:
                loss_correlation_by_item[item_i] = 0.

            if sample_size > 0:
                # One block per building. An unpacked item is the N == 1 case, whose single block
                # is the legacy draw byte-for-byte. The specials above are building-independent.
                item_n_buildings = abs(n_buildings_by_item_id[item['item_id']])

                keep_separate_item = n_buildings_by_item_id[item['item_id']] < 0
                if fuse_emit:
                    # cur_items == 1 wherever alloc_rule != 0 here, so these length-1 slices ARE
                    # the whole cross-item vector write_losses would pass, and the reductions on
                    # them are the identity (setmaxloss, multiplicative) or a cap at tiv (classic)
                    for special in (TIV_IDX, MAX_LOSS_IDX, MEAN_IDX):
                        apply_alloc_rule(losses[special, item_i:item_i + 1], alloc_rule, tiv)
                    # The specials are recomputed above on every entry, so re-applying the cap
                    # after a resume caps fresh values rather than already-capped ones. Only the
                    # header must not be repeated.
                    if resume_state[1] == 0:
                        # The header and the first block are reserved TOGETHER. Writing the header
                        # first and only then finding the block does not fit leaves that header in
                        # the bytes we flush, and re-entry (resume_state[1] still 0) writes it
                        # again -- the reader decodes the second copy as a sidx/loss pair and
                        # rejects the item as carrying a duplicated sidx. max_bytes_per_block
                        # already includes one header, so this reserves two; the 8 spare bytes are
                        # what keep the per-block check below from firing on the block just
                        # reserved.
                        if cursor + gulSampleslevelHeader_size + max_bytes_per_block > byte_mv.shape[0]:
                            resume_state[0] = item_i
                            return cursor, last_processed_coverage_ids_idx
                        cursor = mv_write_item_header(byte_mv, cursor, event_id, item['item_id'])
                    if not keep_separate_item:
                        summed_scratch[:sample_size] = 0

                for building_i in range(resume_state[1], item_n_buildings):
                    # A kept-separate building is a block of its own, so the buffer is checked per
                    # block and the run resumes at the next one. A SUMMED item has no such check:
                    # it emits one block however many buildings it carries, so the reservation made
                    # with its header already covers it, and it could not be interrupted here in
                    # any case -- its accumulator would restart.
                    if fuse_emit and keep_separate_item:
                        if cursor + max_bytes_per_block > byte_mv.shape[0]:
                            resume_state[0] = item_i
                            resume_state[1] = building_i
                            return cursor, last_processed_coverage_ids_idx
                    if lazy_draws:
                        gs = np.uint64(seeds[rng_index])
                        _lh_philox_block(np.uint32(gs & PHILOX_U32_MASK),
                                         np.uint32(gs >> PHILOX_SHIFT32), building_i, sample_size,
                                         perm_scratch[:sample_size], draw_scratch[:sample_size])
                        rndms = draw_scratch[:sample_size]
                    else:
                        rndms = rndms_base[rng_index]
                    if do_correlation and corr_data_by_item_id[item['item_id']]['damage_correlation_value'] > 0:
                        item_corr_data = corr_data_by_item_id[item['item_id']]
                        get_corr_rval(
                            eps_ij[item_corr_data['peril_correlation_group']], rndms,
                            item_corr_data['damage_correlation_value'], arr_min, norm_inv_cdf, arr_inv_factor,
                            arr_min_cdf, norm_cdf, arr_norm_factor, sample_size, z_unif
                        )
                        rndms = z_unif

                    # one column is reused when the item is emitted as it is computed
                    building_col = 0 if fuse_emit else building_i

                    if debug:
                        for sample_idx in range(1, sample_size + 1):
                            building_losses[sample_idx - 1, item_i, building_col] = rndms[sample_idx - 1]
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
                                building_losses[sample_idx - 1, item_i, building_col] = gul
                            else:
                                building_losses[sample_idx - 1, item_i, building_col] = 0

                    if fuse_emit:
                        if alloc_rule != 0:
                            for s_i in range(sample_size):
                                apply_alloc_rule(building_losses[s_i, item_i:item_i + 1, 0], alloc_rule, tiv)
                        if keep_separate_item:
                            cursor = write_packed_building_block(
                                byte_mv, cursor, losses[:, item_i], building_i + 1,
                                building_losses[:, item_i, 0], sample_size, loss_threshold)
                        else:
                            # capped per building, then summed -- the order write_losses uses
                            for s_i in range(sample_size):
                                summed_scratch[s_i] += building_losses[s_i, item_i, 0]

                if fuse_emit:
                    if not keep_separate_item:
                        cursor = write_summed_specials(byte_mv, cursor, losses[:, item_i],
                                                       item_n_buildings,
                                                       loss_correlation_by_item[item_i])
                        for s_i in range(1, sample_size + 1):
                            loss = summed_scratch[s_i - 1]
                            if loss >= loss_threshold:
                                cursor = mv_write_sidx_loss(byte_mv, cursor, s_i, loss)
                    # one delimiter terminates the whole (multi-building) item
                    cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)
                    resume_state[1] = 0       # this item is done; the next starts fresh

        # a fused coverage has already emitted every item as it was computed
        if not fuse_emit:
            cursor = write_losses(
                event_id, sample_size, loss_threshold, losses[:, :items.shape[0]],
                building_losses[:, :items.shape[0], :], items['item_id'],
                n_buildings_by_item_id[items['item_id']],
                loss_correlation_by_item[:items.shape[0]],
                alloc_rule, tiv, byte_mv, cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1
        resume_state[0] = 0                   # the next coverage starts at its first item

    return cursor, last_processed_coverage_ids_idx


def buffered_building_width(alloc_rule, sample_size, coverage_ids, packed_buildings, max_items_by_coverage):
    """How many building columns the per-building sample buffer needs.

    A coverage has to hold all its buildings at once only when the alloc-rule cap runs ACROSS
    items at a fixed building -- ``setmaxloss_items`` and ``split_tiv_*`` read
    ``building_losses[s, :, b]``. With one item on the coverage those reduce to element-local
    operations on a length-1 slice, and with ``alloc_rule`` 0 they do not run at all, so such a
    coverage can emit each building as it is computed and needs a single column. The width is
    therefore set by the coverages that genuinely buffer, not by the portfolio's largest
    location: a packed exposure with one item per coverage drops from max_buildings to 1.

    ``max_items`` is used rather than a per-event item count because the buffer is allocated
    once for the run, so a coverage that is multi-item for any event must be accommodated.

    Passed the three columns rather than the arrays holding them, because gulmc carries the
    count on the items array while gulpy keeps it in n_buildings_by_item_id.

    Args:
        alloc_rule (int): back-allocation rule for the run.
        sample_size (int): logical number of random samples per building (S).
        coverage_ids (numpy.array[int]): per item, the coverage it sits on.
        packed_buildings (numpy.array[int]): per item, the SIGNED building count; only the
            magnitude matters here.
        max_items_by_coverage (numpy.array[int]): per coverage_id, the most items it ever holds.

    Returns:
        int: number of building columns to allocate, at least 1.
    """
    # sample_size 0 writes no samples, so write_losses never indexes the building axis at all
    if alloc_rule == 0 or sample_size <= 0 or coverage_ids.shape[0] == 0:
        return 1
    on_multi_item_coverage = max_items_by_coverage[coverage_ids] > 1
    if not on_multi_item_coverage.any():
        return 1
    return int(np.abs(packed_buildings[on_multi_item_coverage]).max())


@njit(cache=True, fastmath=True)
def write_packed_building_block(byte_mv, cursor, item_specials, b, sample_losses,
                                sample_size, loss_threshold):
    """Emit one building's block of a packed item: its shifted specials, then its samples.

    ``item_specials`` is indexed by the negative special sidx directly (it is a column of
    ``losses``, whose first axis wraps), and its values are building-independent -- only the
    sidx they are written at shifts with ``b``.

    Args:
        byte_mv (numpy.ndarray): byte view of the output buffer.
        cursor (int): index in byte_mv at which to start writing.
        item_specials (numpy.array[oasis_float]): this item's ``losses[:, item_j]`` column.
        b (int): 1-based building index.
        sample_losses (numpy.array[oasis_float]): this building's S sample losses.
        sample_size (int): logical number of random samples per building (S).
        loss_threshold (float): threshold above which random samples are written.

    Returns:
        int: updated cursor.
    """
    for special_idx in SPECIAL_SIDX:
        cursor = mv_write_sidx_loss(byte_mv, cursor, encode_sidx(b, special_idx, sample_size),
                                    item_specials[special_idx])
    for sample_idx in range(1, sample_size + 1):
        loss = sample_losses[sample_idx - 1]
        if loss >= loss_threshold:
            cursor = mv_write_sidx_loss(byte_mv, cursor, encode_sidx(b, sample_idx, sample_size), loss)
    return cursor


@njit(cache=True, fastmath=True)
def write_summed_specials(byte_mv, cursor, item_specials, nb_item, loss_correlation):
    """Emit the specials of an item whose buildings are summed at source.

    Args:
        byte_mv (numpy.ndarray): byte view of the output buffer.
        cursor (int): index in byte_mv at which to start writing.
        item_specials (numpy.array[oasis_float]): this item's ``losses[:, item_j]`` column.
        nb_item (int): how many buildings are summed into this item.
        loss_correlation (oasis_float): the correlation between two of this item's buildings'
            LOSSES, not the copula correlation applied to their draws. The two differ because the
            damage curve attenuates the copula -- see loss_correlation() in gul.core. 0 where
            correlation is off.

    Returns:
        int: updated cursor.
    """
    for special_idx in SPECIAL_SIDX:
        value = item_specials[special_idx]
        if special_idx == CHANCE_OF_LOSS_IDX:
            pass                      # a probability, shared by the buildings
        elif special_idx == STD_DEV_IDX:
            # The buildings of one item share damage_eps_ij[peril_correlation_group], so
            # they are NOT independent: var(sum) = sigma^2 * (N + N(N-1)*r), which is
            # N^2*r for large N rather than N. Scaling by sqrt(N) alone understates
            # sigma by about sqrt(N*r) -- 5.6x at 64 buildings and r 0.7, and it grows
            # with the count.
            #
            # r is the correlation between two buildings' LOSSES, which is NOT the copula
            # correlation: the damage curve attenuates it. Passing the copula value here
            # instead overstated sigma by up to ~39%. gul.core.loss_correlation does the
            # conversion, exactly under the one-factor copula.
            combined = nb_item + nb_item * (nb_item - 1) * loss_correlation
            value = value * sqrt(combined if combined > 0 else nb_item)
        else:
            value = value * nb_item   # mean, tiv and max are additive
        cursor = mv_write_sidx_loss(byte_mv, cursor, special_idx, value)
    return cursor


@njit(cache=True, fastmath=True)
def write_losses(event_id, sample_size, loss_threshold, losses, building_losses,
                 item_ids, n_buildings, loss_correlation, alloc_rule, tiv,
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
        loss_correlation (numpy.array[oasis_float]): per item, the correlation between two of its
            buildings' LOSSES -- not the copula correlation applied to their draws, which the
            damage curve attenuates. 0 where correlation is off. Only a summed item reads it, to
            combine its buildings' variances.
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

    # The same cap the fused path applies per item as it computes -- here over the whole
    # cross-item vector, which is the only difference between the two.
    if alloc_rule != 0:
        for special in (TIV_IDX, MAX_LOSS_IDX, MEAN_IDX):
            apply_alloc_rule(losses[special], alloc_rule, tiv)
        for b in range(max_nb):
            for sample_idx in range(sample_size):
                apply_alloc_rule(building_losses[sample_idx, :, b], alloc_rule, tiv)

    for item_j in range(item_ids.shape[0]):
        cursor = mv_write_item_header(byte_mv, cursor, event_id, item_ids[item_j])
        # Unpack the signed count into an unsigned bound and a flag. The raw value must never reach
        # a range(), which would silently iterate zero times and drop the item's buildings.
        packed_item = n_buildings[item_j]
        nb_item = abs(packed_item)
        keep_separate = packed_item < 0

        if keep_separate:
            for b in range(1, nb_item + 1):
                cursor = write_packed_building_block(byte_mv, cursor, losses[:, item_j], b,
                                                     building_losses[:, item_j, b - 1],
                                                     sample_size, loss_threshold)
        else:
            # summed at source: an ordinary unpacked item covering all nb_item buildings
            cursor = write_summed_specials(byte_mv, cursor, losses[:, item_j], nb_item,
                                           loss_correlation[item_j])
            for sample_idx in range(1, sample_size + 1):
                loss = 0.
                for b in range(nb_item):
                    loss += building_losses[sample_idx - 1, item_j, b]
                if loss >= loss_threshold:
                    cursor = mv_write_sidx_loss(byte_mv, cursor, sample_idx, loss)

        # one delimiter terminates the whole (multi-building) item
        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)  # item delimiter

    return cursor
