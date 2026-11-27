"""This file is the entry point for the gul command for the package."""
import logging
from math import sqrt
import os
import sys
from contextlib import ExitStack

import numpy as np
from numba import njit
from oasislmf.utils.ping import EventProgress

from oasislmf.pytools.common.data import items_dtype
from oasislmf.pytools.common.event_stream import (encode_sidx, mv_write_item_header,
                                                  mv_write_sidx_loss)
from oasislmf.pytools.common.data import areaperil_int, oasis_int
from oasislmf.pytools.common.hashmap import (
    init_dict, unpack as hm_unpack, rehash as hm_rehash,
    _try_add_key as hm_try_add_key, i_add_key_fail as hm_i_add_key_fail,
)
from oasislmf.pytools.gul.common import (SPECIAL_SIDX, CHANCE_OF_LOSS_IDX,
                                         MAX_LOSS_IDX,
                                         MEAN_IDX, STD_DEV_IDX,
                                         TIV_IDX)
from oasislmf.pytools.gul.core import (compute_mean_loss, get_gul,
                                       accumulate_hermite_coeffs, loss_correlation, HERMITE_TERMS,
                                       apply_alloc_rule)
from oasislmf.pytools.gul.io import read_getmodel_stream
from oasislmf.pytools.gul.random import (_lh_philox_block, PHILOX_U32_MASK, PHILOX_SHIFT32,
                                         get_corr_rval)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.common.stage import StreamSink
from oasislmf.pytools.gul import stage as gul_stage
from oasislmf.pytools.utils import redirect_logging

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
        **kwargs: additional keyword arguments. socket_server (str) enables the progress ping and, when numeric, gives
          the port to override; analysis_pk is reported with each ping.

    Raises:
        ValueError: if alloc_rule is not 0, 1, 2, or 3.

    Returns:
        int: 0 if no errors occurred.
    """
    logger.info("starting gulpy")

    stage = gul_stage.GulpyStage(run_dir, sample_size=sample_size, loss_threshold=loss_threshold, alloc_rule=alloc_rule,
                                 debug=debug, random_generator=random_generator, ignore_file_type=ignore_file_type,
                                 peril_filter=peril_filter, ignore_correlation=ignore_correlation)

    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))
        sink = StreamSink(stream_out)

        stream_out.write(stage.header)

        progress = EventProgress(kwargs.get('socket_server', 'False'), kwargs.get("analysis_pk", None))
        for event_data in read_getmodel_stream(streams_in, stage.items, stage.item_map_hm, stage.item_map_hm_keys,
                                               stage.item_map_ja_offsets, stage.coverages, stage.compute, stage.seeds):
            stage.process_cdf_event(event_data, sink)
            progress.event_done()

        progress.finish()
    return 0


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id, coverages, coverage_ids, items_data,
                         last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr, damage_bins,
                         loss_threshold, losses, alloc_rule, do_correlation, eps_ij, corr_data_by_item_id,
                         arr_min, arr_inv_factor, norm_inv_cdf, arr_min_cdf, arr_norm_factor, norm_cdf,
                         z_unif, debug, building_losses, rndms_base,
                         seeds, lazy_draws, draw_scratch, perm_scratch,
                         loss_correlation_by_item, hermite_coeffs,
                         n_buildings_by_item_id, damage_correlation_by_item_id,
                         max_bytes_per_item, byte_mv, cursor):
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
        building_losses (numpy.array[oasis_float]): 3d (sample_size, max_items, max_buildings)
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

        # Every coverage is emitted whole -- there is no return between the first byte of a
        # coverage and its last -- so its output has to fit before we start. The bound
        # OVER-reserves: max_bytes_per_item carries max_emitted_blocks over EVERY item, so a
        # coverage of one-building items is charged for the largest packed item anywhere.
        if cursor + Nitem_ids * max_bytes_per_item > byte_mv.shape[0]:
            return cursor, last_processed_coverage_ids_idx

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

                for building_i in range(item_n_buildings):
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

                    if debug:
                        for sample_idx in range(1, sample_size + 1):
                            building_losses[sample_idx - 1, item_i, building_i] = rndms[sample_idx - 1]
                    else:
                        for sample_idx in range(1, sample_size + 1):
                            # cap `rval` to the maximum `prob_to` value (which should be 1.)
                            rval = rndms[sample_idx - 1]

                            if rval >= prob_to[Nbins - 1]:
                                rval = np.float64(prob_to[Nbins - 1]) - 0.00000003
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

        cursor = write_losses(
            event_id, sample_size, loss_threshold, losses[:, :items.shape[0]],
            building_losses[:, :items.shape[0], :], items['item_id'],
            n_buildings_by_item_id[items['item_id']],
            loss_correlation_by_item[:items.shape[0]],
            alloc_rule, tiv, byte_mv, cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    return cursor, last_processed_coverage_ids_idx


@njit(cache=True, fastmath=True)
def write_packed_building_specials(byte_mv, cursor, item_specials, b, sample_size):
    """Emit one building's analytic sidx.

    Args:
        byte_mv (numpy.ndarray): byte view of the output buffer.
        cursor (int): index in byte_mv at which to start writing.
        item_specials (numpy.array[oasis_float]): this item's ``losses[:, item_j]`` column.
        b (int): 1-based building index.
        sample_size (int): logical number of random samples per building (S).

    Returns:
        int: updated cursor.
    """
    for special_idx in SPECIAL_SIDX:
        cursor = mv_write_sidx_loss(byte_mv, cursor, encode_sidx(b, special_idx, sample_size),
                                    item_specials[special_idx])
    return cursor


@njit(cache=True, fastmath=True)
def write_packed_building_samples(byte_mv, cursor, b, sample_losses, sample_size, loss_threshold):
    """Emit one building's random samples, those above the threshold.

    Args:
        byte_mv (numpy.ndarray): byte view of the output buffer.
        cursor (int): index in byte_mv at which to start writing.
        b (int): 1-based building index.
        sample_losses (numpy.array[oasis_float]): this building's S sample losses.
        sample_size (int): logical number of random samples per building (S).
        loss_threshold (float): threshold above which random samples are written.

    Returns:
        int: updated cursor.
    """
    for sample_idx in range(1, sample_size + 1):
        loss = sample_losses[sample_idx - 1]
        if loss >= loss_threshold:
            cursor = mv_write_sidx_loss(byte_mv, cursor, encode_sidx(b, sample_idx, sample_size), loss)
    return cursor


@njit(cache=True, fastmath=True)
def write_summed_specials(byte_mv, cursor, item_specials, nb_buildings, loss_correlation):
    """Emit the specials of an item whose buildings are summed at source.

    Args:
        byte_mv (numpy.ndarray): byte view of the output buffer.
        cursor (int): index in byte_mv at which to start writing.
        item_specials (numpy.array[oasis_float]): this item's ``losses[:, item_j]`` column.
        nb_buildings (int): how many buildings are summed into this item's loss.
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
            combined = nb_buildings + nb_buildings * (nb_buildings - 1) * loss_correlation
            value = value * sqrt(combined if combined > 0 else nb_buildings)
        else:
            value = value * nb_buildings   # mean, tiv and max are additive
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
            is unpacked into ``nb_buildings``/``keep_separate`` at the top of the write loop -- never
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
    # value would leave max_buildings_coverage at 0 for the keep-separate items, and ranging
    # over it would index building_losses negatively.
    max_buildings_coverage = 0
    for item_j in range(item_ids.shape[0]):
        nb_buildings = abs(n_buildings[item_j])
        if nb_buildings > max_buildings_coverage:
            max_buildings_coverage = nb_buildings

    # The alloc-rule passes below work across items at a fixed building index, so a slot no item
    # on this coverage wrote must read 0 rather than whatever a previous coverage left in the
    # reused buffer. Done here rather than in each compute loop because both bounds are known
    # here: n_buildings per item, and max_buildings_coverage, past which nothing is read at all.
    if alloc_rule != 0:
        for item_j in range(item_ids.shape[0]):
            nb_buildings = abs(n_buildings[item_j])
            for b in range(nb_buildings, max_buildings_coverage):
                for sample_idx in range(sample_size):
                    building_losses[sample_idx, item_j, b] = 0

    # The cap runs across the items at one (sample, building), which is why every building of the
    # coverage has to be resident before any of it is written.
    if alloc_rule != 0:
        for special in (TIV_IDX, MAX_LOSS_IDX, MEAN_IDX):
            apply_alloc_rule(losses[special], alloc_rule, tiv)
        for b in range(max_buildings_coverage):
            for sample_idx in range(sample_size):
                apply_alloc_rule(building_losses[sample_idx, :, b], alloc_rule, tiv)

    for item_j in range(item_ids.shape[0]):
        cursor = mv_write_item_header(byte_mv, cursor, event_id, item_ids[item_j])
        # Unpack the signed count into an unsigned bound and a flag. The raw value must never reach
        # a range(), which would silently iterate zero times and drop the item's buildings.
        packed_item = n_buildings[item_j]
        nb_buildings = abs(packed_item)
        keep_separate = packed_item < 0

        if keep_separate:
            # Every building's analytics, then every building's samples. A building's specials
            # encode to local - NUM_SPECIAL_SIDX * (b - 1), so they ascend as b FALLS, and the
            # samples ascend as b rises: the item comes out ascending end to end, which is what
            # the loss stream requires and what lets a repeated sidx be caught by a comparison.
            # The analytic values do not vary by building, so this only reorders writes.
            for b in range(nb_buildings, 0, -1):
                cursor = write_packed_building_specials(byte_mv, cursor, losses[:, item_j], b,
                                                        sample_size)
            for b in range(1, nb_buildings + 1):
                cursor = write_packed_building_samples(byte_mv, cursor, b,
                                                       building_losses[:, item_j, b - 1],
                                                       sample_size, loss_threshold)
        else:
            # summed at source: an ordinary unpacked item covering all its buildings
            cursor = write_summed_specials(byte_mv, cursor, losses[:, item_j], nb_buildings,
                                           loss_correlation[item_j])
            for sample_idx in range(1, sample_size + 1):
                loss = 0.
                for b in range(nb_buildings):
                    loss += building_losses[sample_idx - 1, item_j, b]
                if loss >= loss_threshold:
                    cursor = mv_write_sidx_loss(byte_mv, cursor, sample_idx, loss)

        # one delimiter terminates the whole (multi-building) item
        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)  # item delimiter

    return cursor
