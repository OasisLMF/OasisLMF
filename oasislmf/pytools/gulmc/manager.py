"""Ground-up loss Monte Carlo (gulmc) manager.

Jagged Array Naming Convention
------------------------------
    <key_name>_ja_id_ind    — optional sparse ID → dense index (id_index.py)
    <key_name>_ja_offsets   — row boundaries: row i spans [offsets[i], offsets[i+1])
    <key_name>_ja_<values>  — one or more parallel flat arrays holding payload data

    Two-level (nested) jagged arrays repeat the pattern on the payload:
    <key_name>_ja_<inner_key>_ja_offsets  — L2 row boundaries
    <key_name>_ja_<inner_key>_ja_<values> — L2 payload data
"""
import atexit
import logging
import os
import sys
import json
from contextlib import ExitStack

import numpy as np
import numba as nb
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import nb_areaperil_int, oasis_float, nb_oasis_int, oasis_int, correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import (PIPE_CAPACITY, ITEM_PACKED_STREAM, ITEM_STREAM, LOSS_STREAM_ID,
                                                  check_packed_item_fits, check_packing_supported,
                                                  max_emitted_blocks, stream_info_to_bytes, write_mv_to_stream)
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.gul.common import MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX, NUM_IDX
from oasislmf.pytools.gul.core import (compute_mean_loss, get_gul, accumulate_hermite_coeffs_into, loss_correlation, HERMITE_TERMS)
from oasislmf.pytools.gul.manager import write_losses, adjust_byte_mv_size
from oasislmf.pytools.gul.random import (generate_correlated_hash_vector, generate_hash,
                                         generate_hash_hazard, get_corr_rval, get_random_generator,
                                         _lh_philox_block, _lh_philox_partial_block, PHILOX_U32_MASK, PHILOX_SHIFT32)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.common.damage_cdf import (add_cdf_row, add_member_cdfs, add_no_loss_mass, cdf_length, convolve_damage_cdf,
                                                pdf_to_cdf)
from oasislmf.pytools.gulmc.items import item_is_aggregate_vuln, item_is_computed
from oasislmf.pytools.gulmc.structure import NO_PLACEMENT_SAMPLE_SIZE
from oasislmf.pytools.gulmc.common import (DAMAGE_TYPE_ABSOLUTE,
                                           DAMAGE_TYPE_DURATION,
                                           DAMAGE_TYPE_RELATIVE,
                                           NP_BASE_ARRAY_SIZE,
                                           NormInversionParameters,
                                           gulSampleslevelHeader_size,
                                           gulSampleslevelRec_size,
                                           haz_arr_type, items_MC_data_type,
                                           gulmc_compute_info_type)
from oasislmf.pytools.common.id_index import get_idx as id_index_get_idx, NOT_FOUND as ID_INDEX_NOT_FOUND
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.ping import EventProgress
from oasislmf.utils.exceptions import OasisException

logger = logging.getLogger(__name__)


CDF_CACHE_EMPTY = nb_int64(-1)
NO_RNG_INDEX = nb_int64(-1)


def validate_coverage_dependency(items, item_is_aggregate, vuln_idx_to_cond_idx):
    """Validate a coverage-dependency configuration against the loaded model data (fail-loud).

    A dependent item (``source_item_id > 0``) is driven by its source item's sampled damage bin
    through a conditional (damage-transition) vulnerability. This checks the invariants the gulmc
    kernel relies on, raising a clear error rather than silently producing wrong losses:

    1. a dependent must not use an aggregate vulnerability (the aggregate assembly path is not
       wired for the damage-bin-indexed conditional matrix);
    2. a dependent must use a conditional vulnerability (present in conditional_vulnerability, i.e.
       ``vuln_idx_to_cond_idx[vulnerability_idx] >= 0``);
    3. an item with no source must NOT use a conditional vulnerability, since there is no source
       damage bin to index it with and the footprint hazard cannot sample it. An item with no
       source and a hazard-indexed vulnerability is simply computed independently — file
       generation leaves such items unpaired deliberately, because only the model's static data
       says which vulnerabilities are conditional.

    The source's sampled damage bin is captured directly during sampling (not re-derived from a
    ratio), so a source coverage may use any damage type (relative / absolute / duration).

    Args:
        items (np.ndarray): items table (coverage_id, vulnerability_id, vulnerability_idx,
            source_item_id).
        item_is_aggregate (np.ndarray[bool]): per item, whether its vulnerability is aggregate.
        vuln_idx_to_cond_idx (np.ndarray): dense vuln idx -> conditional row, or -1 if not conditional.

    Raises:
        OasisException: if any of the three invariants is violated.
    """
    non_agg = ~item_is_aggregate
    # per item: an item is dependent only if it resolved to a source item. A coverage may hold a
    # mix — the key server can place one peril in the same cell as the source and another not.
    is_dependent_item = items['source_item_id'] > 0

    dependent_aggregate = is_dependent_item & ~non_agg
    if np.any(dependent_aggregate):
        bad = np.unique(items['coverage_id'][dependent_aggregate])
        raise OasisException(
            f"coverage dependency: dependent coverage id(s) {bad.tolist()} use an aggregate vulnerability, "
            "which is not supported for a dependent coverage. Use a single conditional vulnerability."
        )

    # membership only makes sense for non-aggregate items (aggregate items carry no single
    # vulnerability_idx): a dependent must use a conditional vuln, an independent a normal one.
    item_is_conditional = np.zeros(len(items), dtype=bool)
    item_is_conditional[non_agg] = vuln_idx_to_cond_idx[items['vulnerability_idx'][non_agg]] >= 0

    dependent_without_conditional = is_dependent_item & non_agg & ~item_is_conditional
    if np.any(dependent_without_conditional):
        bad = np.unique(items['vulnerability_id'][dependent_without_conditional])
        raise OasisException(
            f"coverage dependency: dependent coverage(s) use vulnerability id(s) {bad.tolist()} that are "
            "not in the conditional_vulnerability file; a dependent coverage must use a conditional "
            "(damage-transition) vulnerability. Add it to conditional_vulnerability, or remove the dependency."
        )

    independent_with_conditional = ~is_dependent_item & non_agg & item_is_conditional
    if np.any(independent_with_conditional):
        bad = np.unique(items['vulnerability_id'][independent_with_conditional])
        raise OasisException(
            f"coverage dependency: item(s) with conditional vulnerability id(s) {bad.tolist()} have no "
            "source item at the same areaperil, so they would be computed independently, but a conditional "
            "vulnerability cannot be sampled by the footprint hazard. Provide a source at the same areaperil "
            "(check the key server's areaperil for both coverage types), or give these items a "
            "hazard-indexed vulnerability."
        )


@redirect_logging(exec_name='gulmc')
def mixed_cdf_cache_slots(n_mixed_cdf_groups, budget_slots):
    """Size the mixed cdf cache, in slots of one cdf each: a power of two for masking.

    The cache is cleared every event, since a cdf depends on the event's hazard, and an event
    writes each mixed cdf group at most once while nothing is evicted, so it never needs more slots
    than there are groups. Past ``budget_slots`` it is rounded down to fit: a write is read before
    the next one, so even a single slot gives the right cdfs, only recomputing more.

    Args:
        n_mixed_cdf_groups (int): number of mixed cdf groups, the cache's keys.
        budget_slots (int): slots that fit ``max_cached_vuln_cdf_size_MB``.

    Returns:
        int: the number of slots.
    """
    ideal = 1 << max(n_mixed_cdf_groups - 1, 0).bit_length()
    budget = 1 << (max(budget_slots, 1).bit_length() - 1)
    return min(ideal, budget)


def run(run_dir,
        ignore_file_type,
        sample_size,
        loss_threshold,
        alloc_rule,
        debug,
        random_generator,
        peril_filter=[],
        file_in=None,
        file_out=None,
        data_server=None,
        ignore_correlation=False,
        ignore_haz_correlation=False,
        effective_damageability=False,
        max_cached_vuln_cdf_size_MB=200,
        model_df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
        dynamic_footprint=False,
        section_merge=None,
        **kwargs):
    """Execute the main gulmc workflow.

    Args:
        run_dir (str): the directory of where the process is running
        ignore_file_type (set(str)): file extension to ignore when loading
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        alloc_rule (int): back-allocation rule.
        debug (int): for each random sample, print to the output stream the random loss (if 0), the random value used to draw
          the hazard intensity sample (if 1), the random value used to draw the damage sample (if 2). Defaults to 0.
        random_generator (int): random generator function id.
        peril_filter (list[int], optional): list of perils to include in the computation (if None, all perils will be included). Defaults to [].
        file_in (str, optional): filename of input stream. Defaults to None.
        file_out (str, optional): filename of output stream. Defaults to None.
        data_server (bool, optional): if True, run the data server. Defaults to None.
        ignore_correlation (bool, optional): if True, do not compute correlated random samples. Defaults to False.
        ignore_haz_correlation (bool, optional): if True, do not compute correlated hazard intensity samples. Defaults to False.
        effective_damageability (bool, optional): if True, it uses effective damageability to draw damage samples instead of
          using the full monte carlo approach (i.e., to draw hazard intensity first, then damage).
        max_cached_vuln_cdf_size_MB (int, optional): size in MB of the in-memory cache that stores and reuses the
          effective damage cdfs of the mixed cdf groups. Defaults to 200.
        model_df_engine (str, optional): The engine to use when loading model dataframes. Defaults to OasisPandasReader.
        dynamic_footprint (bool, optional): if True, load the dynamic footprint data and adjust hazard intensities at
          runtime. Defaults to False.
        section_merge (str, optional): how a dynamic footprint merges an areaperil's intensities across the sections of
          one event (see Footprint.load). Defaults to None, the footprint's default.
        **kwargs: additional keyword arguments. socket_server (str) enables the progress pings (a `gul_ready` ping once
          the model is loaded, then `events_complete` counts) and, when numeric, gives the port to override; analysis_pk
          is reported with each ping.

    Raises:
        ValueError: if alloc_rule is not 0, 1, 2, or 3.
        ValueError: if alloc_rule is 1, 2, or 3 when debug is 1 or 2.

    Returns:
        int: 0 if no errors occurred.
    """
    logger.info("starting gulmc")

    ignore_file_type = set(ignore_file_type)

    if alloc_rule not in [0, 1, 2, 3]:
        raise ValueError(f"Expect alloc_rule to be 0, 1, 2, or 3, got {alloc_rule}")

    if debug > 0 and alloc_rule != 0:
        raise ValueError(f"Expect alloc_rule to be 0 if debug is 1 or 2, got {alloc_rule}")

    if data_server:
        logger.debug("data server active")
        FootprintLayerClient.register()
        logger.debug("registered with data server")
        atexit.register(FootprintLayerClient.unregister)
    else:
        logger.debug("data server not active")

    with ExitStack() as stack:
        if file_in is None:
            streams_in = sys.stdin.buffer
        else:
            streams_in = stack.enter_context(open(file_in, 'rb'))

        event_id_mv = memoryview(bytearray(4))
        event_ids = np.ndarray(1, buffer=event_id_mv, dtype='i4')

        # --- load or build read-only structures --------------------------------
        from oasislmf.pytools.gulmc.structure import (
            gulmc_structure_exists, load_gulmc_structure, build_structures, dynamic_footprint_path,
            build_placements,
        )
        # the dynamic footprint's hazard is memory-mapped when create_gulmc_structure saved it
        footprint_shared_dir = None
        if gulmc_structure_exists(run_dir):
            footprint_shared_dir = dynamic_footprint_path(run_dir)
            logger.info("loading pre-computed gulmc structures (shared memory)")
            structures = load_gulmc_structure(run_dir)
        else:
            logger.info("building gulmc structures from input files")
            structures = build_structures(run_dir, ignore_file_type, peril_filter,
                                          dynamic_footprint, model_df_engine, sample_size=sample_size)
        if structures['placement_sample_size'] not in (NO_PLACEMENT_SAMPLE_SIZE, sample_size):
            # placements are drawn per sample when the structure is built; a run with another
            # sample size redraws them rather than read past, or short of, each item's run
            logger.info(f"redrawing placements for {sample_size} samples "
                        f"(the structure drew them for {structures['placement_sample_size']})")
            structures.update(build_placements(
                os.path.join(run_dir, 'input'), structures['items'],
                structures['blended_areaperil_placement_ja_offsets'],
                structures['blended_areaperil_placement_ja_cum_weights'], None, sample_size))

        items = structures['items']
        # coverages needs a writable copy: reconstruct_coverages writes cur_items / start_items per event
        coverages = structures['coverages'].copy()
        item_map_ja_areaperil_ids = structures['item_map_ja_areaperil_ids']
        item_map_ja_offsets = structures['item_map_ja_offsets']
        item_map_ja_mixed_cdf_group_idxs = structures['item_map_ja_mixed_cdf_group_idxs']
        item_map_ja_mixed_cdf_group_areaperil_idxs = structures['item_map_ja_mixed_cdf_group_areaperil_idxs']
        item_map_ja_id_ind = structures['item_map_ja_id_ind']
        item_mixed_cdf_groups = structures['item_mixed_cdf_groups']
        mixed_cdf_group_items_ja_offsets = structures['mixed_cdf_group_items_ja_offsets']
        mixed_cdf_group_items_ja_item_idxs = structures['mixed_cdf_group_items_ja_item_idxs']
        mixed_cdf_group_areaperil_ja_offsets = structures['mixed_cdf_group_areaperil_ja_offsets']
        mixed_cdf_group_areaperil_ja_cell_groups = structures['mixed_cdf_group_areaperil_ja_cell_groups']
        mixed_cdf_group_areaperil_ja_member_blocks = structures['mixed_cdf_group_areaperil_ja_member_blocks']
        placement_item_ind = structures['placement_item_ind']
        blended_areaperil_placement_ja_offsets = structures['blended_areaperil_placement_ja_offsets']
        blended_areaperil_placement_ja_areaperil_j = structures['blended_areaperil_placement_ja_areaperil_j']
        blended_areaperil_ja_placement_offsets = structures['blended_areaperil_ja_placement_offsets']
        blended_areaperil_ja_placements = structures['blended_areaperil_ja_placements']
        placement_item_locations = structures['placement_item_locations']
        placement_item_code_starts = structures['placement_item_code_starts']
        placement_location_ja_offsets = structures['placement_location_ja_offsets']
        placement_location_ja_placements = structures['placement_location_ja_placements']
        placement_location_ja_code_offsets = structures['placement_location_ja_code_offsets']
        placement_location_ja_codes = structures['placement_location_ja_codes']
        blended_areaperil_ja_offsets = structures['blended_areaperil_ja_offsets']
        blended_areaperil_ja_probabilities = structures['blended_areaperil_ja_probabilities']
        # the most areaperils one item reads, sizing the per-areaperil scratch; at least 1
        max_areaperils_per_item = int(np.diff(mixed_cdf_group_areaperil_ja_offsets).max(initial=1))
        vuln_member_ja_offsets = structures['vuln_member_ja_offsets']
        vuln_member_ja_rows = structures['vuln_member_ja_rows']
        vuln_member_ja_weights = structures['vuln_member_ja_weights']
        damage_bins = structures['damage_bins']
        vuln_adj = structures['vuln_adj']
        vuln_cdf = structures['vuln_cdf']
        vuln_cdf_lengths = structures['vuln_cdf_lengths']
        cond_cdf = structures['cond_cdf']
        cond_cdf_lengths = structures['cond_cdf_lengths']
        vuln_idx_to_cond_idx = structures['vuln_idx_to_cond_idx']
        unique_peril_correlation_groups = structures['unique_peril_correlation_groups']
        norm_inv_cdf = structures['norm_inv_cdf']
        norm_cdf = structures['norm_cdf']
        norm_inv_parameters = structures['norm_inv_parameters']
        intensity_bin_peril_ids = structures['intensity_bin_peril_ids']
        intensity_bins = structures['intensity_bins']
        coverage_source_id = structures['coverage_source_id']
        coverage_dependents_ja_offsets = structures['coverage_dependents_ja_offsets']
        coverage_dependents_ja_data = structures['coverage_dependents_ja_data']
        source_item_idx = structures['source_item_idx']
        n_unique_groups = structures['n_unique_groups']
        n_unique_haz_groups = structures['n_unique_haz_groups']
        del structures

        do_coverage_dependency = bool(coverage_dependents_ja_data.shape[0] > 0)

        # gate on the file as well as the forest: if every dependent ends up unpaired the forest is
        # empty, but an item still carrying a conditional vuln has to be caught
        if do_coverage_dependency or cond_cdf.shape[0] > 0:
            if do_coverage_dependency:
                logger.info(f"coverage dependency: switched ON ({coverage_dependents_ja_data.shape[0]} dependent coverages).")
            # items with no areaperil (every placement uncovered) are never computed
            computed = item_is_computed(item_mixed_cdf_groups, mixed_cdf_group_areaperil_ja_offsets)
            validate_coverage_dependency(
                items[computed],
                item_is_aggregate_vuln(item_mixed_cdf_groups, mixed_cdf_group_areaperil_ja_offsets,
                                       mixed_cdf_group_areaperil_ja_member_blocks >= vuln_cdf.shape[0])[computed],
                vuln_idx_to_cond_idx)

        Ndamage_bins_max = vuln_cdf.shape[2]
        Nperil_correlation_groups = unique_peril_correlation_groups.shape[0]
        logger.info(f"Detected {Nperil_correlation_groups} peril correlation groups.")

        if debug:
            # debug writes the random values drawn, one per building; summed at source they would
            # add up to anywhere in [0, N] instead of [0, 1], so every item keeps its buildings separate
            # (a copy: the structure is memory-mapped read-only)
            items = items.copy()
            items['packed_buildings'] = -np.abs(items['packed_buildings'])
        # building-packing: a single stream item per (loc, peril, coverage_type) multiplexes its
        # N buildings into the sample dimension (see write_losses / encode_sidx). Packing is
        # enabled only when some item carries more than one building. alloc_rule caps a coverage's
        # item losses at its TIV; write_losses applies that cap within each building block,
        # the coverage TIV being the per-building share.
        # signed field: take the magnitude, or the keep-separate items (negative) are skipped.
        # It is read off the item's COVERAGE, so a coverage's items cannot disagree on it -- the
        # alloc-rule cap pairs them by building index and relies on that.
        max_buildings = int(np.abs(items['packed_buildings']).max()) if items.shape[0] > 0 else 1
        # only kept-separate items meet either stream ceiling: a summed one writes a single
        # block at sidx 1..S however many buildings it carries
        check_packed_item_fits(max_emitted_blocks(items['packed_buildings']), sample_size)
        check_packing_supported(random_generator, items['packed_buildings'])
        if max_buildings > 1:
            logger.info(f"building-packing ENABLED: up to {max_buildings} buildings packed per item.")

        # import array to store the coverages to be computed
        # coverages are numbered from 1, therefore skip element 0.
        compute = np.zeros(coverages.shape[0] + 1, items_dtype['coverage_id'])

        # coverage dependency scratch/state (all trivial when the feature is off)
        compute_depth = np.zeros(coverages.shape[0] + 1, dtype=np.int32)
        compute_footprint_order = np.zeros(coverages.shape[0] + 1, dtype=items_dtype['coverage_id'])
        # DFS work stack for reordering coverages into (root -> subtree) order: (coverage_id, depth)
        dependency_dfs_stack = np.zeros((coverages.shape[0] + 1, 2), dtype=np.int64)
        # scratch: each present item's position within its coverage, rewritten every event
        item_idx_to_item_j = np.zeros(items.shape[0], dtype=oasis_int)
        # longest dependency chain: sizes the per-depth parent-result stacks
        max_dependency_depth = compute_max_dependency_depth(coverage_source_id) if do_coverage_dependency else 0
        # a coverage needs its sampled damage bins kept only if something below it will read them
        coverage_has_dependents = np.zeros(coverages.shape[0] + 1, dtype=np.int8)
        if do_coverage_dependency:
            coverage_has_dependents[:coverage_dependents_ja_offsets.shape[0] - 1] = (
                np.diff(coverage_dependents_ja_offsets) > 0)
        # Only coverages in the dependency forest touch the stack, so it is sized to the buildings
        # they actually carry rather than to max_buildings -- one outsized location elsewhere in
        # the portfolio would otherwise set the width for the whole run. Dependents are included,
        # not just sources: a dependent reads its source's slice at its OWN building offset, so a
        # dependent packing more buildings than its source would read past a source-only bound.
        if do_coverage_dependency and items.shape[0] > 0:
            in_forest = (coverage_has_dependents[items['coverage_id']] == 1) | (
                coverage_source_id[items['coverage_id']] != 0)
            max_source_buildings = int(np.abs(items['packed_buildings'][in_forest]).max()) if in_forest.any() else 1
        else:
            max_source_buildings = 1
        # per-depth stacks holding a source's result while its subtree is computed, indexed
        # [depth, item_j] so a dependent reads its source's matching peril. The bins are per
        # sample and so per building: laid out as (b - 1) * S + (s - 1), the same encoding the
        # stream uses for a packed sidx.
        max_items_per_coverage = int(np.max(coverages[1:]['max_items']))
        source_damage_bin_stack = np.zeros(
            (max_dependency_depth + 1, max_items_per_coverage,
             max_source_buildings * (sample_size if sample_size > 0 else 1)), dtype=np.int32)
        source_eff_damage_cdf_stack = np.zeros(
            (max_dependency_depth + 1, max_items_per_coverage, Ndamage_bins_max), dtype=oasis_float)
        source_eff_damage_cdf_len_stack = np.zeros((max_dependency_depth + 1, max_items_per_coverage), dtype=np.int64)

        model_storage = get_storage_from_config_path(
            os.path.join(run_dir, 'model_storage.json'),
            os.path.join(run_dir, 'static'),
        )
        logger.debug('import footprint')
        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
            footprint_obj = None
        else:
            footprint_obj = stack.enter_context(Footprint.load(model_storage, ignore_file_type,
                                                df_engine=model_df_engine, areaperil_ids=item_map_ja_areaperil_ids,
                                                section_merge=section_merge,
                                                shared_dir=footprint_shared_dir))
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        # set up streams
        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        # prepare output stream. The aggregation type says whether a reader may meet a packed
        # sidx, and only a kept-separate item emits one -- a run that packs nothing, or packs only
        # items summed at source, declares ITEM_STREAM and is byte-for-byte pre-packing.
        stream_out.write(stream_info_to_bytes(
            LOSS_STREAM_ID,
            ITEM_PACKED_STREAM if max_emitted_blocks(items['packed_buildings']) > 1 else ITEM_STREAM))
        stream_out.write(np.int32(sample_size).tobytes())

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)
        # create the array to store the seeds
        haz_seeds = np.zeros(n_unique_haz_groups, dtype=correlations_dtype['hazard_group_id'])
        vuln_seeds = np.zeros(n_unique_groups, dtype=items_dtype['group_id'])
        # Pre-allocated arrays for group_id -> rng_index mapping. Initialised once; reconstruct_coverages
        # restores the entries it used, so they are never reset in full (their size is that of the
        # portfolio, not of the event).
        group_seq_rng_index = np.full(n_unique_groups, NO_RNG_INDEX, dtype=np.int64)
        hazard_group_seq_rng_index = np.full(n_unique_haz_groups, NO_RNG_INDEX, dtype=np.int64)
        # haz correlation
        if not ignore_haz_correlation and Nperil_correlation_groups > 0 and any(items['hazard_correlation_value'] > 0):
            # there will be some hazard correlation
            do_haz_correlation = True
            haz_peril_correlation_groups = unique_peril_correlation_groups
            haz_corr_seeds = np.zeros(np.max(haz_peril_correlation_groups) + 1, dtype='int64')
        else:
            do_haz_correlation = False
            haz_peril_correlation_groups = unique_peril_correlation_groups[:0]
            haz_corr_seeds = np.zeros(1, dtype='int64')
            if ignore_haz_correlation:
                logger.info(
                    "correlated random number generation for hazard intensity sampling: switched OFF because --ignore-haz-correlation is True.")
            else:
                logger.info("correlated random number generation for hazard intensity sampling: switched OFF because 0 peril correlation groups were detected or "
                            "the hazard correlation value is zero for all peril correlation groups.")

        # damage correlation
        if not ignore_correlation and Nperil_correlation_groups > 0 and any(items['damage_correlation_value'] > 0):
            do_correlation = True
            damage_peril_correlation_groups = unique_peril_correlation_groups
            damage_corr_seeds = np.zeros(np.max(damage_peril_correlation_groups) + 1, dtype='int64')
        else:
            do_correlation = False
            damage_peril_correlation_groups = unique_peril_correlation_groups[:0]
            damage_corr_seeds = np.zeros(1, dtype='int64')
            if ignore_correlation:
                logger.info(
                    "correlated random number generation for damage sampling: switched OFF because --ignore-correlation is True.")
            else:
                logger.info("correlated random number generation for damage sampling: switched OFF because 0 peril correlation groups were detected or "
                            "the damage correlation value is zero for all peril correlation groups.")

        if do_correlation or do_haz_correlation:
            logger.info(f"correlated random number generation for hazard intensity sampling: switched {'ON' if do_haz_correlation else 'OFF'}.")
            logger.info(f"Correlated random number generation for damage sampling: switched  {'ON' if do_correlation else 'OFF'}.")
            logger.info(f"Correlation values for {Nperil_correlation_groups} peril correlation groups have been imported.")
            # norm_inv_parameters, norm_inv_cdf, norm_cdf are pre-computed in structures
        else:
            # override with dummy data structures for correct numba compilation when correlation is OFF
            norm_inv_parameters = np.array((0., 0., 0, 0., 0., 0., 0.), dtype=NormInversionParameters)
            norm_inv_cdf, norm_cdf = np.zeros(1, dtype='float64'), np.zeros(1, dtype='float64')

        # buffer to be re-used to store all the correlated random values
        vuln_z_unif = np.zeros(sample_size, dtype='float64')
        haz_z_unif = np.zeros(sample_size, dtype='float64')

        if effective_damageability is True:
            logger.info("effective_damageability is True: gulmc will draw the damage samples from the effective damageability distribution.")
        else:
            logger.info("effective_damageability is False: gulmc will perform the full Monte Carlo sampling: "
                        "sample the hazard intensity first, then sample the damage from the corresponding vulnerability function.")

        # create buffers to be reused when computing losses
        byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')
        max_items_per_coverage = int(np.max(coverages[1:]['max_items']))
        losses = np.zeros((sample_size + NUM_IDX + 1, max_items_per_coverage), dtype=oasis_float)

        # Generator 2 is counter-based: a building's block is a pure function of the group key
        # and the building index, so it can be produced where it is consumed instead of the
        # whole event's draws being materialised up front. These hold one building's worth.
        vuln_draw_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        haz_draw_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        perm_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        # Per item of the current coverage, the correlation between two of its buildings'
        # LOSSES -- what the variance of their sum actually needs. 0 unless the item is summed
        # and correlated, which is the only case that reads it.
        loss_correlation_by_item = np.zeros(max_items_per_coverage, dtype=oasis_float)
        hermite_coeffs = np.zeros(HERMITE_TERMS, dtype='float64')

        # maximum bytes to be written in the output stream for 1 item. A kept-separate item emits
        # one block of that per building; a summed one emits a single block whatever it carries.
        # One block: the item header, this building's NUM_IDX specials and S samples, and the
        # delimiter. Only the first block of an item carries the header and only the last the
        # delimiter, so charging every block for both is a deliberate over-estimate.
        max_bytes_per_block = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size
        max_bytes_per_item = max_bytes_per_block * max_emitted_blocks(items['packed_buildings'])

        # Size the mixed cdf cache: the effective damage cdfs of the mixed cdf groups, an areaperil's own
        # cdf being its size-1 group's. max_cached_vuln_cdf_size_MB caps it.
        cdf_size_bytes = Ndamage_bins_max * oasis_float.itemsize
        budget_slots = int(max_cached_vuln_cdf_size_MB * 1024 * 1024 // cdf_size_bytes)
        n_mixed_cdf_groups = mixed_cdf_group_items_ja_offsets.shape[0] - 1
        n_mixed_cdf_slots = mixed_cdf_cache_slots(n_mixed_cdf_groups, budget_slots)
        logger.info(f"max vulnerability cdf cache size is {max_cached_vuln_cdf_size_MB}MB")
        logger.info(f"generating a mixed cdf cache of {n_mixed_cdf_slots} slots for {n_mixed_cdf_groups} mixed cdf groups, "
                    f"{n_mixed_cdf_slots * cdf_size_bytes / 1024 / 1024:.3f}MB")

        # maximum bytes to be written in the output stream for 1 item
        event_footprint_obj = FootprintLayerClient if data_server else footprint_obj

        # intensity_bin_peril_ids and intensity_bins are pre-loaded from structures
        if not dynamic_footprint:
            dynamic_footprint = None

        compute_info = np.zeros(1, dtype=gulmc_compute_info_type)[0]

        compute_info['max_bytes_per_item'] = max_bytes_per_item
        compute_info['sample_size'] = sample_size
        compute_info['Ndamage_bins_max'] = Ndamage_bins_max
        compute_info['loss_threshold'] = loss_threshold
        compute_info['alloc_rule'] = alloc_rule
        compute_info['do_correlation'] = do_correlation
        compute_info['do_haz_correlation'] = do_haz_correlation
        compute_info['effective_damageability'] = effective_damageability
        compute_info['debug'] = debug
        compute_info['do_coverage_dependency'] = do_coverage_dependency

        # Under full Monte Carlo, an item of several placements computes the samples
        # prepare_event_samples lists: its live ones, or every one in a debug run.
        list_item_samples = (sample_size > 0 and not effective_damageability
                             and placement_item_locations.shape[0] > 0)
        # A coverage's sample losses, held whole until write_losses emits it (it reduces across
        # items at a fixed building, then sums the buildings): each item's are a run, one after
        # another (see write_losses). An item computed for every sample takes N * S, so without
        # sparse items the largest coverage sizes it; a sparse item takes its live samples only,
        # and prepare_event_samples gives the size each event needs.
        coverage_buildings = np.zeros(coverages.shape[0], dtype=np.int64)
        np.add.at(coverage_buildings, items['coverage_id'], np.abs(items['packed_buildings']).astype(np.int64))
        sample_losses = np.zeros(max(1, sample_size if list_item_samples else int(coverage_buildings.max()) * sample_size),
                                 dtype=oasis_float)
        del coverage_buildings

        # what an event without any sparse item passes
        event_sample_codes = np.zeros(0, dtype=np.int64)
        event_sample_areaperil_j = np.zeros(0, dtype=np.int32)
        event_sample_bins = np.zeros(0, dtype=np.int32)
        event_sample_vuln = np.zeros(0, dtype=np.float64)
        event_sample_haz = np.zeros(0, dtype=np.float64)

        # default random values array for sample_size==0 case
        haz_eps_ij = np.empty((1, sample_size), dtype='float64')
        damage_eps_ij = np.empty((1, sample_size), dtype='float64')

        # one row of sample_size per rng group; not built at all on generator 2, which draws
        # each building's block where it is used
        vuln_rndms_base = np.empty((1, 1), dtype='float64')
        haz_rndms_base = np.empty((1, 1), dtype='float64')

        # Pre-allocate per-event footprint arrays (reused across events, no per-event allocation)
        # fp_ap_inds stores dense areaperil indices (from item_map_ja_id_ind) rather than raw
        # areaperil_ids, so reconstruct_coverages can index item_map_ja_offsets directly without
        # repeating the id_index lookup that was already done in process_areaperils_in_footprint.
        n_max_areaperils = len(item_map_ja_areaperil_ids)
        fp_ap_inds = np.empty(n_max_areaperils, dtype=np.uint32)
        fp_event_rps = np.empty(n_max_areaperils, dtype=np.int32)
        fp_haz_arr_ptr = np.empty(n_max_areaperils + 1, dtype=np.int64)
        # Per group areaperil, this event's hazard row (-1 when its areaperil has no hazard this
        # event) and, under a dynamic footprint, its return period. Only the group areaperils of
        # groups present in the event are written, so they need no reset between events.
        n_mixed_cdf_group_areaperils = int(mixed_cdf_group_areaperil_ja_offsets[-1])
        mixed_cdf_group_areaperil_haz_arr_i = np.empty(n_mixed_cdf_group_areaperils, dtype=oasis_int)
        mixed_cdf_group_areaperil_event_rp = np.zeros(n_mixed_cdf_group_areaperils, dtype=np.int32)
        # per mixed cdf group, the event_tag of the last event it was reached in, and the scratch
        # reconstruct_coverages lists the groups reached in
        mixed_cdf_group_event_tag = np.full(n_mixed_cdf_groups, -1, dtype=np.int64)
        event_mixed_cdf_groups = np.empty(n_mixed_cdf_groups, dtype=np.int32)
        mixed_cdf_group_needs_haz_rng = np.zeros(n_mixed_cdf_groups, dtype=np.bool_)
        event_tag = 0

        # per-item event data, kept across events: reconstruct_coverages only grows it, and every
        # field it reads back is written for the event first
        items_event_data = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_MC_data_type)

        # Pre-allocate the cdf cache once (reused across events, no need to zero)
        mixed_cdf_cache_tag = np.full(n_mixed_cdf_groups, CDF_CACHE_EMPTY, dtype=np.int64)
        mixed_cdf_cache = np.empty((n_mixed_cdf_slots, Ndamage_bins_max), dtype=oasis_float)
        mixed_cdf_cache_nbins = np.zeros(n_mixed_cdf_slots, dtype=np.int32)
        # a group's Hermite terms are kept only when some item sums correlated buildings
        needs_hermite = do_correlation and np.any((items['packed_buildings'] > 1) & (items['damage_correlation_value'] > 0))
        mixed_cdf_group_hermite = np.zeros((n_mixed_cdf_groups if needs_hermite else 0, 2, HERMITE_TERMS), dtype=np.float64)

        # per-event coverage state starts at zero; reconstruct_coverages then only resets the
        # coverages touched by the previous event
        coverages['cur_items'].fill(0)

        progress = EventProgress(kwargs.get('socket_server', 'False'), kwargs.get("analysis_pk", None))
        progress.ready()
        while True:
            if not streams_in.readinto(event_id_mv):
                progress.finish()
                break

            # get the next event_id from the input stream
            compute_info['event_id'] = event_ids[0]
            event_footprint = event_footprint_obj.get_event(event_ids[0])

            if event_footprint is None:
                logger.info(f"event {event_ids[0]} SKIPPED - no footprint")
            else:
                Nhaz_arr_this_event, haz_pdf = process_areaperils_in_footprint(
                    event_footprint,
                    item_map_ja_id_ind,
                    dynamic_footprint,
                    fp_ap_inds,
                    fp_event_rps,
                    fp_haz_arr_ptr)
                if Nhaz_arr_this_event == 0:
                    # no items to be computed for this event
                    progress.event_done()
                    logger.info(f"event {event_ids[0]} SKIPPED - no items")
                    continue

                event_tag += 1
                items_event_data, rng_index, hazard_rng_index, byte_mv = reconstruct_coverages(
                    compute_info,
                    fp_ap_inds,
                    Nhaz_arr_this_event,
                    fp_haz_arr_ptr,
                    fp_event_rps,
                    item_map_ja_offsets,
                    item_map_ja_mixed_cdf_group_idxs,
                    item_map_ja_mixed_cdf_group_areaperil_idxs,
                    items,
                    mixed_cdf_group_areaperil_ja_offsets,
                    mixed_cdf_group_items_ja_offsets,
                    mixed_cdf_group_items_ja_item_idxs,
                    mixed_cdf_group_areaperil_haz_arr_i,
                    mixed_cdf_group_areaperil_event_rp,
                    mixed_cdf_group_event_tag,
                    event_tag,
                    event_mixed_cdf_groups,
                    mixed_cdf_group_needs_haz_rng,
                    coverages,
                    compute,
                    haz_seeds,
                    haz_peril_correlation_groups,
                    haz_corr_seeds,
                    vuln_seeds,
                    damage_peril_correlation_groups,
                    damage_corr_seeds,
                    dynamic_footprint,
                    byte_mv,
                    items_event_data,
                    group_seq_rng_index,
                    hazard_group_seq_rng_index,
                    coverage_source_id,
                    coverage_dependents_ja_offsets,
                    coverage_dependents_ja_data,
                    compute_depth,
                    compute_footprint_order,
                    dependency_dfs_stack,
                    source_item_idx,
                    item_idx_to_item_j
                )

                # since these are never used outside of a sample > 0 branch we can remove the need to
                # generate (and potentially allocate) the random values. As at 2.3.5 the sampling method
                # for random values accounts for 25% of the runtime of the losses step not including
                # the get_event despite having a sample size of 0.
                if sample_size > 0:
                    # One row of sample_size per rng GROUP, which is what a packed item's first
                    # building gets too: random_LatinHypercube_Philox7 is a loop over
                    # _lh_philox_block at building coordinate 0. So this row is shared by every
                    # item of the group, and only buildings 2..N are drawn per item below.
                    vuln_rndms_base = generate_rndm(vuln_seeds[:rng_index], sample_size)
                    haz_rndms_base = generate_rndm(haz_seeds[:hazard_rng_index], sample_size)
                    if hazard_rng_index > 0:
                        haz_eps_ij = generate_rndm(haz_corr_seeds, sample_size, skip_seeds=1)
                    damage_eps_ij = generate_rndm(damage_corr_seeds, sample_size, skip_seeds=1)

                if list_item_samples:
                    (event_sample_codes, event_sample_areaperil_j, event_sample_bins, event_sample_vuln,
                     event_sample_haz, event_sample_losses_size) = prepare_event_samples(
                        compute_info, coverages, compute, compute_depth, coverage_source_id,
                        items_event_data, items, item_mixed_cdf_groups,
                        fp_ap_inds, Nhaz_arr_this_event, item_map_ja_offsets, item_map_ja_mixed_cdf_group_areaperil_idxs,
                        mixed_cdf_group_areaperil_ja_offsets, mixed_cdf_group_areaperil_haz_arr_i,
                        mixed_cdf_group_areaperil_event_rp, dynamic_footprint,
                        blended_areaperil_ja_offsets, blended_areaperil_placement_ja_offsets,
                        blended_areaperil_placement_ja_areaperil_j,
                        blended_areaperil_ja_placement_offsets, blended_areaperil_ja_placements,
                        placement_item_ind, placement_item_locations, placement_item_code_starts,
                        placement_location_ja_offsets, placement_location_ja_placements,
                        placement_location_ja_code_offsets, placement_location_ja_codes,
                        sample_size, max_buildings, vuln_seeds, vuln_rndms_base, rng_index,
                        haz_seeds, haz_rndms_base, hazard_rng_index)
                    if event_sample_losses_size > sample_losses.shape[0]:
                        sample_losses = np.zeros(event_sample_losses_size, dtype=oasis_float)

                # Invalidate the cdf cache for the new event without touching mixed_cdf_cache_tag, whose
                # size is the number of mixed cdf groups of the portfolio: a cached cdf is valid only while
                # mixed_cdf_cache_ctr - tag <= n_mixed_cdf_slots, so advancing the counter by that many
                # invalidates every tag written for previous events.
                compute_info['mixed_cdf_cache_ctr'] += n_mixed_cdf_slots

                processing_done = False
                logger.info(f"event {event_ids[0]} STARTED")
                while not processing_done:
                    resume_point_before = int(compute_info['coverage_i'])
                    try:
                        processing_done = compute_event_losses(
                            compute_info,
                            coverages,
                            compute,
                            items_event_data,
                            items,
                            item_mixed_cdf_groups,
                            mixed_cdf_group_areaperil_ja_offsets,
                            mixed_cdf_group_areaperil_ja_cell_groups,
                            mixed_cdf_group_areaperil_ja_member_blocks,
                            mixed_cdf_group_areaperil_haz_arr_i,
                            mixed_cdf_group_areaperil_event_rp,
                            blended_areaperil_ja_offsets,
                            blended_areaperil_ja_probabilities,
                            mixed_cdf_cache_tag,
                            mixed_cdf_cache,
                            mixed_cdf_cache_nbins,
                            mixed_cdf_group_hermite,
                            max_areaperils_per_item,
                            sample_size,
                            haz_pdf,
                            fp_haz_arr_ptr,
                            vuln_cdf,
                            vuln_cdf_lengths,
                            cond_cdf,
                            cond_cdf_lengths,
                            vuln_member_ja_offsets,
                            vuln_member_ja_rows,
                            vuln_member_ja_weights,
                            damage_bins,
                            losses,
                            vuln_adj,
                            haz_eps_ij,
                            damage_eps_ij,
                            norm_inv_parameters,
                            norm_inv_cdf,
                            norm_cdf,
                            vuln_z_unif,
                            haz_z_unif,
                            byte_mv,
                            dynamic_footprint,
                            intensity_bin_peril_ids,
                            intensity_bins,
                            sample_losses,
                            loss_correlation_by_item,
                            hermite_coeffs,
                            vuln_rndms_base,
                            haz_rndms_base,
                            vuln_seeds,
                            haz_seeds,
                            vuln_draw_scratch,
                            haz_draw_scratch,
                            perm_scratch,
                            event_sample_codes,
                            event_sample_areaperil_j,
                            event_sample_bins,
                            event_sample_vuln,
                            event_sample_haz,
                            coverage_has_dependents,
                            compute_depth,
                            source_damage_bin_stack,
                            source_eff_damage_cdf_stack,
                            source_eff_damage_cdf_len_stack,
                        )
                    except Exception:
                        data = {
                            "event_id": event_ids[0]
                        }
                        with open("event_error.json", "w") as f:
                            json.dump(data, f, default=str)

                        logger.error(f"event id={event_ids[0]} failed in summary")
                        raise
                    # A call that asks to be resumed must have advanced the resume point. It
                    # only stops because the buffer is full, and the buffer is empty on entry, so
                    # if it stops at the same place it will keep stopping there -- an infinite
                    # loop with no error. Note it is the RESUME POINT that has to move, not the
                    # cursor: a resume that restarts an item re-emits the same blocks forever and
                    # writes plenty while never finishing. The sizing in reconstruct_coverages
                    # makes this unreachable today; the check turns a future violation of that
                    # into a failure rather than a hang.
                    if not processing_done and int(compute_info['coverage_i']) <= resume_point_before:
                        raise RuntimeError(
                            f"gulmc made no progress on event {compute_info['event_id']}: it asked "
                            f"to resume at coverage {int(compute_info['coverage_i'])}, no further "
                            f"on than the {resume_point_before} it started from, having written "
                            f"{compute_info['cursor']} bytes into a {byte_mv.shape[0]} byte "
                            f"buffer, which must hold the largest coverage whole.")
                    # write the losses to the output stream
                    write_mv_to_stream(stream_out, byte_mv, compute_info['cursor'])

                logger.info(f"event {event_ids[0]} DONE")

            progress.event_done()

    return 0


@nb.njit(cache=True)
def compute_max_dependency_depth(coverage_source_id):
    """Return the longest parent chain length in the coverage dependency forest.

    Walks the parent links (``coverage_source_id``) up from every coverage and returns the
    deepest chain found. Used to size the per-depth stacks that hold a source coverage's
    result while its dependent subtree is computed.

    Args:
        coverage_source_id (np.ndarray): parent coverage_id per coverage_id (0 = root).

    Returns:
        int: the maximum dependency depth (number of ancestors) over all coverages.
    """
    n_coverages = coverage_source_id.shape[0]
    max_depth = 0
    for coverage_id in range(n_coverages):
        depth = 0
        node = coverage_id
        # the depth guard is a defensive bound; the forest is validated acyclic at build time
        while coverage_source_id[node] != 0 and depth <= n_coverages:
            node = coverage_source_id[node]
            depth += 1
        if depth > max_depth:
            max_depth = depth
    return max_depth


@nb.njit(fastmath=True, cache=True)
def get_gul_from_vuln_cdf(vuln_rval, vuln_cdf, Ndamage_bins, damage_bins, bin_scaling,
                          src_bin_out, out_i):
    # find the damage cdf bin in which the random value `vuln_rval` falls into
    vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins - 1)

    # The sampled bin is the driving signal for coverage dependency, captured here rather than
    # re-derived from the loss. It goes to the caller's array rather than out with the loss: a
    # tuple has to be materialised across the call, and this is called once per sample.
    # src_bin_out is always a sample_size slice, so the store is unconditional and the caller
    # keeps no branch in its loop.
    src_bin_out[out_i] = vuln_bin_idx
    gul = get_gul(
        damage_bins['bin_from'][vuln_bin_idx],
        damage_bins['bin_to'][vuln_bin_idx],
        damage_bins['interpolation'][vuln_bin_idx],
        vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
        vuln_cdf[vuln_bin_idx],
        vuln_rval,
        bin_scaling,
    )
    return gul


@nb.njit(cache=True, fastmath=True, inline='always')
def compute_damage_bin_scaling(damage_bins, Neff_damage_bins, tiv):
    """Determine the tiv scaling factor to apply to gul values for a coverage.

    The scaling depends on the damage type recorded in the last effective damage bin:
    relative functions scale by tiv, absolute by 1, duration converts annual tiv to daily,
    and the default path infers relative-vs-absolute from whether the last bin_to is <= 1.

    A coverage with no tiv has no value at risk, so it yields no loss whatever the damage type.
    Relative and duration functions get that from their tiv factor, but absolute functions carry
    currency directly in the damage bins, so they need the zero explicitly (write_losses' tiv
    split, which caps absolute losses at the tiv, is likewise skipped when tiv is 0). This only
    arises for a coverage dependency source retained with zero tiv (an uninsured building driving
    its insured contents): it must drive its dependents but report no loss of its own. The sampled
    damage bin and the effective damage cdf that do the driving are independent of this scaling.

    Args:
        damage_bins (np.array): damage bin dictionary with bin_to and damage_type fields.
        Neff_damage_bins (int): number of effective damage bins.
        tiv (float): total insured value of the coverage.

    Returns:
        float: the scaling factor applied to damage-bin values to produce gul.
    """
    if tiv <= 0:
        return 0.

    damage_type = damage_bins[Neff_damage_bins - 1]['damage_type']
    if damage_type == DAMAGE_TYPE_RELATIVE:
        damage_bin_scaling = tiv
    elif damage_type == DAMAGE_TYPE_ABSOLUTE:
        damage_bin_scaling = 1
    elif damage_type == DAMAGE_TYPE_DURATION:
        # convert annual tiv to daily
        damage_bin_scaling = tiv / 365
    else:  # default behaviour
        # for relative vulnerability functions, gul are fraction of the tiv
        # for absolute vulnerability functions, gul are absolute values
        damage_bin_scaling = tiv if damage_bins[Neff_damage_bins - 1]['bin_to'] <= 1 else 1.0
    return damage_bin_scaling


@nb.njit(cache=True, fastmath=True, inline='always')
def merge_dynamic_haz_records_per_bin(haz_pdf, haz_start, Nhaz_records, item, intensity_adjustment,
                                      intensity_bin_peril_ids, intensity_bins, bin_slot, haz_bin_id_out, haz_prob_out):
    """Bin a dynamic footprint areaperil's records for one item, summing the probability of the
    records that fall in the same intensity bin.

    A dynamic footprint record holds a raw intensity, binned here because the item's
    intensity_adjustment shifts it first. An areaperil lying in several sections of an event can
    hold many more records than the model has intensity bins, but all the per-hazard-bin work that
    follows (vulnerability rows, their CDFs, the CDF cache, sampling) depends on the bin alone. So
    the hazard axis is the bins the records fall in, never longer than the number of intensity bins,
    which the per-hazard-bin buffers and the CDF cache are sized on.

    Entries are kept in the order of each bin's first record. Records come largest intensity first
    and intensity_bins does not decrease with intensity, so the records of a bin are adjacent and
    the hazard CDF keeps its bin boundaries: a random number picks the same bin as unmerged.

    Args:
        haz_pdf (np.array[haz_arr_type]): hazard pdf records of all areaperils for the event.
        haz_start (int): index of the areaperil's first record in ``haz_pdf``.
        Nhaz_records (int): number of records of the areaperil.
        item (np.void): the item record (uses peril_id).
        intensity_adjustment (int): intensity shift applied before binning.
        intensity_bin_peril_ids (np.array[int32]): sorted unique encoded peril_ids.
        intensity_bins (np.array[int32, 2d]): [peril_idx, intensity_value] -> intensity_bin_id.
        bin_slot (np.array[int64]): scratch indexed by bin id, -1 everywhere on entry and on return.
        haz_bin_id_out (np.array[int32]): receives the bin id of each entry.
        haz_prob_out (np.array[oasis_float]): receives the summed probability of each entry.

    Returns:
        int: the number of entries written.
    """
    peril_id = item['peril_id']
    # find peril index (linear scan of small array, typically 1-3 perils)
    peril_idx = nb_int32(0)
    for pi in range(intensity_bin_peril_ids.shape[0]):
        if intensity_bin_peril_ids[pi] == peril_id:
            peril_idx = nb_int32(pi)
            break
    max_intensity = nb_int32(intensity_bins.shape[1] - 1)
    n_entries = 0
    for record_i in range(haz_start, haz_start + Nhaz_records):
        intensity_val = haz_pdf[record_i]['intensity'] - intensity_adjustment
        if intensity_val < 0:
            intensity_val = nb_int32(0)
        elif intensity_val > max_intensity:
            # past the table, its last intensity stands for that much or more
            intensity_val = max_intensity
        bin_id = intensity_bins[peril_idx, intensity_val]
        entry = bin_slot[bin_id]
        if entry < 0:
            entry = n_entries
            bin_slot[bin_id] = entry
            haz_bin_id_out[entry] = bin_id
            haz_prob_out[entry] = 0.
            n_entries += 1
        haz_prob_out[entry] += haz_pdf[record_i]['probability']
    for entry in range(n_entries):
        bin_slot[haz_bin_id_out[entry]] = -1
    return n_entries


@nb.njit(cache=True, fastmath=True, inline='always')
def fill_haz_bins(haz_pdf, haz_start, Nhaz_records, item, intensity_adjustment, dynamic_footprint,
                  intensity_bin_peril_ids, intensity_bins, bin_slot, haz_bin_id, haz_pdf_prob):
    """Copy an item's hazard intensity bin ids and probabilities into the per-item scratch buffers.

    The item's records are ``haz_pdf[haz_start:haz_start + Nhaz_records]``, read by index and copied
    rather than sliced or field-viewed (see the refcount note in common/event_stream.py). With no
    dynamic footprint each record is one hazard entry, its bin id read from the hazard pdf. With a
    dynamic footprint the records are binned and merged per bin (merge_dynamic_haz_records_per_bin).

    Args:
        haz_pdf (np.array[haz_arr_type]): hazard pdf records of all areaperils for the event.
        haz_start (int): index of this item's first record in ``haz_pdf``.
        Nhaz_records (int): number of records for this item.
        item (np.void): the item record (needs peril_id when dynamic footprint is active).
        intensity_adjustment (int): intensity shift to apply under a dynamic footprint.
        dynamic_footprint (None or object): None if no dynamic footprint, otherwise truthy.
        intensity_bin_peril_ids (np.array[int32]): sorted unique encoded peril_ids.
        intensity_bins (np.array[int32, 2d]): [peril_idx, intensity_value] -> intensity_bin_id.
        bin_slot (np.array[int64]): scratch for merge_dynamic_haz_records_per_bin.
        haz_bin_id (np.array[int32]): output, intensity bin id per hazard entry.
        haz_pdf_prob (np.array[float]): output, probability per hazard entry.

    Returns:
        int: the number of hazard entries written.
    """
    if dynamic_footprint is not None:
        return merge_dynamic_haz_records_per_bin(haz_pdf, haz_start, Nhaz_records, item, intensity_adjustment,
                                                 intensity_bin_peril_ids, intensity_bins, bin_slot,
                                                 haz_bin_id, haz_pdf_prob)
    for haz_i in range(Nhaz_records):
        record = haz_pdf[haz_start + haz_i]
        haz_bin_id[haz_i] = record['intensity_bin_id']
        haz_pdf_prob[haz_i] = record['probability']
    return Nhaz_records


@nb.njit(cache=True, fastmath=True, inline='always')
def group_areaperil_is_live(mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp, mixed_cdf_group_areaperil_idx,
                            protection_rp,
                            dynamic_footprint):
    """Whether an item areaperil can take a loss this event.

    It cannot when its areaperil has no hazard this event, or, under a dynamic footprint, when the
    event's return period there is within the item's protection. The return period is read only for
    an areaperil with hazard this event, which is when reconstruct_coverages wrote it.
    """
    if mixed_cdf_group_areaperil_haz_arr_i[mixed_cdf_group_areaperil_idx] < 0:
        return False
    if dynamic_footprint is not None and protection_rp > 0 \
            and mixed_cdf_group_areaperil_event_rp[mixed_cdf_group_areaperil_idx] < protection_rp:
        return False
    return True


@nb.njit(cache=True, fastmath=True, inline='always')
def footprint_hazard(haz_pdf, haz_arr_ptr, haz_arr_i, item, intensity_adjustment, dynamic_footprint,
                     intensity_bin_peril_ids, intensity_bins, bin_slot, haz_bin_id_buf, haz_pdf_prob_buf,
                     haz_cdf_buf, haz_start):
    """Write an item areaperil's hazard this event into the caller's buffers and return its number of bins.

    The intensity bin ids and the hazard cdf go to ``haz_bin_id_buf`` and ``haz_cdf_buf`` from row
    ``haz_start``, the probabilities to the start of ``haz_pdf_prob_buf``. The records are copied
    rather than viewed: a few values per areaperil cost less than a view (see the
    numba-hot-loop-refcount skill). Only the first ``Nhaz_bins`` rows of each are valid.
    """
    record_start = haz_arr_ptr[haz_arr_i]
    n_entries = fill_haz_bins(haz_pdf, record_start, haz_arr_ptr[haz_arr_i + 1] - record_start, item, intensity_adjustment,
                              dynamic_footprint, intensity_bin_peril_ids, intensity_bins, bin_slot, haz_bin_id_buf[haz_start:],
                              haz_pdf_prob_buf)
    return pdf_to_cdf(haz_pdf_prob_buf, n_entries, haz_cdf_buf[haz_start:])


@nb.njit(cache=True, fastmath=True, inline='always')
def damage_bin_scaling_tiv_part(damage_bins, Neff_damage_bins):
    """Split ``compute_damage_bin_scaling`` into a factor and whether it multiplies the tiv.

    The scaling is ``factor * tiv`` when the second value is True, ``factor`` otherwise (and 0 for
    no tiv). A quantity linear in the scaling, cached for items of any tiv, keeps the two parts
    apart and combines them per item.
    """
    damage_type = damage_bins[Neff_damage_bins - 1]['damage_type']
    if damage_type == DAMAGE_TYPE_RELATIVE:
        return 1., True
    if damage_type == DAMAGE_TYPE_ABSOLUTE:
        return 1., False
    if damage_type == DAMAGE_TYPE_DURATION:
        return 1. / 365, True
    return 1., damage_bins[Neff_damage_bins - 1]['bin_to'] <= 1


@nb.njit(cache=True, fastmath=True, inline='always')
def draw_correlation_samples(compute_info, item, damage_rval_adj, hazard_rng_index, n_samples, sample_idxs,
                             haz_rndms_item, vuln_rndms_item, haz_eps_ij, damage_eps_ij,
                             norm_inv_parameters, norm_inv_cdf, norm_cdf, eps_buf,
                             haz_z_unif, vuln_z_unif):
    """Draw the (optionally correlated) hazard and damage random values of some samples of one item.

    Writes ``n_samples`` values into ``haz_z_unif`` and ``vuln_z_unif`` in place, value ``i`` being
    for sample ``sample_idxs[i]``. When correlation is enabled for the relevant dimension the values
    are drawn through the Gaussian copula (``get_corr_rval``) with the correlated values of those
    samples; otherwise the item's base random values are copied straight across. Hazard values are
    only drawn when ``hazard_rng_index >= 0`` (non-deterministic hazard under full Monte Carlo). The
    damage values are scaled by ``damage_rval_adj``.

    Args:
        compute_info (gulmc_compute_info_type): computation state (do_haz_correlation, do_correlation).
        item (np.void): the item record (correlation values, peril_correlation_group).
        damage_rval_adj (float): the adjustment of the item's vulnerability, 1 for an aggregate one.
        hazard_rng_index (int): < 0 when the hazard is deterministic, in which case no hazard
            values are drawn. Only its sign is used; the caller has already gathered the values.
        n_samples (int): number of samples drawn.
        sample_idxs (np.array[int]): the 0-based sample of each value, ``n_samples`` of them; 0..S-1
            when every sample is drawn.
        haz_rndms_item (np.array[float64]): the item's hazard base random values of those samples,
            in the same order. Under building packing they are one building's.
        vuln_rndms_item (np.array[float64]): the same for damage.
        haz_eps_ij (np.array[float]): correlated random values for hazard sampling, per sample.
        damage_eps_ij (np.array[float]): correlated random values for damage sampling, per sample.
        norm_inv_parameters (NormInversionParameters): parameters for Gaussian inversion.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf lookup table.
        norm_cdf (np.array[float]): Gaussian cdf lookup table.
        eps_buf (np.array[float64]): scratch of at least ``n_samples``, the correlated values of the
            samples drawn.
        haz_z_unif (np.array[float]): output buffer for hazard random values.
        vuln_z_unif (np.array[float]): output buffer for damage random values.
    """
    # hazard random values only exist (hazard_rng_index >= 0) when this areaperil's
    # hazard intensity is non-deterministic and we are running full Monte Carlo.
    if hazard_rng_index >= 0:
        if compute_info['do_haz_correlation'] and item['hazard_correlation_value'] > 0:
            # use correlation definitions to draw correlated random values into haz_z_unif
            group = item['peril_correlation_group']
            for sample_i in range(n_samples):
                eps_buf[sample_i] = haz_eps_ij[group, sample_idxs[sample_i]]
            get_corr_rval(
                eps_buf, haz_rndms_item, item['hazard_correlation_value'],
                norm_inv_parameters['x_min'], norm_inv_cdf, norm_inv_parameters['inv_factor'],
                norm_inv_parameters['cdf_min'], norm_cdf, norm_inv_parameters['norm_factor'],
                n_samples, haz_z_unif
            )
        else:
            for sample_i in range(n_samples):
                haz_z_unif[sample_i] = haz_rndms_item[sample_i]

    if compute_info['do_correlation'] and item['damage_correlation_value'] > 0:
        # use correlation definitions to draw correlated random values into vuln_z_unif
        group = item['peril_correlation_group']
        for sample_i in range(n_samples):
            eps_buf[sample_i] = damage_eps_ij[group, sample_idxs[sample_i]]
        get_corr_rval(
            eps_buf, vuln_rndms_item, item['damage_correlation_value'],
            norm_inv_parameters['x_min'], norm_inv_cdf, norm_inv_parameters['inv_factor'],
            norm_inv_parameters['cdf_min'], norm_cdf, norm_inv_parameters['norm_factor'],
            n_samples, vuln_z_unif
        )
    else:
        # do not use correlation
        for sample_i in range(n_samples):
            vuln_z_unif[sample_i] = vuln_rndms_item[sample_i]

    for sample_i in range(n_samples):
        vuln_z_unif[sample_i] *= damage_rval_adj


@nb.njit(cache=True, fastmath=True, inline='always')
def member_cdf_at(table, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights, block, condition, damage_bin):
    """A member block's damage cdf at one damage bin, under one condition: its members' cdfs by weight."""
    value = 0.
    for member in range(vuln_member_ja_offsets[block], vuln_member_ja_offsets[block + 1]):
        value += vuln_member_ja_weights[member] * table[vuln_member_ja_rows[member], condition, damage_bin]
    return value


@nb.njit(cache=True, fastmath=True, inline='always')
def accumulate_condition_hermite(scaling_factor, table, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                                 block, condition_ids, condition_probs, n_conditions, weight, damage_bins_interpolation,
                                 norm_inv_parameters, norm_inv_cdf, curve_buf, hermite_part, hermite_he_lo, hermite_he_hi):
    """Add the Hermite terms of E[loss | damage uniform] for one areaperil into ``hermite_part``.

    The condition (hazard bin, or source damage bin) is drawn independently of the damage uniform,
    so what the damage copula correlates is the probability-weighted mixture of the per-condition
    curves, ``weight`` times each.
    """
    for condition_i in range(n_conditions):
        probability = condition_probs[condition_i]
        if probability <= 0.:
            continue
        for damage_bin_i in range(table.shape[2]):
            curve_buf[damage_bin_i] = 0.
        add_member_cdfs(curve_buf, 1., table, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                        block, condition_ids[condition_i] - 1, table.shape[2])
        accumulate_hermite_coeffs_into(scaling_factor, curve_buf, damage_bins_interpolation, cdf_length(curve_buf, table.shape[2]),
                                       weight * probability, norm_inv_parameters['x_min'], norm_inv_parameters['inv_factor'],
                                       norm_inv_cdf, hermite_part, hermite_he_lo, hermite_he_hi)


@nb.njit(cache=True, fastmath=True, inline='always')
def cached_mixed_cdf_slot(compute_info, mixed_cdf_group, mixed_cdf_cache_tag, mixed_cdf_cache_mask):
    """The cache slot holding a mixed cdf group's cdf this event, or -1.

    A slot is overwritten only by the write that wraps round to it, so a cdf holds until the ring
    has taken as many writes after it as it has slots.
    """
    stored = mixed_cdf_cache_tag[mixed_cdf_group]
    if stored < 0 or compute_info['mixed_cdf_cache_ctr'] - stored > mixed_cdf_cache_mask + 1:
        return nb_int64(-1)
    return stored & mixed_cdf_cache_mask


@nb.njit(cache=True, fastmath=True, inline='always')
def store_mixed_cdf(compute_info, mixed_cdf_group, cdf, n_bins, mixed_cdf_cache_tag, mixed_cdf_cache, mixed_cdf_cache_nbins):
    """Write a mixed cdf group's cdf to the next slot of the ring and return the slot."""
    mixed_cdf_cache_mask = nb_int64(mixed_cdf_cache.shape[0] - 1)
    slot = compute_info['mixed_cdf_cache_ctr'] & mixed_cdf_cache_mask
    for damage_bin_i in range(n_bins):
        mixed_cdf_cache[slot, damage_bin_i] = cdf[damage_bin_i]
    mixed_cdf_cache_nbins[slot] = nb_int32(n_bins)
    mixed_cdf_cache_tag[mixed_cdf_group] = compute_info['mixed_cdf_cache_ctr']
    compute_info['mixed_cdf_cache_ctr'] += 1
    return slot


@nb.njit(cache=True, fastmath=True)
def areaperil_cdf_slot(compute_info, cell_group, haz_arr_i, member_block, item, intensity_adjustment,
                       haz_pdf, haz_arr_ptr, dynamic_footprint, intensity_bin_peril_ids, intensity_bins, bin_slot,
                       haz_bin_id_buf, haz_pdf_prob_buf, haz_cdf_buf,
                       vuln_cdf, vuln_cdf_lengths, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                       mixed_cdf_cache_tag, mixed_cdf_cache, mixed_cdf_cache_nbins, mixed_cdf_group_hermite,
                       damage_bins, norm_inv_parameters, norm_inv_cdf, cdf_acc, curve_buf, hermite_he_lo, hermite_he_hi):
    """Convolve an areaperil's own effective damage cdf into the cache and return its slot.

    The cdf is its size-1 mixed cdf group's (``cell_group``): the areaperil's hazard this event
    (``haz_arr_i``) weighting the damage cdfs of its vulnerability's members. It does not depend on
    protection, which only decides whether an item counts it. When Hermite terms are kept, the
    group's are computed with it, in the two parts the tiv does and does not multiply.

    Called on a cache miss only, so once per areaperil and vulnerability an event reaches: the caller
    checks the cache inline, which spares a hit the call and its array arguments. Not inlined.
    """
    n_haz_bins = footprint_hazard(haz_pdf, haz_arr_ptr, haz_arr_i, item, intensity_adjustment, dynamic_footprint,
                                  intensity_bin_peril_ids, intensity_bins, bin_slot, haz_bin_id_buf, haz_pdf_prob_buf,
                                  haz_cdf_buf, 0)
    n_bins = convolve_damage_cdf(cdf_acc, vuln_cdf, vuln_cdf_lengths, vuln_member_ja_offsets, vuln_member_ja_rows,
                                 vuln_member_ja_weights, member_block, haz_bin_id_buf, haz_pdf_prob_buf, n_haz_bins)
    if mixed_cdf_group_hermite.shape[0] > 0:
        hermite_per_tiv = mixed_cdf_group_hermite[cell_group, 0]
        hermite_fixed = mixed_cdf_group_hermite[cell_group, 1]
        for k in range(HERMITE_TERMS):
            hermite_per_tiv[k] = 0.
            hermite_fixed[k] = 0.
        scaling_factor, scales_with_tiv = damage_bin_scaling_tiv_part(damage_bins, n_bins)
        hermite_part = hermite_per_tiv if scales_with_tiv else hermite_fixed
        damage_bins_interpolation = damage_bins['interpolation']
        if compute_info['effective_damageability']:
            # one uniform drives the loss, so its cdf is the whole story
            accumulate_hermite_coeffs_into(scaling_factor, cdf_acc, damage_bins_interpolation, n_bins, 1.,
                                           norm_inv_parameters['x_min'], norm_inv_parameters['inv_factor'], norm_inv_cdf,
                                           hermite_part, hermite_he_lo, hermite_he_hi)
        else:
            accumulate_condition_hermite(scaling_factor, vuln_cdf, vuln_member_ja_offsets, vuln_member_ja_rows,
                                         vuln_member_ja_weights, member_block, haz_bin_id_buf, haz_pdf_prob_buf, n_haz_bins,
                                         1., damage_bins_interpolation, norm_inv_parameters, norm_inv_cdf, curve_buf,
                                         hermite_part, hermite_he_lo, hermite_he_hi)
    return store_mixed_cdf(compute_info, cell_group, cdf_acc, n_bins, mixed_cdf_cache_tag, mixed_cdf_cache,
                           mixed_cdf_cache_nbins)


@nb.njit(cache=True, fastmath=True, inline='always')
def fill_damage_row(row_cdf_buf, row_lengths, row, table, table_lengths, vuln_member_ja_offsets, vuln_member_ja_rows,
                    vuln_member_ja_weights, block, condition):
    """Write a member block's damage cdf under one condition into row ``row`` of the item's row buffer.

    A single vulnerability's is its table row; an aggregate's, its members' cdfs by weight. Only the
    first ``row_lengths[row]`` entries are valid.
    """
    n_bins = 0
    for member in range(vuln_member_ja_offsets[block], vuln_member_ja_offsets[block + 1]):
        n_bins = max(n_bins, table_lengths[vuln_member_ja_rows[member], condition])
    for damage_bin_i in range(n_bins):
        row_cdf_buf[row, damage_bin_i] = member_cdf_at(table, vuln_member_ja_offsets, vuln_member_ja_rows,
                                                       vuln_member_ja_weights, block, condition, damage_bin_i)
    row_lengths[row] = n_bins


@nb.njit(cache=True, fastmath=True, inline='always')
def resolve_sample_areaperils(n_samples, areaperil_j_of_sample, item_stamp, areaperil_j_stamp, areaperil_j_live,
                              areaperil_j_to_haz_start, areaperil_j_to_Nhaz_bins, areaperil_j_to_member_block,
                              group_start, mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                              mixed_cdf_group_areaperil_ja_member_blocks, protection_rp, dynamic_footprint,
                              haz_pdf, haz_arr_ptr, item, intensity_adjustment, intensity_bin_peril_ids, intensity_bins,
                              bin_slot, haz_bin_id_buf, haz_pdf_prob_buf, haz_cdf_buf, haz_used):
    """Resolve the areaperils the samples land on, each once per item; returns the hazard rows used.

    At most one per sample whatever the number of areaperils. A sample needs only the areaperil's
    hazard; its damage cdf row is written when first drawn (see ``sample_item_losses``).
    """
    for sample_i in range(n_samples):
        areaperil_j = areaperil_j_of_sample[sample_i]
        if areaperil_j < 0 or areaperil_j_stamp[areaperil_j] == item_stamp:
            continue
        areaperil_j_stamp[areaperil_j] = item_stamp
        group_areaperil_idx = group_start + areaperil_j
        live = group_areaperil_is_live(mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                                       group_areaperil_idx, protection_rp, dynamic_footprint)
        areaperil_j_live[areaperil_j] = live
        if not live:
            continue
        Nhaz_bins = footprint_hazard(
            haz_pdf, haz_arr_ptr, mixed_cdf_group_areaperil_haz_arr_i[group_areaperil_idx], item,
            intensity_adjustment, dynamic_footprint, intensity_bin_peril_ids, intensity_bins, bin_slot,
            haz_bin_id_buf, haz_pdf_prob_buf, haz_cdf_buf, haz_used)
        areaperil_j_to_haz_start[areaperil_j] = haz_used
        areaperil_j_to_Nhaz_bins[areaperil_j] = Nhaz_bins
        areaperil_j_to_member_block[areaperil_j] = mixed_cdf_group_areaperil_ja_member_blocks[group_areaperil_idx]
        haz_used += Nhaz_bins
    return haz_used


@nb.njit(cache=True, fastmath=True, inline='always')
def sample_item_losses(compute_info, sample_size, hazard_rng_index,
                       haz_z_unif, vuln_z_unif, areaperil_j_of_sample, areaperil_j_live,
                       areaperil_j_to_haz_start, areaperil_j_to_Nhaz_bins, areaperil_j_to_member_block,
                       haz_cdf_buf, haz_bin_id_buf, row_cdf_buf, row_lengths, row_stamp, item_stamp,
                       table, table_lengths, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                       mixed_eff_damage_cdf, n_mixed, damage_bin_scaling,
                       damage_bins, out, is_dependent, src_bin_out, parent_bins):
    """Write the per-sample gul (or debug random values) for one building of one item into ``out``.

    Under effective_damageability every sample is drawn from the item's mixed effective damage cdf:
    the building is not placed, so where it stands is folded into the cdf as hazard uncertainty is.
    Otherwise each sample reads the item areaperil its building's placement falls on,
    ``areaperil_j_of_sample[s]``; an item of a single placement reads areaperil 0 throughout. A
    sample on an uncovered placement (-1) or on an areaperil that is dead this event loses nothing.
    On a live areaperil the sample's condition is drawn from the areaperil's hazard cdf (an intensity
    bin), and its loss from the damage cdf of the areaperil's member block under that condition. That
    cdf is written to the item's row buffer the first time a sample needs it, so the search is over a
    short contiguous row; a row is one hazard row of the item (one source damage bin for a dependent).

    Coverage dependency: a dependent item (``is_dependent``, full Monte Carlo) takes as its
    condition its source's per-sample sampled damage bin, read from ``parent_bins``, and reads the
    conditional table. This item's own sampled damage bin is recorded in ``src_bin_out`` so a
    dependent below it in the DFS can consume it. Both are slices of one building's bins, so a
    packed item's buildings each drive their own dependent.

    Args:
        compute_info (gulmc_compute_info_type): computation state (debug, effective_damageability).
        sample_size (int): number of random samples.
        hazard_rng_index (int): index into hazard random values, or < 0 if hazard deterministic.
        haz_z_unif (np.array[float]): hazard random values for this building.
        vuln_z_unif (np.array[float]): damage random values for this building.
        areaperil_j_of_sample (np.array[int32]): per sample, the item areaperil read, -1 for none.
          Not read under effective_damageability.
        areaperil_j_live (np.array[bool]): per item areaperil, whether it can take a loss this event.
        areaperil_j_to_haz_start (np.array[int64]): per item areaperil, its first row in the haz buffers.
        areaperil_j_to_Nhaz_bins (np.array[int64]): per item areaperil, its number of hazard bins.
        areaperil_j_to_member_block (np.array[int32]): per item areaperil, its vulnerability member block.
        haz_cdf_buf (np.array[oasis_float]): hazard cdfs of the item areaperils, packed by row.
        haz_bin_id_buf (np.array[int32]): the intensity bin id of each of those rows.
        row_cdf_buf (np.array[oasis_float, 2d]): per row, the damage cdf the samples on it read.
        row_lengths (np.array[int64]): per row, the length of that cdf.
        row_stamp (np.array[int64]): per row, the item it was written for; a row is written at most
          once per item.
        item_stamp (int): this item's stamp.
        table (np.array[oasis_float, 3d]): the damage cdfs sampled from, ``vuln_cdf`` or, for a
          dependent, ``cond_cdf``.
        table_lengths (np.array[int32, 2d]): their lengths.
        vuln_member_ja_offsets (np.array[oasis_int]): per member block, its members.
        vuln_member_ja_rows (np.array[int32]): per member, its row of ``table``.
        vuln_member_ja_weights (np.array[float64]): per member, its normalised weight.
        mixed_eff_damage_cdf (np.array[oasis_float]): the item's mixed effective damage cdf, its
          first ``n_mixed`` entries valid. Read under effective_damageability only.
        n_mixed (int): length of ``mixed_eff_damage_cdf``.
        damage_bin_scaling (float): the item's tiv scaling.
        damage_bins (np.array): damage bin dictionary.
        out (np.array[oasis_float]): 1-d view of length ``sample_size`` written in place, holding
            this building's samples only.
        is_dependent (bool): True if this item is driven by a source item's sampled damage bin.
        src_bin_out (np.array[int32]): where the sampled damage bin is written, one per sample; 0
            (no damage) for a sample that loses nothing.
        parent_bins (np.array[int32]): the source item's bins for the SAME building, read when
          ``is_dependent``. Aliases ``src_bin_out`` otherwise, which keeps the argument typed
          without giving it a meaning.
    """
    if compute_info['debug'] == 1:  # store the random value used for the hazard sampling instead of the loss
        if hazard_rng_index >= 0:
            for sample_k in range(sample_size):
                out[sample_k] = haz_z_unif[sample_k]
        else:
            # deterministic hazard / effective damageability: no hazard intensity sampled
            out[:] = 0

    elif compute_info['debug'] == 2:  # store the random value used for the damage sampling instead of the loss
        for sample_k in range(sample_size):
            out[sample_k] = vuln_z_unif[sample_k]

    else:  # calculate gul
        if compute_info['effective_damageability']:
            for sample_idx in range(sample_size):
                out[sample_idx] = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx], mixed_eff_damage_cdf,
                                                        n_mixed, damage_bins, damage_bin_scaling,
                                                        src_bin_out, sample_idx)
            return
        for sample_idx in range(sample_size):
            areaperil_j = areaperil_j_of_sample[sample_idx]
            if areaperil_j < 0 or not areaperil_j_live[areaperil_j]:
                out[sample_idx] = 0.
                src_bin_out[sample_idx] = 0
                continue
            if is_dependent:
                # the dependent's condition is the source's sampled damage bin, read straight from
                # the stack -- no ratio round-trip, so a source of any damage type works. Its cdf is
                # the same on every areaperil, so the row is the bin.
                condition = parent_bins[sample_idx]
                row = condition
            else:
                haz_start = areaperil_j_to_haz_start[areaperil_j]
                Nhaz_bins = areaperil_j_to_Nhaz_bins[areaperil_j]
                haz_row = haz_start
                if Nhaz_bins > 1:
                    # find the hazard intensity cdf bin in which the random value falls. The last cdf
                    # value is not searched: a value above it for rounding reasons must still give the
                    # last bin, Nhaz_bins - 1, not one past it.
                    haz_row += binary_search(haz_z_unif[sample_idx], haz_cdf_buf[haz_start:haz_start + Nhaz_bins], Nhaz_bins - 1)
                condition = haz_bin_id_buf[haz_row] - 1
                row = haz_row
            if row_stamp[row] != item_stamp:
                row_stamp[row] = item_stamp
                fill_damage_row(row_cdf_buf, row_lengths, row, table, table_lengths, vuln_member_ja_offsets,
                                vuln_member_ja_rows, vuln_member_ja_weights, areaperil_j_to_member_block[areaperil_j],
                                condition)
            out[sample_idx] = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx], row_cdf_buf[row], row_lengths[row],
                                                    damage_bins, damage_bin_scaling, src_bin_out, sample_idx)


@nb.njit(cache=True, fastmath=True, _nrt=False)
def _compute_event_losses_core(compute_info,
                               coverages,
                               coverage_ids,
                               items_event_data,
                               items,
                               item_mixed_cdf_groups,
                               mixed_cdf_group_areaperil_ja_offsets,
                               mixed_cdf_group_areaperil_ja_cell_groups,
                               mixed_cdf_group_areaperil_ja_member_blocks,
                               mixed_cdf_group_areaperil_haz_arr_i,
                               mixed_cdf_group_areaperil_event_rp,
                               blended_areaperil_ja_offsets,
                               blended_areaperil_ja_probabilities,
                               mixed_cdf_cache_tag,
                               mixed_cdf_cache,
                               mixed_cdf_cache_nbins,
                               mixed_cdf_group_hermite,
                               max_areaperils_per_item,
                               sample_size,
                               haz_pdf,
                               haz_arr_ptr,
                               vuln_cdf,
                               vuln_cdf_lengths,
                               cond_cdf,
                               cond_cdf_lengths,
                               vuln_member_ja_offsets,
                               vuln_member_ja_rows,
                               vuln_member_ja_weights,
                               damage_bins,
                               losses,
                               vuln_adj,
                               haz_eps_ij,
                               damage_eps_ij,
                               norm_inv_parameters,
                               norm_inv_cdf,
                               norm_cdf,
                               vuln_z_unif,
                               haz_z_unif,
                               byte_mv,
                               dynamic_footprint,
                               intensity_bin_peril_ids,
                               intensity_bins,
                               sample_losses,
                               loss_correlation_by_item,
                               hermite_coeffs,
                               vuln_rndms_base,
                               haz_rndms_base,
                               vuln_seeds,
                               haz_seeds,
                               vuln_draw_scratch,
                               haz_draw_scratch,
                               perm_scratch,
                               event_sample_codes,
                               event_sample_areaperil_j,
                               event_sample_bins,
                               event_sample_vuln,
                               event_sample_haz,
                               coverage_has_dependents,
                               compute_depth,
                               source_damage_bin_stack,
                               source_eff_damage_cdf_stack,
                               source_eff_damage_cdf_len_stack,
                               bin_slot,
                               haz_pdf_prob_buf,
                               haz_bin_id_buf,
                               haz_cdf_buf,
                               row_cdf_buf,
                               row_lengths,
                               row_stamp,
                               areaperil_j_live,
                               areaperil_j_stamp,
                               areaperil_j_to_haz_start,
                               areaperil_j_to_Nhaz_bins,
                               areaperil_j_to_member_block,
                               cdf_acc,
                               mix_acc,
                               curve_buf,
                               mixed_eff_damage_cdf,
                               areaperil_j_single,
                               areaperil_j_of_building,
                               source_bin_ids,
                               source_damage_pmf,
                               hermite_he_lo,
                               hermite_he_hi,
                               every_sample,
                               sample_idx_buf,
                               sample_bin_buf,
                               parent_bin_buf,
                               eps_buf,
                               run_offsets,
                               run_lengths,
                               run_code_starts,
                               item_cursor,
                               item_values,
                               sample_acc):
    """Body of compute_event_losses, compiled without the numba runtime (``_nrt=False``).

    Without the runtime numba emits no NRT incref/decref at all, which removes the per-item
    refcounting the inlined helpers' array arguments still cost (see the refcount note in
    common/event_stream.py). In exchange nothing in here, or in any helper it calls, may allocate
    (numba refuses to compile it), so the scratch buffers are allocated by compute_event_losses and
    passed in. Every array here is an argument or a view of one, kept alive by the caller for the
    whole call. ``_nrt`` is a private numba option.

    The arguments are those of compute_event_losses, followed by its scratch buffers.
    """
    mixed_cdf_cache_mask = nb_int64(mixed_cdf_cache.shape[0] - 1)
    Ndamage_bins_max = vuln_cdf.shape[2]
    damage_bins_interpolation = damage_bins['interpolation']  # one field view per call, not per item
    item_stamp = nb_int64(0)
    n_source_bins = cond_cdf.shape[1]

    # we process at least one full coverage at a time, so when we write to stream, we write the whole buffer
    compute_info['cursor'] = 0

    # loop through all the coverages that remain to be computed. `compute` is in DFS order — a
    # root (compute_depth == 0) is followed by its subtree — so a source precedes its dependents.
    for coverage_i in range(compute_info['coverage_i'], compute_info['coverage_n']):
        coverage_id = coverage_ids[coverage_i]
        coverage = coverages[coverage_id]
        depth = compute_depth[coverage_i]
        tiv = coverage['tiv']
        Nitems = coverage['cur_items']
        exposureValue = tiv / Nitems

        # Every coverage is emitted whole -- there is no return between the first byte of a
        # coverage and its last -- so its output has to fit before we start. The bound
        # OVER-reserves: max_bytes_per_item carries max_emitted_blocks over EVERY item in the
        # portfolio, so a coverage of one-building items is charged for the largest packed item
        # that exists anywhere. Safe, since it can only over-reserve, but sizing it from this
        # coverage's own items is a known improvement.
        # Bailing here can land inside a dependency subtree, which is sound: a flush only writes
        # bytes out, and the stacks a dependent reads (source_damage_bin_stack,
        # source_eff_damage_cdf_stack) are caller-owned and outlive the call.
        if compute_info['cursor'] + Nitems * compute_info['max_bytes_per_item'] > byte_mv.shape[0]:
            return False

        coverage_is_dependent = compute_info['do_coverage_dependency'] == 1 and depth > 0
        # nothing below reads this coverage's sampled bins unless it actually has dependents, and
        # the stack is sized on that basis -- storing anyway would write past its width
        has_dependents_below = (compute_info['do_coverage_dependency'] == 1
                                and coverage_has_dependents[coverage_id] == 1)
        # each item's samples are a run of sample_losses, laid out one after another (see write_losses)
        run_offset = nb_int64(0)
        # compute losses for each item, resuming where a flush interrupted this coverage
        for item_j in range(Nitems):
            item_event_data = items_event_data[coverage['start_items'] + item_j]
            run_offsets[item_j] = run_offset
            run_lengths[item_j] = 0
            run_code_starts[item_j] = -1
            rng_index = item_event_data['rng_index']
            hazard_rng_index = item_event_data['hazard_rng_index']
            # signed; this loop only needs how many buildings to draw for. 1 when nothing is
            # packed, which is what makes the unpacked case just N == 1 here.
            n_buildings = abs(item_event_data['packed_buildings'])
            item_idx = item_event_data['item_idx']
            item = items[item_idx]
            # an item is dependent only if it resolved to a source item (< 0 means it did not)
            is_dependent = coverage_is_dependent and item_event_data['source_item_j'] >= 0
            mixed_cdf_group = item_mixed_cdf_groups[item_idx]
            group_start = mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group]
            n_areaperils = mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group + 1] - group_start
            blended_start = blended_areaperil_ja_offsets[item['blended_areaperil_idx']]
            intensity_adjustment = item['intensity_adjustment'] if dynamic_footprint is not None else nb_oasis_int(0)
            protection_rp = item_event_data['return_period']

            item_rho = item_event_data['damage_correlation_value']
            do_hermite = item_event_data['packed_buildings'] > 1 and item_rho > 0.

            # --- the item's effective damage cdf. An areaperil is dead when it has no hazard this
            # event, or (dynamic footprint) when the event's return period there is within the item's
            # protection; dead areaperils and uncovered placements carry no loss, their mass on
            # damage bin 0.
            item_live = True
            if not is_dependent:
                if mixed_cdf_group_areaperil_ja_cell_groups[group_start] == mixed_cdf_group:
                    # an areaperil read directly: its own cdf, which protection does not change, so it
                    # decides here whether the item counts it
                    slot = cached_mixed_cdf_slot(compute_info, mixed_cdf_group, mixed_cdf_cache_tag, mixed_cdf_cache_mask)
                    if slot < 0:
                        slot = areaperil_cdf_slot(
                            compute_info, mixed_cdf_group, mixed_cdf_group_areaperil_haz_arr_i[group_start],
                            mixed_cdf_group_areaperil_ja_member_blocks[group_start], item, intensity_adjustment,
                            haz_pdf, haz_arr_ptr, dynamic_footprint, intensity_bin_peril_ids, intensity_bins, bin_slot,
                            haz_bin_id_buf, haz_pdf_prob_buf, haz_cdf_buf,
                            vuln_cdf, vuln_cdf_lengths, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                            mixed_cdf_cache_tag, mixed_cdf_cache, mixed_cdf_cache_nbins, mixed_cdf_group_hermite,
                            damage_bins, norm_inv_parameters, norm_inv_cdf, cdf_acc, curve_buf, hermite_he_lo,
                            hermite_he_hi)
                    item_live = group_areaperil_is_live(mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                                                        group_start, protection_rp, dynamic_footprint)
                else:
                    slot = cached_mixed_cdf_slot(compute_info, mixed_cdf_group, mixed_cdf_cache_tag, mixed_cdf_cache_mask)
                    if slot < 0:
                        # a blended group: its live areaperils' own cdfs by probability, computed by
                        # its first item this event. Protection is part of its key, so the group's
                        # items share it.
                        needs_hermite = mixed_cdf_group_hermite.shape[0] > 0
                        if needs_hermite:
                            for k in range(HERMITE_TERMS):
                                mixed_cdf_group_hermite[mixed_cdf_group, 0, k] = 0.
                                mixed_cdf_group_hermite[mixed_cdf_group, 1, k] = 0.
                        for damage_bin_i in range(Ndamage_bins_max):
                            mix_acc[damage_bin_i] = 0.
                        live_probability = 0.
                        n_mixed = 0
                        for areaperil_j in range(n_areaperils):
                            group_areaperil_idx = group_start + areaperil_j
                            if not group_areaperil_is_live(mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                                                           group_areaperil_idx, protection_rp, dynamic_footprint):
                                continue
                            cell_group = mixed_cdf_group_areaperil_ja_cell_groups[group_areaperil_idx]
                            cell_slot = cached_mixed_cdf_slot(compute_info, cell_group, mixed_cdf_cache_tag, mixed_cdf_cache_mask)
                            if cell_slot < 0:
                                cell_slot = areaperil_cdf_slot(
                                    compute_info, cell_group, mixed_cdf_group_areaperil_haz_arr_i[group_areaperil_idx],
                                    mixed_cdf_group_areaperil_ja_member_blocks[group_areaperil_idx], item, intensity_adjustment,
                                    haz_pdf, haz_arr_ptr, dynamic_footprint, intensity_bin_peril_ids, intensity_bins, bin_slot,
                                    haz_bin_id_buf, haz_pdf_prob_buf, haz_cdf_buf,
                                    vuln_cdf, vuln_cdf_lengths, vuln_member_ja_offsets, vuln_member_ja_rows, vuln_member_ja_weights,
                                    mixed_cdf_cache_tag, mixed_cdf_cache, mixed_cdf_cache_nbins, mixed_cdf_group_hermite,
                                    damage_bins, norm_inv_parameters, norm_inv_cdf, cdf_acc, curve_buf, hermite_he_lo,
                                    hermite_he_hi)
                            # read at once: the next areaperil's cdf may take its slot
                            p_areaperil = blended_areaperil_ja_probabilities[blended_start + areaperil_j]
                            cell_bins = mixed_cdf_cache_nbins[cell_slot]
                            n_mixed = max(n_mixed, cell_bins)
                            live_probability += p_areaperil
                            add_cdf_row(mix_acc, p_areaperil, mixed_cdf_cache, cell_slot, cell_bins, Ndamage_bins_max)
                            if needs_hermite:
                                # the terms are linear in the weights, so a blended group's are its
                                # areaperils' weighted by p
                                for k in range(HERMITE_TERMS):
                                    mixed_cdf_group_hermite[mixed_cdf_group, 0, k] += p_areaperil * mixed_cdf_group_hermite[cell_group, 0, k]
                                    mixed_cdf_group_hermite[mixed_cdf_group, 1, k] += p_areaperil * mixed_cdf_group_hermite[cell_group, 1, k]
                        # an empty cdf means nothing is live
                        add_no_loss_mass(mix_acc, live_probability, n_mixed)
                        slot = store_mixed_cdf(compute_info, mixed_cdf_group, mix_acc, n_mixed, mixed_cdf_cache_tag,
                                               mixed_cdf_cache, mixed_cdf_cache_nbins)
                n_mixed = mixed_cdf_cache_nbins[slot]
                for damage_bin_i in range(n_mixed):
                    mixed_eff_damage_cdf[damage_bin_i] = mixed_cdf_cache[slot, damage_bin_i]
                if do_hermite:
                    for k in range(HERMITE_TERMS):
                        hermite_coeffs[k] = (tiv * mixed_cdf_group_hermite[mixed_cdf_group, 0, k]
                                             + mixed_cdf_group_hermite[mixed_cdf_group, 1, k]) if tiv > 0 else 0.
            else:
                # A dependent's cdf comes from its source's damage, not from the footprint hazard: the
                # conditions are the source's damage bins and their distribution the source's damage
                # pmf (from its stored effective damage cdf). It is the same on every areaperil it is
                # live on, so it is convolved once and weighted by the probability of those.
                src_j = item_event_data['source_item_j']
                parent_eff_cdf_len = source_eff_damage_cdf_len_stack[depth - 1, src_j]
                prev_cdf = 0.0
                for damage_bin_k in range(n_source_bins):
                    if damage_bin_k < parent_eff_cdf_len:
                        parent_cdf = source_eff_damage_cdf_stack[depth - 1, src_j, damage_bin_k]
                        source_damage_pmf[damage_bin_k] = parent_cdf - prev_cdf
                        prev_cdf = parent_cdf
                    else:
                        source_damage_pmf[damage_bin_k] = 0.0
                live_probability = 0.
                live_block = nb_int32(0)
                for areaperil_j in range(n_areaperils):
                    group_areaperil_idx = group_start + areaperil_j
                    # protection applies to a dependent too: protected, its coverage takes no loss
                    # whatever its source suffered
                    live = group_areaperil_is_live(mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                                                   group_areaperil_idx, protection_rp, dynamic_footprint)
                    areaperil_j_live[areaperil_j] = live
                    areaperil_j_to_member_block[areaperil_j] = mixed_cdf_group_areaperil_ja_member_blocks[group_areaperil_idx]
                    if live:
                        live_probability += blended_areaperil_ja_probabilities[blended_start + areaperil_j]
                        live_block = areaperil_j_to_member_block[areaperil_j]
                n_mixed = 0
                if live_probability > 0.:
                    n_mixed = convolve_damage_cdf(cdf_acc, cond_cdf, cond_cdf_lengths, vuln_member_ja_offsets,
                                                  vuln_member_ja_rows, vuln_member_ja_weights, live_block, source_bin_ids,
                                                  source_damage_pmf, n_source_bins)
                    for damage_bin_i in range(n_mixed):
                        mixed_eff_damage_cdf[damage_bin_i] = live_probability * cdf_acc[damage_bin_i] + 1. - live_probability
                    if do_hermite:
                        for k in range(HERMITE_TERMS):
                            hermite_coeffs[k] = 0.
                        damage_bin_scaling = compute_damage_bin_scaling(damage_bins, n_mixed, tiv)
                        if compute_info['effective_damageability']:
                            accumulate_hermite_coeffs_into(damage_bin_scaling, cdf_acc, damage_bins_interpolation, n_mixed,
                                                           live_probability, norm_inv_parameters['x_min'],
                                                           norm_inv_parameters['inv_factor'], norm_inv_cdf, hermite_coeffs,
                                                           hermite_he_lo, hermite_he_hi)
                        else:
                            # The weights are the source's damage pmf, which treats the source's
                            # buildings as independent, so a correlated source makes this an underestimate.
                            accumulate_condition_hermite(damage_bin_scaling, cond_cdf, vuln_member_ja_offsets,
                                                         vuln_member_ja_rows, vuln_member_ja_weights, live_block,
                                                         source_bin_ids, source_damage_pmf, n_source_bins, live_probability,
                                                         damage_bins_interpolation, norm_inv_parameters, norm_inv_cdf,
                                                         curve_buf, hermite_coeffs, hermite_he_lo, hermite_he_hi)

            if n_mixed == 0 or not item_live:
                # every areaperil dead at this event (no hazard, or protected): the item is skipped
                # whole, samples and analytic sidx alike
                losses[:, item_j] = 0
                # and its run stays empty: no sample lost anything
                if has_dependents_below:
                    # a dependent below this one in the DFS still reads (depth, item_j), so
                    # leave "no damage" there rather than the last coverage's values
                    source_damage_bin_stack[depth, item_j, :] = 0
                    source_eff_damage_cdf_stack[depth, item_j, 0] = 1.
                    source_eff_damage_cdf_len_stack[depth, item_j] = 1
                continue

            # the analytic values are those of the item's effective damage cdf. For a blended item
            # this is the cdf getmodel streams to gulpy for the same item.
            damage_bin_scaling = compute_damage_bin_scaling(damage_bins, n_mixed, tiv)
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                damage_bin_scaling,
                mixed_eff_damage_cdf[:n_mixed],
                damage_bins_interpolation,
                n_mixed,
                damage_bins[n_mixed - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_j] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_j] = chance_of_loss
            losses[TIV_IDX, item_j] = exposureValue
            losses[STD_DEV_IDX, item_j] = std_dev
            losses[MEAN_IDX, item_j] = gul_mean

            if do_hermite:
                loss_correlation_by_item[item_j] = loss_correlation(
                    hermite_coeffs, std_dev * std_dev, item_rho)
            else:
                loss_correlation_by_item[item_j] = 0.

            if sample_size > 0:  # compute random losses
                # full Monte Carlo: record this coverage's per-sample damage bin for any dependent
                # below it in the DFS
                full_monte_carlo = not compute_info['effective_damageability']
                store_source_bin = has_dependents_below and full_monte_carlo
                # the damage random values of a single vulnerability are adjusted, an aggregate's not
                # (its block's adjustment is 1)
                damage_rval_adj = vuln_adj[mixed_cdf_group_areaperil_ja_member_blocks[group_start]]
                table = cond_cdf if is_dependent else vuln_cdf
                table_lengths = cond_cdf_lengths if is_dependent else vuln_cdf_lengths
                # none of the item's areaperils is resolved for its samples yet; a dependent resolved
                # them all above
                item_stamp += 1
                haz_used = 0

                # One run of a building's samples at a time, each drawn, resolved and sampled alike.
                # A sparse item's runs are its listed samples (prepare_event_samples), in code order
                # ((building - 1) * S + sample) and with their base random values; its other samples
                # lost nothing. Any other item's runs are every sample of each of its buildings, with
                # the building's Latin Hypercube block: building 1 is the group's shared row, drawn once
                # for the event; 2..N are drawn here, which reaching at all means the item is packed,
                # so on generator 2, the block generator (check_packing_supported).
                sparse = item_event_data['sparse'] == 1
                sample_k = item_event_data['sample_start']
                sample_end = item_event_data['sample_end']
                if sparse:
                    run_code_starts[item_j] = sample_k
                    run_lengths[item_j] = sample_end - sample_k
                    if is_dependent:
                        # its source's listed samples, ascending like its own, with their damage bins
                        source_k = items_event_data[item_event_data['source_item_i']]['sample_start']
                        source_end = items_event_data[item_event_data['source_item_i']]['sample_end']
                else:
                    run_lengths[item_j] = n_buildings * sample_size
                building_i = 0
                while (sample_k < sample_end) if sparse else (building_i < n_buildings):
                    if sparse:
                        building_i = event_sample_codes[sample_k] // sample_size
                        run_start = sample_k
                        n_samples = 0
                        while sample_k < sample_end and event_sample_codes[sample_k] // sample_size == building_i:
                            sample_idx_buf[n_samples] = event_sample_codes[sample_k] - building_i * sample_size
                            areaperil_j_of_building[n_samples] = event_sample_areaperil_j[sample_k]
                            vuln_draw_scratch[n_samples] = event_sample_vuln[sample_k]
                            haz_draw_scratch[n_samples] = event_sample_haz[sample_k]
                            if is_dependent:
                                # the source's bin at the same building and sample, none where it lost nothing
                                while source_k < source_end and event_sample_codes[source_k] < event_sample_codes[sample_k]:
                                    source_k += 1
                                parent_bin_buf[n_samples] = (event_sample_bins[source_k] if source_k < source_end
                                                             and event_sample_codes[source_k] == event_sample_codes[sample_k]
                                                             else 0)
                            n_samples += 1
                            sample_k += 1
                        samples = sample_idx_buf
                        areaperil_j_of_sample = areaperil_j_of_building
                        vuln_base = vuln_draw_scratch
                        haz_base = haz_draw_scratch
                        out_start = run_offset + run_start - item_event_data['sample_start']
                        # a source keeps its listed samples' bins for its dependents; anything else's go to scratch
                        src_bin_out = event_sample_bins[run_start:sample_k] if store_source_bin else sample_bin_buf
                        parent_bins = parent_bin_buf
                    else:
                        n_samples = sample_size
                        samples = every_sample
                        # every sample reads areaperil 0: an item of several placements is sparse, and
                        # effective damageability samples the mixture
                        areaperil_j_of_sample = areaperil_j_single
                        if building_i == 0:
                            vuln_base = vuln_rndms_base[rng_index]
                        else:
                            vs = np.uint64(vuln_seeds[rng_index])
                            _lh_philox_block(np.uint32(vs & PHILOX_U32_MASK), np.uint32(vs >> PHILOX_SHIFT32), building_i,
                                             sample_size, perm_scratch, vuln_draw_scratch)
                            vuln_base = vuln_draw_scratch
                        if hazard_rng_index < 0:
                            haz_base = vuln_base  # unused; keeps the argument type stable
                        elif building_i == 0:
                            haz_base = haz_rndms_base[hazard_rng_index]
                        else:
                            hs = np.uint64(haz_seeds[hazard_rng_index])
                            _lh_philox_block(np.uint32(hs & PHILOX_U32_MASK), np.uint32(hs >> PHILOX_SHIFT32), building_i,
                                             sample_size, perm_scratch, haz_draw_scratch)
                            haz_base = haz_draw_scratch
                        # this building's slice of the bin stack, same (b - 1) * S + (s - 1) encoding as a
                        # packed sidx: building b of a dependent is driven by building b of its source
                        out_start = run_offset + building_i * sample_size
                        bin_off = building_i * sample_size
                        if store_source_bin:
                            src_bin_out = source_damage_bin_stack[depth, item_j, bin_off: bin_off + sample_size]
                        else:
                            # The sampler stores the bin unconditionally, to keep the branch out of its
                            # per-sample loop, so it needs somewhere in bounds to put a value nothing
                            # will read. bin_off can run past the stack's width, which is only as wide
                            # as a SOURCE item's buildings; the first building's slot never does.
                            src_bin_out = source_damage_bin_stack[depth, item_j, 0: sample_size]
                        if is_dependent:
                            parent_bins = source_damage_bin_stack[depth - 1, item_event_data['source_item_j'],
                                                                  bin_off: bin_off + sample_size]
                        else:
                            parent_bins = src_bin_out  # unused; keeps the argument type stable
                        building_i += 1

                    draw_correlation_samples(compute_info, item, damage_rval_adj, hazard_rng_index, n_samples, samples,
                                             haz_base, vuln_base, haz_eps_ij, damage_eps_ij, norm_inv_parameters,
                                             norm_inv_cdf, norm_cdf, eps_buf, haz_z_unif, vuln_z_unif)
                    if not is_dependent and full_monte_carlo:
                        # a dependent's areaperils were resolved with its cdf above
                        haz_used = resolve_sample_areaperils(
                            n_samples, areaperil_j_of_sample, item_stamp, areaperil_j_stamp, areaperil_j_live,
                            areaperil_j_to_haz_start, areaperil_j_to_Nhaz_bins, areaperil_j_to_member_block,
                            group_start, mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                            mixed_cdf_group_areaperil_ja_member_blocks, protection_rp, dynamic_footprint,
                            haz_pdf, haz_arr_ptr, item, intensity_adjustment, intensity_bin_peril_ids, intensity_bins,
                            bin_slot, haz_bin_id_buf, haz_pdf_prob_buf, haz_cdf_buf, haz_used)
                    sample_item_losses(compute_info, n_samples, hazard_rng_index,
                                       haz_z_unif, vuln_z_unif, areaperil_j_of_sample, areaperil_j_live,
                                       areaperil_j_to_haz_start, areaperil_j_to_Nhaz_bins, areaperil_j_to_member_block,
                                       haz_cdf_buf, haz_bin_id_buf, row_cdf_buf, row_lengths, row_stamp, item_stamp,
                                       table, table_lengths, vuln_member_ja_offsets, vuln_member_ja_rows,
                                       vuln_member_ja_weights, mixed_eff_damage_cdf, n_mixed, damage_bin_scaling,
                                       damage_bins, sample_losses[out_start:out_start + n_samples], is_dependent,
                                       src_bin_out, parent_bins)

            # effective damageability: record the eff-damage CDF instead, for a dependent below
            # to build its damage pmf from; a blended source records its mixed cdf
            if has_dependents_below:
                for damage_bin_k in range(n_mixed):
                    source_eff_damage_cdf_stack[depth, item_j, damage_bin_k] = mixed_eff_damage_cdf[damage_bin_k]
                source_eff_damage_cdf_len_stack[depth, item_j] = n_mixed

            run_offset += run_lengths[item_j]

        # write the losses to the output memoryview. A zero-TIV coverage (an uninsured dependency
        # source) yields zero losses and is written like any other — not special-cased.
        compute_info['cursor'] = write_losses(
            compute_info['event_id'],
            sample_size,
            compute_info['loss_threshold'],
            losses[:, :Nitems],
            items_event_data[coverage['start_items']: coverage['start_items'] + Nitems]['item_id'],
            items_event_data[coverage['start_items']: coverage['start_items'] + Nitems]['packed_buildings'],
            loss_correlation_by_item[:Nitems],
            compute_info['alloc_rule'],
            tiv,
            sample_losses,
            run_offsets[:Nitems],
            run_lengths[:Nitems],
            run_code_starts[:Nitems],
            event_sample_codes,
            item_cursor,
            item_values,
            sample_acc,
            byte_mv,
            compute_info['cursor'])

        # register that another `coverage_id` has been processed
        compute_info['coverage_i'] += 1

    return True


@nb.njit(cache=True, fastmath=True)
def compute_event_losses(compute_info,
                         coverages,
                         coverage_ids,
                         items_event_data,
                         items,
                         item_mixed_cdf_groups,
                         mixed_cdf_group_areaperil_ja_offsets,
                         mixed_cdf_group_areaperil_ja_cell_groups,
                         mixed_cdf_group_areaperil_ja_member_blocks,
                         mixed_cdf_group_areaperil_haz_arr_i,
                         mixed_cdf_group_areaperil_event_rp,
                         blended_areaperil_ja_offsets,
                         blended_areaperil_ja_probabilities,
                         mixed_cdf_cache_tag,
                         mixed_cdf_cache,
                         mixed_cdf_cache_nbins,
                         mixed_cdf_group_hermite,
                         max_areaperils_per_item,
                         sample_size,
                         haz_pdf,
                         haz_arr_ptr,
                         vuln_cdf,
                         vuln_cdf_lengths,
                         cond_cdf,
                         cond_cdf_lengths,
                         vuln_member_ja_offsets,
                         vuln_member_ja_rows,
                         vuln_member_ja_weights,
                         damage_bins,
                         losses,
                         vuln_adj,
                         haz_eps_ij,
                         damage_eps_ij,
                         norm_inv_parameters,
                         norm_inv_cdf,
                         norm_cdf,
                         vuln_z_unif,
                         haz_z_unif,
                         byte_mv,
                         dynamic_footprint,
                         intensity_bin_peril_ids,
                         intensity_bins,
                         sample_losses,
                         loss_correlation_by_item,
                         hermite_coeffs,
                         vuln_rndms_base,
                         haz_rndms_base,
                         vuln_seeds,
                         haz_seeds,
                         vuln_draw_scratch,
                         haz_draw_scratch,
                         perm_scratch,
                         event_sample_codes,
                         event_sample_areaperil_j,
                         event_sample_bins,
                         event_sample_vuln,
                         event_sample_haz,
                         coverage_has_dependents,
                         compute_depth,
                         source_damage_bin_stack,
                         source_eff_damage_cdf_stack,
                         source_eff_damage_cdf_len_stack):
    """Compute ground-up losses for all coverages in a single event.

    Iterates over coverages and their items. Each item reports the effective damage cdf of its
    mixed cdf group, read from the cache or computed by the group's first item this event, and in
    full Monte Carlo draws its samples from the damage cdf table. Results are written into a byte
    buffer for streaming output.

    One cache holds the mixed cdf groups' cdfs, one ring of slots (a power of two) indexed by a
    monotonic write counter: ``mixed_cdf_cache_tag[group]`` is the counter value of the group's
    write, valid until the ring has taken as many writes after it as it has slots. An areaperil's
    own cdf under a vulnerability is its size-1 group's, convolved once per event however many
    blended groups read it; a blended group's is its areaperils' cdfs mixed by probability.

    Args:
        compute_info (gulmc_compute_info_type): computation state (event_id, cursor position,
          coverage range, mixed_cdf_cache_ctr, thresholds, flags).
        coverages (numpy.array[coverage_type]): coverage data indexed by coverage_id.
        coverage_ids (numpy.array[int]): ordered list of coverage_ids to process in this event.
        items_event_data (numpy.array[items_MC_data_type]): per-item event data populated by
          reconstruct_coverages, containing item_idx, rng_index and hazard_rng_index.
        items (np.ndarray): items table merged with correlation parameters.
        item_mixed_cdf_groups (np.array[int32]): per item, its mixed cdf group (see ``build_item_mixed_cdf_groups``).
        mixed_cdf_group_areaperil_ja_offsets (np.array[oasis_int]): per mixed cdf group, its run of group
          areaperils, j-th for the group's j-th areaperil.
        mixed_cdf_group_areaperil_ja_cell_groups (np.array[int32]): per group areaperil, the size-1
          group of its areaperil, which is the group itself for a size-1 group.
        mixed_cdf_group_areaperil_ja_member_blocks (np.array[int32]): per group areaperil, the
          vulnerability member block its damage cdf is read from.
        mixed_cdf_group_areaperil_haz_arr_i (np.array[oasis_int]): this event's hazard row per group
          areaperil, -1 when its areaperil has no hazard this event.
        mixed_cdf_group_areaperil_event_rp (np.array[int32]): this event's return period per group areaperil
          (dynamic footprint only).
        blended_areaperil_ja_offsets (np.array[oasis_int]): per blended areaperil, its areaperils.
        blended_areaperil_ja_probabilities (np.array[float64]): probability the building is on
          each areaperil, the weights of the analytic mixture.
        mixed_cdf_cache_tag (np.array[int64]): per mixed cdf group, the value of
          ``compute_info['mixed_cdf_cache_ctr']`` when its cdf was cached (CDF_CACHE_EMPTY = -1).
        mixed_cdf_cache (np.array[oasis_float, 2d]): ring of mixed cdfs, one per slot; its row
          count is a power of two.
        mixed_cdf_cache_nbins (np.array[int32]): per slot, its cdf's length, 0 for a group none of
          whose areaperils can take a loss this event.
        mixed_cdf_group_hermite (np.array[float64, 3d]): per mixed cdf group, the Hermite terms of its
          cdf without the tiv, as [part the tiv multiplies, part it does not]. Empty when no item
          needs them.
        max_areaperils_per_item (int): the largest K, sizing the per-areaperil scratch.
        sample_size (int): number of random samples to draw.
        haz_pdf (np.array[haz_arr_type]): hazard intensity pdf records for this event.
        haz_arr_ptr (np.array[int64]): indices where each areaperil's hazard records start in haz_pdf.
        vuln_cdf (np.array[oasis_float, 3d]): damage cdf per (vulnerability, intensity bin, damage bin).
        vuln_cdf_lengths (np.array[int32, 2d]): their lengths.
        cond_cdf (np.array[oasis_float, 3d]): damage cdf per (conditional vulnerability, source damage
          bin, damage bin). Empty when no dependency.
        cond_cdf_lengths (np.array[int32, 2d]): their lengths.
        vuln_member_ja_offsets (np.array[oasis_int]): per member block, its members.
        vuln_member_ja_rows (np.array[int32]): per member, its row of ``vuln_cdf`` (``cond_cdf`` for
          a conditional vulnerability).
        vuln_member_ja_weights (np.array[float64]): per member, its normalised weight.
        damage_bins (np.array): damage bin dictionary with bin_from, bin_to, interpolation, damage_type.
        losses (numpy.array[oasis_float]): reusable 2d buffer for loss values.
        vuln_adj (np.array[float]): per member block, the damage random value adjustment.
        haz_eps_ij (np.array[float]): correlated random values for hazard sampling.
        damage_eps_ij (np.array[float]): correlated random values for damage sampling.
        norm_inv_parameters (NormInversionParameters): parameters for Gaussian inversion.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf lookup table.
        norm_cdf (np.array[float]): Gaussian cdf lookup table.
        vuln_z_unif (np.array[float]): reusable buffer for correlated vulnerability random values.
        haz_z_unif (np.array[float]): reusable buffer for correlated hazard random values.
        byte_mv (numpy.array[byte]): output byte buffer for the binary stream.
        dynamic_footprint (None or object): None if no dynamic footprint, otherwise truthy.
        intensity_bin_peril_ids (np.array[int32]): sorted unique encoded peril_ids (length n_perils).
        intensity_bins (np.array[int32, 2d]): shape (n_perils, max_intensity + 1) mapping
          [peril_idx, intensity_value] -> intensity_bin_id.
        sample_losses (numpy.array[oasis_float]): reusable buffer of a coverage's sample losses,
          held whole before write_losses emits it: per item a run, of N * S for an item computed
          for every sample, or of its live samples for a sparse one (see write_losses). Sized for
          the event's largest coverage.
        loss_correlation_by_item (numpy.array[oasis_float]): per item of the current coverage,
          the correlation between two of its buildings' losses, written where the analytic
          moments are and read when its summed specials are emitted. 0 for everything that
          does not report the spread of a sum.
        hermite_coeffs (numpy.array[float64]): length HERMITE_TERMS scratch for that.
        vuln_rndms_base (numpy.array[float64]): 2d (damage rng groups, S) random draws, one row
          per group, which is also every packed item's first building.
        haz_rndms_base (numpy.array[float64]): the same for the hazard rng groups.
        vuln_seeds (numpy.array[int]): per damage rng group, the Philox key. Read for a packed
          item's buildings 2..N, which are drawn per item rather than shared.
        haz_seeds (numpy.array[int]): the same for the hazard rng groups.
        vuln_draw_scratch (numpy.array[float64]): length-S buffer for one building's damage block.
        haz_draw_scratch (numpy.array[float64]): the same for hazard.
        perm_scratch (numpy.array[float64]): length-S scratch the block generator permutes in.
        coverage_has_dependents (np.array[int8]): per coverage_id, 1 if any coverage names it as
          its source. Only those coverages' sampled damage bins are stored, and the bin stack is
          sized on that basis, so storing for any other coverage would write past its width.
        compute_depth (np.array[int32]): per entry of ``coverage_ids``, its depth in the dependency
          forest (0 for a root), matching the DFS order the caller put them in.
        source_damage_bin_stack (np.array[int32]): per-depth, per-item sampled damage bins, holding
          a source coverage's result while its dependent subtree is computed.
        source_eff_damage_cdf_stack (np.array[oasis_float]): per-depth, per-item effective damage
          cdfs, the effective-damageability counterpart of the sampled bins.
        source_eff_damage_cdf_len_stack (np.array[int64]): valid length of each entry of
          ``source_eff_damage_cdf_stack``.
        event_sample_codes (np.array[int64]): the live samples of the sparse items, as item codes
          ``(building - 1) * S + sample``, ascending per item; an item's are those in
          ``[sample_start, sample_end)`` of its items_event_data.
        event_sample_areaperil_j (np.array[int32]): the item areaperil each of them reads.
        event_sample_bins (np.array[int32]): the damage bin each of them drew, written by a source
          of coverage dependency for its dependents to read.
        event_sample_vuln (np.array[float64]): the damage base random value of each of them.
        event_sample_haz (np.array[float64]): the hazard base random value of each of them, 0 when
          the item draws no hazard.

    Returns:
        bool: True if all coverages have been processed, False if the buffer is full and
          the caller should flush and call again.
    """
    # scratch buffers for _compute_event_losses_core, which cannot allocate (compiled without NRT)
    Ndamage_bins_max = vuln_cdf.shape[2]
    # a dynamic footprint's records merged per intensity bin (see fill_haz_bins)
    n_bin_ids = intensity_bins.max() + 1 if intensity_bins.size > 0 else 1
    bin_slot = np.full(n_bin_ids, -1, dtype=np.int64)
    max_haz_bins = max(vuln_cdf.shape[1], n_bin_ids)
    haz_pdf_prob_buf = np.empty(max_haz_bins, dtype=oasis_float)
    # Per item areaperil its samples land on, its hazard rows stay resident until the item's samples
    # are drawn, packed one after another (areaperil_j_to_haz_start): at most S per building, or K.
    # An areaperil is resolved once per item and has no more rows than footprint records, so the
    # rows also stay within this event's records (haz_pdf holds only the areaperils items read).
    # Without that cap, K x intensity bins x damage bins can outgrow memory on a large blend.
    # At least one areaperil's worth even when S is 0, for the effective damage cdf.
    haz_rows = max(1, min(max_haz_bins * max(1, min(max_areaperils_per_item, sample_losses.shape[0])),
                          haz_pdf.shape[0]))
    haz_bin_id_buf = np.empty(haz_rows, dtype=np.int32)
    haz_cdf_buf = np.empty(haz_rows, dtype=oasis_float)
    # per row an item's samples reach -- a hazard row, or a source damage bin for a dependent -- its
    # damage cdf, written once per item when a sample first needs it
    n_rows = max(haz_rows, cond_cdf.shape[1])
    row_cdf_buf = np.empty((n_rows, vuln_cdf.shape[2]), dtype=oasis_float)
    row_lengths = np.zeros(n_rows, dtype=np.int64)
    row_stamp = np.full(n_rows, -1, dtype=np.int64)
    areaperil_j_live = np.zeros(max_areaperils_per_item, dtype=np.bool_)
    # an areaperil is resolved for an item's samples when one first lands there: resolved for this
    # item when its stamp is the item's
    areaperil_j_stamp = np.full(max_areaperils_per_item, -1, dtype=np.int64)
    areaperil_j_to_haz_start = np.zeros(max_areaperils_per_item, dtype=np.int64)
    areaperil_j_to_Nhaz_bins = np.zeros(max_areaperils_per_item, dtype=np.int64)
    areaperil_j_to_member_block = np.zeros(max_areaperils_per_item, dtype=np.int32)
    # an areaperil's cdf and a group's mixture are summed here, in float64, over the whole damage axis
    cdf_acc = np.empty(Ndamage_bins_max, dtype=np.float64)
    mix_acc = np.empty(Ndamage_bins_max, dtype=np.float64)
    curve_buf = np.empty(Ndamage_bins_max, dtype=np.float64)
    # the effective damage cdf of the item: its group's
    mixed_eff_damage_cdf = np.empty(Ndamage_bins_max, dtype=oasis_float)
    # an item of a single placement reads areaperil 0 in every sample; a blended item's building
    # maps its location's placements to its own areaperil_j here
    areaperil_j_single = np.zeros(max(sample_size, 1), dtype=np.int32)
    areaperil_j_of_building = np.zeros(max(sample_size, 1), dtype=np.int32)
    # a dependent's conditions are its source's damage bins, 1-based like intensity bin ids, and
    # their distribution the source's damage pmf
    n_source_bins = cond_cdf.shape[1]
    source_bin_ids = np.arange(1, n_source_bins + 1).astype(np.int32)
    source_damage_pmf = np.empty(max(n_source_bins, 1), dtype=oasis_float)

    # work arrays for accumulate_hermite_coeffs_into
    hermite_he_lo = np.empty(HERMITE_TERMS, dtype=np.float64)
    hermite_he_hi = np.empty(HERMITE_TERMS, dtype=np.float64)

    # the samples drawn at once: every one of a building (every_sample), or a sparse item's live
    # ones of a building, with their (unread) damage bins, a dependent's source bins and their
    # correlated values
    every_sample = np.arange(max(sample_size, 1), dtype=np.int64)
    sample_idx_buf = np.empty(max(sample_size, 1), dtype=np.int64)
    sample_bin_buf = np.empty(max(sample_size, 1), dtype=np.int32)
    parent_bin_buf = np.zeros(max(sample_size, 1), dtype=np.int32)
    eps_buf = np.empty(max(sample_size, 1), dtype=np.float64)
    # per item of a coverage, its run of sample_losses, and write_losses' scratch
    max_items = losses.shape[1]
    run_offsets = np.empty(max_items, dtype=np.int64)
    run_lengths = np.empty(max_items, dtype=np.int64)
    run_code_starts = np.empty(max_items, dtype=np.int64)
    item_cursor = np.empty(max_items, dtype=np.int64)
    item_values = np.empty(max_items, dtype=oasis_float)
    sample_acc = np.empty(max(sample_size, 1), dtype=np.float64)

    return _compute_event_losses_core(compute_info,
                                      coverages,
                                      coverage_ids,
                                      items_event_data,
                                      items,
                                      item_mixed_cdf_groups,
                                      mixed_cdf_group_areaperil_ja_offsets,
                                      mixed_cdf_group_areaperil_ja_cell_groups,
                                      mixed_cdf_group_areaperil_ja_member_blocks,
                                      mixed_cdf_group_areaperil_haz_arr_i,
                                      mixed_cdf_group_areaperil_event_rp,
                                      blended_areaperil_ja_offsets,
                                      blended_areaperil_ja_probabilities,
                                      mixed_cdf_cache_tag,
                                      mixed_cdf_cache,
                                      mixed_cdf_cache_nbins,
                                      mixed_cdf_group_hermite,
                                      max_areaperils_per_item,
                                      sample_size,
                                      haz_pdf,
                                      haz_arr_ptr,
                                      vuln_cdf,
                                      vuln_cdf_lengths,
                                      cond_cdf,
                                      cond_cdf_lengths,
                                      vuln_member_ja_offsets,
                                      vuln_member_ja_rows,
                                      vuln_member_ja_weights,
                                      damage_bins,
                                      losses,
                                      vuln_adj,
                                      haz_eps_ij,
                                      damage_eps_ij,
                                      norm_inv_parameters,
                                      norm_inv_cdf,
                                      norm_cdf,
                                      vuln_z_unif,
                                      haz_z_unif,
                                      byte_mv,
                                      dynamic_footprint,
                                      intensity_bin_peril_ids,
                                      intensity_bins,
                                      sample_losses,
                                      loss_correlation_by_item,
                                      hermite_coeffs,
                                      vuln_rndms_base,
                                      haz_rndms_base,
                                      vuln_seeds,
                                      haz_seeds,
                                      vuln_draw_scratch,
                                      haz_draw_scratch,
                                      perm_scratch,
                                      event_sample_codes,
                                      event_sample_areaperil_j,
                                      event_sample_bins,
                                      event_sample_vuln,
                                      event_sample_haz,
                                      coverage_has_dependents,
                                      compute_depth,
                                      source_damage_bin_stack,
                                      source_eff_damage_cdf_stack,
                                      source_eff_damage_cdf_len_stack,
                                      bin_slot,
                                      haz_pdf_prob_buf,
                                      haz_bin_id_buf,
                                      haz_cdf_buf,
                                      row_cdf_buf,
                                      row_lengths,
                                      row_stamp,
                                      areaperil_j_live,
                                      areaperil_j_stamp,
                                      areaperil_j_to_haz_start,
                                      areaperil_j_to_Nhaz_bins,
                                      areaperil_j_to_member_block,
                                      cdf_acc,
                                      mix_acc,
                                      curve_buf,
                                      mixed_eff_damage_cdf,
                                      areaperil_j_single,
                                      areaperil_j_of_building,
                                      source_bin_ids,
                                      source_damage_pmf,
                                      hermite_he_lo,
                                      hermite_he_hi,
                                      every_sample,
                                      sample_idx_buf,
                                      sample_bin_buf,
                                      parent_bin_buf,
                                      eps_buf,
                                      run_offsets,
                                      run_lengths,
                                      run_code_starts,
                                      item_cursor,
                                      item_values,
                                      sample_acc)


@nb.njit(cache=True, fastmath=True)
def process_areaperils_in_footprint(event_footprint,
                                    areaperil_id_ind,
                                    dynamic_footprint,
                                    ap_inds,
                                    event_rps,
                                    haz_arr_ptr):
    """Process areaperils in the footprint, filtering to those with vulnerability functions.

    Writes into pre-allocated arrays (ap_inds, event_rps, haz_arr_ptr) that are
    owned by the caller and reused across events.

    The buffer stores the dense areaperil index (from `areaperil_id_ind`) rather than
    the raw `areaperil_id`, so downstream consumers (reconstruct_coverages) can index
    `item_map_ja_offsets` directly and skip a second id_index lookup.

    Args:
        event_footprint (np.array[Event or footprint_event_dtype]): footprint entries.
        areaperil_id_ind (np.array): id_index structure for known areaperil_ids.
        dynamic_footprint (bool): true if there is dynamic_footprint.
        ap_inds (np.array[uint32]): pre-allocated output buffer for dense areaperil indices.
        event_rps (np.array[int32]): pre-allocated output buffer for return periods (dynamic only).
        haz_arr_ptr (np.array[int64]): pre-allocated output buffer for hazard pdf offsets.

    Returns:
        Nhaz_arr_this_event (int): number of areaperils stored. If zero, no items have losses.
        haz_pdf (np.array[haz_arr_type]): hazard intensity pdf (freshly sliced).
    """
    footprint_i = 0
    last_areaperil_id = nb_areaperil_int(0)
    last_areaperil_id_start = nb_int64(0)
    haz_arr_i = 0

    Nevent_footprint_entries = len(event_footprint)
    haz_pdf = np.empty(Nevent_footprint_entries, dtype=haz_arr_type)  # max size

    arr_ptr_start = 0
    arr_ptr_end = 0
    haz_arr_ptr[0] = 0

    while footprint_i <= Nevent_footprint_entries:

        if footprint_i < Nevent_footprint_entries:
            areaperil_id = event_footprint[footprint_i]['areaperil_id']
        else:
            areaperil_id = nb_areaperil_int(0)

        if areaperil_id != last_areaperil_id:
            # one areaperil_id is completed

            if last_areaperil_id > 0:
                ap_ind = id_index_get_idx(areaperil_id_ind, last_areaperil_id)
                if ap_ind != ID_INDEX_NOT_FOUND:
                    # if items with this areaperil_id exist, store its dense index
                    ap_inds[haz_arr_i] = ap_ind
                    haz_arr_i += 1

                    # store the hazard intensity pdf
                    arr_ptr_end = arr_ptr_start + (footprint_i - last_areaperil_id_start)
                    haz_pdf['probability'][arr_ptr_start: arr_ptr_end] = event_footprint['probability'][last_areaperil_id_start: footprint_i]
                    haz_pdf['intensity_bin_id'][arr_ptr_start: arr_ptr_end] = event_footprint['intensity_bin_id'][last_areaperil_id_start: footprint_i]
                    if dynamic_footprint is not None:
                        haz_pdf['intensity'][arr_ptr_start: arr_ptr_end] = event_footprint['intensity'][last_areaperil_id_start: footprint_i]
                        # only the first row's return period is the event's (see dynamic_footprint.collapse)
                        event_rps[haz_arr_i - 1] = nb_int32(event_footprint[last_areaperil_id_start]['return_period'])

                    haz_arr_ptr[haz_arr_i] = arr_ptr_end
                    arr_ptr_start = arr_ptr_end

            last_areaperil_id = areaperil_id
            last_areaperil_id_start = footprint_i

        footprint_i += 1

    Nhaz_arr_this_event = haz_arr_i

    return (Nhaz_arr_this_event,
            haz_pdf[:arr_ptr_end])


@nb.njit(cache=True, fastmath=True)
def reconstruct_coverages(compute_info,
                          ap_inds,
                          Nhaz_arr_this_event,
                          haz_arr_ptr,
                          event_rps,
                          item_map_ja_offsets,
                          item_map_ja_mixed_cdf_group_idxs,
                          item_map_ja_mixed_cdf_group_areaperil_idxs,
                          items,
                          mixed_cdf_group_areaperil_ja_offsets,
                          mixed_cdf_group_items_ja_offsets,
                          mixed_cdf_group_items_ja_item_idxs,
                          mixed_cdf_group_areaperil_haz_arr_i,
                          mixed_cdf_group_areaperil_event_rp,
                          mixed_cdf_group_event_tag,
                          event_tag,
                          event_mixed_cdf_groups,
                          mixed_cdf_group_needs_haz_rng,
                          coverages,
                          compute,
                          haz_seeds,
                          haz_peril_correlation_groups,
                          haz_corr_seeds,
                          vuln_seeds,
                          damage_peril_correlation_groups,
                          damage_corr_seeds,
                          dynamic_footprint,
                          byte_mv,
                          items_event_data,
                          group_seq_rng_index,
                          hazard_group_seq_rng_index,
                          coverage_source_id,
                          coverage_dependents_ja_offsets,
                          coverage_dependents_ja_data,
                          compute_depth,
                          compute_footprint_order,
                          dependency_dfs_stack,
                          source_item_idx,
                          item_idx_to_item_j,
                          ):
    """Register each item to its coverage and prepare per-item event data for loss computation.

    First, for each areaperil present in the event footprint, visits every group areaperil that
    reads it (item map order: areaperil, vulnerability, group) and records its hazard row (and
    return period). A group's areaperils are marked as having no hazard on its first visit, so the
    ones its walk never reaches stay at -1. The walk is over item mixed_cdf_groups, not items, so its cost
    does not grow with the number of items sharing a blended areaperil.

    Then every item of each group reached, groups in the order they were reached:

      1. Computes deterministic hash-based random seeds for damage sampling, using group_id.
         Seeds are deduplicated via pre-allocated arrays indexed by sequential group ids.
      2. Maps the item to its coverage structure, tracking the start offset and count.
      3. Stores per-item event data (item_idx, rng_index, ...) in items_event_data, with a hazard
         rng row if any of its areaperils' hazard is non-deterministic.

    An item is thus registered when any of its areaperils has hazard. With K = 1, and one group per
    (areaperil, vulnerability), items are registered in the item map order they always were.

    Args:
        compute_info (gulmc_compute_info_type): computation state; coverage_i, coverage_n,
          and event_id fields are read/written.
        ap_inds (np.array[uint32]): dense areaperil indices present in the event footprint
          (from process_areaperils_in_footprint), length >= Nhaz_arr_this_event.
        Nhaz_arr_this_event (int): number of valid entries in ap_inds.
        haz_arr_ptr (np.array[int64]): per-areaperil offsets into the event hazard pdf; the
          number of hazard intensity bins for areaperil ap_i is haz_arr_ptr[ap_i+1] - haz_arr_ptr[ap_i].
          A count of 1 means the hazard is deterministic and no hazard rng row is needed.
        event_rps (np.array[int32]): parallel array of return periods per areaperil (dynamic only).
        item_map_ja_offsets (np.array[oasis_int]): item map CSR offsets (N_areaperil + 1).
        item_map_ja_mixed_cdf_group_idxs (np.array[oasis_int]): mixed cdf group index per entry.
        item_map_ja_mixed_cdf_group_areaperil_idxs (np.array[oasis_int]): group areaperil index per entry.
        items (np.ndarray): items table merged with correlation parameters, containing
          group_id, hazard_group_id, coverage_id, group_seq_id, hazard_group_seq_id, etc.
        mixed_cdf_group_areaperil_ja_offsets (np.array[oasis_int]): per mixed cdf group, its run of group areaperils.
        mixed_cdf_group_items_ja_offsets (np.array[oasis_int]): per mixed cdf group, its run of items.
        mixed_cdf_group_items_ja_item_idxs (np.array[oasis_int]): the items of each group, in item order.
        mixed_cdf_group_areaperil_haz_arr_i (np.array[oasis_int]): written: hazard row per group areaperil of
          a group reached, -1 when its areaperil has no hazard this event.
        mixed_cdf_group_areaperil_event_rp (np.array[int32]): written: return period per group areaperil
          (dynamic footprint only).
        mixed_cdf_group_event_tag (np.array[int64]): per mixed cdf group, the event_tag it was last reached in.
        event_tag (int): distinct for every event, so a group is reached once per event without
          resetting mixed_cdf_group_event_tag.
        event_mixed_cdf_groups (np.array[int32]): scratch, of one entry per group: the groups reached this
          event, in the order reached.
        mixed_cdf_group_needs_haz_rng (np.array[bool]): scratch per group: whether one of its areaperils'
          hazard is non-deterministic this event, under full Monte Carlo.
        coverages (numpy.array[coverage_type]): coverage data indexed by coverage_id. On entry,
          cur_items is non-zero only for the coverages in ``compute[:compute_info['coverage_n']]``
          (the previous event's), which are the only ones reset here.
        compute (numpy.array[int]): output buffer for the list of coverage_ids to be computed. Must
          still hold the previous event's list on entry, see ``coverages``.
        haz_seeds (numpy.array[int]): output buffer for hazard intensity random seeds.
        haz_peril_correlation_groups (numpy.array[int]): unique peril correlation groups for hazard.
        haz_corr_seeds (numpy.array[int]): output buffer for hazard correlation seeds.
        vuln_seeds (numpy.array[int]): output buffer for damage random seeds.
        damage_peril_correlation_groups (numpy.array[int]): unique peril correlation groups for damage.
        damage_corr_seeds (numpy.array[int]): output buffer for damage correlation seeds.
        dynamic_footprint (None or object): None if no dynamic footprint, otherwise truthy.
        byte_mv (numpy.array[byte]): output byte buffer, may be resized if needed.
        items_event_data (numpy.array[items_MC_data_type]): per-item event data buffer, reused
          across events and grown (doubled) when an event needs more room. Its contents on entry
          are irrelevant: every field read for an item is written for this event first.
        group_seq_rng_index (numpy.array[int64]): pre-allocated array of size n_unique_groups,
          used for O(1) group_id to rng_index mapping. All NO_RNG_INDEX on entry; only the
          entries of this event's groups are set, and they are restored before returning.
        coverage_source_id (numpy.array): source coverage_id per coverage_id, 0 for a root.
        coverage_dependents_ja_offsets (numpy.array[oasis_int]): CSR offsets into
          ``coverage_dependents_ja_data``, indexed by source coverage_id.
        coverage_dependents_ja_data (numpy.array): dependent coverage ids, grouped by source.
        compute_depth (numpy.array[int32]): written per emitted coverage: its depth in the
          dependency forest, 0 for a root.
        compute_footprint_order (numpy.array): scratch holding the present coverages in footprint
          order while ``compute`` is rewritten into DFS order.
        dependency_dfs_stack (numpy.array[int64]): scratch (coverage_id, depth) work stack for the
          DFS reorder.
        source_item_idx (numpy.array[int64]): per item, the index into ``items`` of its source item,
          or -1 when it has none.
        item_idx_to_item_j (numpy.array[oasis_int]): scratch mapping an item index to its position
          within its coverage for this event, used to pair a dependent item with its source item.
        hazard_group_seq_rng_index (numpy.array[int64]): pre-allocated array of size
          n_unique_haz_groups, for hazard_group_id to rng_index mapping. Same contract as
          ``group_seq_rng_index``.

    Returns:
        tuple: (items_event_data, rng_index, hazard_rng_index, byte_mv)
          - items_event_data (numpy.array[items_MC_data_type]): per-item data including
            item_idx, rng_index, hazard_rng_index. The buffer passed in, or a grown copy of it: the
            caller must keep the returned one for the next event.
          - rng_index (int): number of unique damage random seeds generated.
          - hazard_rng_index (int): number of unique hazard random seeds generated. Only hazard
            groups touching a non-deterministic areaperil (and only under full Monte Carlo, not
            effective damageability) are counted, so this is 0 for events with no hazard
            uncertainty. Items not assigned a hazard rng row carry the NO_RNG_INDEX sentinel in
            items_event_data['hazard_rng_index'].
          - byte_mv (numpy.array[byte]): output buffer, possibly resized.
    """
    # init data structures
    # only the previous event's coverages can have cur_items > 0 (see ``coverages``): reset just those
    for position in range(compute_info['coverage_n']):
        coverages[compute[position]]['cur_items'] = 0
    rng_index = 0
    hazard_rng_index = 0
    compute_i = 0
    items_data_i = 0

    # for each areaperil_id, loop over all vulnerability functions used in that areaperil_id and,
    # for each item:
    #  - compute the seeds for the hazard intensity sampling and for the damage sampling
    #  - store data for later processing (hazard cdf index, etc.)
    # Pass 1: for each areaperil with hazard this event, record its hazard row (and return period)
    # in the group areaperils that read it. A group is registered on its first visit, its group
    # areaperils first marked as having no hazard, so those its walk never reaches stay at -1.
    n_event_mixed_cdf_groups = 0
    for ap_i in range(Nhaz_arr_this_event):
        ap_ind = ap_inds[ap_i]
        # A hazard rng row is only consumed when the hazard intensity is non-deterministic
        # (more than one intensity bin for this areaperil) and we are running full Monte Carlo.
        ap_needs_haz_rng = (not compute_info['effective_damageability']) and (haz_arr_ptr[ap_i + 1] - haz_arr_ptr[ap_i] > 1)
        for item_pos in range(item_map_ja_offsets[ap_ind], item_map_ja_offsets[ap_ind + 1]):
            mixed_cdf_group_idx = item_map_ja_mixed_cdf_group_idxs[item_pos]
            if mixed_cdf_group_event_tag[mixed_cdf_group_idx] != event_tag:
                mixed_cdf_group_event_tag[mixed_cdf_group_idx] = event_tag
                event_mixed_cdf_groups[n_event_mixed_cdf_groups] = mixed_cdf_group_idx
                n_event_mixed_cdf_groups += 1
                mixed_cdf_group_needs_haz_rng[mixed_cdf_group_idx] = False
                for mixed_cdf_group_areaperil_idx in range(mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group_idx],
                                                           mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group_idx + 1]):
                    mixed_cdf_group_areaperil_haz_arr_i[mixed_cdf_group_areaperil_idx] = -1
            mixed_cdf_group_areaperil_idx = item_map_ja_mixed_cdf_group_areaperil_idxs[item_pos]
            mixed_cdf_group_areaperil_haz_arr_i[mixed_cdf_group_areaperil_idx] = ap_i
            if dynamic_footprint is not None:
                mixed_cdf_group_areaperil_event_rp[mixed_cdf_group_areaperil_idx] = event_rps[ap_i]
            if ap_needs_haz_rng:
                mixed_cdf_group_needs_haz_rng[mixed_cdf_group_idx] = True

    # Pass 2: register the items of each group reached, in the order the groups were reached and
    # each group's items in item order. With K = 1 and one group per (areaperil, vulnerability), that
    # is the item map's (areaperil, vulnerability, item) order.
    for event_mixed_cdf_group_i in range(n_event_mixed_cdf_groups):
        mixed_cdf_group_idx = event_mixed_cdf_groups[event_mixed_cdf_group_i]
        for mixed_cdf_group_item_pos in range(mixed_cdf_group_items_ja_offsets[mixed_cdf_group_idx],
                                              mixed_cdf_group_items_ja_offsets[mixed_cdf_group_idx + 1]):
            item_idx = mixed_cdf_group_items_ja_item_idxs[mixed_cdf_group_item_pos]
            coverage_id = items[item_idx]['coverage_id']
            coverage = coverages[coverage_id]
            # if this group_id was not seen yet, process it.
            # it assumes that hash only depends on event_id and group_id
            # and that only 1 event_id is processed at a time.
            # Use sequential index for array-based lookup instead of Dict
            group_seq_id = items[item_idx]['group_seq_id']
            # Signed: magnitude is the count, negative means keep the buildings separate.
            # The signed form goes on to items_event_data for the writer.
            item_n_buildings_signed = items[item_idx]['packed_buildings']
            if group_seq_rng_index[group_seq_id] == NO_RNG_INDEX:
                group_seq_rng_index[group_seq_id] = rng_index
                vuln_seeds[rng_index] = generate_hash(items[item_idx]['group_id'], compute_info['event_id'])
                this_rng_index = rng_index
                rng_index += 1
            else:
                this_rng_index = group_seq_rng_index[group_seq_id]

            if coverage['cur_items'] == 0:
                # no items were collected for this coverage yet: set up the structure.
                # Appended in footprint order; reordered into DFS order below when
                # coverage dependency is active.
                compute[compute_i], compute_i = coverage_id, compute_i + 1

                while items_event_data.shape[0] < items_data_i + coverage['max_items']:
                    # if items_data needs to be larger to store all the items, double it in size
                    temp_items_data = np.empty(items_event_data.shape[0] * 2, dtype=items_event_data.dtype)
                    temp_items_data[:items_data_i] = items_event_data[:items_data_i]
                    items_event_data = temp_items_data

                coverage['start_items'], items_data_i = items_data_i, items_data_i + coverage['max_items']

            # append the data of this item
            item_i = coverage['start_items'] + coverage['cur_items']
            # remember where this item landed within its coverage, so a dependent item can be
            # paired with its own source item rather than with whatever shares its position
            item_idx_to_item_j[item_idx] = coverage['cur_items']
            items_event_data[item_i]['item_idx'] = item_idx
            items_event_data[item_i]['item_id'] = items[item_idx]['item_id']
            items_event_data[item_i]['rng_index'] = this_rng_index
            # stored signed: the writer needs the sign to decide separate-vs-summed
            items_event_data[item_i]['packed_buildings'] = item_n_buildings_signed
            # the EFFECTIVE correlation: draw_correlation_samples applies it only when
            # do_correlation is set, and the writer must agree with what was drawn
            items_event_data[item_i]['damage_correlation_value'] = (
                items[item_idx]['damage_correlation_value'] if compute_info['do_correlation'] else 0.)
            # every sample computed, unless prepare_event_samples finds the item can skip its dead ones
            items_event_data[item_i]['sparse'] = 0
            if dynamic_footprint is not None:
                items_event_data[item_i]['intensity_adjustment'] = items[item_idx]['intensity_adjustment']
                items_event_data[item_i]['return_period'] = items[item_idx]['return_period']
            coverage['cur_items'] += 1

            # a hazard rng row only when one of its areaperils' hazard is non-deterministic;
            # otherwise the NO_RNG_INDEX sentinel, never used as an index (guarded at consumption)
            if mixed_cdf_group_needs_haz_rng[mixed_cdf_group_idx]:
                hazard_group_seq_id = items[item_idx]['hazard_group_seq_id']
                if hazard_group_seq_rng_index[hazard_group_seq_id] == NO_RNG_INDEX:
                    hazard_group_seq_rng_index[hazard_group_seq_id] = hazard_rng_index
                    haz_seeds[hazard_rng_index] = generate_hash_hazard(items[item_idx]['hazard_group_id'], compute_info['event_id'])
                    items_event_data[item_i]['hazard_rng_index'] = hazard_rng_index
                    hazard_rng_index += 1
                else:
                    items_event_data[item_i]['hazard_rng_index'] = hazard_group_seq_rng_index[hazard_group_seq_id]
            else:
                items_event_data[item_i]['hazard_rng_index'] = NO_RNG_INDEX

    # Pair each dependent item with its source item. A linked pair shares an areaperil, so both are
    # present together and item_idx_to_item_j holds a value written this event — which is why it
    # needs no reset. build_coverage_dependency_forest enforces that.
    if compute_info['do_coverage_dependency'] == 1:
        for position in range(compute_i):
            coverage = coverages[compute[position]]
            for item_j in range(coverage['cur_items']):
                item_i = coverage['start_items'] + item_j
                src_idx = source_item_idx[items_event_data[item_i]['item_idx']]
                items_event_data[item_i]['source_item_j'] = (
                    item_idx_to_item_j[src_idx] if src_idx >= 0 else -1)

    # Order the coverages to be computed. When coverage dependency is active, reorder the
    # present coverages (currently in footprint order) into DFS order: each root coverage
    # (source == 0) is immediately followed by its present dependent subtree, and each
    # entry's depth is recorded. Otherwise every coverage is an independent root (depth 0).
    if compute_info['do_coverage_dependency'] == 1:
        num_present_coverages = compute_i
        compute_footprint_order[:num_present_coverages] = compute[:num_present_coverages]  # footprint order snapshot
        write_index = 0
        for position in range(num_present_coverages):
            root_coverage_id = compute_footprint_order[position]
            # A coverage is a root when it has no source, or when its source coverage contributes
            # no item to this event: a paired item shares its source item's areaperil, so an absent
            # source coverage means every present item here is unpaired and belongs at depth 0.
            source_coverage_id = coverage_source_id[root_coverage_id]
            if source_coverage_id != 0 and coverages[source_coverage_id]['cur_items'] > 0:
                continue  # not a root: emitted as part of an ancestor's subtree
            stack_pointer = 0
            dependency_dfs_stack[stack_pointer, 0] = root_coverage_id
            dependency_dfs_stack[stack_pointer, 1] = 0
            stack_pointer += 1
            while stack_pointer > 0:
                stack_pointer -= 1
                coverage_id = dependency_dfs_stack[stack_pointer, 0]
                depth = dependency_dfs_stack[stack_pointer, 1]
                compute[write_index] = coverage_id
                compute_depth[write_index] = depth
                write_index += 1
                for dependent_pos in range(coverage_dependents_ja_offsets[coverage_id],
                                           coverage_dependents_ja_offsets[coverage_id + 1]):
                    dependent_coverage_id = coverage_dependents_ja_data[dependent_pos]
                    if coverages[dependent_coverage_id]['cur_items'] > 0:  # only descend into present dependents
                        dependency_dfs_stack[stack_pointer, 0] = dependent_coverage_id
                        dependency_dfs_stack[stack_pointer, 1] = depth + 1
                        stack_pointer += 1
        if write_index != num_present_coverages:
            # Every present coverage is reachable (root when its source is absent, emitted by its
            # source otherwise), so this is a logic error, not a data shape.
            raise RuntimeError(
                "coverage dependency: a present dependent coverage was not reachable from a "
                "present source coverage in this event; aborting rather than producing wrong losses."
            )
        compute_i = write_index
    else:
        for position in range(compute_i):
            compute_depth[position] = 0

    # restore the group -> rng index maps for the groups seen in this event only
    for position in range(compute_i):
        coverage = coverages[compute[position]]
        for item_j in range(coverage['cur_items']):
            item_idx = items_event_data[coverage['start_items'] + item_j]['item_idx']
            group_seq_rng_index[items[item_idx]['group_seq_id']] = NO_RNG_INDEX
            hazard_group_seq_rng_index[items[item_idx]['hazard_group_seq_id']] = NO_RNG_INDEX

    compute_info['coverage_i'] = 0
    compute_info['coverage_n'] = compute_i

    # The buffer has to hold one whole coverage, since a coverage is emitted whole, but not a
    # whole dependency subtree: a flush may fall between the coverages of a subtree.
    # the extra header is what an item's first block is reserved WITH, so a buffer sized to
    # exactly one block would reject that reservation on an empty buffer, forever
    required_bytes = compute_info['max_bytes_per_item']
    for position in range(compute_i):
        coverage_bytes = coverages[compute[position]]['cur_items'] * compute_info['max_bytes_per_item']
        if coverage_bytes > required_bytes:
            required_bytes = coverage_bytes
    byte_mv = adjust_byte_mv_size(byte_mv, required_bytes)

    generate_correlated_hash_vector(haz_peril_correlation_groups, compute_info['event_id'], haz_corr_seeds)
    generate_correlated_hash_vector(damage_peril_correlation_groups, compute_info['event_id'], damage_corr_seeds)

    return (items_event_data,
            rng_index,
            hazard_rng_index,
            byte_mv)


if __name__ == '__main__':
    kwargs = {
        'alloc_rule': 1,
        'debug': 0,
        'file_in': './static/events_p.bin',
        'file_out': '/dev/null',
        'loss_threshold': 0.0,
        'sample_size': 10,
        'effective_damageability': False,
        'ignore_correlation': False,
        'ignore_haz_correlation': False,
        'ignore_file_type': set(),
        'data_server': False,
        'max_cached_vuln_cdf_size_MB': 200,
        'peril_filter': None,
        'random_generator': 2,
        'run_dir': '.',
        'model_df_engine': 'oasis_data_manager.df_reader.reader.OasisPandasReader',
        'dynamic_footprint': False}
    run(**kwargs)


@nb.njit(cache=True)
def prepare_event_samples(compute_info, coverages, coverage_ids, compute_depth, coverage_source_id,
                          items_event_data, items, item_mixed_cdf_groups,
                          ap_inds, Nhaz_arr_this_event, item_map_ja_offsets, item_map_ja_mixed_cdf_group_areaperil_idxs,
                          mixed_cdf_group_areaperil_ja_offsets, mixed_cdf_group_areaperil_haz_arr_i,
                          mixed_cdf_group_areaperil_event_rp, dynamic_footprint,
                          blended_areaperil_ja_offsets, blended_areaperil_placement_ja_offsets,
                          blended_areaperil_placement_ja_areaperil_j,
                          blended_areaperil_ja_placement_offsets, blended_areaperil_ja_placements,
                          placement_item_ind, placement_item_locations, placement_item_code_starts,
                          placement_location_ja_offsets, placement_location_ja_placements,
                          placement_location_ja_code_offsets, placement_location_ja_codes,
                          sample_size, n_buildings_max, vuln_seeds, vuln_rndms_base, n_vuln_rng,
                          haz_seeds, haz_rndms_base, n_haz_rng):
    """List the live samples of the event's blended items, with their base random values.

    An item whose blended areaperil has several placements usually has most of its samples on
    areaperils with no hazard this event, which lose nothing. Such an item is made sparse
    (``items_event_data['sparse']``): only its live samples are computed. They are found from the
    footprint rather than by visiting every sample: the group areaperils with hazard this event,
    then the placements that read them (``blended_areaperil_ja_placements``), then the samples of
    the item's buildings standing there (the inverted placements of its location).

    In a debug run, which writes every sample's random values, every sample of such an item is
    listed, live or not. An item of a single placement is not made sparse: it is live or dead whole. A dependent of
    coverage dependency shares its source's areaperil, so both are sparse or neither: a sparse
    source records its live samples' damage bins (``event_sample_bins``), and its dependent reads
    the bin at each of its own live samples, "no damage" where its source has none.

    A sample's base random values are its rng group's, per building (see ``draw_keyed_samples``),
    so every item of the group reads the same ones.

    Returns:
        tuple: per live sample, ``event_sample_codes`` (int64), its item code ``(building - 1) * S +
        sample``, each sparse item's run ascending; ``event_sample_areaperil_j`` (int32), the item
        areaperil it reads; ``event_sample_bins`` (int32, zeroed), where a source records its damage
        bins; ``event_sample_vuln`` and ``event_sample_haz`` (float64), its damage and
        hazard base random values (the latter 0 when the item draws no hazard); and the most sample
        losses one coverage of the event holds, its sparse items' live samples and N * S for each
        of its other items.
    """
    # the group areaperils with hazard this event, ascending, so that a group's are one run
    n_touched = 0
    for ap_i in range(Nhaz_arr_this_event):
        ap_ind = ap_inds[ap_i]
        n_touched += item_map_ja_offsets[ap_ind + 1] - item_map_ja_offsets[ap_ind]
    touched = np.empty(n_touched, dtype=np.int64)
    n_touched = 0
    for ap_i in range(Nhaz_arr_this_event):
        ap_ind = ap_inds[ap_i]
        for item_pos in range(item_map_ja_offsets[ap_ind], item_map_ja_offsets[ap_ind + 1]):
            touched[n_touched] = item_map_ja_mixed_cdf_group_areaperil_idxs[item_pos]
            n_touched += 1
    touched.sort()

    # per live sample: its code, areaperil, and the keys of its damage and hazard draws,
    # (rng group, building, sample) as rng_index * B * S + code (-1 for no hazard draw)
    capacity = 1024
    codes = np.empty(capacity, dtype=np.int64)
    codes_j = np.empty(capacity, dtype=np.int32)
    vuln_keys = np.empty(capacity, dtype=np.int64)
    haz_keys = np.empty(capacity, dtype=np.int64)
    n_codes = 0
    rng_stride = n_buildings_max * sample_size
    # per rng group, the buildings 1..n drawn as full blocks
    vuln_full = np.zeros(max(n_vuln_rng, 1), dtype=np.int64)
    haz_full = np.zeros(max(n_haz_rng, 1), dtype=np.int64)
    do_coverage_dependency = compute_info['do_coverage_dependency'] == 1
    # the most sample losses one coverage holds (see compute_event_losses' sample_losses)
    max_coverage_samples = 1
    for position in range(compute_info['coverage_n']):
        coverage_id = coverage_ids[position]
        coverage = coverages[coverage_id]
        coverage_is_dependent = do_coverage_dependency and compute_depth[position] > 0
        coverage_samples = 0
        for item_j in range(coverage['cur_items']):
            item_i = coverage['start_items'] + item_j
            item_idx = items_event_data[item_i]['item_idx']
            item = items[item_idx]
            n_buildings = abs(items_event_data[item_i]['packed_buildings'])
            rng = items_event_data[item_i]['rng_index']
            haz_rng = items_event_data[item_i]['hazard_rng_index']
            blend = item['blended_areaperil_idx']
            n_placements = blended_areaperil_placement_ja_offsets[blend + 1] - blended_areaperil_placement_ja_offsets[blend]
            if n_placements <= 1:
                vuln_full[rng] = max(vuln_full[rng], n_buildings)
                if haz_rng >= 0:
                    haz_full[haz_rng] = max(haz_full[haz_rng], n_buildings)
                coverage_samples += n_buildings * sample_size
                continue

            rank = id_index_get_idx(placement_item_ind, item_idx)
            location = placement_item_locations[rank]
            code_start = placement_item_code_starts[rank]
            code_end = code_start + n_buildings * sample_size
            location_start = placement_location_ja_offsets[location]
            location_end = placement_location_ja_offsets[location + 1]
            mixed_cdf_group = item_mixed_cdf_groups[item_idx]
            group_start = mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group]
            group_end = mixed_cdf_group_areaperil_ja_offsets[mixed_cdf_group + 1]
            areaperil_start = blended_areaperil_ja_offsets[blend]
            protection_rp = items_event_data[item_i]['return_period']
            item_first = n_codes
            if compute_info['debug']:
                # a debug run writes every sample's random values: list them all, each with the
                # areaperil its placement reads (-1 unplaced, or on a placement no areaperil covers)
                n_samples = n_buildings * sample_size
                if n_codes + n_samples > capacity:
                    while n_codes + n_samples > capacity:
                        capacity *= 2
                    codes = _resized(codes, n_codes, capacity)
                    codes_j = _resized(codes_j, n_codes, capacity)
                    vuln_keys = _resized(vuln_keys, n_codes, capacity)
                    haz_keys = _resized(haz_keys, n_codes, capacity)
                for k in range(n_samples):
                    codes[n_codes + k] = k
                    codes_j[n_codes + k] = -1
                placement_start = blended_areaperil_placement_ja_offsets[blend]
                for d in range(location_start, location_end):
                    run = placement_location_ja_codes[placement_location_ja_code_offsets[d]:placement_location_ja_code_offsets[d + 1]]
                    areaperil_j = blended_areaperil_placement_ja_areaperil_j[placement_start + placement_location_ja_placements[d]]
                    for c in range(np.searchsorted(run, code_start), np.searchsorted(run, code_end)):
                        codes_j[n_codes + np.int64(run[c]) - code_start] = areaperil_j
                n_codes += n_samples
            touched_i = np.searchsorted(touched, group_start) if not compute_info['debug'] else n_touched
            while touched_i < n_touched and touched[touched_i] < group_end:
                group_areaperil_idx = touched[touched_i]
                touched_i += 1
                if not group_areaperil_is_live(mixed_cdf_group_areaperil_haz_arr_i, mixed_cdf_group_areaperil_event_rp,
                                               group_areaperil_idx, protection_rp, dynamic_footprint):
                    continue
                areaperil_j = group_areaperil_idx - group_start
                for pos in range(blended_areaperil_ja_placement_offsets[areaperil_start + areaperil_j],
                                 blended_areaperil_ja_placement_offsets[areaperil_start + areaperil_j + 1]):
                    placement = blended_areaperil_ja_placements[pos]
                    d = location_start + np.searchsorted(placement_location_ja_placements[location_start:location_end],
                                                         placement)
                    if d == location_end or placement_location_ja_placements[d] != placement:
                        continue  # none of the location's samples stand there
                    run = placement_location_ja_codes[placement_location_ja_code_offsets[d]:placement_location_ja_code_offsets[d + 1]]
                    lo = np.searchsorted(run, code_start)
                    hi = np.searchsorted(run, code_end)
                    if n_codes + hi - lo > capacity:
                        while n_codes + hi - lo > capacity:
                            capacity *= 2
                        codes = _resized(codes, n_codes, capacity)
                        codes_j = _resized(codes_j, n_codes, capacity)
                        vuln_keys = _resized(vuln_keys, n_codes, capacity)
                        haz_keys = _resized(haz_keys, n_codes, capacity)
                    for c in range(lo, hi):
                        codes[n_codes] = np.int64(run[c]) - code_start
                        codes_j[n_codes] = areaperil_j
                        n_codes += 1
            if n_codes - item_first > 1:
                # several placements interleave their samples
                order = np.argsort(codes[item_first:n_codes], kind='mergesort')
                codes[item_first:n_codes] = codes[item_first:n_codes][order]
                codes_j[item_first:n_codes] = codes_j[item_first:n_codes][order]
            for k in range(item_first, n_codes):
                vuln_keys[k] = rng * rng_stride + codes[k]
                haz_keys[k] = haz_rng * rng_stride + codes[k] if haz_rng >= 0 else -1
            items_event_data[item_i]['sparse'] = 1
            items_event_data[item_i]['sample_start'] = item_first
            items_event_data[item_i]['sample_end'] = n_codes
            if coverage_is_dependent and items_event_data[item_i]['source_item_j'] >= 0:
                # its source, computed earlier in this DFS order and sparse too (same areaperil)
                items_event_data[item_i]['source_item_i'] = (coverages[coverage_source_id[coverage_id]]['start_items']
                                                             + items_event_data[item_i]['source_item_j'])
            coverage_samples += n_codes - item_first
        max_coverage_samples = max(max_coverage_samples, coverage_samples)

    return (codes[:n_codes], codes_j[:n_codes],
            np.zeros(n_codes, dtype=np.int32),
            draw_keyed_samples(vuln_keys[:n_codes], vuln_seeds, vuln_rndms_base, vuln_full, sample_size, n_buildings_max),
            draw_keyed_samples(haz_keys[:n_codes], haz_seeds, haz_rndms_base, haz_full, sample_size, n_buildings_max),
            max_coverage_samples)


@nb.njit(cache=True)
def _resized(a, n, capacity):
    """A copy of ``a``'s first ``n`` entries in a new array of ``capacity``."""
    out = np.empty(capacity, dtype=a.dtype)
    out[:n] = a[:n]
    return out


@nb.njit(cache=True)
def draw_keyed_samples(sample_keys, seeds, rndms_base, full_buildings, sample_size, n_buildings_max):
    """The base random value of each key ``rng_index * B * S + (building - 1) * S + sample``; 0 for -1.

    The first building reads the rng group's row, ``rndms_base``. Any other is drawn once per (rng
    group, building) present: the full Latin Hypercube block when the building is among the group's
    first ``full_buildings[rng_index]``, which some item computes whole; otherwise a partial block
    over the distinct samples asked for, the union of what the group's items need, so that they
    agree on each value.
    """
    values = np.zeros(sample_keys.shape[0], dtype=np.float64)
    keys = np.unique(sample_keys[sample_keys >= 0])
    key_values = np.empty(keys.shape[0], dtype=np.float64)
    perms = np.empty(max(sample_size, 1), dtype=np.float64)
    # the identity for _lh_philox_partial_block, which keeps it so
    identity_perms = np.arange(1, max(sample_size, 1) + 1).astype(np.float64)
    swapped = np.empty(max(sample_size, 1), dtype=np.int64)
    block = np.empty(max(sample_size, 1), dtype=np.float64)
    samples = np.empty(max(sample_size, 1), dtype=np.int64)
    run_start = 0
    while run_start < keys.shape[0]:
        run_key = keys[run_start] // sample_size
        run_end = run_start
        while run_end < keys.shape[0] and keys[run_end] // sample_size == run_key:
            samples[run_end - run_start] = keys[run_end] - run_key * sample_size
            run_end += 1
        rng = run_key // n_buildings_max
        building_i = run_key - rng * n_buildings_max
        m = run_end - run_start
        if building_i == 0:
            for i in range(m):
                key_values[run_start + i] = rndms_base[rng, samples[i]]
        else:
            seed = np.uint64(seeds[rng])
            k0 = np.uint32(seed & PHILOX_U32_MASK)
            k1 = np.uint32(seed >> PHILOX_SHIFT32)
            if building_i < full_buildings[rng]:
                _lh_philox_block(k0, k1, building_i, sample_size, perms, block)
                for i in range(m):
                    key_values[run_start + i] = block[samples[i]]
            else:
                _lh_philox_partial_block(k0, k1, building_i, sample_size, samples, m, identity_perms, swapped, block)
                for i in range(m):
                    key_values[run_start + i] = block[i]
        run_start = run_end
    for k in range(sample_keys.shape[0]):
        if sample_keys[k] >= 0:
            values[k] = key_values[np.searchsorted(keys, sample_keys[k])]
    return values
