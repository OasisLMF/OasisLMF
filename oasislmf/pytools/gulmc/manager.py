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
from select import select
import time

import numpy as np
import numba as nb
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import nb_areaperil_int, oasis_float, nb_oasis_int, oasis_int, correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.gul.common import MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX, NUM_IDX
from oasislmf.pytools.gul.core import compute_mean_loss, get_gul
from oasislmf.pytools.gul.manager import write_losses, adjust_byte_mv_size
from oasislmf.pytools.gul.random import (generate_correlated_hash_vector, generate_hash,
                                         generate_hash_hazard, get_corr_rval, get_random_generator)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.gulmc.common import (DAMAGE_TYPE_ABSOLUTE,
                                           DAMAGE_TYPE_DURATION,
                                           DAMAGE_TYPE_RELATIVE,
                                           NP_BASE_ARRAY_SIZE,
                                           NormInversionParameters,
                                           gul_header,
                                           gulSampleslevelHeader_size,
                                           gulSampleslevelRec_size,
                                           haz_arr_type, items_MC_data_type,
                                           gulmc_compute_info_type)
from oasislmf.pytools.common.id_index import get_idx as id_index_get_idx, NOT_FOUND as ID_INDEX_NOT_FOUND
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.ping import oasis_ping, oasis_ping_async
from oasislmf.utils.defaults import SERVER_UPDATE_TIME
from oasislmf.utils.exceptions import OasisException

logger = logging.getLogger(__name__)


CDF_CACHE_EMPTY = nb_int64(-1)
NO_RNG_INDEX = nb_int64(-1)


def validate_coverage_dependency(items, vuln_idx_to_cond_idx):
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
            areaperil_agg_vuln_idx, source_item_id).
        vuln_idx_to_cond_idx (np.ndarray): dense vuln idx -> conditional row, or -1 if not conditional.

    Raises:
        OasisException: if any of the three invariants is violated.
    """
    non_agg = items['areaperil_agg_vuln_idx'] < 0
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
        max_cached_vuln_cdf_size_MB (int, optional): size in MB of the in-memory cache to store and reuse vulnerability cdf. Defaults to 200.
        model_df_engine (str, optional): The engine to use when loading model dataframes. Defaults to OasisPandasReader.
        dynamic_footprint (bool, optional): if True, load the dynamic footprint data and adjust hazard intensities at
          runtime. Defaults to False.
        **kwargs: additional keyword arguments. socket_server (str) enables the progress ping and, when numeric, gives
          the port to override; analysis_pk is reported with each ping.

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
            gulmc_structure_exists, load_gulmc_structure, build_structures,
        )
        if gulmc_structure_exists(run_dir):
            logger.info("loading pre-computed gulmc structures (shared memory)")
            structures = load_gulmc_structure(run_dir)
        else:
            logger.info("building gulmc structures from input files")
            structures = build_structures(run_dir, ignore_file_type, peril_filter,
                                          dynamic_footprint, model_df_engine)

        items = structures['items']
        # coverages needs a writable copy: reconstruct_coverages writes cur_items / start_items per event
        coverages = structures['coverages'].copy()
        item_map_ja_areaperil_ids = structures['item_map_ja_areaperil_ids']
        item_map_ja_offsets = structures['item_map_ja_offsets']
        item_map_ja_vuln_ja_offsets = structures['item_map_ja_vuln_ja_offsets']
        item_map_ja_vuln_ja_item_idxs = structures['item_map_ja_vuln_ja_item_idxs']
        item_map_ja_id_ind = structures['item_map_ja_id_ind']
        item_cdf_group_idx = structures['item_cdf_group_idx']
        n_cdf_groups = structures['n_cdf_groups']
        areaperil_agg_vuln_idx_ja_offsets = structures['areaperil_agg_vuln_idx_ja_offsets']
        areaperil_agg_vuln_idx_ja_data = structures['areaperil_agg_vuln_idx_ja_data']
        damage_bins = structures['damage_bins']
        vuln_adj = structures['vuln_adj']
        vuln_array = structures['vuln_array']
        conditional_vuln_array = structures['conditional_vuln_array']
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

        # coverage dependency is active only when at least one dependent coverage exists;
        # otherwise the forest is empty and gulmc behaves exactly as before.
        do_coverage_dependency = bool(coverage_dependents_ja_data.shape[0] > 0)

        # Dependency is opt-in per location via the keys: a coverage is a dependent only when the
        # key server returns its source coverage type at the same areaperil (resolved in
        # gul_inputs; a mismatched/absent source demotes the coverage to independent). A dependent
        # coverage must use a conditional (damage-transition) vulnerability from the
        # conditional_vulnerability file; an independent coverage must use a normal hazard-indexed
        # one. `vuln_idx_to_cond_idx[vulnerability_idx] >= 0` iff the vulnerability is conditional.
        if do_coverage_dependency or conditional_vuln_array.shape[0] > 0:
            if do_coverage_dependency:
                logger.info(f"coverage dependency: switched ON ({coverage_dependents_ja_data.shape[0]} dependent coverages).")
            validate_coverage_dependency(items, vuln_idx_to_cond_idx)

        Nvulnerability, Ndamage_bins_max, Nintensity_bins = vuln_array.shape
        Nperil_correlation_groups = unique_peril_correlation_groups.shape[0]
        logger.info(f"Detected {Nperil_correlation_groups} peril correlation groups.")

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
        # per-depth, per-item stacks holding the source coverage's result while its subtree is
        # computed. Indexed [depth, item_j] so a dependent item reads its source's matching peril
        # (same item column, since source and dependent span the same areaperils in the same order).
        # Full MC stores the per-sample sampled damage bin; effective damageability stores the eff-damage CDF.
        max_items_per_coverage = int(np.max(coverages[1:]['max_items']))
        source_damage_bin_stack = np.zeros(
            (max_dependency_depth + 1, max_items_per_coverage, sample_size if sample_size > 0 else 1), dtype=np.int32)
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
                                                df_engine=model_df_engine, areaperil_ids=item_map_ja_areaperil_ids))
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        # set up streams
        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        select_stream_list = [stream_out]

        # prepare output stream
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)
        # create the array to store the seeds
        haz_seeds = np.zeros(n_unique_haz_groups, dtype=correlations_dtype['hazard_group_id'])
        vuln_seeds = np.zeros(n_unique_groups, dtype=items_dtype['group_id'])
        # Pre-allocated arrays for group_id -> rng_index mapping (replaces per-event Numba Dicts)
        group_seq_rng_index = np.empty(n_unique_groups, dtype=np.int64)
        hazard_group_seq_rng_index = np.empty(n_unique_haz_groups, dtype=np.int64)

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
        losses = np.zeros((sample_size + NUM_IDX + 1, np.max(coverages[1:]['max_items'])), dtype=oasis_float)

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size

        # define vulnerability cdf cache size
        max_cached_vuln_cdf_size_bytes = max_cached_vuln_cdf_size_MB * 1024 * 1024  # cache size in bytes
        max_Nnumbers_cached_vuln_cdf = max_cached_vuln_cdf_size_bytes // oasis_float.itemsize  # total numbers that can fit in the cache
        max_Nvulnerability_cached_vuln_cdf = max_Nnumbers_cached_vuln_cdf // Ndamage_bins_max  # max number of vulnerability functions that can be stored in cache
        # number of vulnerability functions to be cached, rounded up to power of two for bitwise masking.
        # The cache must fit at least one full multi-block write for a single item — that is
        # 1 effective damage CDF + Nintensity_bins per-bin vuln CDFs — otherwise a block can
        # trample itself while wrapping around, producing silently corrupted CDF reads.
        Nvulns_cached = min(Nvulnerability * Nintensity_bins, max_Nvulnerability_cached_vuln_cdf)
        Nvulns_cached = max(Nvulns_cached, Nintensity_bins + 1)
        Nvulns_cached = 1 << (max(Nvulns_cached, 1) - 1).bit_length()  # next power of two
        cdf_cache_mask = np.int64(Nvulns_cached - 1)
        logger.info(f"max vulnerability cdf cache size is {max_cached_vuln_cdf_size_MB}MB")
        logger.info(
            f"generating a cache of shape ({Nvulns_cached}, {Ndamage_bins_max}) and size {Nvulns_cached * Ndamage_bins_max * oasis_float.itemsize / 1024 / 1024:8.3f}MB")

        # maximum bytes to be written in the output stream for 1 item
        event_footprint_obj = FootprintLayerClient if data_server else footprint_obj

        # intensity_bin_peril_ids and intensity_bins are pre-loaded from structures
        if not dynamic_footprint:
            dynamic_footprint = None

        compute_info = np.zeros(1, dtype=gulmc_compute_info_type)[0]

        compute_info['max_bytes_per_item'] = max_bytes_per_item
        compute_info['Ndamage_bins_max'] = Ndamage_bins_max
        compute_info['loss_threshold'] = loss_threshold
        compute_info['alloc_rule'] = alloc_rule
        compute_info['do_correlation'] = do_correlation
        compute_info['do_haz_correlation'] = do_haz_correlation
        compute_info['effective_damageability'] = effective_damageability
        compute_info['debug'] = debug
        compute_info['do_coverage_dependency'] = do_coverage_dependency

        # default random values array for sample_size==0 case
        haz_rndms_base = np.empty((1, sample_size), dtype='float64')
        vuln_rndms_base = np.empty((1, sample_size), dtype='float64')
        haz_eps_ij = np.empty((1, sample_size), dtype='float64')
        damage_eps_ij = np.empty((1, sample_size), dtype='float64')

        # Pre-allocate per-event footprint arrays (reused across events, no per-event allocation)
        # fp_ap_inds stores dense areaperil indices (from item_map_ja_id_ind) rather than raw
        # areaperil_ids, so reconstruct_coverages can index item_map_ja_offsets directly without
        # repeating the id_index lookup that was already done in process_areaperils_in_footprint.
        n_max_areaperils = len(item_map_ja_areaperil_ids)
        fp_ap_inds = np.empty(n_max_areaperils, dtype=np.uint32)
        fp_event_rps = np.empty(n_max_areaperils, dtype=np.int32)
        fp_haz_arr_ptr = np.empty(n_max_areaperils + 1, dtype=np.int64)

        # Pre-allocate CDF cache once (reused across events, no need to zero)
        cached_vuln_cdfs = np.empty((Nvulns_cached, Ndamage_bins_max), dtype=oasis_float)
        cdf_cache_tag = np.full(n_cdf_groups, CDF_CACHE_EMPTY, dtype=np.int64)
        cdf_cache_nbins = np.zeros(Nvulns_cached, dtype=np.int32)

        counter = 0
        timer = time.time()
        socket_server_val = kwargs.get('socket_server', 'False')
        ping = socket_server_val != 'False'
        ping_port = int(socket_server_val) if ping and str(socket_server_val).isdigit() else None
        while True:
            if not streams_in.readinto(event_id_mv):
                if ping:
                    ping_data = {"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)}
                    if ping_port is not None:
                        ping_data['port_override'] = ping_port
                    oasis_ping(ping_data)
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
                    counter += 1
                    logger.info(f"event {event_ids[0]} SKIPPED - no items")
                    continue

                items_event_data, rng_index, hazard_rng_index, byte_mv = reconstruct_coverages(
                    compute_info,
                    fp_ap_inds,
                    Nhaz_arr_this_event,
                    fp_haz_arr_ptr,
                    fp_event_rps,
                    item_map_ja_offsets,
                    item_map_ja_vuln_ja_offsets,
                    item_map_ja_vuln_ja_item_idxs,
                    items,
                    item_cdf_group_idx,
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
                    # generation of "base" random values for hazard intensity and vulnerability sampling.
                    haz_rndms_base = generate_rndm(haz_seeds[:hazard_rng_index], sample_size)
                    vuln_rndms_base = generate_rndm(vuln_seeds[:rng_index], sample_size)
                    if hazard_rng_index > 0:
                        haz_eps_ij = generate_rndm(haz_corr_seeds, sample_size, skip_seeds=1)
                    damage_eps_ij = generate_rndm(damage_corr_seeds, sample_size, skip_seeds=1)

                # Reset CDF cache lookup per event (cached_vuln_cdfs array is reused, no reallocation)
                cdf_cache_tag[:] = CDF_CACHE_EMPTY
                compute_info['cdf_cache_ctr'] = 0

                processing_done = False
                logger.info(f"event {event_ids[0]} STARTED")
                while not processing_done:
                    try:
                        processing_done = compute_event_losses(
                            compute_info,
                            coverages,
                            compute,
                            items_event_data,
                            items,
                            sample_size,
                            haz_pdf,
                            fp_haz_arr_ptr,
                            vuln_array,
                            conditional_vuln_array,
                            vuln_idx_to_cond_idx,
                            damage_bins,
                            cdf_cache_tag,
                            cdf_cache_nbins,
                            cdf_cache_mask,
                            cached_vuln_cdfs,
                            areaperil_agg_vuln_idx_ja_offsets,
                            areaperil_agg_vuln_idx_ja_data,
                            losses,
                            haz_rndms_base,
                            vuln_rndms_base,
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
                            compute_depth,
                            source_damage_bin_stack,
                            source_eff_damage_cdf_stack,
                            source_eff_damage_cdf_len_stack
                        )
                    except Exception:
                        data = {
                            "event_id": event_ids[0]
                        }
                        with open("event_error.json", "w") as f:
                            json.dump(data, f, default=str)

                        logger.error(f"event id={event_ids[0]} failed in summary")
                        raise
                    # write the losses to the output stream
                    write_start = 0
                    while write_start < compute_info['cursor']:
                        select([], select_stream_list, select_stream_list)
                        write_start += stream_out.write(memoryview(byte_mv[write_start: compute_info['cursor']]))

                logger.info(f"event {event_ids[0]} DONE")

            counter += 1
            if ping and time.time() - timer > SERVER_UPDATE_TIME:
                timer = time.time()
                ping_data = {"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)}
                if ping_port is not None:
                    ping_data['port_override'] = ping_port
                oasis_ping_async(ping_data)
                counter = 0

    return 0


@nb.njit(cache=True, fastmath=True)
def get_last_non_empty(cdf, bin_i):
    """Remove empty bucket from the end

    Args:
        cdf: cumulative distribution
        bin_i: last valid bin index

    Returns:
        last bin index with an increased in the cdf
    """
    last_prob = cdf[bin_i]
    while bin_i > 0 and cdf[bin_i - 1] == last_prob:
        bin_i -= 1
    return bin_i


@nb.njit(cache=True, fastmath=True)
def pdf_to_cdf(pdf, empty_cdf):
    """Return the cumulative distribution from the probality distribution

    Args:
        pdf (np.array[float]): probality distribution
        empty_cdf (np.array[float]): cumulative distribution buffer for output
    Returns:
         cdf (np.array[float]): here we return only the valid part if needed
    """
    cumsum = 0
    i = 0
    while i < pdf.shape[0]:
        cumsum += pdf[i]
        empty_cdf[i] = cumsum
        i += 1
        if cumsum >= 0.999999940:
            break
    i = get_last_non_empty(empty_cdf, i - 1)
    return empty_cdf[: i + 1]


@nb.njit(cache=True, fastmath=True)
def calc_eff_damage_cdf(vuln_pdf, haz_pdf, eff_damage_cdf_empty):
    """Calculate the covoluted cumulative distribution between vulnerability damage and hazard probability distribution

    Args:
        vuln_pdf (np.array[float]) : vulnerability damage probability distribution
        haz_pdf (np.array[float]): hazard probability distribution
        eff_damage_cdf_empty (np.array[float]): output buffer
    Returns:
        eff_damage_cdf (np.array[float]): cdf is stored in eff_damage_cdf_empty, here we return only the valid part if needed
    """
    eff_damage_cdf_cumsum = 0.
    damage_bin_i = 0
    while damage_bin_i < vuln_pdf.shape[1]:
        for haz_i in range(vuln_pdf.shape[0]):
            eff_damage_cdf_cumsum += vuln_pdf[haz_i, damage_bin_i] * haz_pdf[haz_i]

        eff_damage_cdf_empty[damage_bin_i] = eff_damage_cdf_cumsum
        damage_bin_i += 1
        if eff_damage_cdf_cumsum >= 0.999999940:
            break
    damage_bin_i = get_last_non_empty(eff_damage_cdf_empty, damage_bin_i - 1)
    return eff_damage_cdf_empty[:damage_bin_i + 1]


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
def get_gul_from_vuln_cdf(vuln_rval, vuln_cdf, Ndamage_bins, damage_bins, bin_scaling):
    # find the damage cdf bin in which the random value `vuln_rval` falls into
    vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins - 1)

    # compute ground-up losses; also return the sampled damage bin index (needed as the driving
    # signal for coverage dependency — captured here rather than re-derived from the loss).
    gul = get_gul(
        damage_bins['bin_from'][vuln_bin_idx],
        damage_bins['bin_to'][vuln_bin_idx],
        damage_bins['interpolation'][vuln_bin_idx],
        vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
        vuln_cdf[vuln_bin_idx],
        vuln_rval,
        bin_scaling,
    )
    return gul, vuln_bin_idx


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
def compute_haz_bin_id(haz_pdf_record, item, intensity_adjustment, dynamic_footprint,
                       intensity_bin_peril_ids, intensity_bins):
    """Resolve the hazard intensity bin ids for an item's hazard pdf records.

    With no dynamic footprint the intensity bin ids are read directly from the hazard pdf.
    With a dynamic footprint the recorded intensities are shifted by ``intensity_adjustment``,
    clamped to the valid range, and re-mapped to bin ids via the per-peril ``intensity_bins``
    lookup table.

    Args:
        haz_pdf_record (np.array[haz_arr_type]): hazard pdf records for this item's areaperil.
        item (np.void): the item record (needs peril_id when dynamic footprint is active).
        intensity_adjustment (int): intensity shift to apply under a dynamic footprint.
        dynamic_footprint (None or object): None if no dynamic footprint, otherwise truthy.
        intensity_bin_peril_ids (np.array[int32]): sorted unique encoded peril_ids.
        intensity_bins (np.array[int32, 2d]): [peril_idx, intensity_value] -> intensity_bin_id.

    Returns:
        np.array: intensity bin ids, one per hazard pdf record.
    """
    if dynamic_footprint is not None:
        # adjust intensity in dynamic footprint
        haz_intensity = haz_pdf_record['intensity']
        haz_intensity = haz_intensity - intensity_adjustment
        haz_bin_id = np.zeros_like(haz_pdf_record['intensity_bin_id'])
        peril_id = item['peril_id']
        # find peril index (linear scan of small array, typically 1-3 perils)
        peril_idx = nb_int32(0)
        for pi in range(intensity_bin_peril_ids.shape[0]):
            if intensity_bin_peril_ids[pi] == peril_id:
                peril_idx = nb_int32(pi)
                break
        max_intensity = nb_int32(intensity_bins.shape[1] - 1)
        for haz_bin_idx in range(haz_bin_id.shape[0]):
            intensity_val = haz_intensity[haz_bin_idx]
            if intensity_val < 0 or intensity_val > max_intensity:
                intensity_val = nb_int32(0)
            haz_bin_id[haz_bin_idx] = intensity_bins[peril_idx, intensity_val]
    else:
        haz_bin_id = haz_pdf_record['intensity_bin_id']
    return haz_bin_id


@nb.njit(cache=True, fastmath=True, inline='always')
def build_vuln_pdf(item, Nhaz_bins, haz_bin_id, vuln_array, Ndamage_bins_max,
                   areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data, vuln_pdf_empty):
    """Assemble the per-hazard-bin vulnerability pdf for an item.

    For a single (non-aggregate) vulnerability the pdf is read straight out of ``vuln_array``.
    For an aggregate vulnerability the weighted sum of the member vulnerabilities is built from
    the jagged arrays; bins with no probability collapse to 100% no-loss in the first damage bin,
    and the result is normalised by the total weight (or by the member count when all weights are
    zero).

    Args:
        item (np.void): the item record (uses areaperil_agg_vuln_idx / vulnerability_idx).
        Nhaz_bins (int): number of hazard bins for this item.
        haz_bin_id (np.array): intensity bin id per hazard record.
        vuln_array (np.array[float]): 3d vulnerability array
          (Nvulnerability, Ndamage_bins_max, Nintensity_bins).
        Ndamage_bins_max (int): maximum number of damage bins.
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): jagged array offsets.
        areaperil_agg_vuln_idx_ja_data (np.array): merged structured array with 'vuln_idx' and 'weight'.
        vuln_pdf_empty (np.array[float]): reusable buffer sliced to (Nhaz_bins, Ndamage_bins_max).

    Returns:
        np.array[float]: the vulnerability pdf, shape (Nhaz_bins, Ndamage_bins_max).
    """
    vuln_pdf = vuln_pdf_empty[:Nhaz_bins]
    vuln_pdf[:] = 0
    if item['areaperil_agg_vuln_idx'] >= 0:  # aggregate vulnerability — use jagged arrays
        tot_weights = 0.
        blk = item['areaperil_agg_vuln_idx']
        ptr = areaperil_agg_vuln_idx_ja_offsets[blk]
        n_sub = areaperil_agg_vuln_idx_ja_offsets[blk + 1] - ptr
        for j in range(n_sub):
            entry = areaperil_agg_vuln_idx_ja_data[ptr + j]
            vuln_i = entry['vuln_idx']
            weight = np.float64(entry['weight'])
            if weight > 0:
                tot_weights += weight
                for haz_i in range(Nhaz_bins):
                    has_prob = False
                    for damage_bin_i in range(Ndamage_bins_max):
                        if vuln_array[vuln_i, damage_bin_i, haz_bin_id[haz_i] - 1] > 0:
                            has_prob = True
                            vuln_pdf[haz_i, damage_bin_i] += vuln_array[vuln_i, damage_bin_i, haz_bin_id[
                                haz_i] - 1] * weight
                    if not has_prob:
                        # the pdf is all zeros, i.e. probability of no loss is 100%
                        # store it as 100% * weight in the first damage bin
                        vuln_pdf[haz_i, 0] += weight

        if tot_weights > 0:
            vuln_pdf /= tot_weights
        else:
            for j in range(n_sub):
                vuln_i = areaperil_agg_vuln_idx_ja_data[ptr + j]['vuln_idx']
                for haz_i in range(Nhaz_bins):
                    vuln_pdf[haz_i] += vuln_array[vuln_i, :, haz_bin_id[haz_i] - 1]
            vuln_pdf /= n_sub
    else:
        for haz_i in range(Nhaz_bins):
            vuln_pdf[haz_i] = vuln_array[item['vulnerability_idx'], :, haz_bin_id[haz_i] - 1]
    return vuln_pdf


@nb.njit(cache=True, fastmath=True, inline='always')
def resolve_item_cdfs(compute_info, cdf_group, do_calc_vuln_ptf, Nhaz_bins, item, haz_bin_id, haz_pdf_prob,
                      vuln_array, areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
                      cdf_cache_tag, cdf_cache_nbins, cdf_cache_mask, cached_vuln_cdfs,
                      vuln_pdf_empty, eff_damage_cdf_empty, haz_i_to_Ndamage_bins_empty, haz_i_to_vuln_cdf_empty,
                      is_dependent, conditional_vuln_array, vuln_idx_to_cond_idx):
    """Return an item's effective-damage CDF (and per-hazard-bin vuln CDFs), computing or reading the cache.

    On a cache miss the vulnerability pdf is (re)built, the effective damage CDF is computed and
    written to cache slot 0, and — when not running effective_damageability — each per-hazard-bin
    vulnerability CDF is computed and written to the following contiguous slots. On a cache hit the
    same CDFs are read back from those slots. ``compute_info['cdf_cache_ctr']``, ``cdf_cache_tag``,
    ``cdf_cache_nbins`` and ``cached_vuln_cdfs`` are mutated in place.

    The ``haz_i_to_*`` outputs are always returned (sliced from their reusable buffers) so the
    return type is stable; they are only populated when ``effective_damageability`` is False, which
    is also the only case in which the caller reads them.

    Coverage dependency: for a dependent item (``is_dependent``) the vuln pdf is assembled from the
    conditional (damage-transition) matrix ``conditional_vuln_array`` — column ``haz_i`` is the
    dependent's damage pmf given the source landed in damage bin ``haz_i`` — instead of the
    hazard-indexed ``vuln_array``. A dependent's CDFs depend on the source's damage this event, so
    they are event-specific and never cached (the caller forces ``do_calc_vuln_ptf`` for dependents).

    Returns:
        tuple: (eff_damage_cdf, haz_i_to_Ndamage_bins, haz_i_to_vuln_cdf).
    """
    haz_i_to_Ndamage_bins = haz_i_to_Ndamage_bins_empty[:Nhaz_bins]
    haz_i_to_vuln_cdf = haz_i_to_vuln_cdf_empty[:Nhaz_bins]

    if do_calc_vuln_ptf:  # cache miss — compute (and, unless dependent, cache) CDFs
        # we get the vuln_pdf, needed for effcdf and each cdf
        if is_dependent:
            vuln_pdf = vuln_pdf_empty[:Nhaz_bins]
            cond_idx = vuln_idx_to_cond_idx[item['vulnerability_idx']]
            for haz_i in range(Nhaz_bins):
                vuln_pdf[haz_i] = conditional_vuln_array[cond_idx, :, haz_i]
        else:
            vuln_pdf = build_vuln_pdf(item, Nhaz_bins, haz_bin_id, vuln_array, compute_info['Ndamage_bins_max'],
                                      areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data, vuln_pdf_empty)

        # calculate and cache all CDFs as a contiguous block (dependent CDFs are event-specific,
        # so they are computed but not cached)
        eff_damage_cdf = calc_eff_damage_cdf(vuln_pdf, haz_pdf_prob, eff_damage_cdf_empty)
        if not is_dependent:
            cdf_cache_tag[cdf_group] = compute_info['cdf_cache_ctr']
            # slot 0: effective damage CDF
            cache_idx = compute_info['cdf_cache_ctr'] & cdf_cache_mask
            cached_vuln_cdfs[cache_idx, :eff_damage_cdf.shape[0]] = eff_damage_cdf
            cdf_cache_nbins[cache_idx] = nb_int32(eff_damage_cdf.shape[0])
            compute_info['cdf_cache_ctr'] += 1

        if not compute_info['effective_damageability']:  # also (unless dependent) cache per-bin vuln CDFs
            for haz_i in range(Nhaz_bins):
                haz_i_to_Ndamage_bins[haz_i] = pdf_to_cdf(vuln_pdf[haz_i], haz_i_to_vuln_cdf[haz_i]).shape[0]
                if not is_dependent:
                    cache_idx = compute_info['cdf_cache_ctr'] & cdf_cache_mask
                    ndamage_bins = haz_i_to_Ndamage_bins[haz_i]
                    cached_vuln_cdfs[cache_idx, :ndamage_bins] = haz_i_to_vuln_cdf[haz_i][:ndamage_bins]
                    cdf_cache_nbins[cache_idx] = nb_int32(ndamage_bins)
                    compute_info['cdf_cache_ctr'] += 1

    else:  # cache hit — read CDFs from cache
        block_start = cdf_cache_tag[cdf_group]
        cache_idx = block_start & cdf_cache_mask
        eff_damage_cdf = cached_vuln_cdfs[cache_idx, :cdf_cache_nbins[cache_idx]]

        if not compute_info['effective_damageability']:
            for haz_i in range(Nhaz_bins):
                cache_idx = (block_start + 1 + haz_i) & cdf_cache_mask
                ndamage_bins = cdf_cache_nbins[cache_idx]
                haz_i_to_Ndamage_bins[haz_i] = ndamage_bins
                haz_i_to_vuln_cdf[haz_i][:ndamage_bins] = cached_vuln_cdfs[cache_idx, :ndamage_bins]

    return eff_damage_cdf, haz_i_to_Ndamage_bins, haz_i_to_vuln_cdf


@nb.njit(cache=True, fastmath=True, inline='always')
def draw_correlation_samples(compute_info, item, hazard_rng_index, rng_index, sample_size,
                             haz_rndms_base, vuln_rndms_base, haz_eps_ij, damage_eps_ij,
                             norm_inv_parameters, norm_inv_cdf, norm_cdf, vuln_adj,
                             haz_z_unif, vuln_z_unif):
    """Draw the (optionally correlated) hazard and damage random values for one item.

    Writes ``sample_size`` values into ``haz_z_unif`` and ``vuln_z_unif`` in place. When
    correlation is enabled for the relevant dimension the values are drawn through the Gaussian
    copula (``get_corr_rval``); otherwise the item's base random values are copied straight
    across. Hazard values are only drawn when ``hazard_rng_index >= 0`` (non-deterministic hazard
    under full Monte Carlo). For a single (non-aggregate) vulnerability the damage values are
    scaled by the per-vulnerability adjustment.

    Args:
        compute_info (gulmc_compute_info_type): computation state (do_haz_correlation, do_correlation).
        item (np.void): the item record (correlation values, peril_correlation_group, vuln idx).
        hazard_rng_index (int): index into haz_rndms_base, or < 0 if hazard is deterministic.
        rng_index (int): index into vuln_rndms_base for damage sampling.
        sample_size (int): number of random samples to draw.
        haz_rndms_base (np.array[float64]): base random values for hazard sampling.
        vuln_rndms_base (np.array[float64]): base random values for damage sampling.
        haz_eps_ij (np.array[float]): correlated random values for hazard sampling.
        damage_eps_ij (np.array[float]): correlated random values for damage sampling.
        norm_inv_parameters (NormInversionParameters): parameters for Gaussian inversion.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf lookup table.
        norm_cdf (np.array[float]): Gaussian cdf lookup table.
        vuln_adj (np.array[float]): per-vulnerability adjustment factors.
        haz_z_unif (np.array[float]): output buffer for hazard random values.
        vuln_z_unif (np.array[float]): output buffer for damage random values.
    """
    # hazard random values only exist (hazard_rng_index >= 0) when this areaperil's
    # hazard intensity is non-deterministic and we are running full Monte Carlo.
    if hazard_rng_index >= 0:
        if compute_info['do_haz_correlation'] and item['hazard_correlation_value'] > 0:
            # use correlation definitions to draw correlated random values into haz_z_unif
            get_corr_rval(
                haz_eps_ij[item['peril_correlation_group']], haz_rndms_base[hazard_rng_index], item['hazard_correlation_value'],
                norm_inv_parameters['x_min'], norm_inv_cdf, norm_inv_parameters['inv_factor'],
                norm_inv_parameters['cdf_min'], norm_cdf, norm_inv_parameters['norm_factor'],
                sample_size, haz_z_unif
            )
        else:
            haz_z_unif[:] = haz_rndms_base[hazard_rng_index]

    if compute_info['do_correlation'] and item['damage_correlation_value'] > 0:
        # use correlation definitions to draw correlated random values into vuln_z_unif
        get_corr_rval(
            damage_eps_ij[item['peril_correlation_group']], vuln_rndms_base[rng_index], item['damage_correlation_value'],
            norm_inv_parameters['x_min'], norm_inv_cdf, norm_inv_parameters['inv_factor'],
            norm_inv_parameters['cdf_min'], norm_cdf, norm_inv_parameters['norm_factor'],
            sample_size, vuln_z_unif
        )
    else:
        # do not use correlation
        vuln_z_unif[:] = vuln_rndms_base[rng_index]

    if item['areaperil_agg_vuln_idx'] < 0:  # single vuln id (non-aggregate)
        vuln_z_unif *= vuln_adj[item['vulnerability_idx']]


@nb.njit(cache=True, fastmath=True, inline='always')
def sample_item_losses(compute_info, item_j, sample_size, hazard_rng_index, item_event_data,
                       haz_z_unif, vuln_z_unif, haz_cdf_prob, Nhaz_bins,
                       eff_damage_cdf, Neff_damage_bins, haz_i_to_Ndamage_bins, haz_i_to_vuln_cdf,
                       damage_bins, damage_bin_scaling, losses,
                       is_dependent, store_source_bin, source_damage_bin_stack, depth):
    """Write the per-sample gul (or debug random values) for one item into ``losses``.

    In debug modes 1/2 the drawn hazard/damage random values are stored directly. Otherwise the
    gul is computed per sample: under effective_damageability from the single effective damage CDF;
    with a single hazard bin from that bin's vulnerability CDF; and in the general case by drawing
    the hazard bin per sample and using that bin's vulnerability CDF. Return-period protection is
    not handled here: the caller skips a protected item whole, before any sampling.

    Coverage dependency: a dependent item (``is_dependent``, full Monte Carlo) selects its hazard
    bin as its source's per-sample sampled damage bin, read from ``source_damage_bin_stack`` at the
    parent depth, rather than by drawing from the hazard CDF. When ``store_source_bin`` is set (any
    coverage under full-MC coverage dependency), this coverage's own sampled damage bin is recorded
    at ``source_damage_bin_stack[depth]`` so a dependent below it in the DFS can consume it.

    Args:
        compute_info (gulmc_compute_info_type): computation state (debug, effective_damageability).
        item_j (int): column index of this item within the coverage's loss buffer.
        sample_size (int): number of random samples.
        hazard_rng_index (int): index into hazard random values, or < 0 if hazard deterministic.
        item_event_data (np.void): per-item event data (source_item_j for a dependent item).
        haz_z_unif (np.array[float]): hazard random values for this item.
        vuln_z_unif (np.array[float]): damage random values for this item.
        haz_cdf_prob (np.array[float]): hazard intensity cdf.
        Nhaz_bins (int): number of hazard bins.
        eff_damage_cdf (np.array[oasis_float]): effective damage cdf.
        Neff_damage_bins (int): number of effective damage bins.
        haz_i_to_Ndamage_bins (np.array[oasis_int]): per-hazard-bin vulnerability cdf lengths.
        haz_i_to_vuln_cdf (np.array): per-hazard-bin vulnerability cdfs.
        damage_bins (np.array): damage bin dictionary.
        damage_bin_scaling (float): tiv scaling factor.
        losses (np.array[oasis_float]): loss buffer written in place at column item_j.
        is_dependent (bool): True if this item is driven by a source item's sampled damage bin.
        store_source_bin (bool): True if this coverage's own sampled damage bin must be recorded
          for the dependents below it in the DFS.
        source_damage_bin_stack (np.array[int32]): per-depth, per-item sampled damage bins, read at
          the parent depth for a dependent item and written at this coverage's depth.
        depth (int): this coverage's depth in the dependency forest (0 for a root).
    """
    if compute_info['debug'] == 1:  # store the random value used for the hazard sampling instead of the loss
        if hazard_rng_index >= 0:
            losses[1:, item_j] = haz_z_unif[:]
        else:
            # deterministic hazard / effective damageability: no hazard intensity sampled
            losses[1:, item_j] = 0

    elif compute_info['debug'] == 2:  # store the random value used for the damage sampling instead of the loss
        losses[1:, item_j] = vuln_z_unif[:]

    else:  # calculate gul
        if compute_info['effective_damageability']:
            for sample_idx in range(1, sample_size + 1):
                losses[sample_idx, item_j], _ = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], eff_damage_cdf,
                                                                      Neff_damage_bins, damage_bins, damage_bin_scaling)
        elif Nhaz_bins == 1:  # only one hazard possible
            Ndamage_bins = haz_i_to_Ndamage_bins[0]
            vuln_cdf = haz_i_to_vuln_cdf[0][:Ndamage_bins]
            for sample_idx in range(1, sample_size + 1):
                losses[sample_idx, item_j], src_bin = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], vuln_cdf,
                                                                            Ndamage_bins, damage_bins, damage_bin_scaling)
                if store_source_bin:
                    source_damage_bin_stack[depth, item_j, sample_idx - 1] = src_bin
        elif is_dependent:
            # coverage dependency: the dependent's "hazard bin" is the source's sampled damage bin,
            # read straight from the stack (no ratio round-trip, so a source of any damage type
            # works). This coverage may itself be a source (chain), so its own bin is recorded too.
            parent_damage_bin = source_damage_bin_stack[depth - 1, item_event_data['source_item_j']]
            for sample_idx in range(1, sample_size + 1):
                haz_bin_idx = parent_damage_bin[sample_idx - 1]
                Ndamage_bins = haz_i_to_Ndamage_bins[haz_bin_idx]
                vuln_cdf = haz_i_to_vuln_cdf[haz_bin_idx][:Ndamage_bins]
                losses[sample_idx, item_j], src_bin = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], vuln_cdf,
                                                                            Ndamage_bins, damage_bins, damage_bin_scaling)
                if store_source_bin:
                    source_damage_bin_stack[depth, item_j, sample_idx - 1] = src_bin
        else:
            for sample_idx in range(1, sample_size + 1):
                # find the hazard intensity cdf bin in which the random value `haz_z_unif[sample_idx - 1]` falls into
                # we don't need to use last haz_cdf_prob value because if for rounding reason haz_rval
                # is bigger, we want the index Nhaz_bins-1 anyway. if we were using Nhaz_bins,
                # bigger than haz_cdf_prob[-1] haz_rval would have index Nhaz_bins, outside haz_i_to_Ndamage_bins
                haz_bin_idx = binary_search(haz_z_unif[sample_idx - 1], haz_cdf_prob, Nhaz_bins - 1)

                # get the individual vulnerability cdf
                Ndamage_bins = haz_i_to_Ndamage_bins[haz_bin_idx]
                vuln_cdf = haz_i_to_vuln_cdf[haz_bin_idx][:Ndamage_bins]

                losses[sample_idx, item_j], src_bin = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], vuln_cdf,
                                                                            Ndamage_bins, damage_bins, damage_bin_scaling)
                if store_source_bin:
                    source_damage_bin_stack[depth, item_j, sample_idx - 1] = src_bin


@nb.njit(cache=True, fastmath=True)
def compute_event_losses(compute_info,
                         coverages,
                         coverage_ids,
                         items_event_data,
                         items,
                         sample_size,
                         haz_pdf,
                         haz_arr_ptr,
                         vuln_array,
                         conditional_vuln_array,
                         vuln_idx_to_cond_idx,
                         damage_bins,
                         cdf_cache_tag,
                         cdf_cache_nbins,
                         cdf_cache_mask,
                         cached_vuln_cdfs,
                         areaperil_agg_vuln_idx_ja_offsets,
                         areaperil_agg_vuln_idx_ja_data,
                         losses,
                         haz_rndms_base,
                         vuln_rndms_base,
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
                         compute_depth,
                         source_damage_bin_stack,
                         source_eff_damage_cdf_stack,
                         source_eff_damage_cdf_len_stack
                         ):
    """Compute ground-up losses for all coverages in a single event.

    Iterates over coverages and their items, looking up or computing the vulnerability cdf
    for each item, then sampling losses using the pre-generated random numbers. Results are
    written into a byte buffer for streaming output.

    CDF caching uses a monotonic write counter and array-based slot tracking. Each unique
    (areaperil, vuln_id[, intensity_adjustment]) CDF group has a pre-computed index stored in
    eff_cdf_id. cdf_cache_tag[triplet_idx] records the write counter value when the CDFs were
    cached. A slot is valid when cdf_cache_tag[triplet_idx] >= 0 and
    cdf_cache_ctr - cdf_cache_tag[triplet_idx] < cdf_cache_size. Physical slot indexing uses
    bitwise AND with cdf_cache_mask (power-of-two sized cache).

    For effective_damageability=False, CDFs are stored as contiguous blocks:
    slot 0 = effective damage CDF, slots 1..Nhaz_bins = per-intensity-bin vulnerability CDFs.

    Args:
        compute_info (gulmc_compute_info_type): computation state (event_id, cursor position,
          coverage range, cdf_cache_ctr, thresholds, flags).
        coverages (numpy.array[coverage_type]): coverage data indexed by coverage_id.
        coverage_ids (numpy.array[int]): ordered list of coverage_ids to process in this event.
        items_event_data (numpy.array[items_MC_data_type]): per-item event data populated by
          reconstruct_coverages, containing item_idx, haz_arr_i, rng_index, hazard_rng_index,
          and eff_cdf_id (CDF group index).
        items (np.ndarray): items table merged with correlation parameters.
        sample_size (int): number of random samples to draw.
        haz_pdf (np.array[haz_arr_type]): hazard intensity pdf records for this event.
        haz_arr_ptr (np.array[int64]): indices where each areaperil's hazard records start in haz_pdf.
        vuln_array (np.array[float]): 3d vulnerability array of shape
          (Nvulnerability, Ndamage_bins_max, Nintensity_bins).
        damage_bins (np.array): damage bin dictionary with bin_from, bin_to, interpolation, damage_type.
        cdf_cache_tag (np.array[int64]): CDF group index → write counter when cached (CDF_CACHE_EMPTY = -1).
        cdf_cache_nbins (np.array[int32]): physical slot → CDF length (Ndamage_bins).
        cdf_cache_mask (int64): bitmask for physical slot indexing (cdf_cache_size - 1).
        cached_vuln_cdfs (np.array[oasis_float]): 2d cdf cache of shape (cdf_cache_size, Ndamage_bins_max).
        areaperil_agg_vuln_idx_ja_offsets (np.array[oasis_int]): jagged array offsets.
        areaperil_agg_vuln_idx_ja_data (np.array[agg_vuln_idx_weight_dtype]): merged structured array
          with fields 'vuln_idx' (dense vulnerability index) and 'weight' (vulnerability weight).
        losses (numpy.array[oasis_float]): reusable 2d buffer for loss values.
        haz_rndms_base (numpy.array[float64]): base random values for hazard intensity sampling.
        vuln_rndms_base (numpy.array[float64]): base random values for damage sampling.
        vuln_adj (np.array[float]): per-vulnerability adjustment factors.
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
        conditional_vuln_array (np.array[oasis_float]): damage-transition matrices, indexed
          [cond_idx, dependent damage bin - 1, source damage bin - 1]. Empty when no dependency.
        vuln_idx_to_cond_idx (np.array[int64]): dense vuln index -> conditional row, or -1.
        compute_depth (np.array[int32]): per entry of ``coverage_ids``, its depth in the dependency
          forest (0 for a root), matching the DFS order the caller put them in.
        source_damage_bin_stack (np.array[int32]): per-depth, per-item sampled damage bins, holding
          a source coverage's result while its dependent subtree is computed.
        source_eff_damage_cdf_stack (np.array[oasis_float]): per-depth, per-item effective damage
          cdfs, the effective-damageability counterpart of the sampled bins.
        source_eff_damage_cdf_len_stack (np.array[int64]): valid length of each entry of
          ``source_eff_damage_cdf_stack``.

    Returns:
        bool: True if all coverages have been processed, False if the buffer is full and
          the caller should flush and call again.
    """
    cdf_cache_size = nb_int64(cdf_cache_mask + 1)
    # coverage dependency: a dependent's "hazard bins" are the source's damage bins, so its number
    # of hazard bins is num_damage_bins rather than the footprint's num_intensity_bins. The
    # per-hazard-bin scratch buffers must therefore be sized to the larger of the two.
    n_damage_bins_total = damage_bins.shape[0]
    max_haz_bins = max(vuln_array.shape[2], n_damage_bins_total)
    haz_cdf_empty = np.empty(max_haz_bins, dtype=oasis_float)
    vuln_pdf_empty = np.empty((max_haz_bins, compute_info['Ndamage_bins_max']), dtype=vuln_array.dtype)
    eff_damage_cdf_empty = np.empty(compute_info['Ndamage_bins_max'], dtype=oasis_float)
    haz_i_to_Ndamage_bins_empty = np.empty(max_haz_bins, dtype=oasis_int)
    haz_i_to_vuln_cdf_empty = np.empty((max_haz_bins, compute_info['Ndamage_bins_max']), dtype=vuln_array.dtype)
    # the source's damage pmf plays the role of the hazard pdf for a dependent
    source_damage_pmf_empty = np.empty(n_damage_bins_total, dtype=oasis_float)

    # we process at least one full coverage at a time, so when we write to stream, we write the whole buffer
    compute_info['cursor'] = 0

    # loop through all the coverages that remain to be computed. `compute` is in DFS order:
    # each root coverage (compute_depth == 0) is immediately followed by its dependent
    # subtree, so a source is always processed before the dependents that use its result.
    for coverage_i in range(compute_info['coverage_i'], compute_info['coverage_n']):
        coverage_id = coverage_ids[coverage_i]
        coverage = coverages[coverage_id]
        depth = compute_depth[coverage_i]
        tiv = coverage['tiv']
        Nitems = coverage['cur_items']
        exposureValue = tiv / Nitems

        # A root and its dependent subtree are written as one atomic unit so the source's
        # per-sample damage bin (held on source_damage_bin_stack, indexed by depth) stays
        # valid across the whole subtree. We therefore only check the buffer at subtree roots,
        # estimating the bytes for the entire subtree (conservatively assuming all samples are
        # printed).
        if depth == 0:
            subtree_item_count = Nitems
            lookahead_index = coverage_i + 1
            while lookahead_index < compute_info['coverage_n'] and compute_depth[lookahead_index] > 0:
                subtree_item_count += coverages[coverage_ids[lookahead_index]]['cur_items']
                lookahead_index += 1
            if compute_info['cursor'] + subtree_item_count * compute_info['max_bytes_per_item'] > byte_mv.shape[0]:
                return False

        # resolved per item below: such a coverage may also hold items that found no source item,
        # and those are computed independently
        coverage_has_dependents = compute_info['do_coverage_dependency'] == 1 and depth > 0
        # compute losses for each item
        for item_j in range(Nitems):
            item_event_data = items_event_data[coverage['start_items'] + item_j]
            rng_index = item_event_data['rng_index']
            hazard_rng_index = item_event_data['hazard_rng_index']

            item = items[item_event_data['item_idx']]
            # an item is dependent only if it resolved to a source item (< 0 means it did not)
            is_dependent = coverage_has_dependents and item_event_data['source_item_j'] >= 0
            haz_arr_i = item_event_data['haz_arr_i']
            haz_pdf_record = haz_pdf[haz_arr_ptr[haz_arr_i]:haz_arr_ptr[haz_arr_i + 1]]

            if dynamic_footprint is not None:
                intensity_adjustment = item['intensity_adjustment']
                # RP protection skips the item whole, samples and analytic sidx alike. The return
                # period is per item-event, not per hazard bin, so this needs no per-sample check.
                if item_event_data['return_period'] > 0 \
                        and item_event_data['event_rp'] < item_event_data['return_period']:
                    losses[:, item_j] = 0
                    if compute_info['do_coverage_dependency'] == 1:
                        # a dependent below this one in the DFS still reads (depth, item_j), so
                        # leave "no damage" there rather than the last coverage's values
                        source_damage_bin_stack[depth, item_j, :] = 0
                        source_eff_damage_cdf_stack[depth, item_j, 0] = 1.
                        source_eff_damage_cdf_len_stack[depth, item_j] = 1
                    continue
            else:
                intensity_adjustment = nb_oasis_int(0)

            # we calculate this adjusted hazard pdf
            # get the right hazard pdf from the array containing all hazard cdfs
            haz_bin_id = compute_haz_bin_id(haz_pdf_record, item, intensity_adjustment, dynamic_footprint,
                                            intensity_bin_peril_ids, intensity_bins)
            haz_pdf_prob = haz_pdf_record['probability']

            cdf_group = nb_int64(item_event_data['eff_cdf_id'])

            # coverage dependency: replace this dependent's hazard bins with the source's
            # damage bins and its hazard pdf with the source's damage pmf, so the dependent's
            # vulnerability (authored over damage-bin-indexed intensities) is driven directly
            # by the source's damage. The downstream assembly / CDF / sampling is reused.
            if is_dependent:
                # the dependent's "hazard bins" are the source's damage bins; its vulnerability is
                # the conditional matrix (assembled below), and its hazard pdf is the source's
                # damage pmf (derived here from the source's stored effective-damage CDF). The
                # footprint hazard CDF is unused for a dependent, so it is not computed; the name
                # is kept defined (as an empty view) only so numba sees it on all paths.
                Nhaz_bins = n_damage_bins_total
                haz_cdf_prob = haz_cdf_empty[:0]
                src_j = item_event_data['source_item_j']
                parent_eff_cdf = source_eff_damage_cdf_stack[depth - 1, src_j, :source_eff_damage_cdf_len_stack[depth - 1, src_j]]
                haz_pdf_prob = source_damage_pmf_empty
                prev_cdf = 0.0
                for damage_bin_k in range(Nhaz_bins):
                    if damage_bin_k < parent_eff_cdf.shape[0]:
                        haz_pdf_prob[damage_bin_k] = parent_eff_cdf[damage_bin_k] - prev_cdf
                        prev_cdf = parent_eff_cdf[damage_bin_k]
                    else:
                        haz_pdf_prob[damage_bin_k] = 0.0
            else:
                haz_cdf_prob = pdf_to_cdf(haz_pdf_prob, haz_cdf_empty)
                Nhaz_bins = haz_cdf_prob.shape[0]

            # determine if the CDFs for this CDF group are cached. Dependent CDFs are
            # event-specific (they depend on the source's damage this event), so they always
            # miss the (event-independent) cache and are recomputed.
            stored = cdf_cache_tag[cdf_group]
            do_calc_vuln_ptf = is_dependent or (
                stored < 0) or (compute_info['cdf_cache_ctr'] - stored >= cdf_cache_size)

            eff_damage_cdf, haz_i_to_Ndamage_bins, haz_i_to_vuln_cdf = resolve_item_cdfs(
                compute_info, cdf_group, do_calc_vuln_ptf, Nhaz_bins, item, haz_bin_id, haz_pdf_prob,
                vuln_array, areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
                cdf_cache_tag, cdf_cache_nbins, cdf_cache_mask, cached_vuln_cdfs,
                vuln_pdf_empty, eff_damage_cdf_empty, haz_i_to_Ndamage_bins_empty, haz_i_to_vuln_cdf_empty,
                is_dependent, conditional_vuln_array, vuln_idx_to_cond_idx)

            Neff_damage_bins = eff_damage_cdf.shape[0]
            damage_bin_scaling = compute_damage_bin_scaling(damage_bins, Neff_damage_bins, tiv)

            # compute mean loss values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                damage_bin_scaling,
                eff_damage_cdf,
                damage_bins['interpolation'],
                Neff_damage_bins,
                damage_bins[Neff_damage_bins - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_j] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_j] = chance_of_loss
            losses[TIV_IDX, item_j] = exposureValue
            losses[STD_DEV_IDX, item_j] = std_dev
            losses[MEAN_IDX, item_j] = gul_mean

            if sample_size > 0:  # compute random losses
                # coverage dependency (full Monte Carlo): record each coverage's per-sample sampled
                # damage bin so a dependent below it in the DFS can index its conditional vulnerability
                # directly (a dependent in turn consumes its source's stored bins in sample_item_losses).
                store_source_bin = compute_info['do_coverage_dependency'] == 1 and not compute_info['effective_damageability']

                draw_correlation_samples(compute_info, item, hazard_rng_index, rng_index, sample_size,
                                         haz_rndms_base, vuln_rndms_base, haz_eps_ij, damage_eps_ij,
                                         norm_inv_parameters, norm_inv_cdf, norm_cdf, vuln_adj,
                                         haz_z_unif, vuln_z_unif)

                sample_item_losses(compute_info, item_j, sample_size, hazard_rng_index, item_event_data,
                                   haz_z_unif, vuln_z_unif, haz_cdf_prob, Nhaz_bins,
                                   eff_damage_cdf, Neff_damage_bins, haz_i_to_Ndamage_bins, haz_i_to_vuln_cdf,
                                   damage_bins, damage_bin_scaling, losses,
                                   is_dependent, store_source_bin, source_damage_bin_stack, depth)

            # coverage dependency (effective damageability): record this coverage's effective-damage
            # CDF at (depth, item_j) so a dependent below it in the DFS order can build its damage pmf
            # from it. (Full Monte Carlo instead uses the per-sample damage bin captured above.)
            if compute_info['do_coverage_dependency'] == 1:
                num_damage_bins = eff_damage_cdf.shape[0]
                source_eff_damage_cdf_stack[depth, item_j, :num_damage_bins] = eff_damage_cdf
                source_eff_damage_cdf_len_stack[depth, item_j] = num_damage_bins

        # write the losses to the output memoryview. A zero-TIV coverage (e.g. an uninsured
        # dependency source, retained only to drive its dependents) yields zero losses and is
        # written like any other coverage — it is not special-cased.
        compute_info['cursor'] = write_losses(
            compute_info['event_id'],
            sample_size,
            compute_info['loss_threshold'],
            losses[:, :Nitems],
            items_event_data[coverage['start_items']: coverage['start_items'] + Nitems]['item_id'],
            compute_info['alloc_rule'],
            tiv,
            byte_mv,
            compute_info['cursor'])

        # register that another `coverage_id` has been processed
        compute_info['coverage_i'] += 1

    return True


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
                          item_map_ja_vuln_ja_offsets,
                          item_map_ja_vuln_ja_item_idxs,
                          items,
                          item_cdf_group_idx,
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
                          group_seq_rng_index,
                          hazard_group_seq_rng_index,
                          coverage_source_id,
                          coverage_dependents_ja_offsets,
                          coverage_dependents_ja_data,
                          compute_depth,
                          compute_footprint_order,
                          dependency_dfs_stack,
                          source_item_idx,
                          item_idx_to_item_j):
    """Register each item to its coverage and prepare per-item event data for loss computation.

    For each (areaperil_id, vulnerability_id) pair present in the event footprint, iterates
    over all mapped items and:
      1. Computes deterministic hash-based random seeds for hazard and damage sampling,
         using group_id and hazard_group_id respectively. Seeds are deduplicated via
         pre-allocated arrays indexed by sequential group ids.
      2. Maps each item to its coverage structure, tracking the start offset and count.
      3. Stores per-item event data (haz_arr_i, rng_index, hazard_rng_index, eff_cdf_id)
         in the items_event_data array. The eff_cdf_id is the pre-computed CDF group index
         from item_cdf_group_idx.

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
        item_map_ja_offsets (np.array[oasis_int]): L1 CSR offsets (N_areaperil + 1).
        item_map_ja_vuln_ja_offsets (np.array[oasis_int]): L2 CSR offsets (N_pairs + 1).
        item_map_ja_vuln_ja_item_idxs (np.array[oasis_int]): flat item indices into items array.
        items (np.ndarray): items table merged with correlation parameters, containing
          group_id, hazard_group_id, coverage_id, group_seq_id, hazard_group_seq_id, etc.
        item_cdf_group_idx (np.array[int64]): pre-computed mapping from item_idx to CDF group index.
        coverages (numpy.array[coverage_type]): coverage data indexed by coverage_id.
        compute (numpy.array[int]): output buffer for the list of coverage_ids to be computed.
        haz_seeds (numpy.array[int]): output buffer for hazard intensity random seeds.
        haz_peril_correlation_groups (numpy.array[int]): unique peril correlation groups for hazard.
        haz_corr_seeds (numpy.array[int]): output buffer for hazard correlation seeds.
        vuln_seeds (numpy.array[int]): output buffer for damage random seeds.
        damage_peril_correlation_groups (numpy.array[int]): unique peril correlation groups for damage.
        damage_corr_seeds (numpy.array[int]): output buffer for damage correlation seeds.
        dynamic_footprint (None or object): None if no dynamic footprint, otherwise truthy.
        byte_mv (numpy.array[byte]): output byte buffer, may be resized if needed.
        group_seq_rng_index (numpy.array[int64]): pre-allocated array of size n_unique_groups,
          used for O(1) group_id to rng_index mapping (reset to NO_RNG_INDEX each event).
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
          n_unique_haz_groups, for hazard_group_id to rng_index mapping.

    Returns:
        tuple: (items_event_data, rng_index, hazard_rng_index, byte_mv)
          - items_event_data (numpy.array[items_MC_data_type]): per-item data including
            item_idx, haz_arr_i, rng_index, hazard_rng_index, eff_cdf_id.
          - rng_index (int): number of unique damage random seeds generated.
          - hazard_rng_index (int): number of unique hazard random seeds generated. Only hazard
            groups touching a non-deterministic areaperil (and only under full Monte Carlo, not
            effective damageability) are counted, so this is 0 for events with no hazard
            uncertainty. Items not assigned a hazard rng row carry the NO_RNG_INDEX sentinel in
            items_event_data['hazard_rng_index'].
          - byte_mv (numpy.array[byte]): output buffer, possibly resized.
    """
    # init data structures
    # Reset pre-allocated arrays instead of creating new Numba Dicts
    group_seq_rng_index[:] = NO_RNG_INDEX
    hazard_group_seq_rng_index[:] = NO_RNG_INDEX
    rng_index = 0
    hazard_rng_index = 0
    compute_i = 0
    items_data_i = 0
    coverages['cur_items'].fill(0)
    items_event_data = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_MC_data_type)

    # for each areaperil_id, loop over all vulnerability functions used in that areaperil_id and,
    # for each item:
    #  - compute the seeds for the hazard intensity sampling and for the damage sampling
    #  - store data for later processing (hazard cdf index, etc.)
    for ap_i in range(Nhaz_arr_this_event):
        ap_ind = ap_inds[ap_i]
        # A hazard rng row is only consumed when the hazard intensity is non-deterministic
        # (more than one intensity bin for this areaperil) and we are running full Monte Carlo.
        ap_needs_haz_rng = (not compute_info['effective_damageability']) and (haz_arr_ptr[ap_i + 1] - haz_arr_ptr[ap_i] > 1)
        vuln_start = item_map_ja_offsets[ap_ind]
        vuln_end = item_map_ja_offsets[ap_ind + 1]

        for k in range(vuln_start, vuln_end):
            item_start = item_map_ja_vuln_ja_offsets[k]
            item_end = item_map_ja_vuln_ja_offsets[k + 1]
            for item_pos in range(item_start, item_end):
                item_idx = item_map_ja_vuln_ja_item_idxs[item_pos]
                # if this group_id was not seen yet, process it.
                # it assumes that hash only depends on event_id and group_id
                # and that only 1 event_id is processed at a time.
                # Use sequential index for array-based lookup instead of Dict
                group_seq_id = items[item_idx]['group_seq_id']
                if group_seq_rng_index[group_seq_id] == NO_RNG_INDEX:
                    group_seq_rng_index[group_seq_id] = rng_index
                    vuln_seeds[rng_index] = generate_hash(items[item_idx]['group_id'], compute_info['event_id'])
                    this_rng_index = rng_index
                    rng_index += 1
                else:
                    this_rng_index = group_seq_rng_index[group_seq_id]

                if ap_needs_haz_rng:
                    hazard_group_seq_id = items[item_idx]['hazard_group_seq_id']
                    if hazard_group_seq_rng_index[hazard_group_seq_id] == NO_RNG_INDEX:
                        hazard_group_seq_rng_index[hazard_group_seq_id] = hazard_rng_index
                        haz_seeds[hazard_rng_index] = generate_hash_hazard(items[item_idx]['hazard_group_id'], compute_info['event_id'])
                        this_hazard_rng_index = hazard_rng_index
                        hazard_rng_index += 1
                    else:
                        this_hazard_rng_index = hazard_group_seq_rng_index[hazard_group_seq_id]
                else:
                    # deterministic hazard (or effective damageability): no hazard rng row exists.
                    # The NO_RNG_INDEX sentinel is never used as an index (guarded at consumption).
                    this_hazard_rng_index = NO_RNG_INDEX

                coverage_id = items[item_idx]['coverage_id']
                coverage = coverages[coverage_id]
                if coverage['cur_items'] == 0:
                    # no items were collected for this coverage yet: set up the structure.
                    # All present coverages are appended here in footprint order; when
                    # coverage dependency is active this list is reordered into DFS order
                    # (root followed by its dependent subtree) below.
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
                items_event_data[item_i]['haz_arr_i'] = ap_i
                items_event_data[item_i]['rng_index'] = this_rng_index
                items_event_data[item_i]['hazard_rng_index'] = this_hazard_rng_index
                items_event_data[item_i]['eff_cdf_id'] = item_cdf_group_idx[item_idx]
                if dynamic_footprint is not None:
                    items_event_data[item_i]['intensity_adjustment'] = items[item_idx]['intensity_adjustment']
                    items_event_data[item_i]['return_period'] = items[item_idx]['return_period']
                    items_event_data[item_i]['event_rp'] = event_rps[ap_i]

                coverage['cur_items'] += 1

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
        max_subtree_items = 0
        for position in range(num_present_coverages):
            root_coverage_id = compute_footprint_order[position]
            # A coverage is a root when it has no source, or when its source coverage contributes
            # no item to this event: a paired item shares its source item's areaperil, so an absent
            # source coverage means every present item here is unpaired and belongs at depth 0.
            source_coverage_id = coverage_source_id[root_coverage_id]
            if source_coverage_id != 0 and coverages[source_coverage_id]['cur_items'] > 0:
                continue  # not a root: emitted as part of an ancestor's subtree
            subtree_item_count = 0
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
                subtree_item_count += coverages[coverage_id]['cur_items']
                for dependent_pos in range(coverage_dependents_ja_offsets[coverage_id],
                                           coverage_dependents_ja_offsets[coverage_id + 1]):
                    dependent_coverage_id = coverage_dependents_ja_data[dependent_pos]
                    if coverages[dependent_coverage_id]['cur_items'] > 0:  # only descend into present dependents
                        dependency_dfs_stack[stack_pointer, 0] = dependent_coverage_id
                        dependency_dfs_stack[stack_pointer, 1] = depth + 1
                        stack_pointer += 1
            if subtree_item_count > max_subtree_items:
                max_subtree_items = subtree_item_count
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
        max_subtree_items = int(np.max(coverages['cur_items']))

    compute_info['coverage_i'] = 0
    compute_info['coverage_n'] = compute_i
    byte_mv = adjust_byte_mv_size(byte_mv, max_subtree_items * compute_info['max_bytes_per_item'])

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
