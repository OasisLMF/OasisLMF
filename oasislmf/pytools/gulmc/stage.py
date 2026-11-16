"""gulmc as a stage: set up once, then compute one event at a time.

``GulmcStage`` holds everything the ``gulmc`` command builds before its event loop and runs the
loop body for one event on demand. ``oasislmf.pytools.gulmc.manager.run`` (the ``gulmc`` command)
is built on it: it reads event ids from its input stream and hands each to ``process_event``.
"""
import atexit
import json
import logging
import os
from contextlib import ExitStack

import numpy as np

from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import oasis_float, oasis_int, correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import (PIPE_CAPACITY, ITEM_PACKED_STREAM, ITEM_STREAM, LOSS_STREAM_ID,
                                                  check_packed_item_fits, check_packing_supported,
                                                  max_emitted_blocks, stream_info_to_bytes)
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.gul.common import NUM_IDX
from oasislmf.pytools.gul.core import HERMITE_TERMS
from oasislmf.pytools.gul.random import get_random_generator
from oasislmf.pytools.gulmc import manager as gulmc_manager
from oasislmf.pytools.gulmc.common import (NormInversionParameters,
                                           gulSampleslevelHeader_size,
                                           gulSampleslevelRec_size,
                                           gulmc_compute_info_type)
from oasislmf.pytools.gulmc.structure import (gulmc_structure_exists, load_gulmc_structure, build_structures,
                                              dynamic_footprint_path)

# the manager's logger, so the gulmc log reads exactly as it did before the stage was split out of run
logger = logging.getLogger('oasislmf.pytools.gulmc.manager')


class GulmcStage:
    """The gulmc ground-up loss calculation, set up once and then run one event at a time.

    The constructor does everything the ``gulmc`` command does before its first event: it
    validates the options, registers with the data server, loads or builds the model structures,
    opens the footprint, sets up the random generators and allocates the reusable buffers.
    ``process_event`` computes one event and writes exactly the bytes ``gulmc`` writes for it.
    ``header`` is the stream header ``gulmc`` writes before its first event, so ``header``
    followed by the bytes of each event in turn is the ``gulmc`` output stream.

    Paths resolve as in the command-line tool: ``event_error.json`` is written to the current
    working directory.

    The stage is not reentrant: its buffers are reused from one event to the next, and the bytes
    handed to a sink are a view of an internal buffer, valid only until the sink's ``write``
    returns.

    Args:
        run_dir (str): the directory of where the process is running
        sample_size (int): number of random samples to draw.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        alloc_rule (int): back-allocation rule.
        debug (int): for each random sample, print to the output stream the random loss (if 0), the random value used to draw
          the hazard intensity sample (if 1), the random value used to draw the damage sample (if 2).
        random_generator (int): random generator function id.
        ignore_file_type (Iterable[str], optional): file extension to ignore when loading. Defaults to ().
        peril_filter (Iterable[int], optional): perils to include in the computation (if empty, all perils are included). Defaults to ().
        data_server (bool, optional): if True, get the footprint from the data server. Defaults to None.
        ignore_correlation (bool, optional): if True, do not compute correlated random samples. Defaults to False.
        ignore_haz_correlation (bool, optional): if True, do not compute correlated hazard intensity samples. Defaults to False.
        effective_damageability (bool, optional): if True, it uses effective damageability to draw damage samples instead of
          using the full monte carlo approach (i.e., to draw hazard intensity first, then damage).
        max_cached_vuln_cdf_size_MB (int, optional): size in MB of the in-memory cache to store and reuse vulnerability cdf. Defaults to 200.
        model_df_engine (str, optional): The engine to use when loading model dataframes. Defaults to OasisPandasReader.
        dynamic_footprint (bool, optional): if True, load the dynamic footprint data and adjust hazard intensities at
          runtime. Defaults to False.

    Attributes:
        header (bytes): the stream header (stream type and sample size) gulmc writes before its first event.

    Raises:
        ValueError: if alloc_rule is not 0, 1, 2, or 3.
        ValueError: if alloc_rule is 1, 2, or 3 when debug is 1 or 2.
    """

    def __init__(self, run_dir, sample_size, loss_threshold, alloc_rule, debug, random_generator,
                 ignore_file_type=(), peril_filter=(), data_server=None, ignore_correlation=False,
                 ignore_haz_correlation=False, effective_damageability=False,
                 max_cached_vuln_cdf_size_MB=200, model_df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
                 dynamic_footprint=False):
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
            self._setup(stack, run_dir, sample_size, loss_threshold, alloc_rule, debug, random_generator, ignore_file_type,
                        peril_filter, data_server, ignore_correlation, ignore_haz_correlation, effective_damageability,
                        max_cached_vuln_cdf_size_MB, model_df_engine, dynamic_footprint)
            self._stack = stack.pop_all()

    def _setup(self, stack, run_dir, sample_size, loss_threshold, alloc_rule, debug, random_generator, ignore_file_type,
               peril_filter, data_server, ignore_correlation, ignore_haz_correlation, effective_damageability,
               max_cached_vuln_cdf_size_MB, model_df_engine, dynamic_footprint):
        """Build the structures, footprint, random generator state and buffers the event loop uses.

        Args:
            stack (ExitStack): receives the resources the stage holds until ``close``.
            run_dir (str): as for the constructor.
            sample_size (int): as for the constructor.
            loss_threshold (float): as for the constructor.
            alloc_rule (int): as for the constructor.
            debug (int): as for the constructor.
            random_generator (int): as for the constructor.
            ignore_file_type (set(str)): as for the constructor, as a set.
            peril_filter (Iterable[int]): as for the constructor.
            data_server (bool): as for the constructor.
            ignore_correlation (bool): as for the constructor.
            ignore_haz_correlation (bool): as for the constructor.
            effective_damageability (bool): as for the constructor.
            max_cached_vuln_cdf_size_MB (int): as for the constructor.
            model_df_engine (str): as for the constructor.
            dynamic_footprint (bool): as for the constructor.
        """
        # --- load or build read-only structures --------------------------------
        # the dynamic footprint's hazard is memory-mapped when create_gulmc_structure saved it
        footprint_shared_dir = None
        if gulmc_structure_exists(run_dir):
            footprint_shared_dir = dynamic_footprint_path(run_dir)
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
        self.item_map_ja_offsets = structures['item_map_ja_offsets']
        self.item_map_ja_vuln_ja_offsets = structures['item_map_ja_vuln_ja_offsets']
        self.item_map_ja_vuln_ja_item_idxs = structures['item_map_ja_vuln_ja_item_idxs']
        self.item_map_ja_id_ind = structures['item_map_ja_id_ind']
        self.item_cdf_group_idx = structures['item_cdf_group_idx']
        n_cdf_groups = structures['n_cdf_groups']
        self.areaperil_agg_vuln_idx_ja_offsets = structures['areaperil_agg_vuln_idx_ja_offsets']
        self.areaperil_agg_vuln_idx_ja_data = structures['areaperil_agg_vuln_idx_ja_data']
        self.damage_bins = structures['damage_bins']
        self.vuln_adj = structures['vuln_adj']
        vuln_array = structures['vuln_array']
        conditional_vuln_array = structures['conditional_vuln_array']
        vuln_idx_to_cond_idx = structures['vuln_idx_to_cond_idx']
        unique_peril_correlation_groups = structures['unique_peril_correlation_groups']
        norm_inv_cdf = structures['norm_inv_cdf']
        norm_cdf = structures['norm_cdf']
        norm_inv_parameters = structures['norm_inv_parameters']
        self.intensity_bin_peril_ids = structures['intensity_bin_peril_ids']
        self.intensity_bins = structures['intensity_bins']
        coverage_source_id = structures['coverage_source_id']
        coverage_dependents_ja_offsets = structures['coverage_dependents_ja_offsets']
        coverage_dependents_ja_data = structures['coverage_dependents_ja_data']
        self.source_item_idx = structures['source_item_idx']
        n_unique_groups = structures['n_unique_groups']
        n_unique_haz_groups = structures['n_unique_haz_groups']
        del structures

        do_coverage_dependency = bool(coverage_dependents_ja_data.shape[0] > 0)

        # gate on the file as well as the forest: if every dependent ends up unpaired the forest is
        # empty, but an item still carrying a conditional vuln has to be caught
        if do_coverage_dependency or conditional_vuln_array.shape[0] > 0:
            if do_coverage_dependency:
                logger.info(f"coverage dependency: switched ON ({coverage_dependents_ja_data.shape[0]} dependent coverages).")
            gulmc_manager.validate_coverage_dependency(items, vuln_idx_to_cond_idx)

        Nvulnerability, Ndamage_bins_max, Nintensity_bins = vuln_array.shape
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
        self.compute = np.zeros(coverages.shape[0] + 1, items_dtype['coverage_id'])

        # coverage dependency scratch/state (all trivial when the feature is off)
        self.compute_depth = np.zeros(coverages.shape[0] + 1, dtype=np.int32)
        self.compute_footprint_order = np.zeros(coverages.shape[0] + 1, dtype=items_dtype['coverage_id'])
        # DFS work stack for reordering coverages into (root -> subtree) order: (coverage_id, depth)
        self.dependency_dfs_stack = np.zeros((coverages.shape[0] + 1, 2), dtype=np.int64)
        # scratch: each present item's position within its coverage, rewritten every event
        self.item_idx_to_item_j = np.zeros(items.shape[0], dtype=oasis_int)
        # longest dependency chain: sizes the per-depth parent-result stacks
        max_dependency_depth = gulmc_manager.compute_max_dependency_depth(coverage_source_id) if do_coverage_dependency else 0
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
        self.source_damage_bin_stack = np.zeros(
            (max_dependency_depth + 1, max_items_per_coverage,
             max_source_buildings * (sample_size if sample_size > 0 else 1)), dtype=np.int32)
        self.source_eff_damage_cdf_stack = np.zeros(
            (max_dependency_depth + 1, max_items_per_coverage, Ndamage_bins_max), dtype=oasis_float)
        self.source_eff_damage_cdf_len_stack = np.zeros((max_dependency_depth + 1, max_items_per_coverage), dtype=np.int64)

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
                                                shared_dir=footprint_shared_dir))
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        # The aggregation type says whether a reader may meet a packed sidx, and only a
        # kept-separate item emits one -- a run that packs nothing, or packs only items summed at
        # source, declares ITEM_STREAM and is byte-for-byte pre-packing.
        self.header = stream_info_to_bytes(
            LOSS_STREAM_ID,
            ITEM_PACKED_STREAM if max_emitted_blocks(items['packed_buildings']) > 1 else ITEM_STREAM
        ) + np.int32(sample_size).tobytes()

        # set the random generator function
        self.generate_rndm = get_random_generator(random_generator)
        # create the array to store the seeds
        self.haz_seeds = np.zeros(n_unique_haz_groups, dtype=correlations_dtype['hazard_group_id'])
        self.vuln_seeds = np.zeros(n_unique_groups, dtype=items_dtype['group_id'])
        # Pre-allocated arrays for group_id -> rng_index mapping (replaces per-event Numba Dicts)
        self.group_seq_rng_index = np.empty(n_unique_groups, dtype=np.int64)
        self.hazard_group_seq_rng_index = np.empty(n_unique_haz_groups, dtype=np.int64)
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
                logger.info("correlated random number generation for hazard intensity sampling: switched OFF because "
                            "0 peril correlation groups were detected or "
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
                logger.info("correlated random number generation for damage sampling: switched OFF because "
                            "0 peril correlation groups were detected or "
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
        self.vuln_z_unif = np.zeros(sample_size, dtype='float64')
        self.haz_z_unif = np.zeros(sample_size, dtype='float64')

        if effective_damageability is True:
            logger.info("effective_damageability is True: gulmc will draw the damage samples from the effective damageability distribution.")
        else:
            logger.info("effective_damageability is False: gulmc will perform the full Monte Carlo sampling: "
                        "sample the hazard intensity first, then sample the damage from the corresponding vulnerability function.")

        # create buffers to be reused when computing losses
        self.byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')
        self.losses = np.zeros((sample_size + NUM_IDX + 1, max_items_per_coverage), dtype=oasis_float)

        # Per-building sample buffer. write_losses reduces across items at a fixed building and
        # then sums the buildings, so a coverage's every building has to be resident before any of
        # it can be written.
        self.building_losses = np.zeros((max(sample_size, 1), max_items_per_coverage, max_buildings),
                                        dtype=oasis_float)
        # Generator 2 is counter-based: a building's block is a pure function of the group key
        # and the building index, so it can be produced where it is consumed instead of the
        # whole event's draws being materialised up front. These hold one building's worth.
        self.vuln_draw_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        self.haz_draw_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        self.perm_scratch = np.zeros(max(sample_size, 1), dtype='float64')
        # Per item of the current coverage, the correlation between two of its buildings'
        # LOSSES -- what the variance of their sum actually needs. 0 unless the item is summed
        # and correlated, which is the only case that reads it.
        self.loss_correlation_by_item = np.zeros(max_items_per_coverage, dtype=oasis_float)
        self.hermite_coeffs = np.zeros(HERMITE_TERMS, dtype='float64')

        # maximum bytes to be written in the output stream for 1 item. A kept-separate item emits
        # one block of that per building; a summed one emits a single block whatever it carries.
        # One block: the item header, this building's NUM_IDX specials and S samples, and the
        # delimiter. Only the first block of an item carries the header and only the last the
        # delimiter, so charging every block for both is a deliberate over-estimate.
        max_bytes_per_block = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size
        max_bytes_per_item = max_bytes_per_block * max_emitted_blocks(items['packed_buildings'])

        # define vulnerability cdf cache size
        max_cached_vuln_cdf_size_bytes = max_cached_vuln_cdf_size_MB * 1024 * 1024  # cache size in bytes
        max_Nnumbers_cached_vuln_cdf = max_cached_vuln_cdf_size_bytes // oasis_float.itemsize  # total numbers that can fit in the cache
        # max number of vulnerability functions that can be stored in cache
        max_Nvulnerability_cached_vuln_cdf = max_Nnumbers_cached_vuln_cdf // Ndamage_bins_max
        # number of vulnerability functions to be cached, rounded up to power of two for bitwise masking.
        # The cache must fit at least one full multi-block write for a single item — that is
        # 1 effective damage CDF + Nintensity_bins per-bin vuln CDFs — otherwise a block can
        # trample itself while wrapping around, producing silently corrupted CDF reads.
        Nvulns_cached = min(Nvulnerability * Nintensity_bins, max_Nvulnerability_cached_vuln_cdf)
        Nvulns_cached = max(Nvulns_cached, Nintensity_bins + 1)
        Nvulns_cached = 1 << (max(Nvulns_cached, 1) - 1).bit_length()  # next power of two
        self.cdf_cache_mask = np.int64(Nvulns_cached - 1)
        logger.info(f"max vulnerability cdf cache size is {max_cached_vuln_cdf_size_MB}MB")
        logger.info(
            f"generating a cache of shape ({Nvulns_cached}, {Ndamage_bins_max}) and size "
            f"{Nvulns_cached * Ndamage_bins_max * oasis_float.itemsize / 1024 / 1024:8.3f}MB")

        self.event_footprint_obj = FootprintLayerClient if data_server else footprint_obj

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

        # default random values array for sample_size==0 case
        self.haz_eps_ij = np.empty((1, sample_size), dtype='float64')
        self.damage_eps_ij = np.empty((1, sample_size), dtype='float64')

        # one row of sample_size per rng group; not built at all on generator 2, which draws
        # each building's block where it is used
        self.vuln_rndms_base = np.empty((1, 1), dtype='float64')
        self.haz_rndms_base = np.empty((1, 1), dtype='float64')

        # Pre-allocate per-event footprint arrays (reused across events, no per-event allocation)
        # fp_ap_inds stores dense areaperil indices (from item_map_ja_id_ind) rather than raw
        # areaperil_ids, so reconstruct_coverages can index item_map_ja_offsets directly without
        # repeating the id_index lookup that was already done in process_areaperils_in_footprint.
        n_max_areaperils = len(item_map_ja_areaperil_ids)
        self.fp_ap_inds = np.empty(n_max_areaperils, dtype=np.uint32)
        self.fp_event_rps = np.empty(n_max_areaperils, dtype=np.int32)
        self.fp_haz_arr_ptr = np.empty(n_max_areaperils + 1, dtype=np.int64)

        # Pre-allocate CDF cache once (reused across events, no need to zero)
        self.cached_vuln_cdfs = np.empty((Nvulns_cached, Ndamage_bins_max), dtype=oasis_float)
        self.cdf_cache_tag = np.full(n_cdf_groups, gulmc_manager.CDF_CACHE_EMPTY, dtype=np.int64)
        self.cdf_cache_nbins = np.zeros(Nvulns_cached, dtype=np.int32)

        self.sample_size = sample_size
        self.dynamic_footprint = dynamic_footprint
        self.items = items
        self.coverages = coverages
        self.vuln_array = vuln_array
        self.conditional_vuln_array = conditional_vuln_array
        self.vuln_idx_to_cond_idx = vuln_idx_to_cond_idx
        self.norm_inv_parameters = norm_inv_parameters
        self.norm_inv_cdf = norm_inv_cdf
        self.norm_cdf = norm_cdf
        self.haz_peril_correlation_groups = haz_peril_correlation_groups
        self.haz_corr_seeds = haz_corr_seeds
        self.damage_peril_correlation_groups = damage_peril_correlation_groups
        self.damage_corr_seeds = damage_corr_seeds
        self.coverage_source_id = coverage_source_id
        self.coverage_dependents_ja_offsets = coverage_dependents_ja_offsets
        self.coverage_dependents_ja_data = coverage_dependents_ja_data
        self.coverage_has_dependents = coverage_has_dependents
        self.compute_info = compute_info

    def process_event(self, event_id, out):
        """Compute the ground-up losses of one event and write its bytes to ``out``.

        ``out.write`` is called once each time the compute kernel fills its buffer, so zero or
        more times per event; the bytes passed are a view of the stage's buffer and must be
        consumed before ``write`` returns. An event with no footprint, or whose footprint touches
        none of the portfolio's area perils, writes nothing.

        Args:
            event_id (int): the event to compute.
            out: sink with a ``write(data)`` method, ``data`` being a ``memoryview`` of bytes.

        Returns:
            bool: False if the event was skipped because its footprint touches no item, True otherwise
            (including an event with no footprint). ``run`` skips its progress-ping check after
            the former, as gulmc always has.

        Raises:
            RuntimeError: if the compute kernel asks to be resumed without having made progress.
        """
        compute_info = self.compute_info
        event_id = np.int32(event_id)
        # get the next event_id from the input stream
        compute_info['event_id'] = event_id
        event_footprint = self.event_footprint_obj.get_event(event_id)

        if event_footprint is None:
            logger.info(f"event {event_id} SKIPPED - no footprint")
            return True

        Nhaz_arr_this_event, haz_pdf = gulmc_manager.process_areaperils_in_footprint(
            event_footprint,
            self.item_map_ja_id_ind,
            self.dynamic_footprint,
            self.fp_ap_inds,
            self.fp_event_rps,
            self.fp_haz_arr_ptr)
        if Nhaz_arr_this_event == 0:
            # no items to be computed for this event
            logger.info(f"event {event_id} SKIPPED - no items")
            return False

        items_event_data, rng_index, hazard_rng_index, self.byte_mv = gulmc_manager.reconstruct_coverages(
            compute_info,
            self.fp_ap_inds,
            Nhaz_arr_this_event,
            self.fp_haz_arr_ptr,
            self.fp_event_rps,
            self.item_map_ja_offsets,
            self.item_map_ja_vuln_ja_offsets,
            self.item_map_ja_vuln_ja_item_idxs,
            self.items,
            self.item_cdf_group_idx,
            self.coverages,
            self.compute,
            self.haz_seeds,
            self.haz_peril_correlation_groups,
            self.haz_corr_seeds,
            self.vuln_seeds,
            self.damage_peril_correlation_groups,
            self.damage_corr_seeds,
            self.dynamic_footprint,
            self.byte_mv,
            self.group_seq_rng_index,
            self.hazard_group_seq_rng_index,
            self.coverage_source_id,
            self.coverage_dependents_ja_offsets,
            self.coverage_dependents_ja_data,
            self.compute_depth,
            self.compute_footprint_order,
            self.dependency_dfs_stack,
            self.source_item_idx,
            self.item_idx_to_item_j
        )

        sample_size = self.sample_size
        # since these are never used outside of a sample > 0 branch we can remove the need to
        # generate (and potentially allocate) the random values. As at 2.3.5 the sampling method
        # for random values accounts for 25% of the runtime of the losses step not including
        # the get_event despite having a sample size of 0.
        if sample_size > 0:
            # One row of sample_size per rng GROUP, which is what a packed item's first
            # building gets too: random_LatinHypercube_Philox7 is a loop over
            # _lh_philox_block at building coordinate 0. So this row is shared by every
            # item of the group, and only buildings 2..N are drawn per item below.
            self.vuln_rndms_base = self.generate_rndm(self.vuln_seeds[:rng_index], sample_size)
            self.haz_rndms_base = self.generate_rndm(self.haz_seeds[:hazard_rng_index], sample_size)
            if hazard_rng_index > 0:
                self.haz_eps_ij = self.generate_rndm(self.haz_corr_seeds, sample_size, skip_seeds=1)
            self.damage_eps_ij = self.generate_rndm(self.damage_corr_seeds, sample_size, skip_seeds=1)

        # Reset CDF cache lookup per event (cached_vuln_cdfs array is reused, no reallocation)
        self.cdf_cache_tag[:] = gulmc_manager.CDF_CACHE_EMPTY
        compute_info['cdf_cache_ctr'] = 0

        processing_done = False
        logger.info(f"event {event_id} STARTED")
        while not processing_done:
            resume_point_before = int(compute_info['coverage_i'])
            try:
                processing_done = gulmc_manager.compute_event_losses(
                    compute_info,
                    self.coverages,
                    self.compute,
                    items_event_data,
                    self.items,
                    sample_size,
                    haz_pdf,
                    self.fp_haz_arr_ptr,
                    self.vuln_array,
                    self.conditional_vuln_array,
                    self.vuln_idx_to_cond_idx,
                    self.damage_bins,
                    self.cdf_cache_tag,
                    self.cdf_cache_nbins,
                    self.cdf_cache_mask,
                    self.cached_vuln_cdfs,
                    self.areaperil_agg_vuln_idx_ja_offsets,
                    self.areaperil_agg_vuln_idx_ja_data,
                    self.losses,
                    self.vuln_adj,
                    self.haz_eps_ij,
                    self.damage_eps_ij,
                    self.norm_inv_parameters,
                    self.norm_inv_cdf,
                    self.norm_cdf,
                    self.vuln_z_unif,
                    self.haz_z_unif,
                    self.byte_mv,
                    self.dynamic_footprint,
                    self.intensity_bin_peril_ids,
                    self.intensity_bins,
                    self.building_losses,
                    self.loss_correlation_by_item,
                    self.hermite_coeffs,
                    self.vuln_rndms_base,
                    self.haz_rndms_base,
                    self.vuln_seeds,
                    self.haz_seeds,
                    self.vuln_draw_scratch,
                    self.haz_draw_scratch,
                    self.perm_scratch,
                    self.coverage_has_dependents,
                    self.compute_depth,
                    self.source_damage_bin_stack,
                    self.source_eff_damage_cdf_stack,
                    self.source_eff_damage_cdf_len_stack
                )
            except Exception:
                data = {
                    "event_id": event_id
                }
                with open("event_error.json", "w") as f:
                    json.dump(data, f, default=str)

                logger.error(f"event id={event_id} failed in summary")
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
                    f"{compute_info['cursor']} bytes into a {self.byte_mv.shape[0]} byte "
                    f"buffer, which must hold the largest coverage whole.")
            if compute_info['cursor'] > 0:
                out.write(memoryview(self.byte_mv[:compute_info['cursor']]))

        logger.info(f"event {event_id} DONE")
        return True

    def close(self):
        """Release the footprint and any other resource the stage holds."""
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
