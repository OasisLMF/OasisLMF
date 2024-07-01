import atexit
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from select import select

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
from numba import njit
from numba.typed import Dict, List
from numba.types import Tuple as nb_Tuple
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import nb_areaperil_int, oasis_float
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.data_layer.oasis_files.correlations import Correlation, read_correlations
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.manager import get_damage_bins, get_vulns, get_vuln_rngadj_dict, convert_vuln_id_to_index, get_intensity_bin_dict
from oasislmf.pytools.gul.common import MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX, NUM_IDX
from oasislmf.pytools.gul.core import compute_mean_loss, get_gul
from oasislmf.pytools.gul.manager import get_coverages, write_losses, adjust_byte_mv_size
from oasislmf.pytools.gul.random import (compute_norm_cdf_lookup, compute_norm_inv_cdf_lookup,
                                         generate_correlated_hash_vector, generate_hash,
                                         generate_hash_hazard, get_corr_rval, get_random_generator)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.gulmc.aggregate import (
    map_agg_vuln_ids_to_agg_vuln_idxs,
    map_areaperil_vuln_id_to_weight_to_areaperil_vuln_idx_to_weight,
    process_aggregate_vulnerability, process_vulnerability_weights, read_aggregate_vulnerability,
    read_vulnerability_weights)
from oasislmf.pytools.gulmc.common import (AREAPERIL_TO_EFF_VULN_KEY_TYPE,
                                           AREAPERIL_TO_EFF_VULN_VALUE_TYPE,
                                           NP_BASE_ARRAY_SIZE,
                                           Item, Keys,
                                           NormInversionParameters, coverage_type, gul_header,
                                           gulSampleslevelHeader_size, gulSampleslevelRec_size,
                                           haz_cdf_type, items_MC_data_type)
from oasislmf.pytools.gulmc.items import generate_item_map, process_items, read_items
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


VULN_LOOKUP_KEY_TYPE = nb_Tuple((nb_int32, nb_int32))
VULN_LOOKUP_VALUE_TYPE = nb_Tuple((nb_int32, nb_int32))


@njit(cache=True)
def gen_empty_vuln_cdf_lookup(list_size):
    """Generate structures needed to store and retrieve vulnerability cdf in the cache.

    Args:
        list_size (int): maximum number of cdfs to be stored in the cache.

    Returns:
        cached_vuln_cdf_lookup (Dict[VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE]): dict to store
          the map between vuln_id and intensity bin id and the location of the cdf in the cache.
        cached_vuln_cdf_lookup_keys (List[VULN_LOOKUP_VALUE_TYPE]): list of lookup keys.
    """
    cached_vuln_cdf_lookup = Dict.empty(VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE)
    cached_vuln_cdf_lookup_keys = List.empty_list(VULN_LOOKUP_VALUE_TYPE)
    dummy = tuple((nb_int32(-1), nb_int32(-1)))
    for _ in range(list_size):
        cached_vuln_cdf_lookup_keys.append(dummy)

    return cached_vuln_cdf_lookup, cached_vuln_cdf_lookup_keys


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
    """Execute the main gulmc worklow.

    Args:
        run_dir (str): the directory of where the process is running
        ignore_file_type set(str): file extension to ignore when loading
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
        effective_damageability (bool, optional): if True, it uses effective damageability to draw damage samples instead of
          using the full monte carlo approach (i.e., to draw hazard intensity first, then damage).
        max_cached_vuln_cdf_size_MB (int, optional): size in MB of the in-memory cache to store and reuse vulnerability cdf. Defaults to 200.
        model_df_engine: (str) The engine to use when loading model dataframes
    Raises:
        ValueError: if alloc_rule is not 0, 1, 2, or 3.
        ValueError: if alloc_rule is 1, 2, or 3 when debug is 1 or 2.

    Returns:
        int: 0 if no errors occurred.
    """
    logger.info("starting gulmc")

    model_storage = get_storage_from_config_path(
        os.path.join(run_dir, 'model_storage.json'),
        os.path.join(run_dir, 'static'),
    )
    input_path = os.path.join(run_dir, 'input')
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

        # load keys.csv to determine included AreaPerilID from peril_filter
        if peril_filter:
            keys_df = pd.read_csv(os.path.join(input_path, 'keys.csv'), dtype=Keys)
            valid_area_peril_id = keys_df.loc[keys_df['PerilID'].isin(peril_filter), 'AreaPerilID'].to_numpy()
            logger.debug(
                f'Peril specific run: ({peril_filter}), {len(valid_area_peril_id)} AreaPerilID included out of {len(keys_df)}')
        else:
            valid_area_peril_id = None

        logger.debug('import damage bins')
        damage_bins = get_damage_bins(model_storage, ignore_file_type)

        logger.debug('import coverages')
        # coverages are numbered from 1, therefore we skip element 0 in `coverages`
        coverages_tb = get_coverages(input_path, ignore_file_type)
        coverages = np.zeros(coverages_tb.shape[0] + 1, coverage_type)
        coverages[1:]['tiv'] = coverages_tb

        # prepare for stochastic disaggregation
        logger.debug('import aggregate vulnerability definitions and vulnerability weights')
        aggregate_vulnerability = read_aggregate_vulnerability(model_storage, ignore_file_type)
        aggregate_weights = read_vulnerability_weights(model_storage, ignore_file_type)
        agg_vuln_to_vuln_id = process_aggregate_vulnerability(aggregate_vulnerability)

        if aggregate_vulnerability is not None and aggregate_weights is None:
            raise FileNotFoundError(
                f"Vulnerability weights file not found at {model_storage.get_storage_url('', print_safe=True)[1]}"
            )

        # create map of weights by (areaperil_id, vuln_id)
        areaperil_vuln_id_to_weight = process_vulnerability_weights(aggregate_weights, agg_vuln_to_vuln_id)

        logger.debug('import items and correlations tables')
        # since items and correlations have the same granularity (one row per item_id) we merge them on `item_id`.
        correlations_tb = read_correlations(input_path, ignore_file_type)
        items_tb = read_items(input_path, ignore_file_type)
        if len(correlations_tb) != len(items_tb):
            logger.info(
                f"The items table has length {len(items_tb)} while the correlations table has length {len(correlations_tb)}.\n"
                "It is possible that the correlations are not set up properly in the model settings file."
            )

        # merge the tables, using defaults for missing values, and sort the resulting table
        items = rfn.join_by(
            'item_id', items_tb, correlations_tb,
            jointype='leftouter', usemask=False,
            defaults={'peril_correlation_group': 0,
                      'damage_correlation_value': 0.,
                      'hazard_group_id': 0,
                      'hazard_correlation_value': 0.}
        )
        items.sort(order=['areaperil_id', 'vulnerability_id'])
        # build item map
        item_map, areaperil_ids_map = generate_item_map(items, coverages)
        # import array to store the coverages to be computed
        # coverages are numebered from 1, therefore skip element 0.
        compute = np.zeros(coverages.shape[0] + 1, Item.dtype['coverage_id'])

        logger.debug('import peril correlation groups')
        unique_peril_correlation_groups = np.unique(items['peril_correlation_group'])
        Nperil_correlation_groups = unique_peril_correlation_groups.shape[0]
        logger.info(f"Detected {Nperil_correlation_groups} peril correlation groups.")

        vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns, areaperil_dict, used_agg_vuln_ids = process_items(
            items, valid_area_peril_id, agg_vuln_to_vuln_id)

        logger.debug('import footprint')
        footprint_obj = stack.enter_context(Footprint.load(model_storage, ignore_file_type, df_engine=model_df_engine))
        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
        else:
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        logger.debug('import vulnerabilities')
        vuln_adj_dict = get_vuln_rngadj_dict(run_dir, vuln_dict)
        vuln_array, _, _ = get_vulns(model_storage, run_dir, vuln_dict, num_intensity_bins, ignore_file_type, df_engine=model_df_engine)
        Nvulnerability, Ndamage_bins_max, Nintensity_bins = vuln_array.shape
        convert_vuln_id_to_index(vuln_dict, areaperil_to_vulns)

        logger.debug('reconstruct aggregate vulnerability definitions and weights')
        # map each vulnerability_id composing aggregate vulnerabilities to the indices where they are stored in vuln_array
        # here we filter out aggregate vulnerability that are not used in this portfolio, therefore
        # agg_vuln_to_vuln_idxs can contain less aggregate vulnerability ids compared to agg_vuln_to_vuln_id
        agg_vuln_to_vuln_idxs = map_agg_vuln_ids_to_agg_vuln_idxs(used_agg_vuln_ids, agg_vuln_to_vuln_id, vuln_dict)

        # remap (areaperil, vuln_id) to weights to (areaperil, vuln_idx) to weights
        areaperil_vuln_idx_to_weight = map_areaperil_vuln_id_to_weight_to_areaperil_vuln_idx_to_weight(
            areaperil_dict, areaperil_vuln_id_to_weight, vuln_dict)

        # set up streams
        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        select_stream_list = [stream_out]

        # prepare output stream
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())
        cursor = 0

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)
        # create the array to store the seeds
        haz_seeds = np.zeros(len(np.unique(items['hazard_group_id'])), dtype=Correlation.dtype['hazard_group_id'])
        vuln_seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])

        # haz correlation
        do_haz_correlation = False
        if ignore_haz_correlation:
            logger.info("correlated random number generation for hazard intensity sampling: switched OFF because --ignore-haz-correlation is True.")

        else:
            if Nperil_correlation_groups > 0 and any(items['hazard_correlation_value'] > 0):
                do_haz_correlation = True

            else:
                logger.info("correlated random number generation for hazard intensity sampling: switched OFF because 0 peril correlation groups were detected or "
                            "the hazard correlation value is zero for all peril correlation groups.")

        # damage correlation
        do_correlation = False
        if ignore_correlation:
            logger.info("correlated random number generation for damage sampling: switched OFF because --ignore-correlation is True.")

        else:
            if Nperil_correlation_groups > 0 and any(items['damage_correlation_value'] > 0):
                do_correlation = True
            else:
                logger.info("correlated random number generation for damage sampling: switched OFF because 0 peril correlation groups were detected or "
                            "the damage correlation value is zero for all peril correlation groups.")

        if do_correlation or do_haz_correlation:
            logger.info(f"correlated random number generation for hazard intensity sampling: switched {'ON' if do_haz_correlation else 'OFF'}.")
            logger.info(f"Correlated random number generation for damage sampling: switched  {'ON' if do_correlation else 'OFF'}.")

            logger.info(f"Correlation values for {Nperil_correlation_groups} peril correlation groups have been imported.")

            # pre-compute lookup tables for the Gaussian cdf and inverse cdf
            # Notes:
            #  - the size `N` can be increased to achieve better resolution in the Gaussian cdf and inv cdf.
            #  - the function `get_corr_rval` to compute the correlated numbers is not affected by N.
            norm_inv_parameters = np.array((1e-16, 1 - 1e-16, 1000000, -20., 20.), dtype=NormInversionParameters)
            norm_inv_cdf = compute_norm_inv_cdf_lookup(norm_inv_parameters['x_min'], norm_inv_parameters['x_max'], norm_inv_parameters['N'])
            norm_cdf = compute_norm_cdf_lookup(norm_inv_parameters['cdf_min'], norm_inv_parameters['cdf_max'], norm_inv_parameters['N'])

            # buffer to be re-used to store all the correlated random values
            z_unif = np.zeros(sample_size, dtype='float64')

        else:
            # create dummy data structures with proper dtypes to allow correct numba compilation
            norm_inv_parameters = np.array((0., 0., 0, 0., 0.), dtype=NormInversionParameters)
            norm_inv_cdf, norm_cdf = np.zeros(1, dtype='float64'), np.zeros(1, dtype='float64')
            z_unif = np.zeros(1, dtype='float64')

        if effective_damageability is True:
            logger.info("effective_damageability is True: gulmc will draw the damage samples from the effective damageability distribution.")
        else:
            logger.info("effective_damageability is False: gulmc will perform the full Monte Carlo sampling: "
                        "sample the hazard intensity first, then sample the damage from the corresponding vulnerability function.")

        # create buffers to be reused when computing losses
        byte_mv = np.empty(PIPE_CAPACITY * 2, dtype='b')
        losses = np.zeros((sample_size + NUM_IDX + 1, np.max(coverages[1:]['max_items'])), dtype=oasis_float)
        vuln_cdf_empty = np.zeros(Ndamage_bins_max, dtype=oasis_float)
        weighted_vuln_cdf_empty = np.zeros(Ndamage_bins_max, dtype=oasis_float)

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = gulSampleslevelHeader_size + (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size

        # define vulnerability cdf cache size
        max_cached_vuln_cdf_size_bytes = max_cached_vuln_cdf_size_MB * 1024 * 1024  # cahce size in bytes
        max_Nnumbers_cached_vuln_cdf = max_cached_vuln_cdf_size_bytes // oasis_float.itemsize  # total numbers that can fit in the cache
        max_Nvulnerability_cached_vuln_cdf = max_Nnumbers_cached_vuln_cdf // Ndamage_bins_max  # max number of vulnerability funcions that can be stored in cache
        # number of vulnerability functions to be cached
        Nvulns_cached = min(Nvulnerability * Nintensity_bins, max_Nvulnerability_cached_vuln_cdf)
        logger.info(f"max vulnerability cdf cache size is {max_cached_vuln_cdf_size_MB}MB")
        logger.info(
            f"generating a cache of shape ({Nvulns_cached}, {Ndamage_bins_max}) and size {Nvulns_cached * Ndamage_bins_max * oasis_float.itemsize / 1024 / 1024:8.3f}MB")

        # maximum bytes to be written in the output stream for 1 item
        event_footprint_obj = FootprintLayerClient if data_server else footprint_obj

        if dynamic_footprint:
            intensity_bin_dict = get_intensity_bin_dict(os.path.join(run_dir, 'static'))
        else:
            intensity_bin_dict = Dict.empty(nb_int32, nb_int32)
        # to do - intensity adjustment
        # intensity_adjustment = get_intensity_adjustment()

        while True:
            if not streams_in.readinto(event_id_mv):
                break

            # get the next event_id from the input stream
            event_id = event_ids[0]
            event_footprint = event_footprint_obj.get_event(event_id)

            if event_footprint is not None:
                areaperil_ids, Nhaz_cdf_this_event, areaperil_to_haz_cdf, haz_cdf, haz_cdf_ptr, eff_vuln_cdf, areaperil_to_eff_vuln_cdf = process_areaperils_in_footprint(
                    event_footprint, vuln_array,
                    areaperil_to_vulns_idx_dict,
                    areaperil_to_vulns_idx_array,
                    areaperil_to_vulns)

                if Nhaz_cdf_this_event == 0:
                    # no items to be computed for this event
                    continue

                compute_i, items_event_data, rng_index, hazard_rng_index = reconstruct_coverages(event_id,
                                                                                                 areaperil_ids,
                                                                                                 areaperil_ids_map,
                                                                                                 areaperil_to_haz_cdf,
                                                                                                 item_map,
                                                                                                 items,
                                                                                                 coverages,
                                                                                                 compute,
                                                                                                 haz_seeds,
                                                                                                 vuln_seeds
                                                                                                 )

                # generation of "base" random values for hazard intensity and vulnerability sampling
                haz_rndms_base = generate_rndm(haz_seeds[:hazard_rng_index], sample_size)
                vuln_rndms_base = generate_rndm(vuln_seeds[:rng_index], sample_size)

                if do_haz_correlation:
                    haz_corr_seeds = generate_correlated_hash_vector(unique_peril_correlation_groups, event_id)
                    haz_eps_ij = generate_rndm(haz_corr_seeds, sample_size, skip_seeds=1)

                else:
                    # create dummy data structures with proper dtypes to allow correct numba compilation
                    haz_corr_seeds = np.zeros(1, dtype='int64')
                    haz_eps_ij = np.zeros((1, 1), dtype='float64')

                # generate the correlated samples for the whole event, for all peril correlation groups
                if do_correlation:
                    corr_seeds = generate_correlated_hash_vector(unique_peril_correlation_groups, event_id)
                    eps_ij = generate_rndm(corr_seeds, sample_size, skip_seeds=1)

                else:
                    # create dummy data structures with proper dtypes to allow correct numba compilation
                    eps_ij = np.zeros((1, 1), dtype='float64')

                last_processed_coverage_ids_idx = 0

                byte_mv = adjust_byte_mv_size(byte_mv, np.max(coverages['cur_items']) * max_bytes_per_item)

                # create vulnerability cdf cache
                cached_vuln_cdfs = np.zeros((Nvulns_cached, Ndamage_bins_max), dtype=oasis_float)
                cached_vuln_cdf_lookup, lookup_keys = gen_empty_vuln_cdf_lookup(Nvulns_cached)
                next_cached_vuln_cdf = 0

                while last_processed_coverage_ids_idx < compute_i:

                    cursor, last_processed_coverage_ids_idx, next_cached_vuln_cdf = compute_event_losses(
                        event_id,
                        coverages,
                        compute[:compute_i],
                        items_event_data,
                        items,
                        last_processed_coverage_ids_idx,
                        sample_size,
                        haz_cdf,
                        haz_cdf_ptr,
                        areaperil_to_eff_vuln_cdf,
                        eff_vuln_cdf,
                        vuln_array,
                        damage_bins,
                        Ndamage_bins_max,
                        cached_vuln_cdf_lookup,
                        lookup_keys,
                        next_cached_vuln_cdf,
                        cached_vuln_cdfs,
                        agg_vuln_to_vuln_id,
                        agg_vuln_to_vuln_idxs,
                        vuln_dict,
                        areaperil_vuln_idx_to_weight,
                        loss_threshold,
                        losses,
                        vuln_cdf_empty,
                        weighted_vuln_cdf_empty,
                        alloc_rule,
                        do_correlation,
                        do_haz_correlation,
                        haz_rndms_base,
                        vuln_rndms_base,
                        vuln_adj_dict,
                        haz_eps_ij,
                        eps_ij,
                        norm_inv_parameters,
                        norm_inv_cdf,
                        norm_cdf,
                        z_unif,
                        effective_damageability,
                        debug,
                        max_bytes_per_item,
                        byte_mv,
                        cursor,
                        dynamic_footprint,
                        intensity_bin_dict
                    )

                    # write the losses to the output stream
                    write_start = 0
                    while write_start < cursor:
                        select([], select_stream_list, select_stream_list)
                        write_start += stream_out.write(byte_mv[write_start:cursor].tobytes())

                    cursor = 0

                logger.info(f"event {event_id} DONE")

    return 0


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id,
                         coverages,
                         coverage_ids,
                         items_event_data,
                         items,
                         last_processed_coverage_ids_idx,
                         sample_size,
                         haz_cdf,
                         haz_cdf_ptr,
                         areaperil_to_eff_vuln_cdf,
                         eff_vuln_cdf,
                         vuln_array,
                         damage_bins,
                         Ndamage_bins_max,
                         cached_vuln_cdf_lookup,
                         cached_vuln_cdf_lookup_keys,
                         next_cached_vuln_cdf,
                         cached_vuln_cdfs,
                         agg_vuln_to_vuln_id,
                         agg_vuln_to_vuln_idxs,
                         vuln_dict,
                         areaperil_vuln_idx_to_weight,
                         loss_threshold,
                         losses,
                         vuln_cdf_empty,
                         weighted_vuln_cdf_empty,
                         alloc_rule,
                         do_correlation,
                         do_haz_correlation,
                         haz_rndms_base,
                         vuln_rndms_base,
                         vuln_adj_dict,
                         haz_eps_ij,
                         eps_ij,
                         norm_inv_parameters,
                         norm_inv_cdf,
                         norm_cdf,
                         z_unif,
                         effective_damageability,
                         debug,
                         max_bytes_per_item,
                         byte_mv,
                         cursor,
                         dynamic_footprint,
                         intensity_bin_dict):
    """Compute losses for an event.

    Args:
        event_id (int32): event id.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        coverage_ids (numpy.array[int]): array of unique coverage ids used in this event.
        items_data (numpy.array[items_data_type]): items-related data.
        items (np.ndarray): items table merged with correlation parameters.
        last_processed_coverage_ids_idx (int): index of the last coverage_id stored in `coverage_ids` that was fully processed
          and printed to the output stream.
        sample_size (int): number of random samples to draw.
        haz_cdf (np.array[oasis_float]): hazard intensity cdf.
        haz_cdf_ptr (np.array[int]): array with the indices where each cdf record starts in `haz_cdf`.
        areaperil_to_eff_vuln_cdf (dict[ITEM_MAP_KEY_TYPE_internal, int]): map between `(areaperil_id, vuln_idx)` and the location
          where the effective damageability function is stored in `eff_vuln_cdf`.
        eff_vuln_cdf (np.array[oasis_float]): effective damageability cdf.
        vuln_array (np.array[float]): damage pdf for different vulnerability functions, as a function of hazard intensity.
        damage_bins (List[Union[damagebindictionaryCsv, damagebindictionary]]): loaded data from the damage_bin_dict file.
        Ndamage_bins_max (int): maximum number of damage bins.
        cached_vuln_cdf_lookup (Dict[VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE]): dict to store
          the map between vuln_id and intensity bin id and the location of the cdf in the cache.
        cached_vuln_cdf_lookup_keys (List[VULN_LOOKUP_VALUE_TYPE]): list of lookup keys.
        next_cached_vuln_cdf (int): index of the next free slot in the vuln cdf cache.
        cached_vuln_cdfs (np.array[oasis_float]): vulnerability cdf cache.
        agg_vuln_to_vuln_id (dict[int, list[int]]): map of aggregate vulnerability id to list of vulnerability ids.
        agg_vuln_to_vuln_idxs (dict[int, list[int]]): map between aggregate vulnerability id and the list of indices where the individual vulnerability_ids
          that compose it are stored in `vuln_array`.
        vuln_dict (Dict[int, int]): map between vulnerability_id and the index where the vulnerability function is stored in vuln_array.
        areaperil_vuln_idx_to_weight (dict[AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE]): map between the areaperil id and the index where the vulnerability function
          is stored in `vuln_array` and the vulnerability weight.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): array (to be re-used) to store losses for each item.
        vuln_cdf_empty (numpy.array[oasis_float]): array (to be re-used) to store vulnerability cdf.
        weighted_vuln_cdf_empty (numpy.array[oasis_float]): array (to be re-used) to store the weighted vulnerability cdf.
        alloc_rule (int): back-allocation rule.
        do_correlation (bool): if True, compute correlated random samples of damage.
        do_haz_correlation (bool): if True, compute correlated random samples of hazard intensity.
        haz_rndms_base (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed for the hazard intensity sampling.
        vuln_rndms_base (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed for the damage sampling.
        vuln_adj_dict (dict[int, float]): map between vulnerability_id and the adjustment factor to be applied to the (random numbers extracted) vulnerability function.
        haz_eps_ij (np.array[float]): correlated random values of shape `(number of seeds, sample_size)` for hazard sampling.
        eps_ij (np.array[float]): correlated random values of shape `(number of seeds, sample_size)` for damage sampling.
        norm_inv_parameters (NormInversionParameters): parameters for the Normal (Gaussian) inversion functionality.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf.
        norm_cdf (np.array[float]): Gaussian cdf.
        z_unif (np.array[float]): buffer to be re-used to store all the correlated random values.
        effective_damageability (bool): if True, it uses effective damageability to draw damage samples instead of
          using the full monte carlo approach (i.e., to draw hazard intensity first, then damage).
        debug (int): for each random sample, print to the output stream the random loss (if 0),
          the random value used to draw the hazard intensity sample (if 1), the random value used to draw the damage sample (if 2).
        max_bytes_per_item (int): maximum bytes to be written in the output stream for an item.
        byte_mv (numpy.array): byte view of where the output is buffered.
        cursor (int): index of byte_mv where to start writing.

    Returns:
        cursor (int): index of byte_mv where to data has been written.
        last_processed_coverage_ids_idx (int): index of the last coverage_id stored in `coverage_ids` that was fully processed
          and printed to the output stream.
        next_cached_vuln_cdf (int): index of the next free slot in the vuln cdf cache.
    """
    # loop through all the coverages that remain to be computed
    for coverage_i in range(last_processed_coverage_ids_idx, coverage_ids.shape[0]):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']
        Nitems = coverage['cur_items']
        exposureValue = tiv / Nitems

        # estimate max number of bytes needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        est_cursor_bytes = Nitems * max_bytes_per_item

        # return before processing this coverage if the number of free bytes left in the buffer
        # is not sufficient to write out the full coverage
        if cursor + est_cursor_bytes > byte_mv.shape[0]:
            return cursor, last_processed_coverage_ids_idx, next_cached_vuln_cdf

        # compute losses for each item
        for item_j in range(Nitems):
            item_event_data = items_event_data[coverage['start_items'] + item_j]
            item_id = item_event_data['item_id']
            rng_index = item_event_data['rng_index']
            hazard_rng_index = item_event_data['hazard_rng_index']

            item = items[item_event_data['item_idx']]
            areaperil_id = item['areaperil_id']
            vulnerability_id = item['vulnerability_id']

            if not effective_damageability:
                # get the right hazard cdf from the array containing all hazard cdfs
                hazcdf_i = item_event_data['hazcdf_i']
                haz_cdf_record = haz_cdf[haz_cdf_ptr[hazcdf_i]:haz_cdf_ptr[hazcdf_i + 1]]
                haz_cdf_prob = haz_cdf_record['probability']
                haz_cdf_bin_id = haz_cdf_record['intensity_bin_id']
                Nhaz_bins = haz_cdf_ptr[hazcdf_i + 1] - haz_cdf_ptr[hazcdf_i]

            if vulnerability_id in agg_vuln_to_vuln_id:
                # aggregate case: the aggregate effective vuln cdf (agg_eff_vuln_cdf) needs to be computed
                weighted_vuln_cdf = weighted_vuln_cdf_empty
                tot_weights = 0.
                agg_vulns_idx = agg_vuln_to_vuln_idxs[vulnerability_id]

                # here we use loop-unrolling for a more performant code.
                # we explicitly run the first cycle for damage_bin_i=0 in order to cache (eff_vuln_cdf_i, eff_vuln_cdf_Ndamage_bins, weight)
                # that are retrieved through O(1) but costly get calls to numba.typed.Dict.
                # we also filter out the empty cdfs, i.e. the effective cdfs that are completely zero because hazard intensity does not
                # overlap with vulnerability intensity bins; by filtering them once, we can avoid checking in each loop.
                damage_bin_i = nb_int32(0)
                cumsum = 0.
                used_vuln_data = []
                for vuln_i in agg_vulns_idx:
                    eff_vuln_cdf_i, eff_vuln_cdf_Ndamage_bins = areaperil_to_eff_vuln_cdf[(areaperil_id, vuln_i)]
                    if (areaperil_id, vuln_i) in areaperil_vuln_idx_to_weight:
                        weight = np.float64(areaperil_vuln_idx_to_weight[(areaperil_id, vuln_i)])
                    else:
                        weight = np.float64(0.)

                    if eff_vuln_cdf[eff_vuln_cdf_i + eff_vuln_cdf_Ndamage_bins - 1] == 0.:
                        # the cdf is all zeros, i.e. probability of no loss is 100%
                        # store it as (weighted) 1.0 in the first damage bin and do not include it in the next bins.
                        pdf_bin = weight
                    else:
                        pdf_bin = eff_vuln_cdf[eff_vuln_cdf_i] * weight
                        used_vuln_data.append((eff_vuln_cdf_i, eff_vuln_cdf_Ndamage_bins, weight))

                    tot_weights += weight
                    cumsum += pdf_bin

                if tot_weights == 0.:
                    print("Impossible to compute the cdf of the following aggregate vulnerability_id because individual weights are all zero.\n"
                          "Please double check the weights table for the areaperil_id listed below.")
                    print("aggregate vulnerability_id=", vulnerability_id)
                    print("individual vulnerability_ids=", agg_vulns_idx)
                    print("item_id=", item_id)
                    print("event=", event_id)
                    print("areaperil_id=", areaperil_id)
                    print()
                    raise ValueError(
                        "Impossible to compute the cdf of an aggregate vulnerability_id because individual weights are all zero.")

                weighted_vuln_cdf[damage_bin_i] = cumsum / tot_weights

                # continue with the next bins, if necessary
                damage_bin_i = 1
                if cumsum / tot_weights <= 0.999999940:
                    while damage_bin_i < Ndamage_bins_max:
                        for used_vuln in used_vuln_data:
                            eff_vuln_cdf_i, eff_vuln_cdf_Ndamage_bins, weight = used_vuln

                            if damage_bin_i >= eff_vuln_cdf_Ndamage_bins:
                                # this eff_vuln_cdf is finished
                                pdf_bin = 0.
                            else:
                                pdf_bin = eff_vuln_cdf[eff_vuln_cdf_i + damage_bin_i] - eff_vuln_cdf[eff_vuln_cdf_i + damage_bin_i - 1]
                                pdf_bin *= weight

                            cumsum += pdf_bin

                        weighted_vuln_cdf[damage_bin_i] = cumsum / tot_weights
                        damage_bin_i += 1

                        if weighted_vuln_cdf[damage_bin_i - 1] > 0.999999940:
                            break

                Ndamage_bins = damage_bin_i
                eff_damag_cdf_Ndamage_bins = Ndamage_bins
                eff_damag_cdf = weighted_vuln_cdf[:eff_damag_cdf_Ndamage_bins]

            else:
                vuln_i = vuln_dict[vulnerability_id]
                eff_damag_vuln_cdf_i, eff_damag_cdf_Ndamage_bins = areaperil_to_eff_vuln_cdf[(areaperil_id, vuln_i)]
                eff_damag_cdf = eff_vuln_cdf[eff_damag_vuln_cdf_i:eff_damag_vuln_cdf_i + eff_damag_cdf_Ndamage_bins]

            # for relative vulnerability functions, gul are fraction of the tiv
            # for absolute vulnerability functions, gul are absolute values
            computation_tiv = tiv if damage_bins[eff_damag_cdf_Ndamage_bins - 1]['bin_to'] <= 1 else 1.0

            # compute mean loss values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                computation_tiv,
                eff_damag_cdf,
                damage_bins['interpolation'],
                eff_damag_cdf_Ndamage_bins,
                damage_bins[eff_damag_cdf_Ndamage_bins - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_j] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_j] = chance_of_loss
            losses[TIV_IDX, item_j] = exposureValue
            losses[STD_DEV_IDX, item_j] = std_dev
            losses[MEAN_IDX, item_j] = gul_mean

            # compute random losses
            if sample_size > 0:
                if do_haz_correlation:
                    # use correlation definitions to draw correlated random values
                    rho = item['hazard_correlation_value']

                    if rho > 0:
                        get_corr_rval(
                            haz_eps_ij[item['peril_correlation_group']], haz_rndms_base[hazard_rng_index], rho,
                            norm_inv_parameters['x_min'], norm_inv_parameters['x_max'], norm_inv_parameters['N'], norm_inv_cdf,
                            norm_inv_parameters['cdf_min'], norm_inv_parameters['cdf_max'],
                            norm_cdf, sample_size, z_unif
                        )
                        haz_rndms = z_unif

                    else:
                        haz_rndms = haz_rndms_base[hazard_rng_index]

                else:
                    # do not use correlation
                    haz_rndms = haz_rndms_base[hazard_rng_index]

                if do_correlation:
                    # use correlation definitions to draw correlated random values
                    rho = item['damage_correlation_value']

                    if rho > 0:
                        get_corr_rval(
                            eps_ij[item['peril_correlation_group']], vuln_rndms_base[rng_index], rho,
                            norm_inv_parameters['x_min'], norm_inv_parameters['x_max'], norm_inv_parameters['N'], norm_inv_cdf,
                            norm_inv_parameters['cdf_min'], norm_inv_parameters['cdf_max'],
                            norm_cdf, sample_size, z_unif
                        )
                        vuln_rndms = z_unif
                        if vulnerability_id in vuln_adj_dict:
                            vuln_rndms *= vuln_adj_dict[vulnerability_id]

                    else:
                        vuln_rndms = vuln_rndms_base[rng_index]
                        if vulnerability_id in vuln_adj_dict:
                            vuln_rndms *= vuln_adj_dict[vulnerability_id]

                else:
                    # do not use correlation
                    vuln_rndms = vuln_rndms_base[rng_index]
                    if vulnerability_id in vuln_adj_dict:
                        vuln_rndms *= vuln_adj_dict[vulnerability_id]

                if effective_damageability:
                    # draw samples of effective damageability (i.e., intensity-averaged damage probability)

                    for sample_idx in range(1, sample_size + 1):
                        vuln_cdf = eff_damag_cdf
                        Ndamage_bins = eff_damag_cdf_Ndamage_bins

                        vuln_rval = vuln_rndms[sample_idx - 1]

                        if debug == 2:
                            # store the random value used for the damage sampling instead of the loss
                            losses[sample_idx, item_j] = vuln_rval
                            continue

                        # cap `vuln_rval` to the maximum `vuln_cdf` value (which should be 1.)
                        if vuln_rval >= vuln_cdf[Ndamage_bins - 1]:
                            vuln_rval = vuln_cdf[Ndamage_bins - 1] - 0.00000003
                            vuln_bin_idx = Ndamage_bins - 1
                        else:
                            # find the damage cdf bin in which the random value `vuln_rval` falls into
                            vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins)

                        # compute ground-up losses
                        gul = get_gul(
                            damage_bins['bin_from'][vuln_bin_idx],
                            damage_bins['bin_to'][vuln_bin_idx],
                            damage_bins['interpolation'][vuln_bin_idx],
                            vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
                            vuln_cdf[vuln_bin_idx],
                            vuln_rval,
                            computation_tiv,
                        )

                        losses[sample_idx, item_j] = gul * (gul >= loss_threshold)

                else:
                    # full monte carlo (sample hazard intensity and damage independently)

                    if vulnerability_id in agg_vuln_to_vuln_id:
                        # full monte carlo, aggregate vulnerability

                        for sample_idx in range(1, sample_size + 1):

                            # 1) get the intensity bin
                            if Nhaz_bins == 1:
                                # if hazard intensity has no uncertainty, there is no need to sample
                                haz_bin_idx = nb_int32(0)

                            else:
                                # if hazard intensity has a probability distribution, sample it

                                # cap `haz_rval` to the maximum `haz_cdf_prob` value (which should be 1.)
                                haz_rval = haz_rndms[sample_idx - 1]

                                if debug == 1:
                                    # store the random value used for the hazard intensity sampling instead of the loss
                                    losses[sample_idx, item_j] = haz_rval
                                    continue

                                if haz_rval >= haz_cdf_prob[Nhaz_bins - 1]:
                                    haz_bin_idx = nb_int32(Nhaz_bins - 1)
                                else:
                                    # find the hazard intensity cdf bin in which the random value `haz_rval` falls into
                                    haz_bin_idx = nb_int32(binary_search(haz_rval, haz_cdf_prob, Nhaz_bins))

                            # 2) get the hazard intensity bin id
                            haz_int_bin_id = haz_cdf_bin_id[haz_bin_idx]

                            # 3) get the aggregate vulnerability cdf for a given intensity bin id
                            agg_vulns_idx = agg_vuln_to_vuln_idxs[vulnerability_id]
                            weighted_vuln_cdf = weighted_vuln_cdf_empty

                            # cache the weights and compute the total weights
                            tot_weights = 0.
                            used_weights = []
                            for j, vuln_i in enumerate(agg_vulns_idx):
                                if (areaperil_id, vuln_i) in areaperil_vuln_idx_to_weight:
                                    weight = np.float64(areaperil_vuln_idx_to_weight[(areaperil_id, vuln_i)])
                                else:
                                    weight = np.float64(0.)

                                used_weights.append(weight)
                                tot_weights += weight

                            if tot_weights == 0.:
                                print("Impossible to compute the cdf of the following aggregate vulnerability_id because individual weights are all zero.\n"
                                      "Please double check the weights table for the areaperil_id listed below.")
                                print("aggregate vulnerability_id=", vulnerability_id)
                                print("individual vulnerability_ids=", agg_vulns_idx)
                                print("item_id=", item_id)
                                print("event=", event_id)
                                print("areaperil_id=", areaperil_id)
                                print()
                                raise ValueError(
                                    "Impossible to compute the cdf of an aggregate vulnerability_id because individual weights are all zero.")

                            # compute the weighted cdf
                            damage_bin_i = nb_int32(0)
                            cumsum = 0.
                            while damage_bin_i < Ndamage_bins_max:
                                for j, vuln_i in enumerate(agg_vulns_idx):
                                    cumsum += vuln_array[vuln_i, damage_bin_i, haz_int_bin_id - 1] * used_weights[j]

                                weighted_vuln_cdf[damage_bin_i] = cumsum / tot_weights
                                damage_bin_i += 1

                                if weighted_vuln_cdf[damage_bin_i - 1] > 0.999999940:
                                    break

                            Ndamage_bins = damage_bin_i
                            vuln_cdf = weighted_vuln_cdf[:Ndamage_bins]

                            vuln_rval = vuln_rndms[sample_idx - 1]

                            if debug == 2:
                                # store the random value used for the damage sampling instead of the loss
                                losses[sample_idx, item_j] = vuln_rval
                                continue

                            # cap `vuln_rval` to the maximum `vuln_cdf` value (which should be 1.)
                            if vuln_rval >= vuln_cdf[Ndamage_bins - 1]:
                                vuln_rval = vuln_cdf[Ndamage_bins - 1] - 0.00000003
                                vuln_bin_idx = Ndamage_bins - 1
                            else:
                                # find the damage cdf bin in which the random value `vuln_rval` falls into
                                vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins)

                            # compute ground-up losses
                            gul = get_gul(
                                damage_bins['bin_from'][vuln_bin_idx],
                                damage_bins['bin_to'][vuln_bin_idx],
                                damage_bins['interpolation'][vuln_bin_idx],
                                vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
                                vuln_cdf[vuln_bin_idx],
                                vuln_rval,
                                computation_tiv,

                            )

                            losses[sample_idx, item_j] = gul * (gul >= loss_threshold)

                    else:
                        # full monte carlo, individual vulnerability

                        for sample_idx in range(1, sample_size + 1):

                            # 1) get the intensity bin
                            if Nhaz_bins == 1:
                                # if hazard intensity has no uncertainty, there is no need to sample
                                haz_bin_idx = nb_int32(0)

                            else:
                                # if hazard intensity has a probability distribution, sample it

                                # cap `haz_rval` to the maximum `haz_cdf_prob` value (which should be 1.)
                                haz_rval = haz_rndms[sample_idx - 1]

                                if debug == 1:
                                    # store the random value used for the hazard intensity sampling instead of the loss
                                    losses[sample_idx, item_j] = haz_rval
                                    continue

                                if haz_rval >= haz_cdf_prob[Nhaz_bins - 1]:
                                    haz_bin_idx = nb_int32(Nhaz_bins - 1)
                                else:
                                    # find the hazard intensity cdf bin in which the random value `haz_rval` falls into
                                    haz_bin_idx = nb_int32(binary_search(haz_rval, haz_cdf_prob, Nhaz_bins))

                            # 2) get the hazard intensity bin id
                            if dynamic_footprint:
                                haz_int_val = haz_cdf_bin_id[haz_bin_idx]
                                haz_int_bin_id = intensity_bin_dict[haz_int_val]
                            else:
                                haz_int_bin_id = haz_cdf_bin_id[haz_bin_idx]

                            # 3) get the individual vulnerability cdf
                            vuln_i = vuln_dict[vulnerability_id]
                            vuln_cdf, Ndamage_bins, next_cached_vuln_cdf = get_vuln_cdf(vuln_i,
                                                                                        haz_int_bin_id,
                                                                                        cached_vuln_cdf_lookup,
                                                                                        cached_vuln_cdf_lookup_keys,
                                                                                        vuln_array,
                                                                                        vuln_cdf_empty,
                                                                                        Ndamage_bins_max,
                                                                                        cached_vuln_cdfs,
                                                                                        next_cached_vuln_cdf)

                            vuln_rval = vuln_rndms[sample_idx - 1]

                            if debug == 2:
                                # store the random value used for the damage sampling instead of the loss
                                losses[sample_idx, item_j] = vuln_rval
                                continue

                            # cap `vuln_rval` to the maximum `vuln_cdf` value (which should be 1.)
                            if vuln_rval >= vuln_cdf[Ndamage_bins - 1]:
                                vuln_rval = vuln_cdf[Ndamage_bins - 1] - 0.00000003
                                vuln_bin_idx = Ndamage_bins - 1
                            else:
                                # find the damage cdf bin in which the random value `vuln_rval` falls into
                                vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins)

                            # compute ground-up losses
                            # for relative vulnerability functions, gul are scaled by tiv
                            # for absolute vulnerability functions, gul are absolute values
                            gul = get_gul(
                                damage_bins['bin_from'][vuln_bin_idx],
                                damage_bins['bin_to'][vuln_bin_idx],
                                damage_bins['interpolation'][vuln_bin_idx],
                                vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
                                vuln_cdf[vuln_bin_idx],
                                vuln_rval,
                                computation_tiv,
                            )

                            losses[sample_idx, item_j] = gul * (gul >= loss_threshold)

        # write the losses to the output memoryview
        cursor = write_losses(event_id,
                              sample_size,
                              loss_threshold,
                              losses[:, :Nitems],
                              items_event_data[coverage['start_items']: coverage['start_items'] + Nitems]['item_id'],
                              alloc_rule,
                              tiv,
                              byte_mv,
                              cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    return (cursor,
            last_processed_coverage_ids_idx,
            next_cached_vuln_cdf)


@njit(cache=True, fastmath=True)
def get_vuln_cdf(vuln_i,
                 haz_int_bin_id,
                 cached_vuln_cdf_lookup,
                 cached_vuln_cdf_lookup_keys,
                 vuln_array,
                 vuln_cdf_empty,
                 Ndamage_bins_max,
                 cached_vuln_cdfs,
                 next_cached_vuln_cdf):
    """Compute the cdf of a vulnerability function and store it in cache or, if it is already cached, retrieve it.

    Args:
        vuln_i (int): index of the vuln_array matrix where the vulnerability pdf is stored.
        haz_int_bin_id (int): the selected hazard intensity bin id (starts from 1).
        cached_vuln_cdf_lookup (Dict[VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE]): dict to store
          the map between vuln_id and intensity bin id and the location of the cdf in the cache.
        cached_vuln_cdf_lookup_keys (List[VULN_LOOKUP_VALUE_TYPE]): list of lookup keys.
        vuln_array (np.array[float]): damage pdf for different vulnerability functions, as a function of hazard intensity.
        vuln_cdf_empty (numpy.array[oasis_float]): array (to be re-used) to store vulnerability cdf.
        Ndamage_bins_max (int): maximum number of damage bins.
        cached_vuln_cdfs (np.array[oasis_float]): vulnerability cdf cache.
        next_cached_vuln_cdf (int): index of the next free slot in the vuln cdf cache.

    Returns:
        vuln_cdf (np.array[oasis_float]): the desired vulnerability cdf.
        Ndamage_bins (int): number of bins of vuln_cdf.
        next_cached_vuln_cdf (int): updated index of the next free slot in the vuln cdf cache.
    """
    lookup_key = tuple((vuln_i, haz_int_bin_id))
    if lookup_key in cached_vuln_cdf_lookup:
        # cdf is cached
        start, Ndamage_bins = cached_vuln_cdf_lookup[lookup_key]
        vuln_cdf = cached_vuln_cdfs[start, :Ndamage_bins]

    else:
        # cdf has to be computed
        vuln_cdf = vuln_cdf_empty
        damage_bin_i = 0
        cumsum = 0
        while damage_bin_i < Ndamage_bins_max:
            cumsum += vuln_array[vuln_i, damage_bin_i, haz_int_bin_id - 1]
            vuln_cdf[damage_bin_i] = cumsum
            damage_bin_i += 1

            if cumsum > 0.999999940:
                break

        Ndamage_bins = damage_bin_i

        if cached_vuln_cdf_lookup_keys[next_cached_vuln_cdf] in cached_vuln_cdf_lookup:
            # overwrite cache
            cached_vuln_cdf_lookup.pop(cached_vuln_cdf_lookup_keys[next_cached_vuln_cdf])

        # cache the cdf
        cached_vuln_cdfs[next_cached_vuln_cdf, :Ndamage_bins] = vuln_cdf[:Ndamage_bins]
        cached_vuln_cdf_lookup[lookup_key] = tuple((nb_int32(next_cached_vuln_cdf), nb_int32(Ndamage_bins)))
        cached_vuln_cdf_lookup_keys[next_cached_vuln_cdf] = lookup_key
        next_cached_vuln_cdf += 1
        next_cached_vuln_cdf %= cached_vuln_cdfs.shape[0]

    return (vuln_cdf[:Ndamage_bins],
            Ndamage_bins,
            next_cached_vuln_cdf)


@njit(cache=True, fastmath=True)
def process_areaperils_in_footprint(event_footprint,
                                    vuln_array,
                                    areaperil_to_vulns_idx_dict,
                                    areaperil_to_vulns_idx_array,
                                    areaperil_to_vulns):
    """
    Process all the areaperils in the footprint, filtering and retaining only those who have associated vulnerability functions,
    computing the hazard intensity cdf for each of those areaperil_id.

    Args:
        event_footprint (np.array[Event or EventCSV]): footprint, made of one or more event entries.
        vuln_array (np.array[float]): damage pdf for different vulnerability functions, as a function of hazard intensity.
        areaperil_to_vulns_idx_dict (dict[int, int]): areaperil to vulnerability index dictionary.
        areaperil_to_vulns_idx_array (List[IndexType]]): areaperil ID to vulnerability index array.
        areaperil_to_vulns (np.array ): vuln indexes for each area perils

    Returns:
        areaperil_ids (List[int]): list of all areaperil_ids present in the footprint.
        Nhaz_cdf_this_event (int): number of hazard cdf stored for this event. If zero, it means no items have losses in such event.
        areaperil_to_haz_cdf (dict[int, int]): map between the areaperil_id and the hazard cdf index.
        haz_cdf (np.array[oasis_float]): hazard intensity cdf.
        haz_cdf_ptr (np.array[int]): array with the indices where each cdf record starts in `haz_cdf`.
        eff_vuln_cdf (np.array[oasis_float]): effective damageability cdf.
        areaperil_to_eff_vuln_cdf (dict[ITEM_MAP_KEY_TYPE_internal, int]): map between `(areaperil_id, vuln_idx)` and the location
          where the effective damageability function is stored in `eff_vuln_cdf`.
    """
    # init data structures
    haz_prob_start_in_footprint = List.empty_list(nb_int64)
    areaperil_ids = List.empty_list(nb_areaperil_int)

    footprint_i = 0
    last_areaperil_id = nb_areaperil_int(0)
    last_areaperil_id_start = nb_int64(0)
    haz_cdf_i = 0
    areaperil_to_haz_cdf = Dict.empty(nb_areaperil_int, nb_int32)

    Nevent_footprint_entries = len(event_footprint)
    haz_pdf = np.empty(Nevent_footprint_entries, dtype=oasis_float)  # max size
    haz_cdf = np.empty(Nevent_footprint_entries, dtype=haz_cdf_type)  # max size
    Nvulns, Ndamage_bins_max, _ = vuln_array.shape

    eff_vuln_cdf = np.zeros(Nvulns * Ndamage_bins_max, dtype=oasis_float)  # initial size, it is a dynamic array
    cdf_start = 0
    cdf_end = 0
    areaperil_id = 0
    haz_cdf_ptr = List([0])
    eff_vuln_cdf_start = nb_int32(0)
    areaperil_to_eff_vuln_cdf = Dict.empty(AREAPERIL_TO_EFF_VULN_KEY_TYPE, AREAPERIL_TO_EFF_VULN_VALUE_TYPE)

    while footprint_i < Nevent_footprint_entries:

        areaperil_id = event_footprint[footprint_i]['areaperil_id']

        if areaperil_id != last_areaperil_id:
            # one areaperil_id is completed

            if last_areaperil_id > 0:
                if last_areaperil_id in areaperil_to_vulns_idx_dict:
                    # if items with this areaperil_id exist, process and store this areaperil_id
                    areaperil_ids.append(last_areaperil_id)
                    haz_prob_start_in_footprint.append(last_areaperil_id_start)
                    areaperil_to_haz_cdf[last_areaperil_id] = nb_int32(haz_cdf_i)
                    haz_cdf_i += 1

                    # read the hazard intensity pdf and compute the cdf
                    Nhaz_bins_to_read = footprint_i - last_areaperil_id_start
                    cdf_end = cdf_start + Nhaz_bins_to_read
                    cumsum = 0
                    for haz_bin_i in range(Nhaz_bins_to_read):
                        haz_pdf[cdf_start + haz_bin_i] = event_footprint['probability'][last_areaperil_id_start + haz_bin_i]
                        cumsum += haz_pdf[cdf_start + haz_bin_i]
                        haz_cdf[cdf_start + haz_bin_i]['probability'] = cumsum
                        haz_cdf[cdf_start + haz_bin_i]['intensity_bin_id'] = event_footprint['intensity_bin_id'][last_areaperil_id_start + haz_bin_i]

                    # compute effective damageability cdf (internally called `eff_vuln`)
                    # for each vulnerability function associated to this areaperil_id, compute the product between
                    # the hazard intensity probability and the damage probability.
                    areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[last_areaperil_id]]
                    for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
                        while eff_vuln_cdf.shape[0] < eff_vuln_cdf_start + Ndamage_bins_max:
                            # if eff_vuln_cdf.shape needs to be larger to store all the effective cdfs, double it in size
                            temp_eff_vuln_cdf = np.empty(eff_vuln_cdf.shape[0] * 2, dtype=eff_vuln_cdf.dtype)
                            temp_eff_vuln_cdf[:eff_vuln_cdf_start] = eff_vuln_cdf[:eff_vuln_cdf_start]
                            eff_vuln_cdf = temp_eff_vuln_cdf

                        eff_vuln_cdf_cumsum = 0.
                        damage_bin_i = 0
                        while damage_bin_i < Ndamage_bins_max:
                            for haz_bin_i in range(Nhaz_bins_to_read):
                                eff_vuln_cdf_cumsum += vuln_array[
                                    areaperil_to_vulns[vuln_idx], damage_bin_i, haz_cdf[cdf_start + haz_bin_i]['intensity_bin_id'] - 1] * haz_pdf[cdf_start + haz_bin_i]

                            eff_vuln_cdf[eff_vuln_cdf_start + damage_bin_i] = eff_vuln_cdf_cumsum
                            damage_bin_i += 1
                            if eff_vuln_cdf_cumsum > 0.999999940:
                                break

                        Ndamage_bins = damage_bin_i

                        areaperil_to_eff_vuln_cdf[(last_areaperil_id, areaperil_to_vulns[vuln_idx])] = (
                            nb_int32(eff_vuln_cdf_start), nb_int32(Ndamage_bins))
                        eff_vuln_cdf_start += Ndamage_bins

                    haz_cdf_ptr.append(cdf_end)
                    cdf_start = cdf_end

            last_areaperil_id = areaperil_id
            last_areaperil_id_start = footprint_i

        footprint_i += 1

    # here we process the last row of the footprint:
    # this is either the last entry of a cdf started a few lines above or a 1-line cdf.
    # In either case we do not need to check if areaperil_id != last_areaperil_id
    # because we need to store the row anyway.
    if areaperil_id in areaperil_to_vulns_idx_dict:
        areaperil_ids.append(areaperil_id)
        haz_prob_start_in_footprint.append(last_areaperil_id_start)
        areaperil_to_haz_cdf[areaperil_id] = nb_int32(haz_cdf_i)
        haz_cdf_i += 1  # needed to correctly count the number of haz_cdf imported

        Nhaz_bins_to_read = footprint_i - last_areaperil_id_start
        cdf_end = cdf_start + Nhaz_bins_to_read
        cumsum = 0
        for haz_bin_i in range(Nhaz_bins_to_read):
            haz_pdf[cdf_start + haz_bin_i] = event_footprint['probability'][last_areaperil_id_start + haz_bin_i]
            cumsum += haz_pdf[cdf_start + haz_bin_i]
            haz_cdf[cdf_start + haz_bin_i]['probability'] = cumsum
            haz_cdf[cdf_start + haz_bin_i]['intensity_bin_id'] = event_footprint['intensity_bin_id'][last_areaperil_id_start + haz_bin_i]

        # compute effective damageability cdf (internally called `eff_vuln`)
        # for each vulnerability function associated to this areaperil_id, compute the product between
        # the hazard intensity probability and the damage probability.
        areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[last_areaperil_id]]
        for vuln_idx in range(areaperil_to_vulns_idx['start'], areaperil_to_vulns_idx['end']):
            while eff_vuln_cdf.shape[0] < eff_vuln_cdf_start + Ndamage_bins_max:
                # if eff_vuln_cdf.shape needs to be larger to store all the effective cdfs, double it in size
                temp_eff_vuln_cdf = np.empty(eff_vuln_cdf.shape[0] * 2, dtype=eff_vuln_cdf.dtype)
                temp_eff_vuln_cdf[:eff_vuln_cdf_start] = eff_vuln_cdf[:eff_vuln_cdf_start]
                eff_vuln_cdf = temp_eff_vuln_cdf

            eff_vuln_cdf_cumsum = 0.
            damage_bin_i = 0
            while damage_bin_i < Ndamage_bins_max:
                for haz_bin_i in range(Nhaz_bins_to_read):
                    eff_vuln_cdf_cumsum += vuln_array[
                        areaperil_to_vulns[vuln_idx], damage_bin_i, haz_cdf[cdf_start + haz_bin_i]['intensity_bin_id'] - 1] * haz_pdf[cdf_start + haz_bin_i]

                eff_vuln_cdf[eff_vuln_cdf_start + damage_bin_i] = eff_vuln_cdf_cumsum
                damage_bin_i += 1
                if eff_vuln_cdf_cumsum > 0.999999940:
                    break

            Ndamage_bins = damage_bin_i
            areaperil_to_eff_vuln_cdf[(areaperil_id, areaperil_to_vulns[vuln_idx])] = (nb_int32(eff_vuln_cdf_start), nb_int32(Ndamage_bins))
            eff_vuln_cdf_start += Ndamage_bins

        haz_cdf_ptr.append(cdf_end)

    Nhaz_cdf_this_event = haz_cdf_i

    return (areaperil_ids,
            Nhaz_cdf_this_event,
            areaperil_to_haz_cdf,
            haz_cdf[:cdf_end],
            haz_cdf_ptr,
            eff_vuln_cdf,
            areaperil_to_eff_vuln_cdf)


@njit(cache=True, fastmath=True)
def reconstruct_coverages(event_id,
                          areaperil_ids,
                          areaperil_ids_map,
                          areaperil_to_haz_cdf,
                          item_map,
                          items,
                          coverages,
                          compute,
                          haz_seeds,
                          vuln_seeds):
    """Register each item to its coverage, with the location of the corresponding hazard intensity cdf
    in the footprint, compute the random seeds for the hazard intensity and vulnerability samples.

    Args:
        event_id (int32): event id.
        areaperil_ids (List[int]): list of all areaperil_ids present in the footprint.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
          areaperil_id and all the vulnerability ids associated with it.
        areaperil_to_haz_cdf (dict[int, int]): map between the areaperil_id and the hazard cdf index.
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
        items (np.ndarray): items table merged with correlation parameters.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        compute (numpy.array[int]): list of coverage ids to be computed.
        haz_seeds (numpy.array[int]): the random seeds to draw the hazard intensity samples.
        vuln_seeds (numpy.array[int]): the random seeds to draw the damage samples.

    Returns:
        compute_i (int): index of the last coverage id stored in `compute`.
        items_data (numpy.array[items_MC_data_type]): item-related data.
        rng_index (int): number of unique random seeds for damage sampling computed so far.
        hazard_rng_index (int): number of unique random seeds for hazard intensity sampling computed so far.
    """
    # init data structures
    group_id_rng_index = Dict.empty(nb_int32, nb_int64)
    hazard_group_id_hazard_rng_index = Dict.empty(nb_int32, nb_int64)
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
    for areaperil_id in areaperil_ids:

        for vuln_id in areaperil_ids_map[areaperil_id]:
            # register the items to their coverage
            item_key = tuple((areaperil_id, vuln_id))

            for item_idx in item_map[item_key]:
                # if this group_id was not seen yet, process it.
                # it assumes that hash only depends on event_id and group_id
                # and that only 1 event_id is processed at a time.
                group_id = items[item_idx]['group_id']
                if group_id not in group_id_rng_index:
                    group_id_rng_index[group_id] = rng_index
                    vuln_seeds[rng_index] = generate_hash(group_id, event_id)
                    this_rng_index = rng_index
                    rng_index += 1

                else:
                    this_rng_index = group_id_rng_index[group_id]

                hazard_group_id = items[item_idx]['hazard_group_id']
                if hazard_group_id not in hazard_group_id_hazard_rng_index:
                    hazard_group_id_hazard_rng_index[hazard_group_id] = hazard_rng_index
                    haz_seeds[hazard_rng_index] = generate_hash_hazard(hazard_group_id, event_id)
                    this_hazard_rng_index = hazard_rng_index
                    hazard_rng_index += 1

                else:
                    this_hazard_rng_index = hazard_group_id_hazard_rng_index[hazard_group_id]

                coverage_id = items[item_idx]['coverage_id']
                coverage = coverages[coverage_id]
                if coverage['cur_items'] == 0:
                    # no items were collected for this coverage yet: set up the structure
                    compute[compute_i], compute_i = coverage_id, compute_i + 1

                    while items_event_data.shape[0] < items_data_i + coverage['max_items']:
                        # if items_data needs to be larger to store all the items, double it in size
                        temp_items_data = np.empty(items_event_data.shape[0] * 2, dtype=items_event_data.dtype)
                        temp_items_data[:items_data_i] = items_event_data[:items_data_i]
                        items_event_data = temp_items_data

                    coverage['start_items'], items_data_i = items_data_i, items_data_i + coverage['max_items']

                # append the data of this item
                item_i = coverage['start_items'] + coverage['cur_items']
                items_event_data[item_i]['item_idx'] = item_idx
                items_event_data[item_i]['item_id'] = items[item_idx]['item_id']
                items_event_data[item_i]['hazcdf_i'] = areaperil_to_haz_cdf[areaperil_id]
                items_event_data[item_i]['rng_index'] = this_rng_index
                items_event_data[item_i]['hazard_rng_index'] = this_hazard_rng_index

                coverage['cur_items'] += 1

    return (compute_i,
            items_event_data,
            rng_index,
            hazard_rng_index)


if __name__ == '__main__':

    # test_dir = Path(__file__).parent.parent.parent.parent.joinpath("tests") \
    #     .joinpath("assets").joinpath("test_model_2")

    test_dir = Path("runs/losses-20240108105851")

    file_out = test_dir.joinpath('gulpy_mc.bin')
    run(
        run_dir=test_dir,
        ignore_file_type=set(),
        file_in=test_dir.joinpath("input").joinpath('events.bin'),
        file_out=file_out,
        sample_size=10,
        loss_threshold=0.,
        alloc_rule=1,
        debug=0,
        random_generator=1,
        ignore_correlation=False,
        ignore_haz_correlation=False,
        effective_damageability=False,
    )

    # remove temporary file
    if file_out.exists():
        file_out.unlink()
