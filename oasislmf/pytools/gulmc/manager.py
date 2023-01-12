import atexit
import logging
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from select import select

import numpy as np
import pandas as pd
from numba import njit
from numba.typed import Dict, List
from numba.types import Tuple as nb_Tuple
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasislmf.pytools.common import PIPE_CAPACITY
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.data_layer.oasis_files.correlations import \
    CorrelationsData
from oasislmf.pytools.getmodel.common import (Correlation, Keys,
                                              nb_areaperil_int, oasis_float)
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.manager import Item, get_damage_bins, get_vulns
from oasislmf.pytools.gul.common import (CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX,
                                         MEAN_IDX, NP_BASE_ARRAY_SIZE, NUM_IDX,
                                         STD_DEV_IDX, TIV_IDX,
                                         ITEM_MAP_KEY_TYPE_internal,
                                         ITEM_MAP_VALUE_TYPE_internal,
                                         coverage_type, gul_header,
                                         gulSampleslevelHeader_size,
                                         gulSampleslevelRec_size, haz_cdf_type,
                                         items_MC_data_type)
from oasislmf.pytools.gul.core import compute_mean_loss, get_gul
from oasislmf.pytools.gul.manager import get_coverages, write_losses
from oasislmf.pytools.gul.random import (compute_norm_cdf_lookup,
                                         compute_norm_inv_cdf_lookup,
                                         generate_correlated_hash_vector,
                                         generate_hash, generate_hash_haz,
                                         get_corr_rval, get_random_generator)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.gulmc.aggregate import (
    map_agg_vuln_ids_to_agg_vuln_idxs,
    map_areaperil_vuln_id_to_weight_to_areaperil_vuln_idx_to_weight,
    process_aggregate_vulnerability, process_vulnerability_weights,
    read_aggregate_vulnerability, read_vulnerability_weights)
from oasislmf.pytools.gulmc.items import (generate_item_map, process_items,
                                          read_items)

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
        effective_damageability=False,
        max_cached_vuln_cdf_size_MB=200,
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
    Raises:
        ValueError: if alloc_rule is not 0, 1, 2, or 3.
        ValueError: if alloc_rule is 1, 2, or 3 when debug is 1 or 2.

    Returns:
        int: 0 if no errors occurred.
    """
    logger.info("starting gulmc")

    static_path = os.path.join(run_dir, 'static')
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    damage_bins = get_damage_bins(static_path, ignore_file_type)

    # read coverages from file
    coverages_tiv = get_coverages(input_path, ignore_file_type)

    # init the structure for computation
    # coverages are numbered from 1, therefore we skip element 0 in `coverages`
    coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv
    del coverages_tiv

    items = read_items(input_path, ignore_file_type)

    # in-place sort items in order to store them in item_map in the desired order
    # currently numba only supports a simple call to np.sort() with no `order` keyword,
    # so we do the sort here.
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map, areaperil_ids_map = generate_item_map(items, coverages)

    # init array to store the coverages to be computed
    # coverages are numebered from 1, therefore skip element 0.
    compute = np.zeros(coverages.shape[0] + 1, items.dtype['coverage_id'])

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

        # prepare for stochastic disaggregation
        # read aggregate vulnerability definitions and vulnerability weights
        aggregate_vulnerability = read_aggregate_vulnerability(static_path, ignore_file_type)
        aggregate_weights = read_vulnerability_weights(static_path, ignore_file_type)

        # process aggregate vulnerability and vulnerability weights
        agg_vuln_to_vuln_id = process_aggregate_vulnerability(aggregate_vulnerability)

        if aggregate_vulnerability is not None and aggregate_weights is None:
            raise FileNotFoundError(f'Vulnerability weights file not found at {static_path}')

        areaperil_vuln_id_to_weight = process_vulnerability_weights(aggregate_weights, agg_vuln_to_vuln_id)

        logger.debug('init items')
        vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_dict, used_agg_vuln_ids = process_items(
            items, valid_area_peril_id, agg_vuln_to_vuln_id)

        logger.debug('reconstruct aggregate vulnerability definitions and weights')

        # map each vulnerability_id composing aggregate vulnerabilities to the indices where they are stored in vuln_array
        # here we filter out aggregate vulnerability that are not used in this portfolio, therefore
        # agg_vuln_to_vuln_idxs can contain less aggregate vulnerability ids compared to agg_vuln_to_vuln_id
        agg_vuln_to_vuln_idxs = map_agg_vuln_ids_to_agg_vuln_idxs(used_agg_vuln_ids, agg_vuln_to_vuln_id, vuln_dict)

        # remap (areaperil, vuln_id) to weights to (areaperil, vuln_idx) to weights
        areaperil_vuln_idx_to_weight = map_areaperil_vuln_id_to_weight_to_areaperil_vuln_idx_to_weight(
            areaperil_dict, areaperil_vuln_id_to_weight, vuln_dict)

        logger.debug('init footprint')
        footprint_obj = stack.enter_context(Footprint.load(static_path, ignore_file_type))

        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
        else:
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        logger.debug('init vulnerability')

        vuln_array, _, _ = get_vulns(static_path, vuln_dict, num_intensity_bins, ignore_file_type)
        Nvulnerability, Ndamage_bins_max, Nintensity_bins = vuln_array.shape

        # set up streams
        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        select_stream_list = [stream_out]

        # prepare output buffer, write stream header
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2, 3]:
            raise ValueError(f"Expect alloc_rule to be 0, 1, 2, or 3, got {alloc_rule}")

        if debug > 0 and alloc_rule != 0:
            raise ValueError(f"Expect alloc_rule to be 0 if debug is 1 or 2, got {alloc_rule}")

        cursor = 0
        cursor_bytes = 0

        # create the array to store the seeds
        haz_seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])
        vuln_seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])

        do_correlation = False
        if ignore_correlation:
            logger.info("Correlated random number generation: switched OFF because --ignore-correlation is True.")

        else:
            file_path = os.path.join(input_path, 'correlations.bin')
            data = CorrelationsData.from_bin(file_path=file_path).data
            Nperil_correlation_groups = len(data)
            logger.info(f"Detected {Nperil_correlation_groups} peril correlation groups.")

            if Nperil_correlation_groups > 0 and any(data['correlation_value'] > 0):
                do_correlation = True
            else:
                logger.info("Correlated random number generation: switched OFF because 0 peril correlation groups were detected or "
                            "the correlation value is zero for all peril correlation groups.")

        if do_correlation:
            logger.info("Correlated random number generation: switched ON.")

            corr_data_by_item_id = np.ndarray(Nperil_correlation_groups + 1, dtype=Correlation)
            corr_data_by_item_id[0] = (0, 0.)
            corr_data_by_item_id[1:]['peril_correlation_group'] = np.array(data['peril_correlation_group'])
            corr_data_by_item_id[1:]['correlation_value'] = np.array(data['correlation_value'])

            logger.info(
                f"Correlation values for {Nperil_correlation_groups} peril correlation groups have been imported."
            )

            unique_peril_correlation_groups = np.unique(corr_data_by_item_id[1:]['peril_correlation_group'])

            # pre-compute lookup tables for the Gaussian cdf and inverse cdf
            # Notes:
            #  - the size `arr_N` and `arr_N_cdf` can be increased to achieve better resolution in the Gaussian cdf and inv cdf.
            #  - the function `get_corr_rval` to compute the correlated numbers is not affected by arr_N and arr_N_cdf
            arr_min, arr_max, arr_N = 1e-16, 1 - 1e-16, 1000000
            arr_min_cdf, arr_max_cdf, arr_N_cdf = -20., 20., 1000000
            norm_inv_cdf = compute_norm_inv_cdf_lookup(arr_min, arr_max, arr_N)
            norm_cdf = compute_norm_cdf_lookup(arr_min_cdf, arr_max_cdf, arr_N_cdf)

            # buffer to be re-used to store all the correlated random values
            z_unif = np.zeros(sample_size, dtype='float64')

        else:
            # create dummy data structures with proper dtypes to allow correct numba compilation
            corr_data_by_item_id = np.ndarray(1, dtype=Correlation)
            arr_min, arr_max, arr_N = 0, 0, 0
            arr_min_cdf, arr_max_cdf, arr_N_cdf = 0, 0, 0
            norm_inv_cdf, norm_cdf = np.zeros(1, dtype='float64'), np.zeros(1, dtype='float64')
            z_unif = np.zeros(1, dtype='float64')

        if effective_damageability is True:
            logger.info("effective_damageability is True: gulmc will draw the damage samples from the effective damageability distribution.")
        else:
            logger.info("effective_damageability is False: gulmc will perform the full Monte Carlo sampling: "
                        "sample the hazard intensity first, then sample the damage from the corresponding vulnerability function.")

        # create buffers to be reused when computing losses
        losses = np.zeros((sample_size + NUM_IDX + 1, np.max(coverages[1:]['max_items'])), dtype=oasis_float)
        vuln_cdf_empty = np.zeros(Ndamage_bins_max, dtype=oasis_float)
        weighted_vuln_cdf_empty = np.zeros(Ndamage_bins_max, dtype=oasis_float)

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size + 2 * gulSampleslevelHeader_size

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

        while True:
            if not streams_in.readinto(event_id_mv):
                break

            # get the next event_id from the input stream
            event_id = event_ids[0]

            event_footprint = event_footprint_obj.get_event(event_id)

            if event_footprint is not None:

                areaperil_ids, Nhaz_cdf_this_event, areaperil_to_haz_cdf, haz_cdf, haz_cdf_ptr, eff_vuln_cdf, areaperil_to_eff_vuln_cdf = process_areaperils_in_footprint(
                    event_footprint, vuln_array, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array)

                if Nhaz_cdf_this_event == 0:
                    # no items to be computed for this event
                    continue

                compute_i, items_data, rng_index = reconstruct_coverages(event_id, areaperil_ids, areaperil_ids_map, areaperil_to_haz_cdf, item_map,
                                                                         coverages, compute, haz_seeds, vuln_seeds)

                # generation of "base" random values for hazard intensity and vulnerability sampling
                haz_rndms_base = generate_rndm(haz_seeds[:rng_index], sample_size)
                vuln_rndms_base = generate_rndm(vuln_seeds[:rng_index], sample_size)

                # generate the correlated samples for the whole event, for all peril correlation groups
                if do_correlation:
                    corr_seeds = generate_correlated_hash_vector(unique_peril_correlation_groups, event_id)
                    eps_ij = generate_rndm(corr_seeds, sample_size, skip_seeds=1)

                else:
                    # create dummy data structures with proper dtypes to allow correct numba compilation
                    eps_ij = np.zeros((1, 1), dtype='float64')

                last_processed_coverage_ids_idx = 0

                # adjust buff size so that the buffer fits the longest coverage
                buff_size = PIPE_CAPACITY
                max_bytes_per_coverage = np.max(coverages['cur_items']) * max_bytes_per_item
                while buff_size < max_bytes_per_coverage:
                    buff_size *= 2

                # define the raw memory view and its int32 view
                mv_write = memoryview(bytearray(buff_size))
                int32_mv = np.ndarray(buff_size // 4, buffer=mv_write, dtype='i4')

                # create vulnerability cdf cache
                cached_vuln_cdfs = np.zeros((Nvulns_cached, Ndamage_bins_max), dtype=oasis_float)
                cached_vuln_cdf_lookup, lookup_keys = gen_empty_vuln_cdf_lookup(Nvulns_cached)
                next_cached_vuln_cdf = 0

                while last_processed_coverage_ids_idx < compute_i:

                    cursor, cursor_bytes, last_processed_coverage_ids_idx, next_cached_vuln_cdf = compute_event_losses(
                        event_id, coverages, compute[:compute_i], items_data,
                        last_processed_coverage_ids_idx, sample_size, haz_cdf, haz_cdf_ptr,
                        areaperil_to_eff_vuln_cdf,
                        eff_vuln_cdf, vuln_array, damage_bins, Ndamage_bins_max,
                        cached_vuln_cdf_lookup, lookup_keys, next_cached_vuln_cdf,
                        cached_vuln_cdfs,
                        agg_vuln_to_vuln_id, agg_vuln_to_vuln_idxs, vuln_dict, areaperil_vuln_idx_to_weight,
                        loss_threshold, losses, vuln_cdf_empty, weighted_vuln_cdf_empty, alloc_rule, do_correlation, haz_rndms_base, vuln_rndms_base,
                        eps_ij, corr_data_by_item_id, arr_min, arr_max, arr_N, norm_inv_cdf, arr_min_cdf, arr_max_cdf, arr_N_cdf, norm_cdf,
                        z_unif, effective_damageability, debug, max_bytes_per_item, buff_size, int32_mv, cursor
                    )

                    # write the losses to the output stream
                    write_start = 0
                    while write_start < cursor_bytes:
                        select([], select_stream_list, select_stream_list)
                        write_start += stream_out.write(mv_write[write_start:cursor_bytes])

                    cursor = 0

                logger.info(f"event {event_id} DONE")

    return 0


@njit(cache=True, fastmath=True)
def compute_event_losses(event_id,
                         coverages,
                         coverage_ids,
                         items_data,
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
                         haz_rndms,
                         vuln_rndms_base,
                         eps_ij,
                         corr_data_by_item_id,
                         arr_min,
                         arr_max,
                         arr_N,
                         norm_inv_cdf,
                         arr_min_cdf,
                         arr_max_cdf,
                         arr_N_cdf,
                         norm_cdf,
                         z_unif,
                         effective_damageability,
                         debug,
                         max_bytes_per_item,
                         buff_size,
                         int32_mv,
                         cursor):
    """Compute losses for an event.

    Args:
        event_id (int32): event id.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        coverage_ids (numpy.array[int]): array of unique coverage ids used in this event.
        items_data (numpy.array[items_data_type]): items-related data.
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
        vuln_cdf (np.array[oasis_float]): array (to be re-used) to store the damage cdf for each item.
        alloc_rule (int): back-allocation rule.
        do_correlation (bool): if True, compute correlated random samples.
        haz_rndms (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed for the hazard intensity sampling.
        vuln_rndms_base (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed for the damage sampling.
        eps_ij (np.array[float]): correlated random values of shape `(number of seeds, sample_size)`.
        corr_data_by_item_id (np.array[Correlation]): correlation definitions for each item_id.
        arr_min (float): min value of the lookup table for the inverse Gaussian.
        arr_max (float): max value of the lookup table for the inverse Gaussian.
        arr_N (int): array size of the lookup table for the inverse Gaussian.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf.
        arr_min_cdf (float): min value of the grid on which the Gaussian cdf is computed.
        arr_max_cdf (float): max value of the grid on which the Gaussian cdf is computed.
        arr_N_cdf (int): array size of the Gaussian cdf.
        norm_cdf (np.array[float]): Gaussian cdf.
        z_unif (np.array[float]): buffer to be re-used to store all the correlated random values.
        effective_damageability (bool): if True, it uses effective damageability to draw damage samples instead of
          using the full monte carlo approach (i.e., to draw hazard intensity first, then damage).
        debug (int): for each random sample, print to the output stream the random loss (if 0),
          the random value used to draw the hazard intensity sample (if 1), the random value used to draw the damage sample (if 2).
        max_bytes_per_item (int): maximum bytes to be written in the output stream for an item.
        buff_size (int): size in bytes of the output buffer.
        int32_mv (numpy.ndarray): int32 view of the memoryview where the output is buffered.
        cursor (int): index of int32_mv where to start writing.

    Returns:
        cursor (int): index of int32_mv where to start writing.
        cursor_bytes (int): updated value of cursor_bytes, the cursor location in bytes.
        last_processed_coverage_ids_idx (int): index of the last coverage_id stored in `coverage_ids` that was fully processed
          and printed to the output stream.
    """
    # loop through all the coverages that remain to be computed
    for coverage_i in range(last_processed_coverage_ids_idx, coverage_ids.shape[0]):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']
        Nitems = coverage['cur_items']
        exposureValue = tiv / Nitems

        # estimate max number of bytes needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        cursor_bytes = cursor * int32_mv.itemsize
        est_cursor_bytes = Nitems * max_bytes_per_item

        # return before processing this coverage if the number of free bytes left in the buffer
        # is not sufficient to write out the full coverage
        if cursor_bytes + est_cursor_bytes > buff_size:
            return cursor, cursor_bytes, last_processed_coverage_ids_idx, next_cached_vuln_cdf

        items = items_data[coverage['start_items']: coverage['start_items'] + Nitems]

        # compute losses for each item
        for item_i in range(Nitems):
            item = items[item_i]
            rng_index = item['rng_index']
            areaperil_id = item['areaperil_id']
            vulnerability_id = item['vulnerability_id']

            if not effective_damageability:
                # get the right hazard cdf from the array containing all hazard cdfs
                hazcdf_i = item['hazcdf_i']
                haz_cdf_record = haz_cdf[haz_cdf_ptr[hazcdf_i]:haz_cdf_ptr[hazcdf_i + 1]]
                haz_cdf_prob = haz_cdf_record['probability']
                haz_cdf_bin_id = haz_cdf_record['intensity_bin_id']
                Nhaz_bins = haz_cdf_ptr[hazcdf_i + 1] - haz_cdf_ptr[hazcdf_i]

            # if aggregate: agg_eff_vuln_cdf needs to be computed
            if vulnerability_id in agg_vuln_to_vuln_id:
                # aggregate case
                weighted_vuln_cdf = weighted_vuln_cdf_empty
                tot_weights = 0.
                agg_vulns_idx = agg_vuln_to_vuln_idxs[vulnerability_id]

                # here we use loop-unrolling for a more performant code.
                # we explicitly run the first cycle for damage_bin_i=0 to cache (eff_vuln_cdf_i, eff_vuln_cdf_Ndamage_bins, weight)
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
                    print("item_id=", item['item_id'])
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

            # compute mean loss values
            gul_mean, std_dev, chance_of_loss, max_loss = compute_mean_loss(
                tiv,
                eff_damag_cdf,
                damage_bins['interpolation'],
                eff_damag_cdf_Ndamage_bins,
                damage_bins[eff_damag_cdf_Ndamage_bins - 1]['bin_to'],
            )

            losses[MAX_LOSS_IDX, item_i] = max_loss
            losses[CHANCE_OF_LOSS_IDX, item_i] = chance_of_loss
            losses[TIV_IDX, item_i] = exposureValue
            losses[STD_DEV_IDX, item_i] = std_dev
            losses[MEAN_IDX, item_i] = gul_mean

            # compute random losses
            if sample_size > 0:
                if do_correlation:
                    # use correlation definitions to draw correlated random values
                    item_corr_data = corr_data_by_item_id[item['item_id']]
                    rho = item_corr_data['correlation_value']

                    if rho > 0:
                        peril_correlation_group = item_corr_data['peril_correlation_group']

                        get_corr_rval(
                            eps_ij[peril_correlation_group], vuln_rndms_base[rng_index],
                            rho, arr_min, arr_max, arr_N, norm_inv_cdf,
                            arr_min_cdf, arr_max_cdf, arr_N_cdf, norm_cdf, sample_size, z_unif
                        )
                        vuln_rndms = z_unif

                    else:
                        vuln_rndms = vuln_rndms_base[rng_index]

                else:
                    # do not use correlation
                    vuln_rndms = vuln_rndms_base[rng_index]

                for sample_idx in range(1, sample_size + 1):
                    if effective_damageability:
                        vuln_cdf = eff_damag_cdf
                        Ndamage_bins = eff_damag_cdf_Ndamage_bins

                    else:
                        # use the full monte carlo approach: draw samples from the hazard intensity distribution first

                        # 1) get the intensity bin
                        if Nhaz_bins == 1:
                            # if hazard intensity has no uncertainty, there is no need to sample
                            haz_bin_idx = nb_int32(0)

                        else:
                            # if hazard intensity has a probability distribution, sample it

                            # cap `haz_rval` to the maximum `haz_cdf_prob` value (which should be 1.)
                            haz_rval = haz_rndms[rng_index][sample_idx - 1]

                            if debug == 1:
                                # store the random value used for the hazard intensity sampling instead of the loss
                                losses[sample_idx, item_i] = haz_rval
                                continue

                            if haz_rval >= haz_cdf_prob[Nhaz_bins - 1]:
                                haz_bin_idx = nb_int32(Nhaz_bins - 1)
                            else:
                                # find the bin in which the random value `haz_rval` falls into
                                haz_bin_idx = nb_int32(binary_search(haz_rval, haz_cdf_prob, Nhaz_bins))

                        # 2) get the hazard intensity bin id
                        haz_int_bin_id = haz_cdf_bin_id[haz_bin_idx]

                        # 3) get the vulnerability cdf
                        if vulnerability_id in agg_vuln_to_vuln_id:
                            # aggregate case
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
                                print("item_id=", item['item_id'])
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

                        else:
                            # non-aggregate case
                            vuln_i = vuln_dict[vulnerability_id]
                            vuln_cdf, Ndamage_bins, next_cached_vuln_cdf = get_vuln_cdf(
                                vuln_i, haz_bin_idx, haz_int_bin_id, cached_vuln_cdf_lookup, cached_vuln_cdf_lookup_keys, vuln_array, vuln_cdf_empty,
                                Ndamage_bins_max, cached_vuln_cdfs, next_cached_vuln_cdf)

                    # draw samples of damage from the vulnerability function
                    vuln_rval = vuln_rndms[sample_idx - 1]

                    if debug == 2:
                        # store the random value used for the damage sampling instead of the loss
                        losses[sample_idx, item_i] = vuln_rval
                        continue

                    # cap `vuln_rval` to the maximum `vuln_cdf` value (which should be 1.)
                    if vuln_rval >= vuln_cdf[Ndamage_bins - 1]:
                        vuln_rval = vuln_cdf[Ndamage_bins - 1] - 0.00000003
                        vuln_bin_idx = Ndamage_bins - 1
                    else:
                        # find the bin in which the random value `vuln_rval` falls into
                        vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins)

                    # compute ground-up losses
                    gul = get_gul(
                        damage_bins['bin_from'][vuln_bin_idx],
                        damage_bins['bin_to'][vuln_bin_idx],
                        damage_bins['interpolation'][vuln_bin_idx],
                        vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
                        vuln_cdf[vuln_bin_idx],
                        vuln_rval,
                        tiv
                    )

                    if gul >= loss_threshold:
                        losses[sample_idx, item_i] = gul
                    else:
                        losses[sample_idx, item_i] = 0

        # write the losses to the output memoryview
        cursor = write_losses(event_id, sample_size, loss_threshold, losses[:, :items.shape[0]], items['item_id'], alloc_rule, tiv,
                              int32_mv, cursor)

        # register that another `coverage_id` has been processed
        last_processed_coverage_ids_idx += 1

    # update cursor_bytes
    cursor_bytes = cursor * int32_mv.itemsize

    return (cursor,
            cursor_bytes,
            last_processed_coverage_ids_idx,
            next_cached_vuln_cdf)


@njit(cache=True, fastmath=True)
def get_vuln_cdf(vuln_i,
                 haz_bin_idx,
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
        haz_bin_idx (int): index of the selected hazard intensity bin.
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
    lookup_key = tuple((vuln_i, haz_bin_idx))
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

    return (vuln_cdf,
            Ndamage_bins,
            next_cached_vuln_cdf)


@njit(cache=True, fastmath=True)
def process_areaperils_in_footprint(event_footprint,
                                    vuln_array,
                                    areaperil_to_vulns_idx_dict,
                                    areaperil_to_vulns_idx_array):
    """
    Process all the areaperils in the footprint, filtering and retaining only those who have associated vulnerability functions,
    computing the hazard intensity cdf for each of those areaperil_id.

    Args:
        event_footprint (np.array[Event or EventCSV]): footprint, made of one or more event entries.
        vuln_array (np.array[float]): damage pdf for different vulnerability functions, as a function of hazard intensity.
        areaperil_to_vulns_idx_dict (dict[int, int]): areaperil to vulnerability index dictionary.
        areaperil_to_vulns_idx_array (List[IndexType]]): areaperil ID to vulnerability index array.

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

    eff_vuln_cdf = np.zeros((Nvulns * Ndamage_bins_max), dtype=oasis_float)  # max size
    cdf_start = 0
    cdf_end = 0
    haz_cdf_ptr = List([0])
    eff_vuln_cdf_start = nb_int32(0)
    areaperil_to_eff_vuln_cdf = Dict.empty(ITEM_MAP_KEY_TYPE_internal, ITEM_MAP_VALUE_TYPE_internal)

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
                        eff_vuln_cdf_cumsum = 0.
                        damage_bin_i = 0
                        while damage_bin_i < Ndamage_bins_max:
                            for haz_bin_i in range(Nhaz_bins_to_read):
                                eff_vuln_cdf_cumsum += vuln_array[
                                    vuln_idx, damage_bin_i, haz_cdf[cdf_start + haz_bin_i]['intensity_bin_id'] - 1] * haz_pdf[cdf_start + haz_bin_i]

                            eff_vuln_cdf[eff_vuln_cdf_start + damage_bin_i] = eff_vuln_cdf_cumsum
                            damage_bin_i += 1
                            if eff_vuln_cdf_cumsum > 0.999999940:
                                break

                        Ndamage_bins = damage_bin_i

                        areaperil_to_eff_vuln_cdf[(last_areaperil_id, vuln_idx)] = (nb_int32(eff_vuln_cdf_start), nb_int32(Ndamage_bins))
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

            eff_vuln_cdf_cumsum = 0.
            damage_bin_i = 0
            while damage_bin_i < Ndamage_bins_max:
                for haz_bin_i in range(Nhaz_bins_to_read):
                    eff_vuln_cdf_cumsum += vuln_array[
                        vuln_idx, damage_bin_i, haz_cdf[cdf_start + haz_bin_i]['intensity_bin_id'] - 1] * haz_pdf[cdf_start + haz_bin_i]

                eff_vuln_cdf[eff_vuln_cdf_start + damage_bin_i] = eff_vuln_cdf_cumsum
                damage_bin_i += 1
                if eff_vuln_cdf_cumsum > 0.999999940:
                    break

            Ndamage_bins = damage_bin_i
            areaperil_to_eff_vuln_cdf[(areaperil_id, vuln_idx)] = (nb_int32(eff_vuln_cdf_start), nb_int32(Ndamage_bins))
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
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        compute (numpy.array[int]): list of coverage ids to be computed.
        haz_seeds (numpy.array[int]): the random seeds to draw the hazard intensity samples.
        vuln_seeds (numpy.array[int]): the random seeds to draw the damage samples.

    Returns:
        compute_i (int): index of the last coverage id stored in `compute`.
        items_data (numpy.array[items_MC_data_type]): item-related data.
        rng_index (int): number of unique random seeds computed so far.
    """
    # init data structures
    group_id_rng_index = Dict.empty(nb_int32, nb_int64)
    rng_index = 0
    compute_i = 0
    items_data_i = 0
    coverages['cur_items'].fill(0)
    items_data = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_MC_data_type)

    # for each areaperil_id, loop over all vulnerability functions used in that areaperil_id and,
    # for each item:
    #  - compute the seeds for the hazard intensity sampling and for the damage sampling
    #  - store data for later processing (hazard cdf index, etc.)
    for areaperil_id in areaperil_ids:

        for vuln_id in areaperil_ids_map[areaperil_id]:
            # register the items to their coverage
            item_key = tuple((areaperil_id, vuln_id))

            for item in item_map[item_key]:
                item_id, coverage_id, group_id = item

                # if this group_id was not seen yet, process it.
                # it assumes that hash only depends on event_id and group_id
                # and that only 1 event_id is processed at a time.
                if group_id not in group_id_rng_index:
                    group_id_rng_index[group_id] = rng_index
                    haz_seeds[rng_index] = generate_hash_haz(group_id, event_id)
                    vuln_seeds[rng_index] = generate_hash(group_id, event_id)
                    this_rng_index = rng_index
                    rng_index += 1

                else:
                    this_rng_index = group_id_rng_index[group_id]

                coverage = coverages[coverage_id]
                if coverage['cur_items'] == 0:
                    # no items were collected for this coverage yet: set up the structure
                    compute[compute_i], compute_i = coverage_id, compute_i + 1

                    while items_data.shape[0] < items_data_i + coverage['max_items']:
                        # if items_data needs to be larger to store all the items, double it in size
                        temp_items_data = np.empty(items_data.shape[0] * 2, dtype=items_data.dtype)
                        temp_items_data[:items_data_i] = items_data[:items_data_i]
                        items_data = temp_items_data

                    coverage['start_items'], items_data_i = items_data_i, items_data_i + coverage['max_items']

                # append the data of this item
                item_i = coverage['start_items'] + coverage['cur_items']
                items_data[item_i]['item_id'] = item_id
                items_data[item_i]['areaperil_id'] = areaperil_id
                items_data[item_i]['hazcdf_i'] = areaperil_to_haz_cdf[areaperil_id]
                items_data[item_i]['rng_index'] = this_rng_index
                items_data[item_i]['vulnerability_id'] = vuln_id

                coverage['cur_items'] += 1

    return (compute_i,
            items_data,
            rng_index)


if __name__ == '__main__':

    test_dir = Path(__file__).parent.parent.parent.parent.joinpath("tests") \
        .joinpath("assets").joinpath("test_model_2")

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
        ignore_correlation=True,
        effective_damageability=False,
    )

    # remove temporary file
    if file_out.exists():
        file_out.unlink()
