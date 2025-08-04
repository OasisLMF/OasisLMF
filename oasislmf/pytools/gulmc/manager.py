import atexit
import logging
import os
import sys
import json
from contextlib import ExitStack
from select import select

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import numba as nb
from numba.typed import Dict, List
from numba.types import Tuple as nb_Tuple
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasislmf.utils.data import analysis_settings_loader
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import nb_areaperil_int, nb_oasis_float, oasis_float, nb_oasis_int, oasis_int, correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.common.input_files import read_coverages, read_correlations
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.manager import get_damage_bins, get_vulns, get_intensity_bin_dict, encode_peril_id
from oasislmf.pytools.gul.common import MAX_LOSS_IDX, CHANCE_OF_LOSS_IDX, TIV_IDX, STD_DEV_IDX, MEAN_IDX, NUM_IDX
from oasislmf.pytools.gul.core import compute_mean_loss, get_gul
from oasislmf.pytools.gul.manager import write_losses, adjust_byte_mv_size
from oasislmf.pytools.gul.random import (compute_norm_cdf_lookup, compute_norm_inv_cdf_lookup,
                                         generate_correlated_hash_vector, generate_hash,
                                         generate_hash_hazard, get_corr_rval_float, get_random_generator)
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.gulmc.aggregate import (
    process_aggregate_vulnerability, process_vulnerability_weights, read_aggregate_vulnerability,
    read_vulnerability_weights, )
from oasislmf.pytools.gulmc.common import (DAMAGE_TYPE_ABSOLUTE, DAMAGE_TYPE_DURATION, DAMAGE_TYPE_RELATIVE, NP_BASE_ARRAY_SIZE,
                                           Keys, ItemAdjustment,
                                           NormInversionParameters, coverage_type, gul_header,
                                           gulSampleslevelHeader_size, gulSampleslevelRec_size,
                                           haz_arr_type, items_MC_data_type,
                                           gulmc_compute_info_type)
from oasislmf.pytools.gulmc.items import read_items, generate_item_map
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


VULN_LOOKUP_KEY_TYPE = nb_Tuple((nb_int32, nb_areaperil_int, nb_int32, nb_int32, nb_int32))
VULN_LOOKUP_VALUE_TYPE = nb_Tuple((nb_int32, nb_int32))

# parameter for get_corr_rval in a normal cdf
x_min = 1e-16
x_max = 1 - 1e-16
norm_inv_N = 1000000
cdf_min = -20
cdf_max = 20.
inv_factor = (norm_inv_N - 1) / (x_max - x_min)
norm_factor = (norm_inv_N - 1) / (cdf_max - cdf_min)


@nb.njit(cache=True)
def gen_empty_vuln_cdf_lookup(list_size, compute_info):
    """Generate structures needed to store and retrieve vulnerability cdf in the cache.

    Args:
        list_size (int): maximum number of cdfs to be stored in the cache.

    Returns:
        cached_vuln_cdf_lookup (Dict[VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE]): dict to store
          the map between vuln_id and intensity bin id and the location of the cdf in the cache.
        cached_vuln_cdf_lookup_keys (List[VULN_LOOKUP_VALUE_TYPE]): list of lookup keys.
    """
    cached_vuln_cdf_lookup = Dict.empty(VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE)
    cached_vuln_cdf_lookup_keys = List.empty_list(VULN_LOOKUP_KEY_TYPE)
    dummy = tuple((nb_int32(-1), nb_areaperil_int(0), nb_int32(-1), nb_int32(-1), nb_int32(-1)))
    for _ in range(list_size):
        cached_vuln_cdf_lookup_keys.append(dummy)
    compute_info['next_cached_vuln_cdf_i'] = 0
    return cached_vuln_cdf_lookup, cached_vuln_cdf_lookup_keys


def get_dynamic_footprint_adjustments(input_path):
    """Generate intensity adjustment array for dynamic footprint models.

    Args:
        input_path (str): location of the generated adjustments file.

    Returns:
        numpy array with itemid and adjustment factors
    """
    adjustments_fn = os.path.join(input_path, 'item_adjustments.csv')
    if os.path.isfile(adjustments_fn):
        adjustments_tb = np.loadtxt(adjustments_fn, dtype=ItemAdjustment, delimiter=",", skiprows=1, ndmin=1)
    else:
        items_fp = os.path.join(input_path, 'items.csv')
        items_tb = np.loadtxt(items_fp, dtype=items_dtype, delimiter=",", skiprows=1, ndmin=1)
        adjustments_tb = np.array([(i[0], 0, 0) for i in items_tb], dtype=ItemAdjustment)

    return adjustments_tb


def get_peril_id(input_path):
    """
    Get peril_id associated with item_id

    Args:
        input_path (str): The directory path where the 'gul_summary_map.csv' file is located.

    Returns:
        np.ndarray: A structured NumPy array with the following fields:
            - 'item_id' (oasis_int): The item ID as an integer.
            - 'peril_id' (oasis_int): The encoded peril ID as an integer.
    """

    dtype = np.dtype([
        ('item_id', oasis_int),
        ('peril_id', oasis_int)
    ])

    item_peril = pd.read_csv(
        os.path.join(input_path, 'gul_summary_map.csv'),
        usecols=['item_id', 'peril_id']
    )[['item_id', 'peril_id']]

    item_peril['peril_id'] = item_peril['peril_id'].apply(encode_peril_id)

    item_peril = np.array(
        list(item_peril.itertuples(index=False, name=None)),
        dtype=dtype)

    return item_peril


def get_vuln_rngadj(run_dir, vuln_dict):
    """
    Loads vulnerability adjustments from the analysis settings file.

    Args:
        run_dir (str): path to the run directory (used to load the analysis settings)

    Returns: (Dict[nb_int32, nb_float64]) vulnerability adjustments dictionary
    """
    settings_path = os.path.join(run_dir, "analysis_settings.json")
    vuln_adj = np.ones(len(vuln_dict), dtype=oasis_float)
    if not os.path.exists(settings_path):
        logger.debug(f"analysis_settings.json not found in {run_dir}.")
        return vuln_adj
    vulnerability_adjustments_field = analysis_settings_loader(settings_path).get('vulnerability_adjustments', None)
    if vulnerability_adjustments_field is not None:
        adjustments = vulnerability_adjustments_field.get('adjustments', None)
    else:
        adjustments = None
    if adjustments is None:
        logger.debug(f"vulnerability_adjustments not found in {settings_path}.")
        return vuln_adj
    for key, value in adjustments.items():
        if nb_int32(key) in vuln_dict.keys():
            vuln_adj[vuln_dict[nb_int32(key)]] = nb_oasis_float(value)
    return vuln_adj


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
            valid_areaperil_id = keys_df.loc[keys_df['PerilID'].isin(peril_filter), 'AreaPerilID'].to_numpy()
            logger.debug(
                f'Peril specific run: ({peril_filter}), {len(valid_areaperil_id)} AreaPerilID included out of {len(keys_df)}')
        else:
            valid_areaperil_id = None

        logger.debug('import damage bins')
        damage_bins = get_damage_bins(model_storage, ignore_file_type)

        logger.debug('import coverages')
        # coverages are numbered from 1, therefore we skip element 0 in `coverages`
        coverages_tb = read_coverages(input_path, ignore_file_type)
        coverages = np.zeros(coverages_tb.shape[0] + 1, coverage_type)
        coverages[1:]['tiv'] = coverages_tb

        # prepare for stochastic disaggregation
        logger.debug('import aggregate vulnerability definitions and vulnerability weights')
        aggregate_vulnerability = read_aggregate_vulnerability(model_storage, ignore_file_type)
        aggregate_weights = read_vulnerability_weights(model_storage, ignore_file_type)
        agg_vuln_to_vuln_ids = process_aggregate_vulnerability(aggregate_vulnerability)

        if aggregate_vulnerability is not None and aggregate_weights is None:
            raise FileNotFoundError(
                f"Vulnerability weights file not found at {model_storage.get_storage_url('', print_safe=True)[1]}"
            )

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
        items = rfn.merge_arrays((items,
                                  np.empty(items.shape,
                                           dtype=nb.from_dtype(np.dtype([("vulnerability_idx", np.int32)])))),
                                 flatten=True)

        # intensity adjustment
        if dynamic_footprint:
            logger.debug('get dynamic footprint adjustments')
            adjustments_tb = get_dynamic_footprint_adjustments(input_path)
            items = rfn.join_by(
                'item_id', items, adjustments_tb,
                jointype='leftouter', usemask=False,
                defaults={'intensity_adjustment': 0, 'return_period': 0}
            )

        # include peril_id
        if dynamic_footprint:
            logger.debug('get peril_id')
            item_peril = get_peril_id(input_path)
            items = rfn.join_by(
                'item_id', items, item_peril,
                jointype='leftouter', usemask=False,
                defaults={'peril_id': 0}
            )
        items.sort(order=['areaperil_id', 'vulnerability_id'])

        # build item map
        item_map, areaperil_ids_map, vuln_dict, agg_vuln_to_vuln_idxs, areaperil_vuln_idx_to_weight = generate_item_map(
            items,
            coverages,
            valid_areaperil_id,
            agg_vuln_to_vuln_ids)
        if aggregate_weights is not None:
            logger.debug('reconstruct aggregate vulnerability definitions and weights')
            process_vulnerability_weights(areaperil_vuln_idx_to_weight, vuln_dict, aggregate_weights)

        # import array to store the coverages to be computed
        # coverages are numebered from 1, therefore skip element 0.
        compute = np.zeros(coverages.shape[0] + 1, items_dtype['coverage_id'])

        logger.debug('import peril correlation groups')
        unique_peril_correlation_groups = np.unique(items['peril_correlation_group'])
        Nperil_correlation_groups = unique_peril_correlation_groups.shape[0]
        logger.info(f"Detected {Nperil_correlation_groups} peril correlation groups.")

        logger.debug('import footprint')
        footprint_obj = stack.enter_context(Footprint.load(model_storage, ignore_file_type, df_engine=model_df_engine))
        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
        else:
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        logger.debug('import vulnerabilities')
        vuln_adj = get_vuln_rngadj(run_dir, vuln_dict)
        vuln_array, _, _ = get_vulns(model_storage, run_dir, vuln_dict, num_intensity_bins, ignore_file_type, df_engine=model_df_engine)
        Nvulnerability, Ndamage_bins_max, Nintensity_bins = vuln_array.shape

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
        haz_seeds = np.zeros(len(np.unique(items['hazard_group_id'])), dtype=correlations_dtype['hazard_group_id'])
        vuln_seeds = np.zeros(len(np.unique(items['group_id'])), dtype=items_dtype['group_id'])

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

            # pre-compute lookup tables for the Gaussian cdf and inverse cdf
            # Notes:
            #  - the size `N` can be increased to achieve better resolution in the Gaussian cdf and inv cdf.
            #  - the function `get_corr_rval` to compute the correlated numbers is not affected by N.
            norm_inv_parameters = np.array((x_min, x_max, norm_inv_N, cdf_min, cdf_max, inv_factor, norm_factor), dtype=NormInversionParameters)

            norm_inv_cdf = compute_norm_inv_cdf_lookup(norm_inv_parameters['x_min'], norm_inv_parameters['x_max'], norm_inv_parameters['N'])
            norm_cdf = compute_norm_cdf_lookup(norm_inv_parameters['cdf_min'], norm_inv_parameters['cdf_max'], norm_inv_parameters['N'])
        else:
            # create dummy data structures with proper dtypes to allow correct numba compilation
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
            intensity_bin_dict = Dict.empty(nb_Tuple((nb_int32, nb_int32)), nb_int32)
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

        # default random values array for sample_size==0 case
        haz_rndms_base = np.empty((1, sample_size), dtype='float64')
        vuln_rndms_base = np.empty((1, sample_size), dtype='float64')
        haz_eps_ij = np.empty((1, sample_size), dtype='float64')
        damage_eps_ij = np.empty((1, sample_size), dtype='float64')

        while True:
            if not streams_in.readinto(event_id_mv):
                break

            # get the next event_id from the input stream
            compute_info['event_id'] = event_ids[0]
            event_footprint = event_footprint_obj.get_event(event_ids[0])

            if event_footprint is not None:
                areaperil_ids, Nhaz_arr_this_event, areaperil_to_haz_arr_i, haz_pdf, haz_arr_ptr = process_areaperils_in_footprint(
                    event_footprint,
                    areaperil_ids_map,
                    dynamic_footprint)

                if Nhaz_arr_this_event == 0:
                    # no items to be computed for this event
                    continue

                items_event_data, rng_index, hazard_rng_index, byte_mv = reconstruct_coverages(
                    compute_info,
                    areaperil_ids,
                    areaperil_ids_map,
                    areaperil_to_haz_arr_i,
                    item_map,
                    items,
                    coverages,
                    compute,
                    haz_seeds,
                    haz_peril_correlation_groups,
                    haz_corr_seeds,
                    vuln_seeds,
                    damage_peril_correlation_groups,
                    damage_corr_seeds,
                    dynamic_footprint,
                    byte_mv
                )

                # since these are never used outside of a sample > 0 branch we can remove the need to
                # generate (and potentially allocate) the random values. As at 2.3.5 the sampling method
                # for random values accounts for 25% of the runtime of the losses step not including
                # the get_event despite having a sample size of 0.
                if sample_size > 0:
                    # generation of "base" random values for hazard intensity and vulnerability sampling
                    haz_rndms_base = generate_rndm(haz_seeds[:hazard_rng_index], sample_size)
                    vuln_rndms_base = generate_rndm(vuln_seeds[:rng_index], sample_size)
                    haz_eps_ij = generate_rndm(haz_corr_seeds, sample_size, skip_seeds=1)
                    damage_eps_ij = generate_rndm(damage_corr_seeds, sample_size, skip_seeds=1)

                # create vulnerability cdf cache
                cached_vuln_cdfs = np.zeros((Nvulns_cached, compute_info['Ndamage_bins_max']), dtype=oasis_float)
                cached_vuln_cdf_lookup, lookup_keys = gen_empty_vuln_cdf_lookup(Nvulns_cached, compute_info)

                processing_done = False
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
                            haz_arr_ptr,
                            vuln_array,
                            damage_bins,
                            cached_vuln_cdf_lookup,
                            lookup_keys,
                            cached_vuln_cdfs,
                            agg_vuln_to_vuln_idxs,
                            areaperil_vuln_idx_to_weight,
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
                            intensity_bin_dict
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
                        write_start += stream_out.write(byte_mv[write_start: compute_info['cursor']].tobytes())

                logger.info(f"event {event_ids[0]} DONE")
    return 0


@nb.njit(cache=True, fastmath=True)
def get_haz_cdf(item_event_data, haz_cdf, haz_cdf_ptr, dynamic_footprint, intensity_adjustment, intensity_bin_dict):
    # get the right hazard cdf from the array containing all hazard cdfs
    hazcdf_i = item_event_data['hazcdf_i']
    haz_cdf_record = haz_cdf[haz_cdf_ptr[hazcdf_i]:haz_cdf_ptr[hazcdf_i + 1]]
    haz_cdf_prob = haz_cdf_record['probability']

    if dynamic_footprint:
        # adjust intensity in dynamic footprint
        haz_cdf_intensity = haz_cdf_record['intensity']
        haz_cdf_intensity = haz_cdf_intensity - intensity_adjustment
        haz_cdf_intensity = np.where(haz_cdf_intensity < 0, nb_int32(0), haz_cdf_intensity)
        haz_cdf_bin_id = np.zeros_like(haz_cdf_record['intensity_bin_id'])
        for haz_bin_idx in range(haz_cdf_bin_id.shape[0]):
            if haz_cdf_intensity[haz_bin_idx] <= 0:
                haz_cdf_bin_id[haz_bin_idx] = intensity_bin_dict[0]
            else:
                haz_cdf_bin_id[haz_bin_idx] = intensity_bin_dict[haz_cdf_intensity[haz_bin_idx]]
    else:
        haz_cdf_bin_id = haz_cdf_record['intensity_bin_id']
    return haz_cdf_prob, haz_cdf_bin_id


@nb.njit(fastmath=True)
def get_last_non_empty(cdf, bin_i):
    """
    remove empty bucket from the end
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


@nb.njit(fastmath=True)
def pdf_to_cdf(pdf, empty_cdf):
    """
    return the cumulative distribution from the probality distribution
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


@nb.njit(fastmath=True)
def calc_eff_damage_cdf(vuln_pdf, haz_pdf, eff_damage_cdf_empty):
    """
    calculate the covoluted cumulative distribution between vulnerability damage and hazard probability distribution
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


@nb.njit()
def cache_cdf(next_cached_vuln_cdf_i, cached_vuln_cdfs, cached_vuln_cdf_lookup, cached_vuln_cdf_lookup_keys, cdf, cdf_key):
    if cdf_key not in cached_vuln_cdf_lookup:  # already cached
        # cache the cdf
        if cached_vuln_cdf_lookup_keys[next_cached_vuln_cdf_i] in cached_vuln_cdf_lookup:
            # overwrite cache
            cached_vuln_cdf_lookup.pop(cached_vuln_cdf_lookup_keys[next_cached_vuln_cdf_i])

        cached_vuln_cdfs[next_cached_vuln_cdf_i, :cdf.shape[0]] = cdf
        cached_vuln_cdf_lookup[cdf_key] = tuple((nb_int32(next_cached_vuln_cdf_i), nb_int32(cdf.shape[0])))
        cached_vuln_cdf_lookup_keys[next_cached_vuln_cdf_i] = cdf_key
        next_cached_vuln_cdf_i += 1
        next_cached_vuln_cdf_i %= cached_vuln_cdfs.shape[0]
    return next_cached_vuln_cdf_i


@nb.njit(fastmath=True, cache=True)
def get_gul_from_vuln_cdf(vuln_rval, vuln_cdf, Ndamage_bins, damage_bins, bin_scaling):
    # find the damage cdf bin in which the random value `vuln_rval` falls into
    vuln_bin_idx = binary_search(vuln_rval, vuln_cdf, Ndamage_bins - 1)

    # compute ground-up losses
    return get_gul(
        damage_bins['bin_from'][vuln_bin_idx],
        damage_bins['bin_to'][vuln_bin_idx],
        damage_bins['interpolation'][vuln_bin_idx],
        vuln_cdf[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
        vuln_cdf[vuln_bin_idx],
        vuln_rval,
        bin_scaling,
    )


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
                         damage_bins,
                         cached_vuln_cdf_lookup,
                         cached_vuln_cdf_lookup_keys,
                         cached_vuln_cdfs,
                         agg_vuln_to_vuln_idxs,
                         areaperil_vuln_idx_to_weight,
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
                         intensity_bin_dict
                         ):
    """Compute losses for an event.

    Args:
        compute_info (ndarray): information on the state of the computation
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        coverage_ids (numpy.array[int]): array of unique coverage ids used in this event.
        items_event_data (numpy.array[items_data_type]): items-related data.
        items (np.ndarray): items table merged with correlation parameters.
        sample_size (int): number of random samples to draw.
        haz_pdf (np.array[oasis_float]): hazard intensity cdf.
        haz_arr_ptr (np.array[int]): array with the indices where each cdf record starts in `haz_cdf`.
        vuln_array (np.array[float]): damage pdf for different vulnerability functions, as a function of hazard intensity.
        damage_bins (List[Union[damagebindictionaryCsv, damagebindictionary]]): loaded data from the damage_bin_dict file.
        cached_vuln_cdf_lookup (Dict[VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE]): dict to store
          the map between vuln_id and intensity bin id and the location of the cdf in the cache.
        cached_vuln_cdf_lookup_keys (List[VULN_LOOKUP_VALUE_TYPE]): list of lookup keys.
        cached_vuln_cdfs (np.array[oasis_float]): vulnerability cdf cache.
        agg_vuln_to_vuln_idxs (dict[int, list[int]]): map between aggregate vulnerability id and the list of indices where the individual vulnerability_ids
          that compose it are stored in `vuln_array`.
        areaperil_vuln_idx_to_weight (dict[AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE]): map between the areaperil id and the index where the vulnerability function
          is stored in `vuln_array` and the vulnerability weight.
        losses (numpy.array[oasis_float]): array (to be re-used) to store losses for each item.
        haz_rndms_base (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed for the hazard intensity sampling.
        vuln_rndms_base (numpy.array[float64]): 2d array of shape (number of seeds, sample_size) storing the random values
          drawn for each seed for the damage sampling.
        vuln_adj (np.array): map array between vulnerability_idx and the adjustment factor to be applied to the (random numbers extracted) vulnerability function.
        haz_eps_ij (np.array[float]): correlated random values of shape `(number of seeds, sample_size)` for hazard sampling.
        damage_eps_ij (np.array[float]): correlated random values of shape `(number of seeds, sample_size)` for damage sampling.
        norm_inv_parameters (NormInversionParameters): parameters for the Normal (Gaussian) inversion functionality.
        norm_inv_cdf (np.array[float]): inverse Gaussian cdf.
        norm_cdf (np.array[float]): Gaussian cdf.
        vuln_z_unif (np.array[float]): buffer to be re-used to store the correlated random values for vuln.
        haz_z_unif (np.array[float]): buffer to be re-used to store the correlated random values for haz.
        byte_mv (numpy.array): byte view of where the output is buffered.
        dynamic_footprint,
        intensity_bin_dict

    Returns:
        True if processing is done else false
    """
    haz_cdf_empty = np.empty(compute_info['Ndamage_bins_max'], dtype=oasis_float)
    vuln_pdf_empty = np.empty((vuln_array.shape[2], compute_info['Ndamage_bins_max']), dtype=vuln_array.dtype)
    eff_damage_cdf_empty = np.empty(compute_info['Ndamage_bins_max'], dtype=oasis_float)
    haz_i_to_Ndamage_bins_empty = np.empty(vuln_array.shape[2], dtype=oasis_int)
    haz_i_to_vuln_cdf_empty = np.empty((vuln_array.shape[2], compute_info['Ndamage_bins_max']), dtype=vuln_array.dtype)

    # we process at least one full coverage at a time, so when we write to stream, we write the whole buffer
    compute_info['cursor'] = 0

    # loop through all the coverages that remain to be computed
    for coverage_i in range(compute_info['coverage_i'], compute_info['coverage_n']):
        coverage = coverages[coverage_ids[coverage_i]]
        tiv = coverage['tiv']
        Nitems = coverage['cur_items']
        exposureValue = tiv / Nitems

        # estimate max number of bytes needed to output this coverage
        # conservatively assume all random samples are printed (losses>loss_threshold)
        est_cursor_bytes = Nitems * compute_info['max_bytes_per_item']

        # return before processing this coverage if the number of free bytes left in the buffer
        # is not sufficient to write out the full coverage
        if compute_info['cursor'] + est_cursor_bytes > byte_mv.shape[0]:
            return False

        # compute losses for each item
        for item_j in range(Nitems):
            item_event_data = items_event_data[coverage['start_items'] + item_j]
            item_id = item_event_data['item_id']
            rng_index = item_event_data['rng_index']
            hazard_rng_index = item_event_data['hazard_rng_index']

            item = items[item_event_data['item_idx']]
            areaperil_id = item['areaperil_id']
            vulnerability_id = item['vulnerability_id']
            if dynamic_footprint is not None:
                intensity_adjustment = item['intensity_adjustment']
            else:
                intensity_adjustment = nb_oasis_int(0)

            if item['vulnerability_id'] in agg_vuln_to_vuln_idxs:
                agg_vuln_key_id = item['areaperil_id']
            else:
                agg_vuln_key_id = nb_areaperil_int(0)

            haz_arr_i = item_event_data['haz_arr_i']
            haz_pdf_record = haz_pdf[haz_arr_ptr[haz_arr_i]:haz_arr_ptr[haz_arr_i + 1]]

            # we calculate this adjusted hazard pdf
            # get the right hazard pdf from the array containing all hazard cdfs
            if dynamic_footprint is not None:
                # adjust intensity in dynamic footprint
                haz_intensity = haz_pdf_record['intensity']
                haz_intensity = haz_intensity - intensity_adjustment
                haz_bin_id = np.zeros_like(haz_pdf_record['intensity_bin_id'])
                peril_id = item['peril_id']
                for haz_bin_idx in range(haz_bin_id.shape[0]):
                    try:
                        haz_bin_id[haz_bin_idx] = intensity_bin_dict[peril_id, haz_intensity[haz_bin_idx]]
                    except:
                        haz_bin_id[haz_bin_idx] = intensity_bin_dict[peril_id, 0]
            else:
                haz_bin_id = haz_pdf_record['intensity_bin_id']
            haz_pdf_prob = haz_pdf_record['probability']

            eff_damage_cdf_key = tuple((item['vulnerability_id'],
                                        agg_vuln_key_id,
                                        nb_oasis_int(haz_arr_i),
                                        nb_oasis_int(intensity_adjustment),
                                        nb_oasis_int(0)))

            # determine if all the needed cdf are cached
            do_calc_vuln_ptf = eff_damage_cdf_key not in cached_vuln_cdf_lookup
            haz_cdf_prob = pdf_to_cdf(haz_pdf_prob, haz_cdf_empty)
            Nhaz_bins = haz_cdf_prob.shape[0]
            if not compute_info['effective_damageability']:
                for haz_i in range(Nhaz_bins):
                    haz_lookup_key = tuple((item['vulnerability_id'], agg_vuln_key_id, nb_oasis_int(0), intensity_adjustment, haz_bin_id[haz_i]))
                    do_calc_vuln_ptf = do_calc_vuln_ptf or (haz_lookup_key not in cached_vuln_cdf_lookup)

            if do_calc_vuln_ptf:  # some cdf are not cached
                # we get the vuln_pdf, needed for effcdf and each cdf
                vuln_pdf = vuln_pdf_empty[:Nhaz_bins]
                vuln_pdf[:] = 0
                if item['vulnerability_id'] in agg_vuln_to_vuln_idxs:  # we calculate the custom vuln_array for this aggregate
                    tot_weights = 0.
                    agg_vulns_idx = agg_vuln_to_vuln_idxs[item['vulnerability_id']]
                    for j, vuln_i in enumerate(agg_vulns_idx):
                        if (item['areaperil_id'], vuln_i) in areaperil_vuln_idx_to_weight:
                            weight = np.float64(areaperil_vuln_idx_to_weight[(item['areaperil_id'], vuln_i)])
                            if weight > 0:
                                tot_weights += weight
                                for haz_i in range(Nhaz_bins):
                                    has_prob = False
                                    for damage_bin_i in range(compute_info['Ndamage_bins_max']):
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
                        for j, vuln_i in enumerate(agg_vulns_idx):
                            for haz_i in range(Nhaz_bins):
                                vuln_pdf[haz_i] += vuln_array[vuln_i, :, haz_bin_id[haz_i] - 1]
                        vuln_pdf /= len(agg_vulns_idx)
                else:
                    for haz_i in range(Nhaz_bins):
                        vuln_pdf[haz_i] = vuln_array[item['vulnerability_idx'], :, haz_bin_id[haz_i] - 1]

                # calculate and cache all cdf
                eff_damage_cdf = calc_eff_damage_cdf(vuln_pdf, haz_pdf_prob, eff_damage_cdf_empty)
                compute_info['next_cached_vuln_cdf_i'] = cache_cdf(
                    compute_info['next_cached_vuln_cdf_i'], cached_vuln_cdfs, cached_vuln_cdf_lookup,
                    cached_vuln_cdf_lookup_keys, eff_damage_cdf, eff_damage_cdf_key)

                if not compute_info['effective_damageability']:  # we cache all the vuln_cdf needed
                    haz_i_to_Ndamage_bins = haz_i_to_Ndamage_bins_empty[:Nhaz_bins]
                    haz_i_to_vuln_cdf = haz_i_to_vuln_cdf_empty[:Nhaz_bins]
                    for haz_i in range(Nhaz_bins):
                        haz_i_to_Ndamage_bins[haz_i] = pdf_to_cdf(vuln_pdf[haz_i], haz_i_to_vuln_cdf[haz_i]).shape[0]

                        lookup_key = tuple((item['vulnerability_id'], agg_vuln_key_id, nb_oasis_int(0), intensity_adjustment, haz_bin_id[haz_i]))
                        compute_info['next_cached_vuln_cdf_i'] = cache_cdf(
                            compute_info['next_cached_vuln_cdf_i'], cached_vuln_cdfs, cached_vuln_cdf_lookup,
                            cached_vuln_cdf_lookup_keys, haz_i_to_vuln_cdf[haz_i][:haz_i_to_Ndamage_bins[haz_i]],
                            lookup_key)

            else:  # cdf are cached
                start, Ndamage_bins = cached_vuln_cdf_lookup[eff_damage_cdf_key]
                eff_damage_cdf = cached_vuln_cdfs[start, :Ndamage_bins]

                if not compute_info['effective_damageability']:
                    haz_i_to_Ndamage_bins = haz_i_to_Ndamage_bins_empty[:Nhaz_bins]
                    haz_i_to_vuln_cdf = haz_i_to_vuln_cdf_empty[:Nhaz_bins]
                    for haz_i in range(Nhaz_bins):
                        lookup_key = tuple((item['vulnerability_id'], agg_vuln_key_id, nb_oasis_int(0), intensity_adjustment, haz_bin_id[haz_i]))
                        start, Ndamage_bins = cached_vuln_cdf_lookup[lookup_key]

                        haz_i_to_Ndamage_bins[haz_i] = Ndamage_bins
                        haz_i_to_vuln_cdf[haz_i][:Ndamage_bins] = cached_vuln_cdfs[start, :Ndamage_bins]

            Neff_damage_bins = eff_damage_cdf.shape[0]

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
                if compute_info['do_haz_correlation'] and item['hazard_correlation_value'] > 0:
                    # use correlation definitions to draw correlated random values into haz_z_unif
                    get_corr_rval_float(
                        haz_eps_ij[item['peril_correlation_group']], haz_rndms_base[hazard_rng_index], item['hazard_correlation_value'],
                        norm_inv_parameters['x_min'], norm_inv_cdf, norm_inv_parameters['inv_factor'],
                        norm_inv_parameters['cdf_min'], norm_cdf, norm_inv_parameters['norm_factor'],
                        sample_size, haz_z_unif
                    )
                else:
                    haz_z_unif[:] = haz_rndms_base[hazard_rng_index]

                if compute_info['do_correlation'] and item['damage_correlation_value'] > 0:
                    # use correlation definitions to draw correlated random values into vuln_z_unif
                    get_corr_rval_float(
                        damage_eps_ij[item['peril_correlation_group']], vuln_rndms_base[rng_index], item['damage_correlation_value'],
                        norm_inv_parameters['x_min'], norm_inv_cdf, norm_inv_parameters['inv_factor'],
                        norm_inv_parameters['cdf_min'], norm_cdf, norm_inv_parameters['norm_factor'],
                        sample_size, vuln_z_unif
                    )
                else:
                    # do not use correlation
                    vuln_z_unif[:] = vuln_rndms_base[rng_index]

                if agg_vuln_key_id == 0:  # if 0 we have a single vuln id
                    vuln_z_unif *= vuln_adj[item['vulnerability_idx']]

                if compute_info['debug'] == 1:  # store the random value used for the hazard sampling instead of the loss
                    losses[1:, item_j] = haz_z_unif[:]

                elif compute_info['debug'] == 2:  # store the random value used for the damage sampling instead of the loss
                    losses[1:, item_j] = vuln_z_unif[:]

                else:  # calculate gul
                    if compute_info['effective_damageability']:
                        for sample_idx in range(1, sample_size + 1):
                            losses[sample_idx, item_j] = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], eff_damage_cdf,
                                                                               Neff_damage_bins, damage_bins, damage_bin_scaling)
                    elif Nhaz_bins == 1:  # only one hazard possible
                        Ndamage_bins = haz_i_to_Ndamage_bins[0]
                        vuln_cdf = haz_i_to_vuln_cdf[0][:Ndamage_bins]
                        for sample_idx in range(1, sample_size + 1):
                            losses[sample_idx, item_j] = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], vuln_cdf,
                                                                               Ndamage_bins, damage_bins, damage_bin_scaling)
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

                            losses[sample_idx, item_j] = get_gul_from_vuln_cdf(vuln_z_unif[sample_idx - 1], vuln_cdf,
                                                                               Ndamage_bins, damage_bins, damage_bin_scaling)

        # write the losses to the output memoryview
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
                                    present_areaperils,
                                    dynamic_footprint):
    """
    Process all the areaperils in the footprint, filtering and retaining only those who have associated vulnerability functions

    Args:
        event_footprint (np.array[Event or footprint_event_dtype]): footprint, made of one or more event entries.
        present_areaperils (dict[int, int]): areaperil to vulnerability index dictionary.
        dynamic_footprint (boolean): true if there is dynamic_footprint

    Returns:
        areaperil_ids (List[int]): list of all areaperil_ids present in the footprint.
        Nhaz_arr_this_event (int): number of hazard stored for this event. If zero, it means no items have losses in such event.
        areaperil_to_haz_arr_i (dict[int, int]): map between the areaperil_id and the hazard index in haz_arr_ptr.
        haz_pdf (np.array[oasis_float]): hazard intensity pdf.
        haz_arr_ptr (np.array[int]): array with the indices where each hazard intensities record starts in haz arrays (ie, haz_pdf).
    """
    # init data structures
    haz_prob_start_in_footprint = List.empty_list(nb_int64)
    areaperil_ids = List.empty_list(nb_areaperil_int)

    footprint_i = 0
    last_areaperil_id = nb_areaperil_int(0)
    last_areaperil_id_start = nb_int64(0)
    haz_arr_i = 0
    areaperil_to_haz_arr_i = Dict.empty(nb_areaperil_int, nb_oasis_int)

    Nevent_footprint_entries = len(event_footprint)
    haz_pdf = np.empty(Nevent_footprint_entries, dtype=haz_arr_type)  # max size

    arr_ptr_start = 0
    arr_ptr_end = 0
    haz_arr_ptr = List([0])

    while footprint_i <= Nevent_footprint_entries:

        if footprint_i < Nevent_footprint_entries:
            areaperil_id = event_footprint[footprint_i]['areaperil_id']
        else:
            areaperil_id = nb_areaperil_int(0)

        if areaperil_id != last_areaperil_id:
            # one areaperil_id is completed

            if last_areaperil_id > 0:
                if last_areaperil_id in present_areaperils:
                    # if items with this areaperil_id exist, process and store this areaperil_id
                    areaperil_ids.append(last_areaperil_id)
                    haz_prob_start_in_footprint.append(last_areaperil_id_start)
                    areaperil_to_haz_arr_i[last_areaperil_id] = nb_int32(haz_arr_i)
                    haz_arr_i += 1

                    # store the hazard intensity pdf
                    arr_ptr_end = arr_ptr_start + (footprint_i - last_areaperil_id_start)
                    haz_pdf['probability'][arr_ptr_start: arr_ptr_end] = event_footprint['probability'][last_areaperil_id_start: footprint_i]
                    haz_pdf['intensity_bin_id'][arr_ptr_start: arr_ptr_end] = event_footprint['intensity_bin_id'][last_areaperil_id_start: footprint_i]
                    if dynamic_footprint is not None:
                        haz_pdf['intensity'][arr_ptr_start: arr_ptr_end] = event_footprint['intensity'][last_areaperil_id_start: footprint_i]

                    haz_arr_ptr.append(arr_ptr_end)
                    arr_ptr_start = arr_ptr_end

            last_areaperil_id = areaperil_id
            last_areaperil_id_start = footprint_i

        footprint_i += 1

    Nhaz_arr_this_event = haz_arr_i

    return (areaperil_ids,
            Nhaz_arr_this_event,
            areaperil_to_haz_arr_i,
            haz_pdf[:arr_ptr_end],
            haz_arr_ptr)


@nb.njit(cache=True, fastmath=True)
def reconstruct_coverages(compute_info,
                          areaperil_ids,
                          areaperil_ids_map,
                          areaperil_to_haz_arr_i,
                          item_map,
                          items,
                          coverages,
                          compute,
                          haz_seeds,
                          haz_peril_correlation_groups,
                          haz_corr_seeds,
                          vuln_seeds,
                          damage_peril_correlation_groups,
                          damage_corr_seeds,
                          dynamic_footprint,
                          byte_mv):
    """Register each item to its coverage, with the location of the corresponding hazard intensity cdf
    in the footprint, compute the random seeds for the hazard intensity and vulnerability samples.

    Args:
        dynamic_compute_info (dynamic_compute_info_type): ndarray that store all dynamic info on computation
        areaperil_ids (List[int]): list of all areaperil_ids present in the footprint.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
          areaperil_id and all the vulnerability ids associated with it.
        areaperil_to_haz_arr_i (dict[int, int]): map between the areaperil_id and the hazard arr index.
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
        items (np.ndarray): items table merged with correlation parameters.
        coverages (numpy.array[oasis_float]): array with the coverage values for each coverage_id.
        compute (numpy.array[int]): list of coverage ids to be computed.
        haz_seeds (numpy.array[int]): the random seeds to draw the hazard intensity samples.
        haz_peril_correlation_groups (numpy.array[int]): unique peril_correlation_groups for hazard
        haz_corr_seeds (numpy.array[int]): empty buffer to write hazard_corr_seeds
        vuln_seeds (numpy.array[int]): the random seeds to draw the damage samples.
        damage_peril_correlation_groups (numpy.array[int]): unique peril_correlation_groups for damage
        damage_corr_seeds (numpy.array[int]): empty buffer to write damage_corr_seeds
        dynamic_footprint (bollean): true if dynamic_footprint is on
        byte_mv : writing buffer

    Returns:
        compute_i (int): index of the last coverage id stored in `compute`.
        items_data (numpy.array[items_MC_data_type]): item-related data.
        rng_index (int): number of unique random seeds for damage sampling computed so far.
        hazard_rng_index (int): number of unique random seeds for hazard intensity sampling computed so far.
        byte_mv : writing buffer with increased size if needed
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
                    vuln_seeds[rng_index] = generate_hash(group_id, compute_info['event_id'])
                    this_rng_index = rng_index
                    rng_index += 1

                else:
                    this_rng_index = group_id_rng_index[group_id]

                hazard_group_id = items[item_idx]['hazard_group_id']
                if hazard_group_id not in hazard_group_id_hazard_rng_index:
                    hazard_group_id_hazard_rng_index[hazard_group_id] = hazard_rng_index
                    haz_seeds[hazard_rng_index] = generate_hash_hazard(hazard_group_id, compute_info['event_id'])
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
                items_event_data[item_i]['haz_arr_i'] = areaperil_to_haz_arr_i[areaperil_id]
                items_event_data[item_i]['rng_index'] = this_rng_index
                items_event_data[item_i]['hazard_rng_index'] = this_hazard_rng_index
                if dynamic_footprint is not None:
                    items_event_data[item_i]['intensity_adjustment'] = items[item_idx]['intensity_adjustment']
                    items_event_data[item_i]['return_period'] = items[item_idx]['return_period']

                coverage['cur_items'] += 1
    compute_info['coverage_i'] = 0
    compute_info['coverage_n'] = compute_i
    byte_mv = adjust_byte_mv_size(byte_mv, np.max(coverages['cur_items']) * compute_info['max_bytes_per_item'])

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
        'random_generator': 1,
        'run_dir': '.',
        'model_df_engine': 'oasis_data_manager.df_reader.reader.OasisPandasReader',
        'dynamic_footprint': False}
    run(**kwargs)
