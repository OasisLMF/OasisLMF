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
import numpy.lib.recfunctions as rfn
import pandas as pd
import numba as nb
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasislmf.utils.data import analysis_settings_loader
from oasis_data_manager.filestore.config import get_storage_from_config_path
from oasislmf.pytools.common.data import areaperil_int, nb_areaperil_int, nb_oasis_float, oasis_float, nb_oasis_int, oasis_int, correlations_dtype, items_dtype
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.common.hashmap import (
    unpack as hm_unpack, _find_key as hm_find_key, NOT_FOUND as HM_NOT_FOUND,
    init_dict as hm_init_dict, _try_add_key as hm_try_add_key,
    i_add_key_fail as HM_ADD_FAIL, new_slot_bit as HM_NEW_SLOT_BIT, slot_mask as HM_SLOT_MASK,
    rehash as hm_rehash,
)
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
from oasislmf.pytools.gulmc.common import (DAMAGE_TYPE_ABSOLUTE,
                                           DAMAGE_TYPE_DURATION,
                                           DAMAGE_TYPE_RELATIVE,
                                           NP_BASE_ARRAY_SIZE, Keys,
                                           ItemAdjustment,
                                           NormInversionParameters,
                                           coverage_type, gul_header,
                                           gulSampleslevelHeader_size,
                                           gulSampleslevelRec_size,
                                           haz_arr_type, items_MC_data_type,
                                           gulmc_compute_info_type)
from oasislmf.pytools.common.id_index import build as id_index_build, get_idx as id_index_get_idx, NOT_FOUND as ID_INDEX_NOT_FOUND
from oasislmf.pytools.gulmc.items import read_items, generate_item_map
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.ping import oasis_ping
from oasislmf.utils.defaults import SERVER_UPDATE_TIME

logger = logging.getLogger(__name__)


CDF_CACHE_EMPTY = nb_int64(-1)

# parameter for get_corr_rval in a normal cdf
x_min = 1e-16
x_max = 1 - 1e-16
norm_inv_N = 1000000
cdf_min = -20
cdf_max = 20.
inv_factor = (norm_inv_N - 1) / (x_max - x_min)
norm_factor = (norm_inv_N - 1) / (cdf_max - cdf_min)


@nb.njit(cache=True)
def _build_cdf_group_indices(vuln_ja_offsets, vuln_ja_item_idxs, items, dynamic_footprint):
    """Assign a sequential index to each unique CDF-producing group.

    A CDF group is a set of items that share identical vulnerability CDFs. For non-dynamic
    models, each (areaperil, vuln_id) pair — which corresponds to one position in the
    item_map jagged array — gets a single index. For dynamic models, items within the same
    pair may have different intensity_adjustment values that produce different CDFs, so each
    unique adjustment gets its own sub-index.

    Numba compiles two specializations based on whether dynamic_footprint is None or not.

    Args:
        vuln_ja_offsets (np.array[oasis_int]): L2 CSR offsets (N_pairs + 1).
        vuln_ja_item_idxs (np.array[oasis_int]): flat item indices.
        items (np.ndarray): items table (must have 'intensity_adjustment' for dynamic).
        dynamic_footprint: None for static footprints, truthy for dynamic.

    Returns:
        item_cdf_group_idx (np.array[int64]): maps item_idx → CDF group index.
        n_cdf_groups (int): total number of unique CDF groups.
    """
    item_cdf_group_idx = np.empty(len(items), dtype=np.int64)
    n_pairs = len(vuln_ja_offsets) - 1
    cdf_group_cache_id = 0

    if dynamic_footprint is None:
        for k in range(n_pairs):
            start = vuln_ja_offsets[k]
            end = vuln_ja_offsets[k + 1]
            for pos in range(start, end):
                item_cdf_group_idx[vuln_ja_item_idxs[pos]] = cdf_group_cache_id
            cdf_group_cache_id += 1
    else:
        adj_key_storage = np.empty(max(len(items), 1), dtype=oasis_int)
        adj_cache_ids = np.empty(max(len(items), 1), dtype=np.int64)

        for k in range(n_pairs):
            start = vuln_ja_offsets[k]
            end = vuln_ja_offsets[k + 1]
            n_pair_items = end - start
            # fresh hashmap per pair (maps intensity_adjustment → dense index)
            adj_table = hm_init_dict(max(n_pair_items, 1))
            adj_info, adj_lookup, adj_index = hm_unpack(adj_table)

            for pos in range(start, end):
                item_idx = vuln_ja_item_idxs[pos]
                adj = items[item_idx]['intensity_adjustment']

                result = hm_try_add_key(adj_info, adj_lookup, adj_index, adj_key_storage, adj)
                while result == HM_ADD_FAIL:
                    adj_table = hm_rehash(adj_table, adj_key_storage)
                    adj_info, adj_lookup, adj_index = hm_unpack(adj_table)
                    result = hm_try_add_key(adj_info, adj_lookup, adj_index, adj_key_storage, adj)

                dense_idx = adj_index[result & HM_SLOT_MASK]
                if result & HM_NEW_SLOT_BIT:  # new unique adjustment
                    adj_cache_ids[dense_idx] = cdf_group_cache_id
                    cdf_group_cache_id += 1
                item_cdf_group_idx[item_idx] = adj_cache_ids[dense_idx]

    return item_cdf_group_idx, cdf_group_cache_id


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


def get_vuln_rngadj(run_dir, vuln_map, vuln_map_keys):
    """
    Loads vulnerability adjustments from the analysis settings file.

    Args:
        run_dir (str): path to the run directory (used to load the analysis settings)
        vuln_map (np.ndarray[uint8]): packed hashmap table mapping vuln_id to dense index.
        vuln_map_keys (np.ndarray[int32]): array of unique vulnerability ids (hashmap keys).

    Returns: (np.ndarray[oasis_float]) vulnerability adjustments array, indexed by dense vuln index.
    """
    settings_path = os.path.join(run_dir, "analysis_settings.json")
    vuln_adj = np.ones(len(vuln_map_keys), dtype=oasis_float)
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
    hm_info, hm_lookup, hm_index = hm_unpack(vuln_map)
    for key, value in adjustments.items():
        slot = hm_find_key(hm_info, hm_lookup, hm_index, vuln_map_keys, np.int32(int(key)))
        if slot != HM_NOT_FOUND:
            idx = hm_index[slot]
            vuln_adj[idx] = nb_oasis_float(value)
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
        if os.path.exists(os.path.join(input_path, 'keys.csv')):
            keys_df = pd.read_csv(os.path.join(input_path, 'keys.csv'), dtype=Keys)
            if peril_filter:
                valid_areaperil_id = np.unique(keys_df.loc[keys_df['PerilID'].isin(peril_filter), 'AreaPerilID'])
                logger.debug(
                    f'Peril specific run: ({peril_filter}), {len(valid_areaperil_id)} AreaPerilID included out of {len(keys_df)}')
            else:
                valid_areaperil_id = np.unique(keys_df['AreaPerilID'])
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
        agg_vuln_ids, agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids = process_aggregate_vulnerability(aggregate_vulnerability)

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
        if valid_areaperil_id is not None:
            items = items[np.isin(items['areaperil_id'], valid_areaperil_id)]
        items = rfn.merge_arrays((items,
                                  np.empty(items.shape,
                                           dtype=nb.from_dtype(np.dtype([("vulnerability_idx", oasis_int),
                                                                         ("areaperil_agg_vuln_idx", oasis_int)])))),
                                 flatten=True)
        items['areaperil_agg_vuln_idx'] = -1

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
        # Pre-compute sequential indices for group_id and hazard_group_id
        # to enable array-based lookups instead of Numba Dict lookups in reconstruct_coverages
        unique_group_ids_arr, group_seq_ids = np.unique(items['group_id'], return_inverse=True)
        unique_haz_group_ids_arr, haz_group_seq_ids = np.unique(items['hazard_group_id'], return_inverse=True)
        n_unique_groups = len(unique_group_ids_arr)
        n_unique_haz_groups = len(unique_haz_group_ids_arr)
        items = rfn.merge_arrays((items,
                                  np.empty(items.shape,
                                           dtype=nb.from_dtype(np.dtype([("group_seq_id", np.int32),
                                                                         ("hazard_group_seq_id", np.int32)])))),
                                 flatten=True)
        items['group_seq_id'] = group_seq_ids
        items['hazard_group_seq_id'] = haz_group_seq_ids

        items.sort(order=['areaperil_id', 'vulnerability_id'])

        # build item map (two-level jagged array)
        (item_map_ja_areaperil_ids, item_map_ja_offsets,
         item_map_ja_vuln_ids, item_map_ja_vuln_ja_offsets,
         item_map_ja_vuln_ja_item_idxs,
         vuln_map, vuln_map_keys,
         areaperil_agg_vuln_idx_ja_offsets, areaperil_agg_vuln_idx_ja_data,
         areaperil_agg_vuln_idx_ja_areaperil_ids) = generate_item_map(
            items,
            coverages,
            agg_vuln_id_ja_id_ind, agg_vuln_id_ja_offsets, agg_vuln_id_ja_vuln_ids)
        # Build id_index for areaperil_id -> dense index lookup
        item_map_ja_id_ind = id_index_build(item_map_ja_areaperil_ids)

        # Pre-compute CDF group indices: each unique (areaperil, vuln_id) pair (non-dynamic)
        # or (areaperil, vuln_id, intensity_adjustment) group (dynamic) gets a sequential index.
        # Items sharing a CDF group share cached CDFs.
        item_cdf_group_idx, n_cdf_groups = _build_cdf_group_indices(
            item_map_ja_vuln_ja_offsets, item_map_ja_vuln_ja_item_idxs,
            items, dynamic_footprint if dynamic_footprint else None)

        if aggregate_weights is not None:
            logger.debug('reconstruct aggregate vulnerability definitions and weights')
            process_vulnerability_weights(areaperil_agg_vuln_idx_ja_areaperil_ids, areaperil_agg_vuln_idx_ja_data,
                                          vuln_map, vuln_map_keys, aggregate_weights)
        del areaperil_agg_vuln_idx_ja_areaperil_ids  # only needed during setup

        # import array to store the coverages to be computed
        # coverages are numebered from 1, therefore skip element 0.
        compute = np.zeros(coverages.shape[0] + 1, items_dtype['coverage_id'])

        logger.debug('import peril correlation groups')
        unique_peril_correlation_groups = np.unique(items['peril_correlation_group'])
        Nperil_correlation_groups = unique_peril_correlation_groups.shape[0]
        logger.info(f"Detected {Nperil_correlation_groups} peril correlation groups.")

        logger.debug('import footprint')
        footprint_obj = stack.enter_context(Footprint.load(model_storage, ignore_file_type,
                                            df_engine=model_df_engine, areaperil_ids=item_map_ja_areaperil_ids))
        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
        else:
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        logger.debug('import vulnerabilities')
        vuln_adj = get_vuln_rngadj(run_dir, vuln_map, vuln_map_keys)
        vuln_array, _, _ = get_vulns(model_storage, run_dir, vuln_map, vuln_map_keys,
                                     num_intensity_bins, ignore_file_type, df_engine=model_df_engine)
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
        max_cached_vuln_cdf_size_bytes = max_cached_vuln_cdf_size_MB * 1024 * 1024  # cache size in bytes
        max_Nnumbers_cached_vuln_cdf = max_cached_vuln_cdf_size_bytes // oasis_float.itemsize  # total numbers that can fit in the cache
        max_Nvulnerability_cached_vuln_cdf = max_Nnumbers_cached_vuln_cdf // Ndamage_bins_max  # max number of vulnerability functions that can be stored in cache
        # number of vulnerability functions to be cached, rounded up to power of two for bitwise masking
        Nvulns_cached = min(Nvulnerability * Nintensity_bins, max_Nvulnerability_cached_vuln_cdf)
        Nvulns_cached = 1 << (max(Nvulns_cached, 1) - 1).bit_length()  # next power of two
        cdf_cache_mask = np.int64(Nvulns_cached - 1)
        logger.info(f"max vulnerability cdf cache size is {max_cached_vuln_cdf_size_MB}MB")
        logger.info(
            f"generating a cache of shape ({Nvulns_cached}, {Ndamage_bins_max}) and size {Nvulns_cached * Ndamage_bins_max * oasis_float.itemsize / 1024 / 1024:8.3f}MB")

        # maximum bytes to be written in the output stream for 1 item
        event_footprint_obj = FootprintLayerClient if data_server else footprint_obj

        if dynamic_footprint:
            intensity_bin_peril_ids, intensity_bins = get_intensity_bin_dict(os.path.join(run_dir, 'static'))
        else:
            intensity_bin_peril_ids = np.empty(0, dtype=np.int32)
            intensity_bins = np.empty((0, 0), dtype=np.int32)
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

        # Pre-allocate per-event footprint arrays (reused across events, no per-event allocation)
        n_max_areaperils = len(item_map_ja_areaperil_ids)
        fp_areaperil_ids = np.empty(n_max_areaperils, dtype=areaperil_int)
        fp_event_rps = np.empty(n_max_areaperils, dtype=np.int32)
        fp_haz_arr_ptr = np.empty(n_max_areaperils + 1, dtype=np.int64)

        # Pre-allocate CDF cache once (reused across events, no need to zero)
        cached_vuln_cdfs = np.empty((Nvulns_cached, Ndamage_bins_max), dtype=oasis_float)
        cdf_cache_tag = np.full(n_cdf_groups, CDF_CACHE_EMPTY, dtype=np.int64)
        cdf_cache_nbins = np.zeros(Nvulns_cached, dtype=np.int32)

        counter = 0
        timer = time.time()
        ping = kwargs.get('socket_server', 'False') != 'False'
        while True:
            if not streams_in.readinto(event_id_mv):
                if ping:
                    oasis_ping({"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)})
                break

            # get the next event_id from the input stream
            compute_info['event_id'] = event_ids[0]
            event_footprint = event_footprint_obj.get_event(event_ids[0])

            if event_footprint is not None:
                Nhaz_arr_this_event, haz_pdf = process_areaperils_in_footprint(
                    event_footprint,
                    item_map_ja_id_ind,
                    dynamic_footprint,
                    fp_areaperil_ids,
                    fp_event_rps,
                    fp_haz_arr_ptr)
                if Nhaz_arr_this_event == 0:
                    # no items to be computed for this event
                    counter += 1
                    continue

                items_event_data, rng_index, hazard_rng_index, byte_mv = reconstruct_coverages(
                    compute_info,
                    fp_areaperil_ids,
                    Nhaz_arr_this_event,
                    fp_event_rps,
                    item_map_ja_id_ind,
                    item_map_ja_offsets,
                    item_map_ja_vuln_ids,
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
                    hazard_group_seq_rng_index
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

                # Reset CDF cache lookup per event (cached_vuln_cdfs array is reused, no reallocation)
                cdf_cache_tag[:] = CDF_CACHE_EMPTY
                compute_info['cdf_cache_ctr'] = 0

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
                            fp_haz_arr_ptr,
                            vuln_array,
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
                            intensity_bins
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
                oasis_ping({"events_complete": counter, "analysis_pk": kwargs.get("analysis_pk", None)})
                counter = 0

    return 0


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
                         intensity_bins
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

    Returns:
        bool: True if all coverages have been processed, False if the buffer is full and
          the caller should flush and call again.
    """
    cdf_cache_size = nb_int64(cdf_cache_mask + 1)
    haz_cdf_empty = np.empty(vuln_array.shape[2], dtype=oasis_float)
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
            rng_index = item_event_data['rng_index']
            hazard_rng_index = item_event_data['hazard_rng_index']

            item = items[item_event_data['item_idx']]
            if dynamic_footprint is not None:
                intensity_adjustment = item['intensity_adjustment']
                # RP protection: if the item's return period protection exceeds the event's RP, zero loss
                if item_event_data['return_period'] > 0 and item_event_data['event_rp'] < item_event_data['return_period']:
                    losses[:, item_j] = 0
                    continue
            else:
                intensity_adjustment = nb_oasis_int(0)

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
            haz_pdf_prob = haz_pdf_record['probability']

            cdf_group = nb_int64(item_event_data['eff_cdf_id'])
            haz_cdf_prob = pdf_to_cdf(haz_pdf_prob, haz_cdf_empty)
            Nhaz_bins = haz_cdf_prob.shape[0]

            # determine if the CDFs for this CDF group are cached
            stored = cdf_cache_tag[cdf_group]
            do_calc_vuln_ptf = (stored < 0) or (compute_info['cdf_cache_ctr'] - stored >= cdf_cache_size)

            if do_calc_vuln_ptf:  # cache miss — compute and cache CDFs
                # we get the vuln_pdf, needed for effcdf and each cdf
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
                        for j in range(n_sub):
                            vuln_i = areaperil_agg_vuln_idx_ja_data[ptr + j]['vuln_idx']
                            for haz_i in range(Nhaz_bins):
                                vuln_pdf[haz_i] += vuln_array[vuln_i, :, haz_bin_id[haz_i] - 1]
                        vuln_pdf /= n_sub
                else:
                    for haz_i in range(Nhaz_bins):
                        vuln_pdf[haz_i] = vuln_array[item['vulnerability_idx'], :, haz_bin_id[haz_i] - 1]

                # calculate and cache all CDFs as a contiguous block
                eff_damage_cdf = calc_eff_damage_cdf(vuln_pdf, haz_pdf_prob, eff_damage_cdf_empty)
                cdf_cache_tag[cdf_group] = compute_info['cdf_cache_ctr']
                # slot 0: effective damage CDF
                cache_idx = compute_info['cdf_cache_ctr'] & cdf_cache_mask
                cached_vuln_cdfs[cache_idx, :eff_damage_cdf.shape[0]] = eff_damage_cdf
                cdf_cache_nbins[cache_idx] = nb_int32(eff_damage_cdf.shape[0])
                compute_info['cdf_cache_ctr'] += 1

                if not compute_info['effective_damageability']:  # also cache per-bin vuln CDFs
                    haz_i_to_Ndamage_bins = haz_i_to_Ndamage_bins_empty[:Nhaz_bins]
                    haz_i_to_vuln_cdf = haz_i_to_vuln_cdf_empty[:Nhaz_bins]
                    for haz_i in range(Nhaz_bins):
                        haz_i_to_Ndamage_bins[haz_i] = pdf_to_cdf(vuln_pdf[haz_i], haz_i_to_vuln_cdf[haz_i]).shape[0]
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
                    haz_i_to_Ndamage_bins = haz_i_to_Ndamage_bins_empty[:Nhaz_bins]
                    haz_i_to_vuln_cdf = haz_i_to_vuln_cdf_empty[:Nhaz_bins]
                    for haz_i in range(Nhaz_bins):
                        cache_idx = (block_start + 1 + haz_i) & cdf_cache_mask
                        ndamage_bins = cdf_cache_nbins[cache_idx]
                        haz_i_to_Ndamage_bins[haz_i] = ndamage_bins
                        haz_i_to_vuln_cdf[haz_i][:ndamage_bins] = cached_vuln_cdfs[cache_idx, :ndamage_bins]

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

                if item['areaperil_agg_vuln_idx'] < 0:  # single vuln id (non-aggregate)
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
                                    areaperil_id_ind,
                                    dynamic_footprint,
                                    areaperil_ids,
                                    event_rps,
                                    haz_arr_ptr):
    """Process areaperils in the footprint, filtering to those with vulnerability functions.

    Writes into pre-allocated arrays (areaperil_ids, event_rps, haz_arr_ptr) that are
    owned by the caller and reused across events.

    Args:
        event_footprint (np.array[Event or footprint_event_dtype]): footprint entries.
        areaperil_id_ind (np.array): id_index structure for known areaperil_ids.
        dynamic_footprint (boolean): true if there is dynamic_footprint.
        areaperil_ids (np.array[areaperil_int]): pre-allocated output buffer for areaperil ids.
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
                if id_index_get_idx(areaperil_id_ind, last_areaperil_id) != ID_INDEX_NOT_FOUND:
                    # if items with this areaperil_id exist, process and store this areaperil_id
                    areaperil_ids[haz_arr_i] = last_areaperil_id
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
                          areaperil_ids,
                          Nhaz_arr_this_event,
                          event_rps,
                          item_map_ja_id_ind,
                          item_map_ja_offsets,
                          item_map_ja_vuln_ids,
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
                          hazard_group_seq_rng_index):
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
        areaperil_ids (np.array[areaperil_int]): areaperil_ids present in the event footprint
          (from process_areaperils_in_footprint), length >= Nhaz_arr_this_event.
        Nhaz_arr_this_event (int): number of valid entries in areaperil_ids.
        event_rps (np.array[int32]): parallel array of return periods per areaperil (dynamic only).
        item_map_ja_id_ind (np.array): id_index for areaperil_id → dense index.
        item_map_ja_offsets (np.array[oasis_int]): L1 CSR offsets (N_areaperil + 1).
        item_map_ja_vuln_ids (np.array[int32]): vuln_id at each pair position.
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
          used for O(1) group_id to rng_index mapping (reset to -1 each event).
        hazard_group_seq_rng_index (numpy.array[int64]): pre-allocated array of size
          n_unique_haz_groups, for hazard_group_id to rng_index mapping.

    Returns:
        tuple: (items_event_data, rng_index, hazard_rng_index, byte_mv)
          - items_event_data (numpy.array[items_MC_data_type]): per-item data including
            item_idx, haz_arr_i, rng_index, hazard_rng_index, eff_cdf_id.
          - rng_index (int): number of unique damage random seeds generated.
          - hazard_rng_index (int): number of unique hazard random seeds generated.
          - byte_mv (numpy.array[byte]): output buffer, possibly resized.
    """
    # init data structures
    # Reset pre-allocated arrays instead of creating new Numba Dicts
    group_seq_rng_index[:] = -1
    hazard_group_seq_rng_index[:] = -1
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
        areaperil_id = areaperil_ids[ap_i]
        ap_ind = id_index_get_idx(item_map_ja_id_ind, areaperil_id)
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
                if group_seq_rng_index[group_seq_id] == -1:
                    group_seq_rng_index[group_seq_id] = rng_index
                    vuln_seeds[rng_index] = generate_hash(items[item_idx]['group_id'], compute_info['event_id'])
                    this_rng_index = rng_index
                    rng_index += 1
                else:
                    this_rng_index = group_seq_rng_index[group_seq_id]

                hazard_group_seq_id = items[item_idx]['hazard_group_seq_id']
                if hazard_group_seq_rng_index[hazard_group_seq_id] == -1:
                    hazard_group_seq_rng_index[hazard_group_seq_id] = hazard_rng_index
                    haz_seeds[hazard_rng_index] = generate_hash_hazard(items[item_idx]['hazard_group_id'], compute_info['event_id'])
                    this_hazard_rng_index = hazard_rng_index
                    hazard_rng_index += 1
                else:
                    this_hazard_rng_index = hazard_group_seq_rng_index[hazard_group_seq_id]

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
                items_event_data[item_i]['haz_arr_i'] = ap_i
                items_event_data[item_i]['rng_index'] = this_rng_index
                items_event_data[item_i]['hazard_rng_index'] = this_hazard_rng_index
                items_event_data[item_i]['eff_cdf_id'] = item_cdf_group_idx[item_idx]
                if dynamic_footprint is not None:
                    items_event_data[item_i]['intensity_adjustment'] = items[item_idx]['intensity_adjustment']
                    items_event_data[item_i]['return_period'] = items[item_idx]['return_period']
                    items_event_data[item_i]['event_rp'] = event_rps[ap_i]

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
