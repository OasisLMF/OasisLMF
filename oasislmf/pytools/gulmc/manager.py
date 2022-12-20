import os
import sys
import numpy as np
import pandas as pd
import logging
import atexit
from contextlib import ExitStack
from select import select
from pathlib import Path
from numba import njit
from numba.types import (
    uint32 as nb_uint32, int32 as nb_int32, int64 as nb_int64,
    Tuple as nb_Tuple)
from numba.typed import Dict, List

from oasislmf.pytools.common import PIPE_CAPACITY
from oasislmf.pytools.data_layer.oasis_files.correlations import CorrelationsData
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.common import (
    Correlation, Index_type, nb_areaperil_int, oasis_float, Keys)
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.manager import get_damage_bins, Item, get_vulns
from oasislmf.pytools.gul.common import (
    MEAN_IDX, NP_BASE_ARRAY_SIZE, STD_DEV_IDX, TIV_IDX, CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX, NUM_IDX,
    ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE, ITEM_MAP_KEY_TYPE_internal, AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE, ITEM_MAP_VALUE_TYPE_internal,
    items_MC_data_type, gulSampleslevelRec_size, gulSampleslevelHeader_size, coverage_type, gul_header, haz_cdf_type)
from oasislmf.pytools.gul.core import compute_mean_loss, get_gul
from oasislmf.pytools.gul.random import (
    compute_norm_cdf_lookup, compute_norm_inv_cdf_lookup, generate_correlated_hash_vector, generate_hash,
    generate_hash_haz, get_corr_rval, get_random_generator)
from oasislmf.pytools.gul.manager import get_coverages, write_losses
from oasislmf.pytools.gul.utils import append_to_dict_value, binary_search

logger = logging.getLogger(__name__)


@njit(cache=True, fastmath=True)
def generate_item_map(items, coverages):
    """Generate item_map; requires items to be sorted.

    Args:
        items (numpy.ndarray[int32, int32, int32]): 1-d structured array storing
          `item_id`, `coverage_id`, `group_id` for all items.
          items need to be sorted by increasing areaperil_id, vulnerability_id
          in order to output the items in correct order.

    Returns:
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
        areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
          areaperil_id and all the vulnerability ids associated with it.
    """
    item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
    Nitems = items.shape[0]

    areaperil_ids_map = Dict.empty(nb_areaperil_int, Dict.empty(nb_int32, nb_int64))

    for j in range(Nitems):
        append_to_dict_value(
            item_map,
            tuple((items[j]['areaperil_id'], items[j]['vulnerability_id'])),
            tuple((items[j]['id'], items[j]['coverage_id'], items[j]['group_id'])),
            ITEM_MAP_VALUE_TYPE
        )
        coverages[items[j]['coverage_id']]['max_items'] += 1

        if items[j]['areaperil_id'] not in areaperil_ids_map:
            areaperil_ids_map[items[j]['areaperil_id']] = {items[j]['vulnerability_id']: 0}
        else:
            areaperil_ids_map[items[j]['areaperil_id']][items[j]['vulnerability_id']] = 0

    return item_map, areaperil_ids_map


@njit(cache=True)
def process_items(items, agg_vuln_to_vulns, valid_area_peril_id):
    """
    Processes the Items loaded from the file extracting meta data around the vulnerability data.

    Args:
        items: (List[Item]) Data loaded from the vulnerability file
        valid_area_peril_id: array of area_peril_id to be included (if none, all are included)

    Returns: (Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]])
             vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
             areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    areaperil_to_vulns_size = 0
    areaperil_dict = Dict()
    vuln_dict = Dict()
    vuln_idx = 0
    used_agg_vuln_ids = List()
    for i in range(items.shape[0]):
        item = items[i]

        # filter out invalid areaperil id
        if valid_area_peril_id is not None:
            if item['areaperil_id'] not in valid_area_peril_id:
                continue

        # import this vulnerability id if it has not been imported yet
        if item['vulnerability_id'] not in vuln_dict:
            if item['vulnerability_id'] in agg_vuln_to_vulns:
                # vulnerability is aggregate
                for vuln_i in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if vuln_i not in vuln_dict:
                        # import this individual vulnerability_id only if it was not imported already
                        vuln_dict[vuln_i] = np.int32(vuln_idx)
                        vuln_idx += 1

                used_agg_vuln_ids.append(item['vulnerability_id'])

            else:
                # vulnerability is not aggregate
                vuln_dict[item['vulnerability_id']] = np.int32(vuln_idx)
                vuln_idx += 1

        # insert an area dictionary into areaperil_dict under the key of areaperil ID
        if item['areaperil_id'] not in areaperil_dict:
            area_vuln = Dict()

            if item['vulnerability_id'] in agg_vuln_to_vulns:
                for vuln_i in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if vuln_i not in area_vuln:
                        area_vuln[vuln_i] = 0
                        areaperil_to_vulns_size += 1
            else:
                area_vuln[item['vulnerability_id']] = 0
                areaperil_to_vulns_size += 1

            areaperil_dict[item['areaperil_id']] = area_vuln
        else:
            if item['vulnerability_id'] in agg_vuln_to_vulns:
                for vuln_i in agg_vuln_to_vulns[item['vulnerability_id']]:
                    if vuln_i not in areaperil_dict[item['areaperil_id']]:
                        # import this individual vulnerability_id only if it was not imported already
                        areaperil_to_vulns_size += 1
                        areaperil_dict[item['areaperil_id']][vuln_i] = 0
            else:
                if item['vulnerability_id'] not in areaperil_dict[item['areaperil_id']]:
                    areaperil_to_vulns_size += 1
                    areaperil_dict[item['areaperil_id']][item['vulnerability_id']] = 0

    areaperil_to_vulns_idx_dict = Dict()
    areaperil_to_vulns_idx_array = np.empty(len(areaperil_dict), dtype=Index_type)
    areaperil_to_vulns = np.empty(areaperil_to_vulns_size, dtype=np.int32)

    areaperil_i = 0
    vulnerability_i = 0

    for areaperil_id, vulns in areaperil_dict.items():
        areaperil_to_vulns_idx_dict[areaperil_id] = areaperil_i
        areaperil_to_vulns_idx_array[areaperil_i]['start'] = vulnerability_i

        # MT: vulns is a dict and therefore vuln_id are the keys, which are item['vulnerability_id'] stored with
        # areaperil_dict[item['areaperil_id']][item['vulnerability_id']] = 0

        for vuln_id in sorted(vulns):  # sorted is not necessary but doesn't impede the perf and align with cpp getmodel
            areaperil_to_vulns[vulnerability_i] = vuln_id
            vulnerability_i += 1
        areaperil_to_vulns_idx_array[areaperil_i]['end'] = vulnerability_i
        areaperil_i += 1

    return vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_dict, used_agg_vuln_ids


def read_items(input_path, ignore_file_type=set()):
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
        items = np.memmap(items_fname, dtype=Item, mode='r')

    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.loadtxt(items_fname, dtype=Item, delimiter=",", skiprows=1, ndmin=1)

    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items


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
        effective_damageability (bool): if True, it uses effective damageability to draw damage samples instead of
          using the full monte carlo approach (i.e., to draw hazard intensity first, then damage).

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

        # read and store aggregate vulnerability definitions and weights
        # TODO: reorganize this section by defining a read_vulnerability_weights nb function

        @njit(cache=True)
        def gen_empty_agg_vuln_to_vulns():
            return Dict.empty(nb_int32, List.empty_list(nb_int32))

        @njit(cache=True)
        def gen_empty_ap_vuln_weights():
            return Dict.empty(AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE)

        agg_vuln_to_vulns = gen_empty_agg_vuln_to_vulns()
        try:
            # vulnerability_id and aggregate_vulnerability_id are remapped to the internal ids
            # using the vulnd_dict map that contains only the vulnerability_id used in this portfolio.
            d = pd.read_csv(os.path.join(static_path, 'aggregate_vulnerability.csv'))
            d.astype({'aggregate_vulnerability_id': 'int32'})

            # TODO: problem: vulnerability_id that are used only as part of aggregate vulnerability_id will not be present in vuln_dict
            # read all aggregate vulnerability_id without filtering out those that are not used in this portfolio
            # at this stage we read all these definitions, then we read the items and store only the individual vulnerability functions
            # that are used
            for agg, grp in d.groupby('aggregate_vulnerability_id'):
                agg_idx = nb_int32(agg)

                if agg_idx not in agg_vuln_to_vulns:
                    agg_vuln_to_vulns[agg_idx] = List.empty_list(nb_int32)

                for entry in grp['vulnerability_id'].to_list():
                    agg_vuln_to_vulns[agg_idx].append(nb_int32(entry))

        except FileNotFoundError:
            pass

        ap_vuln_weights = gen_empty_ap_vuln_weights()
        if len(agg_vuln_to_vulns) > 0:
            # at least one aggregate vulnerability is defined
            try:
                # TODO: in principle we should filter and add to ap_vuln_weights only if areaperil_id and vulnerability_
                #       are used in this portfolio. Since the reader of weights.csv will change, I don't implement it now.
                d2 = pd.read_csv(os.path.join(static_path, 'weights.csv'))
                for agg, grp in d2.groupby(['areaperil_id', 'vulnerability_id']):
                    ap_vuln_weights[(nb_uint32(agg[0]), nb_int32(agg[1]))] = nb_int32(grp['weight'].to_list()[0])

            except FileNotFoundError:
                pass

        # get agg vuln table
        # vuln_weights = get_vulnerability_weights(input_path, ignore_file_type)

        logger.debug('init items')
        vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_dict, used_agg_vuln_ids = process_items(
            items, agg_vuln_to_vulns, valid_area_peril_id)

        # make agg_vuln_to_vulns_idx
        @njit(cache=True)
        def make_agg_vuln_to_vulns_idx(used_agg_vuln_ids, agg_vuln_to_vulns, vuln_dict):
            """Make map from aggregate vulnerability id to the list of sub-vulnerability ids of which they are composed, where the
            value of the sub-vulnerability id is the internal pointer to the dense array where they are stored.
            """
            agg_vuln_to_vulns_idx = Dict.empty(nb_int32, List.empty_list(nb_int32))
            if len(used_agg_vuln_ids) > 0:
                # at least one aggregate vulnerability is used
                for agg in used_agg_vuln_ids:
                    agg_vuln_to_vulns_idx[agg] = List([vuln_dict[vuln] for vuln in agg_vuln_to_vulns[agg]])
            # else:
            #     agg_vuln_to_vulns_idx[nb_int32(0)] = List([nb_int32(0)])

            return agg_vuln_to_vulns_idx

        agg_vuln_to_vulns_idx = make_agg_vuln_to_vulns_idx(used_agg_vuln_ids, agg_vuln_to_vulns, vuln_dict)

        # make ap_vuln_weights
        @njit(cache=True)
        def make_ap_vuln_idx_weights(areaperil_dict, ap_vuln_weights, vuln_dict):
            """Make map from aggregate vulnerability id to the list of sub-vulnerability ids of which they are composed, where the
            value of the sub-vulnerability id is the internal pointer to the dense array where they are stored.
            """
            ap_vuln_idx_weights = {}
            for ap in areaperil_dict:
                for vuln in areaperil_dict[ap]:
                    ap_vuln_idx_weights[tuple((ap, vuln_dict[vuln]))] = ap_vuln_weights[tuple((ap, vuln))]

            return ap_vuln_idx_weights

        if len(agg_vuln_to_vulns_idx) > 0:
            # at least one aggregate vulnerability is used
            ap_vuln_idx_weights = make_ap_vuln_idx_weights(areaperil_dict, ap_vuln_weights, vuln_dict)
        else:
            ap_vuln_idx_weights = Dict.empty(AGG_VULN_WEIGHTS_KEY_TYPE, AGG_VULN_WEIGHTS_VAL_TYPE)

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
        vuln_prob_to = np.zeros(Ndamage_bins_max, dtype=oasis_float)
        weighted_vuln_to_empty = np.zeros(Ndamage_bins_max, dtype=oasis_float)

        # maximum bytes to be written in the output stream for 1 item
        max_bytes_per_item = (sample_size + NUM_IDX + 1) * gulSampleslevelRec_size + 2 * gulSampleslevelHeader_size

        VULN_LOOKUP_KEY_TYPE = nb_Tuple((nb_int32, nb_int32))
        VULN_LOOKUP_VALUE_TYPE = nb_Tuple((nb_int32, nb_int32))

        @njit(cache=True)
        def gen_empty_vuln_cdf_lookup(list_size):
            cached_vuln_cdf_lookup = Dict.empty(VULN_LOOKUP_KEY_TYPE, VULN_LOOKUP_VALUE_TYPE)
            cached_vuln_cdf_lookup_keys = List.empty_list(VULN_LOOKUP_VALUE_TYPE)
            dummy = tuple((nb_int32(-1), nb_int32(-1)))
            for _ in range(list_size):
                cached_vuln_cdf_lookup_keys.append(dummy)

            return cached_vuln_cdf_lookup, cached_vuln_cdf_lookup_keys

        # define vulnerability cdf cache size
        max_cached_vuln_cdf_size_MB = 200  # cache size in MB
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

                areaperil_ids, Nhaz_cdf_event, areaperil_to_haz_cdf, haz_cdf, haz_cdf_ptr, eff_vuln_cdf, areaperil_to_eff_vuln_cdf = process_areaperils_in_footprint(
                    event_footprint, vuln_array, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array)

                if Nhaz_cdf_event == 0:
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
                        agg_vuln_to_vulns, agg_vuln_to_vulns_idx, vuln_dict, ap_vuln_idx_weights,
                        loss_threshold, losses, vuln_prob_to, weighted_vuln_to_empty, alloc_rule, do_correlation, haz_rndms_base, vuln_rndms_base,
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
                         cached_vuln_cdf_lookup, lookup_keys, next_cached_vuln_cdf,
                         cached_vuln_cdfs,
                         agg_vuln_to_vulns,
                         agg_vuln_to_vulns_idx,
                         vuln_dict,
                         ap_vuln_idx_weights,
                         loss_threshold,
                         losses,
                         vuln_cdf_empty,
                         weighted_vuln_to_empty,
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

    TODO: update docs

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
        eff_vuln_cdf (np.array[oasis_float]): effective damageability cdf.
        vuln_array (np.array[float]): damage pdf for different vulnerability functions, as a function of hazard intensity.
        damage_bins (List[Union[damagebindictionaryCsv, damagebindictionary]]): loaded data from the damage_bin_dict file.
        Ndamage_bins_max (int): maximum number of damage bins.
        loss_threshold (float): threshold above which losses are printed to the output stream.
        losses (numpy.array[oasis_float]): array (to be re-used) to store losses for each item.
        vuln_prob_to (np.array[oasis_float]): array (to be re-used) to store the damage cdf for each item.
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
            hazcdf_i = item['hazcdf_i']

            # get the hazard cdf
            haz_prob_to = haz_cdf[haz_cdf_ptr[hazcdf_i]:haz_cdf_ptr[hazcdf_i + 1]]['probability']
            Nhaz_bins = haz_cdf_ptr[hazcdf_i + 1] - haz_cdf_ptr[hazcdf_i]

            # if aggregate: agg_eff_vuln_cdf needs to be computed
            if vulnerability_id in agg_vuln_to_vulns:
                # aggregate case
                weighted_vuln_to = weighted_vuln_to_empty
                tot_weights = 0.
                agg_vulns_idx = agg_vuln_to_vulns_idx[vulnerability_id]

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
                    weight = np.float64(ap_vuln_idx_weights[(areaperil_id, vuln_i)])

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
                        "Impossible to compute the cdf of the an aggregate vulnerability_id because individual weights are all zero.")

                weighted_vuln_to[damage_bin_i] = cumsum / tot_weights

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

                        weighted_vuln_to[damage_bin_i] = cumsum / tot_weights
                        damage_bin_i += 1

                        if weighted_vuln_to[damage_bin_i - 1] > 0.999999940:
                            break

                Ndamage_bins = damage_bin_i
                eff_damag_cdf_Ndamage_bins = Ndamage_bins
                eff_damag_cdf = weighted_vuln_to[:eff_damag_cdf_Ndamage_bins]

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
                        vuln_prob_to = eff_damag_cdf
                        Ndamage_bins = eff_damag_cdf_Ndamage_bins

                    else:
                        # use the full monte carlo approach: draw samples from the hazard intensity distribution first

                        # 1) get the intensity bin
                        if Nhaz_bins == 1:
                            # if hazard intensity has no uncertainty, there is no need to sample
                            haz_bin_idx = nb_int32(0)

                        else:
                            # if hazard intensity has a probability distribution, sample it

                            # cap `haz_rval` to the maximum `haz_prob_to` value (which should be 1.)
                            haz_rval = haz_rndms[rng_index][sample_idx - 1]

                            if debug == 1:
                                # store the random value used for the hazard intensity sampling instead of the loss
                                losses[sample_idx, item_i] = haz_rval
                                continue

                            if haz_rval >= haz_prob_to[Nhaz_bins - 1]:
                                haz_bin_idx = nb_int32(Nhaz_bins - 1)
                            else:
                                # find the bin in which the random value `haz_rval` falls into
                                haz_bin_idx = nb_int32(binary_search(haz_rval, haz_prob_to, Nhaz_bins))

                        # 2) get the hazard intensity bin id
                        haz_int_bin_id = haz_cdf[haz_cdf_ptr[hazcdf_i] + haz_bin_idx]['intensity_bin_id']

                        # 3) get the vulnerability cdf
                        if vulnerability_id in agg_vuln_to_vulns:
                            # aggregate case
                            agg_vulns_idx = agg_vuln_to_vulns_idx[vulnerability_id]
                            weighted_vuln_to = weighted_vuln_to_empty

                            # cache the weights and compute the total weights
                            tot_weights = 0.
                            used_weights = []
                            for j, vuln_i in enumerate(agg_vulns_idx):
                                weight = np.float64(ap_vuln_idx_weights[(areaperil_id, vuln_i)])
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
                                    "Impossible to compute the cdf of the an aggregate vulnerability_id because individual weights are all zero.")

                            # compute the weighted cdf
                            damage_bin_i = nb_int32(0)
                            cumsum = 0.
                            while damage_bin_i < Ndamage_bins_max:
                                for j, vuln_i in enumerate(agg_vulns_idx):
                                    cumsum += vuln_array[vuln_i, damage_bin_i, haz_int_bin_id - 1] * used_weights[j]

                                weighted_vuln_to[damage_bin_i] = cumsum / tot_weights
                                damage_bin_i += 1

                                if weighted_vuln_to[damage_bin_i - 1] > 0.999999940:
                                    break

                            Ndamage_bins = damage_bin_i
                            vuln_prob_to = weighted_vuln_to[:Ndamage_bins]

                        else:
                            # non-aggregate case
                            vuln_i = vuln_dict[vulnerability_id]
                            vuln_prob_to, Ndamage_bins, next_cached_vuln_cdf = get_vuln_cdf(
                                vuln_i, haz_bin_idx, haz_int_bin_id, cached_vuln_cdf_lookup, vuln_array, vuln_cdf_empty,
                                Ndamage_bins_max, cached_vuln_cdfs, lookup_keys, next_cached_vuln_cdf)

                    # draw samples of damage from the vulnerability function
                    vuln_rval = vuln_rndms[sample_idx - 1]

                    if debug == 2:
                        # store the random value used for the damage sampling instead of the loss
                        losses[sample_idx, item_i] = vuln_rval
                        continue

                    # cap `vuln_rval` to the maximum `vuln_prob_to` value (which should be 1.)
                    if vuln_rval >= vuln_prob_to[Ndamage_bins - 1]:
                        vuln_rval = vuln_prob_to[Ndamage_bins - 1] - 0.00000003
                        vuln_bin_idx = Ndamage_bins - 1
                    else:
                        # find the bin in which the random value `vuln_rval` falls into
                        vuln_bin_idx = binary_search(vuln_rval, vuln_prob_to, Ndamage_bins)

                    # compute ground-up losses
                    gul = get_gul(
                        damage_bins['bin_from'][vuln_bin_idx],
                        damage_bins['bin_to'][vuln_bin_idx],
                        damage_bins['interpolation'][vuln_bin_idx],
                        vuln_prob_to[vuln_bin_idx - 1] * (vuln_bin_idx > 0),
                        vuln_prob_to[vuln_bin_idx],
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
def get_vuln_cdf(vuln_i, haz_bin_idx, haz_int_bin_id, cached_vuln_cdf_lookup, vuln_array, vuln_cdf_empty,
                 Ndamage_bins_max, cached_vuln_cdfs, lookup_keys, next_cached_vuln_cdf):
    """TODO write docstring

    Args:
        vuln_i (_type_): _description_
        haz_bin_idx (_type_): _description_
        haz_int_bin_id (_type_): _description_
        cached_vuln_cdf_lookup (_type_): _description_
        vuln_array (_type_): _description_
        vuln_cdf_empty (_type_): _description_
        Ndamage_bins_max (_type_): _description_
        cached_vuln_cdfs (_type_): _description_
        lookup_keys (_type_): _description_
        next_cached_vuln_cdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    lookup_key = tuple((vuln_i, haz_bin_idx))
    if lookup_key in cached_vuln_cdf_lookup:
        # cdf is cached
        start, Ndamage_bins = cached_vuln_cdf_lookup[lookup_key]
        vuln_prob_to = cached_vuln_cdfs[start, :Ndamage_bins]

    else:
        # cdf has to be computed
        vuln_prob_to = vuln_cdf_empty
        damage_bin_i = 0
        cumsum = 0
        while damage_bin_i < Ndamage_bins_max:
            cumsum += vuln_array[vuln_i, damage_bin_i, haz_int_bin_id - 1]
            vuln_prob_to[damage_bin_i] = cumsum
            damage_bin_i += 1

            if cumsum > 0.999999940:
                break

        Ndamage_bins = damage_bin_i

        if lookup_keys[next_cached_vuln_cdf] in cached_vuln_cdf_lookup:
            # overwrite cache
            cached_vuln_cdf_lookup.pop(lookup_keys[next_cached_vuln_cdf])

        # cache the cdf
        cached_vuln_cdfs[next_cached_vuln_cdf, :Ndamage_bins] = vuln_prob_to[:Ndamage_bins]
        cached_vuln_cdf_lookup[lookup_key] = tuple((nb_int32(next_cached_vuln_cdf), nb_int32(Ndamage_bins)))
        lookup_keys[next_cached_vuln_cdf] = lookup_key
        next_cached_vuln_cdf += 1
        next_cached_vuln_cdf %= cached_vuln_cdfs.shape[0]

    return vuln_prob_to, Ndamage_bins, next_cached_vuln_cdf


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
        Nhaz_cdf_event (int): number of hazard cdf stored for this event. If zero, it means no items have losses in such event.
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

    return (areaperil_ids,
            haz_cdf_i,
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
        .joinpath("assets").joinpath("test_model_1")

    run(
        run_dir=test_dir,
        ignore_file_type=set(),
        file_in=test_dir.joinpath("input").joinpath('events.bin'),
        file_out=test_dir.joinpath('gulpy_mc.bin'),
        sample_size=10,
        loss_threshold=0.,
        alloc_rule=1,
        debug=0,
        random_generator=1,
        ignore_correlation=True,
        effective_damageability=False,
    )
