import os
from contextlib import ExitStack
from random import sample
from reprlib import aRepr
import sys
from tracemalloc import start
import numpy as np
import logging
import pandas as pd
import atexit
from numba.types import int32 as nb_int32, int64 as nb_int64, int8 as nb_int8


from oasislmf.pytools.gul.io import gen_structs, gen_valid_area_peril
from oasislmf.pytools.gul.random import generate_hash
from oasislmf.pytools.getmodel.common import oasis_float
from oasislmf.pytools.getmodel.manager import areaperil_int_relative_size, results_relative_size, buff_size
from oasislmf.pytools.data_layer.footprint_layer import FootprintLayerClient
from oasislmf.pytools.getmodel.manager import get_damage_bins, Item, get_mean_damage_bins
from oasislmf.pytools.getmodel.manager import get_items, get_vulns, convert_vuln_id_to_index
from oasislmf.pytools.getmodel.footprint import Footprint
from oasislmf.pytools.getmodel.common import Keys
from oasislmf.pytools.gul.manager import get_coverages, gul_get_items, generate_item_map
from oasislmf.pytools.gul.random import get_random_generator
from oasislmf.pytools.gul.utils import binary_search
from oasislmf.pytools.gul.common import (
    MEAN_IDX, STD_DEV_IDX, TIV_IDX, CHANCE_OF_LOSS_IDX, MAX_LOSS_IDX, NUM_IDX,
    ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE, GULPY_STREAM_BUFF_SIZE_WRITE,
    gulSampleslevelRec_size, gulSampleslevelHeader_size, coverage_type, gul_header,
)

logger = logging.getLogger(__name__)


def func(vuln_array, areaperil_to_vulns_idx_array, areaperil_to_vulns_idx_dict, sample_size=1, random_generator=1):
    """for one event id"""
    generate_rndm = get_random_generator(random_generator)

    # areaperil_id = [154]
    # areaperil_id = [1]
    probability = np.array([0.3, 0.5, 0.2])
    haz_Nbins = len(probability)
    haz_prob_to = np.cumsum(probability)
    assert haz_prob_to[-1] == 1.

    # SMART thing:
    # if haz_prob_to has 1 bin, i.e. it has no uncertainty, no need to sample
    # --> this makes the code able to treat effective-damageability case

    # sample hazard intensity
    haz_seed = [123456]
    haz_rval = generate_rndm(haz_seed, sample_size)
    haz_rval = 0.55
    haz_bin_idx = binary_search(haz_rval, haz_prob_to, haz_Nbins)

    # get vulnerability (detailed or from aggregate)

    # sample vulnerability
    vuln_seed = [123456]
    vuln_rval = generate_rndm(vuln_seed, sample_size)
    vuln_rval = 0.59
    # areaperil_to_vulns_idx = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[areaperil_id[0]]]['start']
    # vuln_id = areaperil_to_vulns_idx
    # vuln_damage_prob = vuln_array[areaperil_to_vulns_idx, :, haz_bin_idx]

    # mock example
    vuln_array_mock = np.array(
        [[[0.8, 0.6, 0.3], [0.2, 0.4, 0.7]],
         [[1., 0.75, 0.55], [0., 0.25, 0.45]]],
    )
    vuln_array = vuln_array_mock

    Nvuln, Ndamage_bins, Nintensity_bins = vuln_array.shape

    vuln_id = 0
    vuln_damage_prob = vuln_array[vuln_id, :, haz_bin_idx]
    vuln_damage_prob_to = np.cumsum(vuln_damage_prob)
    vuln_Nbins = Ndamage_bins
    vuln_bin_idx = binary_search(vuln_rval, vuln_damage_prob_to, vuln_Nbins)
    # get damage value from vuln_bin_idx

    pass


def run(run_dir, ignore_file_type, sample_size, loss_threshold, alloc_rule, debug,
        random_generator, peril_filter=[], file_in=None, file_out=None, data_server=None, **kwargs):

    logger.info("starting gulpy")

    # TODO: store static_path in a paraparameters file
    static_path = os.path.join(run_dir, 'static')
    # TODO: store input_path in a paraparameters file
    input_path = os.path.join(run_dir, 'input')
    ignore_file_type = set(ignore_file_type)

    # load keys.csv to determine included AreaPerilID from peril_filter
    if peril_filter:
        keys_df = pd.read_csv(os.path.join(input_path, 'keys.csv'), dtype=Keys)
        valid_area_peril_id = keys_df.loc[keys_df['PerilID'].isin(peril_filter), 'AreaPerilID'].to_numpy()
        logger.debug(
            f'Peril specific run: ({peril_filter}), {len(valid_area_peril_id)} AreaPerilID included out of {len(keys_df)}')
    else:
        valid_area_peril_id = None

    # beginning of of gulpy prep
    damage_bins = get_damage_bins(static_path)

    # read coverages from file
    coverages_tiv = get_coverages(input_path)

    # init the structure for computation
    # coverages are numbered from 1, therefore we skip element 0 in `coverages`
    coverages = np.zeros(coverages_tiv.shape[0] + 1, coverage_type)
    coverages[1:]['tiv'] = coverages_tiv
    del coverages_tiv

    items = gul_get_items(input_path)

    # in-place sort items in order to store them in item_map in the desired order
    # currently numba only supports a simple call to np.sort() with no `order` keyword,
    # so we do the sort here.
    items = np.sort(items, order=['areaperil_id', 'vulnerability_id'])
    item_map = generate_item_map(items, coverages)

    # init array to store the coverages to be computed
    # coverages are numebered from 1, therefore skip element 0.
    compute = np.zeros(coverages.shape[0] + 1, items.dtype['coverage_id'])
    # end of gulpy prep

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

        logger.debug('init items')
        vuln_dict, areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns = get_items(
            input_path, ignore_file_type, valid_area_peril_id)

        logger.debug('init footprint')
        footprint_obj = stack.enter_context(Footprint.load(static_path, ignore_file_type))

        if data_server:
            num_intensity_bins: int = FootprintLayerClient.get_number_of_intensity_bins()
            logger.info(f"got {num_intensity_bins} intensity bins from server")
        else:
            num_intensity_bins: int = footprint_obj.num_intensity_bins

        logger.debug('init vulnerability')

        vuln_array, vulns_id, num_damage_bins = get_vulns(static_path, vuln_dict, num_intensity_bins, ignore_file_type)
        # # get agg vuln table
        # vuln_weights = get_vulnerability_weights(static_path, ignore_file_type)

        convert_vuln_id_to_index(vuln_dict, areaperil_to_vulns)
        logger.debug('init mean_damage_bins')
        mean_damage_bins = get_mean_damage_bins(static_path, ignore_file_type)

        # GETMODEL OUT STREAM even_id, areaperil_id, vulnerability_id, num_result, [oasis_float] * num_result
        # max_result_relative_size = 1 + + areaperil_int_relative_size + 1 + 1 + num_damage_bins * results_relative_size

        # mv = memoryview(bytearray(buff_size))

        # int32_mv_OUT = np.ndarray(buff_size // np.int32().itemsize, buffer=mv, dtype=np.int32)

        # header
        # stream_out.write(np.uint32(1).tobytes())

        # GULPY one-time prep
        # set up streams
        if file_out is None or file_out == '-':
            stream_out = sys.stdout.buffer
        else:
            stream_out = stack.enter_context(open(file_out, 'wb'))

        select_stream_list = [stream_out]

        # prepare output buffer, write stream header
        stream_out.write(gul_header)
        stream_out.write(np.int32(sample_size).tobytes())

        # number of bytes to read at a given time.
        number_size = max(gulSampleslevelHeader_size, gulSampleslevelRec_size)

        # define the raw memory view, the int32 view of it, and their respective cursors
        mv_size_bytes = GULPY_STREAM_BUFF_SIZE_WRITE * 2
        mv_write = memoryview(bytearray(mv_size_bytes))
        int32_mv_write = np.ndarray(mv_size_bytes // number_size, buffer=mv_write, dtype='i4')

        # set the random generator function
        generate_rndm = get_random_generator(random_generator)

        if alloc_rule not in [0, 1, 2, 3]:
            raise ValueError(f"Expect alloc_rule to be 0, 1, 2, or 3, got {alloc_rule}")

        cursor = 0
        cursor_bytes = 0

        # create the array to store the seeds
        seeds = np.zeros(len(np.unique(items['group_id'])), dtype=Item.dtype['group_id'])

        from oasislmf.pytools.getmodel.common import oasis_float

        # create buffer to be reused to store all losses for one coverage
        losses_buffer = np.zeros(
            (sample_size + NUM_IDX + 1, np.max(coverages[1:]['max_items'])), dtype=oasis_float)

        while True:
            len_read = streams_in.readinto(event_id_mv)
            if len_read == 0:
                break

            event_id = event_ids[0]

            if data_server:
                event_footprint = FootprintLayerClient.get_event(event_id)
            else:
                event_footprint = footprint_obj.get_event(event_id)

            if event_footprint is not None:
                # load event
                # for cursor_bytes in doCdf(event_id,
                #       num_intensity_bins, event_footprint,
                #       areaperil_to_vulns_idx_dict, areaperil_to_vulns_idx_array, areaperil_to_vulns,
                #       vuln_array, vulns_id, num_damage_bins, mean_damage_bins,
                #                           int32_mv, max_result_relative_size):
                haz_recs, haz_idx_ptr, haz_rng_index, areaperil_ids, vulnerability_ids = gulpy_preprocess_event(
                    haz_seeds)

                # reconstruct coverages and store vulnerability funcs as cdfs
                haz_rndms = generate_rndm(haz_seeds[:haz_rng_index], sample_size)
                compute_i, items_data = reconstruct_coverages(item_map, coverages, compute, seeds, haz_rndms)

                # gulpy

                rndms = generate_rndm(seeds[:rng_index], sample_size)

                last_processed_coverage_ids_idx = 0
                while last_processed_coverage_ids_idx < compute_i:
                    cursor, cursor_bytes, last_processed_coverage_ids_idx = compute_event_losses(
                        event_id, coverages, compute[:compute_i], items_data,
                        last_processed_coverage_ids_idx, sample_size, recs, rec_idx_ptr,
                        damage_bins, loss_threshold, losses_buffer, alloc_rule, rndms, debug,
                        GULPY_STREAM_BUFF_SIZE_WRITE, int32_mv_write, cursor
                    )

                    select([], select_stream_list, select_stream_list)
                    stream_out.write(mv_write[:cursor_bytes])
                    cursor = 0

                logger.info(f"event {event_id} DONE")


def reconstruct_coverages(areaperil_ids, vulnerability_ids, item_map, coverages, compute, seeds,):

    # reconstruct coverage: probably best outsite of this function
    # register the items to their coverage

    for k in range(len(areaperil_ids)):

        item_key = tuple((areaperil_ids[k], vulnerability_ids[k]))

        # draw hazard intensity samples to determine the vulnerability_id slice
        haz_rval = 0.55
        haz_bin_idx = binary_search(haz_rval, haz_prob_to, haz_Nbins)

        vuln_damage_prob = vuln_array[vuln_id, :, haz_bin_idx]
        vuln_damage_prob_to = np.cumsum(vuln_damage_prob)  # calculate cdf
        vuln_Nbins = len(vuln_damage_prob)

        # read damage cdf bins
        start_rec = last_rec_idx_ptr
        end_rec = start_rec + Nbins_to_read
        for j in range(start_rec, end_rec, 1):
            rec[j]['prob_to'] = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0] < - vulnerability function slice
            # cursor += oasis_float_to_int32_size

            rec[j]['bin_mean'] = int32_mv[cursor: cursor + oasis_float_to_int32_size].view(oasis_float)[0] < - read from file
            # cursor += oasis_float_to_int32_size

        rec_idx_ptr.append(rec_idx_ptr[-1] + Nbins_to_read)
        last_rec_idx_ptr = end_rec
        rec_valid_len += Nbins_to_read

        for item in item_map[item_key]:
            item_id, coverage_id, group_id = item

            # if this group_id was not seen yet, process it.
            # it assumes that hash only depends on event_id and group_id
            # and that only 1 event_id is processed at a time.
            if group_id not in group_id_rng_index:
                group_id_rng_index[group_id] = rng_index
                seeds[rng_index] = generate_hash(group_id, last_event_id)
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
            items_data[item_i]['damagecdf_i'] = k
            items_data[item_i]['rng_index'] = this_rng_index

            coverage['cur_items'] += 1


def read_event_footprint():
        event_id_mv = memoryview(bytearray(4))
        event_ids = np.ndarray(1, buffer=event_id_mv, dtype='i4')
    pass


def gulpy_preprocess_event(event_id, event_footprint, coverages, item_map, compute, seeds, valid_area_peril_id=None):


    if valid_area_peril_id is not None:
        valid_area_peril_dict = gen_valid_area_peril(valid_area_peril_id)
    else:
        valid_area_peril_dict = None

    # init data structures
    group_id_rng_index, rec_idx_ptr = gen_structs()
    rng_index = 0
    damagecdf_i = 0
    compute_i = 0
    items_data_i = 0
    coverages['cur_items'].fill(0)
    recs = []

    areaperil_ids = [] # Dict.empty(nb_int32, nb_int64)
    vulnerability_ids = []

    # challenge here is that 1 footprint entry contains
    # event_id areaperil_id intensity_bin prob
    footprint_i = 0
    last_areaperil_id = 0
    prob = []
    areaperil_id_idx = 0

    # init a counter for the local `rec` array
    last_rec_idx_ptr = 0
    rec_valid_len = 0

    prob_rec = np.zeros(1000, dtype=oasis_float)

    while footprint_i < len(event_footprint):
        
        areaperil_id = event_footprint[footprint_i]['areaperil_id']
       
        if areaperil_id != last_areaperil_id:
            if last_areaperil_id > 0:
                # one areaperil_id is completed
                areaperil_ids.append(last_areaperil_id)

                # mapping to get the prob back, e.g.
                # for desired areaperil_id, the intensity prob is
                # prob_rec[]
                # if areaperil_id not in areaperil_ids:
                #     areaperil_ids[areaperil_id] = areaperil_id_idx
                #     areaperil_id_idx += 1

                # if this group_id was not seen yet, process it.
                # it assumes that hash only depends on event_id and group_id
                # and that only 1 event_id is processed at a time.
                if areaperil_id not in haz_rng_index:
                    haz_rng_index[areaperil_id] = haz_rng_index
                    haz_seeds[haz_rng_index] = generate_hash(areaperil_id, event_id)
                    this_haz_rng_index = haz_rng_index
                    haz_rng_index += 1

                else:
                    this_rng_index = group_id_rng_index[group_id]
                    
                # read hazard intensity pdf
                start_rec = last_rec_idx_ptr
                Nbins_to_read = len(prob)
                end_rec = start_rec + Nbins_to_read
                if end_rec > prob_rec.shape[0]:
                    # double its size
                    pass

                for j in range(start_rec, end_rec, 1):
                    prob_rec[j] = prob[j]

                rec_idx_ptr.append(rec_idx_ptr[-1] + Nbins_to_read)
                last_rec_idx_ptr = end_rec
                rec_valid_len += Nbins_to_read

                # one possibility would be to draw the haz intensity random samples here
                # but it is not efficient, probably, as we want to draw all the samples for all seeds
                # so we need to return the seeds

                vulnerability_ids = areaperil_to_vulns_idx_array[areaperil_to_vulns_idx_dict[last_areaperil_id]]
                assert vulnerability_ids['start'] == vulnerability_ids['end']
                vulnerability_id = vulnerability_ids['start']
                vulnerability_ids.append(vulnerability_id)

                # clear the prob array
                prob = []
            
            last_areaperil_id = areaperil_id

        # QUESTION for JOH/Ben: how are the hazard intensity bins provided?
        # is there a maximum number of hazard intensity bins that is set and can be read? -> this would allow avoiding .append() and size a reusable probto
        prob.append(event_footprint[footprint_i]['probability'])



        footprint_i += 1


    return areaperil_ids, vulnerability_ids, 

if __name__ == '__main__':

    run(
        run_dir="/home/mtazzari/repos/OasisPiWind/runs/losses-20220824044200",
        ignore_file_type=set(),
        sample_size=1,
        loss_threshold=0.,
        alloc_rule=1,
        debug=False,
        random_generator=1
    )
