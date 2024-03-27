"""
This file contains the utilities for all the I/O necessary in gulpy.

"""
from select import select

import numpy as np
from numba import njit
from numba.typed import Dict, List
from numba.types import int8 as nb_int8
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64

from oasislmf.pytools.common.data import areaperil_int, areaperil_int_size, oasis_int, oasis_int_size, oasis_float, oasis_float_size
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY, bytes_to_stream_types, mv_read, CDF_STREAM_ID
from oasislmf.pytools.gul.common import (NP_BASE_ARRAY_SIZE, ProbMean,
                                         ProbMean_size,
                                         damagecdfrec_stream, items_data_type)
from oasislmf.pytools.gul.random import generate_hash


@njit(cache=True)
def gen_structs():
    """Generate some data structures needed for the whole computation.

    Returns:
        Dict(int,int), List: map of group ids to random seeds,
          list storing the index where a specific cdf record starts in the `rec` numpy array.

    """
    group_id_rng_index = Dict.empty(nb_int32, nb_int64)
    rec_idx_ptr = List([0])

    return group_id_rng_index, rec_idx_ptr


@njit(cache=True)
def gen_valid_area_peril(valid_area_peril_id):
    valid_area_peril_dict = Dict()
    for i in range(valid_area_peril_id.shape[0]):
        valid_area_peril_dict[valid_area_peril_id[i]] = nb_int8(0)

    return valid_area_peril_dict


def read_getmodel_stream(stream_in, item_map, coverages, compute, seeds, valid_area_peril_id=None, buff_size=PIPE_CAPACITY):
    """Read the getmodel output stream yielding data event by event.

    Args:
        stream_in (buffer-like): input stream, e.g. `sys.stdin.buffer`.
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
        coverages (numpy.ndarray[coverage_type]): array with coverage data.
        compute (numpy.array[int]): list of coverages to be computed.
        seeds (numpy.array[int]): the random seeds for each coverage_id.
        buff_size (int): size in bytes of the read buffer (see note).
        valid_area_peril_id (list[int]): list of valid areaperil_ids.

    Raises:
        ValueError: If the stream type is not 1.

    Yields:
        int, int, numpy.array[items_data_type], numpy.array[oasis_float], numpy.array[int], int:
          event_id, index of the last coverage_id stored in compute, item-related data,
          cdf records, array with the indices of `rec` where each cdf record starts,
          number of unique random seeds computed so far.

    Note:
        It is advisable to set buff_size as 2x the maximum pipe limit (65536 bytes)
        to ensure that the stream is always read in the biggest possible chunks,
        which nominally is the largest between the pipe limit and the remaining memory
        to fill the memoryview.

    """
    # determine stream type
    stream_source_type, stream_agg_type = bytes_to_stream_types(stream_in.read(4))

    # see https://github.com/OasisLMF/ktools/blob/master/docs/md/CoreComponents.md
    if stream_source_type != CDF_STREAM_ID:
        raise ValueError(f"FATAL: Invalid stream type: expect {CDF_STREAM_ID}, got {stream_source_type}.")

    # maximum number of entries is buff_size divided by the minimum entry size
    # (corresponding to a 1-bin only cdf)
    min_size_cdf_entry = damagecdfrec_stream.size + 4 + ProbMean.size

    # each record from getmodel stream is expected to contain:
    # 1 damagecdfrec_stream obj, 1 int (Nbins), a number `Nbins` of ProbMean objects

    # init the memory view to store the stream
    mv = memoryview(bytearray(buff_size))
    byte_mv = np.frombuffer(buffer=mv, dtype='b')

    valid_buf = 0
    last_event_id = -1
    len_read = 1

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

    items_data = np.empty(2 ** NP_BASE_ARRAY_SIZE, dtype=items_data_type)
    select_stream_list = [stream_in]

    while True:
        if valid_buf < buff_size and len_read:
            select(select_stream_list, [], select_stream_list)

            # read the stream from valid_buf onwards
            len_read = stream_in.readinto1(mv[valid_buf:])
            valid_buf += len_read

        if valid_buf == 0:
            # the stream has ended and all the data has been read
            if last_event_id != -1:
                yield last_event_id, compute_i, items_data, np.concatenate(recs), rec_idx_ptr, rng_index
            break

        # read the streamed data into formatted data
        cursor, yield_event, event_id, rec, rec_idx_ptr, last_event_id, compute_i, items_data_i, items_data, rng_index, group_id_rng_index, damagecdf_i = stream_to_data(
            byte_mv, valid_buf, min_size_cdf_entry, last_event_id, item_map, coverages, valid_area_peril_dict,
            compute_i, compute, items_data_i, items_data, seeds, rng_index, group_id_rng_index,
            damagecdf_i, rec_idx_ptr
        )

        if cursor == 0 and len_read == 0:
            # here valid buff > 0, but not enough data to have a full item => the stream stopped prematurely
            raise Exception(
                f"cdf input stream ended prematurely, event_id: {event_id}, valid_buf: {valid_buf}, buff_size: {buff_size}"
            )

        # persist the cdf records read from the stream
        recs.append(rec)

        if yield_event:
            # event is fully read
            yield last_event_id, compute_i, items_data, np.concatenate(recs), rec_idx_ptr, rng_index

            # init the data structures for the next event
            last_event_id = event_id

            # clear data structures
            group_id_rng_index, rec_idx_ptr = gen_structs()
            rng_index = 0
            damagecdf_i = 0
            compute_i = 0
            items_data_i = 0
            coverages['cur_items'].fill(0)
            recs = []

        # move the un-read data to the beginning of the memoryview
        mv[:valid_buf - cursor] = mv[cursor:valid_buf]

        # update the length of the valid data
        valid_buf -= cursor


@njit(cache=True, fastmath=True)
def stream_to_data(byte_mv, valid_buf, size_cdf_entry, last_event_id, item_map, coverages, valid_area_peril_dict,
                   compute_i, compute, items_data_i, items_data, seeds, rng_index, group_id_rng_index, damagecdf_i, rec_idx_ptr):
    """Parse streamed data into data arrays.

    Args:
        byte_mv (ndarray): byte view of the buffer
        valid_buf (int): number of bytes with valid data
        size_cdf_entry (int): size (in bytes) of a single record
        last_event_id (int): event_id of the last event that was completed
        item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
          the mapping between areaperil_id, vulnerability_id to item.
        coverages (numpy.ndarray[coverage_type]): array with coverage data.
        compute_i (int): index of the last coverage id stored in `compute`.
        compute (numpy.array[int]): list of coverage ids to be computed.
        items_data_i (int): index of the last items_data_i stored in `items_data`.
        items_data (numpy.array[items_data_type]): item-related data.
        seeds (numpy.array[int]): the random seeds for each coverage_id.
        rng_index (int): number of unique random seeds computed so far.
        group_id_rng_index (Dict([int,int])): map of group ids to random seeds.
        damagecdf_i (int): index of the last cdf record that has been read from stream and stored in `rec`.
        rec_idx_ptr (numpy.array[int]): array with the indices of `rec` where each cdf record starts.

    Returns:
        int, bool, int, numpy.array[ProbMean], numpy.array[int], int, int, int, numpy.array[items_data_type],
        int, Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]), int:
          number of int numbers read from the int32_mv ndarray, whether the current event (id=`event_id`)
          has been fully read, cdf record, array with the indices of `rec` where each cdf record starts, last or current
          event id, index of the last coverage id stored in `compute`, index of the last items_data_i stored in `items_data`,
          item-related data, number of unique random seeds computed so far, map of group ids to random seeds,
          index of the last cdf record that has been read from stream and stored in `rec`W
    """
    yield_event = False

    # `rec` is a temporary buffer to store the cdf being read
    # conservative choice: size `rec` as if the entire buffer is filled with cdf bins
    rec = np.zeros(valid_buf // ProbMean_size, dtype=ProbMean)

    # int32 memoryview cursor
    cursor = 0

    # init a counter for the local `rec` array
    last_rec_idx_ptr = 0
    rec_valid_len = 0

    while cursor + size_cdf_entry <= valid_buf:

        event_cursor = cursor
        event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

        if event_id != last_event_id:
            # a new event has started
            if last_event_id > 0:
                # if this is not the beginning of the very first event, yield the event that was just completed
                yield_event = True
                cursor = event_cursor
                break

            last_event_id = event_id

        areaperil_id, cursor = mv_read(byte_mv, cursor, areaperil_int, areaperil_int_size)
        vulnerability_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
        Nbins_to_read, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

        if cursor + Nbins_to_read * ProbMean_size > valid_buf:
            # if the next cdf record is not fully contained in the valid buf, then
            # get more data in the buffer and put cursor back at the beginning of this cdf
            cursor = event_cursor
            break

        # peril filter
        if valid_area_peril_dict is not None:
            if areaperil_id not in valid_area_peril_dict:
                cursor += ProbMean_size * Nbins_to_read
                continue

        # read damage cdf bins
        start_rec = last_rec_idx_ptr
        end_rec = start_rec + Nbins_to_read
        for j in range(start_rec, end_rec, 1):
            rec[j]['prob_to'], cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
            rec[j]['bin_mean'], cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)

        rec_idx_ptr.append(rec_idx_ptr[-1] + Nbins_to_read)
        last_rec_idx_ptr = end_rec
        rec_valid_len += Nbins_to_read

        # register the items to their coverage
        item_key = tuple((areaperil_id, vulnerability_id))

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
            items_data[item_i]['damagecdf_i'] = damagecdf_i
            items_data[item_i]['rng_index'] = this_rng_index

            coverage['cur_items'] += 1

        damagecdf_i += 1

    return cursor, yield_event, event_id, rec[:rec_valid_len], rec_idx_ptr, last_event_id, compute_i, items_data_i, items_data, rng_index, group_id_rng_index, damagecdf_i
