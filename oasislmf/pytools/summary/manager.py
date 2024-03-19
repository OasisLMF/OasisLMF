"""
Entry point to the run method of summarypy
read an event loss stream and aggregate it into the relevant summary loss stream

using numba latest version 0.59, we are limited is our usage of classes and have to pass each data structure into the different function
The attributes are name the same throughout the code, below is their definition

Static intermediary data structure:
    summary_pipes : dict summary_set_id to path then updated to stream when reading starts
    summary_sets_id : array of all summary_set_id that have a stream
    summary_set_id_to_summary_set_index : array mapping summary_set_id => summary_set_index
    summary_set_index_to_loss_ptr : mapping array summary_set_index => starting index of summary_id loss (first dimension) in the loss_summary table
    item_id_to_summary_id : mapping array item_id => [summary_id for each summary set]
    item_id_to_risks_i : mapping array item_id => risks_i (index of th corresponding risk)

event based intermediary data structure:
    summary_id_count_per_summary_set : copy of summary_set_index_to_loss_ptr that store the end of list of summary_id for each summary set
    present_summary_id : array that store all the summary_id found in the event
                         each summary set start at summary_set_index_to_loss_ptr and ends at summary_id_count_per_summary_set
    loss_index : store the index in loss_summary of each summary corresponding to an item for quick lookup on read
    loss_summary : 2D array of losses for each summary
                   loss_summary[loss_index[summary_set_index], sidx] = loss
    is_risk_affected : array to store in if a risk has already been affected or not in this event

"""

import numpy as np
import numba as nb
import numba.typed
import pandas as pd

from contextlib import ExitStack
import logging
import os

from oasislmf.pytools.common.data import (load_as_ndarray, oasis_int, nb_oasis_int, oasis_int_size, oasis_float, oasis_float_size,
                                          null_index, summary_xref_dtype)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, stream_info_to_bytes, write_mv_to_stream,
                                                  mv_read, mv_write_summary_header, mv_write_sidx_loss, mv_write_delimiter,
                                                  GUL_STREAM_ID, FM_STREAM_ID, SUMMARY_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY,
                                                  MEAN_IDX, TIV_IDX, NUMBER_OF_AFFECTED_RISK_IDX, MAX_LOSS_IDX)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


SPECIAL_SIDX_COUNT = 6 # 0 is included as a special sidx
SUMMARY_HEADER_SIZE = 2 * oasis_int_size + oasis_float_size + SPECIAL_SIDX_COUNT * (oasis_int_size + oasis_float_size)
SIDX_LOSS_WRITE_SIZE = 2 * (oasis_int_size + oasis_float_size)


SUPPORTED_SUMMARY_SET_ID = list(range(1, 10))

def extract_risk_info(len_item_id, summary_map):
    """
    extract relevant information regarding item and risk mapping from summary_map
    Args:
        len_item_id: number of items
        summary_map: numpy ndarray view of the summary_map

    Returns:
        (number of risk, mapping array item_id => risks_i)
    """
    item_id_to_risks_i = np.empty(len_item_id, oasis_int)
    nb_risk = nb_extract_risk_info(item_id_to_risks_i,
                                summary_map['item_id'].to_numpy(),
                                summary_map['loc_id'].to_numpy(),
                                summary_map['building_id'].to_numpy()),
    return nb_risk, item_id_to_risks_i


@nb.njit
def nb_extract_risk_info(item_id_to_risks_i, summary_map_item_ids, summary_map_loc_ids, summary_map_building_ids):
    loc_id_to_building_risk = numba.typed.Dict.empty(nb_oasis_int, numba.typed.Dict.empty(nb_oasis_int, nb_oasis_int))
    last_risk_i = 0
    for i in range(summary_map_item_ids.shape[0]):
        if summary_map_item_ids[i] in loc_id_to_building_risk:
            building_id_to_risk_i = loc_id_to_building_risk[summary_map_loc_ids[i]]
        else:
            building_id_to_risk_i = numba.typed.Dict.empty(nb_oasis_int, nb_oasis_int)

        if summary_map_building_ids[i] in building_id_to_risk_i:
            risk_i = building_id_to_risk_i[summary_map_building_ids[i]]
        else:
            risk_i = building_id_to_risk_i[summary_map_building_ids[i]] = last_risk_i
            last_risk_i += 1
        item_id_to_risks_i[summary_map_item_ids[i]] = nb_oasis_int(risk_i)

    return last_risk_i


@nb.jit(cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id,
                summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                loss_index, loss_summary, present_summary_id, summary_id_count_per_summary_set,
                item_id_to_risks_i, is_risk_affected, has_affected_risk):
    """read valid part of byte_mv and load relevant data for one event"""
    last_event_id = event_id
    while True:
        if item_id:
            if valid_buff -  cursor < (oasis_int_size + oasis_float_size):
                break
            sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if sidx:
                loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
                loss = 0 if np.isnan(loss) else loss

                ###### do loss read ######
                if sidx > 0 or sidx in [-1, -3, -5]:
                    for summary_set_index in range(summary_sets_id.shape[0]):
                        loss_summary[loss_index[summary_set_index], sidx] += loss
                ##########

            else:
                ##### do item exit ####
                ##########
                cursor += oasis_float_size
                item_id = 0
        else:
            if valid_buff -  cursor < 2 * oasis_int_size:
                break
            event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if event_id != last_event_id:
                if last_event_id: # we have a new event we return the one we just finished
                    return cursor - oasis_int_size, last_event_id, 0, 1
                else: # first pass we store the event we are reading
                    last_event_id = event_id
            item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)

            ##### do new item setup #####
            if has_affected_risk is not None:
                risk_i = item_id_to_risks_i[item_id]
                if is_risk_affected[risk_i]:
                    new_risk = 0
                else:
                    new_risk = is_risk_affected[risk_i] = 1

            for summary_set_index in range(summary_sets_id.shape[0]):
                loss_index[summary_set_index] = nb_oasis_int(summary_set_index_to_loss_ptr[summary_set_index]
                                                          + item_id_to_summary_id[item_id, summary_set_index] - 1)
                # print(summary_set_index_to_loss_ptr[summary_set_index], item_id_to_summary_id[item_id, summary_set_index])
                # print('new_risk', event_id, item_id, new_risk)
                # print('loss_index[summary_set_index]', loss_index[summary_set_index])
                # print('loss_summary[loss_index[summary_set_index], -4]', loss_summary[loss_index[summary_set_index], -4])
                if loss_summary[loss_index[summary_set_index], -4] == 0: # we use sidx 0 to check if this summary_id has already been seen
                    present_summary_id[summary_id_count_per_summary_set[summary_set_index]] = item_id_to_summary_id[item_id, summary_set_index]
                    summary_id_count_per_summary_set[summary_set_index] += 1

                if has_affected_risk is not None:
                    loss_summary[loss_index[summary_set_index], -4] += new_risk

            ##########
    return cursor, event_id, item_id, 0


@nb.njit(cache=True)
def mv_write_event(byte_mv, event_id, len_sample, last_loss_summary_index, last_sidx,
                   output_zeros, has_affected_risk,
                   summary_set_index, summary_set_index_to_loss_ptr, summary_id_count_per_summary_set, present_summary_id, loss_summary,
                   summary_index_cursor, summary_sets_cursor, summary_stream_index):
    """
        load event summary loss into byte_mv

    Args:
        byte_mv: numpy byte view to write to the stream
        event_id: event id
        len_sample: max sample id
        last_loss_summary_index: last summary index written (used to restart from the last summary when buffer was full)
        last_sidx: last sidx written in the buffer (used to restart from the correct sidx when buffer was full

        see other args definition in run method
    """
    cursor = 0
    for loss_summary_index in range(max(summary_set_index_to_loss_ptr[summary_set_index], last_loss_summary_index),
                                    summary_id_count_per_summary_set[summary_set_index]):
        summary_id = present_summary_id[loss_summary_index]

        losses = loss_summary[summary_set_index_to_loss_ptr[summary_set_index] + summary_id - 1]

        if not output_zeros and losses[TIV_IDX] == 0:
            continue

        summary_stream_index[summary_index_cursor]['summary_id'] = summary_id
        summary_stream_index[summary_index_cursor]['offset'] = cursor # we use offset to temporally store the cursor, we set the correct value later on

        if last_sidx == 0:
            if cursor < PIPE_CAPACITY - SUMMARY_HEADER_SIZE:
                cursor = mv_write_summary_header(byte_mv, cursor, event_id, summary_id, losses[TIV_IDX])
                cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, losses[MEAN_IDX])
                if has_affected_risk is not None:
                    cursor = mv_write_sidx_loss(byte_mv, cursor, NUMBER_OF_AFFECTED_RISK_IDX, losses[NUMBER_OF_AFFECTED_RISK_IDX])
                cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, losses[MAX_LOSS_IDX])
                last_sidx = 1
            else:
                return cursor, loss_summary_index, last_sidx, summary_index_cursor

        for sidx in range(last_sidx, len_sample + 1):
            if not output_zeros and losses[sidx] == 0:
                continue
            if cursor < PIPE_CAPACITY - SIDX_LOSS_WRITE_SIZE: # times 2 to accommodate 0,0 if last item
                cursor = mv_write_sidx_loss(byte_mv, cursor, sidx, losses[sidx])
            else:
                return cursor, loss_summary_index, sidx, summary_index_cursor

        cursor = mv_write_delimiter(byte_mv, cursor)
        # set the correct offset for idx file and update summary_sets_cursor
        summary_byte_len = cursor - summary_stream_index[summary_index_cursor]['offset']
        summary_stream_index[summary_index_cursor]['offset'] = summary_sets_cursor[summary_set_index]
        summary_sets_cursor[summary_set_index] += summary_byte_len
        summary_index_cursor += 1

        last_sidx = 0
    return cursor, -1, 0, summary_index_cursor

class SummaryReader(EventReader):
    def __init__(self, summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                 loss_index, loss_summary, present_summary_id, summary_id_count_per_summary_set,
                 item_id_to_risks_i, is_risk_affected, has_affected_risk):
        self.summary_sets_id = summary_sets_id
        self.summary_set_index_to_loss_ptr = summary_set_index_to_loss_ptr
        self.item_id_to_summary_id = item_id_to_summary_id
        self.loss_index = loss_index
        self.loss_summary = loss_summary
        self.present_summary_id = present_summary_id
        self.summary_id_count_per_summary_set = summary_id_count_per_summary_set
        self.item_id_to_risks_i = item_id_to_risks_i
        self.is_risk_affected = is_risk_affected
        self.has_affected_risk = has_affected_risk
        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        return read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.summary_sets_id, self.summary_set_index_to_loss_ptr, self.item_id_to_summary_id,
            self.loss_index, self.loss_summary, self.present_summary_id, self.summary_id_count_per_summary_set,
            self.item_id_to_risks_i, self.is_risk_affected, self.has_affected_risk
        )


def get_summary_set_id_to_summary_set_index(summary_sets_id):
    """create an array mapping summary_set_id => summary_set_index"""
    summary_set_id_to_summary_set_index = np.full(np.max(summary_sets_id) + 1, null_index, 'i4')
    for summary_set_index in range(summary_sets_id.shape[0]):
        summary_set_id_to_summary_set_index[summary_sets_id[summary_set_index]] = summary_set_index
    return summary_set_id_to_summary_set_index


def get_summary_xref_info(summary_xref, summary_sets_id, summary_set_id_to_summary_set_index):
    """
    extract mapping from summary_xref
    """
    summary_set_index_to_loss_ptr = np.zeros(summary_sets_id.shape[0] + 1, oasis_int)
    max_item_id = 0
    for i in range(summary_xref.shape[0]):
        xref = summary_xref[i]
        summary_set_index = summary_set_id_to_summary_set_index[xref['summary_set_id']]
        if summary_set_index == null_index:
            continue
        if xref['summary_id'] > summary_set_index_to_loss_ptr[summary_set_index + 1]:
            summary_set_index_to_loss_ptr[summary_set_index + 1] = xref['summary_id']
        if xref['item_id'] > max_item_id:
            max_item_id = xref['item_id']
    for i in range(1, summary_set_index_to_loss_ptr.shape[0]):
        summary_set_index_to_loss_ptr[i] += summary_set_index_to_loss_ptr[i - 1]

    item_id_to_summary_id = np.full((max_item_id + 1, summary_sets_id.shape[0]), null_index, oasis_int)
    for i in range(summary_xref.shape[0]):
        xref = summary_xref[i]
        summary_set_index = summary_set_id_to_summary_set_index[xref['summary_set_id']]
        if summary_set_index == null_index:
            continue
        item_id_to_summary_id[xref['item_id'], summary_set_index] = xref['summary_id']

    return summary_set_index_to_loss_ptr, item_id_to_summary_id


@redirect_logging('summarypy')
def run(files_in, static_path, low_memory, output_zeros, **kwargs):
    """
    Static intermediary data structure:
        summary_pipes : dict summary_set_id to path then updated to stream when reading starts
        summary_sets_id : array of all summary_set_id that have a stream
        summary_set_id_to_summary_set_index : array mapping summary_set_id => summary_set_index
        summary_set_index_to_loss_ptr : mapping array summary_set_index => starting index of summary_id loss (first dimention) in the loss_summary table
        item_id_to_summary_id : mapping array item_id => [summary_id for each summary set]
        item_id_to_risks_i : mapping array item_id => risks_i (index of th corresponding risk)

    temporary intermediary data structure:
        summary_id_count_per_summary_set : copy of summary_set_index_to_loss_ptr that store the end of list of summary_id for each summary set
        present_summary_id : array that store all the summary_id found in the event
                             each summary set start at summary_set_index_to_loss_ptr and ends at summary_id_count_per_summary_set
        loss_index : store the index in loss_summary of each summary corresponding to an item for quick lookup on read
        loss_summary : 2D array of losses for each summary
                       loss_summary[loss_index[summary_set_index], sidx] = loss
        is_risk_affected : array to store in if a risk has already been affected or not in this event

    Args:
        files_in: list of file path to read event from
        static_path: path to the static files
        low_memory: if true output summary index file
        output_zeros: if true output 0 loss
        **kwargs:

    """
    summary_sets_pipe = {i: kwargs[str(i)] for i in SUPPORTED_SUMMARY_SET_ID if kwargs.get(str(i))}
    summary_sets_id = np.array(list(summary_sets_pipe.keys()))

    with ExitStack() as stack:
        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)

        summary_set_id_to_summary_set_index = get_summary_set_id_to_summary_set_index(summary_sets_id)

        # extract item_id to index in the loss summary
        if stream_source_type == GUL_STREAM_ID:
            summary_xref = load_as_ndarray(static_path, 'gulsummaryxref', summary_xref_dtype)
            summary_map = pd.read_csv(os.path.join(static_path, 'gul_summary_map.csv'),
                                      usecols = ['loc_id', 'item_id', 'building_id', 'coverage_id'],
                                      dtype=oasis_int)
            has_affected_risk = True

        elif  stream_source_type == FM_STREAM_ID:
            summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', summary_xref_dtype)
            if os.path.exists(os.path.join(static_path, 'fm_summary_map.csv')):
                summary_map = pd.read_csv(os.path.join(static_path, 'fm_summary_map.csv'),
                                          usecols = ['loc_id', 'output_id', 'building_id', 'coverage_id'],
                                          dtype=oasis_int,
                                          ).rename(columns={'output_id': 'item_id'})
                has_affected_risk = True
            else:
                has_affected_risk = None # numba use none to optimise function when some part are not used
        else:
            raise Exception(f"unsupported stream type {stream_source_type}")


        summary_set_index_to_loss_ptr, item_id_to_summary_id = get_summary_xref_info(summary_xref, summary_sets_id, summary_set_id_to_summary_set_index)

        if has_affected_risk:
            nb_risk, item_id_to_risks_i = extract_risk_info(item_id_to_summary_id.shape[0], summary_map)
            is_risk_affected = np.zeros(nb_risk, dtype=oasis_int)
        else:
            item_id_to_risks_i = is_risk_affected = np.zeros(0, dtype=oasis_int)

        # init temporary
        present_summary_id = np.zeros(summary_set_index_to_loss_ptr[-1], oasis_int)
        loss_index = np.empty(summary_sets_id.shape[0], oasis_int)
        summary_id_count_per_summary_set = np.array(summary_set_index_to_loss_ptr)
        loss_summary = np.zeros((summary_set_index_to_loss_ptr[-1], len_sample + SPECIAL_SIDX_COUNT), dtype=oasis_float)


        summary_reader = SummaryReader(
                summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                loss_index, loss_summary, present_summary_id, summary_id_count_per_summary_set,
                item_id_to_risks_i, is_risk_affected, has_affected_risk)

        out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

        # data for index file (low_memory==True)
        summary_sets_cursor = np.zeros(summary_sets_id.shape[0], dtype=np.int64)
        summary_stream_index = np.empty(summary_set_index_to_loss_ptr[-1], dtype=np.dtype([('summary_id', oasis_int), ('offset', np.int64)]))


        if low_memory:
            summary_sets_index_pipe = {summary_set_id: stack.enter_context(open(setpath.rsplit('.', 1)[0] + '.idx', 'w'))
                                       for summary_set_id, setpath in summary_sets_pipe.items()}

        for summary_set_index, summary_set_id in enumerate(summary_sets_id):
            summary_pipe = stack.enter_context(open(summary_sets_pipe[summary_set_id], 'wb'))
            summary_sets_pipe[summary_set_id] = summary_pipe

            summary_sets_cursor[summary_set_index] += summary_pipe.write(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM))
            summary_sets_cursor[summary_set_index] += summary_pipe.write(len_sample.tobytes())
            summary_sets_cursor[summary_set_index] += summary_pipe.write(nb_oasis_int(summary_set_id).tobytes())

        for event_id in summary_reader.read_streams(streams_in):
            for summary_set_index, summary_set_id in enumerate(summary_sets_id):
                summary_pipe = summary_sets_pipe[summary_set_id]
                summary_index_cursor = 0
                last_loss_summary_index = 0
                last_sidx = 0
                while True:
                    cursor, loss_summary_index, last_sidx, summary_index_cursor = mv_write_event(
                        out_byte_mv, event_id, len_sample, last_loss_summary_index, last_sidx,
                        output_zeros, has_affected_risk,
                        summary_set_index, summary_set_index_to_loss_ptr, summary_id_count_per_summary_set, present_summary_id,
                        loss_summary,
                        summary_index_cursor, summary_sets_cursor, summary_stream_index
                    )
                    write_mv_to_stream(summary_pipe, out_byte_mv, cursor)
                    if loss_summary_index == -1:
                        break
                if low_memory:
                    # write the summary.idx file
                    np.savetxt(summary_sets_index_pipe[summary_set_id],
                               summary_stream_index[:summary_index_cursor],
                               fmt="%i,%i")



            loss_summary.fill(0)
            present_summary_id.fill(0)
            summary_id_count_per_summary_set.fill(0)
            is_risk_affected.fill(0)
