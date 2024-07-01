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
    summary_set_index_to_present_loss_ptr_end : copy of summary_set_index_to_loss_ptr that store the end of list of summary_id for each summary set
    present_summary_id : array that store all the summary_id found in the event
                         each summary set start at summary_set_index_to_loss_ptr and ends at summary_set_index_to_present_loss_ptr_end
    loss_index : store the index in loss_summary of each summary corresponding to an item for quick lookup on read
    loss_summary : 2D array of losses for each summary
                   loss_summary[loss_index[summary_set_index], sidx] = loss
    is_risk_affected : array to store in if a risk has already been affected or not in this event

"""
import numpy as np
import numba as nb
from numba.typed import Dict as nb_dict
import pandas as pd

from contextlib import ExitStack
import logging
import os
from itertools import zip_longest

from oasislmf.execution.bash import RUNTYPE_GROUNDUP_LOSS, RUNTYPE_INSURED_LOSS, RUNTYPE_REINSURANCE_LOSS
from oasislmf.pytools.common.data import (load_as_ndarray, oasis_int, nb_oasis_int, oasis_int_size, oasis_float, oasis_float_size,
                                          null_index, summary_xref_dtype)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, stream_info_to_bytes, write_mv_to_stream,
                                                  mv_read, mv_write_summary_header, mv_write_sidx_loss, mv_write_delimiter,
                                                  GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID, SUMMARY_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY,
                                                  MEAN_IDX, TIV_IDX, NUMBER_OF_AFFECTED_RISK_IDX, MAX_LOSS_IDX)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


SPECIAL_SIDX_COUNT = 6  # 0 is included as a special sidx
SUMMARY_HEADER_SIZE = 2 * oasis_int_size + oasis_float_size + SPECIAL_SIDX_COUNT * (oasis_int_size + oasis_float_size)
SIDX_LOSS_WRITE_SIZE = 2 * (oasis_int_size + oasis_float_size)


SUPPORTED_SUMMARY_SET_ID = list(range(1, 10))

risk_key_type = nb.types.UniTuple(nb_oasis_int, 2)

summary_info_dtype = np.dtype([('nb_risk', oasis_int), ])

SUPPORTED_RUN_TYPE = [RUNTYPE_GROUNDUP_LOSS, RUNTYPE_INSURED_LOSS, RUNTYPE_REINSURANCE_LOSS]


def create_summary_object_file(static_path, run_type):
    """create and write summary object into static path"""
    summary_objects = get_summary_object(static_path, run_type)
    write_summary_objects(static_path, run_type, *summary_objects)


def load_summary_object(static_path, run_type):
    """load already prepare summary data structure if present otherwise create them"""
    if os.path.isfile(os.path.join(static_path, run_type, 'summary_info.npy')):
        return read_summary_objects(static_path, run_type)
    else:
        return get_summary_object(static_path, run_type)


def get_summary_object(static_path, run_type):
    """read static files to get summary static data structure"""

    # extract item_id to index in the loss summary
    if run_type == RUNTYPE_GROUNDUP_LOSS:
        summary_xref_csv_col_map = {'summary_set_id': 'summaryset_id'}
        summary_xref = load_as_ndarray(static_path, 'gulsummaryxref', summary_xref_dtype, col_map=summary_xref_csv_col_map)
        summary_map = pd.read_csv(os.path.join(static_path, 'gul_summary_map.csv'),
                                  usecols=['loc_id', 'item_id', 'building_id', 'coverage_id'],
                                  dtype=oasis_int)
    elif run_type == RUNTYPE_INSURED_LOSS:
        summary_xref_csv_col_map = {'summary_set_id': 'summaryset_id',
                                    'item_id': 'output'}
        summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', summary_xref_dtype, col_map=summary_xref_csv_col_map)
        summary_map = pd.read_csv(os.path.join(static_path, 'fm_summary_map.csv'),
                                  usecols=['loc_id', 'output_id', 'building_id', 'coverage_id'],
                                  dtype=oasis_int,
                                  ).rename(columns={'output_id': 'item_id'})
    elif run_type == RUNTYPE_REINSURANCE_LOSS:
        summary_xref_csv_col_map = {'summary_set_id': 'summaryset_id',
                                    'item_id': 'output'}
        summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', summary_xref_dtype, col_map=summary_xref_csv_col_map)
        summary_map = None  # numba use none to optimise function when some part are not used
    else:
        raise Exception(f"run type {run_type} not in supported list {SUPPORTED_RUN_TYPE}")

    summary_sets_id = np.sort(np.unique(summary_xref['summary_set_id']))
    summary_set_id_to_summary_set_index = get_summary_set_id_to_summary_set_index(summary_sets_id)
    summary_set_index_to_loss_ptr, item_id_to_summary_id = get_summary_xref_info(summary_xref, summary_sets_id, summary_set_id_to_summary_set_index)

    if summary_map is not None:
        nb_risk, item_id_to_risks_i = extract_risk_info(item_id_to_summary_id.shape[0], summary_map)
    else:
        item_id_to_risks_i = np.zeros(0, dtype=oasis_int)
        nb_risk = 0

    summary_info = np.empty(1, dtype=summary_info_dtype)
    info = summary_info[0]
    info['nb_risk'] = nb_risk

    return summary_info, summary_set_id_to_summary_set_index, summary_set_index_to_loss_ptr, item_id_to_summary_id, item_id_to_risks_i


def write_summary_objects(static_path, run_type, summary_info, summary_set_id_to_summary_set_index,
                          summary_set_index_to_loss_ptr, item_id_to_summary_id, item_id_to_risks_i):
    os.makedirs(os.path.join(static_path, run_type), exist_ok=True)
    np.save(os.path.join(static_path, run_type, 'summary_info'), summary_info)
    np.save(os.path.join(static_path, run_type, 'summary_set_id_to_summary_set_index'), summary_set_id_to_summary_set_index)
    np.save(os.path.join(static_path, run_type, 'summary_set_index_to_loss_ptr'), summary_set_index_to_loss_ptr)
    np.save(os.path.join(static_path, run_type, 'item_id_to_summary_id'), item_id_to_summary_id)
    np.save(os.path.join(static_path, run_type, 'item_id_to_risks_i'), item_id_to_risks_i)


def read_summary_objects(static_path, run_type):
    summary_info = np.load(os.path.join(static_path, run_type, 'summary_info.npy'), mmap_mode='r')
    summary_set_id_to_summary_set_index = np.load(os.path.join(static_path, run_type, 'summary_set_id_to_summary_set_index.npy'), mmap_mode='r')
    summary_set_index_to_loss_ptr = np.load(os.path.join(static_path, run_type, 'summary_set_index_to_loss_ptr.npy'), mmap_mode='r')
    item_id_to_summary_id = np.load(os.path.join(static_path, run_type, 'item_id_to_summary_id.npy'), mmap_mode='r')
    item_id_to_risks_i = np.load(os.path.join(static_path, run_type, 'item_id_to_risks_i.npy'), mmap_mode='r')
    return summary_info, summary_set_id_to_summary_set_index, summary_set_index_to_loss_ptr, item_id_to_summary_id, item_id_to_risks_i


@nb.njit(cache=True)
def nb_extract_risk_info(item_id_to_risks_i, summary_map_item_ids, summary_map_loc_ids, summary_map_building_ids):
    loc_id_building_id_to_building_risk = nb_dict.empty(risk_key_type, nb_oasis_int)
    last_risk_i = 0
    for i in range(summary_map_item_ids.shape[0]):
        loc_id_building_id = (nb_oasis_int(summary_map_loc_ids[i]), nb_oasis_int(summary_map_building_ids[i]))
        if loc_id_building_id in loc_id_building_id_to_building_risk:
            risk_i = loc_id_building_id_to_building_risk[loc_id_building_id]
        else:
            loc_id_building_id_to_building_risk[loc_id_building_id] = risk_i = nb_oasis_int(last_risk_i)
            last_risk_i += 1
        item_id_to_risks_i[summary_map_item_ids[i]] = nb_oasis_int(risk_i)
    return last_risk_i


def extract_risk_info(len_item_id, summary_map):
    """
    extract relevant information regarding item and risk mapping from summary_map
    Args:
        len_item_id: number of items
        summary_map: numpy ndarray view of the summary_map

    Returns:
        (number of risk, mapping array item_id => risks_i)
    """

    item_id_to_risks_i = np.zeros(len_item_id, oasis_int)
    nb_risk = nb_extract_risk_info(
        item_id_to_risks_i,
        summary_map['item_id'].astype(oasis_int).to_numpy(),
        summary_map['loc_id'].astype(oasis_int).to_numpy(),
        summary_map['building_id'].astype(oasis_int).to_numpy())
    return nb_risk, item_id_to_risks_i


@nb.jit(nopython=True, cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id,
                summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                loss_index, loss_summary, present_summary_id, summary_set_index_to_present_loss_ptr_end,
                item_id_to_risks_i, is_risk_affected, has_affected_risk):
    """read valid part of byte_mv and load relevant data for one event"""
    last_event_id = event_id
    while True:
        if item_id:
            if valid_buff - cursor < (oasis_int_size + oasis_float_size):
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
            if valid_buff - cursor < 2 * oasis_int_size:
                break
            event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
            if event_id != last_event_id:
                if last_event_id:  # we have a new event we return the one we just finished
                    for summary_set_index in range(summary_sets_id.shape[0]):  # reorder summary_id for each summary set
                        summary_set_start = summary_set_index_to_loss_ptr[summary_set_index]
                        summary_set_end = summary_set_index_to_present_loss_ptr_end[summary_set_index]
                        present_summary_id[summary_set_start: summary_set_end] = np.sort(present_summary_id[summary_set_start: summary_set_end])
                    return cursor - oasis_int_size, last_event_id, 0, 1
                else:  # first pass we store the event we are reading
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
                # print('loss_summary[loss_index[summary_set_index], NUMBER_OF_AFFECTED_RISK_IDX]', loss_summary[loss_index[summary_set_index], -4])
                if loss_summary[loss_index[summary_set_index], NUMBER_OF_AFFECTED_RISK_IDX] == 0:  # we use sidx 0 to check if this summary_id has already been seen
                    present_summary_id[summary_set_index_to_present_loss_ptr_end[summary_set_index]
                                       ] = item_id_to_summary_id[item_id, summary_set_index]
                    summary_set_index_to_present_loss_ptr_end[summary_set_index] += 1
                    loss_summary[loss_index[summary_set_index], NUMBER_OF_AFFECTED_RISK_IDX] = 1

                elif has_affected_risk is not None:
                    loss_summary[loss_index[summary_set_index], NUMBER_OF_AFFECTED_RISK_IDX] += new_risk
            ##########
    for summary_set_index in range(summary_sets_id.shape[0]):  # reorder summary_id for each summary set
        summary_set_start = summary_set_index_to_loss_ptr[summary_set_index]
        summary_set_end = summary_set_index_to_present_loss_ptr_end[summary_set_index]
        present_summary_id[summary_set_start: summary_set_end] = np.sort(present_summary_id[summary_set_start: summary_set_end])
    return cursor, event_id, item_id, 0


@nb.njit(cache=True)
def mv_write_event(byte_mv, event_id, len_sample, last_loss_summary_index, last_sidx,
                   output_zeros, has_affected_risk,
                   summary_set_index, summary_set_index_to_loss_ptr, summary_set_index_to_present_loss_ptr_end, present_summary_id, loss_summary,
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
                                    summary_set_index_to_present_loss_ptr_end[summary_set_index]):
        summary_id = present_summary_id[loss_summary_index]
        losses = loss_summary[summary_set_index_to_loss_ptr[summary_set_index] + summary_id - 1]

        if not output_zeros and losses[TIV_IDX] == 0:
            continue

        summary_stream_index[summary_index_cursor]['summary_id'] = summary_id
        # we use offset to temporally store the cursor, we set the correct value later on
        summary_stream_index[summary_index_cursor]['offset'] = cursor

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
            if cursor < PIPE_CAPACITY - SIDX_LOSS_WRITE_SIZE:  # times 2 to accommodate 0,0 if last item
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
                 loss_index, loss_summary, present_summary_id, summary_set_index_to_present_loss_ptr_end,
                 item_id_to_risks_i, is_risk_affected, has_affected_risk):
        self.summary_sets_id = summary_sets_id
        self.summary_set_index_to_loss_ptr = summary_set_index_to_loss_ptr
        self.item_id_to_summary_id = item_id_to_summary_id
        self.loss_index = loss_index
        self.loss_summary = loss_summary
        self.present_summary_id = present_summary_id
        self.summary_set_index_to_present_loss_ptr_end = summary_set_index_to_present_loss_ptr_end
        self.item_id_to_risks_i = item_id_to_risks_i
        self.is_risk_affected = is_risk_affected
        self.has_affected_risk = has_affected_risk
        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        return read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.summary_sets_id, self.summary_set_index_to_loss_ptr, self.item_id_to_summary_id,
            self.loss_index, self.loss_summary, self.present_summary_id, self.summary_set_index_to_present_loss_ptr_end,
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


def run(files_in, static_path, run_type, low_memory, output_zeros, **kwargs):
    """
    Args:
        files_in: list of file path to read event from
        run_type: type of the source that is sending the stream
        static_path: path to the static files
        low_memory: if true output summary index file
        output_zeros: if true output 0 loss
        **kwargs:

    """
    summary_sets_path = {}
    error_msg = (f"summary_sets_output expected format is a list of -summary_set_id summary_set_path (ex: -1 S1.bin -2 S2.bin')"
                 f", found '{' '.join(kwargs['summary_sets_output'])}'")
    for summary_set_id, summary_set_path in zip_longest(*[iter(kwargs['summary_sets_output'])] * 2):
        if summary_set_id[0] != '-' or summary_set_path is None:
            raise Exception(error_msg)
        try:
            summary_sets_path[int(summary_set_id[1:])] = summary_set_path
        except ValueError:
            raise Exception(error_msg)
    summary_sets_id = np.array(list(summary_sets_path.keys()))

    with ExitStack() as stack:
        summary_sets_pipe = {i: stack.enter_context(open(summary_set_path, 'wb')) for i, summary_set_path in summary_sets_path.items()}

        if low_memory:
            summary_sets_index_pipe = {summary_set_id: stack.enter_context(open(setpath.rsplit('.', 1)[0] + '.idx', 'w'))
                                       for summary_set_id, setpath in summary_sets_path.items()}

        streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)

        if stream_source_type not in (GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID):
            raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")

        summary_object = load_summary_object(static_path, run_type)
        summary_info, summary_set_id_to_summary_set_index, summary_set_index_to_loss_ptr, item_id_to_summary_id, item_id_to_risks_i = summary_object

        # check all summary_set_id required are defined in summaryxref
        invalid_summary_sets_id = []
        for summary_set_id in summary_sets_id:
            if summary_set_id > summary_set_id_to_summary_set_index.shape[0] or summary_set_id_to_summary_set_index[summary_set_id] == null_index:
                invalid_summary_sets_id.append(summary_set_id)
        if invalid_summary_sets_id:
            raise ValueError(
                f'summary_set_ids {invalid_summary_sets_id} not found in summaryxref available summary_set_id '
                f'{[summary_set_id for summary_set_id, summary_set_index in enumerate(summary_set_id_to_summary_set_index) if summary_set_index != null_index]}')

        has_affected_risk = True if summary_info[0]['nb_risk'] > 0 else None

        # init temporary
        present_summary_id = np.zeros(summary_set_index_to_loss_ptr[-1], oasis_int)
        loss_index = np.empty(summary_sets_id.shape[0], oasis_int)
        summary_set_index_to_present_loss_ptr_end = np.array(summary_set_index_to_loss_ptr)
        loss_summary = np.zeros((summary_set_index_to_loss_ptr[-1], len_sample + SPECIAL_SIDX_COUNT), dtype=oasis_float)
        is_risk_affected = np.zeros(summary_info[0]['nb_risk'], dtype=oasis_int)

        summary_reader = SummaryReader(
            summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
            loss_index, loss_summary, present_summary_id, summary_set_index_to_present_loss_ptr_end,
            item_id_to_risks_i, is_risk_affected, has_affected_risk)

        out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

        # data for index file (low_memory==True)
        summary_sets_cursor = np.zeros(summary_sets_id.shape[0], dtype=np.int64)
        summary_stream_index = np.empty(summary_set_index_to_loss_ptr[-1], dtype=np.dtype([('summary_id', oasis_int), ('offset', np.int64)]))

        for summary_set_index, summary_set_id in enumerate(summary_sets_id):
            summary_pipe = summary_sets_pipe[summary_set_id]
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
                    cursor, last_loss_summary_index, last_sidx, summary_index_cursor = mv_write_event(
                        out_byte_mv, event_id, len_sample, last_loss_summary_index, last_sidx,
                        output_zeros, has_affected_risk,
                        summary_set_index, summary_set_index_to_loss_ptr, summary_set_index_to_present_loss_ptr_end, present_summary_id,
                        loss_summary,
                        summary_index_cursor, summary_sets_cursor, summary_stream_index
                    )
                    write_mv_to_stream(summary_pipe, out_byte_mv, cursor)
                    if last_loss_summary_index == -1:
                        break
                if low_memory:
                    # write the summary.idx file
                    np.savetxt(summary_sets_index_pipe[summary_set_id],
                               summary_stream_index[:summary_index_cursor],
                               fmt="%i,%i")

            loss_summary.fill(0)
            present_summary_id.fill(0)
            summary_set_index_to_present_loss_ptr_end[:] = summary_set_index_to_loss_ptr
            is_risk_affected.fill(0)


@redirect_logging(exec_name='summarypy')
def main(create_summarypy_files, static_path, run_type, **kwargs):
    if create_summarypy_files:
        create_summary_object_file(static_path, run_type)
    else:
        run(static_path=static_path, run_type=run_type, **kwargs)
