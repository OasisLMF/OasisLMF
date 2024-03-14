import select

import numpy as np
import numba as nb
import numba.typed
import pandas as pd

from contextlib import ExitStack
import logging
import os
import sys

from oasislmf.pytools.common.data import (load_as_ndarray, oasis_int, nb_oasis_int, oasis_int_size, oasis_float, oasis_float_size,
                                          null_index, summary_xref_dtype)
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, stream_info_to_bytes,
                                                  mv_read, load_summary_header, load_sidx_loss, load_delimiter,
                                                  GUL_STREAM_ID, FM_STREAM_ID, SUMMARY_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY,
                                                  MEAN_IDX, TIV_IDX, NUMBER_OF_AFFECTED_RISK_IDX, MAX_LOSS_IDX)
from oasislmf.pytools.utils import redirect_logging

logger = logging.getLogger(__name__)


SPECIAL_SIDX_COUNT = 6 # 0 is included as a special sidx
SUMMARY_HEADER_LEN = 2 * oasis_int_size + oasis_float_size + SPECIAL_SIDX_COUNT * (oasis_int_size + oasis_float_size)

def extract_risk_info(len_item_id, summary_map):
    item_to_risks_i = np.empty(len_item_id, oasis_int)
    nb_risk = nb_extract_risk_info(item_to_risks_i,
                                summary_map['item_id'].to_numpy(),
                                summary_map['loc_id'].to_numpy(),
                                summary_map['building_id'].to_numpy()),
    return nb_risk, item_to_risks_i


@nb.njit
def nb_extract_risk_info(item_to_risks_i, summary_map_item_ids, summary_map_loc_ids, summary_map_building_ids):
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
        item_to_risks_i[summary_map_item_ids[i]] = nb_oasis_int(risk_i)

    return last_risk_i


@nb.jit(cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id,
                summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                loss_index, loss_summary, present_summary_id, summary_id_count_per_summary_set,
                item_id_to_risks_i, is_risk_affected, has_affected_risk):

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

            print('event_id', event_id, 'item_id', item_id)
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
def load_event(byte_mv, event_id, len_sample, output_zeros, last_loss_summary_index, last_sidx,
               summary_set_index, summary_set_index_to_loss_ptr, summary_id_count_per_summary_set, present_summary_id, loss_summary,
               summary_index_cursor, summary_sets_cursor, summary_stream_index, has_affected_risk):
    """

    Args:
        byte_mv: byte np.array view of the memory view use to write in the output
        event_id: event_id
        len_sample: max sidx
        output_zeros: if false, filter 0 loss
        last_loss_summary_index: index of the summary we need to restart at
        last_sidx: sidx we need to restart at
        summary_set_index: index of the current summary set
        summary_set_index_to_loss_ptr: pointer to where the summury set loss start in loss_summary
        summary_id_count_per_summary_set:
        present_summary_id:
        loss_summary:

    Returns:

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
            if cursor + SUMMARY_HEADER_LEN < PIPE_CAPACITY:
                cursor = load_summary_header(byte_mv, cursor, event_id, summary_id, losses[TIV_IDX])
                cursor = load_sidx_loss(byte_mv, cursor, MEAN_IDX, losses[MEAN_IDX])
                if has_affected_risk is not None:
                    cursor = load_sidx_loss(byte_mv, cursor, NUMBER_OF_AFFECTED_RISK_IDX, losses[NUMBER_OF_AFFECTED_RISK_IDX])
                cursor = load_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, losses[MAX_LOSS_IDX])
                last_sidx = 1
            else:
                return cursor, loss_summary_index, last_sidx, summary_index_cursor

        for sidx in range(last_sidx, len_sample + 1):
            if not output_zeros and losses[sidx] == 0:
                continue
            if cursor + 2 * (oasis_int_size + oasis_float_size) < PIPE_CAPACITY: # times 2 to accommodate 0,0 if last item
                cursor = load_sidx_loss(byte_mv, cursor, sidx, losses[sidx])
            else:
                return cursor, loss_summary_index, sidx, summary_index_cursor

        cursor = load_delimiter(byte_mv, cursor)
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

    def item_exit(self):
        pass

    def event_read_log(self, **kwargs):
        pass


@redirect_logging('summarypy')
def run(files_in, static_path, low_memory, output_zeros, **kwargs):
    # summary_pipes dict summary_set_id to stream
    # summary_sets_id array of all summary_set_id that have a stream

    summary_sets_pipe = {i: kwargs[str(i)] for i in range(10) if kwargs.get(str(i))}
    summary_sets_id = np.array(list(summary_sets_pipe.keys()))

    print(files_in, static_path, low_memory, output_zeros, summary_sets_pipe, summary_sets_id)

    with ExitStack() as stack:
        streams_in, (stream_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)

        # map summary set id to summary set index
        summary_set_id_to_summary_set_index = np.full(10, null_index, 'i4')
        for summary_set_index in range(summary_sets_id.shape[0]):
            summary_set_id_to_summary_set_index[summary_sets_id[summary_set_index]] = summary_set_index

        print('summary_set_map', summary_set_id_to_summary_set_index)

        # extract item_id to index in the loss summary
        if stream_type == GUL_STREAM_ID:
            summary_xref = load_as_ndarray(static_path, 'gulsummaryxref', summary_xref_dtype)
            summary_map = pd.read_csv(os.path.join(static_path, 'gul_summary_map.csv'),
                                      usecols = ['loc_id', 'item_id', 'building_id', 'coverage_id'],
                                      dtype=oasis_int)
            has_affected_risk = True

        elif  stream_type == FM_STREAM_ID:
            summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', summary_xref_dtype)
            if os.path.exists(os.path.join(static_path, 'fm_summary_map.csv')):
                summary_map = pd.read_csv(os.path.join(static_path, 'fm_summary_map.csv'),
                                          usecols = ['loc_id', 'output_id', 'building_id', 'coverage_id'],
                                          dtype=oasis_int,
                                          ).rename(columns={'output_id': 'item_id'})
                has_affected_risk = True
            else:
                has_affected_risk = None
        else:
            raise Exception(f"unsupported stream type {stream_type}")

        print(stream_type)
        print(summary_xref)
        print(summary_map)
        0/0
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
        print('summary_set_index_to_loss_ptr', summary_set_index_to_loss_ptr)

        item_id_to_summary_id = np.full((max_item_id + 1, summary_sets_id.shape[0]), null_index, oasis_int)
        for i in range(summary_xref.shape[0]):
            xref = summary_xref[i]
            summary_set_index = summary_set_id_to_summary_set_index[xref['summary_set_id']]
            if summary_set_index == null_index:
                continue
            item_id_to_summary_id[xref['item_id'], summary_set_index] = xref['summary_id']

        present_summary_id = np.zeros(summary_set_index_to_loss_ptr[-1], oasis_int)
        loss_index = np.empty(summary_sets_id.shape[0], oasis_int)
        summary_id_count_per_summary_set = np.array(summary_set_index_to_loss_ptr)
        loss_summary = np.zeros((summary_set_index_to_loss_ptr[-1], len_sample + SPECIAL_SIDX_COUNT), dtype=oasis_float)

        if has_affected_risk:
            nb_risk, item_id_to_risks_i = extract_risk_info(item_id_to_summary_id.shape[0], summary_map)
            is_risk_affected = np.zeros(nb_risk, dtype=oasis_int)
            print(item_id_to_risks_i)
        else:
            item_id_to_risks_i = is_risk_affected = np.zeros(0, dtype=oasis_int)

        # summary_sets_id: list of summary_set_id
        # summary_set_to_index : map summary_set_id to summary_set_index
        # summary_set_ptr : map summary_set_id to where it starts in the loss
        # summary_ptr : ptr to where the summary id is in summary_id_present
        # summary_id_present: starting at each summary_set_ptr, list all summary id present, 0 means no more summary for this set
        # summary_xref_map : map between ( item_id , summary) => index 1 in loss_summary
        # loss_summary: loss for (summary_set_id, summary_id) via index = (summary_set_ptr[summary_set_id] + summary_id, sidx)


        summary_reader = SummaryReader(
                summary_sets_id, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                loss_index, loss_summary, present_summary_id, summary_id_count_per_summary_set,
                item_id_to_risks_i, is_risk_affected, has_affected_risk)

        out_mv = memoryview(bytearray(PIPE_CAPACITY))
        out_byte_mv = np.frombuffer(buffer=out_mv, dtype='b')

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
            last_loss_summary_index = 0
            last_sidx = 0
            for summary_set_index, summary_set_id in enumerate(summary_sets_id):
                summary_pipe = summary_sets_pipe[summary_set_id]
                summary_index_cursor = 0
                while True:
                    cursor, loss_summary_index, last_sidx, summary_index_cursor = load_event(
                        out_byte_mv, event_id, len_sample, output_zeros, last_loss_summary_index, last_sidx,
                        summary_set_index, summary_set_index_to_loss_ptr, summary_id_count_per_summary_set, present_summary_id, loss_summary,
                        summary_index_cursor, summary_sets_cursor, summary_stream_index, has_affected_risk
                    )
                    written = 0
                    while written < cursor:
                        _, writable, exceptional =  select.select([],[summary_pipe], [summary_pipe])
                        if exceptional:
                            raise IOError(f'error with input stream, {exceptional}')
                        written += summary_pipe.write(out_mv[:cursor])
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
