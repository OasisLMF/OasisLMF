"""Summary stage: set up once, then aggregate a loss stream into summary streams one block of events at a time.

``SummaryStage`` holds everything ``summarypy`` builds before its event loop (the summary objects read from
the static files, the aggregation arrays and the reader) and processes events handed to it in memory. The
``summarypy`` command is built on it (see ``manager.run``), so both share one implementation of the per-event
work. The Numba kernels that read a loss stream into the aggregation arrays (``read_buffer``) and write one
event of a summary set (``mv_write_event``, ``fill_event_rows``) live here too.

Static intermediary data structure::

    summary_sets_id : array of all summary_set_id that have a stream
    summary_set_indices : array of the summary_set_index of each summary_set_id that has a stream
    summary_set_id_to_summary_set_index : array mapping summary_set_id => summary_set_index
    summary_set_index_to_loss_ptr : mapping array summary_set_index => starting index of summary_id loss (first dimension) in the loss_summary table
    item_id_to_summary_id : mapping array item_id => [summary_id for each summary set]
    item_id_to_risks_i : mapping array item_id => risks_i (index of th corresponding risk)
    risk_i_to_n_buildings : mapping array risks_i => number of buildings the risk stands for (1 unless packed)

event based intermediary data structure::

    summary_set_index_to_present_loss_ptr_end : copy of summary_set_index_to_loss_ptr that store the end of list of summary_id for each summary set
    present_summary_id : array that store all the summary_id found in the event
                         each summary set start at summary_set_index_to_loss_ptr and ends at summary_set_index_to_present_loss_ptr_end
    loss_index : store the index in loss_summary of each summary corresponding to an item for quick lookup on read
    loss_summary : 2D array of losses for each summary
                   loss_summary[loss_index[summary_set_index], sidx] = loss
    is_risk_affected : array to store in if a risk has already been affected or not in this event
"""
import json
import logging
import os

import numba as nb
import numpy as np
import pyarrow.parquet as pq
from numba.typed import Dict as nb_dict
from numpy.lib.recfunctions import repack_fields

from oasislmf.pytools.common.data import (building_count_dtype, def_to_type_and_size, load_as_ndarray, oasis_int, nb_oasis_int, oasis_float,
                                          null_index, fm_summary_xref_dtype, gul_summary_xref_dtype,
                                          loss_pair_dtype, loss_pair_size, summary_stream_index_dtype)
from oasislmf.pytools.common.event_stream import (EventReader, bytes_to_stream_types, stream_info_to_bytes,
                                                  mv_read, mv_write_summary_header, mv_write_sidx_loss,
                                                  GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID, SUMMARY_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY,
                                                  LOSS_STREAM_AGG_TYPES,
                                                  MEAN_IDX, TIV_IDX, NUMBER_OF_AFFECTED_RISK_IDX, MAX_LOSS_IDX,
                                                  NUM_SPECIAL_SIDX, decode_local_sidx)
from oasislmf.pytools.common.run_types import RUNTYPE_GROUNDUP_LOSS, RUNTYPE_INSURED_LOSS, RUNTYPE_REINSURANCE_LOSS, LOSS_RUNTYPES

logger = logging.getLogger(__name__)

STREAM_HEADER_SIZE = 8

event_id_dtype, event_id_dtype_size = def_to_type_and_size('event_id')
item_id_dtype, item_id_dtype_size = def_to_type_and_size('item_id')
loss_dtype, loss_dtype_size = def_to_type_and_size('loss')
summaryset_id_dtype, summaryset_id_dtype_size = def_to_type_and_size('summaryset_id')
_, summary_id_dtype_size = def_to_type_and_size('summary_id')

SPECIAL_SIDX_COUNT = 6  # 0 is included as a special sidx
SUMMARY_HEADER_SIZE = event_id_dtype_size + summary_id_dtype_size + loss_dtype_size + SPECIAL_SIDX_COUNT * (loss_pair_size)
SUMMARY_META_SIZE = 4 + 4 + summaryset_id_dtype_size  # (stream_type, sample size , summary_set_id)
SIDX_LOSS_WRITE_SIZE = 2 * (loss_pair_size)


risk_key_type = nb.types.UniTuple(nb_oasis_int, 2)

summary_info_dtype = np.dtype([('nb_risk', oasis_int), ])

summary_map_dtype = np.dtype([('loc_id', oasis_int), ('item_id', oasis_int), ('building_id', oasis_int)])
summary_map_with_buildings_dtype = np.dtype(summary_map_dtype.descr + [('number_of_buildings', building_count_dtype)])

SUPPORTED_RUN_TYPE = LOSS_RUNTYPES


def load_summary_object(static_path, run_type):
    """Load already prepare summary data structure if present otherwise create them"""
    if os.path.isfile(os.path.join(static_path, run_type, 'summary_info.npy')):
        return read_summary_objects(static_path, run_type)
    else:
        return get_summary_object(static_path, run_type)


def load_summary_map(static_path, name, col_map=None):
    """Load a summary map, with its ``number_of_buildings`` column when it carries one.

    The column is written under building packing, where one row stands for that many buildings.
    A map written before packing existed, or assembled by hand, does not have it.

    The columns are looked up in the file ``load_as_ndarray`` reads, in its order of preference. A
    ``.bin`` has no header to look in and is read without the column.
    """
    dtype = summary_map_dtype
    path = os.path.join(static_path, name)
    if not os.path.isfile(path + '.bin'):
        columns = set()
        if os.path.isfile(path + '.parquet'):
            columns = set(pq.read_schema(path + '.parquet').names)
        elif os.path.isfile(path + '.csv'):
            with open(path + '.csv') as f:
                columns = {h.strip() for h in f.readline().split(',')}
        if 'number_of_buildings' in columns:
            dtype = summary_map_with_buildings_dtype
    return load_as_ndarray(static_path, name, dtype, col_map=col_map)


def get_summary_object(static_path, run_type):
    """Read static files to get summary static data structure"""
    # extract item_id to index in the loss summary
    if run_type == RUNTYPE_GROUNDUP_LOSS:
        summary_xref = load_as_ndarray(static_path, 'gulsummaryxref', gul_summary_xref_dtype)
        summary_map = load_summary_map(static_path, 'gul_summary_map')

    elif run_type == RUNTYPE_INSURED_LOSS:
        summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', fm_summary_xref_dtype)
        summary_xref = summary_xref.astype(gul_summary_xref_dtype)  # Change dtype to keep consistent column names
        summary_map = load_summary_map(static_path, 'fm_summary_map', col_map={'item_id': 'output_id'})

    elif run_type == RUNTYPE_REINSURANCE_LOSS:
        summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', fm_summary_xref_dtype)
        summary_xref = summary_xref.astype(gul_summary_xref_dtype)  # Change dtype to keep consistent column names
        summary_map = None  # numba use none to optimise function when some part are not used

    else:
        raise Exception(f"run type {run_type} not in supported list {SUPPORTED_RUN_TYPE}")

    summary_sets_id = np.sort(np.unique(summary_xref['summaryset_id']))
    summary_set_id_to_summary_set_index = get_summary_set_id_to_summary_set_index(summary_sets_id)
    summary_set_index_to_loss_ptr, item_id_to_summary_id = get_summary_xref_info(summary_xref, summary_sets_id, summary_set_id_to_summary_set_index)

    if summary_map is not None:
        nb_risk, item_id_to_risks_i, risk_i_to_n_buildings = extract_risk_info(item_id_to_summary_id.shape[0], summary_map)
    else:
        item_id_to_risks_i = np.zeros(0, dtype=oasis_int)
        risk_i_to_n_buildings = np.zeros(0, dtype=building_count_dtype)
        nb_risk = 0

    summary_info = np.empty(1, dtype=summary_info_dtype)
    info = summary_info[0]
    info['nb_risk'] = nb_risk

    return (summary_info, summary_set_id_to_summary_set_index, summary_set_index_to_loss_ptr, item_id_to_summary_id,
            item_id_to_risks_i, risk_i_to_n_buildings)

def read_summary_objects(static_path, run_type):
    summary_info = np.load(os.path.join(static_path, run_type, 'summary_info.npy'), mmap_mode='r')
    summary_set_id_to_summary_set_index = np.load(os.path.join(static_path, run_type, 'summary_set_id_to_summary_set_index.npy'), mmap_mode='r')
    summary_set_index_to_loss_ptr = np.load(os.path.join(static_path, run_type, 'summary_set_index_to_loss_ptr.npy'), mmap_mode='r')
    item_id_to_summary_id = np.load(os.path.join(static_path, run_type, 'item_id_to_summary_id.npy'), mmap_mode='r')
    item_id_to_risks_i = np.load(os.path.join(static_path, run_type, 'item_id_to_risks_i.npy'), mmap_mode='r')
    n_buildings_path = os.path.join(static_path, run_type, 'risk_i_to_n_buildings.npy')
    if os.path.isfile(n_buildings_path):
        risk_i_to_n_buildings = np.load(n_buildings_path, mmap_mode='r')
    else:  # objects prepared before building packing: every risk is one building
        risk_i_to_n_buildings = np.ones(summary_info[0]['nb_risk'], dtype=building_count_dtype)
    return (summary_info, summary_set_id_to_summary_set_index, summary_set_index_to_loss_ptr, item_id_to_summary_id,
            item_id_to_risks_i, risk_i_to_n_buildings)


@nb.njit(cache=True)
def nb_extract_risk_info(item_id_to_risks_i, risk_i_to_n_buildings, summary_map, row_n_buildings):
    loc_id_building_id_to_building_risk = nb_dict.empty(risk_key_type, nb_oasis_int)
    last_risk_i = 0
    for i in range(summary_map.shape[0]):
        loc_id_building_id = (nb_oasis_int(summary_map["loc_id"][i]), nb_oasis_int(summary_map["building_id"][i]))
        if loc_id_building_id in loc_id_building_id_to_building_risk:
            risk_i = loc_id_building_id_to_building_risk[loc_id_building_id]
        else:
            loc_id_building_id_to_building_risk[loc_id_building_id] = risk_i = nb_oasis_int(last_risk_i)
            # the rows of one risk share its location, so they carry the same count
            risk_i_to_n_buildings[risk_i] = row_n_buildings[i]
            last_risk_i += 1
        item_id_to_risks_i[summary_map["item_id"][i]] = nb_oasis_int(risk_i)
    return last_risk_i


def extract_risk_info(len_item_id, summary_map):
    """Extract relevant information regarding item and risk mapping from summary_map

    Args:
        len_item_id: number of items
        summary_map: numpy ndarray view of the summary_map

    Returns:
        (number of risk, mapping array item_id => risks_i, mapping array risks_i => number of buildings)
    """
    item_id_to_risks_i = np.zeros(len_item_id, oasis_int)
    risk_i_to_n_buildings = np.ones(summary_map.shape[0], building_count_dtype)
    if 'number_of_buildings' in summary_map.dtype.names:
        row_n_buildings = np.maximum(np.asarray(summary_map['number_of_buildings'], dtype=building_count_dtype), 1)
    else:
        row_n_buildings = np.ones(summary_map.shape[0], building_count_dtype)
    nb_risk = nb_extract_risk_info(
        item_id_to_risks_i,
        risk_i_to_n_buildings,
        repack_fields(summary_map[['loc_id', 'item_id', 'building_id']]),
        row_n_buildings)
    return nb_risk, item_id_to_risks_i, risk_i_to_n_buildings[:nb_risk]


@nb.jit(nopython=True, cache=True)
def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id,
                summary_set_indices, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                loss_index, loss_summary, present_summary_id, summary_set_index_to_present_loss_ptr_end,
                item_id_to_risks_i, risk_i_to_n_buildings, is_risk_affected, has_affected_risk):
    """Read valid part of byte_mv and load relevant data for one event"""
    last_event_id = event_id
    # A summary is per item, so a packed item's buildings share a bucket: decode onto the sample the
    # index represents and let the accumulation below add them. Identity for an ordinary stream.
    # Its affected risks are counted by building too, as row disaggregation counts them: a packed
    # risk stands for risk_i_to_n_buildings of them.
    max_sidx_val = loss_summary.shape[1] - SPECIAL_SIDX_COUNT
    while True:
        if item_id:
            n_sidx_loss = (valid_buff - cursor) // loss_pair_size
            if n_sidx_loss == 0:
                break
            sidx_loss_view = byte_mv[cursor:cursor + n_sidx_loss * loss_pair_size].view(loss_pair_dtype)
            terminated = False
            for k in range(n_sidx_loss):
                sidx = sidx_loss_view[k]['sidx']
                if not sidx:
                    ##### do item exit ####
                    ##########
                    cursor += (k + 1) * loss_pair_size
                    item_id = 0
                    terminated = True
                    break
                loss = sidx_loss_view[k]['loss']
                loss = 0 if np.isnan(loss) else loss

                ###### do loss read ######
                # Only the positive branch divides by the sample size, and a zero-sample stream carries no
                # positive sidx -- but it does carry per-building specials, which must still be decoded.
                if (max_sidx_val > 0 and sidx > max_sidx_val) or sidx < -NUM_SPECIAL_SIDX:
                    sidx = decode_local_sidx(sidx, max_sidx_val)
                if sidx > 0 or sidx in [-1, -3, -5]:
                    for summary_set_index in summary_set_indices:
                        loss_summary[loss_index[summary_set_index], sidx] += loss
                ##########
            if not terminated:
                cursor += n_sidx_loss * loss_pair_size
        else:
            if valid_buff - cursor < (event_id_dtype_size + item_id_dtype_size):
                break
            event_id, cursor = mv_read(byte_mv, cursor, event_id_dtype, event_id_dtype_size)
            if event_id != last_event_id:
                if last_event_id:  # we have a new event we return the one we just finished
                    return cursor - event_id_dtype_size, last_event_id, 0, 1
                else:  # first pass we store the event we are reading
                    last_event_id = event_id
            item_id, cursor = mv_read(byte_mv, cursor, item_id_dtype, item_id_dtype_size)

            ##### do new item setup #####
            if has_affected_risk is not None:
                risk_i = item_id_to_risks_i[item_id]
                risk_count = risk_i_to_n_buildings[risk_i]
                if is_risk_affected[risk_i]:
                    new_risk = 0
                else:
                    is_risk_affected[risk_i] = 1
                    new_risk = risk_count
            else:
                risk_count = 1

            for summary_set_index in summary_set_indices:
                if item_id >= item_id_to_summary_id.shape[0] or item_id_to_summary_id[item_id, summary_set_index] == null_index:
                    raise ValueError('summarypy: item_id', item_id, 'has no summaryxref row for summary set index', summary_set_index)
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
                    loss_summary[loss_index[summary_set_index], NUMBER_OF_AFFECTED_RISK_IDX] = risk_count

                elif has_affected_risk is not None:
                    loss_summary[loss_index[summary_set_index], NUMBER_OF_AFFECTED_RISK_IDX] += new_risk
            ##########
    # Buffer exhausted; caller may re-enter to accumulate more sidx_loss for this event.
    # SummaryReader.finalize_event handles the per-event sort once, just before yield.
    return cursor, event_id, item_id, 0


@nb.njit(cache=True)
def mv_write_event(byte_mv, event_id, len_sample, last_loss_summary_index, last_sidx,
                   output_zeros, has_affected_risk,
                   summary_set_index, summary_set_index_to_loss_ptr, summary_set_index_to_present_loss_ptr_end, present_summary_id, loss_summary,
                   summary_index_cursor, summary_sets_cursor, summary_stream_index):
    """Load event summary loss into byte_mv

    Args:
        byte_mv: numpy byte view to write to the stream
        event_id: event id
        len_sample: max sample id
        last_loss_summary_index: last summary index written (used to restart from the last summary when buffer was full)
        last_sidx: last sidx written in the buffer (used to restart from the correct sidx when buffer was full
        output_zeros: if False, summaries and samples with a zero loss are skipped
        has_affected_risk: None when the number of affected risks is not tracked, otherwise the affected risk data
        summary_set_index: index of the summary set being written
        summary_set_index_to_loss_ptr: start offset of each summary set in loss_summary
        summary_set_index_to_present_loss_ptr_end: end offset of each summary set in present_summary_id
        present_summary_id: summary ids present in this event, per summary set
        loss_summary: the loss values to write, indexed by summary set offset and summary id
        summary_index_cursor: next free slot in summary_stream_index
        summary_sets_cursor: running byte offset written so far for each summary set
        summary_stream_index: index records (summary_id, offset) written alongside the stream

    Returns:
        the cursor reached in byte_mv, the summary index to resume from (-1 when the event is
        complete), the sidx to resume from, and the updated summary_index_cursor
    """
    cursor = 0
    for loss_summary_index in range(max(summary_set_index_to_loss_ptr[summary_set_index], last_loss_summary_index),
                                    summary_set_index_to_present_loss_ptr_end[summary_set_index]):
        summary_id = present_summary_id[loss_summary_index]
        losses = loss_summary[summary_set_index_to_loss_ptr[summary_set_index] + summary_id - 1]

        if not output_zeros and losses[TIV_IDX] == 0 and losses[MEAN_IDX] == 0:
            continue

        summary_stream_index[summary_index_cursor]['summary_id'] = summary_id
        # we use offset to temporally store the cursor, we set the correct value later on
        summary_stream_index[summary_index_cursor]['offset'] = cursor

        if last_sidx == 0:
            if cursor < PIPE_CAPACITY - SUMMARY_HEADER_SIZE:
                cursor = mv_write_summary_header(byte_mv, cursor, event_id, summary_id, losses[TIV_IDX])
                cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, losses[MAX_LOSS_IDX])
                if has_affected_risk is not None:
                    cursor = mv_write_sidx_loss(byte_mv, cursor, NUMBER_OF_AFFECTED_RISK_IDX, losses[NUMBER_OF_AFFECTED_RISK_IDX])
                cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, losses[MEAN_IDX])
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

        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)
        # set the correct offset for idx file and update summary_sets_cursor
        summary_byte_len = cursor - summary_stream_index[summary_index_cursor]['offset']
        summary_stream_index[summary_index_cursor]['offset'] = summary_sets_cursor[summary_set_index]
        summary_sets_cursor[summary_set_index] += summary_byte_len
        summary_index_cursor += 1

        last_sidx = 0
    return cursor, -1, 0, summary_index_cursor


@nb.njit(cache=True)
def write_selt_row(rows, row_i, event_id, summary_id, sidx, loss, impacted_exposure):
    row = rows[row_i]
    row['EventId'] = event_id
    row['SummaryId'] = summary_id
    row['SampleId'] = sidx
    row['Loss'] = loss
    row['ImpactedExposure'] = impacted_exposure
    return row_i + 1


@nb.njit(cache=True)
def fill_event_rows(rows, row_i, event_id, len_sample, last_loss_summary_index, last_sidx,
                    output_zeros, has_affected_risk,
                    summary_set_index, summary_set_index_to_loss_ptr, summary_set_index_to_present_loss_ptr_end, present_summary_id,
                    loss_summary):
    """Load event summary loss into a table of SELT rows, in the same order as the binary stream.

    Each record of the binary summary stream becomes one (EventId, SummaryId, SampleId, Loss, ImpactedExposure)
    row, the special sidx (max loss, number of affected risks, mean) included, and zero losses are skipped
    under the same rules as mv_write_event.

    Args:
        rows: SELT_dtype array to fill
        row_i: index of the next free row in rows
        event_id: event id
        len_sample: max sample id
        last_loss_summary_index: summary index to resume from when rows was full on the previous call
        last_sidx: sidx to resume from when rows was full on the previous call
        output_zeros: if False, summaries and samples with a zero loss are skipped
        has_affected_risk: None when the number of affected risks is not tracked, otherwise the affected risk data
        summary_set_index: index of the summary set being written
        summary_set_index_to_loss_ptr: start offset of each summary set in loss_summary
        summary_set_index_to_present_loss_ptr_end: end offset of each summary set in present_summary_id
        present_summary_id: summary ids present in this event, per summary set
        loss_summary: the loss values to write, indexed by summary set offset and summary id

    Returns:
        the next free row in rows, the summary index to resume from (-1 when the event is complete)
        and the sidx to resume from
    """
    n_header_rows = 3 if has_affected_risk is not None else 2
    for loss_summary_index in range(max(summary_set_index_to_loss_ptr[summary_set_index], last_loss_summary_index),
                                    summary_set_index_to_present_loss_ptr_end[summary_set_index]):
        summary_id = present_summary_id[loss_summary_index]
        losses = loss_summary[summary_set_index_to_loss_ptr[summary_set_index] + summary_id - 1]
        tiv = losses[TIV_IDX]

        if not output_zeros and tiv == 0 and losses[MEAN_IDX] == 0:
            continue

        if last_sidx == 0:
            if row_i + n_header_rows > rows.shape[0]:
                return row_i, loss_summary_index, last_sidx
            row_i = write_selt_row(rows, row_i, event_id, summary_id, MAX_LOSS_IDX, losses[MAX_LOSS_IDX], tiv)
            if has_affected_risk is not None:
                row_i = write_selt_row(rows, row_i, event_id, summary_id, NUMBER_OF_AFFECTED_RISK_IDX,
                                       losses[NUMBER_OF_AFFECTED_RISK_IDX], tiv)
            row_i = write_selt_row(rows, row_i, event_id, summary_id, MEAN_IDX, losses[MEAN_IDX], tiv)
            last_sidx = 1

        for sidx in range(last_sidx, len_sample + 1):
            if not output_zeros and losses[sidx] == 0:
                continue
            if row_i >= rows.shape[0]:
                return row_i, loss_summary_index, sidx
            row_i = write_selt_row(rows, row_i, event_id, summary_id, sidx, losses[sidx], tiv)

        last_sidx = 0
    return row_i, -1, 0

class SummaryReader(EventReader):
    """Read an event loss stream and aggregate it into the relevant summary loss streams."""

    def __init__(self, summary_set_indices, summary_set_index_to_loss_ptr, item_id_to_summary_id,
                 loss_index, loss_summary, present_summary_id, summary_set_index_to_present_loss_ptr_end,
                 item_id_to_risks_i, risk_i_to_n_buildings, is_risk_affected, has_affected_risk):
        self.summary_set_indices = summary_set_indices
        self.summary_set_index_to_loss_ptr = summary_set_index_to_loss_ptr
        self.item_id_to_summary_id = item_id_to_summary_id
        self.loss_index = loss_index
        self.loss_summary = loss_summary
        self.present_summary_id = present_summary_id
        self.summary_set_index_to_present_loss_ptr_end = summary_set_index_to_present_loss_ptr_end
        self.item_id_to_risks_i = item_id_to_risks_i
        self.risk_i_to_n_buildings = risk_i_to_n_buildings
        self.is_risk_affected = is_risk_affected
        self.has_affected_risk = has_affected_risk
        self.logger = logger

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        return read_buffer(
            byte_mv, cursor, valid_buff, event_id, item_id,
            self.summary_set_indices, self.summary_set_index_to_loss_ptr, self.item_id_to_summary_id,
            self.loss_index, self.loss_summary, self.present_summary_id, self.summary_set_index_to_present_loss_ptr_end,
            self.item_id_to_risks_i, self.risk_i_to_n_buildings, self.is_risk_affected, self.has_affected_risk
        )

    def finalize_event(self, **kwargs):
        # Sort the slice of summary_ids seen this event so mv_write_event emits in summary_id order.
        for summary_set_index in self.summary_set_indices:
            start = self.summary_set_index_to_loss_ptr[summary_set_index]
            end = self.summary_set_index_to_present_loss_ptr_end[summary_set_index]
            self.present_summary_id[start:end] = np.sort(self.present_summary_id[start:end])


def get_summary_set_id_to_summary_set_index(summary_sets_id):
    """Create an array mapping summary_set_id => summary_set_index"""
    summary_set_id_to_summary_set_index = np.full(np.max(summary_sets_id) + 1, null_index, 'i4')
    for summary_set_index in range(summary_sets_id.shape[0]):
        summary_set_id_to_summary_set_index[summary_sets_id[summary_set_index]] = summary_set_index
    return summary_set_id_to_summary_set_index


def get_summary_xref_info(summary_xref, summary_sets_id, summary_set_id_to_summary_set_index):
    """Extract mapping from summary_xref"""
    summary_set_index_to_loss_ptr = np.zeros(summary_sets_id.shape[0] + 1, oasis_int)
    max_item_id = 0
    for i in range(summary_xref.shape[0]):
        xref = summary_xref[i]
        summary_set_index = summary_set_id_to_summary_set_index[xref['summaryset_id']]
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
        summary_set_index = summary_set_id_to_summary_set_index[xref['summaryset_id']]
        if summary_set_index == null_index:
            continue
        item_id_to_summary_id[xref['item_id'], summary_set_index] = xref['summary_id']

    return summary_set_index_to_loss_ptr, item_id_to_summary_id


def parse_input_header(input_header):
    """Validate the header of a summarypy input stream as summarypy does and return its sample count.

    Args:
        input_header (bytes): the 8-byte header of the upstream loss stream (gul item stream, il/ri loss
            stream, plain or with packed buildings).

    Returns:
        np.int32: the sample count (max sidx) of the stream.

    Raises:
        Exception: the stream type or the aggregation type is not one summarypy reads.
    """
    input_header = bytes(input_header)
    if len(input_header) != STREAM_HEADER_SIZE:
        raise ValueError(f'stream header must be {STREAM_HEADER_SIZE} bytes, got {len(input_header)}')
    stream_source_type, stream_agg_type = bytes_to_stream_types(input_header[:4])
    len_sample = np.frombuffer(input_header[4:], dtype=np.int32)[0]

    if stream_source_type not in (GUL_STREAM_ID, FM_STREAM_ID, LOSS_STREAM_ID):
        raise Exception(f"unsupported stream type {stream_source_type}, {stream_agg_type}")
    # summarypy sums over the sidx it is given, so a packed stream needs no second path -- but
    # an aggregation type we have not defined must be refused rather than silently summed.
    if stream_agg_type not in LOSS_STREAM_AGG_TYPES:
        raise Exception(f"unsupported stream aggregation type {stream_agg_type}, expected one of "
                        f"{LOSS_STREAM_AGG_TYPES} (item, or item with packed buildings)")
    return len_sample


class SummaryStage:
    """One summarypy step (one run type), set up once and fed events in memory.

    For every event it aggregates the item losses into the summaries of each requested summary set and
    writes, per set, exactly the bytes summarypy writes for that event to the set's output.

    The aggregation keeps no state from one event to the next: everything an event loads is cleared once
    its output is written. The only running state is the byte offset of each set's output, kept for the
    ``.idx`` records of ``low_memory`` mode, which are offsets from the start of the output file (header
    included). ``reset()`` restarts them for a new file.

    Paths resolve as in ``summarypy``, so ``static_path`` is relative to the working directory.

    Args:
        input_header (bytes): the 8-byte header of the upstream loss stream. It is validated as summarypy
            validates its input and gives the sample count.
        static_path (str): folder holding the summary static files (``input``, or ``input/RI_<n>`` for ri).
        run_type (str): ``gul``, ``il`` or ``ri``.
        summary_set_ids (iterable of int): the summary sets to write, all defined in summaryxref.
        low_memory (bool): also write, per event and set, the ``.idx`` records summarypy ``-m`` writes.
        output_zeros (bool): write summaries and samples with a zero loss (summarypy ``-z``).
        **kwargs: ignored, so that the summarypy command-line options can be passed through.

    Attributes:
        headers (dict): summary_set_id -> the 12-byte summary stream header summarypy writes before the
            first event of that set (it carries the set id, so it differs per set).
        reader (SummaryReader): the reader loading events into the aggregation arrays.
        len_sample (np.int32): the sample count of the input and output streams.
    """

    def __init__(self, input_header, static_path, run_type, summary_set_ids, low_memory=False,
                 output_zeros=False, **kwargs):
        len_sample = parse_input_header(input_header)
        summary_sets_id = np.array(list(summary_set_ids))

        (summary_info, summary_set_id_to_summary_set_index, summary_set_index_to_loss_ptr, item_id_to_summary_id,
         item_id_to_risks_i, risk_i_to_n_buildings) = load_summary_object(static_path, run_type)

        invalid_summary_sets_id = []
        for summary_set_id in summary_sets_id:
            if summary_set_id < 1 or summary_set_id >= summary_set_id_to_summary_set_index.shape[0] or summary_set_id_to_summary_set_index[summary_set_id] == null_index:
                invalid_summary_sets_id.append(summary_set_id)
        if invalid_summary_sets_id:
            raise ValueError(
                f'summary_set_ids {invalid_summary_sets_id} not found in summaryxref available summary_set_id '
                f'{[summary_set_id for summary_set_id, summary_set_index in enumerate(summary_set_id_to_summary_set_index) if summary_set_index != null_index]}')

        self.len_sample = len_sample
        self.low_memory = low_memory
        self.output_zeros = output_zeros
        self.has_affected_risk = True if summary_info[0]['nb_risk'] > 0 else None

        # losses are read and written by each requested set's own summary_set_index (its position in
        # summaryxref), so the order of the requested sets doesn't change what is written for each set,
        # and sets that weren't requested are never touched
        self.summary_set_ids = [int(summary_set_id) for summary_set_id in summary_sets_id]
        self.summary_set_indices = summary_set_id_to_summary_set_index[summary_sets_id]
        self.summary_set_index_to_loss_ptr = summary_set_index_to_loss_ptr
        nb_xref_summary_sets = summary_set_index_to_loss_ptr.shape[0] - 1

        self.present_summary_id = np.zeros(summary_set_index_to_loss_ptr[-1], oasis_int)
        loss_index = np.empty(nb_xref_summary_sets, oasis_int)
        self.summary_set_index_to_present_loss_ptr_end = np.array(summary_set_index_to_loss_ptr)
        self.loss_summary = np.zeros((summary_set_index_to_loss_ptr[-1], len_sample + SPECIAL_SIDX_COUNT), dtype=oasis_float)
        self.is_risk_affected = np.zeros(summary_info[0]['nb_risk'], dtype=oasis_int)

        self.reader = SummaryReader(
            self.summary_set_indices, summary_set_index_to_loss_ptr, item_id_to_summary_id,
            loss_index, self.loss_summary, self.present_summary_id, self.summary_set_index_to_present_loss_ptr_end,
            item_id_to_risks_i, risk_i_to_n_buildings, self.is_risk_affected, self.has_affected_risk)

        self.out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

        self.headers = {
            summary_set_id: (stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM) + len_sample.tobytes()
                             + summaryset_id_dtype.type(summary_set_id).tobytes())
            for summary_set_id in self.summary_set_ids
        }

        # data for the index file (low_memory)
        self.summary_sets_cursor = np.zeros(nb_xref_summary_sets, dtype=np.int64)
        self.summary_stream_index = np.empty(summary_set_index_to_loss_ptr[-1], dtype=summary_stream_index_dtype)
        self.event_id = None
        self.reset()

    @property
    def header(self):
        """bytes: the summary stream header, when the stage writes a single summary set (see ``headers``)."""
        if len(self.headers) != 1:
            raise ValueError(f"the stage writes {len(self.headers)} summary sets, each with its own header: use headers[summary_set_id]")
        return next(iter(self.headers.values()))

    def process_block(self, block, outs, idx_outs=None, row_writers=None):
        """Aggregate each event of an in-memory block, in order, and write it to every summary set's output.

        Args:
            block (np.ndarray): 1-D int8 array holding one or more whole events of the input loss stream,
                without the stream header. It may be read-only.
            outs (dict): summary_set_id -> sink for the binary summary stream of that set, for every set.
            idx_outs (dict, optional): summary_set_id -> sink for the ``.idx`` records of that set; required
                when the stage was built with ``low_memory``, ignored otherwise.
            row_writers (dict, optional): summary_set_id -> ``manager.SummaryRowWriter``; when given, each set
                is written as SELT rows instead of a binary stream and ``outs`` is ignored.
        """
        self.process_events(self.reader.read_block(block), outs, idx_outs, row_writers)

    def process_events(self, events, outs=None, idx_outs=None, row_writers=None):
        """Write each event yielded by ``events`` to every summary set's output.

        ``events`` must be an iterator over event ids produced by ``self.reader`` (``read_block`` or
        ``read_streams``), which loads each event into the aggregation arrays before yielding it. On failure
        the last event read is recorded in ``event_error.json`` in the working directory.

        Args:
            events: iterator of event ids loaded by ``self.reader``.
            outs (dict): summary_set_id -> sink for the binary summary stream of that set.
            idx_outs (dict, optional): summary_set_id -> sink for the ``.idx`` records (``low_memory``).
            row_writers (dict, optional): summary_set_id -> ``manager.SummaryRowWriter`` for csv/parquet output.
        """
        if row_writers is None:
            if outs is None:
                raise ValueError("outs is required for binary output")
            if self.low_memory and idx_outs is None:
                raise ValueError("idx_outs is required when the stage is built with low_memory")
        try:
            for event_id in events:
                self.event_id = event_id
                if row_writers is None:
                    self._write_event(event_id, outs, idx_outs)
                else:
                    self._write_event_rows(event_id, row_writers)
                self._clear_event()
        except Exception:
            data = {
                "event_id": self.event_id
            }
            with open("event_error.json", "w") as f:
                json.dump(data, f, default=str)

            logger.error(f"event id={self.event_id} failed in summary")
            raise

    def _write_event(self, event_id, outs, idx_outs):
        """Write the loaded event to the binary output (and ``.idx`` records) of each summary set."""
        out_byte_mv = self.out_byte_mv
        for summary_set_id, summary_set_index in zip(self.summary_set_ids, self.summary_set_indices):
            out = outs[summary_set_id]
            summary_index_cursor = 0
            last_loss_summary_index = 0
            last_sidx = 0
            while True:
                cursor, last_loss_summary_index, last_sidx, summary_index_cursor = mv_write_event(
                    out_byte_mv, event_id, self.len_sample, last_loss_summary_index, last_sidx,
                    self.output_zeros, self.has_affected_risk,
                    summary_set_index, self.summary_set_index_to_loss_ptr, self.summary_set_index_to_present_loss_ptr_end,
                    self.present_summary_id, self.loss_summary,
                    summary_index_cursor, self.summary_sets_cursor, self.summary_stream_index
                )
                if cursor:
                    out.write(out_byte_mv[:cursor])
                if last_loss_summary_index == -1:
                    break
            if self.low_memory:
                idx_outs[summary_set_id].write(self.summary_stream_index[:summary_index_cursor].tobytes())

    def _write_event_rows(self, event_id, row_writers):
        """Write the loaded event as SELT rows to the row writer of each summary set."""
        for summary_set_id, summary_set_index in zip(self.summary_set_ids, self.summary_set_indices):
            writer = row_writers[summary_set_id]
            last_loss_summary_index = 0
            last_sidx = 0
            while True:
                writer.row_i, last_loss_summary_index, last_sidx = fill_event_rows(
                    writer.rows, writer.row_i, event_id, self.len_sample, last_loss_summary_index, last_sidx,
                    self.output_zeros, self.has_affected_risk,
                    summary_set_index, self.summary_set_index_to_loss_ptr, self.summary_set_index_to_present_loss_ptr_end,
                    self.present_summary_id, self.loss_summary
                )
                if last_loss_summary_index == -1:
                    break
                writer.flush()

    def _clear_event(self):
        """Clear what the event just written loaded into the aggregation arrays."""
        self.loss_summary.fill(0)
        self.present_summary_id.fill(0)
        self.summary_set_index_to_present_loss_ptr_end[:] = self.summary_set_index_to_loss_ptr
        self.is_risk_affected.fill(0)

    def reset(self):
        """Make the stage behave as a freshly built one for the events that follow.

        Restarts the ``.idx`` byte offsets of every set just after its header, for a new output file, and
        clears the aggregation arrays (which only an event that failed part way can leave dirty).
        """
        self.summary_sets_cursor.fill(0)
        for summary_set_id, summary_set_index in zip(self.summary_set_ids, self.summary_set_indices):
            self.summary_sets_cursor[summary_set_index] = len(self.headers[summary_set_id])
        self._clear_event()
        self.event_id = None

    def close(self):
        """Release nothing: the stage holds only memory. Present for the stage contract."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
