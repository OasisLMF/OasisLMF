"""WIP draft: EventReader based reader for the rl inuring-priority summary streams.

See manager.py for the stream format and design notes.

SummaryCombineReader only reads: it yields each event as soon as one input (inuring priority)
has sent it in full, closed by its E:S0 sentinel. Combining the inputs is left to the caller.
"""
import logging

import numba as nb
import numpy as np

from oasislmf.pytools.common.data import def_to_type_and_size, fm_summary_xref_dtype, load_as_ndarray
from oasislmf.pytools.common.event_stream import EventReader, init_streams_in, mv_read, ITEM_STREAM, SUMMARY_STREAM_ID

logger = logging.getLogger(__name__)

event_id_dtype, event_id_dtype_size = def_to_type_and_size('event_id')
summary_id_dtype, summary_id_dtype_size = def_to_type_and_size('summary_id')
summaryset_id_dtype, summaryset_id_dtype_size = def_to_type_and_size('summaryset_id')
sidx_dtype, sidx_dtype_size = def_to_type_and_size('sidx')
loss_dtype, loss_dtype_size = def_to_type_and_size('loss')

RECORD_HEADER_SIZE = event_id_dtype_size + summary_id_dtype_size + loss_dtype_size
PAIR_SIZE = sidx_dtype_size + loss_dtype_size
SENTINEL_SUMMARY_ID = 0
MAX_SPECIAL_SIDX = 3  # special sidx summarypy can write per record: -5 (max loss), -4 (affected risks), -1 (mean)

read_state_dtype = np.dtype([
    ('event_id', event_id_dtype),      # event currently being read on this input (0 = between events)
    ('summary_id', summary_id_dtype),  # record whose pairs are being read (0 = expect a record header)
    ('n_records', np.int64),           # records of the current event stored so far
    ('n_pairs', np.int64),             # (sidx, loss) pairs of the current event stored so far
])


class CombineStreamError(Exception):
    pass


def init_summary_streams_in(files_in, stack):
    """Open the input summary streams and check that their headers agree.

    init_streams_in only reads the first 8 bytes of the header (stream type, len_sample),
    the summary stream header also carries the summary_set_id that is read here.

    Args:
        files_in: list of input summary stream paths
        stack: contextlib ExitStack the streams are added to

    Returns:
        (streams_in, len_sample, summary_set_id)
    """
    streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
    if stream_source_type != SUMMARY_STREAM_ID or stream_agg_type != ITEM_STREAM:
        raise CombineStreamError(f"unsupported stream type {stream_source_type}, {stream_agg_type}, expected summary stream")

    summary_set_ids = [np.frombuffer(stream_in.read(summaryset_id_dtype_size), summaryset_id_dtype)[0] for stream_in in streams_in]
    if len(set(summary_set_ids)) > 1:
        raise CombineStreamError(f"input streams have different summary_set_id {summary_set_ids}")

    return streams_in, len_sample, summary_set_ids[0]


def get_max_summary_id(static_path, summary_set_id):
    """Largest summary_id of summary_set_id in the fmsummaryxref file of static_path.

    An event holds at most one record per summary_id, so this bounds the records per event.
    """
    summary_xref = load_as_ndarray(static_path, 'fmsummaryxref', fm_summary_xref_dtype)
    summary_ids = summary_xref['summary_id'][summary_xref['summaryset_id'] == summary_set_id]
    if summary_ids.shape[0] == 0:
        raise CombineStreamError(f"summary_set_id {summary_set_id} not found in fmsummaryxref in {static_path}")
    return int(summary_ids.max())


class EventBuffer:
    """Records of the event being read on one input, in CSR form.

    Record i is summary_ids[i] with expval[i], and its pairs are
    sidx[offsets[i]:offsets[i + 1]], loss[offsets[i]:offsets[i + 1]].

    Sized for the largest possible event: one record per summary_id, each with every sample
    plus the special sidx.
    """

    def __init__(self, max_summary_id, len_sample):
        max_pairs = max_summary_id * (len_sample + MAX_SPECIAL_SIDX)
        self.summary_ids = np.empty(max_summary_id, dtype=summary_id_dtype)
        self.expval = np.empty(max_summary_id, dtype=loss_dtype)
        self.offsets = np.empty(max_summary_id + 1, dtype=np.int64)
        self.sidx = np.empty(max_pairs, dtype=sidx_dtype)
        self.loss = np.empty(max_pairs, dtype=loss_dtype)


@nb.njit(cache=True)
def read_buffer(byte_mv, cursor, valid_buff, state, summary_ids, expval, offsets, sidx, loss):
    """Read the valid part of byte_mv for one input and store the records of the current event.

    The arrays are sized for the largest possible event (see EventBuffer), so running out of
    room means the stream breaks that bound and is reported as an error.

    Returns:
        cursor, event_id, summary_id (non zero if a record is only partly read), yield_event
        yield_event is 1 right after the E:S0 sentinel, so an event is known to be complete
        without reading ahead into the next event.
    """
    while True:
        if state['summary_id']:  # inside a record: read (sidx, loss) pairs
            if valid_buff - cursor < PAIR_SIZE:
                return cursor, state['event_id'], state['summary_id'], 0
            pair_sidx, cursor = mv_read(byte_mv, cursor, sidx_dtype, sidx_dtype_size)
            pair_loss, cursor = mv_read(byte_mv, cursor, loss_dtype, loss_dtype_size)
            if pair_sidx == 0:
                state['summary_id'] = 0
                continue
            if state['n_pairs'] - offsets[state['n_records'] - 1] == sidx.shape[0] // summary_ids.shape[0]:
                raise ValueError('rlcombine: more (sidx, loss) pairs in a record than len_sample + special sidx allow')
            sidx[state['n_pairs']] = pair_sidx
            loss[state['n_pairs']] = pair_loss
            state['n_pairs'] += 1

        else:  # expect a record header (a sentinel needs its (0, 0) pair too)
            if valid_buff - cursor < RECORD_HEADER_SIZE + PAIR_SIZE:
                return cursor, state['event_id'], 0, 0
            record_event_id, cursor = mv_read(byte_mv, cursor, event_id_dtype, event_id_dtype_size)
            summary_id, cursor = mv_read(byte_mv, cursor, summary_id_dtype, summary_id_dtype_size)
            record_expval, cursor = mv_read(byte_mv, cursor, loss_dtype, loss_dtype_size)

            if state['event_id'] == 0:
                state['event_id'] = record_event_id
            elif record_event_id != state['event_id']:
                raise ValueError('rlcombine: event not closed by an S0 sentinel record')

            if summary_id == SENTINEL_SUMMARY_ID:
                pair_sidx, cursor = mv_read(byte_mv, cursor, sidx_dtype, sidx_dtype_size)
                cursor += loss_dtype_size
                if pair_sidx != 0:
                    raise ValueError('rlcombine: S0 sentinel record must only hold the (0, 0) terminator')
                offsets[state['n_records']] = state['n_pairs']
                event_id = state['event_id']
                state['event_id'] = 0
                return cursor, event_id, 0, 1

            if summary_id < 1 or summary_id > summary_ids.shape[0]:
                raise ValueError('rlcombine: summary_id not in the summary set (see fmsummaryxref)')
            if state['n_records'] == summary_ids.shape[0]:
                raise ValueError('rlcombine: more records in an event than summary_ids in the summary set')
            summary_ids[state['n_records']] = summary_id
            expval[state['n_records']] = record_expval
            offsets[state['n_records']] = state['n_pairs']
            state['n_records'] += 1
            state['summary_id'] = summary_id


class SummaryCombineReader(EventReader):
    """Read the summary streams of several inuring priorities and yield each event per input.

    EventReader.read_streams reads one whole event from whichever input is ready, so events
    are yielded in the order they arrive, interleaved across inputs. Each input's stream is
    expected to close every event with an E:S0 sentinel record.

    Each input gets one EventBuffer sized for the largest possible event, so memory is
    n_inputs * max_summary_id * (len_sample + 3) * 8 bytes for the pairs.
    """

    def __init__(self, n_inputs, len_sample, max_summary_id):
        self.logger = logger
        self.state = np.zeros(n_inputs, dtype=read_state_dtype)  # one row per input (file_idx)
        self.buffers = [EventBuffer(max_summary_id, len_sample) for _ in range(n_inputs)]
        self.file_idx = 0

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx=0, **kwargs):
        self.file_idx = file_idx
        buffer = self.buffers[file_idx]
        return read_buffer(byte_mv, cursor, valid_buff, self.state[file_idx],
                           buffer.summary_ids, buffer.expval, buffer.offsets, buffer.sidx, buffer.loss)

    def event_read_log(self, event_id):
        pass

    def read_events(self, streams_in):
        """Yield every event of every input as soon as that input has sent it in full.

        Yields:
            (file_idx, event_id, summary_ids, expval, offsets, sidx, loss): file_idx is the input
            (index in streams_in); record i is summary_ids[i] with expval[i] and the pairs
            sidx[offsets[i]:offsets[i + 1]], loss[offsets[i]:offsets[i + 1]].
            The arrays are views that are only valid until the next iteration.
        """
        for event_id in self.read_streams(streams_in):
            file_idx = self.file_idx
            state = self.state[file_idx]
            if state['event_id'] != 0:  # read_streams also yields an event cut off by the end of the stream
                raise CombineStreamError(f"input {file_idx} ended in the middle of event {event_id}")
            buffer = self.buffers[file_idx]
            n_records, n_pairs = state['n_records'], state['n_pairs']
            yield (file_idx, event_id, buffer.summary_ids[:n_records], buffer.expval[:n_records],
                   buffer.offsets[:n_records + 1], buffer.sidx[:n_pairs], buffer.loss[:n_pairs])
            state['n_records'] = 0
            state['n_pairs'] = 0

        if (self.state['summary_id'] != 0).any() or (self.state['event_id'] != 0).any():
            raise CombineStreamError("an input ended in the middle of an event")
