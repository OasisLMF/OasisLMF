"""rlcombine

read multiple summary loss streams from multiple rl inuring priorities and combine

the input is the summary loss stream from each individual IP for a given summaryset (and process)

i.e. fifo/rl_IP1_S1_combine_P1, fifo/rl_IP2_S1_combine_P2, ... fifo/rl_IPN_S1_combine_P2

The summary streams from each individual IP are structured something like this:

header │ E1:S2 │ E1:S5 │ E1:S9 │ E7:S1 │ E7:S5 │ E3:S2 │ …
         └──── event 1 ────┘     └─ event 7 ─┘  └ event 3 …

STREAM HEADER (12 B, once): [agg type=1 (3 B) | source type=3 (1 B)] int32 len_sample=N | int32 summary_set_id

and each summary record (e.g. E1:S2) consists of:
    single header (12 bytes): int32 event_id, int32 summary_id, float32 expval
    multiple records (8 bytes each): (int32 sidx, float32 loss)
        sidx -5, -4, -1 are the max, affected risk and mean analytical loss respectively
        sidx of a positive integar is a single sample
        sidx 0 = terminator (end of summary record)

To combine summary streams from multiple IP we need to know when a particular
inuring priority has sent the final event_id record, including when an event_id
has no loss and no records. Events with no loss are never sent. Hence we need a
sentinel record to notify when an `event_id` has finished. This should only
happen if --combine-out is passed to summarypy and this type of stream will
only be passed to this module (not the output components) and will be stripped
from this stream in this module before passing on. Propose using a `summary_id`
of 0 for the sentinel record for example `E10:S0` and the sidx 0 terminator
afterwards to preserve consistency with the other parsers. Then you can know
when a record is done for a particular `event_id` without reading the next
record in the stream.

We can use the fact that that the event_ids should be aligned and the
summary_ids should be sorted as they get passed to rlcombine to do a merge with
these streams. #TODO - check this

The pseudo code for this module looks something like:

open inputs; read the 12-byte header of each
    all must have source type == SUMMARY (3), agg type == ITEM (1),
    and equal len_sample and summary_set_id               → else error
write output header (same three values); out_offset = 12

for each input k: head[k] = next_record(k)                  # non-blocking fill, see below
if every head is EOF: done                                   # empty run
if any head is EOF: error

loop:
    E = head[0].event_id
    if any head[k].event_id != E: error (event sequence mismatch)

    S = min(head[k].summary_id, with S0 = ∞)

    if S == ∞:                                               # every input has closed event E
        for each k: head[k] = next_record(k)
        if every head is EOF: done
        if any head is EOF: error
        continue

    acc = new summary record (E, S)
    for each k with head[k].summary_id == S:
        combine head[k] into acc
        prev[k] = S
        head[k] = next_record(k)
        if head[k] is EOF: error (event E not closed by S0)
        if head[k] is not S0:
            if head[k].event_id != E: error (event E not closed by S0)
            if head[k].summary_id <= prev[k]: error (summary_id not strictly increasing)

    if -z or acc.tiv != 0 or acc.mean != 0:
        if -m: write (S, out_offset) to .idx
        write acc to output buffer (flushing and splitting across 64 KB chunks)
        out_offset += record size

next_record(k):
    while stream k has no complete record (terminator not yet buffered) and not EOF:
        select on all open inputs; for each ready input j, read what is available into buf[j]
    parse one record from buf[k] (or return EOF)

Method of combining records depends on the type of the field:

    sidx 1...N         | regular sample                  | sum
    sidx -1            | mean                            | sum (no. samples are the same)
    sidx -2            | std dev                         | not in summary stream → error if seen
    header expval (-3) | TIV (passed through fm terms)   | sum
    sidx -4            | chance of loss / affected risks | not written for -t ri → error if seen
    sidx -5            | max loss                        | sum


# TODO: check that tiv and max loss are actually calculated per term in the RI case and not just passed through.
    - what to do with these fields will depend on this output.
"""
import logging

import numba as nb
import numpy as np

from oasislmf.pytools.common.data import def_to_type_and_size
from oasislmf.pytools.common.event_stream import (EventReader, init_streams_in, mv_read,
                                                  ITEM_STREAM, MAX_LOSS_IDX, MEAN_IDX, SUMMARY_STREAM_ID)

logger = logging.getLogger(__name__)

event_id_dtype, event_id_dtype_size = def_to_type_and_size('event_id')
summary_id_dtype, summary_id_dtype_size = def_to_type_and_size('summary_id')
summaryset_id_dtype, summaryset_id_dtype_size = def_to_type_and_size('summaryset_id')
sidx_dtype, sidx_dtype_size = def_to_type_and_size('sidx')
loss_dtype, loss_dtype_size = def_to_type_and_size('loss')

RECORD_HEADER_SIZE = event_id_dtype_size + summary_id_dtype_size + loss_dtype_size
PAIR_SIZE = sidx_dtype_size + loss_dtype_size
SENTINEL_SUMMARY_ID = 0

# accumulator columns: 0 = max loss (sidx -5), 1 = mean (sidx -1), 1 + s = sample s
COL_MAX_LOSS = 0
COL_MEAN = 1
N_SPECIAL_COLS = 2

read_state_dtype = np.dtype([
    ('event_id', event_id_dtype),      # event currently being read on this input (0 = between events)
    ('summary_id', summary_id_dtype),  # record whose pairs are being read (0 = expect a record header)
    ('slot', np.int64),                # accumulator slot holding event_id
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


@nb.njit(cache=True)
def get_slot(event_id, slot_event, slot_seq, next_seq):
    """Return the accumulator slot of event_id, allocating a free one on first sight."""
    free = -1
    for i in range(slot_event.shape[0]):
        if slot_event[i] == event_id:
            return i
        if free == -1 and slot_event[i] == -1:
            free = i
    if free == -1:
        raise ValueError('rlcombine: no free event slot, an input is too far ahead of the others (increase n_slots)')
    slot_event[free] = event_id
    slot_seq[free] = next_seq[0]
    next_seq[0] += 1
    return free


@nb.njit(cache=True)
def read_buffer(byte_mv, cursor, valid_buff, state,
                slot_event, slot_seq, next_seq, slot_count, acc, expval, seen, present, present_len):
    """Read the valid part of byte_mv for one input and add its records into the event slots.

    Every value is summed: samples, mean (-1), max loss (-5) and the header expval.

    Returns:
        cursor, event_id, summary_id (non zero if a record is only partly read), yield_event
        yield_event is 1 right after the E:S0 sentinel, so an event is known to be complete
        on this input without reading ahead into the next event.
    """
    while True:
        if state['summary_id']:  # inside a record: read (sidx, loss) pairs
            if valid_buff - cursor < PAIR_SIZE:
                return cursor, state['event_id'], state['summary_id'], 0
            sidx, cursor = mv_read(byte_mv, cursor, sidx_dtype, sidx_dtype_size)
            loss, cursor = mv_read(byte_mv, cursor, loss_dtype, loss_dtype_size)
            if sidx == 0:
                state['summary_id'] = 0
                continue
            if sidx > 0:
                col = N_SPECIAL_COLS - 1 + sidx
            elif sidx == MEAN_IDX:
                col = COL_MEAN
            elif sidx == MAX_LOSS_IDX:
                col = COL_MAX_LOSS
            else:
                raise ValueError('rlcombine: unexpected special sidx in summary stream (only -5 and -1 are supported)')
            acc[state['slot'], state['summary_id'], col] += loss

        else:  # expect a record header (a sentinel needs its (0, 0) pair too)
            if valid_buff - cursor < RECORD_HEADER_SIZE + PAIR_SIZE:
                return cursor, state['event_id'], 0, 0
            record_event_id, cursor = mv_read(byte_mv, cursor, event_id_dtype, event_id_dtype_size)
            summary_id, cursor = mv_read(byte_mv, cursor, summary_id_dtype, summary_id_dtype_size)
            record_expval, cursor = mv_read(byte_mv, cursor, loss_dtype, loss_dtype_size)

            if state['event_id'] == 0:
                state['event_id'] = record_event_id
                state['slot'] = get_slot(record_event_id, slot_event, slot_seq, next_seq)
            elif record_event_id != state['event_id']:
                raise ValueError('rlcombine: event not closed by an S0 sentinel record')
            slot = state['slot']

            if summary_id == SENTINEL_SUMMARY_ID:
                sidx, cursor = mv_read(byte_mv, cursor, sidx_dtype, sidx_dtype_size)
                cursor += loss_dtype_size
                if sidx != 0:
                    raise ValueError('rlcombine: S0 sentinel record must only hold the (0, 0) terminator')
                slot_count[slot] += 1
                event_id = state['event_id']
                state['event_id'] = 0
                return cursor, event_id, 0, 1

            if summary_id < 0 or summary_id >= seen.shape[1]:
                raise ValueError('rlcombine: summary_id out of range (check max_summary_id)')
            if not seen[slot, summary_id]:
                seen[slot, summary_id] = True
                present[slot, present_len[slot]] = summary_id
                present_len[slot] += 1
            expval[slot, summary_id] += record_expval
            state['summary_id'] = summary_id


class SummaryCombineReader(EventReader):
    """Read the summary streams of several inuring priorities and sum them per (event_id, summary_id).

    EventReader.read_streams is push based: it reads one whole event from whichever input is
    ready and yields its event_id. The same event therefore arrives from each input at a
    different time, so each event in flight gets an accumulator "slot". An event is complete
    once every input has sent its E:S0 sentinel; read_combined_events then yields it, sorted
    by summary_id, and frees the slot.

    Reading one input until its event is complete cannot stall the other inputs' producers:
    summarypy only starts writing an event once it has read all of it, so once the first byte
    of an event arrives the rest of it and its S0 follow without waiting on anything upstream.

    Memory: n_slots * (max_summary_id + 1) * (len_sample + 2) * 8 bytes.
    """

    def __init__(self, n_inputs, len_sample, max_summary_id, n_slots=64):
        self.logger = logger
        self.n_inputs = n_inputs
        self.state = np.zeros(n_inputs, dtype=read_state_dtype)  # one row per input (file_idx)

        self.slot_event = np.full(n_slots, -1, dtype=np.int64)
        self.slot_seq = np.zeros(n_slots, dtype=np.int64)  # order in which events were first seen
        self.next_seq = np.zeros(1, dtype=np.int64)
        self.slot_count = np.zeros(n_slots, dtype=np.int64)  # inputs that have closed the event

        self.acc = np.zeros((n_slots, max_summary_id + 1, len_sample + N_SPECIAL_COLS), dtype=np.float64)
        self.expval = np.zeros((n_slots, max_summary_id + 1), dtype=np.float64)
        self.seen = np.zeros((n_slots, max_summary_id + 1), dtype=np.bool_)
        self.present = np.zeros((n_slots, max_summary_id + 1), dtype=summary_id_dtype)
        self.present_len = np.zeros(n_slots, dtype=np.int64)

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, file_idx=0, **kwargs):
        return read_buffer(
            byte_mv, cursor, valid_buff, self.state[file_idx],
            self.slot_event, self.slot_seq, self.next_seq, self.slot_count,
            self.acc, self.expval, self.seen, self.present, self.present_len
        )

    def event_read_log(self, event_id):
        pass

    def read_combined_events(self, streams_in):
        """Read all inputs and yield each combined event, in input event order.

        Yields:
            (event_id, summary_ids, expval, losses) with summary_ids sorted ascending,
            expval[i] the summed expval of summary_ids[i] and losses[i] its summed
            columns (max loss, mean, sample 1 .. len_sample). The arrays are only valid
            until the next iteration.
        """
        for event_id in self.read_streams(streams_in):
            slot = int(np.flatnonzero(self.slot_event == event_id)[0])
            if self.slot_count[slot] < self.n_inputs:
                continue  # other inputs have not closed this event yet

            older = (self.slot_event != -1) & (self.slot_seq < self.slot_seq[slot])
            if older.any():
                raise CombineStreamError(f"event {event_id} complete while earlier events {self.slot_event[older].tolist()} "
                                         "are not: the inputs do not carry the same event sequence")

            summary_ids = np.sort(self.present[slot, :self.present_len[slot]])
            yield event_id, summary_ids, self.expval[slot, summary_ids], self.acc[slot, summary_ids]
            self._release(slot, summary_ids)

        # read_streams also yields incomplete events at end of stream, so check what is left
        if (self.state['summary_id'] != 0).any() or (self.state['event_id'] != 0).any():
            raise CombineStreamError("an input ended in the middle of an event")
        if (self.slot_event != -1).any():
            raise CombineStreamError(f"events never completed on every input: {self.slot_event[self.slot_event != -1].tolist()}")

    def _release(self, slot, summary_ids):
        self.acc[slot, summary_ids] = 0
        self.expval[slot, summary_ids] = 0
        self.seen[slot, summary_ids] = False
        self.present_len[slot] = 0
        self.slot_count[slot] = 0
        self.slot_event[slot] = -1
