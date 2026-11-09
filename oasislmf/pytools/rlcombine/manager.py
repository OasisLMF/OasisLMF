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
from contextlib import ExitStack
import logging
import numpy as np
import numba as nb

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY, ITEM_STREAM, MAX_LOSS_IDX, MEAN_IDX, SUMMARY_STREAM_ID, TIV_IDX, init_streams_in, mv_write_sidx_loss, mv_write_summary_header, stream_info_to_bytes, write_mv_to_stream

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, def_to_type_and_size, null_index, summary_stream_index_dtype
from oasislmf.pytools.summary.manager import load_summary_object
from oasislmf.pytools.utils import redirect_logging
from oasislmf.utils.exceptions import OasisException

from oasislmf.pytools.rlcombine.reader import PAIR_SIZE, RECORD_HEADER_SIZE, SummaryCombineReader

logger = logging.getLogger(__name__)

summaryset_id_dtype, summaryset_id_dtype_size = def_to_type_and_size('summaryset_id')
event_id_dtype, event_id_dtype_size = def_to_type_and_size('event_id')
summary_id_dtype, summary_id_dtype_size = def_to_type_and_size('summary_id')
loss_dtype, loss_dtype_size = def_to_type_and_size('loss')
sidx_dtype, sidx_dtype_size = def_to_type_and_size('sidx')
loss_dtype, loss_dtype_size = def_to_type_and_size('loss')

record_dtype = np.dtype([
    ('event_id', event_id_dtype),
    ('summary_id',  summary_id_dtype),
    ('expval',  loss_dtype),
])

sidx_pair_dtype = np.dtype([
    ('sidx', sidx_dtype),
    ('loss', loss_dtype),
])

class EventAccumulator:
    """Combined losses of the event being merged, indexed by summary_id - 1."""

    def __init__(self, max_summary_id, len_sample):
        self.present = np.zeros(max_summary_id, dtype=np.bool_)
        self.expval = np.zeros(max_summary_id, dtype=loss_dtype)
        self.max_loss = np.zeros(max_summary_id, dtype=loss_dtype)
        self.mean = np.zeros(max_summary_id, dtype=loss_dtype)
        self.losses = np.zeros((max_summary_id, len_sample), dtype=loss_dtype)

    def reset(self):
        self.present = False
        self.expval = 0
        self.max_loss = 0
        self.mean = 0
        self.losses = 0

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
        raise OasisException(f"unsupported stream type {stream_source_type}, {stream_agg_type}, expected summary stream")

    summary_set_ids = [np.frombuffer(stream_in.read(summaryset_id_dtype_size), summaryset_id_dtype)[0] for stream_in in streams_in]
    if len(set(summary_set_ids)) > 1:
        raise OasisException(f"input streams have different summary_set_id {summary_set_ids}")

    return streams_in, len_sample, summary_set_ids[0]

def parse_intermediate_outputs(intermediate_outputs_arg):
    intermediate_outputs = {}
    if intermediate_outputs_arg is None:
        return intermediate_outputs

    error_msg = f'intermediate-outputs expected input of form input=path, found {intermediate_outputs_arg}'
    try:
        for arg in intermediate_outputs_arg:
            idx, path = arg.split('=', 1)
            intermediate_outputs[int(idx)] = path
    except ValueError:
        raise Exception(error_msg)

    return intermediate_outputs


def get_max_summary_id__single_static_path(static_path, summary_set_id):
    _, set_id_to_idx, set_idx_to_loss, _, _ = load_summary_object(static_path, 'rl')
    set_idx = set_id_to_idx[summary_set_id]
    if set_idx == null_index:
        raise OasisException(f'summary_set_id {summary_set_id} not in {static_path} rlsummaryxref')
    return set_idx_to_loss[set_idx + 1] - set_idx_to_loss[set_idx]


def get_max_summary_id(static_paths, summary_set_id):
    max_summary_id = -1
    for static_path in static_paths:
        summary_id = get_max_summary_id__single_static_path(static_path, summary_set_id)
        if max_summary_id < summary_id:
            max_summary_id = summary_id
    return max_summary_id


@nb.njit(cache=True)
def accumulate_input_events(summary_ids, expval_event, offsets_event, sidx_event, loss_event,
                            present, expval, max_loss, mean, losses):
    """
    Add one event buffer to the event accumulator.
    """
    previous_summary_id = 0
    for i in range(summary_ids.shape[0]):
        summary_id = summary_ids[i]
        if summary_id < previous_summary_id:
            raise ValueError('rlcombine: summary_ids are not strictly increasing')
        previous_summary_id = summary_id

        row = summary_id - 1
        present[row] = True
        expval[row] += expval_event
        for j in range(offsets_event[i], offsets_event[i+1]):
            if sidx_event[j] > 0:
                losses[row, sidx_event[j]-1] += loss_event[j]
            elif sidx_event[j] == MAX_LOSS_IDX:
                max_loss[row] += loss_event[j]
            elif sidx_event[j] == MEAN_IDX:
                mean[row] += loss_event[j]
            else:
                raise ValueError('rlcombine: unknown special sidx found. Only support MEAN_IDX and MAX_LOSS_IDX')


def combine_events(event_buffers, acc):
    for event_buffer in event_buffers:
        accumulate_input_events(event_buffer['summary_ids'], event_buffer['expval'], event_buffer['offsets'],
                                event_buffer['sidx'], event_buffer['loss'], acc.present, acc.expval,
                                acc.max_loss, acc.mean, acc.losses)

    return (np.flatnonzero(acc.present) + 1).astype(summary_id_dtype)


@nb.njit(cache=True)
def mv_write_combined_event(byte_mv, event_id, len_sample, present_ids, expval, max_loss, mean, losses,
                            output_zeros, last_record, last_sidx, stream_offset, idx_buffer, idx_cursor):
    cursor = 0
    for i in range(last_record, present_ids.shape[0]):
        summary_id = present_ids[i]
        row = summary_id - 1

        if last_sidx == 0:
            if cursor + RECORD_HEADER_SIZE + 3 * PAIR_SIZE > PIPE_CAPACITY:
                return cursor, i, last_sidx, idx_cursor
            idx_buffer[idx_cursor]['summary_id'] = summary_id
            idx_buffer[idx_buffer]['offset'] = stream_offset + cursor
            idx_cursor += 1
            cursor = mv_write_summary_header(byte_mv, cursor, event_id, summary_id, expval[row])
            cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, max_loss[row])
            cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, mean[row])
            last_sidx = 1

        for sidx in range(last_sidx, len_sample + 1):
            loss = losses[row, sidx - 1]
            if not output_zeros and loss == 0:
                continue

            if cursor + 2 * PAIR_SIZE > PIPE_CAPACITY:
                return cursor, i, sidx, idx_cursor

            cursor = mv_write_sidx_loss(byte_mv, cursor, sidx, loss)

        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)
        last_sidx = 0
    return cursor, -1, 0, idx_cursor


def write_combined_event(pipe, idx_pipe, byte_mv, event_id, len_sample, present_ids, acc,
                         output_zeros, stream_offsets, file_idx, idx_buffer):
    '''
    Write combined events into PIPE_CAPACITY chunks and .idx records if idx_pipe.
    '''
    last_record, last_sidx, idx_cursor = 0, 0, 0
    while True:
        cursor, last_record, last_sidx, idx_cursor = mv_write_combined_event(
                byte_mv, event_id, len_sample, present_ids, acc.expval, acc.max_loss,
                acc.mean, acc.losses, output_zeros, last_record, last_sidx, stream_offsets[file_idx],
                idx_buffer, idx_cursor)
        write_mv_to_stream(pipe, byte_mv, cursor)
        stream_offsets[file_idx] += cursor
        if last_record == -1:
            break

    if idx_pipe is not None:
        idx_pipe.write(idx_buffer[:idx_cursor].tobytes())


@nb.njit(cache=True)
def mv_write_passthrough_event(byte_mv, last_record, last_pair, idx_buffer, idx_cursor, idx_file_cursors,
                               event_id, summary_ids, expval, sidx, loss, offsets):
    n_records = summary_ids.shape[0]
    cursor = 0

    for i in range(last_record, n_records):
        if last_pair == -1: # start of new record
            if cursor + RECORD_HEADER_SIZE + PAIR_SIZE > PIPE_CAPACITY:
                return cursor, i, -1, idx_cursor
            idx_buffer[idx_cursor]['summary_id'] = summary_ids[i]
            idx_buffer[idx_cursor]['offset'] = idx_file_cursors + cursor
            idx_cursor += 1
            cursor = mv_write_summary_header(byte_mv, cursor, event_id, summary_ids[i], expval[i])
            last_pair = offsets[i]

        for j in range(last_pair, offsets[i+1]):
            if cursor + PAIR_SIZE * 2 > PIPE_CAPACITY: # pair + terminator
                return cursor, i, j, idx_cursor
            cursor = mv_write_sidx_loss(byte_mv, cursor, sidx[j], loss[j])

        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0) # terminator
        last_pair = -1
    return cursor -1, -1, idx_cursor


def write_passthrough_event(pipe, idx_pipe, byte_mv, event_id, event_buffer,
                            idx_file_cursors, file_idx, idx_buffer):
    last_record, last_pair, idx_cursor = 0, -1, 0
    while True:
        cursor, last_record, last_pair, idx_cursor = mv_write_passthrough_event(
                byte_mv, last_record, last_pair, idx_buffer, idx_cursor, idx_file_cursors[file_idx], event_id,
                event_buffer['summary_ids'], event_buffer['expval'], event_buffer['sidx'], event_buffer['loss'],
                event_buffer['offsets'])

        write_mv_to_stream(pipe, byte_mv, cursor)
        idx_file_cursors[file_idx] += cursor

        if last_record == -1:
            break

    if idx_pipe is not None:
        idx_pipe.write(idx_buffer[:idx_cursor].tobytes())


def run(summary_files_in, output_summary_file, static_paths, low_memory, intermediate_outputs, output_zeros, **kwargs):
    intermediate_outputs = parse_intermediate_outputs(intermediate_outputs)

    if len(summary_files_in) != len(static_paths):
        raise OasisException('each summary stream file needs a corresponding static path')

    n_inputs = len(summary_files_in)

    if any([k > n_inputs - 1 for k in intermediate_outputs.keys()]):
        raise ValueError(f'Index of requested intermediate-outputs is out of bound. Requested values: {intermediate_outputs.keys()}')

    out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')
    partial_out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

    with ExitStack() as stack:
        # gather inputs and read meta
        streams_in, len_sample, summary_set_id = init_summary_streams_in(summary_files_in, stack)
        max_summary_id = get_max_summary_id(static_paths, summary_set_id)

        acc = EventAccumulator(max_summary_id, len_sample)

        # prepare idx files
        idx_buffer = np.empty(max_summary_id, dtype=summary_stream_index_dtype)
        idx_file_cursors = np.zeros(n_inputs + 1, dtype=np.int64)

        # prepare final output
        output_pipe = stack.enter_context(open(output_summary_file, 'wb'))
        idx_file_cursors[-1] += output_pipe.write(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM))
        idx_file_cursors[-1] += output_pipe.write(len_sample.tobytes())
        idx_file_cursors[-1] += output_pipe.write(summary_id_dtype.type(summary_set_id).tobytes())

        # prepare the intermediate outputs
        intermediate_pipes = {i: stack.enter_context(open(intermediate_path, 'wb')) for i, intermediate_path in intermediate_outputs.items()}
        for i, int_pipe in intermediate_pipes.items():
            idx_file_cursors[i] += int_pipe.write(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM))
            idx_file_cursors[i] += int_pipe.write(len_sample.tobytes())
            idx_file_cursors[i] += int_pipe.write(summary_id_dtype.type(summary_set_id).tobytes())


        # prepare index pipes
        output_index_pipe = None
        intermediate_idx_pipes = {}
        if low_memory:
            output_index_name = output_summary_file.rsplit('.', 1)[0] + '.idx'
            output_index_pipe = stack.enter_context(open(output_index_name, 'wb'))
            intermediate_idx_pipes = {i: stack.enter_context(open(intermediate_path.rsplit('.', 1)[0] + '.idx', 'wb')) for i, intermediate_path in intermediate_outputs.items()}

        summary_stream_reader = SummaryCombineReader(n_inputs, max_summary_id, len_sample)
        input_queues = [[] for _ in range(n_inputs)]

        for file_idx, event_id, event_buffer in summary_stream_reader.read_events(streams_in):
            # pass through requested IP
            if file_idx in intermediate_pipes:
                write_passthrough_event(intermediate_pipes[file_idx],
                                        intermediate_idx_pipes.get(file_idx),
                                        out_byte_mv, event_id, event_buffer, idx_file_cursors, file_idx, idx_buffer)

            input_queues[file_idx].append((event_id, event_buffer))
            if any([len(input_queue) == 0 for input_queue in input_queues]):
                continue

            # every input sent an event, now combine
            curr_events = [input_queue.pop(0) for input_queue in input_queues]
            event_id = curr_events[0][0]

            if any([curr_event[0] != event_id for curr_event in curr_events]):
                raise OasisException('event sequence mismatch in the streams')

            present_ids = combine_events(curr_events, acc)
            write_combined_event(output_pipe, output_index_pipe, out_byte_mv, event_id, len_sample,
                                 present_ids, acc, output_zeros, idx_file_cursors, file_idx, idx_buffer)
            acc.reset()

        if any([len(input_queue) != 0 for input_queue in input_queues]):
               raise OasisException('Summary streams only partially combined.')


@redirect_logging(exec_name='rlcombine')
def main(summary_files_in, output_summary_file, static_paths, low_memory,
         intermediate_outputs, output_zeros, **kwargs):
    run(summary_files_in, output_summary_file, static_paths, low_memory,
        intermediate_outputs, output_zeros, **kwargs)
