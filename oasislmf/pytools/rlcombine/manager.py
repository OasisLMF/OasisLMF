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

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY, ITEM_STREAM, MAX_LOSS_IDX, MEAN_IDX, SUMMARY_STREAM_ID, TIV_IDX, init_streams_in, mv_write_sidx_loss, mv_write_summary_header, write_mv_to_stream

from oasislmf.pytools.common.data import DEFAULT_BUFFER_SIZE, def_to_type_and_size, null_index
from oasislmf.pytools.summary.manager import load_summary_object
from oasislmf.utils.exceptions import OasisException

from oasislmf.pytools.rlcombine.reader import SummaryCombineReader

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

def merge_events(input_queues, n_inputs, len_sample, out_byte_mv):
    cursor = 0

    event_id = None
    curr_events = [input_queues[k].pop(0) for k in range(n_inputs)]


    event_id = curr_events[0][0]
    if any([curr_events[k][0] != event_id for k in range(n_inputs)]):
        raise OasisException('event sequence mismatch in the streams')

    curr_event_buffers = [ce[1] for ce in curr_events]
    curr_record_offset = [0 for _ in range(n_inputs)]

    previous_summary_id = [0 for _ in range(n_inputs)]
    remaining_inputs = list(range(n_inputs))
    finished = [False for _ in range(n_inputs)]
    while True:
        # Check if all inputs are finished for current event
        for i, k in enumerate(remaining_inputs):
            if curr_record_offset[k] == curr_event_buffers[k]['summary_ids'].shape[0]:
                finished[i] = True

        if all(finished):
            return cursor

        curr_summary_id = min([curr_event_buffers[k]['summary_ids'][curr_record_offset[k]] for k in remaining_inputs])

        # create a new record + loss array
        new_record = np.empty(1, dtype=record_dtype)[0]
        new_record['event_id'] = event_id
        new_record['summary_id'] = curr_summary_id
        new_record['expval'] = 0.

        regular_loss_pairs = np.zeros(len_sample, dtype=sidx_pair_dtype)
        regular_loss_pairs['sidx'] = np.arange(len_sample, dtype=sidx_dtype) + 1
        special_loss_pairs = np.empty(2, dtype=sidx_pair_dtype)
        special_loss_pairs[0]['sidx'] = MAX_LOSS_IDX
        special_loss_pairs[1]['sidx'] = MEAN_IDX

        regular_loss_mask = np.zeros(len_sample, dtype=bool)

        print(f'New record... event_id: {event_id}, summary_id: {curr_summary_id}')
        print('---')
        print()

        for k in remaining_inputs:
            k_curr_record_offset = curr_record_offset[k]
            k_curr_buffer = curr_event_buffers[k]
            k_curr_summary_id = k_curr_buffer['summary_ids'][k_curr_record_offset]
            if k_curr_summary_id != curr_summary_id:
                continue
            elif k_curr_summary_id < previous_summary_id[k]:
                raise OasisException(f'summaryids are not strictly increasing for file idx {k}')

            # sum expval
            new_record['expval'] += k_curr_buffer['expval'][k_curr_record_offset]

            start_offset = k_curr_buffer['offsets'][k_curr_record_offset]
            end_offset = k_curr_buffer['offsets'][k_curr_record_offset+1]

            print(f'Combining from input buffer {k}...')
            print(f"expval: {k_curr_buffer['expval'][k_curr_record_offset]}")

            print(f"sidx: {k_curr_buffer['sidx'][start_offset: end_offset]}")
            print(f"loss: {k_curr_buffer['loss'][start_offset: end_offset]}")
            for i in range(start_offset, end_offset):
                _curr_sidx = k_curr_buffer['sidx'][i]
                _curr_loss = k_curr_buffer['loss'][i]

                if _curr_sidx < 0:
                    if _curr_sidx == MAX_LOSS_IDX:
                        special_loss_pairs[0]['loss'] += _curr_loss
                    elif _curr_sidx == MEAN_IDX:
                        special_loss_pairs[1]['loss'] += _curr_loss
                    else:
                        raise ValueError(f'unexpected special sidx{_curr_sidx} found')
                    continue

                regular_loss_pairs[_curr_sidx - 1]['loss'] += _curr_loss
                regular_loss_mask[_curr_sidx - 1] = True

            curr_record_offset[k] += 1
            previous_summary_id[k] = curr_summary_id

        regular_loss_pairs = regular_loss_pairs[regular_loss_mask]

        print('---')
        print()

        print('Combined record: ')
        print(f"expval: {new_record['expval']}")
        print(f"regular loss pairs: {regular_loss_pairs}")
        print(f"special loss pairs: {special_loss_pairs}")

        print('---')
        print()

        # need to handle output properly
        cursor = add_record_to_byte_mv(out_byte_mv, cursor, new_record, regular_loss_pairs, special_loss_pairs)

        print('Moving onto the next record...')

        print('---')
        print('\n\n')


def add_record_to_byte_mv(byte_mv, cursor, record, loss_pairs, special_loss_pairs, output_zeros=False):
    print('Writing record...')
    print(f'event_id: {record["event_id"]}, summary_id: {record["summary_id"]}, expval: {record["expval"]}')
    cursor = mv_write_summary_header(byte_mv, cursor, record['event_id'], record['summary_id'], record['expval'])
    cursor = mv_write_sidx_loss(byte_mv, cursor, MAX_LOSS_IDX, special_loss_pairs[0]['loss'])
    cursor = mv_write_sidx_loss(byte_mv, cursor, MEAN_IDX, special_loss_pairs[1]['loss'])

    for _curr_loss_pair in loss_pairs:
        sidx, loss = _curr_loss_pair['sidx'], _curr_loss_pair['loss']
        if not output_zeros and loss == 0:
            continue
        cursor = mv_write_sidx_loss(byte_mv, cursor, sidx, loss)

    cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)
    return cursor


def run(summary_files_in, output_summary_file, static_paths, low_memory,  **kwargs):

    if len(summary_files_in) != len(static_paths):
        raise OasisException('each summary stream file needs a corresponding static path')

    n_inputs = len(summary_files_in)
    inuring_priority_info_dict = { i: {'summary_fp': summary_files_in[i],
                                       'static_path': static_paths[i],
                                       'stream_in': None,
                                       'max_summary_id': None} for i in range(n_inputs)}

    out_byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')

    with ExitStack() as stack:
        output_pipe = stack.enter_context(open(output_summary_file, 'wb'))
        if low_memory:
            output_index_name = output_summary_file.rsplit('.', 1)[0] + '.idx'
            output_index_pipe = stack.enter_context(open(output_index_name, 'wb'))
        streams_in, len_sample, summary_set_id = init_summary_streams_in(summary_files_in, stack)
        max_summary_id = get_max_summary_id(static_paths, summary_set_id)

        summary_stream_reader = SummaryCombineReader(n_inputs, max_summary_id, len_sample)
        input_queues = [[] for _ in range(n_inputs)]

        file_idx_events = {i: [] for i in range(n_inputs)}
        n_event_ids_written = 0
        event_ids_seen = set()
        for file_idx, event_id, event_buffer in summary_stream_reader.read_events(streams_in):
            event_ids_seen.add(event_id)
            print(f'read file_idx: {file_idx}')
            print(f'event_id read: {event_id}')
            file_idx_events[file_idx].append(event_id)
            input_queues[file_idx].append((event_id, event_buffer))
            print(f'length of input queue {file_idx}: {len(input_queues[file_idx])}')

            if any([len(input_queues[k]) == 0 for k in range(n_inputs)]):
                print('Waiting to fill all queues...')
                continue

            cursor = merge_events(input_queues, n_inputs,len_sample, out_byte_mv)
            write_mv_to_stream(output_pipe, out_byte_mv, cursor)
            n_event_ids_written += 1

        print("Finished combining streams...")
        print(f"No. event ids written: {n_event_ids_written}")
        print(f"No. event ids seen: {len(event_ids_seen)}")
        print(f"Missing event_ids: {event_ids_seen.discard(n_event_ids_written)}")
