"""Entry point to the run method of rlcombine
combine the reinsurer's loss (rl) summary streams of every inuring priority (IP) into one stream

Each input is the ``summarypy -t rl -s`` stream of one IP for the same summary set and process,
e.g. ``rl_IP1_S1_rlcombine_P1 ... rl_IPN_S1_rlcombine_P1``. The ``-s`` flag closes every event with a
sentinel record ``E:S0`` (summary_id 0), so an event is known to be complete even when an IP has no
loss records for it. The sentinels are not written to any output.

For each event:

1. ``SummaryCombineReader`` reads the inputs as data arrives and yields an input's event once its
   sentinel is read. Every IP sends the events in the same order.
2. If the input is requested with ``-I``, the event is written unchanged to its pass-through stream
   (``rl_IP{i}_S{j}``).
3. Once every input has sent the event, records with the same summary_id are summed across inputs:
   the samples, the mean (sidx -1), the max loss (sidx -5) and the expval (TIV) in the header. The
   combined event is written to the output stream (``rl_S{j}``) in summary_id order.

The writers stop before their PIPE_CAPACITY buffer is full and resume after it is flushed, so events
and single records can be larger than PIPE_CAPACITY. With ``-m``, a ``.idx`` file of
(summary_id, offset) records is written next to each output.
"""
from contextlib import ExitStack
import logging
import numpy as np
import numba as nb

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY, ITEM_STREAM, MAX_LOSS_IDX, MEAN_IDX, SUMMARY_STREAM_ID, init_streams_in, mv_write_sidx_loss, mv_write_summary_header, stream_info_to_bytes, write_mv_to_stream

from oasislmf.pytools.common.data import def_to_type_and_size, null_index, summary_stream_index_dtype
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
    ('summary_id', summary_id_dtype),
    ('expval', loss_dtype),
])

sidx_pair_dtype = np.dtype([
    ('sidx', sidx_dtype),
    ('loss', loss_dtype),
])


class EventAccumulator:
    """Combined losses of the event being merged, one row per summary_id (row = summary_id - 1).

    Attributes:
        present (ndarray[bool]): True for the summary_ids that have a record in the event
        expval (ndarray[loss_dtype]): summed expval (TIV) of each summary_id
        max_loss (ndarray[loss_dtype]): summed max loss (sidx -5) of each summary_id
        mean (ndarray[loss_dtype]): summed mean loss (sidx -1) of each summary_id
        losses (ndarray[loss_dtype]): summed sample losses, shape (max_summary_id, len_sample)
    """

    def __init__(self, max_summary_id, len_sample):
        """Allocate the combined loss arrays, set to zero.

        Args:
            max_summary_id (int): largest summary_id in the summary set
            len_sample (int): number of samples
        """
        self.present = np.zeros(max_summary_id, dtype=np.bool_)
        self.expval = np.zeros(max_summary_id, dtype=loss_dtype)
        self.max_loss = np.zeros(max_summary_id, dtype=loss_dtype)
        self.mean = np.zeros(max_summary_id, dtype=loss_dtype)
        self.losses = np.zeros((max_summary_id, len_sample), dtype=loss_dtype)

    def reset(self, present_ids):
        """Clear the rows used by the last event, in place.

        Args:
            present_ids (ndarray[summary_id_dtype]): summary_ids of the last event, as returned by combine_events
        """
        rows = present_ids - 1
        self.present[rows] = False
        self.expval[rows] = 0
        self.max_loss[rows] = 0
        self.mean[rows] = 0
        self.losses[rows] = 0


def init_summary_streams_in(files_in, stack):
    """Open the input summary streams and check that their headers agree.

    init_streams_in only reads the stream type and len_sample, the summary_set_id that follows them
    in a summary stream header is read here.

    Args:
        files_in (List[str]): paths of the input summary streams, one per inuring priority
        stack (ExitStack): context stack the opened streams are added to

    Returns:
        streams_in (List[BinaryIO]): opened input streams, positioned after the header
        len_sample (int): number of samples
        summary_set_id (summaryset_id_dtype): summary set of the streams

    Raises:
        OasisException: if a stream is not a summary stream or the streams have different summary_set_id
    """
    streams_in, (stream_source_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
    if stream_source_type != SUMMARY_STREAM_ID or stream_agg_type != ITEM_STREAM:
        raise OasisException(f"unsupported stream type {stream_source_type}, {stream_agg_type}, expected summary stream")

    summary_set_ids = [np.frombuffer(stream_in.read(summaryset_id_dtype_size), summaryset_id_dtype)[0] for stream_in in streams_in]
    if len(set(summary_set_ids)) > 1:
        raise OasisException(f"input streams have different summary_set_id {summary_set_ids}")

    return streams_in, len_sample, summary_set_ids[0]


def parse_intermediate_outputs(intermediate_outputs_arg):
    """Parse the -I/--intermediate-outputs arguments.

    Args:
        intermediate_outputs_arg (List[str] | None): arguments of the form ``input_index=path``, where
            input_index is the 0-based position of the input in -i

    Returns:
        intermediate_outputs (Dict[int, str]): pass-through output path for each input index

    Raises:
        Exception: if an argument is not of the form ``input_index=path``
    """
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
    """Get the largest summary_id of a summary set from the rlsummaryxref in static_path.

    Args:
        static_path (str): static folder of one inuring priority (e.g. input/RI_1)
        summary_set_id (summaryset_id_dtype): summary set to look up

    Returns:
        max_summary_id (int): largest summary_id of the summary set

    Raises:
        OasisException: if the summary set is not in the rlsummaryxref
    """
    _, set_id_to_idx, set_idx_to_loss, _, _ = load_summary_object(static_path, 'rl')
    set_idx = set_id_to_idx[summary_set_id]
    if set_idx == null_index:
        raise OasisException(f'summary_set_id {summary_set_id} not in {static_path} rlsummaryxref')
    return set_idx_to_loss[set_idx + 1] - set_idx_to_loss[set_idx]


def get_max_summary_id(static_paths, summary_set_id):
    """Get the largest summary_id of a summary set across all inuring priorities.

    Args:
        static_paths (List[str]): static folder of each inuring priority
        summary_set_id (summaryset_id_dtype): summary set to look up

    Returns:
        max_summary_id (int): largest summary_id of the summary set in any static folder
    """
    max_summary_id = -1
    for static_path in static_paths:
        summary_id = get_max_summary_id__single_static_path(static_path, summary_set_id)
        if max_summary_id < summary_id:
            max_summary_id = summary_id
    return max_summary_id


@nb.njit(cache=True)
def accumulate_input_events(summary_ids, expval_event, offsets_event, sidx_event, loss_event,
                            present, expval, max_loss, mean, losses):
    """Add the records of one input's event to the event accumulator arrays.

    Record i of the event is summary_ids[i] with expval_event[i] and the (sidx, loss) pairs
    sidx_event[offsets_event[i]:offsets_event[i + 1]], loss_event[offsets_event[i]:offsets_event[i + 1]].

    Args:
        summary_ids (ndarray[summary_id_dtype]): summary_id of each record, strictly increasing
        expval_event (ndarray[loss_dtype]): expval (TIV) of each record
        offsets_event (ndarray[int64]): start of each record's pairs in sidx_event / loss_event
        sidx_event (ndarray[sidx_dtype]): sidx of each pair
        loss_event (ndarray[loss_dtype]): loss of each pair
        present (ndarray[bool]): EventAccumulator.present, updated in place
        expval (ndarray[loss_dtype]): EventAccumulator.expval, updated in place
        max_loss (ndarray[loss_dtype]): EventAccumulator.max_loss, updated in place
        mean (ndarray[loss_dtype]): EventAccumulator.mean, updated in place
        losses (ndarray[loss_dtype]): EventAccumulator.losses, updated in place

    Raises:
        ValueError: if the summary_ids are not strictly increasing or a special sidx other than
            MAX_LOSS_IDX and MEAN_IDX is found
    """
    previous_summary_id = 0
    for i in range(summary_ids.shape[0]):
        summary_id = summary_ids[i]
        if summary_id <= previous_summary_id:
            raise ValueError('rlcombine: summary_ids are not strictly increasing')
        previous_summary_id = summary_id

        row = summary_id - 1
        present[row] = True
        expval[row] += expval_event[i]
        for j in range(offsets_event[i], offsets_event[i + 1]):
            if sidx_event[j] > 0:
                losses[row, sidx_event[j] - 1] += loss_event[j]
            elif sidx_event[j] == MAX_LOSS_IDX:
                max_loss[row] += loss_event[j]
            elif sidx_event[j] == MEAN_IDX:
                mean[row] += loss_event[j]
            else:
                raise ValueError('rlcombine: unknown special sidx found. Only support MEAN_IDX and MAX_LOSS_IDX')


def combine_events(event_buffers, acc):
    """Sum the same event from every input into acc.

    Args:
        event_buffers (List[Dict[str, ndarray]]): the event from each input, as yielded by
            SummaryCombineReader.read_events
        acc (EventAccumulator): accumulator, empty before the call

    Returns:
        present_ids (ndarray[summary_id_dtype]): summary_ids with a record in any input, sorted
    """
    for event_buffer in event_buffers:
        accumulate_input_events(event_buffer['summary_ids'], event_buffer['expval'], event_buffer['offsets'],
                                event_buffer['sidx'], event_buffer['loss'], acc.present, acc.expval,
                                acc.max_loss, acc.mean, acc.losses)

    return (np.flatnonzero(acc.present) + 1).astype(summary_id_dtype)


@nb.njit(cache=True)
def mv_write_combined_event(byte_mv, event_id, len_sample, present_ids, expval, max_loss, mean, losses,
                            output_zeros, last_record, last_sidx, stream_offset, idx_buffer, idx_cursor):
    """Write the combined event to byte_mv, from record last_record and sample last_sidx.

    Stops before byte_mv is full: a record header is only started if it fits with its two special
    sidx and the terminator, and a sample only if the terminator still fits after it. Call again with
    the returned last_record and last_sidx after flushing byte_mv.

    Args:
        byte_mv (ndarray): numpy byte view of the PIPE_CAPACITY output buffer
        event_id (event_id_dtype): event id
        len_sample (int): number of samples
        present_ids (ndarray[summary_id_dtype]): summary_ids to write, as returned by combine_events
        expval (ndarray[loss_dtype]): EventAccumulator.expval
        max_loss (ndarray[loss_dtype]): EventAccumulator.max_loss
        mean (ndarray[loss_dtype]): EventAccumulator.mean
        losses (ndarray[loss_dtype]): EventAccumulator.losses
        output_zeros (bool): if False, samples with a zero loss are skipped
        last_record (int): index in present_ids to resume from (0 for a new event)
        last_sidx (int): sample to resume from, 0 if the record header has not been written yet
        stream_offset (int): bytes already written to the output stream, used for the .idx offsets
        idx_buffer (ndarray[summary_stream_index_dtype]): .idx records of the event, filled in place
        idx_cursor (int): next free slot in idx_buffer

    Returns:
        cursor (int): bytes written to byte_mv
        last_record (int): record to resume from, -1 when the event is complete
        last_sidx (int): sample to resume from
        idx_cursor (int): next free slot in idx_buffer
    """
    cursor = 0
    for i in range(last_record, present_ids.shape[0]):
        summary_id = present_ids[i]
        row = summary_id - 1

        if last_sidx == 0:
            if cursor + RECORD_HEADER_SIZE + 3 * PAIR_SIZE > PIPE_CAPACITY:
                return cursor, i, last_sidx, idx_cursor
            idx_buffer[idx_cursor]['summary_id'] = summary_id
            idx_buffer[idx_cursor]['offset'] = stream_offset + cursor
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
    """Write the combined event to pipe in PIPE_CAPACITY chunks, and its .idx records to idx_pipe.

    Args:
        pipe (BinaryIO): combined output stream
        idx_pipe (BinaryIO | None): .idx stream of the combined output, None without -m
        byte_mv (ndarray): numpy byte view of the PIPE_CAPACITY output buffer
        event_id (event_id_dtype): event id
        len_sample (int): number of samples
        present_ids (ndarray[summary_id_dtype]): summary_ids to write, as returned by combine_events
        acc (EventAccumulator): combined losses of the event
        output_zeros (bool): if False, samples with a zero loss are skipped
        stream_offsets (ndarray[int64]): bytes written so far to each output, updated in place
        file_idx (int): index of this output in stream_offsets (-1 for the combined output)
        idx_buffer (ndarray[summary_stream_index_dtype]): buffer for the .idx records of the event
    """
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
    """Write one input's event unchanged to byte_mv, from record last_record and pair last_pair.

    Stops before byte_mv is full: a record header is only started if it fits with the terminator,
    and a pair only if the terminator still fits after it. Call again with the returned last_record
    and last_pair after flushing byte_mv.

    Args:
        byte_mv (ndarray): numpy byte view of the PIPE_CAPACITY output buffer
        last_record (int): record to resume from (0 for a new event)
        last_pair (int): pair to resume from, -1 if the record header has not been written yet
        idx_buffer (ndarray[summary_stream_index_dtype]): .idx records of the event, filled in place
        idx_cursor (int): next free slot in idx_buffer
        idx_file_cursors (int): bytes already written to the pass-through stream, used for the .idx offsets
        event_id (event_id_dtype): event id
        summary_ids (ndarray[summary_id_dtype]): summary_id of each record
        expval (ndarray[loss_dtype]): expval (TIV) of each record
        sidx (ndarray[sidx_dtype]): sidx of each pair
        loss (ndarray[loss_dtype]): loss of each pair
        offsets (ndarray[int64]): start of each record's pairs in sidx / loss

    Returns:
        cursor (int): bytes written to byte_mv
        last_record (int): record to resume from, -1 when the event is complete
        last_pair (int): pair to resume from, -1 if the next record starts with its header
        idx_cursor (int): next free slot in idx_buffer
    """
    n_records = summary_ids.shape[0]
    cursor = 0

    for i in range(last_record, n_records):
        if last_pair == -1:  # start of new record
            if cursor + RECORD_HEADER_SIZE + PAIR_SIZE > PIPE_CAPACITY:
                return cursor, i, -1, idx_cursor
            idx_buffer[idx_cursor]['summary_id'] = summary_ids[i]
            idx_buffer[idx_cursor]['offset'] = idx_file_cursors + cursor
            idx_cursor += 1
            cursor = mv_write_summary_header(byte_mv, cursor, event_id, summary_ids[i], expval[i])
            last_pair = offsets[i]

        for j in range(last_pair, offsets[i + 1]):
            if cursor + PAIR_SIZE * 2 > PIPE_CAPACITY:  # pair + terminator
                return cursor, i, j, idx_cursor
            cursor = mv_write_sidx_loss(byte_mv, cursor, sidx[j], loss[j])

        cursor = mv_write_sidx_loss(byte_mv, cursor, 0, 0)  # terminator
        last_pair = -1
    return cursor, -1, -1, idx_cursor


def write_passthrough_event(pipe, idx_pipe, byte_mv, event_id, event_buffer,
                            idx_file_cursors, file_idx, idx_buffer):
    """Write one input's event to its pass-through stream in PIPE_CAPACITY chunks, and its .idx records to idx_pipe.

    Args:
        pipe (BinaryIO): pass-through output stream of the input
        idx_pipe (BinaryIO | None): .idx stream of the pass-through output, None without -m
        byte_mv (ndarray): numpy byte view of the PIPE_CAPACITY output buffer
        event_id (event_id_dtype): event id
        event_buffer (Dict[str, ndarray]): the input's event, as yielded by SummaryCombineReader.read_events
        idx_file_cursors (ndarray[int64]): bytes written so far to each output, updated in place
        file_idx (int): index of the input
        idx_buffer (ndarray[summary_stream_index_dtype]): buffer for the .idx records of the event
    """
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
    """Combine the rl summary streams of every inuring priority into one summary stream.

    Args:
        summary_files_in (List[str]): ``summarypy -t rl -s`` stream of each inuring priority
        output_summary_file (str): path of the combined summary stream
        static_paths (List[str]): static folder of each inuring priority, in the same order as summary_files_in
        low_memory (bool): if True, write a .idx file next to each output
        intermediate_outputs (List[str] | None): pass-through outputs, as ``input_index=path``
        output_zeros (bool): if True, samples with a zero loss are written to the combined output
        **kwargs: other command line options, unused

    Raises:
        OasisException: if the inputs do not match their static paths, the events of the inputs are not
            in the same order, or an input ends before the others
        ValueError: if an intermediate output refers to an input that does not exist
    """
    intermediate_outputs = parse_intermediate_outputs(intermediate_outputs)

    if len(summary_files_in) != len(static_paths):
        raise OasisException('each summary stream file needs a corresponding static path')

    n_inputs = len(summary_files_in)

    if any(k < 0 or k > n_inputs - 1 for k in intermediate_outputs.keys()):
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
            intermediate_idx_pipes = {i: stack.enter_context(open(intermediate_path.rsplit('.', 1)[0] + '.idx', 'wb'))
                                      for i, intermediate_path in intermediate_outputs.items()}

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

            present_ids = combine_events([ev[1] for ev in curr_events], acc)
            write_combined_event(output_pipe, output_index_pipe, out_byte_mv, event_id, len_sample,
                                 present_ids, acc, output_zeros, idx_file_cursors, -1, idx_buffer)
            acc.reset(present_ids)

        if any([len(input_queue) != 0 for input_queue in input_queues]):
            raise OasisException('Summary streams only partially combined.')


@redirect_logging(exec_name='rlcombine')
def main(summary_files_in, output_summary_file, static_paths, low_memory,
         intermediate_outputs, output_zeros, **kwargs):
    """Entry point of rlcombine, see run for the arguments."""
    run(summary_files_in, output_summary_file, static_paths, low_memory,
        intermediate_outputs, output_zeros, **kwargs)
