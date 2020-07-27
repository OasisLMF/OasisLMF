"""
Module responsible for reading and writing FM stream.

Some detail on the data layout.
event id fron the stream can be:
 - negative for special values like the mean or the standard deviation.
 - positive for normal sample id
in the several data matrix:
 - index 0 is not used
 - length is the given len_sample + the number of special values + 1
 - value are arranged so the index in the data correspond to the sample id,
   the special values are place at the end therefor.
   ie: [0, sidx_1, sidx_2, ...sidx_len_sample, sidx_-3, sidx_-2, sidx_-1]
   here we use python ability to understand negative indexes to equate sidx and index simplifying index management


"""
from .financial_structure import PROFILE
from .common import float_equal_precision, nb_oasis_float, nb_oasis_int
from .queue import QueueTerminated

import sys
from numba import njit, int32
import numpy as np
import select
import logging
logger = logging.getLogger(__name__)


#gul_header = np.int32(1 << 24).tobytes()
fm_header = np.int32(1 | 2 << 24).tobytes()


buff_size = 1_048_576 * 32
number_size = 4
nb_number = buff_size // number_size

EXTRA_VALUES = 3


@njit(cache=True)
def stream_to_loss_table(stream_as_int, stream_as_float, valid_len, last_event_cursor, node_to_index, losses, not_null):
    last_event_id = stream_as_int[last_event_cursor]
    cursor = last_event_cursor

    while cursor < valid_len - 2:
        event_id, cursor = stream_as_int[cursor], cursor + 1
        if event_id != last_event_id:
            return last_event_id, last_event_cursor, 1

        i, cursor = node_to_index[(nb_oasis_int(1), nb_oasis_int(0), nb_oasis_int(stream_as_int[cursor]), PROFILE)][1], cursor + 1
        not_null[i] = True

        while cursor < valid_len - 2:
            sidx, cursor = nb_oasis_int(stream_as_int[cursor]), cursor + 1
            loss, cursor = nb_oasis_float(stream_as_float[cursor]), cursor + 1
            if sidx == 0:
                last_event_cursor = cursor
                break
            elif sidx == -2:
                pass

            losses[i, sidx] = 0 if np.isnan(loss) else loss

    return last_event_id, last_event_cursor, 0


def read_stream_header(stream_obj):
    stream_type = stream_obj.read(4)
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]
    return stream_type, len_sample


def read_event(stream_in, node_to_index, len_inputs, len_sample):

    losses = np.zeros([len_inputs, len_sample + EXTRA_VALUES + 1], dtype=np.float32)
    not_null = np.zeros([len_inputs, ], dtype=np.bool)

    buf = bytearray(buff_size)
    mv = memoryview(buf)

    stream_as_int32 = np.ndarray(nb_number, buffer=mv, dtype=np.int32)
    stream_as_float32 = np.ndarray(nb_number, buffer=mv, dtype=np.float32)

    last_event_cursor = 0
    read_cursor = 0

    while True:
        readable, _, exceptional = select.select([stream_in], [], [stream_in])
        if exceptional:
            raise IOError(f'error with input stream, {exceptional}')
        len_read = readable[0].readinto1(mv[read_cursor:])
        valid_buf = len_read + read_cursor

        if len_read == 0:
            break

        while True:
            last_event_id, last_event_cursor, full_event = stream_to_loss_table(stream_as_int32, stream_as_float32, valid_buf // number_size,
                                                                 last_event_cursor, node_to_index, losses, not_null)
            if full_event:
                yield last_event_id, losses, not_null
                losses = np.zeros([len_inputs, len_sample + EXTRA_VALUES +1], dtype=np.float32)
                not_null = np.zeros([len_inputs, ], dtype=np.bool)
            else:
                read_cursor = valid_buf - number_size * last_event_cursor
                mv[:read_cursor] = mv[number_size * last_event_cursor: valid_buf]
                last_event_cursor = 0
                break
    try:
        yield last_event_id, losses,  not_null
    except UnboundLocalError:  # Nothing was read
        pass


def queue_event_reader(event_queue, file_in, node_to_index, len_indexes):
    try:
        if file_in is None:
            stream_in = sys.stdin.buffer
        else:
            stream_in = open(file_in, 'rb')
        try:
            stream_type, len_sample = read_stream_header(stream_in)

            for event in read_event(stream_in, node_to_index, len_indexes, len_sample):
                logger.debug(f"reading {event[0]}")
                try:
                    event_queue.put(event)
                except QueueTerminated:
                    logger.warning(f"stopped because exception was raised")
                    break
            logger.info(f"reading done")
        finally:
            if file_in:
                stream_in.close()
    except Exception:
        event_queue.terminate = True
        raise


@njit(cache=True)
def load_event(as_int, as_float, event_id, output_item_index, losses, not_null):
    cursor = 0
    for output in output_item_index:
        if not_null[output['index']]:
            as_int[cursor], cursor = event_id, cursor + 1
            as_int[cursor], cursor = output['output_id'], cursor + 1
            loss = losses[output['index']]

            for i in [-3, -1]:
                as_int[cursor], cursor = i, cursor + 1
                as_float[cursor], cursor = loss[i], cursor + 1

            for i in range(1, loss.shape[0] - EXTRA_VALUES):
                if loss[i] > float_equal_precision:
                    as_int[cursor], cursor = i, cursor + 1
                    as_float[cursor], cursor = loss[i], cursor + 1

            as_int[cursor], cursor = 0, cursor + 1
            as_float[cursor], cursor = 0, cursor + 1
    return cursor


class EventWriter:
    def __init__(self, files_out, output_item_index, len_sample):
        self.files_out = files_out
        self.output_item_index = output_item_index

        self.len_sample = len_sample # all normal sidx plus the extra value plus 1 (index 0)
        self.loss_shape_1 = len_sample + EXTRA_VALUES + 1
        nb_values = (2 * (len_sample + EXTRA_VALUES + 1)) * len(output_item_index)  # (event_id + item_id + len_sample) * number of items
        self.int_size = np.dtype(np.int32).itemsize
        self.mv = memoryview(bytearray(nb_values * self.int_size))

        self.as_int = np.ndarray(nb_values, buffer=self.mv, dtype=np.int32)
        self.as_float = np.ndarray(nb_values, buffer=self.mv, dtype=np.float32)

    def __enter__(self):
        if self.files_out is None:
            self.stream_out = sys.stdout.buffer
        else:
            self.stream_out = open(self.files_out, 'wb')
        # prepare output buffer
        self.stream_out.write(fm_header)
        self.stream_out.write(np.int32(self.len_sample).tobytes())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.files_out:
            self.stream_out.close()

    def write(self, event):
        event_id, loss, not_null = event
        if self.loss_shape_1 != loss.shape[1]:
            raise ValueError(f"event {event_id} has a different sample len {loss.shape[1]}")

        cursor = load_event(self.as_int, self.as_float, event_id, self.output_item_index, loss, not_null) * self.int_size
        self.stream_out.write(self.mv[:cursor])


def queue_event_writer(event_queue, files_out, output_item_index, sentinel):
    try:
        event = event_queue.get()
        if event != sentinel:
            event_id, loss, not_null = event
            len_sample = loss.shape[1] - EXTRA_VALUES - 1

            with EventWriter(files_out, output_item_index, len_sample) as event_writer_handler:
                logger.debug(f"writing {event[0]}")
                event_writer_handler.write(event)
                logger.debug(f"writen {event[0]}")

                while True:
                    event = event_queue.get()
                    if event == sentinel:
                        break

                    logger.debug(f"writing {event[0]}")
                    event_writer_handler.write(event)
                    logger.debug(f"writen {event[0]}")
        logger.info(f"writing done")
    except Exception:
        event_queue.terminate = True
        raise
