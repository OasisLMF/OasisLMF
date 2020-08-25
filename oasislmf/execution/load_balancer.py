#!/usr/bin/env python
import argparse

import select
import numpy as np
from numba import njit
import time
from io import BytesIO
import queue
import concurrent.futures

import logging
logger = logging.getLogger(__name__)


last_event_padding = b'\x00\x00\x00\x00\x00\x00\x00\x00'
number_size = 4
CHECK_STOPPER_DURATION = 1


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--pipe-in', help='names of the input file_path', nargs='+')
parser.add_argument('-o', '--pipe-out', help='names of the output file_path', nargs='+')
parser.add_argument('-r', '--read-size', help='maximum size of chunk read from input', default=1_048_576, type=int)
parser.add_argument('-w', '--write-size', help='maximum size of chunk read from input', default=1024, type=int)
parser.add_argument('-q', '--queue-size', help='maximum size of the queue', default=50, type=int)
parser.add_argument('-v', '--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50',
                    default=30, type=int)


class ProducerStopped(RuntimeError):
    pass


@njit(cache=True)
def get_next_event_index(read_buffer, last_event_index, last_event_id, max_cursor):
    """
    try to get the index of the end of the event
    if found return the index and 0 to indicate it is found
    if not found return the index of the last item parsed and the last event id

    :param sub: byte array to parse
    :param last_item_index: last index parsed
    :param last_event_id: last event idea parsed (0 means no event)
    :return: last index parsed, last event idea parsed (0 means the chunk sub[:last_item_index] is a full event
    """
    cursor = last_event_index
    while cursor < max_cursor - 4:
        cur_event_id = read_buffer[cursor]
        if last_event_id != cur_event_id:
            if last_event_id == 0:
                last_event_id = read_buffer[cursor]
            else:
                return cursor, last_event_id, 1

        cursor += 2
        while cursor < max_cursor - 2:
            sidx = read_buffer[cursor]
            cursor += 2
            if sidx == 0:
                last_event_index = cursor
                break
    return last_event_index, last_event_id, 0


def produce(even_queue, event, stopper):
    while not stopper[0]:
        try:
            even_queue.put(event, timeout=CHECK_STOPPER_DURATION)
            break
        except queue.Full:
            pass
    else:
        raise ProducerStopped()


def producer(in_stream, pipeline, read_size, stopper):
    read_buffer = memoryview(bytearray(read_size))
    buf_as_int32 = np.ndarray(read_size // number_size, buffer=read_buffer, dtype=np.int32)
    event_buffer = BytesIO()
    left_over = 0
    last_event_index = 0
    last_event_id = 0

    wait_read_time = 0
    read_input_time = 0
    parse_input_time = 0
    buffer_management_time = 0

    while True:
        tw = time.time()
        select.select([in_stream], [], [])
        tr = time.time()
        wait_read_time += tr - tw
        len_read = in_stream.readinto1(read_buffer[left_over:])
        read_input_time += time.time() - tr
        if not len_read:
            in_stream.close()
            event_buffer.write(read_buffer[:left_over])
            if read_buffer[left_over - 8:left_over-4] != b'\x00\x00\x00\x00':
                event_buffer.write(last_event_padding)
            event_buffer.seek(0)
            produce(pipeline, event_buffer, stopper)
            break

        valid_buf = len_read + left_over

        while True:
            tp = time.time()
            event_index, last_event_id, event_finished = get_next_event_index(buf_as_int32, last_event_index, last_event_id,
                                                                              valid_buf // number_size)
            tm = time.time()
            parse_input_time += tm - tp

            event_buffer.write(read_buffer[last_event_index * number_size: event_index * number_size])

            if event_finished:
                event_buffer.seek(0)
                produce(pipeline, event_buffer, stopper)
                event_buffer = BytesIO()
                last_event_index = event_index
                last_event_id = 0
            else:
                left_over = valid_buf - number_size * event_index
                read_buffer[:left_over] = read_buffer[number_size * event_index: valid_buf]
                last_event_index = 0
                break
            buffer_management_time += time.time() - tm

    return wait_read_time, read_input_time, parse_input_time, buffer_management_time


def consumer(out_stream, pipeline, write_size, sentinel, stopper):
    s_tot=0
    w_tot=0
    p_tot=0
    while True:
        tp = time.time()
        event_buf = pipeline.get()
        p_tot += time.time() - tp
        if event_buf is sentinel:
            break
        else:
            while True:
                data = event_buf.read(write_size)
                if not data:
                    break
                ts = time.time()
                select.select([], [out_stream], [])
                tw = time.time()
                s_tot += tw-ts
                try:
                    out_stream.write(data)
                except:
                    stopper[0] = True
                    raise
                w_tot += time.time() - tw
    return s_tot, w_tot, p_tot


def balance(pipe_in, pipe_out, read_size, write_size, queue_size):
    """
    Load balance events for a list of input fil_path to a list of output fil_path

    :param pipe_in: list of fil_path
        fil_path to take as input
    :param pipe_out: list of fil_path
        fil_path to take as input
    :param read_size: int
        size of the maximum amount of Byte read from one input at a time
    :param queue_size: int
        maximum size ofthe buffer queue

    """
    inputs = [open(p, 'rb') for p in pipe_in]
    outputs = [open(p, 'wb') for p in pipe_out]

    pipeline = queue.Queue(maxsize=queue_size)
    sentinel = object()
    stopper = np.zeros(1, dtype=np.bool)
    try:
        # check stream input header and write it to the stream output
        headers = set([s.read(8) for s in inputs])
        if len(headers) != 1:
            raise Exception('input streams have different header type')
        header = headers.pop()
        [s.write(header) for s in outputs]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(inputs)+len(outputs)) as executor:
            producer_task = [executor.submit(producer, s, pipeline, read_size, stopper) for s in inputs]
            consumer_task = [executor.submit(consumer, s, pipeline, write_size, sentinel, stopper) for s in outputs]
            try:
                prod_t = [t.result() for t in producer_task]
            finally:
                for _ in pipe_out:
                    pipeline.put(sentinel)

            wait_read_time, read_input_time, parse_input_time, buffer_management_time = 0, 0, 0, 0
            for t in prod_t:
                wait_read_time += t[0]
                read_input_time += t[1]
                parse_input_time += t[2]
                buffer_management_time += t[3]

            cons_t = [t.result() for t in consumer_task]

            wait_write_time, write_output_time, wait_pipeline = 0, 0, 0
            for t in cons_t:
                wait_write_time += t[0]
                write_output_time += t[1]
                wait_pipeline += t[2]

            logger.info(f"""
    wait_read_time = {wait_read_time}, {wait_read_time / len(inputs)}
    read_input_time = {read_input_time}, {read_input_time / len(inputs)}
    parse_input_time = {parse_input_time}, {parse_input_time / len(inputs)}
    buffer_management_time = {buffer_management_time}, {buffer_management_time / len(inputs)}
    wait_write_time = {wait_write_time}, {wait_write_time / len(outputs)}
    write_output_time = {write_output_time}, {write_output_time / len(outputs)}
    wait_pipeline = {wait_pipeline}, {wait_pipeline / len(outputs)}
    """)


    finally:
        [s.close() for s in inputs]
        [s.close() for s in outputs]


def main():
    kwargs = vars(parser.parse_args())

    # add handler to fm logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    balance(**kwargs)


if __name__ == '__main__':
    main()
