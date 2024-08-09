"""
Contain all common function and attribute to help read the event stream containing the losses
"""

import selectors
from select import select
import sys

import numpy as np
import numba as nb

from .data import oasis_int, oasis_int_size, oasis_float, oasis_float_size

# streams
PIPE_CAPACITY = 65536  # bytes

# stream source type
CDF_STREAM_ID = 0
GUL_STREAM_ID = 1  # deprecated use LOSS_STREAM_ID
FM_STREAM_ID = 2  # deprecated use LOSS_STREAM_ID
LOSS_STREAM_ID = 2
SUMMARY_STREAM_ID = 3

# stream aggregation type (represent the time of aggregation used
ITEM_STREAM = 1
COVERAGE_STREAM = 2


# special sample id
MEAN_IDX = -1
STD_DEV_IDX = -2
TIV_IDX = -3
CHANCE_OF_LOSS_IDX = NUMBER_OF_AFFECTED_RISK_IDX = -4
MAX_LOSS_IDX = -5


def stream_info_to_bytes(stream_source_type, stream_agg_type):
    """
    From Stream source type and aggregation type produce the stream header
    Args:
        stream_source_type (np.int32):
        stream_agg_type (np.int32):

    Returns:
        return bytes
    """
    return np.array([stream_agg_type], '<i4').tobytes()[:3] + np.int8(stream_source_type).tobytes()


def bytes_to_stream_types(stream_header):
    """
    Read the stream header and return the information on stream type
    Args:
        stream_header: bytes

    Returns:
        (stream source type (np.int32), stream aggregation type (np.int32))
    """
    return np.frombuffer(stream_header[3:], 'i1')[0], np.frombuffer(stream_header[:3] + b'\x00', '<i4')[0]


def read_stream_info(stream_obj):
    """
    from open stream object return the information that characterize the stream (stream_source_type, stream_agg_type, len_sample)
    Args:
        stream_obj: open stream
    Returns:
        (stream_source_type, stream_agg_type, len_sample) as np.int32 triplet
    """
    stream_source_type, stream_agg_type = bytes_to_stream_types(stream_obj.read(4))
    len_sample = np.frombuffer(stream_obj.read(4), dtype=np.int32)[0]
    return stream_source_type, stream_agg_type, len_sample


def get_streams_in(files_in, stack):
    if files_in is None:
        streams_in = [sys.stdin.buffer]
    elif isinstance(files_in, list):
        streams_in = [stack.enter_context(open(file_in, 'rb')) for file_in in files_in]
    else:
        streams_in = [stack.enter_context(open(files_in, 'rb'))]
    return streams_in


def get_and_check_header_in(streams_in):
    streams_info = [read_stream_info(stream_in) for stream_in in streams_in]
    if len(set(streams_info)) > 1:
        raise IOError(f"multiple stream type detected in streams {dict(enumerate(streams_info))}")
    return streams_info[0]


def init_streams_in(files_in, stack):
    """
    if files_in use stdin as stream in
    otherwise open each path in files_in, read the header, check that they are the same, and return the streams and their info
    Args:
        files_in: none or a list of path
        stack: contextlib stack to add the open stream to

    Returns:
        list of open streams and their info
    """
    streams_in = get_streams_in(files_in, stack)
    return streams_in, get_and_check_header_in(streams_in)


@nb.jit(nopython=True, cache=True)
def mv_read(byte_mv, cursor, _dtype, itemsize):
    """
    read a certain dtype from numpy byte view starting at cursor, return the value and the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        _dtype: data type of the object
        itemsize: size of the data type

    Returns:
        (object value, end of object index)
    """
    return byte_mv[cursor:cursor + itemsize].view(_dtype)[0], cursor + itemsize


@nb.jit(nopython=True, cache=True)
def mv_write(byte_mv, cursor, _dtype, itemsize, value) -> int:
    """
    load an object into the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        _dtype: data type of the object
        itemsize: size of the data type
        value: value to write

    Returns:
        end of object index
    """
    byte_mv[cursor:cursor + itemsize].view(_dtype)[0] = value
    return cursor + itemsize


@nb.jit(nopython=True, cache=True)
def mv_write_summary_header(byte_mv, cursor, event_id, summary_id, exposure_value) -> int:
    """
    write a summary header to the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        event_id: event id
        summary_id: summary id
        exposure_value: exposure value

    Returns:
        end of object index
    """
    # print(event_id, summary_id, exposure_value)
    cursor = mv_write(byte_mv, cursor, oasis_int, oasis_int_size, event_id)
    cursor = mv_write(byte_mv, cursor, oasis_int, oasis_int_size, summary_id)
    cursor = mv_write(byte_mv, cursor, oasis_float, oasis_float_size, exposure_value)
    return cursor


@nb.jit(nopython=True, cache=True)
def mv_write_item_header(byte_mv, cursor, event_id, item_id) -> int:
    """
    write a item header to the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        event_id: event id
        item_id: item id

    Returns:
        end of object index
    """
    # print(event_id, item_id)
    cursor = mv_write(byte_mv, cursor, oasis_int, oasis_int_size, event_id)
    cursor = mv_write(byte_mv, cursor, oasis_int, oasis_int_size, item_id)
    return cursor


@nb.jit(nopython=True, cache=True)
def mv_write_sidx_loss(byte_mv, cursor, sidx, loss) -> int:
    """
    write sidx and loss to the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        sidx: sample id
        loss: loss

    Returns:
        end of object index
    """
    # print('    ', sidx, loss)
    cursor = mv_write(byte_mv, cursor, oasis_int, oasis_int_size, sidx)
    cursor = mv_write(byte_mv, cursor, oasis_float, oasis_float_size, loss)
    return cursor


@nb.jit(nopython=True, cache=True)
def mv_write_delimiter(byte_mv, cursor) -> int:
    """
    write the item delimiter (0,0) to the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start

    Returns:
        end of delimiter index
    """
    cursor = mv_write(byte_mv, cursor, oasis_int, oasis_int_size, 0)
    cursor = mv_write(byte_mv, cursor, oasis_float, oasis_float_size, 0)
    # print('end', cursor)
    return cursor


class EventReader:
    """
    Abstract class to read event stream

    This class provide a generic interface to read multiple event stream using:
    - selector : handle back pressure, the program is paused and don't use resource if nothing is in the stream buffer
    - memoryview : read a chuck (PIPE_CAPACITY) of data at a time then work on it using a numpy byte view of this buffer

    To use those methods need to be implemented:
    - __init__(self, ...) the constructor with all data structure needed to read and store the event stream
    - read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id)
        simply point to a local numba.jit function name read_buffer (a template is provided bellow)
        this function should implement the specific logic of where and how to store the event information.

    Those to method may be overwritten
    - item_exit(self):
        specific logic to do when an item is finished (only executed once the stream is finished but no 0,0 closure was present)

    - event_read_log(self):
        what kpi to log when a full event is read

    usage snippet:
        with ExitStack() as stack:
            streams_in, (stream_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
            reader = CustomReader(<read relevant attributes>)
            for event_id in reader.read_streams(streams_in):
                <event logic>

    """

    @staticmethod
    def register_streams_in(selector_class, streams_in):
        """
        Data from input process is generally sent by event block, meaning once a stream receive data, the complete event is
        going to be sent in a short amount of time.
        Therefore, we can focus on each stream one by one using their specific selector 'stream_selector'.

        """
        main_selector = selector_class()
        stream_data = []
        for stream_in in streams_in:
            mv = memoryview(bytearray(PIPE_CAPACITY))
            byte_mv = np.frombuffer(buffer=mv, dtype='b')

            stream_selector = selector_class()
            stream_selector.register(stream_in, selectors.EVENT_READ)
            data = {'mv': mv,
                    'byte_mv': byte_mv,
                    'cursor': 0,
                    'valid_buff': 0,
                    'stream_selector': stream_selector
                    }
            stream_data.append(data)
            main_selector.register(stream_in, selectors.EVENT_READ, data)
        return main_selector, stream_data

    def read_streams(self, streams_in):
        """
        read multiple stream input, yield each event id and load relevant value according to the specific read_buffer implemented in subclass
        Args:
            streams_in: streams to read

        Returns:
            event id generator
        """
        try:
            main_selector, stream_data = self.register_streams_in(selectors.DefaultSelector, streams_in)
            self.logger.debug("Streams read with DefaultSelector")
        except PermissionError:  # Fall back option if stream_in contain regular files
            main_selector, stream_data = self.register_streams_in(selectors.SelectSelector, streams_in)
            self.logger.debug("Streams read with SelectSelector")
        try:
            while main_selector.get_map():
                for sKey, _ in main_selector.select():
                    event = self.read_event(sKey.fileobj, main_selector, **sKey.data)

                    if event:
                        event_id, cursor, valid_buff = event
                        sKey.data['cursor'] = cursor
                        sKey.data['valid_buff'] = valid_buff
                        self.event_read_log(event_id)
                        yield event_id

            # Stream is read, we need to check if there is remaining event to be parsed
            for data in stream_data:
                if data['cursor'] < data['valid_buff']:
                    byte_mv = data['byte_mv']
                    cursor = data['cursor']
                    valid_buff = data['valid_buff']
                    yield_event = True
                    while yield_event:
                        cursor, event_id, item_id, yield_event = self.read_buffer(byte_mv, cursor, valid_buff, 0, 0)

                        if event_id:
                            self.item_exit()
                            self.event_read_log(event_id)
                            yield event_id

        finally:
            main_selector.close()

    def read_event(self, stream_in, main_selector, stream_selector, mv, byte_mv, cursor, valid_buff):
        """
        read one event from stream_in
        close and remove the stream from main_selector when all is read
        Args:
            stream_in: stream to read
            main_selector: selector that contain all the streams
            stream_selector:  this stream selector
            mv: buffer memoryview
            byte_mv: numpy byte view of the buffer
            cursor: current cursor of the memory view
            valid_buff: valid data in memory view

        Returns:
            event_id, cursor, valid_buff
        """
        event_id = 0
        item_id = 0

        while True:
            if valid_buff < PIPE_CAPACITY:
                stream_selector.select()
                len_read = stream_in.readinto1(mv[valid_buff:])
                valid_buff += len_read

                if len_read == 0:
                    stream_selector.close()
                    main_selector.unregister(stream_in)
                    if event_id:
                        self.item_exit()
                        return event_id, cursor, valid_buff

                    break
            cursor, event_id, item_id, yield_event = self.read_buffer(byte_mv, cursor, valid_buff, event_id, item_id)

            if yield_event:
                if 2 * cursor > valid_buff:
                    mv[:valid_buff - cursor] = mv[cursor: valid_buff]
                    valid_buff -= cursor
                    cursor = 0
                return event_id, cursor, valid_buff
            else:
                mv[:valid_buff - cursor] = mv[cursor: valid_buff]
                valid_buff -= cursor
                cursor = 0

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id):
        raise NotImplementedError

    def item_exit(self):
        pass

    def event_read_log(self, event_id):
        pass


# read_buffer template,
# add you input argument and implement what is needed in the three steps ('do new item setup', 'do loss read', 'do item exit')
#
# @nb.jit(nopython=True, cache=True)
# def read_buffer(byte_mv, cursor, valid_buff, event_id, item_id, [array necessary to load and store the event data]):
#
#     last_event_id = event_id
#     while True:
#         if item_id:
#             if valid_buff -  cursor < (oasis_int_size + oasis_float_size):
#                 break
#             sidx, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
#             if sidx:
#                 loss, cursor = mv_read(byte_mv, cursor, oasis_float, oasis_float_size)
#                 loss = 0 if np.isnan(loss) else loss
#
#                 ###### do loss read ######
#                 ##########
#
#             else:
#                 ##### do item exit ####
#                 ##########
#                 cursor += oasis_float_size
#                 item_id = 0
#         else:
#             if valid_buff -  cursor < 2 * oasis_int_size:
#                 break
#             event_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
#             if event_id != last_event_id:
#                 if last_event_id: # we have a new event we return the one we just finished
#                     return cursor - oasis_int_size, last_event_id, 0, 1
#                 else: # first pass we store the event we are reading
#                     last_event_id = event_id
#             item_id, cursor = mv_read(byte_mv, cursor, oasis_int, oasis_int_size)
#
#             ##### do new item setup #####
#             ##########
#     return cursor, event_id, item_id, 0


def write_mv_to_stream(stream, byte_mv, cursor):
    """
    Write numpy byte array view to stream
    - use select to handle forward pressure
    - use a while loop in case the stream is non-blocking (meaning the ammount of byte written is not guarantied to be cursor len)
    Args:
        stream: stream to write to
        byte_mv: numpy byte view of the buffer to write
        cursor: ammount of byte to write

    """
    written = 0
    while written < cursor:
        _, writable, exceptional = select([], [stream], [stream])
        if exceptional:
            raise IOError(f'error with input stream, {exceptional}')
        written += stream.write(byte_mv[written:cursor].tobytes())
