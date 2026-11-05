"""Contain all common function and attribute to help read the event stream containing the losses"""

from oasislmf.utils.exceptions import OasisStreamException
import selectors
from select import select
import sys

import numpy as np
import numba as nb
from numba.core import cgutils, config as nb_config, types
from numba.extending import intrinsic, overload

from .data import def_to_type_and_size

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
NUM_SPECIAL_SIDX = 5
MEAN_IDX = -1
STD_DEV_IDX = -2
TIV_IDX = -3
CHANCE_OF_LOSS_IDX = NUMBER_OF_AFFECTED_RISK_IDX = -4
MAX_LOSS_IDX = -5


# Load type info
event_id_type, event_id_size = def_to_type_and_size("event_id")
item_id_type, item_id_size = def_to_type_and_size("item_id")
summary_id_dtype, summary_id_size = def_to_type_and_size("summary_id")
sidx_type, sidx_size = def_to_type_and_size("sidx")
loss_type, loss_size = def_to_type_and_size("loss")


def stream_info_to_bytes(stream_source_type, stream_agg_type):
    """From Stream source type and aggregation type produce the stream header

    Args:
        stream_source_type (np.int32): id of the tool that produced the stream
        stream_agg_type (np.int32): id of the aggregation level of the stream

    Returns:
        return bytes
    """
    return np.array([stream_agg_type], '<i4').tobytes()[:3] + np.int8(stream_source_type).tobytes()


def bytes_to_stream_types(stream_header):
    """Read the stream header and return the information on stream type

    Args:
        stream_header: bytes

    Returns:
        (stream source type (np.int32), stream aggregation type (np.int32))
    """
    return np.frombuffer(stream_header[3:], 'i1')[0], np.frombuffer(stream_header[:3] + b'\x00', '<i4')[0]


def read_stream_info(stream_obj):
    """From open stream object return the information that characterize the stream (stream_source_type, stream_agg_type, len_sample)

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
    """If files_in use stdin as stream in
    otherwise open each path in files_in, read the header, check that they are the same, and return the streams and their info

    Args:
        files_in: none or a list of path
        stack: contextlib stack to add the open stream to

    Returns:
        list of open streams and their info
    """
    streams_in = get_streams_in(files_in, stack)
    return streams_in, get_and_check_header_in(streams_in)


# mv_read / mv_write move one scalar at a time between a byte buffer and a typed value. They are
# the innermost call of every stream reader and writer, so they run once per sidx/loss pair.
#
# Refcounting in hot loops (issue #2240). In numba, every array view (slice, field view, `.view()`)
# and every call that passes an array costs an atomic NRT incref/decref pair. numba <= 0.65 deleted
# these pairs from small helpers (removerefctpass); numba PR #10511 removed that pass in 0.66, and
# llvmlite's RefPrune cannot remove a pair across a call layer or a `raise`. So per-value loops in
# pytools avoid creating views and avoid calls that take arrays; other modules refer to this note.
#
# The natural numba spelling, `byte_mv[cursor:cursor + itemsize].view(_dtype)[0]`, builds two
# temporary arrays per call, and its refcounting was ~40% of gulmc's run time. A single typed view
# of the whole buffer is not an option: the stream mixes field types (loss is f4 or f8 depending on
# OASIS_FLOAT), so fields sit at offsets that are not multiples of their own size.
#
# Instead, under JIT, the value is loaded/stored straight through a pointer to byte `cursor`
# (unaligned, native byte order, exactly what `.view()` does), with no array temporaries at all.
# mv_read / mv_write themselves stay plain Python so they still work without JIT (from Python
# callers and under NUMBA_DISABLE_JIT); the @overload below is what jitted callers compile.
#
# Overrun is checked only under NUMBA_BOUNDSCHECK, like every other numba array access. An
# always-on check costs ~17% of gulmc on numba >= 0.66: its `raise` branch sits between the
# incref/decref numba now emits around each inlined write, and llvmlite's RefPrune cannot remove a
# pair that spans a raise. Callers size the buffer before writing (gulpy/gulmc via
# adjust_byte_mv_size and a per-subtree space check, fmpy/summarypy by testing the cursor against
# PIPE_CAPACITY before each record); CI runs with the check on. The setting is read when the caller
# is compiled: a cache built without NUMBA_BOUNDSCHECK keeps the check off (see CLAUDE.md), and a
# per-function `@njit(boundscheck=True)` does not enable it.
#
# The helpers below marked `pragma: no cover` run only while numba compiles a jitted caller. The
# uploaded coverage comes from the NUMBA_DISABLE_JIT run, where nothing is compiled; the JIT test
# jobs and test_event_stream_mv.py exercise them.


def _scalar_type_of(dtype):  # pragma: no cover
    """Numba scalar type named by a dtype argument, or None if it is not a plain scalar.

    Callers pass both `np.dtype(...)` instances (typed as DType) and scalar classes such as
    `np.int32` (typed as NumberClass). A structured dtype (a Record) returns None, so it takes the
    view-based fallback rather than a scalar load/store.
    """
    if isinstance(dtype, types.DType):
        scalar_type = dtype.dtype
    elif isinstance(dtype, types.NumberClass):
        scalar_type = dtype.instance_type
    else:
        return None
    return scalar_type if _is_scalar(scalar_type) else None


def _is_contiguous_byte_array(byte_mv):  # pragma: no cover
    return isinstance(byte_mv, types.Array) and byte_mv.ndim == 1 and byte_mv.layout == 'C' and byte_mv.dtype.bitwidth == 8


def _is_scalar(value):  # pragma: no cover
    return isinstance(value, (types.Number, types.Boolean))


def _check_in_buffer(byte_mv, cursor, itemsize):  # pragma: no cover
    pass  # pure-Python path: the view-based code raises on overrun by itself


@overload(_check_in_buffer)
def _check_in_buffer_jit(byte_mv, cursor, itemsize):  # pragma: no cover
    if not nb_config.BOUNDSCHECK:
        return lambda byte_mv, cursor, itemsize: None

    def impl(byte_mv, cursor, itemsize):
        if cursor < 0 or cursor + itemsize > byte_mv.shape[0]:
            raise IndexError("event stream cursor out of the byte buffer")
    return impl


def _pointer_at(context, builder, sig, args, scalar_type):  # pragma: no cover
    """LLVM pointer to `scalar_type` at byte offset `cursor` (args[1]) of the byte array args[0]."""
    ary = context.make_array(sig.args[0])(context, builder, args[0])
    offset = context.cast(builder, args[1], sig.args[1], types.intp)
    addr = cgutils.pointer_add(builder, ary.data, offset)
    return builder.bitcast(addr, context.get_value_type(scalar_type).as_pointer())


# The two intrinsics are only called from the mv_read / mv_write overloads below, which have already
# checked for a contiguous (and, to store, mutable) byte buffer, a dtype and a scalar value.
@intrinsic
def _unaligned_load(typingctx, byte_mv, cursor, dtype):  # pragma: no cover
    scalar_type = _scalar_type_of(dtype)

    def codegen(context, builder, sig, args):
        return builder.load(_pointer_at(context, builder, sig, args, scalar_type), align=1)
    return scalar_type(byte_mv, cursor, dtype), codegen


@intrinsic
def _unaligned_store(typingctx, byte_mv, cursor, dtype, value):  # pragma: no cover
    scalar_type = _scalar_type_of(dtype)

    def codegen(context, builder, sig, args):
        # same implicit conversion as `view[0] = value`, e.g. a float64 loss into an f4 field
        val = context.cast(builder, args[3], sig.args[3], scalar_type)
        builder.store(val, _pointer_at(context, builder, sig, args, scalar_type), align=1)
        return context.get_dummy_value()
    return types.none(byte_mv, cursor, dtype, value), codegen


def _mv_read_view(byte_mv, cursor, _dtype, itemsize):
    return byte_mv[cursor:cursor + itemsize].view(_dtype)[0], cursor + itemsize


def _mv_write_view(byte_mv, cursor, _dtype, itemsize, value):
    byte_mv[cursor:cursor + itemsize].view(_dtype)[0] = value
    return cursor + itemsize


def mv_read(byte_mv, cursor, _dtype, itemsize):
    """Read a certain dtype from numpy byte view starting at cursor, return the value and the index of the end of the object

    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        _dtype: data type of the object
        itemsize: size of the data type

    Returns:
        (object value, end of object index)
    """
    return _mv_read_view(byte_mv, cursor, _dtype, itemsize)


@overload(mv_read)
def _mv_read_jit(byte_mv, cursor, _dtype, itemsize):  # pragma: no cover
    if _scalar_type_of(_dtype) is None or not _is_contiguous_byte_array(byte_mv):
        return _mv_read_view  # unusual buffer: keep the view-based implementation

    def impl(byte_mv, cursor, _dtype, itemsize):
        _check_in_buffer(byte_mv, cursor, itemsize)
        return _unaligned_load(byte_mv, cursor, _dtype), cursor + itemsize
    return impl


@nb.njit(cache=True)
def reservation_overflows(idx, reservation, capacity, name):
    """Check whether reserving `reservation` more rows in a fixed-size output buffer
    (used before reading a summary's raw data, so its worst-case output is guaranteed
    to fit - see elt/manager.py and plt/manager.py's read_buffer) would overflow it.

    Args:
        idx: current write position in the output buffer
        reservation: worst-case number of rows this summary could add
        capacity: output buffer's total size
        name: output type name, used only in the raised error message (e.g. "SELT")

    Returns:
        bool: True if the buffer needs to be flushed before this summary can be read.
            False if there's room.

    Raises:
        ValueError: if idx == 0 (buffer already empty) and it still doesn't fit -
            flushing an empty buffer can never make more room, so returning True here
            would have the caller loop forever instead of making progress.
    """
    if idx + reservation > capacity:
        if idx == 0:
            raise ValueError(
                f"{name} reservation of {reservation} rows for a single summary exceeds the "
                f"output buffer capacity of {capacity}; increase OASIS_DEFAULT_BUFFER_SIZE."
            )
        return True
    return False


def mv_write(byte_mv, cursor, _dtype, itemsize, value) -> int:
    """Load an object into the numpy byte view at index cursor, return the index of the end of the object

    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        _dtype: data type of the object
        itemsize: size of the data type
        value: value to write

    Returns:
        end of object index
    """
    return _mv_write_view(byte_mv, cursor, _dtype, itemsize, value)


@overload(mv_write)
def _mv_write_jit(byte_mv, cursor, _dtype, itemsize, value):  # pragma: no cover
    if _scalar_type_of(_dtype) is None or not _is_contiguous_byte_array(byte_mv) or not byte_mv.mutable:
        return _mv_write_view  # unusual buffer: keep the view-based implementation

    if _is_scalar(value):
        def impl(byte_mv, cursor, _dtype, itemsize, value):
            _check_in_buffer(byte_mv, cursor, itemsize)
            _unaligned_store(byte_mv, cursor, _dtype, value)
            return cursor + itemsize
        return impl

    if isinstance(value, types.Array) and value.ndim == 0 and _is_scalar(value.dtype):
        # a 0-d array (e.g. a numpy scalar reaching jitted code as `arr[()]`-able), which the
        # view-based version accepted through setitem; store the element it holds
        def impl_0d(byte_mv, cursor, _dtype, itemsize, value):
            _check_in_buffer(byte_mv, cursor, itemsize)
            _unaligned_store(byte_mv, cursor, _dtype, value[()])
            return cursor + itemsize
        return impl_0d

    return _mv_write_view  # any other value type: keep the view-based implementation's semantics


@nb.jit(nopython=True)
def mv_write_summary_header(byte_mv, cursor, event_id, summary_id, exposure_value) -> int:
    """Wrapper for cached write a summary header to the numpy byte view at index cursor, return the index of the end of the object

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
    return mv_write_summary_header_cached(byte_mv, cursor, event_id, summary_id, exposure_value,
                                          event_id_type, event_id_size, summary_id_dtype, summary_id_size,
                                          loss_type, loss_size)


@nb.jit(nopython=True, cache=True)
def mv_write_summary_header_cached(byte_mv, cursor, event_id, summary_id, exposure_value,
                                   event_id_type, event_id_size,
                                   summary_id_dtype, summary_id_size,
                                   exposure_value_dtype, exposure_value_size) -> int:
    """Cached write a summary header to the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        event_id: event id
        summary_id: summary id
        exposure_value: exposure value
        event_id_type: type info for event id
        event_id_size: size of event id in bytes
        summary_id_dtype: type info for summary id
        summary_id_size: size of summary id
        exposure_value_dtype: type info for exposure value
        exposure_value_size: size of exposure value

    Returns:
        end of object index
    """
    cursor = mv_write(byte_mv, cursor, event_id_type, event_id_size, event_id)
    cursor = mv_write(byte_mv, cursor, summary_id_dtype, summary_id_size, summary_id)
    cursor = mv_write(byte_mv, cursor, exposure_value_dtype, exposure_value_size, exposure_value)
    return cursor


@nb.jit(nopython=True)
def mv_write_item_header(byte_mv, cursor, event_id, item_id) -> int:
    """Wrapper function for cached mv_write_item_header. writes an item header to the numpy byte view,
    return index of the end of the object

    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        event_id: event id
        item_id: item id

    Returns:
        end of object index
    """
    # print(event_id, item_id)
    cursor = mv_write(byte_mv, cursor, event_id_type, event_id_size, event_id)
    cursor = mv_write(byte_mv, cursor, item_id_type, item_id_size, item_id)
    return cursor


@nb.jit(nopython=True)
def mv_write_sidx_loss(byte_mv, cursor, sidx, loss) -> int:
    """Write sidx and loss to the numpy byte view at index cursor, return the index of the end of the object

    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        sidx: sample id
        loss: loss

    Returns:
        end of object index
    """
    return mv_write_sidx_loss_cached(byte_mv, cursor, sidx, loss,
                                     sidx_type=sidx_type, sidx_size=sidx_size,
                                     loss_type=loss_type, loss_size=loss_size)


# inlined so byte_mv is not increfed/decrefed across a call boundary once per sidx/loss pair;
# measured: ~14% less gulmc run time on numba 0.68, neutral on 0.61 (issue #2240).
# Inlining the outer, uncached wrappers too was measured to slow numba 0.61 down, so they stay calls.
@nb.jit(nopython=True, cache=True, inline='always')
def mv_write_sidx_loss_cached(byte_mv, cursor, sidx, loss, sidx_type,
                              loss_type, sidx_size, loss_size) -> int:
    """Cached write sidx and loss to the numpy byte view at index cursor, return the index of the end of the object
    Args:
        byte_mv: numpy byte view
        cursor: index of where the object start
        sidx: sample id
        loss: loss
        sidx_type: sample id type info
        loss_type: loss type info
        sidx_size: sample id size in bytes
        loss_size: loss size in bytes

    Returns:
        end of object index
    """
    # print('    ', sidx, loss)
    cursor = mv_write(byte_mv, cursor, sidx_type, sidx_size, sidx)
    cursor = mv_write(byte_mv, cursor, loss_type, loss_size, loss)
    return cursor


class EventReader:
    """Abstract class to read event stream

    This class provides a generic interface to read multiple event streams using:

    - **selector**: handle back pressure — the program is paused and doesn't use resources if
      nothing is in the stream buffer
    - **memoryview**: read a chunk (PIPE_CAPACITY) of data at a time then work on it using a
      numpy byte view of this buffer

    These methods need to be implemented:

    - ``__init__(self, ...)``: the constructor with all data structures needed to read and store
      the event stream
    - ``read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id)``: point to a local
      numba.jit function named read_buffer (a template is provided below); it should implement
      the specific logic of where and how to store the event information

    These methods may be overwritten:

    - ``item_exit(self)``: specific logic to do when an item is finished (only executed once the
      stream is finished but no 0,0 closure was present)
    - ``event_read_log(self)``: what kpi to log when a full event is read

    Usage snippet::

        with ExitStack() as stack:
            streams_in, (stream_type, stream_agg_type, len_sample) = init_streams_in(files_in, stack)
            reader = CustomReader(<read relevant attributes>)
            for event_id in reader.read_streams(streams_in):
                <event logic>
    """

    @staticmethod
    def register_streams_in(selector_class, streams_in):
        """Data from input process is generally sent by event block, meaning once a stream receive data, the complete event is
        going to be sent in a short amount of time.
        Therefore, we can focus on each stream one by one using their specific selector 'stream_selector'.
        """
        main_selector = selector_class()
        stream_data = []
        for file_idx, stream_in in enumerate(streams_in):
            mv = memoryview(bytearray(PIPE_CAPACITY))
            byte_mv = np.frombuffer(buffer=mv, dtype='b')

            stream_selector = selector_class()
            stream_selector.register(stream_in, selectors.EVENT_READ)
            data = {'mv': mv,
                    'byte_mv': byte_mv,
                    'cursor': 0,
                    'valid_buff': 0,
                    'stream_selector': stream_selector,
                    'file_idx': file_idx,
                    }
            stream_data.append(data)
            main_selector.register(stream_in, selectors.EVENT_READ, data)
        return main_selector, stream_data

    def read_streams(self, streams_in):
        """Read multiple stream input, yield each event id and load relevant value according to the specific read_buffer implemented in subclass

        Args:
            streams_in: streams to read

        Yields:
            int: each event id read from the streams
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
                        self.finalize_event()
                        yield event_id

            # Stream is read, we need to check if there is remaining event to be parsed
            for data in stream_data:
                if data['cursor'] < data['valid_buff']:
                    byte_mv = data['byte_mv']
                    cursor = data['cursor']
                    valid_buff = data['valid_buff']
                    file_idx = data['file_idx']
                    yield_event = True
                    while yield_event:
                        cursor, event_id, item_id, yield_event = self.read_buffer(byte_mv, cursor, valid_buff, 0, 0, file_idx=file_idx)

                        if event_id:
                            self.item_exit()
                            self.event_read_log(event_id)
                            self.finalize_event()
                            yield event_id

        finally:
            main_selector.close()

    def read_event(self, stream_in, main_selector, stream_selector, mv, byte_mv, cursor, valid_buff, file_idx):
        """Read one event from stream_in
        close and remove the stream from main_selector when all is read

        Args:
            stream_in: stream to read
            main_selector: selector that contain all the streams
            stream_selector:  this stream selector
            mv: buffer memoryview
            byte_mv: numpy byte view of the buffer
            cursor: current cursor of the memory view
            valid_buff: valid data in memory view
            file_idx: file index

        Returns:
            event_id, cursor, valid_buff
        """
        event_id = 0
        item_id = 0
        try:
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
                cursor, event_id, item_id, yield_event = self.read_buffer(byte_mv, cursor, valid_buff, event_id, item_id, file_idx=file_idx)

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

        except Exception as e:
            self.logger.error(str(e))
            raise OasisStreamException("Error reading stream", e)

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        raise NotImplementedError

    def item_exit(self):
        pass

    def finalize_event(self):
        """Hook called by read_streams just before yielding each event_id.
        Subclasses use this to perform any per-event finishing work that read_buffer
        cannot do on its own — e.g. sorting accumulated state. Default is a no-op.
        """
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
    """Write numpy byte array view to stream
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
