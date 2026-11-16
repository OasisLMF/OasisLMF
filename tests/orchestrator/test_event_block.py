import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.common.stage import EventBlock


def test_new_block_is_empty():
    block = EventBlock()
    assert len(block) == 0
    assert block.size == 0
    assert block.capacity == 2 * PIPE_CAPACITY
    view = block.view()
    assert view.dtype == np.int8
    assert view.shape == (0,)


@pytest.mark.parametrize('data', [b'\x01\x02\xff', bytearray(b'\x01\x02\xff'), memoryview(b'\x01\x02\xff')])
def test_write_buffer_protocol_objects(data):
    block = EventBlock()
    assert block.write(data) == 3
    assert block.size == 3
    assert block.tobytes() == b'\x01\x02\xff'
    np.testing.assert_array_equal(block.view(), np.array([1, 2, -1], dtype=np.int8))


def test_writes_append():
    block = EventBlock()
    block.write(b'ab')
    block.write(b'')
    block.write(b'cde')
    assert len(block) == 5
    assert block.tobytes() == b'abcde'


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.bool_, 'S1', 'V1'])
def test_write_one_byte_numpy_dtypes(dtype):
    raw = bytes([0, 1, 1, 0, 1])
    arr = np.frombuffer(raw, dtype=dtype)
    block = EventBlock()
    assert block.write(arr) == 5
    assert block.tobytes() == raw


def test_write_non_contiguous_numpy_slice():
    arr = np.arange(20, dtype=np.int8)
    sliced = arr[::3]
    assert not sliced.flags.c_contiguous
    block = EventBlock()
    assert block.write(sliced) == len(sliced)
    assert block.tobytes() == sliced.tobytes()


def test_write_reversed_and_2d_arrays():
    arr = np.arange(12, dtype=np.uint8)
    block = EventBlock()
    block.write(arr[::-1])
    block.write(arr.reshape(3, 4).T)
    assert block.tobytes() == arr[::-1].tobytes() + arr.reshape(3, 4).T.tobytes()


def test_write_non_contiguous_memoryview():
    mv = memoryview(bytes(range(10)))[::2]
    assert not mv.c_contiguous
    block = EventBlock()
    assert block.write(mv) == 5
    assert block.tobytes() == bytes(range(0, 10, 2))


def test_write_multi_byte_array_appends_raw_bytes():
    arr = np.array([1, -2, 300], dtype=np.int32)
    block = EventBlock()
    assert block.write(arr) == 12
    assert block.write(memoryview(arr)) == 12
    assert block.tobytes() == arr.tobytes() * 2


def test_write_read_only_array():
    arr = np.frombuffer(b'xyz', dtype=np.int8)
    assert not arr.flags.writeable
    block = EventBlock()
    block.write(arr)
    assert block.tobytes() == b'xyz'


def test_grows_by_doubling_and_keeps_data():
    block = EventBlock(capacity=4)
    block.write(b'abc')
    assert block.capacity == 4
    block.write(b'de')
    assert block.capacity == 8
    block.write(b'x' * 20)
    assert block.capacity == 32
    assert block.tobytes() == b'abcde' + b'x' * 20


def test_view_is_no_copy():
    block = EventBlock(capacity=16)
    block.write(b'abcd')
    view = block.view()
    assert view.shape == (4,)
    assert np.shares_memory(view, block.view())
    assert view.base is not None
    view[0] = ord('z')
    assert block.tobytes() == b'zbcd'


def test_clear_keeps_capacity():
    block = EventBlock(capacity=4)
    block.write(b'0123456789')
    capacity = block.capacity
    block.clear()
    assert len(block) == 0
    assert block.capacity == capacity
    block.write(b'ab')
    assert block.tobytes() == b'ab'


def test_block_view_round_trips_through_block():
    first = EventBlock()
    first.write(np.array([7, 8, 9], dtype=np.int32))
    second = EventBlock()
    second.write(first.view())
    assert second.tobytes() == first.tobytes()
