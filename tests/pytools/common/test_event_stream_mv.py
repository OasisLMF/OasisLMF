"""Contract of the event-stream scalar codec (mv_write / mv_read) as jitted callers see it.

Streams mix field widths (loss is f4 or f8 depending on OASIS_FLOAT), so a field can start at an
offset that is not a multiple of its own size. The end-to-end suites run with the default f4 and
never hit that case, so it is pinned here: whatever the dtype mix and offset, the bytes written must
be the ones numpy would produce, and must read back unchanged.
"""
import numba as nb
import numpy as np
import pytest
from numba.core import config as nb_config

from oasislmf.pytools.common.event_stream import mv_read, mv_write

ID_DTYPES = ['i2', 'i4', 'u4', 'i8', 'u8']
LOSS_DTYPES = ['f4', 'f8']


@nb.njit
def _write_records(byte_mv, start, ids, losses, id_dtype, id_size, loss_dtype, loss_size):
    cursor = start
    for k in range(ids.shape[0]):
        cursor = mv_write(byte_mv, cursor, id_dtype, id_size, ids[k])
        cursor = mv_write(byte_mv, cursor, loss_dtype, loss_size, losses[k])
    return cursor


@nb.njit
def _read_records(byte_mv, start, n, id_dtype, id_size, loss_dtype, loss_size, ids_out, losses_out):
    cursor = start
    for k in range(n):
        ids_out[k], cursor = mv_read(byte_mv, cursor, id_dtype, id_size)
        losses_out[k], cursor = mv_read(byte_mv, cursor, loss_dtype, loss_size)
    return cursor


@pytest.mark.parametrize('loss_dtype', LOSS_DTYPES)
@pytest.mark.parametrize('id_dtype', ID_DTYPES)
@pytest.mark.parametrize('start', [0, 1, 3])
def test_mixed_width_records_round_trip_at_any_offset(id_dtype, loss_dtype, start):
    id_dtype, loss_dtype = np.dtype(id_dtype), np.dtype(loss_dtype)
    rng = np.random.default_rng(0)
    n = 50
    ids = rng.integers(0, 30000, n).astype(id_dtype)
    losses = rng.random(n).astype(loss_dtype)
    record = np.dtype([('id', id_dtype), ('loss', loss_dtype)])  # packed, like the stream
    byte_mv = np.zeros(start + n * record.itemsize + 7, dtype='b')

    end = _write_records(byte_mv, start, ids, losses, id_dtype, id_dtype.itemsize, loss_dtype, loss_dtype.itemsize)

    expected = np.empty(n, dtype=record)
    expected['id'], expected['loss'] = ids, losses
    assert end == start + n * record.itemsize
    assert byte_mv[start:end].tobytes() == expected.tobytes()
    assert not byte_mv[:start].any() and not byte_mv[end:].any()  # nothing written outside the records

    ids_out, losses_out = np.zeros(n, id_dtype), np.zeros(n, loss_dtype)
    assert _read_records(byte_mv, start, n, id_dtype, id_dtype.itemsize, loss_dtype, loss_dtype.itemsize,
                         ids_out, losses_out) == end
    np.testing.assert_array_equal(ids_out, ids)
    np.testing.assert_array_equal(losses_out, losses)


@nb.njit
def _write_scalar_class_and_float64(byte_mv):
    # np.int32 (a scalar class, not a dtype instance) and a float64 value into an f4 field
    cursor = mv_write(byte_mv, 0, np.int32, 4, 7)
    return mv_write(byte_mv, cursor, np.dtype('f4'), 4, 0.1)


def test_value_converted_to_field_dtype():
    byte_mv = np.zeros(8, dtype='b')
    assert _write_scalar_class_and_float64(byte_mv) == 8
    assert byte_mv.tobytes() == np.int32(7).tobytes() + np.float32(0.1).tobytes()


@nb.njit
def _write_0d(byte_mv, value_0d):
    return mv_write(byte_mv, 0, np.dtype('f4'), 4, value_0d)


def test_zero_dim_array_value_is_written_as_its_element():
    # no pytools caller does this, but main accepted it
    byte_mv = np.zeros(4, dtype='b')
    assert _write_0d(byte_mv, np.array(2.5)) == 4
    assert byte_mv.tobytes() == np.float32(2.5).tobytes()


@nb.njit
def _read_one(byte_mv):
    return mv_read(byte_mv, 0, np.dtype('i4'), 4)


def test_read_from_read_only_buffer():
    byte_mv = np.frombuffer(np.int32(42).tobytes(), dtype='b')
    assert _read_one(byte_mv) == (42, 4)


@nb.njit
def _write_one(byte_mv, cursor):
    return mv_write(byte_mv, cursor, np.dtype('f8'), 8, 1.0)


# Under JIT the overrun check exists only with NUMBA_BOUNDSCHECK (as for any numba indexing); the
# pure-Python path (NUMBA_DISABLE_JIT) always raises.
@pytest.mark.skipif(not (nb_config.BOUNDSCHECK or nb_config.DISABLE_JIT),
                    reason="overrun is only checked under NUMBA_BOUNDSCHECK or NUMBA_DISABLE_JIT")
@pytest.mark.parametrize('call', [
    lambda: _write_one(np.zeros(10, dtype='b'), 3),  # last byte would land past the end
    lambda: _write_one(np.zeros(10, dtype='b'), -1),
    lambda: _read_one(np.zeros(3, dtype='b')),
])
def test_overrunning_the_buffer_raises(call):
    with pytest.raises((ValueError, IndexError)):
        call()


@nb.njit
def _write_any(byte_mv, _dtype, value):
    return mv_write(byte_mv, 0, _dtype, 4, value)


@nb.njit
def _read_any(byte_mv, _dtype):
    return mv_read(byte_mv, 0, _dtype, 4)


# Under JIT there is no slow fallback: anything but a scalar (or 0-d array) into a C-contiguous byte
# buffer is a compile error rather than a silently slower code path.
@pytest.mark.skipif(nb_config.DISABLE_JIT, reason="type checks happen when numba compiles the caller")
@pytest.mark.parametrize('call', [
    lambda: _write_any(np.zeros(8, dtype='b'), np.dtype('f4'), np.zeros(1, dtype='f4')),  # 1-d array value, even of size 1
    lambda: _write_any(np.zeros(8, dtype='i4'), np.dtype('f4'), 1.0),  # not a byte buffer
    lambda: _write_any(np.zeros(8, dtype='b')[::2], np.dtype('f4'), 1.0),  # not C-contiguous
    lambda: _write_any(np.zeros(8, dtype=np.bool_), np.dtype('f4'), 1.0),  # bool buffer
    lambda: _write_any(np.zeros(8, dtype='b'), np.dtype([('a', 'i4')]), 1),  # structured dtype
    lambda: _read_any(np.zeros(8, dtype='i4'), np.dtype('f4')),
])
def test_unsupported_types_fail_to_compile(call):
    with pytest.raises(nb.core.errors.TypingError):
        call()
