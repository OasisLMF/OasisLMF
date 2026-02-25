import pickle
import socket
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oasislmf.pytools.data_layer.footprint_layer import (
    FootprintLayer,
    FootprintLayerClient,
    OperationEnum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_header(operation: OperationEnum, event_id: int = 0) -> bytes:
    """Build a 16-byte header matching the protocol."""
    return operation.value + event_id.to_bytes(8, byteorder='big') + b'\x00' * 4


# ---------------------------------------------------------------------------
# OperationEnum
# ---------------------------------------------------------------------------

def test_operation_enum_values_are_4_byte_big_endian():
    assert OperationEnum.GET_DATA.value == (1).to_bytes(4, byteorder='big')
    assert OperationEnum.GET_NUM_INTENSITY_BINS.value == (2).to_bytes(4, byteorder='big')
    assert OperationEnum.REGISTER.value == (3).to_bytes(4, byteorder='big')
    assert OperationEnum.UNREGISTER.value == (4).to_bytes(4, byteorder='big')
    assert OperationEnum.POLL_DATA.value == (5).to_bytes(4, byteorder='big')


def test_operation_enum_lookup_from_bytes():
    assert OperationEnum((1).to_bytes(4, byteorder='big')) == OperationEnum.GET_DATA


def test_operation_enum_invalid_value_raises():
    with pytest.raises(ValueError):
        OperationEnum((99).to_bytes(4, byteorder='big'))


# ---------------------------------------------------------------------------
# FootprintLayer._extract_header
# ---------------------------------------------------------------------------

def test_extract_header_get_data_returns_event_id():
    header = _make_header(OperationEnum.GET_DATA, event_id=42)
    operation, event_id = FootprintLayer._extract_header(header)
    assert operation == OperationEnum.GET_DATA
    assert event_id == 42


def test_extract_header_get_data_event_id_zero():
    header = _make_header(OperationEnum.GET_DATA, event_id=0)
    _, event_id = FootprintLayer._extract_header(header)
    assert event_id == 0


def test_extract_header_get_data_large_event_id():
    header = _make_header(OperationEnum.GET_DATA, event_id=2**32 - 1)
    _, event_id = FootprintLayer._extract_header(header)
    assert event_id == 2**32 - 1


@pytest.mark.parametrize("op", [
    OperationEnum.GET_NUM_INTENSITY_BINS,
    OperationEnum.REGISTER,
    OperationEnum.UNREGISTER,
    OperationEnum.POLL_DATA,
])
def test_extract_header_non_get_data_returns_none_event_id(op):
    header = _make_header(op, event_id=99)
    operation, event_id = FootprintLayer._extract_header(header)
    assert operation == op
    assert event_id is None


# ---------------------------------------------------------------------------
# FootprintLayer._stream_footprint_data
# ---------------------------------------------------------------------------

def test_stream_footprint_data_small_array_single_chunk():
    data = np.array([1, 2, 3], dtype=np.int32)
    mock_conn = MagicMock(spec=socket.socket)

    FootprintLayer._stream_footprint_data(data, mock_conn, event_id=1)

    mock_conn.sendall.assert_called_once_with(pickle.dumps(data))


def test_stream_footprint_data_large_array_multiple_chunks():
    # ~80 KB pickled — exceeds the 60 000-byte chunk size
    data = np.zeros(10000, dtype=np.float64)
    mock_conn = MagicMock(spec=socket.socket)

    FootprintLayer._stream_footprint_data(data, mock_conn, event_id=1)

    raw = pickle.dumps(data)
    expected_chunks = [raw[i:i + 60000] for i in range(0, len(raw), 60000)]
    assert mock_conn.sendall.call_count == len(expected_chunks)
    actual_calls = [c.args[0] for c in mock_conn.sendall.call_args_list]
    assert actual_calls == expected_chunks


def test_stream_footprint_data_reassembled_matches_original():
    data = np.array([10, 20, 30], dtype=np.float32)
    chunks = []
    mock_conn = MagicMock(spec=socket.socket)
    mock_conn.sendall.side_effect = lambda chunk: chunks.append(chunk)

    FootprintLayer._stream_footprint_data(data, mock_conn, event_id=5)

    np.testing.assert_array_equal(pickle.loads(b"".join(chunks)), data)


# ---------------------------------------------------------------------------
# FootprintLayer.__init__ / _define_socket
# ---------------------------------------------------------------------------

@patch('oasislmf.pytools.data_layer.footprint_layer.socket.socket')
def test_footprint_layer_init_sets_attributes(mock_socket_cls):
    mock_socket_cls.return_value = MagicMock()
    storage = MagicMock()

    layer = FootprintLayer(storage=storage, total_expected=5)

    assert layer.storage is storage
    assert layer.total_expected == 5
    assert layer.count == 0
    assert layer.total_served == 0
    assert layer.file_data is None


@patch('oasislmf.pytools.data_layer.footprint_layer.socket.socket')
def test_footprint_layer_init_binds_and_listens(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock

    FootprintLayer(storage=MagicMock(), total_expected=1)

    mock_sock.bind.assert_called_once_with(('127.0.0.1', 8080))
    mock_sock.listen.assert_called_once_with(100)


@patch('oasislmf.pytools.data_layer.footprint_layer.socket.socket')
def test_footprint_layer_init_sets_so_reuseaddr(mock_socket_cls):
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock

    FootprintLayer(storage=MagicMock(), total_expected=1)

    mock_sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


# ---------------------------------------------------------------------------
# FootprintLayerClient.poll
# ---------------------------------------------------------------------------

def test_client_poll_returns_true_when_server_reachable():
    with patch.object(FootprintLayerClient, '_get_socket', return_value=MagicMock()):
        assert FootprintLayerClient.poll() is True


def test_client_poll_returns_false_on_connection_refused():
    with patch.object(FootprintLayerClient, '_get_socket', side_effect=ConnectionRefusedError):
        assert FootprintLayerClient.poll() is False


# ---------------------------------------------------------------------------
# FootprintLayerClient.get_number_of_intensity_bins
# ---------------------------------------------------------------------------

def test_client_get_number_of_intensity_bins_sends_correct_operation():
    mock_sock = MagicMock()
    mock_sock.recv.return_value = (100).to_bytes(8, byteorder='big')

    with patch.object(FootprintLayerClient, '_get_socket', return_value=mock_sock):
        result = FootprintLayerClient.get_number_of_intensity_bins()

    mock_sock.sendall.assert_called_once_with(OperationEnum.GET_NUM_INTENSITY_BINS.value)
    assert result == 100
    mock_sock.close.assert_called_once()


# ---------------------------------------------------------------------------
# FootprintLayerClient.get_event
# ---------------------------------------------------------------------------

def test_client_get_event_sends_correct_header():
    data = np.array([1.0, 2.0], dtype=np.float32)
    raw = pickle.dumps(data)
    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [raw, b'']

    with patch.object(FootprintLayerClient, '_get_socket', return_value=mock_sock):
        FootprintLayerClient.get_event(event_id=7)

    expected = OperationEnum.GET_DATA.value + (7).to_bytes(8, byteorder='big')
    mock_sock.sendall.assert_called_once_with(expected)


def test_client_get_event_reassembles_chunked_response():
    data = np.array([10, 20, 30], dtype=np.int64)
    raw = pickle.dumps(data)
    mid = len(raw) // 2
    mock_sock = MagicMock()
    mock_sock.recv.side_effect = [raw[:mid], raw[mid:], b'']

    with patch.object(FootprintLayerClient, '_get_socket', return_value=mock_sock):
        result = FootprintLayerClient.get_event(event_id=1)

    np.testing.assert_array_equal(result, data)


def test_client_get_event_returns_none_on_empty_response():
    mock_sock = MagicMock()
    mock_sock.recv.return_value = b''

    with patch.object(FootprintLayerClient, '_get_socket', return_value=mock_sock):
        result = FootprintLayerClient.get_event(event_id=1)

    assert result is None
