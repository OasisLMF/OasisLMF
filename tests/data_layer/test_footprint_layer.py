import pickle
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oasis_data_manager.filestore.backends.local import LocalStorage

from oasislmf.pytools.data_layer.footprint_layer import (
    FootprintLayer,
    FootprintLayerClient,
    OperationEnum,
    main,
    portfolio_areaperil_ids,
)
from oasislmf.pytools.getmodel.footprint import Footprint

TEST_MODEL_1 = Path(__file__).parent.parent / 'assets' / 'test_model_1'


def _make_header(operation: OperationEnum, event_id: int = 0) -> bytes:
    """Build a 16-byte header matching the protocol."""
    return operation.value + event_id.to_bytes(8, byteorder='big') + b'\x00' * 4


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


def test_client_poll_returns_true_when_server_reachable():
    with patch.object(FootprintLayerClient, '_get_socket', return_value=MagicMock()):
        assert FootprintLayerClient.poll() is True


def test_client_poll_returns_false_on_connection_refused():
    with patch.object(FootprintLayerClient, '_get_socket', side_effect=ConnectionRefusedError):
        assert FootprintLayerClient.poll() is False


def test_client_get_number_of_intensity_bins_sends_correct_operation():
    mock_sock = MagicMock()
    mock_sock.recv.return_value = (100).to_bytes(8, byteorder='big')

    with patch.object(FootprintLayerClient, '_get_socket', return_value=mock_sock):
        result = FootprintLayerClient.get_number_of_intensity_bins()

    mock_sock.sendall.assert_called_once_with(OperationEnum.GET_NUM_INTENSITY_BINS.value)
    assert result == 100
    mock_sock.close.assert_called_once()


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


# --- issue #2220: servedata with footprints other than csv -------------------

@pytest.mark.parametrize('items_file', ['items.csv', 'items.bin'])
def test_portfolio_areaperil_ids_read_from_items(tmp_path, items_file):
    shutil.copy(TEST_MODEL_1 / 'input' / items_file, tmp_path / items_file)
    expected = np.unique(np.loadtxt(TEST_MODEL_1 / 'input' / 'items.csv', delimiter=',', skiprows=1,
                                    usecols=2, dtype=np.int64))

    np.testing.assert_array_equal(portfolio_areaperil_ids(str(tmp_path)), expected)


def test_portfolio_areaperil_ids_none_without_items(tmp_path):
    assert portfolio_areaperil_ids(str(tmp_path)) is None
    assert portfolio_areaperil_ids(str(tmp_path / 'missing')) is None


@patch('oasislmf.pytools.data_layer.footprint_layer.socket.socket')
def test_get_event_data_needs_no_footprint_index(mock_socket_cls):
    """Parquet and dynamic footprints have no footprint_index; serving an event used to fail on it."""
    event = np.array([1, 2, 3])
    layer = FootprintLayer(storage=MagicMock(), total_expected=1)
    layer.file_data = SimpleNamespace(get_event=lambda event_id: event if event_id == 4 else None)

    assert layer.get_event_data(4) is event
    assert layer.get_event_data(5) is None


@pytest.mark.parametrize('ignore_file_type', [{'csv'}, {'bin', 'idx'}], ids=['bin', 'csv'])
@patch.object(FootprintLayer, '_define_socket')
def test_get_event_data_serves_an_event_more_than_once(mock_define_socket, ignore_file_type):
    """The csv footprint's index entry was deleted after the first request, so a repeat got nothing."""
    storage = LocalStorage(root_dir=str(TEST_MODEL_1 / 'static'))
    layer = FootprintLayer(storage=storage, total_expected=1)
    with Footprint.load(storage, ignore_file_type=ignore_file_type) as footprint:
        layer.file_data = footprint
        event_id = 1
        first = layer.get_event_data(event_id)
        second = layer.get_event_data(event_id)

    assert first is not None and len(first) > 0
    np.testing.assert_array_equal(first, second)


@patch('oasislmf.pytools.data_layer.footprint_layer.Footprint.load')
@patch('oasislmf.pytools.data_layer.footprint_layer.socket.socket')
def test_listen_loads_footprint_for_the_portfolio_areaperils(mock_socket_cls, mock_load):
    """A dynamic footprint needs the areaperils; without them it read input/keys.csv, absent for parquet keys."""
    connection = MagicMock()
    connection.recv.return_value = _make_header(OperationEnum.UNREGISTER)
    mock_socket_cls.return_value.accept.return_value = (connection, ('127.0.0.1', 1))
    areaperil_ids = np.array([3, 7])

    FootprintLayer(storage=MagicMock(), total_expected=0, areaperil_ids=areaperil_ids).listen()

    assert mock_load.call_args.kwargs['areaperil_ids'] is areaperil_ids


@patch.object(FootprintLayer, '_stream_footprint_data')
@patch.object(FootprintLayer, '_define_socket')
def test_listen_serves_a_get_data_request(mock_define_socket, mock_stream):
    """A GET_DATA request is answered from the loaded footprint, then UNREGISTER stops the server."""
    storage = LocalStorage(root_dir=str(TEST_MODEL_1 / 'static'))
    layer = FootprintLayer(storage=storage, total_expected=0, ignore_file_type={'csv'})
    get_data, unregister = MagicMock(), MagicMock()
    get_data.recv.return_value = _make_header(OperationEnum.GET_DATA, event_id=1)
    unregister.recv.return_value = _make_header(OperationEnum.UNREGISTER)
    layer.socket = MagicMock()
    layer.socket.accept.side_effect = [(get_data, ('127.0.0.1', 1)), (unregister, ('127.0.0.1', 2))]

    layer.listen()

    with Footprint.load(storage, ignore_file_type={'csv'}) as footprint:
        expected = footprint.get_event(1)
    assert mock_stream.call_count == 1
    np.testing.assert_array_equal(mock_stream.call_args.kwargs['event_data'], expected)
    assert mock_stream.call_args.kwargs['connection'] is get_data


@patch.object(FootprintLayer, 'listen')
@patch.object(FootprintLayer, '_define_socket')
def test_main_filters_the_footprint_to_the_input_items(mock_define_socket, mock_listen, tmp_path, monkeypatch):
    shutil.copy(TEST_MODEL_1 / 'input' / 'items.csv', tmp_path / 'items.csv')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('sys.argv', ['servedata', str(TEST_MODEL_1 / 'static'), '2', '--input-path', str(tmp_path)])
    created = []
    monkeypatch.setattr(FootprintLayer, '__init__', _recording_init(FootprintLayer.__init__, created))

    main()

    assert mock_listen.call_count == 1
    layer, = created
    assert layer.total_expected == 2
    np.testing.assert_array_equal(layer.areaperil_ids, portfolio_areaperil_ids(str(tmp_path)))


def _recording_init(init, created):
    """Wrap ``FootprintLayer.__init__`` so a test can inspect the server ``main`` builds."""

    def recording_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        created.append(self)
    return recording_init
