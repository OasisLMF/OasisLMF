import json
import socket
import threading
import time
import sys
import pytest
from unittest.mock import MagicMock, patch

from oasislmf.utils.ping import oasis_ping_socket
from oasislmf.utils.socket_server import GulProgressServer, gul_description
from oasislmf.utils.socket_server import main as server_main


def get_free_port():
    """Grab an available TCP port from the OS."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return port


def test_server_increments_counter():
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port)
    server.start()

    assert server.counter == 0
    target = (server.host, server.port)
    data = json.dumps({"events_complete": 5})
    assert oasis_ping_socket(target, data) is True
    time.sleep(0.1)
    assert server.counter == 5

    server.stop()


def test_server_handles_multiple_clients():
    port = get_free_port()
    server = GulProgressServer(20, host="127.0.0.1", port=port)
    server.start()
    target = (server.host, server.port)

    for i in [3, 7, 10]:
        data = json.dumps({"events_complete": i})
        assert oasis_ping_socket(target, data) is True
    time.sleep(0.1)
    assert server.counter == 20

    server.stop()


def test_server_handles_missing_key():
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port)
    server.start()
    target = (server.host, server.port)
    data = json.dumps({"hello": "world"})

    initial = server.counter
    assert oasis_ping_socket(target, data) is True
    time.sleep(0.1)
    assert server.counter == initial

    server.stop()


def test_server_handles_unknown_fields_logs_warning(caplog):
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port)
    server.start()
    target = (server.host, server.port)
    payload = {"unknown_field": "some_value"}

    with caplog.at_level("WARNING", logger="root"):
        assert oasis_ping_socket(target, json.dumps(payload)) is True
        time.sleep(0.1)

    assert server.counter == 0
    assert any("Json received with no valid fields" in r.message for r in caplog.records)

    server.stop()


def test_server_handles_invalid_json(caplog):
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port)
    server.start()
    target = (server.host, server.port)

    with caplog.at_level("WARNING", logger="root"):
        assert oasis_ping_socket(target, "not valid json {{{") is True
        time.sleep(0.1)

    assert server.counter == 0
    assert any("Invalid json received" in r.message for r in caplog.records)

    server.stop()


def test_server_can_stop_and_not_accept():
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port)
    server.start()
    target = (server.host, server.port)
    data = json.dumps({"events_complete": 1})
    server.stop()
    for i in range(10):
        time.sleep(1)
        if not oasis_ping_socket(target, data):
            return True
    assert False


@pytest.mark.parametrize("argv", [['a'], ['a', 'b'], ['a', '10', 'c']])
def test_main_invalid_call(argv):
    sys.argv = argv
    with pytest.raises(SystemExit):
        server_main()


def test_main_finishes_valid_call():
    with (patch('time.sleep', side_effect=[None, StopIteration]),
          patch('oasislmf.utils.socket_server.GulProgressServer') as fake_server_class):
        mock_server = MagicMock()
        fake_server_class.return_value.__enter__.return_value = mock_server
        mock_server.counter = 10
        sys.argv = ['a', '10']
        server_main()


def test_server_terminate():
    port = get_free_port()
    server = GulProgressServer(91, host="127.0.0.1", port=port)
    server.start()
    with patch.object(server, "stop", wraps=server.stop) as mock_stop:
        assert oasis_ping_socket(("127.0.0.1", port), json.dumps({"terminate": "terminate"})) is True
        time.sleep(0.1)
        assert server.counter == server.total
        mock_stop.assert_called_once()


def test_main_with_explicit_port():
    port = get_free_port()
    with (patch('time.sleep', side_effect=[None, StopIteration]),
          patch('oasislmf.utils.socket_server.GulProgressServer') as fake_server_class):
        mock_server = MagicMock()
        fake_server_class.return_value.__enter__.return_value = mock_server
        mock_server.counter = 5
        sys.argv = ['a', '5', str(port)]
        server_main()

    fake_server_class.assert_called_once_with(5, port=port, gul_processes=0, outputs_total=0)


def test_main_passes_gul_processes():
    with (patch('time.sleep', side_effect=[None, StopIteration]),
          patch('oasislmf.utils.socket_server.GulProgressServer') as fake_server_class):
        mock_server = MagicMock()
        fake_server_class.return_value.__enter__.return_value = mock_server
        mock_server.counter = 5
        sys.argv = ['a', '5', '--gul-processes', '4']
        server_main()

    fake_server_class.assert_called_once_with(5, port=None, gul_processes=4, outputs_total=0)


def test_server_counts_gul_ready():
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port, gul_processes=2)
    server.start()
    target = (server.host, server.port)
    assert oasis_ping_socket(target, json.dumps({"gul_ready": 1})) is True
    time.sleep(0.1)
    assert server.gul_ready == 1
    assert server.counter == 0

    server.stop()


@pytest.mark.parametrize("counter,gul_ready,gul_processes,expected", [
    (0, 0, 4, "Gul starting (0/4 ready)"),
    (0, 3, 4, "Gul starting (3/4 ready)"),
    (0, 4, 4, "Gul events completed"),
    (2, 1, 4, "Gul events completed"),   # events arriving means GUL is past start-up
    (0, 0, 0, "Gul events completed"),   # no GUL process reports gul_ready
])
def test_gul_description(counter, gul_ready, gul_processes, expected):
    server = GulProgressServer(10, host="127.0.0.1", port=0, gul_processes=gul_processes)
    server.counter = counter
    server.gul_ready = gul_ready
    assert gul_description(server) == expected


def test_server_counts_outputs_complete():
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port, outputs_total=3)
    server.start()
    target = (server.host, server.port)
    for _ in range(2):
        assert oasis_ping_socket(target, json.dumps({"outputs_complete": 1})) is True
    time.sleep(0.1)
    assert server.outputs_complete == 2
    assert server.counter == 0

    server.stop()


def test_main_waits_for_outputs_after_events():
    """With --outputs, main keeps running after the GUL events until every output process reports."""
    port = get_free_port()
    sys.argv = ['a', '2', str(port), '--outputs', '2']
    server_thread = threading.Thread(target=server_main, daemon=True)
    server_thread.start()
    target = ("127.0.0.1", port)
    for _ in range(50):
        if oasis_ping_socket(target, json.dumps({"events_complete": 2})):
            break
        time.sleep(0.1)

    time.sleep(2)
    assert server_thread.is_alive()
    for _ in range(2):
        assert oasis_ping_socket(target, json.dumps({"outputs_complete": 1})) is True
    server_thread.join(timeout=5)
    assert not server_thread.is_alive()
