import json
import socket
import time
import sys
import pytest
from unittest.mock import MagicMock, patch

from oasislmf.utils.ping import oasis_ping_socket
from oasislmf.utils.socket_server import GulProgressServer
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


def test_main_invalid_call():
    sys.argv = []
    with pytest.raises(ValueError):
        server_main()
    sys.argv = ['a']
    with pytest.raises(ValueError):
        server_main()
    sys.argv = ['a', 'b']
    with pytest.raises(TypeError):
        server_main()


def test_main_finishes_valid_call():
    with (patch('time.sleep', side_effect=[None, StopIteration]),
          patch('oasislmf.utils.socket_server.GulProgressServer') as fake_server_class):
        mock_server = MagicMock()
        fake_server_class.return_value.__enter__.return_value = mock_server
        mock_server.counter = 10
        sys.argv = ['a', 10]
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
