import json
import os
from unittest.mock import patch, MagicMock
from oasislmf.utils.ping import oasis_ping, oasis_ping_socket, oasis_ping_websocket


def test_oasis_ping_websocket_path():
    data = {"analysis_pk": 123}
    with (patch.dict(os.environ, {"OASIS_WEBSOCKET_URL": "ws://fakehost", "OASIS_WEBSOCKET_PORT": "9999"}),
          patch("oasislmf.utils.ping.oasis_ping_websocket", return_value=True) as mock_ws,
          patch("oasislmf.utils.ping.oasis_ping_socket") as mock_sock):
        result = oasis_ping(data)

    mock_ws.assert_called_once()
    mock_sock.assert_not_called()
    assert result is True

    called_url, called_msg = mock_ws.call_args[0]
    assert called_url == "ws://fakehost:9999/ws/analysis-status/"
    assert json.loads(called_msg) == data


def test_oasis_ping_analysis_pk_missing_env():
    data = {"analysis_pk": 123}
    with (patch.dict(os.environ, {}, clear=True),
          patch("oasislmf.utils.ping.oasis_ping_websocket") as mock_ws,
          patch("oasislmf.utils.ping.oasis_ping_socket") as mock_sock):
        result = oasis_ping(data)

    assert result is False
    mock_ws.assert_not_called()
    mock_sock.assert_not_called()


def test_oasis_ping_socket_path():
    data = {"hello": "world"}
    with (patch.dict(os.environ, {"OASIS_SOCKET_SERVER_IP": "1.2.3.4", "OASIS_SOCKET_SERVER_PORT": "4321"}),
          patch("oasislmf.utils.ping.oasis_ping_websocket") as mock_ws,
          patch("oasislmf.utils.ping.oasis_ping_socket", return_value=True) as mock_sock):
        result = oasis_ping(data)

    mock_ws.assert_not_called()
    mock_sock.assert_called_once()
    assert result is True

    called_target, called_msg = mock_sock.call_args[0]
    assert called_target == ("1.2.3.4", 4321)
    assert json.loads(called_msg) == data


def test_oasis_ping_socket_success():
    fake_sock = MagicMock()
    fake_sock.__enter__.return_value = fake_sock

    with patch("socket.socket", return_value=fake_sock):
        result = oasis_ping_socket(("127.0.0.1", 9999), '{"hello": "world"}')

    assert result is True
    fake_sock.connect.assert_called_once_with(("127.0.0.1", 9999))
    fake_sock.sendall.assert_called_once_with(b'{"hello": "world"}')


def test_oasis_ping_socket_connection_refused():
    fake_sock = MagicMock()
    fake_sock.__enter__.return_value = fake_sock
    fake_sock.connect.side_effect = ConnectionRefusedError

    with patch("socket.socket", return_value=fake_sock):
        result = oasis_ping_socket(("127.0.0.1", 9999), '{"hello": "world"}')

    assert result is False
    fake_sock.sendall.assert_not_called()


def test_oasis_ping_websocket_success():
    fake_ws = MagicMock()
    with patch("websocket.WebSocket", return_value=fake_ws):
        result = oasis_ping_websocket("ws://fakehost:1234/ws", '{"hello": "world"}')
    assert result is True
    fake_ws.connect.assert_called_once_with("ws://fakehost:1234/ws")
    fake_ws.send.assert_called_once_with('{"hello": "world"}')
    fake_ws.close.assert_called_once()


def test_oasis_ping_websocket_failure():
    fake_ws = MagicMock()
    fake_ws.connect.side_effect = Exception("boom")
    with patch("websocket.WebSocket", return_value=fake_ws):
        result = oasis_ping_websocket("ws://fakehost:1234/ws", '{"hello": "world"}')
    assert result is False
    fake_ws.send.assert_not_called()
