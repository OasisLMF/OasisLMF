import json
import os
import threading
import time
from unittest.mock import patch, MagicMock
import pytest
import requests
import oasislmf.utils.ping as ping_module
from oasislmf.utils.ping import (oasis_ping, oasis_ping_socket, oasis_ping_websocket, oasis_ping_http, oasis_ping_async,
                                 oasis_ping_progress, wait_for_async_ping, EventProgress)


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


def test_oasis_ping_http_path():
    data = {"analysis_pk": 123}
    with (patch.dict(os.environ, {"OASIS_ANALYSIS_STATUS_URL": "http://fakehost/analysis-status/"}, clear=True),
          patch("oasislmf.utils.ping.oasis_ping_http", return_value=True) as mock_http,
          patch("oasislmf.utils.ping.oasis_ping_websocket") as mock_ws,
          patch("oasislmf.utils.ping.oasis_ping_socket") as mock_sock):
        result = oasis_ping(data)

    mock_http.assert_called_once_with("http://fakehost/analysis-status/", data)
    mock_ws.assert_not_called()
    mock_sock.assert_not_called()
    assert result is True


def test_oasis_ping_http_path_falls_back_to_websocket():
    data = {"analysis_pk": 123}
    with (patch.dict(os.environ, {
        "OASIS_ANALYSIS_STATUS_URL": "http://fakehost/analysis-status/",
        "OASIS_WEBSOCKET_URL": "ws://fakehost",
        "OASIS_WEBSOCKET_PORT": "9999",
    }, clear=True),
            patch("oasislmf.utils.ping.oasis_ping_http", return_value=False) as mock_http,
            patch("oasislmf.utils.ping.oasis_ping_websocket", return_value=True) as mock_ws,
            patch("oasislmf.utils.ping.oasis_ping_socket") as mock_sock):
        result = oasis_ping(data)

    mock_http.assert_called_once()
    mock_ws.assert_called_once()
    mock_sock.assert_not_called()
    assert result is True


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


def test_oasis_ping_socket_unreachable_host():
    fake_sock = MagicMock()
    fake_sock.__enter__.return_value = fake_sock
    fake_sock.connect.side_effect = OSError(113, "No route to host")

    with patch("socket.socket", return_value=fake_sock):
        result = oasis_ping_socket(("127.0.0.1", 9999), '{"hello": "world"}')

    assert result is False


def test_oasis_ping_websocket_success():
    fake_ws = MagicMock()
    with patch("websocket.WebSocket", return_value=fake_ws):
        result = oasis_ping_websocket("ws://fakehost:1234/ws", '{"hello": "world"}')
    assert result is True
    fake_ws.connect.assert_called_once_with("ws://fakehost:1234/ws", timeout=1)
    fake_ws.send.assert_called_once_with('{"hello": "world"}')
    fake_ws.close.assert_called_once()


def test_oasis_ping_websocket_failure():
    fake_ws = MagicMock()
    fake_ws.connect.side_effect = Exception("boom")
    with patch("websocket.WebSocket", return_value=fake_ws):
        result = oasis_ping_websocket("ws://fakehost:1234/ws", '{"hello": "world"}')
    assert result is False
    fake_ws.send.assert_not_called()


def test_oasis_ping_http_success():
    fake_response = MagicMock()
    with patch("requests.post", return_value=fake_response) as mock_post:
        result = oasis_ping_http("http://fakehost/analysis-status/", {"hello": "world"})

    assert result is True
    mock_post.assert_called_once_with("http://fakehost/analysis-status/", json={"hello": "world"}, timeout=1)
    fake_response.raise_for_status.assert_called_once()


def test_oasis_ping_http_failure():
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("boom")):
        result = oasis_ping_http("http://fakehost/analysis-status/", {"hello": "world"})

    assert result is False


def test_oasis_ping_http_bad_status():
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
    with patch("requests.post", return_value=fake_response):
        result = oasis_ping_http("http://fakehost/analysis-status/", {"hello": "world"})

    assert result is False


def test_oasis_ping_port_override():
    data = {"events_complete": 5, "port_override": 9999}
    with (patch.dict(os.environ, {"OASIS_SOCKET_SERVER_IP": "1.2.3.4", "OASIS_SOCKET_SERVER_PORT": "4321"}),
          patch("oasislmf.utils.ping.oasis_ping_socket", return_value=True) as mock_sock):
        oasis_ping(data)

    called_target, called_msg = mock_sock.call_args[0]
    assert called_target == ("1.2.3.4", 9999)
    parsed = json.loads(called_msg)
    assert parsed["events_complete"] == 5


def test_oasis_ping_no_port_override_uses_env():
    data = {"events_complete": 3}
    with (patch.dict(os.environ, {"OASIS_SOCKET_SERVER_IP": "1.2.3.4", "OASIS_SOCKET_SERVER_PORT": "4321"}),
          patch("oasislmf.utils.ping.oasis_ping_socket", return_value=True) as mock_sock):
        oasis_ping(data)

    called_target, _ = mock_sock.call_args[0]
    assert called_target == ("1.2.3.4", 4321)


def test_oasis_ping_async_calls_oasis_ping_off_caller_thread():
    data = {"hello": "world"}
    caller_thread = threading.current_thread()
    done = threading.Event()
    calls = []

    def fake_oasis_ping(d):
        calls.append((d, threading.current_thread()))
        done.set()

    with patch("oasislmf.utils.ping.oasis_ping", side_effect=fake_oasis_ping):
        oasis_ping_async(data)
        assert done.wait(timeout=2), "oasis_ping_async did not call oasis_ping in time"

    assert calls == [(data, calls[0][1])]
    assert calls[0][1] is not caller_thread


def test_oasis_ping_async_drops_update_when_previous_still_in_flight():
    started = threading.Event()
    release = threading.Event()

    def slow_oasis_ping(d):
        started.set()
        release.wait(timeout=2)

    with patch("oasislmf.utils.ping.oasis_ping", side_effect=slow_oasis_ping) as mock_ping:
        assert oasis_ping_async({"events_complete": 1}) is True
        assert started.wait(timeout=2), "first ping never started"

        # A second update arriving while the first is still in flight is dropped,
        # not queued behind it, and the caller is told so.
        assert oasis_ping_async({"events_complete": 2}) is False

        release.set()
        for _ in range(200):
            if not ping_module._ping_lock.locked():
                break
            time.sleep(0.01)

    mock_ping.assert_called_once_with({"events_complete": 1})


@pytest.mark.parametrize("socket_server,expected", [
    ('8888', {"outputs_complete": 1, "analysis_pk": None, "port_override": 8888}),
    ('True', {"outputs_complete": 1, "analysis_pk": 3}),
])
def test_oasis_ping_progress_sends(socket_server, expected):
    analysis_pk = expected["analysis_pk"]
    with patch("oasislmf.utils.ping.oasis_ping", return_value=True) as mock_ping:
        assert oasis_ping_progress(socket_server, analysis_pk, outputs_complete=1) is True
    mock_ping.assert_called_once_with(expected)


@pytest.mark.parametrize("socket_server", ['False', None])
def test_oasis_ping_progress_disabled(socket_server):
    with patch("oasislmf.utils.ping.oasis_ping") as mock_ping:
        assert oasis_ping_progress(socket_server, None, outputs_complete=1) is False
    mock_ping.assert_not_called()


@pytest.mark.parametrize("sent,failures", [(True, 0), (False, 1)])
def test_oasis_ping_async_calls_on_failure_only_when_not_sent(sent, failures):
    failed = []
    with patch("oasislmf.utils.ping.oasis_ping", return_value=sent):
        assert oasis_ping_async({"events_complete": 1}, on_failure=lambda: failed.append(1)) is True
        assert wait_for_async_ping(2)
    assert len(failed) == failures


def test_oasis_ping_async_calls_on_failure_when_ping_raises():
    failed = []
    with patch("oasislmf.utils.ping.oasis_ping", side_effect=RuntimeError("boom")):
        oasis_ping_async({"events_complete": 1}, on_failure=lambda: failed.append(1))
        assert wait_for_async_ping(2)
    assert failed == [1]


def test_event_progress_disabled_sends_nothing():
    with (patch("oasislmf.utils.ping.oasis_ping") as mock_ping,
          patch("oasislmf.utils.ping.oasis_ping_async") as mock_async):
        progress = EventProgress('False')
        progress.ready()
        progress.event_done()
        progress.finish()
    mock_ping.assert_not_called()
    mock_async.assert_not_called()


def test_event_progress_keeps_count_of_failed_ping():
    """A periodic ping that was dispatched but did not get through is reported by `finish`."""
    sent = []

    def fake_oasis_ping(data):
        sent.append(data["events_complete"])
        return len(sent) > 1

    with (patch("oasislmf.utils.ping.oasis_ping", side_effect=fake_oasis_ping),
          patch("oasislmf.utils.ping.SERVER_UPDATE_TIME", -1)):
        progress = EventProgress('8888')
        progress.event_done()
        assert wait_for_async_ping(2)
        progress.finish()

    assert sent == [1, 1]


def test_event_progress_finish_waits_for_ping_in_flight():
    """`finish` sends the remaining count only after the in-flight ping, so no count is sent twice or lost."""
    release = threading.Event()
    sent = []

    def slow_oasis_ping(data):
        if not sent:
            release.wait(timeout=2)
        sent.append(data["events_complete"])
        return True

    with (patch("oasislmf.utils.ping.oasis_ping", side_effect=slow_oasis_ping),
          patch("oasislmf.utils.ping.SERVER_UPDATE_TIME", -1)):
        progress = EventProgress('8888')
        progress.event_done()
        progress.event_done()
        threading.Timer(0.2, release.set).start()
        progress.finish()

    assert sent == [1, 1]
