import json
import websocket
import socket
import os
import logging
import requests
import threading
import time
from oasislmf.utils.defaults import SERVER_DEFAULT_PORT, SERVER_DEFAULT_IP, SERVER_UPDATE_TIME

logger = logging.getLogger(__name__)

# Guards `oasis_ping_async` so a stuck ping target can never pile up more than
# one in-flight ping thread within this process.
_ping_lock = threading.Lock()

# How long a process waits at exit for a ping still being sent by `oasis_ping_async`.
ASYNC_PING_EXIT_WAIT = 3


def platform_progress_configured():
    """Whether the environment names a platform target for progress updates.

    Returns:
        Boolean: True if ``OASIS_ANALYSIS_STATUS_URL`` or both ``OASIS_WEBSOCKET_URL`` and
            ``OASIS_WEBSOCKET_PORT`` are set
    """
    has_websocket = all(item in os.environ for item in ['OASIS_WEBSOCKET_URL', 'OASIS_WEBSOCKET_PORT'])
    return has_websocket or 'OASIS_ANALYSIS_STATUS_URL' in os.environ


def progress_enabled(socket_server):
    """Whether a pytool's ``--socket-server`` flag turns progress updates on.

    Args:
        socket_server (str): value of the tool's ``--socket-server`` flag, see `progress_ping_data`.

    Returns:
        Boolean: False if the flag is 'False' or None, True otherwise
    """
    return socket_server not in (None, 'False')


def add_progress_args(parser):
    """Adds the ``--socket-server`` and ``--analysis-pk`` flags a pytool uses to report progress.

    Args:
        parser (argparse.ArgumentParser): the tool's argument parser.
    """
    parser.add_argument('--socket-server', default='False',
                        help='Report completion to a progress server: a port for the local socket server, or True for the default target')
    parser.add_argument('--analysis-pk', default=None, help='Only used by platform to link run to an analysis')


def pop_progress_args(kwargs):
    """Removes the flags added by `add_progress_args` from a tool's parsed arguments.

    Args:
        kwargs (dict): the parsed arguments, e.g. ``vars(parser.parse_args())``.

    Returns:
        (str, str): the ``--socket-server`` and ``--analysis-pk`` values
    """
    return kwargs.pop('socket_server'), kwargs.pop('analysis_pk')


def oasis_ping(data):
    """Sends a JSON message to either an HTTP endpoint, a websocket server, or a socket server.


    If `analysis_pk` is in the data, targets are tried in order until one succeeds:
        - if `OASIS_ANALYSIS_STATUS_URL` is in environment, POSTs the message to that URL.
        - if `OASIS_WEBSOCKET_URL` and `OASIS_WEBSOCKET_PORT` are in environment, sends a websocket message.
        - if neither is configured, or all configured targets fail, no message gets through.
    Else, a message sent to `OASIS_SOCKET_SERVER_IP` `OASIS_SOCKET_SERVER_PORT` defaulted to 127.0.0.1 8888.

    If ``data`` contains a ``port_override`` key, that port is used in place of the default/env-var port
    when connecting to the socket server. The key is stripped before the message is sent.

    For a specific target, use `oasis_ping_http`, `oasis_ping_socket` or `oasis_ping_websocket` directly.

    Args:
        data (dict): dictionary of data: JSON serialisable

    Returns:
        Boolean: whether attempted call gets through
    """
    if data.get('analysis_pk', None) is not None:
        attempted = False
        if 'OASIS_ANALYSIS_STATUS_URL' in os.environ:
            attempted = True
            url = os.environ['OASIS_ANALYSIS_STATUS_URL']
            logger.debug(f"Sending ping to {url}: {data}")
            if oasis_ping_http(url, data):
                return True
        if all(item in os.environ for item in ['OASIS_WEBSOCKET_URL', 'OASIS_WEBSOCKET_PORT']):
            attempted = True
            msg = json.dumps(data)
            ws_url = f"{os.environ['OASIS_WEBSOCKET_URL']}:{os.environ['OASIS_WEBSOCKET_PORT']}/ws/analysis-status/"
            logger.debug(f"Sending ping to {ws_url}: {msg}")
            if oasis_ping_websocket(ws_url, msg):
                return True
        if not attempted:
            logger.error("Missing environment variables `OASIS_ANALYSIS_STATUS_URL` or "
                         "`OASIS_WEBSOCKET_URL`/`OASIS_WEBSOCKET_PORT`.")
        return False
    port_override = data.pop('port_override', None)
    msg = json.dumps(data)
    target_port = int(port_override) if port_override is not None else int(os.environ.get("OASIS_SOCKET_SERVER_PORT", SERVER_DEFAULT_PORT))
    target = (os.environ.get("OASIS_SOCKET_SERVER_IP", SERVER_DEFAULT_IP), target_port)
    logger.debug(f"Sending ping to {target}: {msg}")
    return oasis_ping_socket(target, msg)


def progress_ping_data(socket_server, analysis_pk=None, **progress):
    """Builds the payload of a progress update for a pytool run with a ``--socket-server`` flag.

    Args:
        socket_server (str): value of the tool's ``--socket-server`` flag. 'False' or None disables updates, a numeric
            value is the port of the local socket server, any other value uses the default target.
        analysis_pk (int, optional): platform analysis the update belongs to.
        **progress: progress fields to send, e.g. ``outputs_complete=1``.

    Returns:
        dict or None: the payload for `oasis_ping` or `oasis_ping_async`, or None if updates are disabled
    """
    if not progress_enabled(socket_server):
        return None
    data = {**progress, "analysis_pk": analysis_pk}
    if str(socket_server).isdigit():
        data['port_override'] = int(socket_server)
    return data


def oasis_ping_progress(socket_server, analysis_pk=None, **progress):
    """Sends a progress update for a pytool run with a ``--socket-server`` flag.

    Args:
        socket_server (str): value of the tool's ``--socket-server`` flag, see `progress_ping_data`.
        analysis_pk (int, optional): platform analysis the update belongs to.
        **progress: progress fields to send, e.g. ``outputs_complete=1``.

    Returns:
        Boolean: whether an update was sent and got through
    """
    data = progress_ping_data(socket_server, analysis_pk, **progress)
    if data is None:
        return False
    return oasis_ping(data)


def oasis_ping_async(data, on_failure=None):
    """Sends a ping without blocking the caller.

    `oasis_ping` makes a network call (HTTP POST or websocket connect) that can
    block for up to the configured timeout when the target is unreachable but
    not actively refusing the connection (wrong hostname, dropped packets, a
    NetworkPolicy). Calling it directly from a per-event progress ping inside a
    compute loop means a broken ping target stalls the calculation itself. This
    runs `oasis_ping` on a background daemon thread instead, so a stuck ping
    can never block the caller.

    Only one ping is ever in flight at a time per process: if a previous call
    is still pending when this is invoked, the new update is not sent rather
    than letting a persistently broken target pile up threads. Progress counts
    such as `events_complete` are increments, so callers must keep any count
    that was not sent and include it in their next ping.

    A dispatched ping can still fail to get through. ``on_failure`` is then
    called on the background thread before the next ping can be dispatched, so
    a caller can put the count back.

    Args:
        data (dict): dictionary of data: JSON serialisable
        on_failure (callable, optional): called with no arguments if the
            dispatched ping does not get through

    Returns:
        Boolean: True if the ping was dispatched, False if it was dropped because
            a previous ping is still in flight
    """
    if not _ping_lock.acquire(blocking=False):
        logger.debug(f"Skipping ping, previous ping still in flight: {data}")
        return False

    def _send():
        sent = False
        try:
            sent = oasis_ping(data)
        finally:
            if not sent and on_failure is not None:
                on_failure()
            _ping_lock.release()

    threading.Thread(target=_send, daemon=True).start()
    return True


def wait_for_async_ping(timeout):
    """Waits for a ping sent by `oasis_ping_async` to finish.

    Args:
        timeout (float): longest time to wait, in seconds

    Returns:
        Boolean: True if no ping is in flight, False if one was still being sent after ``timeout``
    """
    if not _ping_lock.acquire(timeout=timeout):
        return False
    _ping_lock.release()
    return True


class EventProgress:
    """Reports a GUL process's progress without blocking it or losing completed events.

    Completed events are sent at most every SERVER_UPDATE_TIME seconds through `oasis_ping_async`.
    A count stays pending until its ping gets through: if the ping is dropped or fails, the count
    goes out with the next update. `finish` waits for a ping still in flight, then sends what is left.

    Args:
        socket_server (str): value of the tool's ``--socket-server`` flag, see `progress_ping_data`.
        analysis_pk (int, optional): platform analysis the updates belong to.
    """

    def __init__(self, socket_server, analysis_pk=None):
        self.socket_server = socket_server
        self.analysis_pk = analysis_pk
        self.enabled = progress_enabled(socket_server)
        self._pending = 0
        self._lock = threading.Lock()
        self._last_update = time.time()

    def ready(self):
        """Reports that the process has loaded the model and is about to process events."""
        if self.enabled:
            oasis_ping_async(progress_ping_data(self.socket_server, self.analysis_pk, gul_ready=1))

    def event_done(self):
        """Counts one completed event, sending the pending count if an update is due."""
        if not self.enabled:
            return
        with self._lock:
            self._pending += 1
        if time.time() - self._last_update > SERVER_UPDATE_TIME:
            self._last_update = time.time()
            count = self._take_pending()
            if not oasis_ping_async(self._data(count), on_failure=lambda: self._add_pending(count)):
                self._add_pending(count)

    def finish(self):
        """Sends every completed event not yet reported, after waiting up to ASYNC_PING_EXIT_WAIT for a ping in flight."""
        if not self.enabled:
            return
        wait_for_async_ping(ASYNC_PING_EXIT_WAIT)
        oasis_ping(self._data(self._take_pending()))

    def _take_pending(self):
        with self._lock:
            count, self._pending = self._pending, 0
        return count

    def _add_pending(self, count):
        with self._lock:
            self._pending += count

    def _data(self, count):
        return progress_ping_data(self.socket_server, self.analysis_pk, events_complete=count)


def oasis_ping_socket(target, data):
    """Sends a JSON message to a target socket

    Args:
        target ((str, int)): IP and port to hit
        data (str): JSON dumped string

    Returns:
        Boolean: whether attempted call gets through
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as oasis_socket:
            oasis_socket.connect(target)
            oasis_socket.sendall(data.encode('utf-8'))
        return True
    except OSError as e:
        logger.error(f"oasis_ping_socket could not connect: {e}")
        return False


def oasis_ping_http(url, data):
    """Sends a JSON message to a target HTTP endpoint via POST.

    Args:
        url (str): URL to hit (e.g. "http://oasis-server:8000/analysis-status/")
        data (dict): dictionary of data: JSON serialisable

    Returns:
        Boolean: whether attempted call gets through
    """
    try:
        response = requests.post(url, json=data, timeout=1)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"oasis_ping_http could not connect: {e}")
        return False


def oasis_ping_websocket(ws_url, data):
    """Sends a JSON message to a target websocket

    Args:
        ws_url (str): URL to hit (e.g. "ws://oasis-websocket:8001/ws/analysis-status/")
        data (str): JSON dumped string

    Returns:
        Boolean: whether attempted call gets through
    """
    ws = websocket.WebSocket()
    try:
        ws.connect(ws_url, timeout=1)
        ws.send(data)
        return True
    except Exception as e:
        logger.error(f"oasis_ping_websocket could not connect: {e}")
        return False
    finally:
        ws.close()
