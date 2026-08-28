import json
import websocket
import socket
import os
import logging
import requests
from oasislmf.utils.defaults import SERVER_DEFAULT_PORT, SERVER_DEFAULT_IP

logger = logging.getLogger(__name__)


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
    except (ConnectionError, TimeoutError, socket.gaierror) as e:
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
