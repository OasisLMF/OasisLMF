import json
import websocket
import socket
import os
import logging


def oasis_ping(data):
    """
    Sends a JSON message to either a websocket server or a socket server.

    If `analysis_pk` is in the data, `OASIS_WEBSOCKET_URL` and `OASIS_WEBSOCKET_URL` are in environment, sends a websocket message.
    If `analysis_pk` but missing variables, no message sent.
    Else, websocket sent to `OASIS_SOCKET_SERVER_IP` `OASIS_SOCKET_SERVER_PORT` defaulted to 127.0.0.1 8888.

    For a specific socket or websocket, use `oasis_ping_socket` or `oasis_ping_websocket` with the target location.

    Args:
        data (dict): dictionary of data: JSON serialisable

    Returns:
        Boolean: whether attempted call gets through
    """
    msg = json.dumps(data)
    if data.get('analysis_pk', None) is not None:
        if all(item in os.environ for item in ['OASIS_WEBSOCKET_URL', 'OASIS_WEBSOCKET_PORT']):
            return oasis_ping_websocket(f"{os.environ['OASIS_WEBSOCKET_URL']}:{os.environ['OASIS_WEBSOCKET_PORT']}/ws/analysis-status/", msg)
        logging.error("Missing environment variables `OASIS_WEBSOCKET_URL` and `OASIS_WEBSOCKET_PORT`.")
        return False
    target = (os.environ.get("OASIS_SOCKET_SERVER_IP", "127.0.0.1"), int(os.environ.get("OASIS_SOCKET_SERVER_PORT", 8888)))
    return oasis_ping_socket(target, msg)


def oasis_ping_socket(target, data):
    """
    Sends a JSON message to a target socket

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
    except ConnectionRefusedError as e:
        logging.error(f"oasis_ping_socket could not connect: {e}")
        return False


def oasis_ping_websocket(ws_url, data):
    """
    Sends a JSON message to a target websocket

    Args:
        ws_url (str): URL to hit (e.g. "ws://oasis-websocket:8001/ws/analysis-status/")
        data (str): JSON dumped string

    Returns:
        Boolean: whether attempted call gets through
    """
    ws = websocket.WebSocket()
    try:
        ws.connect(ws_url)
        ws.send(data)
        return True
    except Exception as e:
        logging.error(f"oasis_ping_websocket could not connect: {e}")
        return False
    finally:
        ws.close()
