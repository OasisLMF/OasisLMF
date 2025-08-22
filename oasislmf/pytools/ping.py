import json
import sys
import websocket
import logging
import socket
import os


def main():
    try:
        ws_url = sys.argv[1]
        message = sys.argv[2]
        data = json.loads(message)
    except Exception as e:
        logging.error("Ping called incorrectly: required call 'oasis-ping <location> <json>'")
        logging.error(f"error={str(e)}")
        logging.error(f"ws_url={ws_url}")
        logging.error(f"message={message}")
        return
    oasis_ping(data)


def oasis_ping(data):
    data = json.dumps(data)
    if os.environ.get('OASIS_WEBSOCKET_URL', None) is None:
        return oasis_ping_socket(data)
    oasis_ping_websocket(f"{os.environ['OASIS_WEBSOCKET_URL']}:{os.environ['OASIS_WEBSOCKET_PORT']}/ws/analysis-status/", data)


def oasis_ping_socket(data):
    oasis_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    oasis_socket.connect(("0.0.0.0", 8888))
    oasis_socket.send(data.encode('utf-8'))


def oasis_ping_websocket(ws_url, data):
    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        ws.send(data)
        ws.close()
        logging.info("Post sent successfully")
    except Exception:
        logging.error("Ping failed to call")
