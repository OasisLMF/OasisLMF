import json
import sys
import websocket
import logging


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
    oasis_ping(ws_url, data)


def oasis_ping(ws_url, data):
    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        ws.send(json.dumps(data))
        ws.close()
        logging.info("Post sent successfully")
    except Exception:
        logging.error("Ping failed to call")
