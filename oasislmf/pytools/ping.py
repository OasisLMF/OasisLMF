import json
import sys
import websocket
import logging


def main():
    try:
        ws_url = sys.argv[1]
        message = sys.argv[2]
        data = json.loads(message)
    except Exception:
        logging.error("Ping called incorrectly: required call 'oasis-ping <location> <json>'")
        return 0

    try:
        ws = websocket.WebSocket()
        ws.connect(ws_url)
        ws.send(json.dumps(data))
        ws.close()
        logging.info("Post sent successfully")
    except Exception:
        logging.error("Ping failed to call")

    return 0
