import socket
import threading
import json
import os
import time
import sys
from tqdm import tqdm
from oasislmf.utils.defaults import SERVER_UPDATE_TIME, SERVER_DEFAULT_PORT, SERVER_DEFAULT_IP


class GulProgressServer:
    def __init__(self, host=None, port=None):
        """
        Args:
            host (str, optional): Non-default host to use. Defaults to `OASIS_SOCKET_SERVER_IP` or SERVER_DEFAULT_IP if unset.
            port (int, optional): Non-default port to use. Defaults to `OASIS_SOCKET_SERVER_PORT` or SERVER_DEFAULT_PORT if unset.
        """
        self.host = os.environ.get('OASIS_SOCKET_SERVER_IP', SERVER_DEFAULT_IP) if host is None else host
        self.port = int(os.environ.get('OASIS_SOCKET_SERVER_PORT', SERVER_DEFAULT_PORT)) if port is None else port
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.running = False
        self._accept_thread = False

    def start(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self):
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(client_socket,), daemon=True).start()
            except OSError:
                break  # socket was closed

    def _handle_client(self, client_socket):
        data = self._read_all(client_socket)
        payload = json.loads(data)
        with self.counter_lock:
            self.counter += int(payload.get("events_complete", 0))

    def _read_all(self, client_socket):
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = client_socket.recv(1024)
            if not chunk:  # client disconnected
                break
            buffer += chunk
        return buffer.decode("utf-8").strip()

    def stop(self):
        self.running = False
        self.server_socket.close()
        if self._accept_thread:
            self._accept_thread.join(timeout=1)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


def main():
    if len(sys.argv) < 2:
        raise ValueError("Socket server must be called with an argument for pbar length")
    try:
        total = int(sys.argv[1])
    except Exception:
        raise TypeError("Socket server argument must be an integer")
    with (GulProgressServer() as server,
          tqdm(total=total, unit="events", desc="Gul events completed", leave=True) as pbar):
        counter = 0
        while counter < total:
            with server.counter_lock:
                if counter != server.counter:
                    pbar.update(server.counter - counter)
                    counter = server.counter
            time.sleep(SERVER_UPDATE_TIME)
