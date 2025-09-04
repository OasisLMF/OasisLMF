import socket
import threading
import json
import os


class GulProgressServer:
    def __init__(self, host=None, port=None):
        """
        Args:
            host (str, optional): Non-default host to use. Defaults to `OASIS_SOCKET_SERVER_IP` or `127.0.0.1` if unset.
            port (int, optional): Non-default port to use. Defaults to `OASIS_SOCKET_SERVER_PORT` or 8888 if unset.
        """
        self.host = os.environ.get('OASIS_SOCKET_SERVER_IP', "127.0.0.1") if host is None else host
        self.port = int(os.environ.get('OASIS_SOCKET_SERVER_PORT', 8888)) if port is None else port
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
        data = client_socket.recv(1024)
        payload = json.loads(data.decode("utf-8").strip())
        with self.counter_lock:
            self.counter += int(payload.get("events_complete", 0))
        try:
            client_socket.sendall(b"OK\n")
        finally:
            client_socket.close()

    def stop(self):
        self.running = False
        self.server_socket.close()
        if self._accept_thread:
            self._accept_thread.join(timeout=1)
