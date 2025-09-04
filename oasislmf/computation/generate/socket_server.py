import socket
import threading
import json


class GulProgressServer:
    def __init__(self, host="127.0.0.1", port=8888):
        self.host = host
        self.port = int(port)
        self.counter = 0
        self.counter_lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()

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
