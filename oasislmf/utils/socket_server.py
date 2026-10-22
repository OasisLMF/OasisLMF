import argparse
import socket
import threading
import json
import os
import time
from oasislmf.utils.rounded_tqdm import rounded_tqdm
from oasislmf.utils.defaults import SERVER_UPDATE_TIME, SERVER_DEFAULT_PORT, SERVER_DEFAULT_IP
import logging


class GulProgressServer:
    def __init__(self, total, host=None, port=None, gul_processes=0, outputs_total=0):
        """Initialise the progress server.

        Args:
            total (int): How many events the server has
            host (str, optional): Non-default host to use. Defaults to `OASIS_SOCKET_SERVER_IP` or SERVER_DEFAULT_IP if unset.
            port (int, optional): Non-default port to use. Defaults to `OASIS_SOCKET_SERVER_PORT` or SERVER_DEFAULT_PORT if unset.
            gul_processes (int, optional): How many GUL processes will report `gul_ready` once they have
                loaded the model and are about to process events. Defaults to 0 (none report).
            outputs_total (int, optional): How many output processes (katpy, aalpy, lecpy) will report
                `outputs_complete` once the GUL events are done. Defaults to 0.
        """
        self.host = os.environ.get('OASIS_SOCKET_SERVER_IP', SERVER_DEFAULT_IP) if host is None else host
        self.port = int(os.environ.get('OASIS_SOCKET_SERVER_PORT', SERVER_DEFAULT_PORT)) if port is None else port
        self.total = total
        self.counter = 0
        self.gul_processes = gul_processes
        self.gul_ready = 0
        self.outputs_total = outputs_total
        self.outputs_complete = 0
        self.counter_lock = threading.Lock()
        self.running = False
        self._accept_thread = False

    def start(self):
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        try:
            self.server_socket.bind((self.host, self.port))
        except OSError:  # Desired socket unavailable
            self.server_socket.bind((self.host, 0))
            self.port = self.server_socket.getsockname()[1]
            logging.info(f"Port for socket server changed to {self.port}")
            os.environ['OASIS_SOCKET_SERVER_PORT'] = str(self.port)

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
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            logging.warning(f"Invalid json received: {data}")
            return
        with self.counter_lock:
            if 'terminate' in payload:
                self.counter = self.total
                self.outputs_complete = self.outputs_total
                self.stop()
            elif "events_complete" in payload:
                self.counter += int(payload.get("events_complete", 0))
            elif "gul_ready" in payload:
                self.gul_ready += int(payload.get("gul_ready", 0))
            elif "outputs_complete" in payload:
                self.outputs_complete += int(payload.get("outputs_complete", 0))
            else:
                logging.warning(f"Json received with no valid fields {payload}")

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


def gul_description(server):
    """Progress bar label: GUL processes still starting up, or events being completed."""
    if server.counter or server.gul_ready >= server.gul_processes:
        return "Gul events completed"
    return f"Gul starting ({server.gul_ready}/{server.gul_processes} ready)"


def main():
    parser = argparse.ArgumentParser(description="Show progress updates sent by a local loss run")
    parser.add_argument('total', type=int, help='Number of events in the run')
    parser.add_argument('port', type=int, nargs='?', default=None, help='Port to listen on')
    parser.add_argument('--gul-processes', type=int, default=0,
                        help='Number of GUL processes that report when they are ready to process events')
    parser.add_argument('--outputs', type=int, default=0,
                        help='Number of output processes (katpy, aalpy, lecpy) that report when they finish')
    args = parser.parse_args()

    with GulProgressServer(args.total, port=args.port, gul_processes=args.gul_processes,
                           outputs_total=args.outputs) as server:
        with rounded_tqdm(total=args.total, unit="events", desc=gul_description(server), leave=True) as pbar:
            counter = 0
            while counter < args.total:
                with server.counter_lock:
                    pbar.set_description_str(gul_description(server), refresh=False)
                    if counter != server.counter:
                        pbar.update(server.counter - counter)
                        counter = server.counter
                    else:
                        pbar.refresh()
                time.sleep(SERVER_UPDATE_TIME)

        if args.outputs:
            with rounded_tqdm(total=args.outputs, unit="outputs", desc="Outputs completed", leave=True) as pbar:
                counter = 0
                while counter < args.outputs:
                    with server.counter_lock:
                        if counter != server.outputs_complete:
                            pbar.update(server.outputs_complete - counter)
                            counter = server.outputs_complete
                    time.sleep(SERVER_UPDATE_TIME)
