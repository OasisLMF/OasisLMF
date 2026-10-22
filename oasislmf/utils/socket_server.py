import argparse
import socket
import threading
import json
import os
from oasislmf.utils.rounded_tqdm import rounded_tqdm
from oasislmf.utils.defaults import SERVER_UPDATE_TIME, SERVER_DEFAULT_PORT, SERVER_DEFAULT_IP
import logging


class GulProgressServer:
    def __init__(self, total, host=None, port=None, gul_processes=0):
        """Initialise the progress server.

        The run reports in two stages. GUL processes send `gul_ready` once they have loaded the model and
        `events_complete` counts as they go. Once the GUL pipeline has finished, the script sends `outputs_total`
        (the number of output processes), each output process sends `outputs_complete` when it finishes, and the
        script sends `outputs_done` at the end of the run.

        Args:
            total (int): How many events the server has
            host (str, optional): Non-default host to use. Defaults to `OASIS_SOCKET_SERVER_IP` or SERVER_DEFAULT_IP if unset.
            port (int, optional): Non-default port to use. Defaults to `OASIS_SOCKET_SERVER_PORT` or SERVER_DEFAULT_PORT if unset.
            gul_processes (int, optional): How many GUL processes will report `gul_ready` once they have
                loaded the model and are about to process events. Defaults to 0 (none report).
        """
        self.host = os.environ.get('OASIS_SOCKET_SERVER_IP', SERVER_DEFAULT_IP) if host is None else host
        self.port = int(os.environ.get('OASIS_SOCKET_SERVER_PORT', SERVER_DEFAULT_PORT)) if port is None else port
        self.total = total
        self.counter = 0
        self.gul_processes = gul_processes
        self.gul_ready = 0
        self.outputs_total = None
        self.outputs_complete = 0
        self.finished = False
        self.counter_lock = threading.Lock()
        self.updated = threading.Event()
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
                if self.outputs_total is not None:
                    self.outputs_complete = self.outputs_total
                self.finished = True
                self.stop()
            elif "events_complete" in payload:
                self.counter += int(payload.get("events_complete", 0))
            elif "gul_ready" in payload:
                self.gul_ready += int(payload.get("gul_ready", 0))
            elif "outputs_total" in payload:
                self.outputs_total = int(payload["outputs_total"])
            elif "outputs_complete" in payload:
                self.outputs_complete += int(payload.get("outputs_complete", 0))
            elif "outputs_done" in payload:
                self.finished = True
            else:
                logging.warning(f"Json received with no valid fields {payload}")
        self.updated.set()

    def _read_all(self, client_socket):
        buffer = b""
        while not buffer.endswith(b"\n"):
            chunk = client_socket.recv(1024)
            if not chunk:  # client disconnected
                break
            buffer += chunk
        return buffer.decode("utf-8").strip()

    def wait_for_update(self):
        """Block until a message arrives or SERVER_UPDATE_TIME seconds pass."""
        self.updated.wait(SERVER_UPDATE_TIME)
        self.updated.clear()

    def gul_done(self):
        """Whether the GUL stage is over: every event counted, or the script has moved on to the outputs."""
        return self.counter >= self.total or self.outputs_total is not None or self.finished

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


def show_gul_progress(server):
    """Draw the GUL bar until the GUL stage is over.

    The bar is filled once the script reports the outputs stage has started, so events whose
    `events_complete` count was lost do not hold up the run's progress display.
    """
    with rounded_tqdm(total=server.total, unit="events", desc=gul_description(server), leave=True) as pbar:
        shown = 0
        while True:
            with server.counter_lock:
                done = server.gul_done()
                count = server.total if done else server.counter
                pbar.set_description_str(gul_description(server), refresh=False)
            if count != shown:
                pbar.update(count - shown)
                shown = count
            else:
                pbar.refresh()
            if done:
                return
            server.wait_for_update()


def show_outputs_progress(server):
    """Draw the outputs bar once the script has sent `outputs_total`, until every output reports or the run ends."""
    while True:
        with server.counter_lock:
            outputs_total, finished = server.outputs_total, server.finished
        if outputs_total is not None or finished:
            break
        server.wait_for_update()
    if not outputs_total:
        return

    with rounded_tqdm(total=outputs_total, unit="outputs", desc="Outputs completed", leave=True) as pbar:
        shown = 0
        while True:
            with server.counter_lock:
                count = min(server.outputs_complete, outputs_total)
                done = count >= outputs_total or server.finished
            if count != shown:
                pbar.update(count - shown)
                shown = count
            if done:
                return
            server.wait_for_update()


def main():
    parser = argparse.ArgumentParser(description="Show progress updates sent by a local loss run")
    parser.add_argument('total', type=int, help='Number of events in the run')
    parser.add_argument('port', type=int, nargs='?', default=None, help='Port to listen on')
    parser.add_argument('--gul-processes', type=int, default=0,
                        help='Number of GUL processes that report when they are ready to process events')
    args = parser.parse_args()

    with GulProgressServer(args.total, port=args.port, gul_processes=args.gul_processes) as server:
        show_gul_progress(server)
        show_outputs_progress(server)
