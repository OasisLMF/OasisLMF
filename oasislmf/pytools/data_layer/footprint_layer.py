from random import randint
import argparse
import atexit
import datetime
import logging
import os
import pickle
import socket
import time
from contextlib import ExitStack
from enum import Enum
from typing import Optional, Set, Tuple, List

import numpy as np

from oasis_data_manager.filestore.backends.local import LocalStorage
from oasis_data_manager.filestore.backends.base import BaseStorage
from oasislmf.pytools.getmodel.common import SECTION_MERGE_DEFAULT, SECTION_MERGE_OPTIONS
from oasislmf.pytools.getmodel.footprint import Footprint

POINTER_PATH = str(os.path.dirname(os.path.realpath(__file__))) + "/pointer_flag.txt"
TCP_IP = '127.0.0.1'
TCP_PORT = 8080
PROCESSES_SUPPORTED = 100

MODEL_LOG_PATH = str(os.getcwd()) + f"/{randint(1, 900)}_model_log.txt"

# how long a client keeps trying to reach a server that is not (or no longer) listening
REGISTER_TIMEOUT_SECONDS = 300
REGISTER_RETRY_SECONDS = 0.1


class FootprintServerError(Exception):
    """An error raised by the footprint server while serving a request, sent back to the client
    in place of the event's data so that the client stops rather than waits for data that never comes.
    """


class OperationEnum(Enum):
    """Defines the different types of operations supported via bytes. To be passed through TCP port first to tell the
    server what type of operation is required.
    """
    GET_DATA = (1).to_bytes(4, byteorder='big')
    GET_NUM_INTENSITY_BINS = (2).to_bytes(4, byteorder='big')
    REGISTER = (3).to_bytes(4, byteorder='big')
    UNREGISTER = (4).to_bytes(4, byteorder='big')
    POLL_DATA = (5).to_bytes(4, byteorder='big')


class FootprintLayer:
    """Class to accessing the footprint data via TCP ports.

    Attributes:
        static_path (str): path to the static file to load the data
        ignore_file_type (Set[str]): collection of file types to ignore when loading
        file_data (Optional[Footprint]): footprint object to load
        socket (Optional[socket.socket]): the TCP socket in which data is sent
        count (int): the number of processes currently relying on the server
        total_expected (int): the total number of reliant processes expected
        total_served (int): the total number of processes that have ever registered through the server's lifetime
    """

    def __init__(
        self,
        storage: BaseStorage,
        total_expected: int,
        ignore_file_type: Set[str] = set(),
        df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
        section_merge=None,
        input_path="input",
    ) -> None:
        """The constructor for the FootprintLayer class.

        Args:
            storage (BaseStorage): the storage connector the footprint data is loaded from
            total_expected (int): the total number of reliant processes expected
            ignore_file_type (Set[str]): collection of file types to ignore when loading
            df_engine (str): the engine to use when loading dataframes
            section_merge (str or None): how a dynamic footprint merges an areaperil's intensities
                across the sections of one event (see Footprint.load)
            input_path (str or None): the run's input directory, whose items give the areaperils the
                footprint is served for, as gulmc and getmodel load it themselves. None, or a directory
                without items, serves every areaperil, which a dynamic footprint cannot do.
        """
        self.storage = storage
        self.ignore_file_type: Set[str] = ignore_file_type
        self.file_data: Optional[Footprint] = None
        self.socket: Optional[socket.socket] = None
        self.count: int = 0
        self.total_expected: int = total_expected
        self.total_served: int = 0
        self.df_engine = df_engine
        self.section_merge = section_merge
        self.input_path = input_path
        self._define_socket()

    def _define_socket(self) -> None:
        """Defines the self.socket attribute to the port and localhost.

        Returns: None
        """
        logging.info(f"defining socket for TCP server: {datetime.datetime.now()}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_address = (TCP_IP, TCP_PORT)
        self.socket.bind(server_address)
        self.socket.listen(PROCESSES_SUPPORTED)

    def _establish_shutdown_procedure(self) -> None:
        """Establishes the steps for shutdown, writing the pointer, and making sure that the pointer will be deleted
        and the self.socket is shutdown once the process running the server is exited.

        Returns: None
        """
        logging.info(f"establishing shutdown procedure: {datetime.datetime.now()}")

    def _portfolio_areaperil_ids(self) -> Optional[np.ndarray]:
        """The areaperil_ids of the run's items, for which the footprint is served.

        A client only uses the areaperils of its items, which are those of the run's items (a peril filter
        only narrows them), so serving these alone gives it the same footprint as loading it itself. A
        dynamic footprint needs them, to know which hazard to load.

        Returns: (np.ndarray or None) the unique areaperil_ids, or None if there is no input directory or items file
        """
        if self.input_path is None or not os.path.isdir(self.input_path):
            logging.info(f"no input directory {self.input_path}: serving the footprint for every areaperil")
            return None

        from oasislmf.pytools.gulmc.items import read_items
        try:
            items = read_items(self.input_path, self.ignore_file_type)
        except FileNotFoundError:
            logging.info(f"no items in {self.input_path}: serving the footprint for every areaperil")
            return None
        areaperil_ids = np.unique(items['areaperil_id'])
        logging.info(f"serving the footprint for the {len(areaperil_ids)} areaperils of the items in {self.input_path}")
        return areaperil_ids

    @staticmethod
    def _stream_footprint_data(event_data: np.array, connection: socket.socket, event_id: int) -> None:
        """Serialises data then splits it into chunks of 500 in turn streaming through a connection.

        Args:
            event_data (np.array): the data to be serialised and streamed through a connection
            connection (socket.socket): the connection that the data is going to be streamed through
            event_id (int): the id of the event the data belongs to, currently not used by the
                streaming itself
        """
        raw_data: bytes = pickle.dumps(event_data)

        raw_data_buffer: List[bytes] = [raw_data[i:i + 60000] for i in range(0, len(raw_data), 60000)]

        for chunk in raw_data_buffer:
            connection.sendall(chunk)

    @staticmethod
    def _extract_header(header_data: bytes) -> Tuple[OperationEnum, Optional[int]]:
        """Extracts the operation type and event_id from the header.

        Args:
            header_data: (bytes) header data sent from the client and recieved via TCP

        Returns: (Tuple[OperationEnum, Optional[int]]) event_id is None if the Operation is not GET_DATA
        """
        operation_number_data = header_data[:4]
        event_id_data = header_data[4:12]

        operation: OperationEnum = OperationEnum(operation_number_data)

        if operation == OperationEnum.GET_DATA:
            event_id: Optional[int] = int.from_bytes(event_id_data, 'big')
        else:
            event_id: Optional[int] = None

        return operation, event_id

    def listen(self) -> None:
        """Fires off the process with an event loop serving footprint data.

        Returns: None
        """
        self._establish_shutdown_procedure()

        with ExitStack() as stack:
            footprint_obj = stack.enter_context(Footprint.load(self.storage,
                                                               ignore_file_type=self.ignore_file_type,
                                                               df_engine=self.df_engine,
                                                               areaperil_ids=self._portfolio_areaperil_ids(),
                                                               section_merge=self.section_merge))
            self.file_data = footprint_obj
            logging.info(f"serving {type(footprint_obj).__name__} footprint: {datetime.datetime.now()}")
            while True:
                connection, client_address = self.socket.accept()
                try:
                    if self._serve(connection, client_address):
                        logging.info(f"breaking event loop: {datetime.datetime.now()}")
                        self.socket.shutdown(socket.SHUT_RDWR)
                        break
                # Catch all errors, send to logger and keep running
                except Exception:
                    logging.exception(f"request from {client_address} failed")
                finally:
                    connection.close()

    def _serve(self, connection: socket.socket, client_address) -> bool:
        """Serves one request.

        Args:
            connection (socket.socket): the connection the request came through
            client_address: the address of the client, for the log

        Returns: (bool) True if every expected client has unregistered and the server should stop
        """
        data = connection.recv(16)
        if not data:
            return False
        operation, event_id = self._extract_header(header_data=data)

        if operation == OperationEnum.GET_DATA:
            try:
                event_data = self.file_data.get_event(event_id)
            except Exception as e:
                # the client is waiting on this event: tell it, rather than leave it to time out
                logging.exception(f"event_id {event_id} could not be served")
                event_data = FootprintServerError(f"footprint server failed to serve event {event_id}: {e!r}")
            FootprintLayer._stream_footprint_data(event_data=event_data, connection=connection, event_id=event_id)

        elif operation == OperationEnum.GET_NUM_INTENSITY_BINS:
            number_of_intensity_bins = int(self.file_data.num_intensity_bins)
            connection.sendall(number_of_intensity_bins.to_bytes(8, byteorder='big'))

        elif operation == OperationEnum.REGISTER:
            self.count += 1
            self.total_served += 1
            logging.info(f"connection registered: {self.count} for {client_address} {datetime.datetime.now()}")

        elif operation == OperationEnum.UNREGISTER:
            self.count -= 1
            logging.info(f"connection unregistered: {self.count} for {client_address} {datetime.datetime.now()}")
            return self.count <= 0 and self.total_expected == self.total_served
        return False


class FootprintLayerClient:
    """Connects to the footprint server via TCP."""
    @classmethod
    def poll(cls) -> bool:
        """Checks to see if data server is running.

        Returns: (bool)
        """
        try:
            _ = cls._get_socket()
            return True
        except ConnectionRefusedError as e:
            logging.error('Failed to find server: {}'.format(e))
            return False

    @classmethod
    def _get_socket(cls) -> socket.socket:
        """Gets the socket using the cls.TCP_IP and cls.TCP_PORT.

        Returns: (socket.socket) a connected socket
        """
        current_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        current_socket.connect((TCP_IP, TCP_PORT))
        current_socket.settimeout(10)
        return current_socket

    @classmethod
    def register(cls) -> None:
        """Registers the client with the server.

        Returns: None
        """
        deadline = time.monotonic() + REGISTER_TIMEOUT_SECONDS
        while not FootprintLayerClient.poll():
            if time.monotonic() > deadline:
                raise FootprintServerError(
                    f"no footprint server (servedata) listening on {TCP_IP}:{TCP_PORT} after {REGISTER_TIMEOUT_SECONDS}s; "
                    f"if it was started, it has stopped: see its log, footprint_tcp_server.log")
            time.sleep(REGISTER_RETRY_SECONDS)

        current_socket = cls._get_socket()
        data: bytes = OperationEnum.REGISTER.value
        current_socket.sendall(data)
        current_socket.close()

    @classmethod
    def unregister(cls) -> None:
        """Unregisters the client with the data server.

        Returns: None
        """
        current_socket = cls._get_socket()
        data: bytes = OperationEnum.UNREGISTER.value
        current_socket.sendall(data)
        current_socket.close()
        atexit.unregister(cls.unregister)

    @classmethod
    def get_number_of_intensity_bins(cls) -> int:
        """Gets the number of intensity bins from the footprint data.

        Returns: (int) the number of intensity bins
        """
        current_socket = cls._get_socket()

        data: bytes = OperationEnum.GET_NUM_INTENSITY_BINS.value
        current_socket.sendall(data)
        intensity_bins_data = current_socket.recv(8)
        current_socket.close()
        return int.from_bytes(intensity_bins_data, 'big')

    @classmethod
    def get_event(cls, event_id: int) -> np.array:
        """Gets the footprint data from the footprint based off the event_id.

        Args:
            event_id: (int) the event ID of the data required

        Returns: (np.array) footprint data based off the event_id, or None if the event has none

        Raises:
            FootprintServerError: if the server failed to get the event
        """
        current_socket = cls._get_socket()

        data: bytes = OperationEnum.GET_DATA.value + int(event_id).to_bytes(8, byteorder='big')
        current_socket.sendall(data)

        raw_data_buffer: List[bytes] = []

        while True:
            raw_data = current_socket.recv(6000)
            if not raw_data:
                break
            raw_data_buffer.append(raw_data)

        if raw_data_buffer:
            event_data = pickle.loads(b"".join(raw_data_buffer))
            if isinstance(event_data, FootprintServerError):
                raise event_data
            return event_data


def main() -> None:
    logging.basicConfig(
        filename='footprint_tcp_server.log',
        filemode='w',
        format='%(name)s - %(levelname)s - %(message)s',
        level=os.environ.get("OASIS_PACKAGE_LOG_LEVEL", "INFO")
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("p", help="path to static file", type=str)
    parser.add_argument("n", help="number of processes expected to be reliant on server", type=int)
    parser.add_argument("--df-engine", help="The engine to use when loading dataframes",
                        default="oasis_data_manager.df_reader.reader.OasisPandasReader", )
    parser.add_argument("--section-merge", default=SECTION_MERGE_DEFAULT, choices=SECTION_MERGE_OPTIONS,
                        help="How a dynamic footprint merges the intensities of an areaperil lying in several "
                             f"sections of one event. Default: {SECTION_MERGE_DEFAULT}")
    parser.add_argument("--input-path", default="input",
                        help="The run's input directory: the footprint is served for the areaperils of its items. "
                             "Default: input")
    args = parser.parse_args()
    server = FootprintLayer(LocalStorage(root_dir=args.p), total_expected=args.n, df_engine=args.df_engine,
                            section_merge=args.section_merge, input_path=args.input_path)
    server.listen()


if __name__ == "__main__":
    main()
