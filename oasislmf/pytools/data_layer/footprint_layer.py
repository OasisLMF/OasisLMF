import argparse
import atexit
import datetime
import logging
import math
import os
import pickle
import socket
from contextlib import ExitStack
from enum import Enum
from typing import Optional, Set, Tuple, List

import numpy as np

from oasislmf.pytools.getmodel.footprint import Footprint

# configuring process meta data
logging.basicConfig(
    filename='footprint_tcp_server.log',
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    level=os.environ.get("LOGLEVEL", "INFO")
)
POINTER_PATH = str(os.path.dirname(os.path.realpath(__file__))) + "/pointer_flag.txt"
TCP_IP = '127.0.0.1'
TCP_PORT = 8080
PROCESSES_SUPPORTED = 100

from random import randint
MODEL_LOG_PATH = str(os.getcwd()) + f"/{randint(1,900)}_model_log.txt"


class OperationEnum(Enum):
    """
    Defines the different types of operations supported via bytes. To be passed through TCP port first to tell the
    server what type of operation is required.
    """
    GET_DATA = (1).to_bytes(4, byteorder='big')
    GET_NUM_INTENSITY_BINS = (2).to_bytes(4, byteorder='big')
    REGISTER = (3).to_bytes(4, byteorder='big')
    UNREGISTER = (4).to_bytes(4, byteorder='big')
    POLL_DATA = (5).to_bytes(4, byteorder='big')


class FootprintLayer:
    """
    This class is responsible for accessing the footprint data via TCP ports.

    Attributes:
        static_path (str): path to the static file to load the data
        ignore_file_type (Set[str]): collection of file types to ignore when loading
        file_data (Optional[Footprint]): footprint object to load
        socket (Optional[socket.socket]): the TCP socket in which data is sent
        count (int): the number of processes currently relying on the server
        total_expected (int): the total number of reliant processes expected
        total_served (int): the total number of processes that have ever registered through the server's lifetime
    """
    def __init__(self, static_path: str, total_expected: int, ignore_file_type: Set[str] = set()) -> None:
        """
        The constructor for the FootprintLayer class.

        Args:
            static_path: (str) path to the static file to load the data
            ignore_file_type: (Set[str]) collection of file types to ignore when loading
            total_expected: (int) the total number of reliant processes expected

        """
        self.static_path: str = static_path
        self.ignore_file_type: Set[str] = ignore_file_type
        self.file_data: Optional[Footprint] = None
        self.socket: Optional[socket.socket] = None
        self.count: int = 0
        self.total_expected: int = total_expected
        self.total_served: int = 0
        self._define_socket()

    def _define_socket(self) -> None:
        """
        Defines the self.socket attribute to the port and localhost.

        Returns: None
        """
        logging.info(f"defining socket for TCP server: {datetime.datetime.now()}")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (TCP_IP, TCP_PORT)
        self.socket.bind(server_address)
        self.socket.listen(PROCESSES_SUPPORTED)

    def _establish_shutdown_procedure(self) -> None:
        """
        Establishes the steps for shutdown, writing the pointer, and making sure that the pointer will be deleted
        and the self.socket is shutdown once the process running the server is exited.

        Returns: None
        """
        logging.info(f"establishing shutdown procedure: {datetime.datetime.now()}")
        # atexit.register(_shutdown_port, self.socket)
        pass

    @staticmethod
    def _stream_footprint_data(event_data: np.array, connection: socket.socket, event_id: int) -> None:
        """
        Serialises data then splits it into chunks of 500 in turn streaming through a connection.

        Args:
            event_data: (np.array) the data to be serialised and streamed through a connection
            connection: (socket.socket) the connection that the data is going to be streamed through

        Returns: None
        """
        raw_data: bytes = pickle.dumps(event_data)

        raw_data_buffer: List[bytes] = [raw_data[i:i + 60000] for i in range(0, len(raw_data), 60000)]

        for chunk in raw_data_buffer:
            connection.sendall(chunk)

    @staticmethod
    def _extract_header(header_data: bytes) -> Tuple[OperationEnum, Optional[int]]:
        """
        Extracts the operation type and event_id from the header.

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
        """
        Fires off the process with an event loop serving footprint data.

        Returns: None
        """
        self._establish_shutdown_procedure()

        with ExitStack() as stack:
            footprint_obj = stack.enter_context(Footprint.load(static_path=self.static_path,
                                                               ignore_file_type=self.ignore_file_type))
            self.file_data = footprint_obj
            while True:
                try:
                    connection, client_address = self.socket.accept()
                    data = connection.recv(16)

                    if data:
                        operation, event_id = self._extract_header(header_data=data)

                        if operation == OperationEnum.GET_DATA:
                            event_data = self.file_data.get_event(event_id=event_id)

                            if event_id in self.file_data.footprint_index:
                                logging.info(f'event_id "{event_id}" retrieved from footprint index')
                                del self.file_data.footprint_index[event_id]
                            else:
                                logging.error(f'event_id "{event_id}" not in footprint_index')

                            FootprintLayer._stream_footprint_data(event_data=event_data, connection=connection, event_id=event_id)

                        elif operation == OperationEnum.GET_NUM_INTENSITY_BINS:

                            number_of_intensity_bins = self.file_data.num_intensity_bins
                            connection.sendall(number_of_intensity_bins.to_bytes(8, byteorder='big'))

                        elif operation == OperationEnum.REGISTER:
                            self.count += 1
                            self.total_served += 1
                            logging.info(f"connection registered: {self.count} for {client_address} {datetime.datetime.now()}")

                        elif operation == OperationEnum.UNREGISTER:
                            self.count -= 1
                            logging.info(f"connection unregistered: {self.count} for {client_address} {datetime.datetime.now()}")
                            if self.count <= 0 and self.total_expected == self.total_served:
                                logging.info(f"breaking event loop: {datetime.datetime.now()}")
                                self.socket.shutdown(socket.SHUT_RDWR)
                                break
                        connection.close()
                # Catch all errors, send to logger and keep running        
                except Exception as e:
                    logging.error(e)
            connection.close()


class FootprintLayerClient:
    """
    This class is responsible for connecting to the footprint server via TCP.
    """
    @classmethod
    def poll(cls) -> bool:
        """
        Checks to see if data server is running.

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
        """
        Gets the socket using the cls.TCP_IP and cls.TCP_PORT.

        Returns: (socket.socket) a connected socket
        """
        current_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        current_socket.connect((TCP_IP, TCP_PORT))
        current_socket.settimeout(10)
        return current_socket

    @classmethod
    def _define_shutdown_procedure(cls) -> None:
        """
        Unregisters the client to the server on exit of the process.

        Returns: None
        """
        atexit.register(cls.unregister)

    @classmethod
    def register(cls) -> None:
        """
        Registers the client with the server.

        Returns: None
        """
        connection_viable: bool = False
        while connection_viable is False:
            connection_viable = FootprintLayerClient.poll()

        current_socket = cls._get_socket()
        data: bytes = OperationEnum.REGISTER.value
        current_socket.sendall(data)
        current_socket.close()
        # cls._define_shutdown_procedure()

    @classmethod
    def unregister(cls) -> None:
        """
        Unregisters the client with the data server.

        Returns: None
        """
        current_socket = cls._get_socket()
        data: bytes = OperationEnum.UNREGISTER.value
        current_socket.sendall(data)
        current_socket.close()
        atexit.unregister(cls.unregister)

    @classmethod
    def get_number_of_intensity_bins(cls) -> int:
        """
        Gets the number of intensity bins from the footprint data.

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
        """
        Gets the footprint data from the footprint based off the event_id.

        Args:
            event_id: (int) the event ID of the data required

        Returns: (np.array) footprint data based off the event_id
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

        return pickle.loads(b"".join(raw_data_buffer))


def _shutdown_port(connection: socket.socket) -> None:
    logging.info(f"socket is shutting down: {datetime.datetime.now()}")
    connection.shutdown(socket.SHUT_RDWR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("p", help="path to static file", type=str)
    parser.add_argument("n", help="number of processes expected to be reliant on server", type=int)
    args = parser.parse_args()
    server = FootprintLayer(static_path=args.p, total_expected=args.n)
    server.listen()


if __name__ == "__main__":
    main()

