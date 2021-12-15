import pickle
import socket
from contextlib import ExitStack
from enum import Enum
from multiprocessing import Process
from typing import Optional, Set, Tuple, List
import math

import numpy as np

from oasislmf.pytools.getmodel.footprint import Footprint


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
    """
    def __init__(self, static_path: str, ignore_file_type: Set[str] = set()) -> None:
        """
        The constructor for the FootprintLayer class.

        Args:
            static_path: (str) path to the static file to load the data
            ignore_file_type: (Set[str]) collection of file types to ignore when loading
        """
        self.static_path: str = static_path
        self.ignore_file_type: Set[str] = ignore_file_type
        self.file_data: Optional[Footprint] = None
        self.socket: Optional[socket.socket] = None
        self.count: int = 0
        self._define_socket()

    def _define_socket(self) -> None:
        """
        Defines the self.socket attribute to the port and localhost.

        Returns: None
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('127.0.0.1', 8080)
        self.socket.bind(server_address)
        self.socket.listen(1)

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
        with ExitStack() as stack:
            footprint_obj = stack.enter_context(Footprint.load(static_path=self.static_path,
                                                               ignore_file_type=self.ignore_file_type))
            self.file_data = footprint_obj

            while True:

                connection, client_address = self.socket.accept()
                data = connection.recv(16)

                if data:
                    operation, event_id = self._extract_header(header_data=data)

                    if operation == OperationEnum.GET_DATA:
                        event_data = self.file_data.get_event(event_id=event_id)
                        try:
                            del self.file_data.footprint_index[event_id]
                        except KeyError:
                            # TODO => log the key error
                            pass
                        raw_data = pickle.dumps(event_data)
                        number_of_chunks: int = int(math.ceil(len(raw_data) / 500))
                        raw_data_buffer = [raw_data[i:i + 500] for i in range(0, len(raw_data), 500)]

                        connection.send(number_of_chunks.to_bytes(32, byteorder='big'))

                        for chunk in raw_data_buffer:
                            connection.send(chunk)

                    elif operation == OperationEnum.GET_NUM_INTENSITY_BINS:
                        number_of_intensity_bins = self.file_data.num_intensity_bins
                        connection.send(number_of_intensity_bins.to_bytes(8, byteorder='big'))

                    elif operation == OperationEnum.REGISTER:
                        self.count += 1

                    elif operation == OperationEnum.UNREGISTER:
                        self.count -= 1
                        if self.count <= 0:
                            break
                    connection.close()
            connection.close()


class FootprintLayerClient:
    """
    This class is responsible for connecting to the footprint server via TCP.
    ClassAttributes:
        TCP_IP (str): the host of the server
        TCP_PORT (int): the port the server is on
    """
    TCP_IP = '127.0.0.1'
    TCP_PORT = 8080

    @classmethod
    def _get_socket(cls) -> socket.socket:
        """
        Gets the socket using the cls.TCP_IP and cls.TCP_PORT.

        Returns: (socket.socket) a connected socket
        """
        current_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        current_socket.connect((cls.TCP_IP, cls.TCP_PORT))  # this is where it could be hanging
        current_socket.settimeout(10)
        return current_socket

    @classmethod
    def _register(cls) -> None:
        current_socket = cls._get_socket()
        data: bytes = OperationEnum.REGISTER.value
        current_socket.sendall(data)
        current_socket.close()

    @classmethod
    def register(cls, static_path: str) -> None:
        try:
            cls._register()
        except ConnectionRefusedError:
            footprint_layer = FootprintLayer(static_path=static_path)
            server_process = Process(target=footprint_layer.listen)
            server_process.start()
            cls._register()

    @classmethod
    def unregister(cls) -> None:
        current_socket = cls._get_socket()
        data: bytes = OperationEnum.UNREGISTER.value
        current_socket.sendall(data)
        current_socket.close()

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

        number_of_chunks: bytes = current_socket.recv(32)
        number_of_chunks: int = int.from_bytes(number_of_chunks, 'big')

        raw_data_buffer: List[bytes] = []
        for _ in range(number_of_chunks):
            raw_data_buffer.append(current_socket.recv(500))

        return pickle.loads(b"".join(raw_data_buffer))


def _shutdown_socket(running_socket: socket.socket):
    print("the shutdown function is firing")
    running_socket.close()


def main():
    import atexit
    test = FootprintLayer("/home/maxwellflitton/Documents/github/oasislmf-get-model-testing/data/100/static/")
    atexit.register(_shutdown_socket, test.socket)
    test.listen()


if __name__ == "__main__":
    main()

