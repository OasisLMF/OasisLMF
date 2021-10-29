import struct
from enum import Enum
from typing import List
import zlib


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CompressionEnum(Enum):
    NO_HAZARD_UNCERTAINTY = 0
    HAS_HAZARD_UNCERTAINTY = 1
    INDEX_FILE_HAS_UNCOMPRESSED_SIZE = 2
    INDEX_FILE_HAS_UNCOMPRESSED_SIZE_AND_THERE_IS_HAZARD_UNCERTAINTY = 3


class FootprintIndexBinReader(metaclass=Singleton):
    """
    This class is responsible for loading binary files for the footprint index.

    Attributes:
        footprint_path (str): the path pointing to the footprint file
        path (str): the path pointing to the footprint index file
        chunk_size (int): the number of bytes being read per row
        header_size (int): the number of bytes the header takes up
        compression (bool): True if the file is compressed, False if not
        number_of_intensity_bins (int): the number of intensity bins the data has
        compression_type (CompressionEnum): the type of compression the file has
    """
    def __init__(self, footprint_path: str, path: str) -> None:
        """
        The constructor for the FootprintIndexBinReader class.

        Args:
            footprint_path: (str) the path pointing to the footprint file
            path: (str) the path pointing to the footprint index file
        """
        self.footprint_path = footprint_path
        self.path = path
        self.chunk_size = 20
        self.header_size = 8
        self.compression = False
        header = self.header
        self.number_of_intensity_bins = header["number of intensity bins"]
        self.compression_type = self.map_compression(header["compression type"])
        self.map_chunk_size()

    def map_chunk_size(self) -> None:
        """
        Maps the type of compression updating the self.chunk_size if needed.

        Returns: None
        """
        compressed_statuses = [
            CompressionEnum.INDEX_FILE_HAS_UNCOMPRESSED_SIZE,
            CompressionEnum.INDEX_FILE_HAS_UNCOMPRESSED_SIZE_AND_THERE_IS_HAZARD_UNCERTAINTY
        ]
        if self.compression_type in compressed_statuses:
            self.chunk_size = 28
            self.compression = True

    def read(self) -> tuple:
        """
        Generator reading the binary file.

        Returns: (Tuple) compressed => event_id, offset, size, uncompressed_size
                         uncompressed => event_id, offset, size
        """
        with open(self.path, "rb") as file:
            data = "placeholder"
            while data:
                data = file.read(self.chunk_size)
                if data is None:
                    break
                yield self.process_data(data)

    def read_slice(self, number: int):
        pass

    def process_data(self, data: bytes) -> tuple:
        """
        Processes a chunk of bytes usually read from the file.

        Args:
            data: (bytes) data read from the

        Returns: (Tuple) compressed => event_id, offset, size, uncompressed_size
                         uncompressed => event_id, offset, size
        """
        event_id = int.from_bytes(data[:4], "little")
        offset = int.from_bytes(data[4:12], "little")
        size = int.from_bytes(data[12:20], "little")
        if self.compression is True:
            uncompressed_size = int.from_bytes(data[20:28], "little")
            return event_id, offset, size, uncompressed_size
        return event_id, offset, size

    @staticmethod
    def map_compression(compression_type: int) -> CompressionEnum:
        """
        Maps the compression type based off the compression type number which is usually extracted from the header.

        Args:
            compression_type: (int) the type of compression to be mapped

        Returns: (CompressionEnum) compression type
        """
        return CompressionEnum(compression_type)

    @property
    def header(self) -> dict:
        placeholder = {}
        with open(self.footprint_path, "rb") as file:
            data = file.read(8)
        placeholder["number of intensity bins"] = int.from_bytes(data[:4], "little")
        placeholder["compression type"] = int.from_bytes(data[4:8], "little")
        return placeholder

    @property
    def zipped(self) -> bool:
        if self.path[-2:] == ".z":
            return True
        return False


class FootprintReader(metaclass=Singleton):
    """
    This class is responsible reading the footprint file.

    Attributes:
        path (str): path pointing to the footprint binary file
        chunk_size (int): size of chunks for each row of the file

    How to use:
        footprint_index = FootprintIndexBinReader("./static/footprint.bin", "./static/footprint.idx")
        footprint = FootprintReader("./footprint.bin", 12)

        generator = footprint.read_slices()
        counter = 0
        for i in footprint_index.read():
            next(generator)
            event_id = i[0]
            start = i[1]
            size = i[2]
            read_data = generator.send(size)
    """
    def __init__(self, path: str, chunk_size: int = 12) -> None:
        """
        The constructor of the FootprintReader class.

        Args:
            path: (str) path pointing to the footprint binary file
            chunk_size: (int) size of chunks for each row of the file
        """
        self.path = path
        self.chunk_size = chunk_size

    @staticmethod
    def process_data(data: bytes) -> tuple:
        """
        Processes a chunk of bytes usually read from the file.

        Args:
            data: (bytes) data read from the

        Returns: (tuple) areaperil_id, intensity_bin_id, probability
        """
        areaperil_id = int.from_bytes(data[:4], "little")
        intensity_bin_id = int.from_bytes(data[4:8], "little")
        probability = struct.unpack('f', data[8:12])[0]

        return areaperil_id, intensity_bin_id, probability

    @staticmethod
    def process_compressed_data(data: bytes, uncompressed_size: int) -> tuple:
        data = zlib.decompress(data, bufsize=uncompressed_size)
        areaperil_id = int.from_bytes(data[:4], "little")
        intensity_bin_id = int.from_bytes(data[4:8], "little")
        probability = struct.unpack('f', data[8:12])[0]

        return areaperil_id, intensity_bin_id, probability

    def read_slices(self) -> List[tuple]:
        """
        This is a generator that reads data for each event_id.
        To activate the generator, you will have to send the size of the chunk into the generator.

        Returns: (List[tuple]) processed data from the data slice
        """
        with open(self.path, "rb") as file:
            data: bytes = file.read(8)
            while data:
                size = yield
                data = file.read(size)
                if data is None:
                    break
                chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
                yield [self.process_data(i) for i in chunks]

    def read_compressed_slices(self) -> List[tuple]:
        """
        This is a generator that reads data for each event_id.
        To activate the generator, you will have to send the size of the chunk into the generator.

        Returns: (List[tuple]) processed data from the data slice
        """
        with open(self.path, "rb") as file:
            data: bytes = file.read(8)
            while data:
                size = yield
                data = file.read(size)
                if data is None:
                    break
                chunks = [data[i:i + self.chunk_size] for i in range(0, len(data), self.chunk_size)]
                yield [self.process_compressed_data(i, 60000) for i in chunks]

    def read(self) -> tuple:
        """
        This is a generator that produces a row from the

        Returns: (tuple) processed row from the file
        """
        with open(self.path, "rb") as file:
            data = file.read(8)
            while data:
                data = file.read(self.chunk_size)
                if data is None:
                    break
                yield self.process_data(data)
