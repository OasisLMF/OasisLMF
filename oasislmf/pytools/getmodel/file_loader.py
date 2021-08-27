import _io
from os import path
from typing import Optional, Any, Dict, List, Tuple, Union
import struct

from pandas import DataFrame, read_csv, read_parquet

from .enums import FileTypeEnum
from .errors import NotSupportedError


class FileLoader:
    """
    This class is responsible for reading and writing data to files.

    Attributes:
        path (str): path to the file that we are either reading or writing to
        label (str): description of what type of data is being processed for error raising
        extension (str): the file extension that belongs to the file (used for mapping functions in the self.READ_MAP)
        file_name (str): the name of the file being loaded
    """
    READ_MAP: Dict[str, Any] = {
        FileTypeEnum.CSV.value: (read_csv, "to_csv"),
        FileTypeEnum.BIN.value: (open, "to_csv"),
        FileTypeEnum.PARQUET.value: (read_parquet, "to_parquet")
    }

    def __init__(self, file_path: str, label: str) -> None:
        """
        The constructor for the FileLoader class.

        Args:
            file_path: (str) path to the file that we are either reading or writing to.
            label: (str) description of what type of data is being processed for error raising
        """
        self.path: str = file_path
        self.label: str = label
        self.extension: str = file_path.split(".")[-1]
        self.file_name: str = file_path.split(".")[-2].split("/")[-1]
        self._read_function: Any = self.get_read_function()
        self._value: Optional[DataFrame] = None

    def _process_bytes(self, data: bytes) -> DataFrame:
        """
        Takes in bytes, processes them, and returns them as a DataFrame based on the profiles for the bin files. The
        file profile is selected using the self.file_name.

        Args:
            data: (bytes) data to be processed, usually from a .bin file. Please refer to the encoding_map to see
                          what files are supported, the keys in the encoding_map are the names of the files supported.

        Returns: (DataFrame) with data from the bin file.
        """
        encoding_map = {
            "vulnerability": {
                "struct": struct.Struct("iiif"),
                "byte chunk": 16,
                "columns": ["vulnerability_id", "intensity_bin_index", "damage_bin_index", "prob"]
            },
            "footprint": {
                "struct": struct.Struct("iiif"),
                "byte chunk": 16,
                "columns": ["event_id", "areaperil_id", "intensity_bin_index", "prob"]
            },
            "damage_bin_dict": {
                "struct": struct.Struct("ifff"),
                "byte chunk": 16,
                "columns": ["bin_index", "bin_from", "bin_to", "interpolation"]
            },
            "events": {
                "struct": struct.Struct("i"),
                "byte chunk": 4,
                "columns": ["event_id"]
            }
        }
        chunk: int = encoding_map[self.file_name]["byte chunk"]
        unpack_struct: struct.Struct = encoding_map[self.file_name]["struct"]
        raw_data: List[bytes] = [data[i:i + chunk] for i in range(0, len(data), chunk)]

        buffer = []
        for i in range(0, len(raw_data)):
            try:
                buffer.append(unpack_struct.unpack(raw_data[i]))
            except struct.error:
                break
        return DataFrame(buffer, columns=encoding_map[self.file_name]["columns"])

    def get_read_function(self) -> Any:
        """
        Uses the self.extension to get the correct read function from the self.READ_MAP.

        Returns: (Any) the read function required to read data from the file
        """
        read_function: Optional[Any] = self.READ_MAP.get(self.extension)
        if read_function is None:
            raise NotSupportedError(message=f"{self.extension} not supported")
        return read_function[0]

    def read(self) -> DataFrame:
        """
        Reads data from the file.

        Returns: (DataFrame) data from the file
        """
        if not path.isfile(path=self.path):
            raise FileNotFoundError(f"{self.label} cannot be found under the path: {self.path}")
        return self._read_function(self.path)

    def clear_cache(self) -> None:
        """
        Wipes the self._value so the file can be read again.

        Returns: None
        """
        del self._value
        self._value = None

    @property
    def value(self) -> DataFrame:
        """
        Gets data from the file that the self.path is pointing to.

        Returns: (DataFrame) data from file
        """
        if self._value is None:
            self._value = self.read()
            if isinstance(self._value, _io.TextIOWrapper):
                with open(self.path, 'rb') as file:
                    bytes_data = file.read()
                self._value = self._process_bytes(data=bytes_data)
        return self._value

    @value.setter
    def value(self, value) -> None:
        self._value = value
