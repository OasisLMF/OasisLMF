from os import path
from typing import Optional, Any, Dict

from pandas import DataFrame, read_csv

from .enums import FileTypeEnum
from .errors import NotSupportedError


class FileLoader:
    """
    This class is responsible for reading and writing data to files.

    Attributes:
        path (str): path to the file that we are either reading or writing to
        label (str): description of what type of data is being processed for error raising
        extension (str): the file extension that belongs to the file (used for mapping functions in the self.READ_MAP)
    """
    READ_MAP: Dict[str, Any] = {
        FileTypeEnum.CSV.value: (read_csv, "to_csv"),
        FileTypeEnum.BIN.value: (read_csv, "to_csv")
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
        self._read_function: Any = self.get_read_function()
        self._value: Optional[DataFrame] = None

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
        return self._value
