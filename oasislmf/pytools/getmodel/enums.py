from enum import Enum


class FileTypeEnum(Enum):
    """
    Defines the type of files supported.
    """
    CSV = "csv"
    BIN = "bin"
    PARQUET = "parquet"


class LoadingTypes(Enum):
    """
    Defines the type of storage mechanisms supported.
    """
    FILE = "file"
