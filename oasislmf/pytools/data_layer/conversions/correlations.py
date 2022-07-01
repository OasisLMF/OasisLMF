"""
This file defines the functions that convert correlations.
"""
import os

from oasislmf.pytools.data_layer.oasis_files.correlations import CorrelationsData


def convert_csv_to_bin(file_path: str) -> None:
    """
    Converts the correlations data from a CSV file to a binary file.

    Args:
        file_path: (str) path to the file being converted

    Returns: None
    """
    data = CorrelationsData.from_csv(file_path=file_path)
    data.to_bin(file_path=file_path.replace("csv", "bin"))


def convert_bin_to_csv(file_path: str) -> None:
    """
    Converts the correlations data from a binary file to a CSV file.

    Args:
        file_path: (str) path to the file being converted

    Returns: None
    """
    data = CorrelationsData.from_bin(file_path=file_path)
    data.to_csv(file_path=file_path.replace("bin", "csv"))


def convert_csv_to_bin_main() -> None:
    """
    The command line entry point for converting CSV correlations file to a binary file.

    Returns: None
    """
    file_path = str(os.getcwd()) + "/correlations.csv"
    convert_csv_to_bin(file_path=file_path)


def convert_bin_to_csv_main() -> None:
    """
    The command line entry point for converting binary correlations file to a CSV file.

    Returns: None
    """
    file_path = str(os.getcwd()) + "/correlations.bin"
    convert_bin_to_csv(file_path=file_path)
