"""
This file defines the functions that convert correlations.
"""
import sys
from pathlib import Path

from oasislmf.pytools.data_layer.oasis_files.correlations import CorrelationsData


def convert_csv_to_bin(file_path: str, file_out_path: str = "") -> None:
    """
    Converts the correlations data from a CSV file to a binary file.

    Args:
        file_path: (str) path to the file being converted

    Returns: None
    """
    CorrelationsData.from_csv(file_path=file_path).to_bin(
        file_path=Path(file_path).with_suffix(".bin") if not file_out_path else file_out_path
    )


def convert_bin_to_csv(file_path: str, file_out_path: str = "") -> None:
    """
    Converts the correlations data from a binary file to a CSV file.

    Args:
        file_path: (str) path to the file being converted

    Returns: None
    """
    CorrelationsData.from_bin(file_path=file_path).to_csv(file_out_path if file_out_path else sys.stdout.buffer)
