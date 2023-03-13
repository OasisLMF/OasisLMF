"""
This file defines the loading and saving for correlations data.
"""
import logging
import os
from typing import Optional

import numba as nb
import numpy as np
import pandas as pd

from oasislmf.pytools.common import oasis_float

logger = logging.getLogger(__name__)


Correlation = nb.from_dtype(np.dtype([
    ('item_id', np.int32),
    ("peril_correlation_group", np.int32),
    ("damage_correlation_value", oasis_float),
    ("hazard_group_id", np.int32),
    ("hazard_correlation_value", oasis_float)
]))


class CorrelationsData:
    """
    This class is responsible for managing the loading and saving of correlation data from binary and CSV files.

    Attributes:
        data (Optional[pd.DataFrame): correlation data that is either loaded or saved
    """
    COLUMNS = list(Correlation.dtype.names)

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        The constructor for the CorrelationsData class.

        Args:
            data: (Optional[pd.DataFrame] default is None but if supplied must have the columns listed in CorrelationsData.COLUMNS.
        """
        self.data: Optional[pd.DataFrame] = data

    @staticmethod
    def from_csv(file_path: str) -> "CorrelationsData":
        """
        Loads correlations data from a CSV file.

        Args:
            file_path: (str) the path to the CSV file housing the data

        Returns: (CorrelationsData) the loaded data from the CSV file
        """
        return CorrelationsData(data=pd.read_csv(file_path))

    @staticmethod
    def from_bin(file_path: str) -> "CorrelationsData":
        """
        Loads correlations data from a binary file.

        Args:
            file_path: (str) the path to the binary file housing the data

        Returns: (CorrelationsData) the loaded data from the binary file
        """
        return CorrelationsData(data=pd.DataFrame(np.fromfile(file_path, dtype=Correlation)))

    def to_csv(self, file_path: str) -> None:
        """
        Writes self.data to a CSV file.

        Args:
            file_path: (str) the file path for the CSV file to be written to

        Returns: None
        """
        self.data.to_csv(file_path, index=False)

    def to_bin(self, file_path: str) -> None:
        """
        Writes self.data to a binary file.

        Args:
            file_path: (str) the file path for the binary file to be written to

        Returns: None
        """
        data = np.array(list(self.data.itertuples(index=False)), dtype=Correlation)
        data.tofile(file_path)


def read_correlations(input_path, ignore_file_type=set()):
    """Load the correlations from the correlations file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
        vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
        areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))

    if "correlations.bin" in input_files and "bin" not in ignore_file_type:
        correlations_fname = os.path.join(input_path, 'correlations.bin')
        logger.debug(f"loading {correlations_fname}")

        try:
            correlations = np.memmap(correlations_fname, dtype=Correlation, mode='r')

        except ValueError:
            logger.debug("binary file is empty, numpy.memmap failed. trying to read correlations.csv.")
            correlations = read_correlations(input_path, ignore_file_type={'bin'})

    elif "correlations.csv" in input_files and "csv" not in ignore_file_type:
        correlations_fname = os.path.join(input_path, 'correlations.csv')
        logger.debug(f"loading {correlations_fname}")
        correlations = np.loadtxt(correlations_fname, dtype=Correlation, delimiter=",", skiprows=1, ndmin=1)

    else:
        raise FileNotFoundError(f'correlations file not found at {input_path}')

    return correlations
