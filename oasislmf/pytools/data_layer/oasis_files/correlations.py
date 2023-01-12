"""
This file defines the loading and saving for correlations data.
"""
from typing import Optional

import numpy as np
import pandas as pd

from oasislmf.pytools.getmodel.common import Correlation


class CorrelationsData:
    """
    This class is responsible for managing the loading and saving of correlation data from binary and CSV files.

    Attributes:
        data (Optional[pd.DataFrame): correlation data that is either loaded or saved
    """
    COLUMNS = ["item_id", "peril_correlation_group", "correlation_value"]

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        The constructor for the CorrelationsData class.

        Args:
            data: (Optional[pd.DataFrame] default is None but if supplied must have the following columns:
                                          [item_id, peril_correlation_group, correlation_value]
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
        data = pd.DataFrame(np.fromfile(file_path, dtype=Correlation))
        data["item_id"] = list(range(1, len(data) + 1))
        data = data[["item_id", "peril_correlation_group", "correlation_value"]]
        return CorrelationsData(data=data)

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
        data = np.array(list(self.data.drop("item_id", axis=1).itertuples(index=False)), dtype=Correlation)
        data.tofile(file_path)
