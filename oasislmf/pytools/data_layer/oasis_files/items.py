"""
This file defines the loading and saving for items data.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import numba as nb
import numpy as np
import pandas as pd

from oasislmf.pytools.common import areaperil_int

logger = logging.getLogger(__name__)


Item = nb.from_dtype(np.dtype([('item_id', np.int32),
                               ('coverage_id', np.int32),
                               ('areaperil_id', areaperil_int),
                               ('vulnerability_id', np.int32),
                               ('group_id', np.int32),
                               ('hazard_group_id', np.int32)
                               ]))

Item_legacy = nb.from_dtype(np.dtype([('item_id', np.int32),
                                      ('coverage_id', np.int32),
                                      ('areaperil_id', areaperil_int),
                                      ('vulnerability_id', np.int32),
                                      ('group_id', np.int32)
                                      ]))


class ItemsData:
    """
    This class is responsible for managing the loading and saving of items data from binary and CSV files.

    Attributes:
        data (Optional[pd.DataFrame): items data that is either loaded or saved
    """
    COLUMNS = list(Item.dtype.names)
    COLUMNS_legacy = list(Item_legacy.dtype.names)

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        The constructor for the ItemsData class.

        Args:
            data: (Optional[pd.DataFrame] default is None but if supplied must have the columns listed in ItemsData.COLUMNS.
        """
        self.data: Optional[pd.DataFrame] = data

    def sort(self, by: str | List[str]) -> None:
        self.data.sort_values(by, inplace=True)

    @staticmethod
    def from_csv(file_path: str):
        """
        Loads items data from a CSV file.

        Args:
            file_path: (str) the path to the CSV file housing the data

        Returns: (ItemsData) the loaded data from the CSV file
        """
        try:
            data = pd.read_csv(file_path, dtype=Item)
        except ValueError:
            data = pd.read_csv(file_path, dtype=Item_legacy)

        return ItemsData(data)

    @staticmethod
    def from_bin(file_path: str, legacy: bool = False) -> "ItemsData":
        """
        Loads correlations data from a binary file.

        Args:
            file_path: (str) the path to the binary file housing the data
            legacy: (bool) if True, it uses the Item_legacy definition.

        Returns: (ItemsData) the loaded data from the binary file.

        Note:
          There is no way to automatically infer whether a binary file uses the legacy definition or not.
          Therefore, it must be specified by the user.
        """
        if legacy:
            data = pd.DataFrame(np.fromfile(file_path, dtype=Item_legacy), columns=ItemsData.COLUMNS_legacy)
        else:
            data = pd.DataFrame(np.fromfile(file_path, dtype=Item), columns=ItemsData.COLUMNS)

        return ItemsData(data=data)

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
        try:
            data = np.array(list(self.data.itertuples(index=False)), dtype=Item)
        except ValueError:
            data = np.array(list(self.data.itertuples(index=False)), dtype=Item_legacy)

        data.tofile(file_path)


def read_items(input_path, ignore_file_type=set()):
    """Load the items from the items file.

    Args:
        input_path (str): the path pointing to the file
        ignore_file_type (Set[str]): file extension to ignore when loading.

    Returns:
        Tuple[Dict[int, int], List[int], Dict[int, int], List[Tuple[int, int]], List[int]]
          vulnerability dictionary, vulnerability IDs, areaperil to vulnerability index dictionary,
          areaperil ID to vulnerability index array, areaperil ID to vulnerability array
    """
    input_files = set(os.listdir(input_path))

    if "items.bin" in input_files and "bin" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.bin')
        logger.debug(f"loading {items_fname}")
        items = np.memmap(items_fname, dtype=Item, mode='r')

    elif "items.csv" in input_files and "csv" not in ignore_file_type:
        items_fname = os.path.join(input_path, 'items.csv')
        logger.debug(f"loading {items_fname}")
        items = np.loadtxt(items_fname, dtype=Item, delimiter=",", skiprows=1, ndmin=1)

    else:
        raise FileNotFoundError(f'items file not found at {input_path}')

    return items
