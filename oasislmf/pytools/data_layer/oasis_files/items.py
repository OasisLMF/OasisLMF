"""
This file defines the loading and saving for items data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from oasislmf.pytools.getmodel.common import Item

import logging

logger = logging.getLogger(__name__)


class ItemsData:
    """
    This class is responsible for managing the loading and saving of correlation data from binary and CSV files.

    Attributes:
        data (Optional[pd.DataFrame): correlation data that is either loaded or saved
    """
    COLUMNS = ["item_id", "peril_correlation_group", "damage_correlation_value"]

    def __init__(self, data: Optional[pd.DataFrame] = None) -> None:
        """
        The constructor for the ItemsData class.

        Args:
            data: (Optional[pd.DataFrame] default is None but if supplied must have the following columns:
                                          [item_id, peril_correlation_group, damage_correlation_value]
        """
        self.data: Optional[pd.DataFrame] = data

    def read(self, input_path: str | Path, ignore_file_type: set, inplace=True) -> None:
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
            self.items = np.memmap(items_fname, dtype=Item, mode='r')

        elif "items.csv" in input_files and "csv" not in ignore_file_type:
            items_fname = os.path.join(input_path, 'items.csv')

            if inplace:
                self.data = self.from_csv(os.path.join(input_path, 'items.csv'))
            else:

        else:
            raise FileNotFoundError(f'items file not found at {input_path}')

    @staticmethod
    def from_csv(items_fname: str):
        """
        Loads items data from a CSV file.

        Args:
            file_path: (str) the path to the CSV file housing the data

        Returns: (ItemsData) the loaded data from the CSV file
        """
        logger.debug(f"loading {items_fname}")

        return ItemsData(np.loadtxt(items_fname, dtype=Item, delimiter=",", skiprows=1, ndmin=1))

    def sort(self, by: str | List[str]):
        self.data = np.sort(self.data, order=by)

    @staticmethod
    # @njit(cache=True, fastmath=True)
    def generate_item_map(items, coverages):
        """Generate item_map; requires items to be sorted.

        Args:
            items (numpy.ndarray[int32, int32, int32]): 1-d structured array storing
            `item_id`, `coverage_id`, `group_id` for all items.
            items need to be sorted by increasing areaperil_id, vulnerability_id
            in order to output the items in correct order.

        Returns:
            item_map (Dict[ITEM_MAP_KEY_TYPE, ITEM_MAP_VALUE_TYPE]): dict storing
            the mapping between areaperil_id, vulnerability_id to item.
            areaperil_ids_map (Dict[int, Dict[int, int]]) dict storing the mapping between each
            areaperil_id and all the vulnerability ids associated with it.
        """
        item_map = Dict.empty(ITEM_MAP_KEY_TYPE, List.empty_list(ITEM_MAP_VALUE_TYPE))
        Nitems = items.shape[0]

        areaperil_ids_map = Dict.empty(nb_areaperil_int, Dict.empty(nb_int32, nb_int64))

        for j in range(Nitems):
            append_to_dict_value(
                item_map,
                tuple((items[j]['areaperil_id'], items[j]['vulnerability_id'])),
                tuple((items[j]['id'], items[j]['coverage_id'], items[j]['group_id'])),
                ITEM_MAP_VALUE_TYPE
            )
            coverages[items[j]['coverage_id']]['max_items'] += 1

            if items[j]['areaperil_id'] not in areaperil_ids_map:
                areaperil_ids_map[items[j]['areaperil_id']] = {items[j]['vulnerability_id']: 0}
            else:
                areaperil_ids_map[items[j]['areaperil_id']][items[j]['vulnerability_id']] = 0

        return item_map, areaperil_ids_map

    @staticmethod
    def from_bin(file_path: str) -> "ItemsData":
        """
        Loads correlations data from a binary file.

        Args:
            file_path: (str) the path to the binary file housing the data

        Returns: (ItemsData) the loaded data from the binary file
        """
        data = pd.DataFrame(np.fromfile(file_path, dtype=Item))
        data["item_id"] = list(range(1, len(data) + 1))
        data = data[["item_id", "peril_correlation_group", "damage_correlation_value"]]
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
        data = np.array(list(self.data.drop("item_id", axis=1).itertuples(index=False)), dtype=Item)
        data.tofile(file_path)
