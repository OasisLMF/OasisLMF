"""
This file tests the loading, saving, and conversions for items data to and from CSV and binary files.
"""
import os
from unittest import TestCase, main

import pandas as pd

from oasislmf.pytools.data_layer.oasis_files.items import Item, ItemsData


class TestItems(TestCase):

    @staticmethod
    def file_path(file_name: str) -> str:
        return os.path.dirname(os.path.realpath(__file__)) + f"/meta_data/{file_name}"

    @staticmethod
    def _wipe_file(file_path: str) -> None:
        if os.path.exists(path=file_path):
            os.remove(file_path)

    def setUp(self) -> None:
        self.data_path: str = TestItems.file_path(file_name="items.csv")
        self.bin_path: str = TestItems.file_path(file_name="items.bin")
        self.data: pd.DataFrame = pd.read_csv(self.data_path, dtype=Item)
        self.test: ItemsData = ItemsData(data=self.data)

        self.cache_path: str = TestItems.file_path(file_name="cache.csv")
        self.cache_bin_path: str = TestItems.file_path(file_name="cache.bin")
        self.check_file_path: str = TestItems.file_path(file_name="check.csv")

    def tearDown(self) -> None:
        self._wipe_file(file_path=self.cache_path)
        self._wipe_file(file_path=self.cache_bin_path)
        self._wipe_file(file_path=self.check_file_path)

    def test___init__(self):
        test_one = ItemsData()
        test_two = ItemsData(data=self.data)

        self.assertEqual(None, test_one.data)
        self.assertTrue(test_two.data.equals(self.data))

    def test_from_csv(self):
        test: ItemsData = ItemsData.from_csv(file_path=self.data_path)

        self.assertIsInstance(test, ItemsData)
        self.assertTrue(test.data.equals(self.data))

    def test_to_csv(self):
        self.test.to_csv(file_path=self.cache_path)
        data: pd.DataFrame = pd.read_csv(self.data_path, dtype=Item)
        self.assertTrue(data.equals(self.test.data))

    def test_from_bin(self):
        test = ItemsData.from_bin(file_path=self.bin_path)
        test.data.to_csv(self.check_file_path, index=False)

        check_data = pd.read_csv(self.check_file_path, dtype=Item)
        self.assertTrue(check_data.equals(self.data))

    def test_to_bin(self):
        self.test.to_bin(file_path=self.cache_bin_path)

        test_one = ItemsData.from_bin(file_path=self.cache_bin_path)
        test_one.data.to_csv(self.check_file_path, index=False)
        test_one.to_csv(file_path=self.check_file_path)

        test_two = ItemsData.from_csv(file_path=self.check_file_path)
        self.assertTrue(self.data.equals(test_two.data))


if __name__ == "__main__":
    main()
