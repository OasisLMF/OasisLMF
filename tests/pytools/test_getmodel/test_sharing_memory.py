from unittest import main, TestCase

# from oasislmf.pytools.getmodel.get_model import get_items, get_vulns, Footprint
from oasislmf.pytools.getmodel.sharing_memory import SharedMemoryManager, DataEnum
from oasislmf.pytools.getmodel.manager import get_items, get_vulns, Footprint
from mmap import mmap

# import numpy as np
# import numba as nb
# import pyarrow.parquet as pq


class SharedMemoryTests(TestCase):

    def setUp(self) -> None:
        self.test_name: DataEnum = DataEnum.VULNERABILITY

    def test_shared_ids(self):
        vulns_dict = get_items(input_path="./")[0]

        test_one = SharedMemoryManager(name=self.test_name, static_path="./", vuln_dict=vulns_dict)
        vuln_array_pointer_one, vulns_id_pointer_two = test_one.pointers

        test_two = SharedMemoryManager(name=self.test_name, static_path="./", vuln_dict=vulns_dict)
        vuln_array_pointer_one_two, vulns_id_pointer_two_two = test_one.pointers

        # asserts that we have mmaps from
        self.assertEqual(mmap, type(vuln_array_pointer_one.buf.obj))
        self.assertEqual(mmap, type(vulns_id_pointer_two.buf.obj))
        self.assertEqual(mmap, type(vuln_array_pointer_one_two.buf.obj))
        self.assertEqual(mmap, type(vulns_id_pointer_two_two.buf.obj))

        self.assertEqual(True, test_one.loaded_from_file)
        self.assertEqual(False, test_two.loaded_from_file)

    def test_pointer(self):
        test = SharedMemoryManager(name=self.test_name, size=20)
        outcome = test.get_reference
        print(test)


if __name__ == "__main__":
    main()
