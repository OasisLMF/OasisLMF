from unittest import main, TestCase
from mock import patch

from oasislmf.pytools.getmodel.get_model import get_items, load_items, get_vulns
from oasislmf.pytools.getmodel.vulnerability import vulnerability_to_parquet, Vulnerability, get_array, iter_table
from oasislmf.pytools.getmodel.common import oasis_float
from numpy.testing import assert_almost_equal

import numpy as np
import numba as nb
import pyarrow as pa
import pyarrow.parquet as pq
import os
import pickle
import json


@nb.jit(cache=True, fastmath=True)
def reconstruct_parquet_vulnerability(flat_array, number_of_vulnerability_ids):
    placeholder = np.zeros((50, 50, 50), dtype=oasis_float)
    counter = 0
    while counter < flat_array.shape[0]:
        placeholder[counter] = np.reshape(flat_array[counter], (-1, number_of_vulnerability_ids))
        counter += 1
    return placeholder


class GetModelTests(TestCase):

    def test_load_items(self):
        vulns_dict = get_items(input_path="./", file_type="bin")[0]
        first_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50, file_type="bin")[0]
        parquet_handle = pq.ParquetDataset(os.path.join("./static/", "vulnerability.parquet"), use_legacy_dataset=False,
                                           filters=[("vulnerability_id", "in", list(vulns_dict.keys()))])
        vuln_table = parquet_handle.read()
        vuln_array = vuln_table[1].to_numpy()
        outcome = reconstruct_parquet_vulnerability(vuln_array, 50)
        self.assertEqual(outcome.all(), first_outcome.all())
        # assert_almost_equal(outcome, first_outcome)

    def test_get_vulns(self):
        vulns_dict = get_items(input_path="./", file_type="bin")[0]
        first_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50, file_type="bin")

        # TODO => ask what the dimensions map to (50, 50, 50) => (x, y, z)
        # vulnerability array has a size of 125,000 and a shape of (50, 50, 50)
        vulnerability_array = first_outcome[0]

        second_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50,
                                   file_type="parquet")

        self.assertEqual(second_outcome[0].size, vulnerability_array.size)
        self.assertEqual(second_outcome[0].all(), vulnerability_array.all())
        np.save("./old", vulnerability_array)
        np.save("./new", second_outcome[0])

    def test_filtering(self):
        outcome = get_items(input_path="./", file_type="bin")

        # use legacy dataset set to True enables us to pass filters to all columns
        table2 = pq.ParquetDataset("./vulnerability_dataset/part_0.parquet", use_legacy_dataset=False,
                                   filters=[("vulnerability_id", "in", list(outcome[0].keys()))])

        table2 = table2.read()
        meta = table2.schema.metadata
        number_of_vulnerability_ids = int(meta[b"num_vulnerability_id"].decode("utf-8"))
        number_of_intensity_bins = int(meta[b"num_intensity_bins"].decode("utf-8"))
        number_of_damage_bins = int(meta[b"num_damage_bins"].decode("utf-8"))
        print(f"here is the length of the data {len(table2)}")
        # print(f"here is the first column: {table2[0]}")
        print(f"here is the first column: {len(table2[0])}")
        # print(f"here is the second column: {table2[1][0]}")
        print(f"here is the second column: {len(table2[1])}")
        print(f"here is the third column: {len(table2[1][0])}")
        print(f"here is the fourth column: {table2[1][0][0]}")
        print(table2)


if __name__ == "__main__":
    main()
