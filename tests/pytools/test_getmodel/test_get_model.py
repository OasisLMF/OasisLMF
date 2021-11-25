from unittest import main, TestCase

from oasislmf.pytools.getmodel.manager import get_items, get_vulns, Footprint

import numpy as np
import numba as nb
import pyarrow.parquet as pq
from pyarrow import memory_map


class GetModelTests(TestCase):
    """
    This class can be used to test the get model functions in a testing environment. Running this is an effective way
    for isolated development. Patches and mocks can be used to speed up the development however, the get model is
    functional so mocking is not essential. The functions below are commented out as they will automatically run in
    an continuous integration. However, they can be uncommented and used for debugging and development. The example
    data in this directory has the following config:

    {
        "num_vulnerabilities": 50,
        "num_intensity_bins": 50,
        "num_damage_bins": 50,
        "vulnerability_sparseness": 0.5,
        "num_events": 500,
        "num_areaperils": 100,
        "areaperils_per_event": 100,
        "intensity_sparseness": 0.5,
        "num_periods": 1000,
        "num_locations": 1000,
        "coverages_per_location": 3,
        "num_layers": 1
    }

    The data was generated using the oasislmf-get-model-testing repo and has the following formats:

    CSV => all files
    bin => all files
    parquet => vulnerability only
    """
    def test_init(self):
        pass

    # def test_load_parquet(self):
    #     vulns_dict = get_items(input_path="./")[0]
    #     parquet_handle = pq.ParquetDataset("./vulnerability_dataset/part_0.parquet",
    #                                        use_legacy_dataset=False,
    #                                        filters=[("vulnerability_id", "in", list(vulns_dict))],
    #                                        memory_map=True)
    #     vuln_table = parquet_handle.read()
    #     vuln_meta = vuln_table.schema.metadata
    #     num_damage_bins = int(vuln_meta[b"num_damage_bins"].decode("utf-8"))
    #     number_of_intensity_bins = int(vuln_meta[b"num_intensity_bins"].decode("utf-8"))
    #     vuln_array = np.vstack(vuln_table['vuln_array'].to_numpy()).reshape(vuln_table['vuln_array'].length(),
    #                                                                         num_damage_bins,
    #                                                                         number_of_intensity_bins)
    #     vulns_id = vuln_table['vulnerability_id'].to_numpy()
    #
    #     parquet_handle_two = pq.ParquetDataset("./vulnerability_dataset/part_0.parquet",
    #                                        use_legacy_dataset=False,
    #                                        filters=[("vulnerability_id", "in", list(vulns_dict))],
    #                                        memory_map=True)
    #     vuln_table_two = parquet_handle.read()
    #     vuln_meta_two = vuln_table_two.schema.metadata
    #     num_damage_bins_two = int(vuln_meta[b"num_damage_bins"].decode("utf-8"))
    #     number_of_intensity_bins_two = int(vuln_meta[b"num_intensity_bins"].decode("utf-8"))
    #     vuln_array_two = np.vstack(vuln_table_two['vuln_array'].to_numpy()).reshape(vuln_table_two['vuln_array'].length(),
    #                                                                         num_damage_bins_two,
    #                                                                         number_of_intensity_bins_two)
    #     vulns_id_two = vuln_table['vulnerability_id'].to_numpy()
    #     print(id(vuln_table))
    #     print(id(vuln_table_two))

    # def test_load_footprint(self):
    #     with Footprint.load(static_path="./static/") as test:
    #         outcome = test
    #     print(outcome)

    # def test_update(self):
    #     vulns_dict = get_items(input_path="./", file_type="bin")[0]
    #     second_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50,
    #                                file_type="parquet")


    # def test_get_vulns(self):
    #     vulns_dict = get_items(input_path="./")[0]
    #     first_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50,
    #                               ignore_file_type={"parquet"})
    #     vulnerability_array = first_outcome[0]
    #
    #     second_outcome = get_vulns(static_path="./static/", vuln_dict=vulns_dict, num_intensity_bins=50)
    #     second_vulnerability_array = second_outcome[0]
    #     print()
    #     print(second_vulnerability_array)
    #     print(vulnerability_array)


if __name__ == "__main__":
    main()
