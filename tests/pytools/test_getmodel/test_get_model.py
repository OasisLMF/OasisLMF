from unittest import main, TestCase

from oasislmf.pytools.getmodel.manager import get_items, get_vulns, Footprint

import numpy as np
import numba as nb
import pyarrow.parquet as pq
from pyarrow import memory_map
from contextlib import ExitStack
from typing import Any
from oasislmf.pytools.getmodel.custom_footprint import get_custom_footprint_loader


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
    # def test_init(self):
    #     outcome = get_custom_footprint_loader(file_path="./copy_test:SomeObject")
    #     print(outcome)

    def test_footprint(self):
        pass

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
