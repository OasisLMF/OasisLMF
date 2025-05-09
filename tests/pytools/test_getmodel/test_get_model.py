from unittest import main, TestCase, mock
from unittest.mock import patch

from numba.typed import Dict
from numba import int32 as nb_int32, float64 as nb_float64

import os
import shutil
import tempfile

from oasislmf.pytools.getmodel.manager import get_vulns
from oasislmf.pytools.getmodel.vulnerability import vulnerability_to_parquet
from oasis_data_manager.filestore.backends.local import LocalStorage

import numpy as np
import numba as nb
import pandas as pd
import subprocess


class TestGetVulns(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vuln_dict_base = {1: 2, 2: 0, 3: 1}
        self.vuln_dict = Dict.empty(key_type=nb.int32, value_type=nb.int32)
        for key, value in self.vuln_dict_base.items():
            self.vuln_dict[key] = value
        self.num_intensity_bins = 2
        self.mock_vuln_data = np.array([
            (1, 1, 1, 0.330),
            (1, 1, 2, 0.333),
            (1, 1, 3, 0.337),
            (1, 2, 1, 0.330),
            (1, 2, 2, 0.333),
            (1, 2, 3, 0.337),
            (2, 1, 1, 0.7),
            (2, 1, 2, 0.1),
            (2, 1, 3, 0.2),
            (2, 2, 1, 0.1),
            (2, 2, 2, 0.2),
            (2, 2, 3, 0.7),
            (3, 1, 1, 0.4),
            (3, 1, 2, 0.3),
            (3, 1, 3, 0.3),
            (3, 2, 1, 0.3),
            (3, 2, 2, 0.3),
            (3, 2, 3, 0.4),
            (4, 1, 1, 0.9),
            (4, 1, 2, 0.05),
            (4, 1, 3, 0.05),
            (4, 2, 1, 0.05),
            (4, 2, 2, 0.05),
            (4, 2, 3, 0.9),
        ], dtype=[('vulnerability_id', 'i4'), ('intensity_bin_id', 'i4'), ('damage_bin_id', 'i4'), ('probability', 'f4')])
        self.mock_vuln_adj_data = np.array([
            (33, 1, 1, 0.4),
            (33, 1, 2, 0.3),
            (33, 1, 3, 0.3),
            (2, 1, 1, 0.83),
            (2, 1, 2, 0.08),
            (2, 1, 3, 0.08),
            (2, 2, 1, 0.08),
            (2, 2, 2, 0.08),
            (2, 2, 3, 0.83),
            (0, 2, 1, 0.3),
            (0, 2, 2, 0.3),
            (0, 2, 3, 0.4),
        ], dtype=[('vulnerability_id', 'i4'), ('intensity_bin_id', 'i4'), ('damage_bin_id', 'i4'), ('probability', 'f4')])
        self.expected_outputs = {
            'vuln_array_adj': np.array([[[0.83, 0.08], [0.08, 0.08], [0.08, 0.83]],
                                        [[0.4, 0.3], [0.3, 0.3], [0.3, 0.4]],
                                        [[0.330, 0.330], [0.333, 0.333], [0.337, 0.337]]], dtype=np.float32),
            'vuln_array': np.array([[[0.7, 0.1], [0.1, 0.2], [0.2, 0.7]],
                                    [[0.4, 0.3], [0.3, 0.3], [0.3, 0.4]],
                                    [[0.330, 0.330], [0.333, 0.333], [0.337, 0.337]]], dtype=np.float32),
            'vulns_id': np.array([2, 3, 1], dtype=np.int32),
            'num_damage_bins': 3
        }
        self.ignore_file_type = {"parquet", "bin", "csv"}
        self.vuln_dict_adj = self.create_vuln_dict_from_mock_data(self.mock_vuln_adj_data)

        # create csv file
        df = pd.DataFrame(self.mock_vuln_data)
        csv_path = os.path.join(self.temp_dir, 'vulnerability.csv')
        df.to_csv(csv_path, index=False)

        # create bin + idx file
        subprocess.run("vulnerabilitytobin -d 3 -i < vulnerability.csv", check=True, capture_output=False, shell=True, cwd=self.temp_dir)
        # rename vulnerability.bin to vul.bin and vulnerability.idx to vul.idx
        os.rename(os.path.join(self.temp_dir, 'vulnerability.bin'), os.path.join(self.temp_dir, 'vul.bin'))
        os.rename(os.path.join(self.temp_dir, 'vulnerability.idx'), os.path.join(self.temp_dir, 'vul.idx'))

        # create bin file
        bin_path = os.path.join(self.temp_dir, 'vulnerability.bin')
        with open(bin_path, 'wb') as f:
            # Calculate num_damage_bins as the maximum damage_bin_id
            num_damage_bins = np.max(self.mock_vuln_data['damage_bin_id'])

            # Write num_damage_bins as the header
            header = np.array([num_damage_bins], dtype='i4')
            f.write(header.tobytes())

            # Write the mock_vuln_data records
            for record in self.mock_vuln_data:
                f.write(record.tobytes())

        # create parquet file
        vulnerability_to_parquet(self.temp_dir)

        # create adjustment csv file
        df_adj = pd.DataFrame(self.mock_vuln_adj_data)
        csv_path_adj = os.path.join(self.temp_dir, 'vulnerability_adj.csv')
        df_adj.to_csv(csv_path_adj, index=False)
        self.static_path = self.temp_dir

    @staticmethod
    def create_vuln_dict_from_mock_data(mock_vuln_adj_data):
        vuln_dict = {}
        for record in mock_vuln_adj_data:
            vuln_id, intensity_bin_id, damage_bin_id, probability = record
            if vuln_id not in vuln_dict:
                vuln_dict[vuln_id] = []
            vuln_dict[vuln_id].append([intensity_bin_id, damage_bin_id, probability])
        return vuln_dict

    def map_vulns_id_to_indices(self, vulns_id):
        """
        Create a mapping from vulnerability IDs to their indices in the vuln_array to compare them.
        """
        return {v_id: idx for idx, v_id in enumerate(vulns_id)}

    def test_get_vulns(self):

        model_storage = LocalStorage(root_dir=self.static_path, cache_dir=None)
        for file_type in ['csv', 'bin', 'parquet']:
            ignore_file_types = self.ignore_file_type - {file_type}
            vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                              self.num_intensity_bins, ignore_file_type=ignore_file_types)
            self.assertIsNotNone(vuln_array)
            self.assertEqual(num_damage_bins, self.expected_outputs['num_damage_bins'])
            for vuln_id in vulns_id:
                expected_index = self.vuln_dict_base[vuln_id]
                actual_index = np.where(vulns_id == vuln_id)[0][0]
                self.assertTrue(np.array_equal(vuln_array[actual_index], self.expected_outputs['vuln_array'][expected_index]))

        ignore_file_types = {'csv', 'parquet'}

        # Test index file: delete vulnerability.bin and rename vul.bin to vulnerability.bin and vul.idx to vulnerability.idx
        os.remove(os.path.join(self.temp_dir, 'vulnerability.bin'))
        os.rename(os.path.join(self.temp_dir, 'vul.bin'), os.path.join(self.temp_dir, 'vulnerability.bin'))
        os.rename(os.path.join(self.temp_dir, 'vul.idx'), os.path.join(self.temp_dir, 'vulnerability.idx'))
        vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                          self.num_intensity_bins, ignore_file_type=ignore_file_types)
        self.assertIsNotNone(vuln_array)
        self.assertEqual(num_damage_bins, self.expected_outputs['num_damage_bins'])
        for vuln_id in vulns_id:
            expected_index = self.vuln_dict_base[vuln_id]
            actual_index = np.where(vulns_id == vuln_id)[0][0]
            self.assertTrue(np.array_equal(vuln_array[actual_index], self.expected_outputs['vuln_array'][expected_index]))

    def test_get_vulns_adj(self):
        model_storage = LocalStorage(root_dir=self.static_path, cache_dir=None)
        for file_type in ['csv', 'bin', 'parquet']:
            ignore_file_types = self.ignore_file_type - {file_type}
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('oasislmf.pytools.getmodel.manager.analysis_settings_loader', return_value={'vulnerability_adjustments': {
                    'replace_file': str(os.path.join(self.static_path, "vulnerability_adj.csv"))}
                }) as mock_manager, \
                    mock.patch('oasislmf.utils.data.analysis_settings_loader',
                               return_value={'vulnerability_adjustments': {
                                   'replace_file': str(os.path.join(self.static_path, "vulnerability_adj.csv"))}
                               }) as mock_util:
                    vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                                      self.num_intensity_bins, ignore_file_type=ignore_file_types)
                    self.assertIsNotNone(vuln_array)
                    self.assertEqual(num_damage_bins, self.expected_outputs['num_damage_bins'])
                    for vuln_id in vulns_id:
                        expected_index = self.vuln_dict_base[vuln_id]
                        actual_index = np.where(vulns_id == vuln_id)[0][0]
                        self.assertTrue(np.array_equal(vuln_array[actual_index], self.expected_outputs['vuln_array_adj'][expected_index]))

        for file_type in ['csv', 'bin', 'parquet']:
            ignore_file_types = self.ignore_file_type - {file_type}
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('oasislmf.pytools.getmodel.manager.analysis_settings_loader', return_value={'vulnerability_adjustments': {
                    'replace_file': str(os.path.join(self.static_path, "vulnerability_adj.csv"))}
                }) as mock_manager, \
                    mock.patch('oasislmf.utils.data.analysis_settings_loader',
                               return_value={'vulnerability_adjustments': {
                                   'replace_file': str(os.path.join(self.static_path, "vulnerability_adj.csv"))}
                               }) as mock_util:

                    vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                                      self.num_intensity_bins, ignore_file_type=ignore_file_types)
                    self.assertIsNotNone(vuln_array)
                    self.assertEqual(num_damage_bins, self.expected_outputs['num_damage_bins'])
                    for vuln_id in vulns_id:
                        expected_index = self.vuln_dict_base[vuln_id]
                        actual_index = np.where(vulns_id == vuln_id)[0][0]
                        self.assertTrue(np.array_equal(vuln_array[actual_index], self.expected_outputs['vuln_array_adj'][expected_index]))

        ignore_file_types = {'csv', 'parquet'}

        # Test index file: delete vulnerability.bin and rename vul.bin to vulnerability.bin and vul.idx to vulnerability.idx
        os.remove(os.path.join(self.temp_dir, 'vulnerability.bin'))
        os.rename(os.path.join(self.temp_dir, 'vul.bin'), os.path.join(self.temp_dir, 'vulnerability.bin'))
        os.rename(os.path.join(self.temp_dir, 'vul.idx'), os.path.join(self.temp_dir, 'vulnerability.idx'))
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('oasislmf.pytools.getmodel.manager.analysis_settings_loader', return_value={'vulnerability_adjustments': {
                'replace_file': str(os.path.join(self.static_path, "vulnerability_adj.csv"))}
            }) as mock_manager, \
                    mock.patch('oasislmf.utils.data.analysis_settings_loader',
                               return_value={'vulnerability_adjustments': {
                                   'replace_file': str(os.path.join(self.static_path, "vulnerability_adj.csv"))}
                               }) as mock_util:
                vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                                  self.num_intensity_bins, ignore_file_type=ignore_file_types)
                self.assertIsNotNone(vuln_array)
                self.assertEqual(num_damage_bins, self.expected_outputs['num_damage_bins'])
                for vuln_id in vulns_id:
                    expected_index = self.vuln_dict_base[vuln_id]
                    actual_index = np.where(vulns_id == vuln_id)[0][0]
                    self.assertTrue(np.array_equal(vuln_array[actual_index], self.expected_outputs['vuln_array_adj'][expected_index]))

    def test_get_vulns_compare(self):
        # First, run the function without any adjustments
        vuln_arrays_no_adj = {}
        vuln_ids_no_adj = {}
        model_storage = LocalStorage(root_dir=self.static_path, cache_dir=None)
        for file_type in ['csv', 'bin', 'parquet']:
            ignore_file_types = self.ignore_file_type - {file_type}
            vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                              self.num_intensity_bins, ignore_file_type=ignore_file_types)
            vuln_arrays_no_adj[file_type] = vuln_array
            vuln_ids_no_adj[file_type] = vulns_id

        # Now, create a mock adjustment that is equivalent to the original data
        vuln_dict_adj = self.create_vuln_dict_from_mock_data(self.mock_vuln_data)

        # Run the function with the mock adjustment
        vuln_arrays_with_adj = {}
        vuln_ids_with_adj = {}

        for file_type in ['csv', 'bin', 'parquet']:
            ignore_file_types = self.ignore_file_type - {file_type}
            with mock.patch('os.path.exists', return_value=True):
                with mock.patch('oasislmf.pytools.getmodel.manager.analysis_settings_loader',
                                return_value={'vulnerability_adjustments': {
                                    'replace_data': vuln_dict_adj}
                                }) as mock_manager, \
                        mock.patch('oasislmf.utils.data.analysis_settings_loader',
                                   return_value={'vulnerability_adjustments': {
                                       'replace_data': vuln_dict_adj}
                                   }) as mock_util:
                    vuln_array, vulns_id, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                                      self.num_intensity_bins, ignore_file_type=ignore_file_types)
                    vuln_arrays_with_adj[file_type] = vuln_array
                    vuln_ids_with_adj[file_type] = vulns_id

        # Compare the results with and without adjustments
        for file_type in ['csv', 'bin', 'parquet']:
            for vuln_id in vuln_ids_no_adj[file_type]:
                idx_no_adj = np.where(vuln_ids_no_adj[file_type] == vuln_id)[0][0]
                idx_with_adj = np.where(vuln_ids_with_adj[file_type] == vuln_id)[0][0]
                self.assertTrue(np.array_equal(vuln_arrays_no_adj[file_type][idx_no_adj], vuln_arrays_with_adj[file_type][idx_with_adj]))

        ignore_file_types = {'csv', 'parquet'}
        os.remove(os.path.join(self.temp_dir, 'vulnerability.bin'))
        os.rename(os.path.join(self.temp_dir, 'vul.bin'), os.path.join(self.temp_dir, 'vulnerability.bin'))
        os.rename(os.path.join(self.temp_dir, 'vul.idx'), os.path.join(self.temp_dir, 'vulnerability.idx'))
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('oasislmf.pytools.getmodel.manager.analysis_settings_loader',
                            return_value={'vulnerability_adjustments': {
                                'replace_data': vuln_dict_adj}
                            }) as mock_manager, \
                mock.patch('oasislmf.utils.data.analysis_settings_loader',
                           return_value={'vulnerability_adjustments': {
                               'replace_data': vuln_dict_adj}
                           }) as mock_util:
                vuln_array_idx_adj, vulns_id_idx_adj, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                                                  self.num_intensity_bins, ignore_file_type=ignore_file_types)
        vuln_array_idx, vulns_id_idx, num_damage_bins = get_vulns(model_storage, self.static_path, self.vuln_dict,
                                                                  self.num_intensity_bins, ignore_file_type=ignore_file_types)
        for vuln_id in vulns_id_idx:
            idx_no_adj = np.where(vulns_id_idx == vuln_id)[0][0]
            idx_with_adj = np.where(vulns_id_idx_adj == vuln_id)[0][0]
            self.assertTrue(np.array_equal(vuln_array_idx[idx_no_adj], vuln_array_idx_adj[idx_with_adj]))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


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
    pass
    # def test_outcome(self):
    #     from multiprocessing.shared_memory import SharedMemory

    # def test_convert_footprint_to_parquet(self):
    #     from oasislmf.pytools.getmodel.footprint import Footprint
    #     from contextlib import ExitStack
    #     import json
    #     import numpy as np
    #     import pandas as pd
    #
    #     static_path: str = "./conversions/"
    #
    #     with ExitStack() as stack:
    #         footprint_obj = stack.enter_context(Footprint.load(static_path=static_path,
    #                                                            ignore_file_type={
    #                                                                'footprint.bin.z',
    #                                                                'footprint.idx.z'
    #                                                                }
    #                                                            )
    #                                             )
    #         # areaperil_id, intensity_bin_id, probability
    #
    #         buffer = []
    #
    #         for key in footprint_obj.footprint_index.keys():
    #             buffer.append(footprint_obj.get_event(key))
    #
    #         data = np.concatenate(buffer)
    #
    #         meta_data = {
    #             "num_intensity_bins": footprint_obj.num_intensity_bins,
    #             "has_intensity_uncertainty": True if footprint_obj.has_intensity_uncertainty is 1 else False
    #         }
    #         print(meta_data)
    #         print(data[0])
    #         df = pd.DataFrame(data)
    #         print(df.head())
    #
    # with open(f'{static_path}/footprint_parquet_meta.json', 'w') as outfile:
    #     json.dump(meta_data, outfile)

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
    #     from oasislmf.pytools.getmodel.footprint import Footprint
    #     import pandas as pd
    #     import pyarrow as pa
    #     import pyarrow.parquet as pq
    #
    #     with Footprint.load(static_path="./conversions/") as test:
    #         outcome = test
    #         print(outcome)
    # print("here is the outcome: ", outcome.footprint_index)

    # buffer = []
    #
    # for key in outcome.footprint_index.keys():
    #     row = outcome.footprint_index[key]
    #     row["event_id"] = key
    #     buffer.append(row)
    # df = pd.DataFrame(buffer)
    # print(df.head())
    # table = pa.Table.from_pandas(df)
    # pq.write_table(table, "./footprint.parquet")

    # print(outcome.footprint_index.head())

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
