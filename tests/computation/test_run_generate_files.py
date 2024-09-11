import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import os

import oasislmf
from oasislmf.manager import OasisManager

from .data.common import *
from .test_computation import ComputationChecker


class TestGenerateFiles(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()

        # Args
        cls.default_args = cls.manager._params_generate_oasis_files()
        cls.pre_hook_args = cls.manager._params_exposure_pre_analysis()
        cls.gen_files_args = cls.manager._params_generate_files()
        cls.post_file_gen_args = cls.manager._params_post_file_gen()

    def setUp(self):
        # Tempfiles
        self.tmp_dirs = self.create_tmp_dirs([a for a in self.default_args.keys() if 'dir' in a])
        self.tmp_files = self.create_tmp_files(
            [a for a in self.default_args.keys() if 'csv' in a] +
            [a for a in self.default_args.keys() if 'json' in a]
        )
        self.min_args = {
            'oed_location_csv': self.tmp_files['oed_location_csv'].name,
            'keys_data_csv': self.tmp_files['keys_data_csv'].name,
        }
        self.write_json(self.tmp_files.get('model_settings_json'), MIN_MODEL_SETTINGS)
        self.write_str(self.tmp_files.get('oed_location_csv'), MIN_LOC)
        self.write_str(self.tmp_files.get('oed_accounts_csv'), MIN_ACC)
        self.write_str(self.tmp_files.get('keys_data_csv'), MIN_KEYS)

    def test_args__default_combine(self):
        expt_combined_args = self.combine_args([
            self.pre_hook_args,
            self.gen_files_args,
            self.post_file_gen_args
        ])
        self.assertEqual(expt_combined_args, self.default_args)

    def test_generate__without_pre_analysis(self):
        files_mock = MagicMock()
        files_dir = self.tmp_dirs.get('oasis_files_dir').name
        files_mock._get_output_dir.return_value = files_dir

        call_args = self.min_args
        with patch.object(oasislmf.computation.run.generate_files, 'GenerateFiles', files_mock):
            self.manager.generate_oasis_files(**call_args)

        files_called_kwargs = self.called_args(files_mock)
        expected_called_kwargs = self.combine_args([call_args, {"oasis_files_dir": files_dir}])

        files_mock.assert_called_once()
        for key, expected_value in expected_called_kwargs.items():
            self.assertEqual(files_called_kwargs[key], expected_value)

    def test_generate__with_pre_analysis(self):
        pre_mock = MagicMock()
        files_mock = MagicMock()
        files_dir_obj = self.tmp_dirs.get('oasis_files_dir')
        files_dir_obj.cleanup()  # delete file to check if its created

        model_settings = self.tmp_files.get('model_settings_json').name
        account = self.tmp_files.get('oed_accounts_csv').name

        self.write_json(self.tmp_files.get('exposure_pre_analysis_setting_json'), {
            "override_loc_num": 'test_loc_1',
            "override_acc_num": 'test_acc_1',
        })

        call_args = self.combine_args([self.min_args, {
            "model_settings_json": model_settings,
            "oasis_files_dir": files_dir_obj.name,
            "oed_accounts_csv": account,
            "exposure_pre_analysis_module": FAKE_PRE_ANALYSIS_MODULE,
            "exposure_pre_analysis_setting_json": self.tmp_files.get('exposure_pre_analysis_setting_json').name
        }])

        with patch.object(oasislmf.computation.run.generate_files, 'GenerateFiles', files_mock):
            self.manager.generate_oasis_files(**call_args)

        exposure_data = files_mock.call_args.kwargs['exposure_data']
        self.assertEqual(exposure_data.location.dataframe['LocNumber'][0], 'test_loc_1')
        self.assertEqual(exposure_data.account.dataframe['AccNumber'][0], 'test_acc_1')

        files_called_kwargs = self.called_args(files_mock)
        expected_called_kwargs = self.combine_args([call_args])

        files_mock.assert_called_once()
        for key, expected_value in expected_called_kwargs.items():
            self.assertEqual(files_called_kwargs[key], expected_value)
        self.assertTrue(os.path.isdir(files_dir_obj.name))

    def test_generate__return_vaules(self):
        pass
