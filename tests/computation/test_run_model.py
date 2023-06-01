import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import os

import oasislmf
from oasislmf.manager import OasisManager

from .data.common import *
from .test_computation import ComputationChecker


class TestRunModel(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()

        # Args
        cls.default_args = cls.manager._params_run_model()
        cls.pre_hook_args = cls.manager._params_exposure_pre_analysis()
        cls.gen_files_args = cls.manager._params_generate_files()
        cls.gen_loss_args = cls.manager._params_generate_losses()

        # Tempfiles
        cls.tmp_dirs = cls.create_tmp_dirs([a for a in cls.default_args.keys() if 'dir' in a])
        cls.tmp_files = cls.create_tmp_files(
            [a for a in cls.default_args.keys() if 'csv' in a] +
            [a for a in cls.default_args.keys() if 'json' in a]
        )

    def setUp(self):
        self.min_args = {
            'oed_location_csv': self.tmp_files['oed_location_csv'].name,
            'analysis_settings_json': self.tmp_files['analysis_settings_json'].name,
            'keys_data_csv': self.tmp_files['keys_data_csv'].name,
            'model_data_dir': self.tmp_dirs['model_data_dir'].name
        }
        self.write_json(self.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)
        self.write_json(self.tmp_files.get('model_settings_json'), MIN_MODEL_SETTINGS)
        self.write_str(self.tmp_files.get('oed_location_csv'), MIN_LOC)
        self.write_str(self.tmp_files.get('oed_accounts_csv'), MIN_ACC)
        self.write_str(self.tmp_files.get('keys_data_csv'), MIN_KEYS)

    def test_args__default_combine(self):
        expt_combined_args = self.combine_args([
            self.pre_hook_args,
            self.gen_files_args,
            self.gen_loss_args,
        ])
        self.assertEqual(expt_combined_args, self.default_args)

    def test_model_run__without_pre_analysis(self):
        files_mock = MagicMock()
        losses_mock = MagicMock()
        run_dir = self.tmp_dirs.get('model_run_dir').name
        losses_mock._get_output_dir.return_value = run_dir

        call_args = self.min_args
        with patch.object(oasislmf.computation.run.model, 'GenerateFiles', files_mock), \
                patch.object(oasislmf.computation.run.model, 'GenerateLosses', losses_mock):
            self.manager.run_model(**call_args)

        files_called_kwargs = self.called_args(files_mock)
        losses_called_kwargs = self.called_args(losses_mock)
        expected_called_kwargs = self.combine_args([call_args,
                                                    {
                                                        'model_run_dir': run_dir,
                                                        'oasis_files_dir': os.path.join(run_dir, 'input')
                                                    }
                                                    ])

        files_mock.assert_called_once()
        losses_mock.assert_called_once()
        self.assertEqual(files_called_kwargs, expected_called_kwargs)
        self.assertEqual(losses_called_kwargs, expected_called_kwargs)

    def test_model_run__with_pre_analysis(self):
        pre_mock = MagicMock()
        files_mock = MagicMock()
        losses_mock = MagicMock()

        run_dir = self.tmp_dirs.get('model_run_dir').name
        model_settings = self.tmp_files.get('model_settings_json').name
        account = self.tmp_files.get('oed_accounts_csv').name

        self.write_json(self.tmp_files.get('exposure_pre_analysis_setting_json'), {
            "override_loc_num": 'test_loc_1',
            "override_acc_num": 'test_acc_1',
        })

        call_args = self.combine_args([self.min_args, {
            "model_run_dir": run_dir,
            "model_settings_json": model_settings,
            "oed_accounts_csv": account,
            "exposure_pre_analysis_module": FAKE_PRE_ANALYSIS_MODULE,
            "exposure_pre_analysis_setting_json": self.tmp_files.get('exposure_pre_analysis_setting_json').name
        }])

        with patch.object(oasislmf.computation.run.model, 'GenerateFiles', files_mock), \
                patch.object(oasislmf.computation.run.model, 'GenerateLosses', losses_mock):
            self.manager.run_model(**call_args)

        exposure_data = files_mock.call_args.kwargs['exposure_data']
        self.assertEqual(exposure_data.location.dataframe['LocNumber'][0], 'test_loc_1')
        self.assertEqual(exposure_data.account.dataframe['AccNumber'][0], 'test_acc_1')

        files_called_kwargs = self.called_args(files_mock)
        losses_called_kwargs = self.called_args(losses_mock)
        expected_called_kwargs = self.combine_args([call_args,
                                                    {
                                                        'model_run_dir': run_dir,
                                                        'oasis_files_dir': os.path.join(run_dir, 'input')
                                                    }
                                                    ])

        files_mock.assert_called_once()
        losses_mock.assert_called_once()
        self.assertEqual(files_called_kwargs, expected_called_kwargs)
        self.assertEqual(losses_called_kwargs, expected_called_kwargs)
