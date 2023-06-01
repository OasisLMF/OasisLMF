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
        cls.default_args = cls.manager._params_generate_files()
        cls.pre_hook_args = cls.manager._params_exposure_pre_analysis()
        cls.gen_files_args = cls.manager._params_generate_files()

        # Tempfiles
        cls.tmp_dirs = cls.create_tmp_dirs([a for a in cls.default_args.keys() if 'dir' in a])
        cls.tmp_files = cls.create_tmp_files(
            [a for a in cls.default_args.keys() if 'csv' in a] +
            [a for a in cls.default_args.keys() if 'json' in a]
        )

    def setUp(self):
        import ipdb; ipdb.set_trace()
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
