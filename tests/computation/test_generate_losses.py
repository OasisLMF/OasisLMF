
import pathlib
import os
import logging
import shutil

from unittest import mock
from unittest.mock import patch, Mock, ANY

from ods_tools.oed.common import OdsException
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.path import setcwd
from oasislmf.manager import OasisManager
from .data.common import *
from .test_computation import ComputationChecker


TEST_DIR = pathlib.Path(os.path.realpath(__file__)).parent.parent
LOOKUP_CONFIG = TEST_DIR.joinpath('model_preparation').joinpath('meta_data').joinpath('lookup_config.json')


class TestGenLosses(ComputationChecker):

    @classmethod
    def setUpClass(cls):
        cls.manager = OasisManager()
        cls.default_args = cls.manager._params_generate_losses()
        cls.files_args = cls.manager._params_generate_files()

        # Create Tempfiles (losses)
        cls.tmp_dirs = cls.create_tmp_dirs([a for a in cls.default_args.keys() if 'dir' in a])
        cls.tmp_files = cls.create_tmp_files(
            [a for a in cls.default_args.keys() if 'csv' in a] +
            [a for a in cls.default_args.keys() if 'path' in a] +
            [a for a in cls.default_args.keys() if 'json' in a]
        )
        # Create Tempfiles (files)
        cls.tmp_oasis_files = cls.create_tmp_files(
            [a for a in cls.files_args.keys() if 'csv' in a] +
            [a for a in cls.files_args.keys() if 'path' in a] +
            [a for a in cls.files_args.keys() if 'json' in a]
        )
        # tmp dirs for holding oasis files
        cls.tmp_oasis_files_dirs = cls.create_tmp_dirs([
            'oasis_files_dir__gul',
            'oasis_files_dir__il',
            'oasis_files_dir__ri',
        ])
        # Write minimum data to gen files
        cls.write_json(cls.tmp_oasis_files.get('lookup_complex_config_json'), MIN_RUN_SETTINGS)
        cls.write_str(cls.tmp_oasis_files.get('oed_location_csv'), MIN_LOC)
        cls.write_str(cls.tmp_oasis_files.get('oed_accounts_csv'), MIN_ACC)
        cls.write_str(cls.tmp_oasis_files.get('oed_info_csv'), MIN_INF)
        cls.write_str(cls.tmp_oasis_files.get('oed_scope_csv'), MIN_SCP)
        cls.write_str(cls.tmp_oasis_files.get('keys_data_csv'), MIN_KEYS)
        cls.write_str(cls.tmp_oasis_files.get('keys_errors_csv'), MIN_KEYS_ERR)

        # Write minimum data for gen losses
        cls.write_json(cls.tmp_files.get('analysis_settings_json'), MIN_RUN_SETTINGS)

        # Args for generating sample data
        cls.args_gen_files_gul = {
            'lookup_config_json': LOOKUP_CONFIG,
            'oed_location_csv': cls.tmp_oasis_files['oed_location_csv'].name,
            'oasis_files_dir': cls.tmp_oasis_files_dirs.get('oasis_files_dir__gul').name
        }
        cls.args_gen_files_il = {
            **cls.args_gen_files_gul,
            'oed_accounts_csv': cls.tmp_oasis_files['oed_accounts_csv'].name,
            'oasis_files_dir': cls.tmp_oasis_files_dirs.get('oasis_files_dir__il').name
        }
        cls.args_gen_files_ri = {
            **cls.args_gen_files_il,
            'oed_info_csv': cls.tmp_oasis_files['oed_info_csv'].name,
            'oed_scope_csv': cls.tmp_oasis_files['oed_scope_csv'].name,
            'oasis_files_dir': cls.tmp_oasis_files_dirs.get('oasis_files_dir__ri').name
        }

        # args for generating model data
        cls.required_args_model_data = [
            'num_vulnerabilities',
            'num_intensity_bins',
            'num_damage_bins',
            'num_events',
            'num_areaperils',
            'num_periods',
        ]
        cls.args_model_data = {k: 10 for k in cls.required_args_model_data}

        # populate model data dir
        cls.model_data_dir = cls.tmp_dirs.get('model_data_dir').name
        cls.manager.generate_dummy_model_files(**{
            **cls.args_model_data,
            'target_dir': cls.model_data_dir
        })
        for src_file in pathlib.Path(cls.model_data_dir).glob('*/*.*'):
            shutil.copy(src_file, cls.model_data_dir)
        shutil.rmtree(pathlib.Path(cls.model_data_dir).joinpath('static'))
        shutil.rmtree(pathlib.Path(cls.model_data_dir).joinpath('input'))

    def setUp(self):
        # set base args
        self.min_args = {
            'analysis_settings_json': self.tmp_files.get('analysis_settings_json').name,
            'oasis_files_dir': self.args_gen_files_gul['oasis_files_dir'],
            'model_data_dir': self.model_data_dir
        }

    def test_losses__no_input__exception_raised(self):
        with self.assertRaises(OasisException) as context:
            self.manager.generate_losses()
        expected_err_msg = 'parameter oasis_files_dir is required'
        self.assertIn(expected_err_msg, str(context.exception))

    def test_losses__run_gul(self):
        self.manager.generate_files(**self.args_gen_files_gul)
        self.manager.generate_losses(**self.min_args)

    def test_losses__run_il(self):
        self.manager.generate_files(**self.args_gen_files_il)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, IL_RUN_SETTINGS)
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_il['oasis_files_dir'],
        }
        self.manager.generate_losses(**call_args)

    def test_losses__run_ri(self):
        self.manager.generate_files(**self.args_gen_files_ri)
        run_settings = self.tmp_files.get('analysis_settings_json')
        self.write_json(run_settings, RI_RUN_SETTINGS)
        call_args = {
            **self.min_args,
            'oasis_files_dir': self.args_gen_files_ri['oasis_files_dir'],
        }
        self.manager.generate_losses(**call_args)
