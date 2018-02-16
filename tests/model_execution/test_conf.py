from __future__ import unicode_literals

import json
import uuid
from unittest import TestCase

import os
import io

from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import sampled_from
from pathlib2 import Path

from oasislmf.model_execution.conf import create_analysis_settings_json
from oasislmf.model_execution.files import GENERAL_SETTINGS_FILE, MODEL_SETTINGS_FILE, GUL_SUMMARIES_FILE, \
    IL_SUMMARIES_FILE
from oasislmf.utils.exceptions import OasisException


class CreateAnalysisSettingsJson(TestCase):
    def create_fake_directory(self, d):
        Path(os.path.join(d, GENERAL_SETTINGS_FILE)).touch()
        Path(os.path.join(d, MODEL_SETTINGS_FILE)).touch()
        Path(os.path.join(d, GUL_SUMMARIES_FILE)).touch()
        Path(os.path.join(d, IL_SUMMARIES_FILE)).touch()

    def test_directory_does_not_exist___oasis_exception_is_raised(self):
        with self.assertRaises(OasisException):
            create_analysis_settings_json('/tmp/non_existing_dir_{}'.format(uuid.uuid4().hex))

    @given(sampled_from([GENERAL_SETTINGS_FILE, MODEL_SETTINGS_FILE, GUL_SUMMARIES_FILE, IL_SUMMARIES_FILE]))
    def test_input_file_is_missing___error_is_raised(self, to_delete):
        with TemporaryDirectory() as d:
            self.create_fake_directory(d)

            os.remove(os.path.join(d, to_delete))

            with self.assertRaises(OasisException):
                create_analysis_settings_json(d)

    def test_general_settings_file_contains_properties___properties_are_loaded_into_general_settings(self):
        with TemporaryDirectory() as d:
            self.create_fake_directory(d)

            with io.open(os.path.join(d, GENERAL_SETTINGS_FILE), 'w', encoding='utf-8') as f:
                f.writelines([
                    'first,1,int\n',
                    'second,foo,str\n',
                    'third,2.2,float\n',
                ])

            data = json.loads(create_analysis_settings_json(d))

            self.assertEqual(data['first'], 1)
            self.assertEqual(data['second'], 'foo')
            self.assertEqual(data['third'], 2.2)

    def test_model_settings_file_contains_properties___properties_are_loaded_into_model_settings(self):
        with TemporaryDirectory() as d:
            self.create_fake_directory(d)

            with io.open(os.path.join(d, MODEL_SETTINGS_FILE), 'w', encoding='utf-8') as f:
                f.writelines([
                    'first,3,int\n',
                    'second,bar,str\n',
                    'third,4.4,float\n',
                ])

            data = json.loads(create_analysis_settings_json(d))

            self.assertEqual(data['model_settings'], {
                'first': 3,
                'second': 'bar',
                'third': 4.4,
            })

    def test_gul_settings_contains_data___data_is_stored_in_gul_summaries(self):
        with TemporaryDirectory() as d:
            self.create_fake_directory(d)

            with io.open(os.path.join(d, GUL_SUMMARIES_FILE), 'w', encoding='utf-8') as f:
                f.writelines([
                    '1,leccalc_foo,TRUE\n',
                    '1,leccalc_bar,FALSE\n',
                    '1,another,FALSE\n',
                    '2,leccalc_boo,FALSE\n',
                    '2,leccalc_far,TRUE\n',
                    '2,different,TRUE\n',
                ])

            data = json.loads(create_analysis_settings_json(d))

            self.assertEqual(data['gul_summaries'], [
                {'id': 1, 'another': False, 'leccalc': {'leccalc_foo': True, 'leccalc_bar': False}},
                {'id': 2, 'different': True, 'leccalc': {'leccalc_boo': False, 'leccalc_far': True}},
            ])

    def test_il_settings_contains_data___data_is_stored_in_il_summaries(self):
        with TemporaryDirectory() as d:
            self.create_fake_directory(d)

            with io.open(os.path.join(d, IL_SUMMARIES_FILE), 'w', encoding='utf-8') as f:
                f.writelines([
                    '1,leccalc_foo,TRUE\n',
                    '1,leccalc_bar,FALSE\n',
                    '1,another,FALSE\n',
                    '2,leccalc_boo,FALSE\n',
                    '2,leccalc_far,TRUE\n',
                    '2,different,TRUE\n',
                ])

            data = json.loads(create_analysis_settings_json(d))

            self.assertEqual(data['il_summaries'], [
                {'id': 1, 'another': False, 'leccalc': {'leccalc_foo': True, 'leccalc_bar': False}},
                {'id': 2, 'different': True, 'leccalc': {'leccalc_boo': False, 'leccalc_far': True}},
            ])
