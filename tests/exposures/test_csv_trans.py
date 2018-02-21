import filecmp
import os
import unittest


# find the root of git repo & import class under test
# Set Dir Vars
from backports.tempfile import TemporaryDirectory
from hypothesis import given
from hypothesis.strategies import integers
from pathlib2 import Path

from oasislmf.exposures.csv_trans import Translator
from oasislmf.utils.diff import unified_diff

data_dir = str(Path(__file__).parent.joinpath('csv_trans_data'))
input_data_dir = str(Path(data_dir, 'input'))
expected_data_dir = str(Path(data_dir, 'expected'))


class CsvTrans(unittest.TestCase):
    @given(integers(min_value=1, max_value=10))
    def test_source_to_canonical(self, chunk_size):
        with TemporaryDirectory() as d:
            output_file = os.path.join(d, 'canonical.csv')

            translator = Translator(
                os.path.join(input_data_dir, 'source.csv'),
                output_file,
                os.path.join(input_data_dir, 'source_to_canonical.xsd'),
                os.path.join(input_data_dir, 'source_to_canonical.xslt'),
                chunk_size=chunk_size,
                append_row_nums=True
            )
            translator()

            diff = unified_diff(output_file, os.path.join(expected_data_dir, 'canonical.csv'), as_string=True)
            self.assertEqual(0, len(diff), diff)

    @given(integers(min_value=1, max_value=10))
    def test_canonical_to_model(self, chunk_size):
        with TemporaryDirectory() as d:
            output_file = os.path.join(d, 'model.csv')

            translator = Translator(
                os.path.join(input_data_dir, 'canonical.csv'),
                output_file,
                os.path.join(input_data_dir, 'canonical_to_model.xsd'),
                os.path.join(input_data_dir, 'canonical_to_model.xslt'),
                chunk_size=chunk_size
            )
            translator()
            self.assertTrue(filecmp.cmp(output_file, os.path.join(expected_data_dir, 'model.csv')))
