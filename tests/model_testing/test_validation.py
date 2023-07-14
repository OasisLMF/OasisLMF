from tempfile import TemporaryDirectory
from unittest import TestCase

from oasislmf.utils.exceptions import OasisException
from oasislmf.validation.model_data import csv_validity_test


class TestValidation(TestCase):

    def test_csv_validity_test___model_data_directory_is_empty(self):
        """
        Test csv_validity_test when model data directory is empty. Raises
        OasisException.
        """

        with TemporaryDirectory() as model_data_dir:

            self.assertRaises(OasisException, csv_validity_test, model_data_dir)
