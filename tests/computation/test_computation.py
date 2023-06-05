
__all__ = [
    'ComputationChecker',
]

from os import path
import json
import io
import os
from collections import ChainMap
import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest


class ComputationChecker(unittest.TestCase):

    @staticmethod
    def create_tmp_files(file_list):
        return {f: NamedTemporaryFile() for f in file_list}

    @staticmethod
    def create_tmp_dirs(dirs_list):
        return {d: TemporaryDirectory() for d in dirs_list}

    @staticmethod
    def write_json(tmpfile, data):
        with open(tmpfile.name, mode='w') as f:
            f.write(json.dumps(data))

    @staticmethod
    def write_str(tmpfile, data):
        with open(tmpfile.name, mode='w') as f:
            f.write(data)

    @staticmethod
    def read_file(filepath):
        with open(filepath, mode='rb') as f:
            return f.read()

    @staticmethod
    def combine_args(dict_list):
        return dict(ChainMap(*dict_list))

    @staticmethod
    def called_args(mock_obj):
        return {k: v for k, v in mock_obj.call_args.kwargs.items() if isinstance(v, (str, int))}

    @pytest.fixture(autouse=True)
    def logging_fixtures(self, caplog):
        self._caplog = caplog


# class TestComputation(unittest.TestCase):
