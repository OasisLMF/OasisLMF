
__all__ = [
    'ComputationChecker',
]

import contextlib
import json
from collections import ChainMap
import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest


class ComputationChecker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = contextlib.ExitStack()

    def create_tmp_files(self, file_list):
        return {f: self.stack.enter_context(NamedTemporaryFile(prefix=self.__class__.__name__)) for f in file_list}

    def tearDown(self):
        super().tearDown()
        self.stack.close()

    def create_tmp_dirs(self, dirs_list):
        return {d: self.tmp_dir() for d in dirs_list}

    def tmp_dir(self):
        dir_ = TemporaryDirectory()
        self.stack.enter_context(dir_)
        return dir_

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
