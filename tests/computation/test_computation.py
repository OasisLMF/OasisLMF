
__all__ = [
    'ComputationChecker',
]

import contextlib
import json
from collections import ChainMap
import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
import pytest

from oasislmf.computation.generate.files import GenerateFiles


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


def test_get_signature_preserves_falsy_defaults():
    """Params whose default is falsy (False / 0 / '') must keep that default instead of None """
    sig = GenerateFiles.get_signature()
    assert sig is not None
    params = sig.parameters

    # boolean flags that default to False must report False, not None
    for name in ("verbose", "intermediary_csv", "disable_summarise_exposure", "write_ri_tree"):
        assert params[name].default is False, \
            f"{name} default should be False, got {params[name].default!r}"

    # a param with a genuine string default is still populated correctly
    assert isinstance(params["base_df_engine"].default, str)

    # a param with no default at all keeps the None placeholder
    assert params["oasis_files_dir"].default is None
