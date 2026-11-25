"""pytest configuration"""
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _use_tmp_log_dir(tmp_path_factory):
    session_tmp = tmp_path_factory.mktemp("oasis_session", numbered=True)
    os.environ['OASIS_TMPDIR'] = str(session_tmp)
    os.environ['OASIS_PYTEST_REDIRECT_LOGS'] = '1'
    yield
    del os.environ['OASIS_PYTEST_REDIRECT_LOGS']
    del os.environ['OASIS_TMPDIR']


@pytest.fixture
def footprint_reads(monkeypatch):
    """The path of every parquet read a dynamic footprint makes, in order."""
    from oasislmf.pytools.getmodel.footprint import FootprintParquetDynamic

    reads = []
    original_get_df_reader = FootprintParquetDynamic.get_df_reader

    def recording_get_df_reader(self, filepath, **kwargs):
        reads.append(filepath)
        return original_get_df_reader(self, filepath, **kwargs)

    monkeypatch.setattr(FootprintParquetDynamic, 'get_df_reader', recording_get_df_reader)
    return reads
