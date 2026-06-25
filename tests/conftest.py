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
