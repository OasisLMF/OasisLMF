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


def pytest_addoption(parser):
    parser.addoption(
        '--gul-rtol', type=float, default=1e-10,
        help='Relative tolerance between expected values and results, default is "1e-10"'
    )
    parser.addoption(
        '--gul-atol', type=float, default=1e-8,
        help='Absolute tolerance between expected values and results, default is "1e-8"'
    )
    parser.addoption(
        '--gulmc-generate-missing-expected', action='store_true', default=False,
        help='If True, generate the expected files for the tests that lack them (e.g., newly added tests). Default: False.'
    )
    parser.addoption(
        '--update-expected', action='store_true', default=False,
        help='If True, update all the expected files, overwriting them if they exist. Default: False.'
    )
    parser.addoption(
        '--fm-keep-output', action='store_true', default=False,
        help='If True, keep the test results (useful for debugging purposes). Default: False.'
    )
