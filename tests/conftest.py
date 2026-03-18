"""pytest configuration"""
import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _use_tmp_log_dir():
    os.environ['OASIS_PYTEST_TMP_LOGDIR'] = '1'
    yield
    del os.environ['OASIS_PYTEST_TMP_LOGDIR']


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
