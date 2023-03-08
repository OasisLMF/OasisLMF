"""pytest configuration"""


def pytest_addoption(parser):
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
