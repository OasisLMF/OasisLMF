"""pytest configuration"""


def pytest_addoption(parser):
    parser.addoption(
        '--generate-expected', action='store_true', default=False,  # dest='generate_expected',
        help='If True, it generates the expected files instead of running the test. Default: False.'
    )
    parser.addoption(
        '--overwrite-expected', action='store_true', default=False,  # dest='overwrite_expected',
        help='If True, it overwrites the expected files even if they exist. Default: False.'
    )
