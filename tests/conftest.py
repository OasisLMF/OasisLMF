"""pytest configuration"""


def pytest_addoption(parser):
    parser.addoption(
        '--generate-expected', action='store_true', default=False,  # dest='generate_expected',
        help='If True, it generates the expected files instead of running the test'
    )
