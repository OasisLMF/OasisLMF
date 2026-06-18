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
    parser.addoption(
        '--remine-hash-collisions', action='store_true', default=False,
        help='If True, brute-force a fresh colliding-int32-keys set for the hashmap '
             'collision test and print it for paste-back. Use when the hash function '
             'in oasislmf.pytools.common.hashmap changes. Adds ~5-15 s to the run.'
    )
