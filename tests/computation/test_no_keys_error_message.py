from oasislmf.computation.generate.files import no_keys_error_message


def test_names_the_keys_errors_file_that_was_written():
    msg = no_keys_error_message('/run/input/keys.parquet', '/run/input/keys-errors.parquet')
    assert 'Check the `keys-errors.parquet` file' in msg
    assert msg.endswith('File path: /run/input/keys-errors.parquet')


def test_keys_errors_file_follows_keys_file_format_when_not_given():
    """The errors file is written in the same format as the keys file (#2227)."""
    msg = no_keys_error_message('/run/input/keys.parquet')
    assert 'Check the `keys-errors.parquet` file' in msg
    assert msg.endswith('File path: /run/input/keys-errors.parquet')

    msg = no_keys_error_message('/run/input/keys.csv')
    assert 'Check the `keys-errors.csv` file' in msg
    assert msg.endswith('File path: /run/input/keys-errors.csv')


def test_no_keys_file():
    msg = no_keys_error_message(None)
    assert msg.startswith('No successful lookup results found in the keys file')
    assert msg.endswith('File path: Missing')
