from unittest import TestCase

import zlib

from hypothesis import (
    given,
    HealthCheck,
    settings,
)
from hypothesis.strategies import text
from mock import patch, Mock

from oasislmf.utils.compress import compress_string
from oasislmf.utils.exceptions import OasisException

MOCKED_CHUNK_SIZE = 100


@patch('oasislmf.utils.compress.CHUNK_SIZE', MOCKED_CHUNK_SIZE)
class CompressData(TestCase):
    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(data=text(min_size=0, max_size=MOCKED_CHUNK_SIZE - 1))
    def test_data_is_less_than_than_the_chunk_size___result_is_the_compressed_version_of_the_full_data(self, data):
        compressor = zlib.compressobj()
        expected = compressor.compress(data.encode('utf-8')) + compressor.flush()

        result = compress_string(data)

        self.assertEqual(expected, result)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(data=text(min_size=MOCKED_CHUNK_SIZE, max_size=MOCKED_CHUNK_SIZE))
    def test_data_is_equal_to_the_the_chunk_size___result_is_the_compressed_version_of_the_full_data(self, data):
        compressor = zlib.compressobj()
        expected = compressor.compress(data.encode('utf-8')) + compressor.flush()

        result = compress_string(data)

        self.assertEqual(expected, result)

    @settings(suppress_health_check=[HealthCheck.too_slow])
    @given(front=text(min_size=MOCKED_CHUNK_SIZE, max_size=MOCKED_CHUNK_SIZE), overflow=text(min_size=0, max_size=MOCKED_CHUNK_SIZE - 1))
    def test_data_is_larger_than_the_chunk_size___result_is_the_concatenated_compressed_version_of_the_chunks(self, front, overflow):
        compressor = zlib.compressobj()
        expected = compressor.compress(front.encode('utf-8')) + compressor.compress(overflow.encode('utf-8')) + compressor.flush()

        result = compress_string(front + overflow)

        self.assertEqual(expected, result)

    def test_compress_raises_a_zlib_error___error_is_converted_to_oasis_error(self):
        class FakeCompressor(object):
            def compress(self, *args, **kwargs):
                raise zlib.error()

        with patch('zlib.compressobj', Mock(return_value=FakeCompressor())), self.assertRaises(OasisException):
            compress_string('data')

    def test_compress_raises_a_non_zlib_error___error_is_raised_without_converting(self):
        class FakeCompressor(object):
            def compress(self, *args, **kwargs):
                raise ValueError()

        with patch('zlib.compressobj', Mock(return_value=FakeCompressor())), self.assertRaises(ValueError):
            compress_string('data')
