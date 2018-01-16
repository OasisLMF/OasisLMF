from unittest import TestCase

import zlib

from hypothesis import given
from hypothesis.strategies import binary
from mock import patch

from oasislmf.utils.compress import compress_data

MOCKED_CHUNK_SIZE = 100


@patch('oasislmf.utils.compress.CHUNK_SIZE', MOCKED_CHUNK_SIZE)
class CompressData(TestCase):
    @given(binary(min_size=0, max_size=MOCKED_CHUNK_SIZE - 1))
    def test_data_is_less_than_than_the_chunk_size___result_is_the_compressed_version_of_the_full_data(self, data):
        compressor = zlib.compressobj()
        expected = compressor.compress(data) + compressor.flush()

        result = compress_data(data)

        self.assertEqual(expected, result)

    @given(binary(min_size=MOCKED_CHUNK_SIZE, max_size=MOCKED_CHUNK_SIZE))
    def test_data_is_equal_to_the_the_chunk_size___result_is_the_compressed_version_of_the_full_data(self, data):
        compressor = zlib.compressobj()
        expected = compressor.compress(data) + compressor.flush()

        result = compress_data(data)

        self.assertEqual(expected, result)

    @given(binary(min_size=MOCKED_CHUNK_SIZE, max_size=MOCKED_CHUNK_SIZE), binary(min_size=0, max_size=MOCKED_CHUNK_SIZE - 1))
    def test_data_is_larger_than_the_chunk_size___result_is_the_concatenated_compressed_version_of_the_chunks(self, front, overflow):
        compressor = zlib.compressobj()
        expected = compressor.compress(front) + compressor.compress(overflow) + compressor.flush()

        result = compress_data(front + overflow)

        self.assertEqual(expected, result)
