__all__ = [
    'compress_string',
    'decompress_string',
    'CHUNK_SIZE',
]

import zlib

from .exceptions import OasisException


CHUNK_SIZE = 5 * 10 ** 8  # 500 Mb


def compress_string(st: str) -> bytes:
    """
    Compresses strings using the zlib library.

    Adapted from a StackOverflow.com solution by Dmitry Skryabin

        https://stackoverflow.com/a/36056646/7556955

    with a modification to set block/chunk size to 500 Mb (5 x 10^8 bytes).

    :param s: Input string to be compressed
    :type s: str

    :return: Compressed string as bytes
    :rtype: bytes
    """
    _st = ''.join(st).encode('utf-8')
    compressed = b''
    begin = 0
    compressor = zlib.compressobj()

    try:
        while begin < len(_st):
            compressed += compressor.compress(_st[begin:begin + CHUNK_SIZE])
            begin += CHUNK_SIZE

        compressed += compressor.flush()
    except zlib.error as e:
        raise OasisException("Exception raised in 'compress_string'", e)

    return compressed


def decompress_string(bt: bytes) -> str:
    """
    Decompresses zlib-compressed strings

    :param bt: zlib-compressed string
    :type bt: bytes

    :return: Decompressed (Unicode) string
    :rtype: str
    """
    decompressor = zlib.decompressobj()

    return decompressor.decompress(bt).decode('utf-8')
