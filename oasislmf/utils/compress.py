import zlib

from .exceptions import OasisException


def compress_data(s):
    """
    Compress large data strings.

    Adapted from a StackOverflow.com solution by Dmitry Skryabin

        https://stackoverflow.com/a/36056646/7556955

    with a modification to set block/chunk size to 500 Mb (5 x 10^8 bytes).
    """

    compressed = ''
    begin = 0
    chunk_size = 5 * 10 ** 8  # 500 Mb
    compressor = zlib.compressobj()

    try:
        while begin < len(s):
            compressed += compressor.compress(s[begin:begin + chunk_size])
            begin += chunk_size

        compressed += compressor.flush()
    except zlib.error as e:
        raise OasisException(str(e))

    return compressed
