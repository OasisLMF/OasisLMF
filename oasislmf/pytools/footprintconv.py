import argparse
from contextlib import ExitStack
import logging
import mmap
import numpy as np
import os
from zlib import compress

from .getmodel.common import (FootprintHeader, EventIndexBin, EventIndexBinZ,
                              footprint_filename, footprint_index_filename, zfootprint_filename, zfootprint_index_filename)
from .getmodel.footprint import uncompressedMask

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def compress_footprint(static_path, decompressed_size=True, compression_level=-1):
    with ExitStack() as stack:
        footprint_obj = stack.enter_context(open(os.path.join(static_path, footprint_filename), 'rb'))
        footprint_map = mmap.mmap(footprint_obj.fileno(), length=0, access=mmap.ACCESS_READ)
        footprint_header = np.frombuffer(bytearray(footprint_map[:FootprintHeader.itemsize]), dtype=FootprintHeader)
        # The input footprint.idx's own entry layout (with or without d_size) is determined by
        # the header bit as it already is on disk -- independent of decompressed_size, which only
        # controls the *output* zipped index's layout below.
        input_has_d_size = bool(footprint_header['has_intensity_uncertainty'].item() & uncompressedMask)
        input_index_dtype = EventIndexBinZ if input_has_d_size else EventIndexBin

        if decompressed_size:
            footprint_header['has_intensity_uncertainty'] |= uncompressedMask
            index_dtype = EventIndexBinZ
        else:
            # The output index is written without d_size below; clear the bit even if the input
            # file had it set (e.g. written by csvtobin -d without -z), or the output header would
            # claim 28-byte entries while the index actually written uses 20-byte ones.
            footprint_header['has_intensity_uncertainty'] &= ~uncompressedMask
            index_dtype = EventIndexBin

        zfootprint_obj = stack.enter_context(open(os.path.join(static_path, zfootprint_filename), 'wb'))
        zfootprint_obj.write(footprint_header.tobytes())

        footprint_idx_bin = np.memmap(os.path.join(static_path, footprint_index_filename), dtype=input_index_dtype, mode='r')
        zfootprint_idx_bin = np.memmap(os.path.join(static_path, zfootprint_index_filename), dtype=index_dtype, mode='w+',
                                       shape=footprint_idx_bin.shape)

        offset = 8
        in_size = 8
        for i in range(footprint_idx_bin.shape[0]):
            index = footprint_idx_bin[i]
            zindex = zfootprint_idx_bin[i]
            zindex['event_id'] = index['event_id']
            zindex['offset'] = offset
            if decompressed_size:
                zindex['d_size'] = index['size']

            data_in = footprint_map[index['offset']: index['offset'] + index['size']]
            data_out = compress(data_in, compression_level)
            zindex['size'] = len(data_out)
            offset += len(data_out)
            in_size += index['size']
            zfootprint_obj.write(data_out)

        logger.info(f'compressed {os.path.join(static_path, footprint_filename)} from {in_size} B => {offset} B')


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--static-path', help='path to the footprint', default='.')
parser.add_argument('-l', '--compression-level', help='integer from 0 to 9 or -1 controlling the level of compression',
                    default=-1, type=int)
parser.add_argument('--decompressed_size', help='if True add the decompressed size to the index file', action='store_true')
parser.add_argument('-v', '--logging-level', help='logging level (debug:10, info:20, warning:30, error:40, critical:50)',
                    default=20, type=int)


def footprintconvpy():
    kwargs = vars(parser.parse_args())

    # add handler to fm logger
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logging_level = kwargs.pop('logging_level')
    logger.setLevel(logging_level)

    compress_footprint(**kwargs)


if __name__ == '__main__':
    footprintconvpy()
