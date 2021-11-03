from zlib import compress, decompress
from contextlib import ExitStack
import os
import mmap
import numba as nb
import numpy as np
import pandas as pd


areaperil_int = np.dtype(os.environ.get('AREAPERIL_TYPE', 'u4'))
oasis_float = np.dtype(os.environ.get('OASIS_FLOAT', 'f4'))

FootprintHeader = nb.from_dtype(np.dtype([('num_intensity_bins', np.int32),
                                          ('has_intensity_uncertainty', np.int32)
                                         ]))

Event = nb.from_dtype(np.dtype([('areaperil_id', areaperil_int),
                                ('intensity_bin_id', np.int32),
                                ('probability', oasis_float)
                              ]))

IndexBin = nb.from_dtype(np.dtype([('event_id', np.int32),
                                   ('offset', np.int64),
                                   ('size', np.int64)
                                  ]))

IndexBinZ = nb.from_dtype(np.dtype([('event_id', np.int32),
                                   ('offset', np.int64),
                                   ('size', np.int64),
                                   ('d_size', np.int64)
                                  ]))
footprint_header_size = 8
uncompressedMask = 1 << 1

zfootprint_filename = 'footprint.bin.z'
zfootprint_index_filename = 'footprint.idx.z'


class Footprint:

    def __init__(self, static_path):
        self.static_path = static_path
        self.stack = ExitStack()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)



class FootprintBinZ(Footprint):

    def __enter__(self):
        file_obj = self.stack.enter_context(open(os.path.join(self.static_path, zfootprint_filename), mode='rb'))
        self.zfootprint = mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ)

        footprint_header = np.frombuffer(bytearray(self.zfootprint[:footprint_header_size]), dtype=FootprintHeader)
        self.uncompressed_size = (footprint_header['has_intensity_uncertainty'] & uncompressedMask) >> 1
        if self.uncompressed_size:
            self.index_dtype = IndexBinZ
        else:
            self.index_dtype = IndexBin

        zfootprint_mmap = np.memmap(os.path.join(static_path, zfootprint_index_filename), dtype=self.index_dtype, mode='r')
        self.footprint_index = pd.DataFrame(zfootprint_mmap, columns=zfootprint_mmap.dtype.names).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        event_info = self.footprint_index[event_id]
        zdata = self.zfootprint[event_info['offset']: event_info['offset']+event_info['size']]
        data = decompress(zdata)
        return np.frombuffer(data, Event)


if __name__ == "__main__":
    static_path = #put your static path here

    with FootprintBinZ(static_path) as footprint_obj:
        print(footprint_obj.footprint_index)
        print(footprint_obj.get_event(1377))
