from zlib import decompress
from contextlib import ExitStack
import os
import mmap
import numpy as np
import pandas as pd

from .common import (FootprintHeader, EventIndexBin, EventIndexBinZ, Event, EventCSV,
                     footprint_filename, footprint_index_filename, zfootprint_filename, zfootprint_index_filename,
                     csvfootprint_filename)

uncompressedMask = 1 << 1
intensityMask = 1


class Footprint:
    def __init__(self, static_path):
        self.static_path = static_path
        self.stack = ExitStack()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)

    @classmethod
    def load(cls, static_path):
        priorities = [FootprintBinZ, FootprintBin, FootprintCsv]
        for footprint_class in priorities:
            for filename in footprint_class.footprint_filenames:
                if not os.path.isfile(os.path.join(static_path, filename)):
                    valid = False
                    break
            else:
                valid = True

            if valid:
                return footprint_class(static_path)
        else:
            raise Exception(f"no valid footprint in {static_path}")

    def get_event(self, event_id):
        raise NotImplementedError()


class FootprintCsv(Footprint):
    footprint_filenames = [csvfootprint_filename]

    def __enter__(self):
        self.footprint = np.genfromtxt(os.path.join(self.static_path, "footprint.csv"), dtype=EventCSV, delimiter=",")
        self.num_intensity_bins = max(self.footprint['intensity_bin_id'])

        footprint_df = pd.DataFrame(self.footprint, columns=self.footprint.dtype.names)
        self.has_intensity_uncertainty = footprint_df.groupby(['event_id','areaperil_id']).size().max() > 1

        footprint_index_df = footprint_df.groupby('event_id', as_index=False).size()
        footprint_index_df['offset'] = footprint_index_df['size'].cumsum() - footprint_index_df['size']
        footprint_index_df.set_index('event_id', inplace=True)
        self.footprint_index = footprint_index_df.to_dict('index')

        return self

    def get_event(self, event_id):
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return self.footprint[event_info['offset']: event_info['offset'] + event_info['size']]


class FootprintBin(Footprint):
    footprint_filenames = [footprint_filename, footprint_index_filename]

    def __enter__(self):
        self.footprint = self.stack.enter_context(mmap.mmap(os.open(os.path.join(self.static_path, footprint_filename), flags=os.O_RDONLY),
                                                             length=0, access=mmap.ACCESS_READ))
        footprint_header = np.frombuffer(bytearray(self.footprint[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty  = int(footprint_header['has_intensity_uncertainty'] & intensityMask)

        footprint_mmap = np.memmap(os.path.join(self.static_path, footprint_index_filename), dtype=EventIndexBin, mode='r')
        self.footprint_index = pd.DataFrame(footprint_mmap, columns=footprint_mmap.dtype.names).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return np.frombuffer(self.footprint[event_info['offset']: event_info['offset'] + event_info['size']], Event)


class FootprintBinZ(Footprint):
    footprint_filenames = [zfootprint_filename, zfootprint_index_filename]

    def __enter__(self):
        self.zfootprint = self.stack.enter_context(mmap.mmap(os.open(os.path.join(self.static_path, zfootprint_filename), flags=os.O_RDONLY),
                                                             length=0, access=mmap.ACCESS_READ))

        footprint_header = np.frombuffer(bytearray(self.zfootprint[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty  = int(footprint_header['has_intensity_uncertainty'] & intensityMask)
        self.uncompressed_size = int((footprint_header['has_intensity_uncertainty'] & uncompressedMask) >> 1)
        if self.uncompressed_size:
            self.index_dtype = EventIndexBinZ
        else:
            self.index_dtype = EventIndexBin

        zfootprint_mmap = np.memmap(os.path.join(self.static_path, zfootprint_index_filename), dtype=self.index_dtype, mode='r')
        self.footprint_index = pd.DataFrame(zfootprint_mmap, columns=zfootprint_mmap.dtype.names).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            zdata = self.zfootprint[event_info['offset']: event_info['offset']+event_info['size']]
            data = decompress(zdata)
            return np.frombuffer(data, Event)
