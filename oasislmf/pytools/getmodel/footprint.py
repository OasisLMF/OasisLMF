"""
This file houses the classes that load the footprint data from compressed, binary, and CSV files.
"""
import mmap
import logging
import os
from contextlib import ExitStack
from zlib import decompress

import numpy as np
import pandas as pd

from .common import (FootprintHeader, EventIndexBin, EventIndexBinZ, Event, EventCSV,
                     footprint_filename, footprint_index_filename, zfootprint_filename, zfootprint_index_filename,
                     csvfootprint_filename)

logger = logging.getLogger(__name__)

uncompressedMask = 1 << 1
intensityMask = 1


class Footprint:
    """
    This class is the base class for the footprint loaders.

    Attributes:
        static_path (str): the path to the static files directory
        stack (ExitStack): the context manager that combines other context managers and cleanup functions
    """
    def __init__(self, static_path) -> None:
        """
        The constructor for the Footprint class.

        Args:
            static_path: (str) the path to the static files directory
        """
        self.static_path = static_path
        self.stack = ExitStack()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)

    @classmethod
    def load(cls, static_path, ignore_file_type=set()):
        """
        Loads the loading classes defined in this file checking to see if the files are in the static path
        whilst doing so. The loading goes through the hierarchy with the following order:

        -> compressed binary file
        -> binary file
        -> CSV file

        If the compressed binary file is present, this will be loaded. If it is not, then the binary file will be loaded
        and so on.

        Args:
            static_path: (str) the path to the static files directory
            ignore_file_type: (Set[str]) type of file to be skipped in the hierarchy

        Returns: (Union[FootprintBinZ, FootprintBin, FootprintCsv]) the loaded class
        """
        priorities = [FootprintBinZ, FootprintBin, FootprintCsv]
        for footprint_class in priorities:
            for filename in footprint_class.footprint_filenames:
                if (not os.path.isfile(os.path.join(static_path, filename))
                        or filename.rsplit('.', 1)[-1] in ignore_file_type):
                    valid = False
                    break
            else:
                valid = True

            if valid:
                for filename in footprint_class.footprint_filenames:
                    logger.debug(f"loading {os.path.join(static_path, filename)}")
                return footprint_class(static_path)
        else:
            raise Exception(f"no valid footprint in {static_path}")

    def get_event(self, event_id):
        raise NotImplementedError()


class FootprintCsv(Footprint):
    """
    This class is responsible for loading footprint data from CSV.

    Attributes (when in context):
        footprint (np.array[EventCSV]): event data loaded from the CSV file
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
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
        """
        Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (EventCSV) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return self.footprint[event_info['offset']: event_info['offset'] + event_info['size']]


class FootprintBin(Footprint):
    """
    This class is responsible loading the event data from the footprint binary files.

    Attributes (when in context):
        footprint (mmap.mmap): loaded data from the binary file which has header and then Event data
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [footprint_filename, footprint_index_filename]

    def __enter__(self):
        self.footprint = self.stack.enter_context(
            mmap.mmap(os.open(os.path.join(self.static_path, footprint_filename), flags=os.O_RDONLY),
                      length=0, access=mmap.ACCESS_READ
                      )
        )
        footprint_header = np.frombuffer(bytearray(self.footprint[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)

        footprint_mmap = np.memmap(
            os.path.join(self.static_path, footprint_index_filename),
            dtype=EventIndexBin,
            mode='r'
        )

        self.footprint_index = pd.DataFrame(
            footprint_mmap,
            columns=footprint_mmap.dtype.names
        ).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        """
        Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (Event) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return np.frombuffer(self.footprint[event_info['offset']: event_info['offset'] + event_info['size']], Event)


class FootprintBinZ(Footprint):
    """
    This class is responsible for loading event data from compressed event data.

    Attributes (when in context):
        zfootprint (mmap.mmap): loaded data from the compressed binary file which has header and then Event data
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        uncompressed_size (int): the size in which the data is when it is decompressed
        index_dtype (Union[EventIndexBinZ, EventIndexBin]) the data type
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [zfootprint_filename, zfootprint_index_filename]

    def __enter__(self):
        self.zfootprint = self.stack.enter_context(mmap.mmap(os.open(os.path.join(self.static_path, zfootprint_filename), flags=os.O_RDONLY),
                                                             length=0, access=mmap.ACCESS_READ))

        footprint_header = np.frombuffer(bytearray(self.zfootprint[:FootprintHeader.size]), dtype=FootprintHeader)
        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)
        self.uncompressed_size = int((footprint_header['has_intensity_uncertainty'] & uncompressedMask) >> 1)
        if self.uncompressed_size:
            self.index_dtype = EventIndexBinZ
        else:
            self.index_dtype = EventIndexBin

        zfootprint_mmap = np.memmap(os.path.join(self.static_path, zfootprint_index_filename), dtype=self.index_dtype, mode='r')
        self.footprint_index = pd.DataFrame(zfootprint_mmap, columns=zfootprint_mmap.dtype.names).set_index('event_id').to_dict('index')
        return self

    def get_event(self, event_id):
        """
        Gets the event from self.zfootprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (Event) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            zdata = self.zfootprint[event_info['offset']: event_info['offset']+event_info['size']]
            data = decompress(zdata)
            return np.frombuffer(data, Event)
