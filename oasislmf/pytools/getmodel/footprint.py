"""
This file houses the classes that load the footprint data from compressed, binary, and CSV files.
"""
import json
import logging
import pickle
import mmap
import os
from contextlib import ExitStack
from typing import Dict, List, Union
from zlib import decompress

import numpy as np
import pandas as pd
import numba as nb

from oasis_data_manager.df_reader.config import clean_config, InputReaderConfig, get_df_reader
from oasis_data_manager.df_reader.reader import OasisReader
from oasis_data_manager.filestore.backends.base import BaseStorage
from .common import (
    FootprintHeader, EventIndexBin, EventIndexBinZ, Event,
    EventDynamic, footprint_filename, footprint_index_filename,
    zfootprint_filename, zfootprint_index_filename,
    csvfootprint_filename, parquetfootprint_filename,
    parquetfootprint_meta_filename, event_defintion_filename,
    hazard_case_filename, fp_format_priorities,
    parquetfootprint_chunked_dir,
    parquetfootprint_chunked_lookup, footprint_bin_lookup
)
from oasislmf.pytools.common.data import footprint_event_dtype

logger = logging.getLogger(__name__)

uncompressedMask = 1 << 1
intensityMask = 1


CURRENT_DIRECTORY = str(os.getcwd())


@nb.njit(cache=True)
def has_number_in_range(areaperil_ids, min_areaperil_id, max_areaperil_id):
    for apid in areaperil_ids:
        if min_areaperil_id <= apid <= max_areaperil_id:
            return True
    return False


def df_to_numpy(dataframe, dtype, columns={}) -> np.array:
    """

    Args:
        dataframe: DataFrame to convert to numpy
        dtype: numpy dtype of the output ndarray
        columns: optional dict-like object (with get method) mapping np_column => dataframe_column if they are different
    Returns:
        numpy nd array

    >>> dataframe = pd.DataFrame({'a':[1,2], 'b':[0.0, 1.0]})
    >>> dtype = np.dtype([('a', np.int64), ('c', np.float32),])
    >>> columns = {'c': 'b'}
    >>> df_to_numpy(dataframe, dtype, columns)
    array([(1, 0.), (2, 1.)], dtype=[('a', '<i8'), ('c', '<f4')])
    """
    numpy_data = np.empty(len(dataframe), dtype=dtype)
    for np_column in dtype.fields.keys():
        numpy_data[:][np_column] = dataframe[columns.get(np_column, np_column)].to_numpy()
    return numpy_data


@nb.njit(cache=True)
def get_event_map(event_ids):
    event_map = {}
    cur_event_id = event_ids[0]
    last_idx = 0
    for cur_idx in range(1, len(event_ids)):
        if event_ids[cur_idx] != cur_event_id:
            event_map[cur_event_id] = (last_idx, cur_idx)
            cur_event_id = event_ids[cur_idx]
            last_idx = cur_idx
    event_map[cur_event_id] = (last_idx, len(event_ids))
    return event_map


class OasisFootPrintError(Exception):
    """
    Raises exceptions when loading footprints.
    """

    def __init__(self, message: str) -> None:
        """
        The constructor of the OasisFootPrintError class.

        Args:
            message: (str) the message to be raised
        """
        super().__init__(message)


class Footprint:
    """
    This class is the base class for the footprint loaders.

    Attributes:
        storage (BaseStorage): the storage object used to lookup files
        stack (ExitStack): the context manager that combines other context managers and cleanup functions
    """

    def __init__(
            self, storage: BaseStorage,
            df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
            areaperil_ids=None
    ) -> None:
        """
        The constructor for the Footprint class.

        Args:
            storage (BaseStorage): the storage object used to lookup files
        """
        self.storage = storage
        self.stack = ExitStack()
        self.df_engine = df_engine
        if areaperil_ids is not None:
            self.areaperil_ids = np.unique(areaperil_ids)
        else:
            self.areaperil_ids = None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)

    @staticmethod
    def get_footprint_fmt_priorities():
        """
        Get list of footprint file format classes in order of priority.

        Returns: (list) footprint file format classes
        """
        format_to_class = {
            'parquet_chunk': FootprintParquetChunk,
            'parquet': FootprintParquet, 'csv': FootprintCsv,
            'binZ': FootprintBinZ, 'bin': FootprintBin,
            'parquet_dynamic': FootprintParquetDynamic,
        }
        priorities = [format_to_class[fmt] for fmt in fp_format_priorities if fmt in format_to_class]

        return priorities

    @classmethod
    def load(
        cls,
        storage: BaseStorage,
        ignore_file_type=set(),
        df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
        areaperil_ids=None,
        **kwargs
    ):
        """
        Loads the loading classes defined in this file checking to see if the files are in the static path
        whilst doing so. The loading goes through the hierarchy with the following order:

        -> parquet
        -> compressed binary file
        -> binary file
        -> CSV file
        -> parquet (with dynamic generation)

        If the compressed binary file is present, this will be loaded. If it
        is not, then the binary file will be loaded
        and so on.

        Args:
            storage (BaseStorage): the storage object used to lookup files
            ignore_file_type (Set[str]): type of file to be skipped in the hierarchy. This can be a choice of:

            parquet
            json
            z
            bin
            idx

        Returns: (Union[FootprintBinZ, FootprintBin, FootprintCsv]) the loaded class
        """
        for footprint_class in cls.get_footprint_fmt_priorities():
            for filename in footprint_class.footprint_filenames:
                if (not storage.exists(filename) or filename.rsplit('.', 1)[-1] in ignore_file_type):
                    valid = False
                    break
            else:
                valid = True
            if valid:
                for filename in footprint_class.footprint_filenames:
                    logger.debug(f"loading {filename}")
                return footprint_class(storage, df_engine=df_engine, areaperil_ids=areaperil_ids)
        else:
            if storage.isfile("footprint.parquet"):
                raise OasisFootPrintError(
                    message="footprint.parquet needs to be partitioned in order to work, please see: "
                    "oasislmf.pytools.data_layer.conversions.footprint => convert_bin_to_parquet"
                )
            raise OasisFootPrintError(message="no valid footprint found")

    def get_event(self, event_id):
        raise NotImplementedError()

    def get_df_reader(self, filepath, **kwargs) -> OasisReader:
        # load the base df engine config and add the connection parameters
        df_reader_config = clean_config(InputReaderConfig(filepath=filepath, engine=self.df_engine))
        df_reader_config["engine"]["options"]["storage"] = self.storage

        return get_df_reader(df_reader_config, **kwargs)

    @staticmethod
    def prepare_df_data(data_frame: pd.DataFrame) -> np.array:
        """
        Reads footprint data from a parquet file.

        Returns: (np.array) footprint data loaded from the parquet file
        """
        return df_to_numpy(data_frame, Event)

    def areaperil_in_range(self, event_id, events_dict):
        if self.areaperil_ids is None:
            return True  # If its none we are searching all the places
        if event_id not in events_dict:
            return False
        min_areaperil_id, max_areaperil_id = events_dict[event_id]
        return has_number_in_range(
            self.areaperil_ids, min_areaperil_id, max_areaperil_id
        )


class FootprintCsv(Footprint):
    """
    This class is responsible for loading footprint data from CSV.

    Attributes (when in context):
        footprint (np.array[footprint_event_dtype]): event data loaded from the CSV file
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [csvfootprint_filename]

    def __enter__(self):
        self.reader = pd.read_csv()
        self.reader = self.get_df_reader("footprint.csv", dtype=footprint_event_dtype)

        self.num_intensity_bins = self.reader.query(lambda df: df['intensity_bin_id'].max())

        self.has_intensity_uncertainty = self.reader.query(
            lambda df: df.groupby(
                ['event_id', 'areaperil_id']
            ).size().max() > 1
        )

        def _fn(df):
            footprint_index_df = df.groupby('event_id', as_index=False).size()
            footprint_index_df['offset'] = (footprint_index_df['size'].cumsum() - footprint_index_df['size'])

            footprint_index_df.set_index('event_id', inplace=True)
            return footprint_index_df
        self.footprint_index = self.reader.query(_fn).to_dict('index')

        return self

    def get_event(self, event_id):
        """
        Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[footprint_event_dtype]) the event that was extracted
        """
        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        else:
            return self.prepare_df_data(self.reader.filter(lambda df: df[df["event_id"] == event_id]).as_pandas())


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
        footprint_file = self.stack.enter_context(self.storage.with_fileno(footprint_filename))

        self.footprint = mmap.mmap(footprint_file.fileno(), length=0, access=mmap.ACCESS_READ)

        footprint_header = np.frombuffer(bytearray(self.footprint[:FootprintHeader.size]), dtype=FootprintHeader)

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)

        f = self.stack.enter_context(self.storage.with_fileno(footprint_index_filename))
        footprint_mmap = np.memmap(f, dtype=EventIndexBin, mode='r')

        self.footprint_index = pd.DataFrame(footprint_mmap, columns=footprint_mmap.dtype.names).set_index('event_id').to_dict('index')
        try:
            lookup_file = self.storage.with_fileno(footprint_bin_lookup)
            with lookup_file as f:
                lookup = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            df = pickle.loads(lookup)

            self.events_dict = {
                row.event_id: (row.min_areaperil_id, row.max_areaperil_id)
                for row in df.itertuples(index=False)
            }
        except FileNotFoundError:
            self.events_dict = None

        return self

    def get_event(self, event_id):
        """
        Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array(Event)) the event that was extracted
        """
        if self.events_dict:
            if not self.areaperil_in_range(event_id, self.events_dict):
                return None

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
        zfootprint_file = self.stack.enter_context(self.storage.with_fileno(zfootprint_filename))
        self.zfootprint = mmap.mmap(zfootprint_file.fileno(), length=0, access=mmap.ACCESS_READ)

        footprint_header = np.frombuffer(bytearray(self.zfootprint[:FootprintHeader.size]), dtype=FootprintHeader)

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'])
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'] & intensityMask)
        self.uncompressed_size = int((footprint_header['has_intensity_uncertainty'] & uncompressedMask) >> 1)

        if self.uncompressed_size:
            self.index_dtype = EventIndexBinZ
        else:
            self.index_dtype = EventIndexBin

        f = self.stack.enter_context(self.storage.with_fileno(zfootprint_index_filename))

        zfootprint_mmap = np.memmap(f, dtype=self.index_dtype, mode='r')
        self.footprint_index = pd.DataFrame(zfootprint_mmap, columns=zfootprint_mmap.dtype.names).set_index('event_id').to_dict('index')

        try:
            lookup_file = self.storage.with_fileno(footprint_bin_lookup)
            with lookup_file as f:
                lookup = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)
            df = pickle.loads(lookup)

            self.events_dict = {
                row.event_id: (row.min_areaperil_id, row.max_areaperil_id)
                for row in df.itertuples(index=False)
            }
        except FileNotFoundError:
            self.events_dict = None

        return self

    def get_event(self, event_id):
        """
        Gets the event from self.zfootprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """

        event_info = self.footprint_index.get(event_id)
        if event_info is None:
            return
        elif self.events_dict:
            if not self.areaperil_in_range(event_id, self.events_dict):
                return None
        else:
            zdata = self.zfootprint[event_info['offset']: event_info['offset'] + event_info['size']]
            data = decompress(zdata)
            return np.frombuffer(data, Event)


class FootprintParquet(Footprint):
    """
    This class is responsible for loading event data from parquet event data.

    Attributes (when in context):
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames: List[str] = [parquetfootprint_filename, parquetfootprint_meta_filename]

    def __enter__(self):
        with self.storage.open(parquetfootprint_meta_filename, 'r') as outfile:
            meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

        self.num_intensity_bins = int(meta_data['num_intensity_bins'])
        self.has_intensity_uncertainty = int(meta_data['has_intensity_uncertainty'] & intensityMask)

        return self

    def get_event(self, event_id: int):
        """
        Gets the event data from the partitioned parquet data file.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        dir_path = f"footprint.parquet/event_id={event_id}/"
        if self.storage.exists(dir_path):
            reader = self.get_df_reader(dir_path)
            numpy_data = self.prepare_df_data(data_frame=reader.as_pandas())
            return numpy_data
        else:
            return np.empty(0, dtype=Event)


class FootprintParquetChunk(Footprint):
    footprint_filenames = [parquetfootprint_chunked_dir, parquetfootprint_meta_filename, parquetfootprint_chunked_lookup]
    current_reader = None
    current_partition = None

    def __enter__(self):
        with self.storage.open(parquetfootprint_meta_filename, 'r') as outfile:
            meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

        self.num_intensity_bins = int(meta_data['num_intensity_bins'])
        self.has_intensity_uncertainty = int(meta_data['has_intensity_uncertainty'] & intensityMask)

        self.footprint_lookup_map = (self.get_df_reader("footprint_lookup.parquet").as_pandas()
                                     .set_index('event_id')
                                     .to_dict('index'))
        if self.areaperil_ids is not None:
            self.areaperil_ids_filter = [("areaperil_id", "in", list(self.areaperil_ids))]
        else:
            self.areaperil_ids_filter = None
        return self

    def get_event(self, event_id: int):
        """
        Gets the event data from the partitioned
        parquetfootprint_chunked_filename data file.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        event_info = self.footprint_lookup_map.get(event_id)
        if event_info is None:
            return None

        partition = event_info["partition"]
        if partition != self.current_partition:
            self.current_partition = partition
            self.current_df = self.get_df_reader(os.path.join(parquetfootprint_chunked_dir, f"footprint_{partition}.parquet"),
                                                 filters=self.areaperil_ids_filter).as_pandas()
            if len(self.current_df):
                self.event_map = get_event_map(self.current_df["event_id"].to_numpy())
            else:
                self.event_map = {}

        if event_id in self.event_map:
            start_idx, end_idx = self.event_map[event_id]
            return self.prepare_df_data(data_frame=self.current_df[start_idx: end_idx])
        else:
            return None

    def get_events(self, partition):
        reader = self.get_df_reader(os.path.join(parquetfootprint_chunked_dir, f"footprint_{partition}.parquet"))

        df = reader.as_pandas()
        events = [self.prepare_df_data(data_frame=group) for _, group in df.groupby("event_id")]
        return events


class FootprintParquetDynamic(Footprint):
    """
    This class is responsible for loading event data from parquet dynamic event sets and maps
    It will build the footprint from the underlying event defintion and hazard case files

    Attributes (when in context):
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty. Only "no" is supported
        return periods (list): the list of return periods in the model. Not currently used
    """
    footprint_filenames: List[str] = [event_defintion_filename, hazard_case_filename, parquetfootprint_meta_filename]

    def __enter__(self):
        with self.storage.open(parquetfootprint_meta_filename, 'r') as outfile:
            meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

        self.num_intensity_bins = int(meta_data['num_intensity_bins'])
        self.has_intensity_uncertainty = int(meta_data['has_intensity_uncertainty'] & intensityMask)

        self.df_location_sections = pd.read_csv('input/sections.csv')
        self.location_sections = set(list(self.df_location_sections['section_id']))
        if self.areaperil_ids is None:
            self.areaperil_ids = pd.read_csv('input/keys.csv', usecols=['AreaPerilID']).AreaPerilID.unique()

        return self

    def get_event(self, event_id: int):
        """
        Gets the event data from the partitioned parquet data file.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        event_defintion_reader = self.get_df_reader(event_defintion_filename, filters=[("event_id", "==", event_id)])
        df_event_defintion = event_defintion_reader.as_pandas()
        event_sections = list(df_event_defintion['section_id'])
        sections = list(set(event_sections) & self.location_sections)

        if len(sections) > 0:
            df_hazard_case = {}
            for section in sections:
                hazard_case_reader = self.get_df_reader(
                    f'{hazard_case_filename}/section_id={int(section)}',
                    filters=[("areaperil_id", "in", self.areaperil_ids)]
                )
                df_hazard_case[section] = hazard_case_reader.as_pandas()
                df_hazard_case[section]['section_id'] = section
            df_hazard_case = pd.concat(df_hazard_case, ignore_index=True)

            from_cols = ['section_id', 'areaperil_id', 'intensity']
            to_cols = from_cols + ['interpolation', 'return_period']

            df_hazard_case_from = df_hazard_case.merge(
                df_event_defintion, left_on=['section_id', 'return_period'], right_on=['section_id', 'rp_from'])[from_cols].rename(
                    columns={'intensity': 'from_intensity'})

            df_hazard_case_to = df_hazard_case.merge(
                df_event_defintion, left_on=['section_id', 'return_period'], right_on=['section_id', 'rp_to'])[to_cols].rename(
                    columns={'intensity': 'to_intensity'})

            df_footprint = df_hazard_case_from.merge(df_hazard_case_to, on=['section_id', 'areaperil_id'], how='outer')
            df_footprint['from_intensity'] = df_footprint['from_intensity'].fillna(0)

            if len(df_footprint.index) > 0:
                df_footprint['intensity'] = np.floor(df_footprint.from_intensity + (
                    (df_footprint.to_intensity - df_footprint.from_intensity) * df_footprint.interpolation))
                df_footprint['intensity'] = df_footprint['intensity'].astype('int')
                df_footprint = df_footprint.sort_values('intensity', ascending=False)
                df_footprint = df_footprint.drop_duplicates(subset=['areaperil_id'], keep='first')
                df_footprint['intensity_bin_id'] = 0  # Placeholder for intensity bin ID
                df_footprint['probability'] = 1
            else:
                df_footprint.loc[:, 'intensity'] = []
                df_footprint.loc[:, 'intensity_bin_id'] = []
                df_footprint.loc[:, 'probability'] = []

            numpy_data = np.empty(len(df_footprint), dtype=EventDynamic)
            for column in ['areaperil_id', 'intensity_bin_id', 'intensity', 'probability', 'return_period']:
                numpy_data[:][column] = df_footprint[column].to_numpy()

            return numpy_data


if __name__ == "__main__":
    import doctest
    doctest.testmod()
