"""This file houses the classes that load the footprint data from compressed, binary, and CSV files."""
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
    parquetfootprint_chunked_lookup, footprint_bin_lookup,
    SECTION_MERGE_OPTIONS, SECTION_MERGE_DEFAULT,
)
from oasislmf.pytools.common.data import footprint_event_dtype, areaperil_int

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
    """Convert a pandas DataFrame to a numpy structured array.

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
    """Raises exceptions when loading footprints."""

    def __init__(self, message: str) -> None:
        """The constructor of the OasisFootPrintError class.

        Args:
            message: (str) the message to be raised
        """
        super().__init__(message)


class Footprint:
    """This class is the base class for the footprint loaders.

    Attributes:
        storage (BaseStorage): the storage object used to lookup files
        stack (ExitStack): the context manager that combines other context managers and cleanup functions
    """

    def __init__(
            self, storage: BaseStorage,
            df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
            areaperil_ids=None
    ) -> None:
        """The constructor for the Footprint class.

        Args:
            storage (BaseStorage): the storage object used to lookup files
            df_engine (str): the engine to use when loading dataframes
            areaperil_ids (list): areaperil_ids that will be useful
        """
        self.storage = storage
        self.stack = ExitStack()
        self.df_engine = df_engine
        if areaperil_ids is not None:
            self.areaperil_ids = np.array(areaperil_ids, dtype=areaperil_int)
        else:
            self.areaperil_ids = None

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)

    @staticmethod
    def get_footprint_fmt_priorities():
        """Get list of footprint file format classes in order of priority.

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
        section_merge=None,
        **kwargs
    ):
        """Loads the loading classes defined in this file checking to see if the files are in the static path
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

            df_engine (str): the engine to use when loading dataframes
            areaperil_ids (list): areaperil_ids to filter the loaded footprint to
            section_merge (str or None): how a dynamic footprint merges an areaperil's intensities
                across the sections of one event, one of SECTION_MERGE_OPTIONS. None takes the
                default. Ignored by the other footprint formats, which have no sections.
            **kwargs: additional keyword arguments, accepted and ignored so that callers can
                forward a wider parameter dict

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
                if footprint_class is FootprintParquetDynamic:
                    return footprint_class(storage, df_engine=df_engine, areaperil_ids=areaperil_ids,
                                           section_merge=section_merge)
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
        """Reads footprint data from a parquet file.

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
    """This class is responsible for loading footprint data from CSV.

    Attributes (when in context):
        footprint (np.array[footprint_event_dtype]): event data loaded from the CSV file
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty
        footprint_index (dict): map of footprint IDs with the index in the data
    """
    footprint_filenames = [csvfootprint_filename]

    def __enter__(self):
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
        """Gets the event from self.footprint based off the event ID passed in.

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
    """This class is responsible loading the event data from the footprint binary files.

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

        footprint_header = np.frombuffer(bytearray(self.footprint[:FootprintHeader.itemsize]), dtype=FootprintHeader)

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'].item())
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'].item() & intensityMask)

        f = self.stack.enter_context(self.storage.with_fileno(footprint_index_filename))
        footprint_mmap = np.memmap(f, dtype=EventIndexBin, mode='r')

        self.footprint_index = np.array(footprint_mmap)
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
        """Gets the event from self.footprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array(Event)) the event that was extracted
        """
        if self.events_dict:
            if not self.areaperil_in_range(event_id, self.events_dict):
                return None

        idx = np.searchsorted(self.footprint_index['event_id'], event_id)
        if idx >= len(self.footprint_index) or self.footprint_index['event_id'][idx] != event_id:
            return None
        event_info = self.footprint_index[idx]
        return np.frombuffer(self.footprint[int(event_info['offset']): int(event_info['offset']) + int(event_info['size'])], Event)


class FootprintBinZ(Footprint):
    """This class is responsible for loading event data from compressed event data.

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

        footprint_header = np.frombuffer(bytearray(self.zfootprint[:FootprintHeader.itemsize]), dtype=FootprintHeader)

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'].item())
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'].item() & intensityMask)
        self.uncompressed_size = int((footprint_header['has_intensity_uncertainty'].item() & uncompressedMask) >> 1)

        if self.uncompressed_size:
            self.index_dtype = EventIndexBinZ
        else:
            self.index_dtype = EventIndexBin

        f = self.stack.enter_context(self.storage.with_fileno(zfootprint_index_filename))

        zfootprint_mmap = np.memmap(f, dtype=self.index_dtype, mode='r')
        self.footprint_index = np.array(zfootprint_mmap)

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
        """Gets the event from self.zfootprint based off the event ID passed in.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        idx = np.searchsorted(self.footprint_index['event_id'], event_id)
        if idx >= len(self.footprint_index) or self.footprint_index['event_id'][idx] != event_id:
            return None
        event_info = self.footprint_index[idx]
        if self.events_dict:
            if not self.areaperil_in_range(event_id, self.events_dict):
                return None
        zdata = self.zfootprint[int(event_info['offset']): int(event_info['offset']) + int(event_info['size'])]
        data = decompress(zdata)
        return np.frombuffer(data, Event)


class FootprintParquet(Footprint):
    """This class is responsible for loading event data from parquet event data.

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

        if self.areaperil_ids is not None:
            self.areaperil_ids_filter = [("areaperil_id", "in", self.areaperil_ids)]
        else:
            self.areaperil_ids_filter = None

        return self

    def get_event(self, event_id: int):
        """Gets the event data from the partitioned parquet data file.

        Args:
            event_id: (int) the ID belonging to the Event being extracted

        Returns: (np.array[Event]) the event that was extracted
        """
        dir_path = f"footprint.parquet/event_id={event_id}/"
        if self.storage.exists(dir_path):
            reader = self.get_df_reader(dir_path, filters=self.areaperil_ids_filter)
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
        """Gets the event data from the partitioned footprint_<partition>.parquet files
        under parquetfootprint_chunked_dir.

        Args:
            event_id (int): the ID belonging to the Event being extracted

        Returns:
            np.array[Event]: the event that was extracted, or None if the event is absent from the
                lookup map or from its partition
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


def validate_section_merge(section_merge):
    """Resolve the section_merge option, failing at load on an unknown value.

    Raises here rather than at the first event with a multi-section areaperil: a portfolio may
    hold no such areaperil, and a mistyped option must not go unnoticed until one does.
    """
    if section_merge is None:
        return SECTION_MERGE_DEFAULT
    if section_merge not in SECTION_MERGE_OPTIONS:
        raise OasisFootPrintError(
            f"unknown section_merge {section_merge!r}, valid options are {', '.join(SECTION_MERGE_OPTIONS)}")
    return section_merge


class FootprintParquetDynamic(Footprint):
    """This class is responsible for loading event data from parquet dynamic event sets and maps
    It will build the footprint from the underlying event defintion and hazard case files

    Either file may be partitioned by section_id (i.e. be a directory of section_id=N/
    subdirectories) or be a single unpartitioned parquet file, independently of the other.
    Both layouts are read the same way, by pushing a section_id filter down to the reader.

    The hazard case is bulk-loaded at __enter__ for this portfolio's sections and areaperils,
    which is all get_event ever needs of it. The event definition is bulk-loaded too when it
    is partitioned, giving indexed per-event lookups; when it is unpartitioned it is instead
    filtered to the event per get_event call.

    Both files may also be sparse: a section that is absent from the event definition has no
    events affecting it, and a section absent from the hazard case is unaffected by the
    modelled perils. Either absence makes the locations of that section not at risk, which is
    reported to the caller as get_event returning None rather than as an error.

    Attributes (when in context):
        num_intensity_bins (int): number of intensity bins in the data
        has_intensity_uncertainty (bool): if the data has uncertainty. Only "no" is supported
        return periods (list): the list of return periods in the model. Not currently used
    """
    footprint_filenames: List[str] = [event_defintion_filename, hazard_case_filename, parquetfootprint_meta_filename]

    def __init__(self, storage: BaseStorage, df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
                 areaperil_ids=None, section_merge=None) -> None:
        """The constructor for the FootprintParquetDynamic class.

        Args:
            storage (BaseStorage): the storage object used to lookup files
            df_engine (str): the engine to use when loading dataframes
            areaperil_ids (list): areaperil_ids that will be useful
            section_merge (str or None): how an areaperil's intensities are merged across the
                sections of one event, one of SECTION_MERGE_OPTIONS. None takes SECTION_MERGE_DEFAULT.

        Raises:
            OasisFootPrintError: if section_merge is not a known option
        """
        super().__init__(storage, df_engine=df_engine, areaperil_ids=areaperil_ids)
        self.section_merge = validate_section_merge(section_merge)

    def _is_partitioned_by_section(self, filename):
        """Check whether a parquet file is a dataset partitioned by section_id.

        This is a property of the model data alone: it must not depend on which sections the
        portfolio happens to use, as a partitioned file is allowed to hold none of them.

        Args:
            filename (str): the parquet file or dataset directory to inspect

        Returns: (bool) True if the file is a directory of section_id=N/ partitions
        """
        if not self.storage.exists(filename) or self.storage.isfile(filename):
            return False

        return any(
            os.path.basename(str(entry).rstrip('/')).startswith('section_id=')
            for entry in self.storage.listdir(filename)
        )

    def __enter__(self):
        with self.storage.open(parquetfootprint_meta_filename, 'r') as outfile:
            meta_data: Dict[str, Union[int, bool]] = json.load(outfile)

        self.num_intensity_bins = int(meta_data['num_intensity_bins'])
        self.has_intensity_uncertainty = int(meta_data['has_intensity_uncertainty'] & intensityMask)

        self.df_location_sections = pd.read_csv('input/sections.csv')
        self.location_sections = set(list(self.df_location_sections['section_id']))
        if self.areaperil_ids is None:
            self.areaperil_ids = pd.read_csv('input/keys.csv', usecols=['AreaPerilID']).AreaPerilID.unique()

        self.event_definition_partitioned = self._is_partitioned_by_section(event_defintion_filename)

        self.areaperil_ids_filter = [("areaperil_id", "in", self.areaperil_ids)]
        self.absent_sections_reported = {}

        self.df_event_definition = None
        self.df_hazard_case = None
        self.event_set = set()

        if self.event_definition_partitioned:
            self.get_event = self._get_event_partitioned
            if not self._load_event_definitions():
                return self
        else:
            self.get_event = self._get_event_flat

        self._load_hazard_case()
        return self

    def _report_absent_sections(self, filename, absent):
        reported = self.absent_sections_reported.setdefault(filename, set())
        newly_absent = sorted(set(absent) - reported)
        if not newly_absent:
            return

        reported.update(newly_absent)
        logger.info(f"sections {newly_absent} have no data in {filename} for this portfolio, "
                    f"so their locations are treated as not at risk")

    def _read_sections(self, filename, sections, filters=None):
        """Read the requested sections of a parquet file, treating absent sections as empty.

        Args:
            filename (str): the parquet file or dataset directory to read
            sections (iterable): the section_ids to read
            filters (list): optional pyarrow filters pushed down to the reader

        Returns: (pd.DataFrame) the concatenated sections with a section_id column, or an
            empty DataFrame if none of the sections holds any data.
        """
        sections = sorted(int(section) for section in sections)
        if not sections:
            return pd.DataFrame()

        section_filters = (filters or []) + [("section_id", "in", sections)]
        df_sections = self.get_df_reader(filename, filters=section_filters).as_pandas()
        # a hive partition key is read back as a category, which does not survive the
        # fillna and merge in _build_footprint the way the flat file's integer column does
        df_sections['section_id'] = df_sections['section_id'].astype('int64')
        present = set() if df_sections.empty else set(df_sections['section_id'])
        self._report_absent_sections(filename, set(sections) - present)
        return df_sections

    def _load_event_definitions(self):
        """Bulk-load the event definitions of this portfolio's sections.

        Returns: (bool) False if no event in the model data affects the portfolio at all
        """
        df_event_definition = self._read_sections(event_defintion_filename, self.location_sections)
        if df_event_definition.empty:
            logger.warning(f"no section of this portfolio is in {event_defintion_filename}, "
                           f"so no event affects it and every loss will be zero")
            return False

        # sorted so that the per-event .loc lookups in _get_event_partitioned are indexed;
        # a stable sort keeps the row order the model data was written in
        self.df_event_definition = df_event_definition.set_index('event_id').sort_index(kind='stable')
        self.event_set = set(df_event_definition['event_id'].unique())
        return True

    def _load_hazard_case(self):
        """Bulk-load the hazard of this portfolio's sections and areaperils.

        The hazard case does not depend on the event, so it is read once here for both
        get_event paths rather than per call.
        """
        df_hazard_case = self._read_sections(
            hazard_case_filename, self.location_sections, filters=self.areaperil_ids_filter)
        if df_hazard_case.empty:
            # the modelled perils leave every section of this portfolio unaffected
            logger.warning(f"no section of this portfolio has hazard in {hazard_case_filename}, "
                           f"so every loss will be zero")
            return

        self.df_hazard_case = df_hazard_case.set_index('section_id').sort_index(kind='stable')

    def _get_event_partitioned(self, event_id):
        """Fast path: the event definition is indexed in memory, so look the event up."""
        if event_id not in self.event_set:
            return None

        return self._build_event(self.df_event_definition.loc[[event_id]].reset_index())

    def _get_event_flat(self, event_id):
        """Lazy path: an unpartitioned event definition is filtered to the event per call."""
        df_event_definition = self.get_df_reader(
            event_defintion_filename, filters=[("event_id", "==", event_id)]).as_pandas()
        if df_event_definition.empty:
            return None

        return self._build_event(df_event_definition)

    def _build_event(self, df_event_definition):
        """Build one event's footprint from the sections it affects that carry hazard.

        Args:
            df_event_definition (pd.DataFrame): the event definition rows of a single event

        Returns: (np.array[EventDynamic]) the footprint, or None if no section of the event
            is at risk. The hazard case holds only this portfolio's sections, so intersecting
            with its index drops both the sections outside the portfolio and those the
            modelled perils leave unaffected.
        """
        if self.df_hazard_case is None:
            return None

        sections = self.df_hazard_case.index.intersection(df_event_definition['section_id'].unique())
        if sections.empty:
            return None

        df_hazard_case = self.df_hazard_case.loc[sections].reset_index()
        df_event_definition = df_event_definition[df_event_definition['section_id'].isin(sections)]

        return self._build_footprint(df_hazard_case, df_event_definition)

    def _build_footprint(self, df_hazard_case, df_event_definition):
        """Build the interpolated footprint from hazard case and event definition DataFrames.

        Args:
            df_hazard_case: (pd.DataFrame) hazard case data with section_id, areaperil_id, return_period, intensity,
                and optionally probability (float) for stochastic hazard. When probability is absent, all rows are
                treated as deterministic (probability=1). Several rows per (section_id, areaperil_id, return_period)
                form a discrete distribution of intensity; the rp_from and rp_to brackets may hold different
                numbers of them, with different probabilities (see pair_by_level).
            df_event_definition: (pd.DataFrame) event definition with section_id, rp_from, rp_to, interpolation

        A return period bracket that the hazard case does not cover for an areaperil is read as
        intensity 0, so the intensity interpolates from or toward zero rather than being dropped.

        An areaperil lying in several sections of the event is then merged into one distribution
        by the section_merge option (see merge_sections).

        Returns: (np.array[EventDynamic]) the interpolated footprint, or None if empty
        """
        if 'probability' not in df_hazard_case.columns:
            df_hazard_case = df_hazard_case.assign(probability=1.0)

        # one row per section for the event being built
        df_event_sections = df_event_definition.drop_duplicates('section_id')
        atom_cols = ['group', 'member', 'intensity', 'probability']

        # member 0 is the rp_from bracket, member 1 the rp_to bracket; a (section, areaperil)
        # pair is packed into one integer group key, areaperil_id being 32 bits
        brackets = []
        for member, rp_col in enumerate(['rp_from', 'rp_to']):
            bracket = df_hazard_case.merge(df_event_sections[['section_id', rp_col]],
                                           left_on=['section_id', 'return_period'], right_on=['section_id', rp_col])
            bracket['group'] = (bracket['section_id'].to_numpy(np.int64) << 32) | bracket['areaperil_id'].to_numpy(np.int64)
            brackets.append(bracket.assign(member=member)[atom_cols])
        # an areaperil missing from one bracket reads it as intensity 0, a single atom
        groups = np.union1d(brackets[0]['group'].unique(), brackets[1]['group'].unique())
        for member, bracket in enumerate(brackets):
            missing = np.setdiff1d(groups, bracket['group'].unique(), assume_unique=True)
            if len(missing):
                brackets[member] = pd.concat([bracket, pd.DataFrame(
                    {'group': missing, 'member': member, 'intensity': 0, 'probability': 1.0})])

        atoms = pd.concat(brackets, ignore_index=True)
        if len(atoms) == 0:
            return None
        paired = pair_by_level(atoms['group'].to_numpy(), atoms['member'].to_numpy(),
                               atoms['intensity'].to_numpy(), atoms['probability'].to_numpy())

        # both members are present at every level, member 0 first
        group = paired['group'][0::2]
        df_footprint = pd.DataFrame({
            'section_id': group >> 32,
            'areaperil_id': group & 0xFFFFFFFF,
            'probability': paired['probability'][0::2],
            'from_intensity': paired['intensity'][0::2],
            'to_intensity': paired['intensity'][1::2],
        })
        df_event_sections = df_event_sections.set_index('section_id')
        interpolation = df_footprint['section_id'].map(df_event_sections['interpolation'])
        df_footprint['return_period'] = df_footprint['section_id'].map(df_event_sections['rp_to'])
        # left unfloored until the sections are merged: a sum of floors loses up to a unit per section
        df_footprint['intensity'] = df_footprint.from_intensity + (
            (df_footprint.to_intensity - df_footprint.from_intensity) * interpolation)

        df_footprint = merge_sections(df_footprint, self.section_merge)
        df_footprint['intensity'] = np.floor(df_footprint['intensity']).astype('int')

        # Collapse realisations that produce the same intensity after interpolation,
        # summing their probabilities, then normalise per areaperil to absorb float drift.
        df_footprint = df_footprint.groupby(
            ['areaperil_id', 'intensity', 'return_period'], as_index=False
        )['probability'].sum()
        df_footprint['probability'] /= df_footprint.groupby('areaperil_id')['probability'].transform('sum')
        df_footprint = df_footprint.sort_values(['areaperil_id', 'intensity'], ascending=[True, False])

        df_footprint['intensity_bin_id'] = 0

        return df_to_numpy(df_footprint, EventDynamic)


# cumulative probabilities are matched across distributions as integers in units of
# 10**-LEVEL_DECIMALS: 1/3 + 1/3 and 2/3 must meet, not leave a sliver of probability between them
LEVEL_DECIMALS = 12
LEVEL_SCALE = 10 ** LEVEL_DECIMALS


def pair_by_level(group, member, intensity, probability):
    """Pair several discrete distributions of intensity by cumulative probability level.

    Within each group, every member is a discrete distribution given as (intensity, probability)
    atoms. The members' cumulative probability breakpoints are merged into one sorted set of
    levels; each interval between consecutive levels becomes one output row per member, carrying
    that member's atom at the level and the interval's width as its probability. Combining the
    members row by row (interpolating, taking the larger, ...) then couples them comonotonically:
    the j-th quantile of one meets the j-th quantile of the other, whatever the number of atoms
    or their probabilities.

    With the same number of atoms at the same probabilities in every member this is exactly
    pairing by rank. Each member's probabilities are normalised to sum to 1.

    Args:
        group (np.ndarray[int]): group of each atom.
        member (np.ndarray[int]): member of each atom, a distribution within its group.
        intensity (np.ndarray): intensity of each atom.
        probability (np.ndarray[float]): probability of each atom within its member.

    Returns:
        dict[str, np.ndarray]: one entry per (group, level interval, member) of the group, sorted
            by group, level and member: group, level (the interval's upper bound, in units of
            1 / LEVEL_SCALE), member, probability (the interval width), intensity, and atom, the
            position in the input arrays of the atom it carries.
    """
    order = np.lexsort((intensity, member, group))
    group, member, probability = group[order], member[order], probability[order]

    # one segment per (group, member), atoms ascending in intensity
    segment_start = _segments(group, member)
    segment = np.repeat(np.arange(len(segment_start)), np.diff(np.r_[segment_start, len(group)]))

    # cumulative probability per segment, normalised, as an integer level; per segment so the
    # running sum never accumulates the float error of the whole event
    cumulative = pd.Series(probability).groupby(segment).cumsum().to_numpy()
    total = np.add.reduceat(probability, segment_start)
    level = np.rint(cumulative / total[segment] * LEVEL_SCALE).astype(np.int64)
    level[np.r_[segment_start[1:] - 1, len(level) - 1]] = LEVEL_SCALE  # each segment ends at exactly 1

    # the union of the members' levels, per group
    level_order = np.lexsort((level, group))
    level_group, level_value = group[level_order], level[level_order]
    distinct = np.empty(len(level_order), dtype=bool)
    distinct[:1] = True
    distinct[1:] = (level_group[1:] != level_group[:-1]) | (level_value[1:] != level_value[:-1])
    level_group, level_value = level_group[distinct], level_value[distinct]
    first_of_group = np.r_[True, level_group[1:] != level_group[:-1]]
    width = np.diff(level_value, prepend=0)
    width[first_of_group] = level_value[first_of_group]

    grid_level, grid_segment = _cross_with_segments(level_group, group[segment_start])

    # each member's atom at each level: the first in its segment whose level reaches it. Levels
    # are ranked so (segment, level) packs into one sortable int64 key without overflow
    ranks, level_rank = np.unique(np.r_[level, level_value], return_inverse=True)
    atom_key = segment * (len(ranks) + 1) + level_rank[:len(level)]
    grid_key = grid_segment * (len(ranks) + 1) + level_rank[len(level):][grid_level]
    atom = np.searchsorted(atom_key, grid_key, side='left')

    return {
        'group': level_group[grid_level],
        'level': level_value[grid_level],
        'member': member[atom],
        'probability': width[grid_level] / LEVEL_SCALE,
        'intensity': intensity[order][atom],
        'atom': order[atom],
    }


def _segments(group, member):
    """Start of each run of equal (group, member) in arrays sorted by group then member."""
    new_segment = np.empty(len(group), dtype=bool)
    new_segment[:1] = True
    new_segment[1:] = (group[1:] != group[:-1]) | (member[1:] != member[:-1])
    return np.flatnonzero(new_segment)


def _cross_with_segments(point_group, segment_group):
    """Cross every point with every segment of the same group.

    Both inputs must be sorted by group. Returns the point index and the segment index of each
    pair, ordered by point then segment.
    """
    first_segment = np.searchsorted(segment_group, point_group, side='left')
    segment_count = np.searchsorted(segment_group, point_group, side='right') - first_segment
    pair_point = np.repeat(np.arange(len(point_group)), segment_count)
    offset = np.arange(len(pair_point)) - np.repeat(np.cumsum(segment_count) - segment_count, segment_count)
    return pair_point, first_segment[pair_point] + offset


def _merge_comonotone(df, combine_sum):
    """Pair the sections' distributions by cumulative level, then take the largest, or the sum.

    The return period kept at each level is the section giving the largest intensity there,
    the larger return period on a tie.
    """
    return_period = df['return_period'].to_numpy()
    paired = pair_by_level(df['areaperil_id'].to_numpy(np.int64), df['section_id'].to_numpy(np.int64),
                           df['intensity'].to_numpy(np.float64), df['probability'].to_numpy(np.float64))
    group, level, intensity = paired['group'], paired['level'], paired['intensity']
    rp = return_period[paired['atom']]

    block_start = _segments(group, level)
    block_end = np.r_[block_start[1:], len(group)] - 1
    block = np.repeat(np.arange(len(block_start)), np.diff(np.r_[block_start, len(group)]))
    winner = np.lexsort((rp, intensity, block))[block_end]

    return pd.DataFrame({
        'areaperil_id': group[block_start],
        'intensity': np.add.reduceat(intensity, block_start) if combine_sum else intensity[winner],
        'probability': paired['probability'][block_start],
        'return_period': rp[winner],
    })


def merge_sections_max(df):
    """Largest intensity, the sections' uncertainties moving together (comonotone)."""
    return _merge_comonotone(df, combine_sum=False)


def merge_sections_sum(df):
    """Sum of the sections' intensities, their uncertainties moving together (comonotone)."""
    return _merge_comonotone(df, combine_sum=True)


def merge_sections_max_independent(df):
    """Largest intensity, the sections' uncertainties independent.

    P(max <= x) is the product of the sections' P(D <= x), evaluated at every intensity any of
    them can take: at most as many atoms as the sections have together, no cartesian product.
    The return period of an atom is that of the section it comes from, the larger on a tie.
    """
    df = df.sort_values(['areaperil_id', 'section_id', 'intensity'], kind='stable')
    group = df['areaperil_id'].to_numpy(np.int64)
    intensity = df['intensity'].to_numpy(np.float64)
    segment_start = _segments(group, df['section_id'].to_numpy(np.int64))
    segment = np.repeat(np.arange(len(segment_start)), np.diff(np.r_[segment_start, len(group)]))
    probability = df['probability'].to_numpy(np.float64)
    cdf = pd.Series(probability).groupby(segment).cumsum().to_numpy() / np.add.reduceat(probability, segment_start)[segment]

    # the intensities any section can take, per areaperil, with the largest return period giving each
    values = df.groupby(['areaperil_id', 'intensity'], sort=True)['return_period'].max().reset_index()
    value_group, value = values['areaperil_id'].to_numpy(np.int64), values['intensity'].to_numpy(np.float64)

    # each section's P(D <= v): its last atom at or below v, packed into one sortable key per segment
    ranks, rank = np.unique(np.r_[intensity, value], return_inverse=True)
    atom_key = segment * (len(ranks) + 1) + rank[:len(intensity)]
    pair_value, pair_segment = _cross_with_segments(value_group, group[segment_start])
    below = np.searchsorted(atom_key, pair_segment * (len(ranks) + 1) + rank[len(intensity):][pair_value], side='right') - 1
    section_cdf = np.where(below >= segment_start[pair_segment], cdf[np.maximum(below, 0)], 0.0)

    value_start = np.flatnonzero(np.r_[True, pair_value[1:] != pair_value[:-1]])
    merged_cdf = np.multiply.reduceat(section_cdf, value_start)
    first_of_group = np.r_[True, value_group[1:] != value_group[:-1]]
    merged_probability = np.diff(merged_cdf, prepend=0.0)
    merged_probability[first_of_group] = merged_cdf[first_of_group]

    keep = merged_probability > 1e-15
    return pd.DataFrame({
        'areaperil_id': value_group[keep],
        'intensity': value[keep],
        'probability': merged_probability[keep],
        'return_period': values['return_period'].to_numpy()[keep],
    })


SECTION_MERGES = {
    'max': merge_sections_max,
    'max_independent': merge_sections_max_independent,
    'sum': merge_sections_sum,
}


def merge_sections(df_footprint, section_merge):
    """Merge, per areaperil, the distributions of the sections of one event it lies in.

    An areaperil in a single section passes through unchanged, whatever the option.

    Args:
        df_footprint (pd.DataFrame): section_id, areaperil_id, intensity, probability and
            return_period, a distribution per (section_id, areaperil_id).
        section_merge (str): one of SECTION_MERGE_OPTIONS.

    Returns:
        pd.DataFrame: areaperil_id, intensity, probability and return_period, a distribution
            per areaperil_id.
    """
    columns = ['areaperil_id', 'intensity', 'probability', 'return_period']
    n_sections = df_footprint.groupby('areaperil_id')['section_id'].transform('nunique')
    multi = n_sections.to_numpy() > 1
    if not multi.any():
        return df_footprint[columns]
    return pd.concat([df_footprint.loc[~multi, columns], SECTION_MERGES[section_merge](df_footprint.loc[multi])],
                     ignore_index=True)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
