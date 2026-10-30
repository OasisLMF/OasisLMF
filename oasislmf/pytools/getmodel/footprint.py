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
from oasislmf.pytools.common.data import (
    footprint_event_dtype, areaperil_int, memmap_or_empty, is_first_entry_zlib_compressed
)
from oasislmf.pytools.getmodel.dynamic_footprint import (
    LEVEL_SCALE, MERGE_MAX, MERGE_MAX_INDEPENDENT, MERGE_SUM, collapse, interpolate_brackets, is_sorted, merge_sections)

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
        compressed (bool): if the events are zlib compressed despite the uncompressed file names
    """
    footprint_filenames = [footprint_filename, footprint_index_filename]

    def __enter__(self):
        footprint_file = self.stack.enter_context(self.storage.with_fileno(footprint_filename))

        self.footprint = mmap.mmap(footprint_file.fileno(), length=0, access=mmap.ACCESS_READ)

        footprint_header = np.frombuffer(bytearray(self.footprint[:FootprintHeader.itemsize]), dtype=FootprintHeader)

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'].item())
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'].item() & intensityMask)
        # the idx entries carry the decompressed size when the header says so, even uncompressed
        uncompressed_size = int((footprint_header['has_intensity_uncertainty'].item() & uncompressedMask) >> 1)

        f = self.stack.enter_context(self.storage.with_fileno(footprint_index_filename))
        idx_dtype = EventIndexBinZ if uncompressed_size else EventIndexBin
        self.footprint_index = np.array(memmap_or_empty(f, idx_dtype))
        self.compressed = self._is_compressed()
        if self.compressed:
            logger.warning(f"{footprint_filename} holds zlib compressed events, reading it as a compressed footprint. "
                           f"Compressed footprints should be named {zfootprint_filename} and {zfootprint_index_filename}")
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

    def _is_compressed(self):
        """Whether the events are zlib compressed, as csvtobin writes them when zipping under the
        uncompressed file names.

        Returns: (bool) True if the events are compressed
        """
        return is_first_entry_zlib_compressed(self.footprint, self.footprint_index)

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
        data = self.footprint[int(event_info['offset']): int(event_info['offset']) + int(event_info['size'])]
        if self.compressed:
            data = decompress(data)
        return np.frombuffer(data, Event)


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
        self.footprint_index = np.array(memmap_or_empty(f, self.index_dtype))

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
        self.confluence_areaperils = np.empty(0, np.uint64)
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

        validate_event_definition(df_event_definition)
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

        validate_hazard_case(df_hazard_case)
        # in the order the footprint build walks it, so that no event has to sort it again
        df_hazard_case = df_hazard_case.sort_values(['section_id', 'areaperil_id', 'return_period', 'intensity'],
                                                    kind='stable')
        self.confluence_areaperils = confluence_areaperils(df_hazard_case)
        self.df_hazard_case = df_hazard_case.set_index('section_id')

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

        validate_event_definition(df_event_definition)
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

        sections = self.df_hazard_case.index.intersection(df_event_definition['section_id'].unique()).sort_values()
        if sections.empty:
            return None

        df_hazard_case = self.df_hazard_case.loc[sections].reset_index()
        df_event_definition = df_event_definition[df_event_definition['section_id'].isin(sections)]

        return self._build_footprint(df_hazard_case, df_event_definition, self.confluence_areaperils)

    def _build_footprint(self, df_hazard_case, df_event_definition, confluences=None):
        """Build the interpolated footprint from hazard case and event definition DataFrames.

        Args:
            df_hazard_case: (pd.DataFrame) hazard case data with section_id, areaperil_id, return_period, intensity,
                and optionally probability (float) for stochastic hazard. When probability is absent, all rows are
                treated as deterministic (probability=1). Several rows per (section_id, areaperil_id, return_period)
                form a discrete distribution of intensity; the rp_from and rp_to brackets may hold different
                numbers of them, with different probabilities (see dynamic_footprint.interpolate_brackets).
                Rows sorted by section_id, areaperil_id, return_period, intensity, as the hazard case is at
                load, are used as they are; any other order is sorted first.
            df_event_definition: (pd.DataFrame) event definition with section_id, rp_from, rp_to, interpolation
            confluences: (np.ndarray or None) sorted areaperil_ids lying in more than one section of the
                hazard case, as recorded at load; worked out from df_hazard_case when None.

        A return period bracket that the hazard case does not cover for an areaperil is read as
        intensity 0, so the intensity interpolates from or toward zero rather than being dropped.

        An areaperil lying in several sections of the event is then merged into one distribution
        by the section_merge option (see dynamic_footprint.merge_sections).

        Returns: (np.array[EventDynamic]) the interpolated footprint, or None if empty. Each
            areaperil's rows are together, largest intensity first.
        """
        if 'probability' not in df_hazard_case.columns:
            df_hazard_case = df_hazard_case.assign(probability=1.0)

        # one row per section for the event being built, in section order; a section given twice with
        # different brackets has no single meaning
        df_event_sections = df_event_definition.drop_duplicates(
            ['section_id', 'rp_from', 'rp_to', 'interpolation']).sort_values('section_id')
        event_id = df_event_sections['event_id'].iloc[0] if 'event_id' in df_event_sections else ''
        repeated = df_event_sections['section_id'].duplicated()
        if repeated.any():
            raise OasisFootPrintError(
                f"event {event_id} defines "
                f"section(s) {sorted(df_event_sections.loc[repeated, 'section_id'].unique().tolist())} more than once, "
                f"with different rp_from, rp_to or interpolation")
        event_section = df_event_sections['section_id'].to_numpy(np.int64)
        if len(event_section) == 0 or len(df_hazard_case) == 0:
            return None

        hazard_section = df_hazard_case['section_id'].to_numpy(np.int64)
        hazard_areaperil = df_hazard_case['areaperil_id'].to_numpy(np.uint64)
        hazard_rp = df_hazard_case['return_period'].to_numpy(np.int64)
        hazard_intensity = df_hazard_case['intensity'].to_numpy(np.float64)
        hazard_probability = df_hazard_case['probability'].to_numpy(np.float64)
        if not is_sorted(hazard_section, hazard_areaperil, hazard_rp, hazard_intensity):
            order = np.lexsort((hazard_intensity, hazard_rp, hazard_areaperil, hazard_section))
            hazard_section, hazard_areaperil, hazard_rp = hazard_section[order], hazard_areaperil[order], hazard_rp[order]
            hazard_intensity, hazard_probability = hazard_intensity[order], hazard_probability[order]

        # each hazard row's section in the event; the rows at its rp_from and at its rp_to are the two
        # brackets, each in the hazard case's order: section, areaperil, intensity
        index = np.minimum(np.searchsorted(event_section, hazard_section), len(event_section) - 1)
        in_event = event_section[index] == hazard_section
        interpolation = df_event_sections['interpolation'].to_numpy(np.float64)[index]
        streams = []
        for rp_col in ('rp_from', 'rp_to'):
            rows = np.flatnonzero(in_event & (hazard_rp == df_event_sections[rp_col].to_numpy(np.int64)[index]))
            streams += [hazard_section[rows], hazard_areaperil[rows], hazard_intensity[rows],
                        hazard_probability[rows], interpolation[rows]]
        if len(streams[0]) == 0 and len(streams[5]) == 0:
            return None
        section_id, areaperil_id, intensity, width = interpolate_brackets(*streams)
        probability = width / LEVEL_SCALE
        return_period = df_event_sections['rp_to'].to_numpy(np.int64)[np.searchsorted(event_section, section_id)]

        # areaperils lying in several sections are merged; intensities stay unfloored until then, as a
        # sum of floors would lose up to a unit per section
        if confluences is None:
            confluences = confluence_areaperils(df_hazard_case)
        if len(confluences):
            at = np.minimum(np.searchsorted(confluences, areaperil_id), len(confluences) - 1)
            merged = confluences[at] == areaperil_id
            if merged.any():
                order = np.flatnonzero(merged)
                order = order[np.lexsort((intensity[order], section_id[order], areaperil_id[order]))]
                # the merge re-derives each section's levels from the exact integer widths
                merged_rows = merge_sections(areaperil_id[order], section_id[order], intensity[order], width[order],
                                             return_period[order], SECTION_MERGE_MODES[self.section_merge])
                kept = ~merged
                areaperil_id, intensity, probability, return_period = (
                    np.concatenate((column[kept], merged_column))
                    for column, merged_column in zip((areaperil_id, intensity, probability, return_period), merged_rows))

        # the hazard case is in the int32 range, but an extrapolating weight or a sum of sections can
        # leave it, and the cast in collapse would wrap silently
        int32 = np.iinfo(np.int32)
        out_of_range = ~np.isfinite(intensity) | (intensity < int32.min) | (intensity >= int32.max + 1.0)
        if out_of_range.any():
            raise OasisFootPrintError(
                f"event {event_id}: {int(out_of_range.sum())} interpolated or merged intensity(ies) outside the int32 "
                f"range, e.g. {intensity[out_of_range][0]} at areaperil_id {areaperil_id[out_of_range][0]}; check the "
                f"event's interpolation weights and the section merge ({self.section_merge})")
        areaperil_id, intensity, probability, return_period = collapse(areaperil_id, intensity, probability, return_period)
        footprint = np.empty(len(areaperil_id), dtype=EventDynamic)
        footprint['areaperil_id'] = areaperil_id
        footprint['intensity_bin_id'] = 0
        footprint['intensity'] = intensity
        footprint['probability'] = probability
        footprint['return_period'] = return_period
        return footprint


SECTION_MERGE_MODES = {
    'max': MERGE_MAX,
    'max_independent': MERGE_MAX_INDEPENDENT,
    'sum': MERGE_SUM,
}


def refuse_rows(filename, df, reason, bad):
    """Raise an OasisFootPrintError counting the rows of `df` flagged by `bad`, with the first as an example."""
    example = df.loc[bad].iloc[0].to_dict()
    raise OasisFootPrintError(f"{filename}: {int(bad.sum())} row(s) with {reason}, e.g. {example}")


def not_whole_numbers(values):
    """Flag the values that are missing or not whole numbers; return periods are int32 from the footprint on."""
    values = values.to_numpy(np.float64)
    return ~np.isfinite(values) | (values != np.floor(values))


def validate_event_definition(df_event_definition):
    """Refuse event definition values the footprint build cannot use.

    rp_from and rp_to are matched against the hazard case's return periods, so they must be whole
    numbers; interpolation must be finite, as a NaN weight would make every intensity of the
    section NaN. A weight outside [0, 1] extrapolates and is allowed.
    """
    for rp_col in ('rp_from', 'rp_to'):
        bad = not_whole_numbers(df_event_definition[rp_col])
        if bad.any():
            refuse_rows(event_defintion_filename, df_event_definition, f"an {rp_col} that is missing or not a whole number", bad)
    bad = ~np.isfinite(df_event_definition['interpolation'].to_numpy(np.float64))
    if bad.any():
        refuse_rows(event_defintion_filename, df_event_definition, "an interpolation that is missing or not finite", bad)


def validate_hazard_case(df_hazard_case):
    """Refuse hazard case values the footprint build cannot represent, once, at load.

    Return periods must be whole numbers, as they are int32 from the footprint on. Intensities are
    written as int32 after flooring, so they must be finite and in range; each (section, areaperil,
    return period) distribution needs non-negative probabilities with a positive total, as they are
    normalised per distribution.
    """
    def refuse(reason, bad):
        refuse_rows(hazard_case_filename, df_hazard_case, reason, bad)

    bad = not_whole_numbers(df_hazard_case['return_period'])
    if bad.any():
        refuse("a return_period that is missing or not a whole number", bad)
    intensity = df_hazard_case['intensity'].to_numpy(np.float64)
    int32 = np.iinfo(np.int32)
    bad = ~np.isfinite(intensity) | (intensity < int32.min) | (intensity > int32.max)
    if bad.any():
        refuse("an intensity that is not finite or outside the int32 range", bad)
    if 'probability' in df_hazard_case.columns:
        probability = df_hazard_case['probability'].to_numpy(np.float64)
        bad = ~np.isfinite(probability) | (probability < 0)
        if bad.any():
            refuse("a probability that is not finite or negative", bad)
        total = df_hazard_case.groupby(['section_id', 'areaperil_id', 'return_period'], observed=True)[
            'probability'].transform('sum').to_numpy(np.float64)
        bad = total <= 0
        if bad.any():
            refuse("probabilities adding up to 0 for their (section_id, areaperil_id, return_period)", bad)


def confluence_areaperils(df_hazard_case):
    """The areaperil_ids lying in more than one section of the hazard case, sorted.

    Only those can need a section merge; with none, as in most models, the merge is skipped.
    """
    pairs = df_hazard_case[['areaperil_id', 'section_id']].drop_duplicates()
    counts = pairs['areaperil_id'].value_counts()
    return np.sort(counts.index[counts.to_numpy() > 1].to_numpy(np.uint64))


if __name__ == "__main__":
    import doctest
    doctest.testmod()
