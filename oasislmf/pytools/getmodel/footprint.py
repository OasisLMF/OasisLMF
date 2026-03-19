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

from numba.typed import Dict, List
from numba.types import Tuple as nb_Tuple
from numba.types import int32 as nb_int32
from numba.types import int64 as nb_int64
from oasislmf.pytools.common.data import nb_areaperil_int, nb_oasis_float, oasis_float, nb_oasis_int, oasis_int, correlations_dtype, items_dtype
from oasislmf.pytools.gulmc.common import haz_arr_type

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
    OFPT_dir, v2_parquet_nested_dir, v2_parquet_flat_dir
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

    FORMAT_TO_CLASS = None  # populated after all subclasses are defined

    @classmethod
    def _get_format_to_class(cls):
        """Get mapping from format name to Footprint subclass.

        Returns:
            dict: format name -> class
        """
        if cls.FORMAT_TO_CLASS is None:
            cls.FORMAT_TO_CLASS = {
                'OFPT': FootprintOFPT,
                'v2_parquet_nested': FootprintV2ParquetNested,
                'v2_parquet_flat': FootprintV2ParquetFlat,
                'parquet_chunk': FootprintParquetChunk,
                'parquet': FootprintParquet, 'csv': FootprintCsv,
                'binZ': FootprintBinZ, 'bin': FootprintBin,
                'parquet_dynamic': FootprintParquetDynamic,
            }
        return cls.FORMAT_TO_CLASS

    @staticmethod
    def get_footprint_fmt_priorities():
        """
        Get list of footprint file format classes in order of priority.

        Returns: (list) footprint file format classes
        """
        format_to_class = Footprint._get_format_to_class()
        priorities = [format_to_class[fmt] for fmt in fp_format_priorities if fmt in format_to_class]

        return priorities

    @classmethod
    def load(
        cls,
        storage: BaseStorage,
        ignore_file_type=set(),
        df_engine="oasis_data_manager.df_reader.reader.OasisPandasReader",
        areaperil_ids=None,
        footprint_format=None,
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
                parquet, json, z, bin, idx
            footprint_format (str, optional): force a specific footprint format by name
                (e.g. 'bin', 'binZ', 'OFPT', 'v2_parquet_nested', 'v2_parquet_flat').
                Bypasses the priority scan when set. Defaults to None (auto-detect).

        Returns: (Footprint) the loaded footprint subclass instance
        """
        if footprint_format is not None:
            format_to_class = cls._get_format_to_class()
            if footprint_format not in format_to_class:
                raise OasisFootPrintError(
                    message=f"Unknown footprint format '{footprint_format}'. "
                    f"Valid formats: {', '.join(format_to_class.keys())}"
                )
            footprint_class = format_to_class[footprint_format]
            logger.info(f"Using forced footprint format: {footprint_format}")
            return footprint_class(storage, df_engine=df_engine, areaperil_ids=areaperil_ids)

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

@nb.njit(cache=True, nopython=True)
def OFPT_process_event(oed_areaperils, event_areaperil_ids, cum_offsets, probabilities, intensity_bin_ids):
    # init data structures
    match_areaperil_ids = np.empty_like(event_areaperil_ids)
    areaperil_to_haz_arr_i = Dict.empty(nb_areaperil_int, nb_oasis_int)

    arr_ptr_start = 0
    arr_ptr_end = 0
    haz_pdf = np.empty(len(probabilities), dtype=haz_arr_type)  # max size
    haz_arr_ptr = List([0])

    match_ap_i = 0
    present_ap_i = 0
    event_ap_i = 0

    while present_ap_i < len(oed_areaperils) and event_ap_i < len(event_areaperil_ids):
        if oed_areaperils[present_ap_i] < event_areaperil_ids[event_ap_i]:
            present_ap_i += 1
            continue
        if oed_areaperils[present_ap_i] > event_areaperil_ids[event_ap_i]:
            event_ap_i += 1
            continue
        match_areaperil_ids[match_ap_i] = oed_areaperils[present_ap_i]
        areaperil_to_haz_arr_i[oed_areaperils[present_ap_i]] = nb_int32(match_ap_i)
        arr_ptr_end = arr_ptr_start + cum_offsets[event_ap_i + 1] - cum_offsets[event_ap_i]
        haz_pdf['probability'][arr_ptr_start: arr_ptr_end] = probabilities[
            cum_offsets[event_ap_i]: cum_offsets[event_ap_i + 1]]
        haz_pdf['intensity_bin_id'][arr_ptr_start: arr_ptr_end] = intensity_bin_ids[
            cum_offsets[event_ap_i]: cum_offsets[event_ap_i + 1]]

        haz_arr_ptr.append(arr_ptr_end)
        arr_ptr_start = arr_ptr_end

        match_ap_i += 1
        present_ap_i += 1
        event_ap_i += 1

    return (match_areaperil_ids[:match_ap_i],
            match_ap_i,
            areaperil_to_haz_arr_i,
            haz_pdf[:arr_ptr_end],
            haz_arr_ptr)


class FootprintOFPT(Footprint):
    footprint_filenames = ["footprint_OFPT"]

    def __enter__(self):
        from oasislmf.pytools.common.ofpt import NpMemMap, OFPTScanner
        from pathlib import Path
        root_dir = "/home/sstruzik/OasisPiWind/model_data/PiWind/"
        OFPT_DIR = Path(root_dir, "footprint_OFPT")
        self.OFPT_scanner = OFPTScanner(OFPT_DIR, NpMemMap)

        self.num_intensity_bins = pd.read_csv('static/intensity_bin_dict.csv')['bin_index'].max() + 1

        return self

    def get_event(self, event_id):
        return self.OFPT_scanner.get_event(event_id, self.areaperil_ids)
        event_header, event_chunks = self.OFPT_scanner.get_event_info(event_id)
        if has_number_in_range(self.areaperil_ids, event_chunks["min_areaperil_id"], event_chunks["max_areaperil_id"]):
            self.OFPT_scanner.get_event(event_header, event_chunks, self.areaperil_ids)
        else:
            return None
        breakpoint()

        # # event_id to file hex
        # event_hex = f"{event_id:08x}"
        # even_file = f"{OFPT_dir}/{event_hex[:2]}/{event_hex[2:4]}/{event_hex[4:6]}.ofpt"
        # if self.cur_file != even_file: # new event we need to load:
        #     self.cur_file = even_file
        # event_header, event_chunks = self.OFPT_scanner.get_event_info(event_id)

    def get_event_items(self, event_id,
                        oed_areaperils,
                        dynamic_footprint):
        """
        areaperil_to_haz_arr_i should not be a dict


        Args:
            event_id:
            oed_areaperils:
            dynamic_footprint:

        Returns:

        """
        event_chunk =  self.get_event(event_id)
        if event_chunk is None:
            return None, 0, None, None, None
        return OFPT_process_event(oed_areaperils, *event_chunk)


class FootprintV2ParquetNested(Footprint):
    """Footprint loader for V2 nested Parquet format.

    Reads from the ``footprint_v2_nested/footprint/`` directory tree.
    Each file holds up to 256 events. Each row contains one (event_id,
    areaperils_id) pair with list columns for probabilities and
    intensity_bin_ids.
    """
    footprint_filenames = [v2_parquet_nested_dir]

    def __enter__(self):
        import pyarrow.parquet as pq

        self.fp_root = os.path.join(self.storage.root_dir, v2_parquet_nested_dir, "footprint")

        # Read metadata from the first parquet file found
        for dirpath, _, filenames in os.walk(self.fp_root):
            for fname in sorted(filenames):
                if fname.endswith(".parquet"):
                    meta = pq.read_metadata(os.path.join(dirpath, fname))
                    custom = meta.schema.to_arrow_schema().metadata
                    self.num_intensity_bins = int(custom[b"oasis:max_intensity_bin_id"])
                    self.has_intensity_uncertainty = int(custom[b"oasis:has_intensity_uncertainty"])
                    return self

        raise OasisFootPrintError("No parquet files found in footprint_v2_nested")

    def _event_path(self, event_id):
        """Compute the parquet file path for a given event_id.

        Args:
            event_id (int): The event identifier.

        Returns:
            str: Absolute path to the parquet file.
        """
        b3 = (event_id >> 24) & 0xFF
        b2 = (event_id >> 16) & 0xFF
        b1 = (event_id >> 8) & 0xFF
        return os.path.join(self.fp_root, f"{b3:02X}", f"{b2:02X}", f"{b1:02X}.parquet")

    def get_event(self, event_id):
        """Get event data in columnar format from nested parquet.

        Args:
            event_id (int): The event identifier.

        Returns:
            tuple or None: (areaperil_ids, cum_offsets, probabilities, intensity_bin_ids)
                or None if the event is not found.
        """
        import pyarrow.parquet as pq
        import pyarrow.compute as pc

        path = self._event_path(event_id)
        if not os.path.exists(path):
            return None

        table = pq.read_table(path, filters=[("event_id", "=", int(event_id))])
        if len(table) == 0:
            return None

        # Sort by areaperils_id
        table = table.sort_by("areaperils_id")

        areaperils_ids = table["areaperils_id"].combine_chunks().to_numpy().astype(areaperil_int)
        prob_list = table["probabilities"]
        bin_list = table["intensity_bin_ids"]

        # Flatten list columns
        flat_probs = pc.list_flatten(prob_list).combine_chunks().to_numpy().astype(np.float32)
        flat_bins = pc.list_flatten(bin_list).combine_chunks().to_numpy().astype(np.int32)

        # Build cumulative offsets (scenarios per areaperil)
        n_per_row = pc.list_value_length(prob_list).combine_chunks().to_numpy()
        cum_offsets = np.zeros(len(n_per_row) + 1, dtype=np.int32)
        np.cumsum(n_per_row, out=cum_offsets[1:])

        return areaperils_ids, cum_offsets, flat_probs, flat_bins

    def get_event_items(self, event_id, oed_areaperils, dynamic_footprint):
        """Get filtered event data ready for gulmc processing.

        Args:
            event_id (int): The event identifier.
            oed_areaperils (np.ndarray): Sorted array of OED areaperil IDs.
            dynamic_footprint: Dynamic footprint flag (unused).

        Returns:
            tuple: (areaperil_ids, count, areaperil_to_haz_arr_i, haz_pdf, haz_arr_ptr)
        """
        event_chunk = self.get_event(event_id)
        if event_chunk is None:
            return None, 0, None, None, None
        return OFPT_process_event(oed_areaperils, *event_chunk)


class FootprintV2ParquetFlat(Footprint):
    """Footprint loader for V2 flat Parquet format.

    Reads from the ``footprint_v2_flat/footprint/`` directory tree.
    Each file holds up to 256 events. Each row contains one
    (event_id, areaperils_id, scenario_index, intensity_index) tuple
    with scalar columns.
    """
    footprint_filenames = [v2_parquet_flat_dir]

    def __enter__(self):
        import pyarrow.parquet as pq

        self.fp_root = os.path.join(self.storage.root_dir, v2_parquet_flat_dir, "footprint")

        # Read metadata from the first parquet file found
        for dirpath, _, filenames in os.walk(self.fp_root):
            for fname in sorted(filenames):
                if fname.endswith(".parquet"):
                    meta = pq.read_metadata(os.path.join(dirpath, fname))
                    custom = meta.schema.to_arrow_schema().metadata
                    self.num_intensity_bins = int(custom[b"oasis:max_intensity_bin_id"])
                    self.has_intensity_uncertainty = int(custom[b"oasis:has_intensity_uncertainty"])
                    return self

        raise OasisFootPrintError("No parquet files found in footprint_v2_flat")

    def _event_path(self, event_id):
        """Compute the parquet file path for a given event_id.

        Args:
            event_id (int): The event identifier.

        Returns:
            str: Absolute path to the parquet file.
        """
        b3 = (event_id >> 24) & 0xFF
        b2 = (event_id >> 16) & 0xFF
        b1 = (event_id >> 8) & 0xFF
        return os.path.join(self.fp_root, f"{b3:02X}", f"{b2:02X}", f"{b1:02X}.parquet")

    def get_event(self, event_id):
        """Get event data in columnar format from flat parquet.

        Args:
            event_id (int): The event identifier.

        Returns:
            tuple or None: (areaperil_ids, cum_offsets, probabilities, intensity_bin_ids)
                or None if the event is not found.
        """
        import pyarrow.parquet as pq

        path = self._event_path(event_id)
        if not os.path.exists(path):
            return None

        table = pq.read_table(path, filters=[("event_id", "=", int(event_id))])
        if len(table) == 0:
            return None

        df = table.to_pandas()
        df = df.sort_values(["areaperils_id", "scenario_index", "intensity_index"])

        areaperils_ids_list = []
        cum_offsets = [0]
        all_probs = []
        all_bins = []

        for apid, grp in df.groupby("areaperils_id", sort=True):
            areaperils_ids_list.append(apid)
            k = int(grp["intensity_index"].max()) + 1

            if k == 1:
                # Each row is one scenario
                n_scenarios = len(grp)
                all_probs.extend(grp["probability"].values.tolist())
                all_bins.extend(grp["intensity_bin_id"].values.tolist())
                cum_offsets.append(cum_offsets[-1] + n_scenarios)
            else:
                # K>1: group by scenario, collect bins in column-major order
                probs = grp[grp["intensity_index"] == 0].sort_values("scenario_index")["probability"].values
                n_scenarios = len(probs)
                all_probs.extend(probs.tolist())
                for iidx in range(k):
                    type_bins = grp[grp["intensity_index"] == iidx].sort_values("scenario_index")["intensity_bin_id"].values
                    all_bins.extend(type_bins.tolist())
                cum_offsets.append(cum_offsets[-1] + n_scenarios)

        return (
            np.array(areaperils_ids_list, dtype=areaperil_int),
            np.array(cum_offsets, dtype=np.int32),
            np.array(all_probs, dtype=np.float32),
            np.array(all_bins, dtype=np.int32),
        )

    def get_event_items(self, event_id, oed_areaperils, dynamic_footprint):
        """Get filtered event data ready for gulmc processing.

        Args:
            event_id (int): The event identifier.
            oed_areaperils (np.ndarray): Sorted array of OED areaperil IDs.
            dynamic_footprint: Dynamic footprint flag (unused).

        Returns:
            tuple: (areaperil_ids, count, areaperil_to_haz_arr_i, haz_pdf, haz_arr_ptr)
        """
        event_chunk = self.get_event(event_id)
        if event_chunk is None:
            return None, 0, None, None, None
        return OFPT_process_event(oed_areaperils, *event_chunk)


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

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'].item())
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'].item() & intensityMask)

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

        self.num_intensity_bins = int(footprint_header['num_intensity_bins'].item())
        self.has_intensity_uncertainty = int(footprint_header['has_intensity_uncertainty'].item() & intensityMask)
        self.uncompressed_size = int((footprint_header['has_intensity_uncertainty'].item() & uncompressedMask) >> 1)

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

        if self.areaperil_ids is not None:
            self.areaperil_ids_filter = [("areaperil_id", "in", self.areaperil_ids)]
        else:
            self.areaperil_ids_filter = None

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
