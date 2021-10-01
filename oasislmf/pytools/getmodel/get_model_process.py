import numba as nb
import numpy as np
import struct
import sys
from numba import jit, int64
from pandas import DataFrame, merge, concat
from typing import Optional, Tuple, Any

from oasislmf.utils.data import merge_dataframes
from .data_access_layers import FileDataAccessLayer
from .descriptors import HeaderTypeDescriptor
from .enums import FileTypeEnum
from .loader_mixin import FileLoader


import line_profiler
profile = line_profiler.LineProfiler()


@jit((int64[:], int64[:], int64[:]), nopython=True)
def define_data_series(vul_ids, inten_ids, damage_bin_maxs):
    vun_list = []

    for i in range(len(vul_ids)):
        vul_id = vul_ids[i]
        inten_id = inten_ids[i]

        for x in range(1, damage_bin_maxs[i] + 1):
            vun_list.append((
                vul_id,
                inten_id,
                x,
                0.0,
            ))
    return vun_list


@jit((int64, int64[:]), nopython=True)
def get_finish_position(identifier, data_vector):
    starting_point = 0
    finishing_point = 0
    match = False

    for i in data_vector:
        if i == identifier:
            match = True
        elif match is True:
            break
        else:
            starting_point += 1
        finishing_point += 1
    return starting_point, finishing_point


class GetModelProcess:
    """
    This class is responsible for loading data from a range of sources and merging them to build a model.

    Attributes:
        data_path (str): the path to the data files needed to construct the model
        model (Optional[DataFrame]): the model for K-tools
        events (Optional[DataFrame]): events preloaded (if None, will load from a file)
        stream_type (int): the type of stream that will be loaded into the header of the stream

    Properties:
        result (Tuple[DataFrame, DataFrame, DataFrame]): the constructed model, empty DataFrame, and damage_bin

    NOTE: This class uses the ModelLoaderMixin so it's data attributes are defined there
    """
    STREAM_HEADER: HeaderTypeDescriptor = HeaderTypeDescriptor()

    def __init__(self, data_path: str, footprint_index_dictionary: dict,
                 events: Optional[DataFrame] = None, file_type: FileTypeEnum = FileTypeEnum.CSV) -> None:
        """
        The constructor for the GetModelProcess class.

        Args:
            data_path: (str) the path to the data files needed to construct the model
            events (Optional[DataFrame]): events preloaded (if None, will load from a file)
            file_type (FileTypeEnum): the file type that the get model loads data from
        """
        self.fdal: FileDataAccessLayer = FileDataAccessLayer(extension=file_type, data_path=data_path)
        self.data_path: str = data_path
        self._vulnerabilities: Optional[FileLoader] = None
        self._footprint: Optional[FileLoader] = None
        self._damage_bin: Optional[FileLoader] = None
        self._events: Optional[FileLoader] = None
        self._items: Optional[FileLoader] = None
        self.model: Optional[DataFrame] = None
        self.events: Optional[Any] = events
        self.stream_type: int = 1
        self.footprint_index_dictionary: dict = footprint_index_dictionary
        self.event_id: int = self.events["event_id"].to_numpy()[0]

    def filter_footprint(self):
        start, stop = self.footprint_index_dictionary[self.event_id]
        filtered_footprint = self.fdal.footprint.value.to_numpy()[start: stop]

        if isinstance(self.fdal.items.value, DataFrame):
            self.fdal.items.value = self.fdal.items.value.to_numpy()

        highest_group_id = self.fdal.items.value[-1][4]
        number_of_vulnerability_ids = int(len(self.fdal.items.value) / highest_group_id)
        item_area_peril_ids = sorted([x[2] for x in self.fdal.items.value[0: number_of_vulnerability_ids]])

        item_position_map = {}
        areaperil_id_pointer = 0
        areaperil_id_end_pointer = len(item_area_peril_ids) - 1

        for i in range(0, len(filtered_footprint)):
            if filtered_footprint[i][1] in item_area_peril_ids:
                item_position_map[item_area_peril_ids[areaperil_id_pointer]] = i
                areaperil_id_pointer += 1

            if areaperil_id_pointer > areaperil_id_end_pointer:
                break

        buffer = np.array([[0, 0, 0, 0]])
        for i in sorted(list(item_position_map.keys())):
            row = filtered_footprint[item_position_map[i]: item_position_map[i] + 1]
            row = np.concatenate((np.array([self.event_id]), row[0]))
            buffer = np.concatenate((buffer, np.array([row])))

        self.model = buffer[1:]

    def merge_vulnerability_with_footprint(self):
        vulnerability_pointer = 0
        end_pointer = len(self.fdal.vulnerabilities.value) - 1

        buffer = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])

        for i in self.model:
            matched = False
            match_ended = False
            footprint_row = i
            position_map = {}
            cached_vulnerability_id: Optional[int] = None

            # generating a dict for matching ascending bin index with vulnerability probabilities
            while match_ended is False:
                vulnerability_row = self.fdal.vulnerabilities.value[vulnerability_pointer]

                if matched is True and footprint_row[2] == vulnerability_row[1]:
                    position_map[vulnerability_row[2]] = vulnerability_row[3]

                elif matched is False and footprint_row[2] == vulnerability_row[1]:
                    cached_vulnerability_id = vulnerability_row[0]
                    position_map[vulnerability_row[2]] = vulnerability_row[3]
                    matched = True

                elif matched is True and footprint_row[2] != vulnerability_row[1]:
                    position_map[vulnerability_row[2]] = vulnerability_row[3]
                    match_ended = True

                if vulnerability_pointer == end_pointer:
                    match_ended = True

                vulnerability_pointer += 1

            cached_bin_index = 1
            cached_probability = 0

            while cached_bin_index <= max(position_map.keys()):
                probability = position_map.get(cached_bin_index, 0.0)
                prob_to = probability * footprint_row[3]
                cached_probability += prob_to

                trimmed_vulnerability_row = np.array(
                    [cached_vulnerability_id, cached_bin_index, probability, cached_probability])
                merged_row = np.concatenate((footprint_row, trimmed_vulnerability_row))
                buffer = np.concatenate((buffer, np.array([merged_row])))
                cached_bin_index += 1

        # event_id	areaperil_id	intensity_bin_id	footprint_probability	vulnerability_id
        # damage_bin_id	vulnerability_probability	prob_to
        self.model = buffer[1:]

    def merge_damage_bin_dict(self):
        damage_map = {}

        for i in self.fdal.damage_bin.value.to_numpy():
            damage_map[i[0]] = i[1:]

        buffer = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])

        for i in self.model:
            interpol = np.array([damage_map[i[5]][2]])
            buffer = np.concatenate((buffer, [np.concatenate((i, interpol))]))

        self.model = buffer[1:]

    def deploy_final_model(self):
        df = DataFrame(self.model, columns=['event_id',
                                          'areaperil_id',
                                          'intensity_bin_id',
                                          'footprint_probability',
                                          'vulnerability_id',
                                          'damage_bin_id',
                                          'vulnerability_probability',
                                          'prob_to',
                                          'interpol'])
        self.model = df

    def print_stream(self) -> None:
        """
        Prints out the stream for cdftocsv.

        Returns: None
        """

        for i in self.model.groupby(["event_id", "areaperil_id", "vulnerability_id"]):

            df = i[1]
            header_row = df.iloc[0]

            sys.stdout.buffer.write(struct.pack("i", int(header_row.event_id)))
            sys.stdout.buffer.write(struct.pack("i", int(header_row.areaperil_id)))
            sys.stdout.buffer.write(struct.pack("i", int(header_row.vulnerability_id)))

            sys.stdout.buffer.write(struct.Struct('i').pack(int(len(df.index))))

            for _, row in df.iterrows():
                sys.stdout.buffer.write(struct.pack("f", float(row.prob_to)))
                sys.stdout.buffer.write(struct.pack("f", float(row.interpol)))  # this is the interpolation

    def run(self) -> None:
        """
        Runs all the functions to construct the model in sequence.

        Returns: None
        """
        if self.should_run is True:
            self.filter_footprint()
            self.merge_vulnerability_with_footprint()
            self.merge_damage_bin_dict()
            self.deploy_final_model()

    @property
    def should_run(self) -> bool:
        if self.footprint_index_dictionary.get(self.event_id) is None:
            return False
        return True

    @property
    def relevant_events(self):
        start, end = self.footprint_index_dictionary[self.event_id]
        return self.fdal.footprint.value.to_numpy()[start: end]
