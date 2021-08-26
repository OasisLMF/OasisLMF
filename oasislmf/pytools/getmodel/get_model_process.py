import struct
import sys
from typing import Optional, Tuple, Any

import numpy as np
from numba import jit, int64
from pandas import DataFrame, merge, concat

from oasislmf.utils.data import merge_dataframes
from .descriptors import HeaderTypeDescriptor
from .enums import FileTypeEnum
from .loader_mixin import ModelLoaderMixin, FileLoader


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


class GetModelProcess(ModelLoaderMixin):
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

    def __init__(self, data_path: str,
                 events: Optional[DataFrame] = None, file_type: FileTypeEnum = FileTypeEnum.CSV) -> None:
        """
        The constructor for the GetModelProcess class.

        Args:
            data_path: (str) the path to the data files needed to construct the model
            events (Optional[DataFrame]): events preloaded (if None, will load from a file)
            file_type (FileTypeEnum): the file type that the get model loads data from
        """
        super().__init__(extension=file_type)
        self.data_path: str = data_path
        self._vulnerabilities: Optional[FileLoader] = None
        self._footprint: Optional[FileLoader] = None
        self._damage_bin: Optional[FileLoader] = None
        self._events: Optional[FileLoader] = None
        self._items: Optional[FileLoader] = None
        self.model: Optional[DataFrame] = None
        self.events: Optional[Any] = events
        self.stream_type: int = 1

    def merge_complex_items(self) -> None:
        pass

    def filter_footprint(self) -> None:
        """
        Filters out rows from the self.model that do not have the combination of "areaperil_id" and "vulnerability_id"
        from the self.items.

        Returns: None
        """
        coefficient = np.max(self.items.value["vulnerability_id"])
        self.items.value["filter_code"] = self.items.value["vulnerability_id"] + self.items.value["areaperil_id"] * (
                coefficient + 1
        )
        self.items.value.drop_duplicates(subset='filter_code', inplace=True)

        self.model["filter_code"] = self.model["vulnerability_id"] + self.model["area_peril_id"] * (
                coefficient + 1
        )
        self.model = self.model[self.model["filter_code"].isin(self.items.value["filter_code"].to_list())]
        del self.model['filter_code']

    def merge_vulnerabilities(self) -> None:
        """
        Merges the self.vulnerabilities data into the self.model and calculates the "cum_prob".

        Returns: None
        """
        # find that MAX damage_bin_id for each row in the vulnerability file
        # cut the vulnerabilities that are not represented in the items
        vun_filter = list(self.items.value.drop_duplicates(subset=['vulnerability_id'])["vulnerability_id"])
        self.vulnerabilities.value = self.vulnerabilities.value[
            self.vulnerabilities.value["vulnerability_id"].isin(vun_filter)]

        # find that MAX damage_bin_id for each row in the vulnerability file
        vun_max = self.vulnerabilities.value.groupby(
            ['vulnerability_id', 'intensity_bin_id']
        )['damage_bin_id'].max().reset_index().rename(columns={"damage_bin_id": "damage_bin_max"})

        vun_list = define_data_series(vul_ids=vun_max["vulnerability_id"].values,
                                      inten_ids=vun_max["intensity_bin_id"].values,
                                      damage_bin_maxs=vun_max["damage_bin_max"].values)

        vun_fill_empty = DataFrame(vun_list,
                                   columns=["vulnerability_id", "intensity_bin_id", "damage_bin_id", "probability"])
        vun_fill_empty.reset_index(drop=True, inplace=True)

        # filter model by area peril ID
        area_filter = list(self.items.value.drop_duplicates(subset=['areaperil_id'])["areaperil_id"])
        self.model = self.model[self.model["area_peril_id"].isin(area_filter)]

        # merge the two DataFrames so 'damage_bin_id' is sequential, [1 ... row.damage_bin_max]
        # where the 'probability' has a valid value use that otherwise fill the missing 'damage_bin_id' entries with 0.0
        vulnerabilities_no_gap = merge_dataframes(
            self.vulnerabilities.value,
            vun_fill_empty,
            ['vulnerability_id', 'intensity_bin_id', 'damage_bin_id'],
            how='right').fillna(0.0)

        # merge the
        self.model.set_index("intensity_bin_id", inplace=True)
        vulnerabilities_no_gap.set_index("intensity_bin_id", inplace=True)
        self.model = self.model.join(vulnerabilities_no_gap, how="left")

        self.model.rename(columns={"probability": "vulnerability_probability"}, inplace=True)
        # self.vulnerabilities.clear_cache()

    def merge_damage_bin_dict(self) -> None:
        """
        Merges the self.damage_bin into the self.model.

        Returns: None
        """
        self.model = merge(
            self.model, self.damage_bin.value[['bin_index', 'interpolation']],
            how='inner', left_on='damage_bin_id', right_on='bin_index'
        )
        # Drop unrequired column to free memory
        del self.model["bin_index"]
        # self.model.drop('bin_index', axis=1, inplace=True)

        # Restore initial order and drop unrequired order column
        self.model.sort_values(
            by=['order', 'area_peril_id', 'vulnerability_id', 'damage_bin_id'],
            ascending=True, inplace=True
        )

        del self.model["order"]
        self.damage_bin.clear_cache()

    def merge_model_with_footprint(self) -> None:
        """
        Merges the self.events into the self.model.

        Returns: None
        """
        event_ids: np.ndarray = self.events.value["event_id"].to_numpy()  # add in chunking and stream load
        self.model = DataFrame({"event_id": event_ids})
        self.model["order"] = self.model.index

        self.footprint.value.rename(columns={'areaperil_id': 'area_peril_id', "probability": "footprint_probability"},
                                    inplace=True)

        self.model = merge(self.model, self.footprint.value, how='inner', on='event_id')

        self.footprint.clear_cache()

    def calculate_probability_of_damage(self) -> None:
        """
        Calculates the "prob_to" columns for the model by multiplying the probability of the event happening by the
        probability of damage occuring.

        Returns: None
        """
        self.model["prob_to"] = self.model["footprint_probability"] * self.model["vulnerability_probability"]

    def define_columns_for_saving(self) -> None:
        """
        Trims the self.model DataFrame removing columns that are not needed and rename columns required for later
        sections in k-tools.

        Returns: None
        """
        self.model = self.model[["event_id", "area_peril_id", "vulnerability_id", "damage_bin_id", "prob_to",
                                 "interpolation"]]

        self.model.rename(columns={
            "event_id": "event_id",
            "area_peril_id": "areaperil_id",
            "vulnerability_id": "vulnerability_id",
            "damage_bin_id": "bin_index",
            "prob_to": "prob_to",
            "interpolation": "bin_mean"
        }, inplace=True)

    def calculate_cum_sum(self) -> None:
        """
        Calculates the cumulative probability based on the event_id, areaperil_id, and vulnerability_id and assigns
        it to teh "prob_to" column.

        Returns: None
        """
        self.model.sort_values(by=['vulnerability_id'])

        buffer = []
        for i in self.model.groupby(["event_id", "areaperil_id", "vulnerability_id"]):
            df = i[1]
            df["prob_to"] = df["prob_to"].cumsum()
            buffer.append(df)

        self.model = concat(buffer)

    def print_stream(self) -> None:
        """
        Prints out the stream for cdftocsv.

        Returns: None
        """
        self.model.sort_values(by=['vulnerability_id'])
        sys.stdout.buffer.write(self.STREAM_HEADER)

        for i in self.model.groupby(["event_id", "areaperil_id", "vulnerability_id"]):

            df = i[1]
            header_row = df.iloc[0]

            sys.stdout.buffer.write(struct.pack("i", int(header_row.event_id)))
            sys.stdout.buffer.write(struct.pack("i", int(header_row.areaperil_id)))
            sys.stdout.buffer.write(struct.pack("i", int(header_row.vulnerability_id)))

            sys.stdout.buffer.write(struct.Struct('i').pack(int(len(df.index))))

            for _, row in df.iterrows():
                sys.stdout.buffer.write(struct.pack("f", float(row.prob_to)))
                sys.stdout.buffer.write(struct.pack("f", float(row.bin_mean)))

    def run(self) -> None:
        """
        Runs all the functions to construct the model in sequence.

        Returns: None
        """
        self.merge_model_with_footprint()
        self.merge_complex_items()
        self.merge_vulnerabilities()
        self.filter_footprint()
        self.merge_damage_bin_dict()
        self.calculate_probability_of_damage()
        self.define_columns_for_saving()
        self.calculate_cum_sum()

    @property
    def result(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        return self.model, DataFrame(), self.damage_bin.value
