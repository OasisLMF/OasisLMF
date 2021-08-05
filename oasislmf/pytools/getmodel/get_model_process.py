import struct
import sys
from typing import Optional, Tuple, List, Any

import numpy as np
from pandas import DataFrame, merge

from .descriptors import HeaderTypeDescriptor
from .loader_mixin import ModelLoaderMixin, FileLoader


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

    def __init__(self, data_path: str, events: Optional[DataFrame] = None) -> None:
        """
        The constructor for the GetModelProcess class.

        Args:
            data_path: (str) the path to the data files needed to construct the model
            events (Optional[DataFrame]): events preloaded (if None, will load from a file)
        """
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


        Returns:
        """
        filter_buffer = [str(i["areaperil_id"]) + str(i["vulnerability_id"]) for i in list(self.items.value.T.to_dict().values())]
        self.model["filter_code"] = self.model["area_peril_id"].astype(str) + self.model["vulnerability_id"].astype(str)
        self.model = self.model[self.model["filter_code"].isin(filter_buffer)]
        del self.model['filter_code']

    def merge_vulnerabilities(self) -> None:
        """
        Merges the self.vulnerabilities data into the self.model and calculates the "cum_prob".

        Returns: None
        """
        # Calculate cummulative probability
        self.vulnerabilities.value['cum_prob'] = self.vulnerabilities.value.groupby(
            ['vulnerability_id', 'intensity_bin_id']
        ).cumsum()['probability']

        # we need to add vulnerability_id to the self.model (THIS IS WHERE THE PROBLEM IS, IT'S SUPPOSED TO BE:
        # on=['vulnerability_id', 'intensity_bin_id'])

        self.model = merge(self.model, self.vulnerabilities.value, how='inner',
                           on=['intensity_bin_id', 'intensity_bin_id'])
        # Drop unrequired columns and dataframe to free memory
        self.model.rename(columns={"probability": "vulnerability_probability"}, inplace=True)
        self.model.drop(['intensity_bin_id'], axis=1, inplace=True)
        self.vulnerabilities.clear_cache()

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
        self.model.drop('bin_index', axis=1, inplace=True)

        # Restore initial order and drop unrequired order column
        self.model.sort_values(
            by=['order', 'area_peril_id', 'vulnerability_id', 'damage_bin_id'],
            ascending=True, inplace=True
        )

        self.model.drop('order', axis=1, inplace=True)

    def merge_model_with_footprint(self) -> None:
        """
        Merges the self.events into the self.model.

        Returns: None
        """
        event_ids: np.ndarray = self.events.value["event_id"].to_numpy()  # add in chunking and stream load
        self.model = DataFrame({"event_id": event_ids})
        self.model["order"] = self.model.index
        # self.footprint.value.drop('probability', axis=1, inplace=True)

        self.footprint.value.rename(columns={'areaperil_id': 'area_peril_id', "probability": "footprint_probability"},
                                    inplace=True)
        # self.footprint.value.drop('footprint_probability', axis=1, inplace=True)
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

    @property
    def result(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        return self.model, DataFrame(), self.damage_bin.value

    @property
    def stream(self) -> List[bytes]:
        """
        Loops through the self.model converting it into binary data.

        Returns: (List[bytes]) self.model in binary form
        """
        buffer = [self.STREAM_HEADER]
        number_of_rows: int = len(self.model.index)

        for i in list(self.model.T.to_dict().values()):
            s = struct.Struct('IIIIf')
            values = (
                int(i["event_id"]),
                int(i["areaperil_id"]),
                int(i["vulnerability_id"]),
                int(number_of_rows),
                float(i["prob_to"])
            )
            buffer.append(s.pack(*values))
        return buffer

    def print_stream(self) -> None:
        """
        Prints out the stream for cdftocsv.

        Returns: Nonehttps://www.nhs.uk/conditions/coronavirus-covid-19/
        """
        self.model.sort_values(by=['vulnerability_id'])
        number_of_rows: int = len(self.model.index)
        sys.stdout.buffer.write(self.STREAM_HEADER)

        ordered_data = list(self.model.T.to_dict().values())

        for i in range(0, number_of_rows):
            sys.stdout.buffer.write(struct.Struct('i').pack(int(ordered_data[i]["event_id"])))
            sys.stdout.buffer.write(struct.Struct('i').pack(int(ordered_data[i]["areaperil_id"])))
            sys.stdout.buffer.write(struct.Struct('i').pack(int(ordered_data[i]["vulnerability_id"])))
            # logger.info(f"here is the number of rows: {number_of_rows}")
            # sys.stdout.buffer.write(struct.Struct('I').pack(int(number_of_rows)))

            buffer = []
            for x in range(i, number_of_rows):
                if ordered_data[x]["vulnerability_id"] == ordered_data[i]["vulnerability_id"]:
                    buffer.append(struct.Struct('f').pack(float(ordered_data[x]["prob_to"])))
                    buffer.append(struct.Struct('f').pack(float(ordered_data[x]["bin_mean"])))
                    # sys.stdout.buffer.write(struct.Struct('f').pack(float(ordered_data[x]["prob_to"])))
                    # sys.stdout.buffer.write(struct.Struct('f').pack(float(ordered_data[x]["bin_mean"])))
                else:
                    break

            sys.stdout.buffer.write(struct.Struct('i').pack(int(len(buffer) / 2)))

            for y in buffer:
                sys.stdout.buffer.write(y)
