from typing import Optional, Tuple

import numpy as np
from pandas import DataFrame, merge

from .loader_mixin import ModelLoaderMixin, FileLoader


class GetModelProcess(ModelLoaderMixin):
    """
    This class is responsible for loading data from a range of sources and merging them to build a model.

    Attributes:
        data_path (str): the path to the data files needed to construct the model
        model (Optional[DataFrame]): the model for K-tools

    Properties:
        result (Tuple[DataFrame, DataFrame, DataFrame]): the constructed model, empty DataFrame, and damage_bin

    NOTE: This class uses the ModelLoaderMixin so it's data attributes are defined there
    """
    def __init__(self, data_path: str) -> None:
        """
        The constructor for the GetModelProcess class.

        Args:
            data_path: (str) the path to the data files needed to construct the model
        """
        self.data_path: str = data_path
        self._vulnerabilities: Optional[FileLoader] = None
        self._footprint: Optional[FileLoader] = None
        self._damage_bin: Optional[FileLoader] = None
        self._events: Optional[FileLoader] = None
        self.model: Optional[DataFrame] = None

    def merge_complex_items(self) -> None:
        pass

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
        self.merge_damage_bin_dict()
        self.calculate_probability_of_damage()
        self.define_columns_for_saving()

    @property
    def result(self) -> Tuple[DataFrame, DataFrame, DataFrame]:
        return self.model, DataFrame(), self.damage_bin.value
