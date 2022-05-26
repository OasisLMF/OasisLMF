"""
This file defines the ModelSettings class which manages the loading and mapping of model settings data.
"""
import json
import os
from typing import Optional


class ModelSettings:
    """
    This class is responsible for loading and mapping the data around model settings.
    """
    def __init__(self) -> None:
        """
        The constructor for the ModelSettings class.
        """
        self._data: Optional[dict] = None
        self._correlation_map: Optional[dict] = None

    def _load_data(self) -> dict:
        """
        Loads the data from the stashed file directory.
        Returns: (dict) model settings data
        """
        with open(self.file_directory) as file:
            return json.loads(file.read())

    def get_correlation_value(self, peril_correlation_group: int) -> float:
        """
        Gets the correlation value for a peril correlation group.
        Args:
            peril_correlation_group: (int) the ID of the peril correlation group that we are going to extract the
                                           correlation value for
        Returns: (float) the correlation value
        """
        return self.correlation_map[peril_correlation_group].get("correlation_value", 0.0)

    @staticmethod
    def define_path(file_path: str) -> None:
        """
        Stashes the path to the model settings data to run the model.
        Args:
            file_path: (str) the path to the model settings JSON file
        Returns: None
        """
        with open(ModelSettings().file_directory_stash, "w") as file:
            file.write(file_path)

    @property
    def file_directory(self) -> str:
        if not os.path.exists(path=self.file_directory_stash):
            raise FileNotFoundError(
                "stash file not found please implement the defile_path function before trying to read "
                "model settings data"
            )
        with open(self.file_directory_stash) as file:
            return file.read()

    @property
    def file_directory_stash(self) -> str:
        return str(os.path.dirname(os.path.realpath(__file__))) + "/file_stash.txt"

    @property
    def data(self) -> dict:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    @property
    def correlation_map(self) -> Optional[dict]:
        if self._correlation_map is None:
            data = self.data
            package = dict()

            for supported_peril in data.get("lookup_settings", {"supported_perils": []}).get("supported_perils", []):
                peril_correlation_group: Optional[int] = supported_peril.get("peril_correlation_group")
                if peril_correlation_group is not None:
                    package[peril_correlation_group] = supported_peril
                else:
                    return None

            for correlation_setting in data.get("correlation_ settings", []):
                package[correlation_setting["peril_correlation_group"]]["correlation_value"] = correlation_setting["correlation_value"]

            self._correlation_map = package
        return self._correlation_map
