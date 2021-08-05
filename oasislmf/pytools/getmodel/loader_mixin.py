import os
from typing import Dict

from pandas import read_csv, DataFrame

from .file_loader import FileLoader


class ModelLoaderMixin:
    """
    This Mixin class is responsible for loading data for the get model.
    """
    FILE_MAP: Dict[str, str] = {
        "vulnerabilities": "vulnerability.csv",
        "footprint": "footprint.csv",
        "damage_bin": "damage_bin_dict.csv",
        "events": "events.csv",
        "items": "items.csv"
    }

    @staticmethod
    def check_bin_file_exists_and_read_it(file_desc, input_fp, conversion_tool):
        """
        Check model data or complex items binary file exists, convert to csv using
        ktools executables and return file contents as dataframe
        :param file_desc: brief description of file to be opened
        :dtype file_desc: str
        :param input_fp: file path to binary file
        :dtype input_fp: str
        :param conversion_tool: ktools binary to csv conversion tool executable
        :dtype: str
        :return: data from binary file
        :dtype: pandas.DataFrame
        """

        if not os.path.exists(input_fp):
            raise Exception(
                f'{file_desc} file {os.path.abspath(input_fp)} does not exist.'
            )
        with os.popen(f'{conversion_tool} < {input_fp}') as p:
            input_df = read_csv(p)
        input_df.columns = input_df.columns.str.replace(' ', '')
        input_df.columns = input_df.columns.str.replace('"', '')

        return input_df

    def load_data_if_none(self, name: str) -> None:
        """
        Loads the data from the file in the directory of the self.data_path setting the data to the self._{name}
        attribute.

        Args:
            name: (str) this is doing to be used to get the name of the file using the self.FILE_MAP

        Returns: None
        """
        if getattr(self, f"_{name}") is None:
            file_handler: FileLoader = FileLoader(file_path=self.data_path + f"/{self.FILE_MAP[name]}",
                                                  label=name)
            setattr(self, f"_{name}", file_handler)

    @property
    def items(self) -> FileLoader:
        self.load_data_if_none(name="items")
        return self._items

    @property
    def vulnerabilities(self) -> FileLoader:
        self.load_data_if_none(name="vulnerabilities")
        return self._vulnerabilities

    @property
    def footprint(self) -> FileLoader:
        self.load_data_if_none(name="footprint")
        return self._footprint

    @property
    def damage_bin(self) -> FileLoader:
        self.load_data_if_none(name="damage_bin")
        return self._damage_bin

    @property
    def events(self) -> FileLoader:
        self.load_data_if_none(name="events")
        return self._events

    @items.setter
    def items(self, value) -> None:
        self._items = value

    @vulnerabilities.setter
    def vulnerabilities(self, value) -> None:
        self._vulnerabilities = value

    @footprint.setter
    def footprint(self, value) -> None:
        self._footprint = value

    @damage_bin.setter
    def damage_bin(self, value) -> None:
        self._damage_bin = value

    @events.setter
    def events(self, value) -> None:
        if value is not None:
            placeholder = FileLoader(file_path=self.data_path + f"/{self.FILE_MAP['events']}",
                                     label="events")
            placeholder.value = value
            value = placeholder
        self._events = value
