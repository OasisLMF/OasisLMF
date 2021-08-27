from .descriptors import FileMapDescriptor
from .enums import FileTypeEnum
from .file_loader import FileLoader


class ModelLoaderMixin:
    """
    This Mixin class is responsible for loading data for the get model.
    """
    FILE_MAP = FileMapDescriptor()

    def __init__(self, extension: FileTypeEnum = FileTypeEnum.CSV) -> None:
        """
        The constructor for the ModelLoaderMixin class.

        Args:
            extension: (FileTypeEnum) type of extension to be used
        """
        self.extension: FileTypeEnum = extension

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
