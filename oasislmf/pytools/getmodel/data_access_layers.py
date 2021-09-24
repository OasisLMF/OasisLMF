from .loader_mixin import ModelFileLoaderMixin
from .enums import FileTypeEnum
from typing import Dict, Optional
import numba as nb
from .file_loader import FileLoader


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@nb.jit(cache=True, nopython=True)
def make_footprint_index_dict(footprint_index):
    footprint_map = nb.typed.Dict()
    cached_event_id = footprint_index[0]
    cached_offset = 0

    for i in range(0, len(footprint_index)):
        if footprint_index[i] != cached_event_id:
            footprint_map[cached_event_id] = (cached_offset, i - 1)
            cached_offset = i
            cached_event_id = footprint_index[i]
    return footprint_map


class FileDataAccessLayer(ModelFileLoaderMixin, metaclass=Singleton):

    def __init__(self, data_path, extension: FileTypeEnum = FileTypeEnum.CSV) -> None:
        super().__init__(extension=extension)
        self.data_path: str = data_path
        self._vulnerabilities: Optional[FileLoader] = None
        self._footprint: Optional[FileLoader] = None
        self._damage_bin: Optional[FileLoader] = None
        self._events: Optional[FileLoader] = None
        self._items: Optional[FileLoader] = None
        self.data_path = data_path
        self.extension: FileTypeEnum = extension
        self.footprint_dict = make_footprint_index_dict(
            footprint_index=self.footprint.value["event_id"].to_numpy())
