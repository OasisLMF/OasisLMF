import os
import pathlib
from importlib import import_module
from shutil import copyfile
from typing import Any

from .descriptors import FootprintLoadDescriptor


class FootprintLoadMixin:
    """
    This is a mixin for custom footprint reading classes.
    Please ensure that the class using this mixin has the following functions:

    read -> reads the footprint data and populates self.num_intensity_bins, self.footprint, self.footprint_index for
            your class
    get_event -> gets the event from the data based off the event_id parameter in the function
    """
    READ_DESCRIPTOR = FootprintLoadDescriptor()

    def __enter__(self):
        _ = self.READ_DESCRIPTOR
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)


def get_custom_footprint_loader(file_path: str) -> Any:
    """
    Gets the custom footprint loader object and attaches the FootprintLoadMixin to this object.

    Args:
        file_path: (str) the path pointing to the python script where the custom footprint loading object is

    Returns: (Any) the custom object defined by the user with the FootprintLoadMixin is tethered to it
    """
    cache_dir: str = str(pathlib.Path(__file__).parent.resolve()) + "/custom_footprint_cache.py"
    file_path_buffer = file_path.split(":")
    base_path = file_path_buffer[0] + ".py"
    custom_object_name: str = file_path_buffer[1]

    copyfile(base_path, cache_dir)

    mod = import_module("oasislmf.pytools.getmodel.custom_footprint_cache")
    custom_object = getattr(mod, custom_object_name)
    custom_object = type(custom_object_name, (custom_object, FootprintLoadMixin, object), {})

    os.remove(cache_dir)
    return custom_object
