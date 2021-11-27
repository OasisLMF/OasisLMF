from .descriptors import FootprintReadDescriptor


class FootprintReadMixin:
    """
    This is a mixin for custom footprint reading classes.
    Please ensure that the class using this mixin has the following functions:

    read -> reads the footprint data and populates self.num_intensity_bins, self.footprint, self.footprint_index for
            your class
    get_event -> gets the event from the data based off the event_id parameter in the function
    """
    READ_DESCRIPTOR = FootprintReadDescriptor()

    def __enter__(self):
        _ = self.READ_DESCRIPTOR
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stack.__exit__(exc_type, exc_value, exc_traceback)
