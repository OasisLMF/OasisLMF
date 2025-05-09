from oasis_data_manager.errors import OasisException

__all__ = [
    'OasisException'
]


class OasisStreamException(OasisException):
    def __init__(self, msg, original_exception=None):
        super().__init__(msg, original_exception)
