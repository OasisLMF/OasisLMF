from oasis_data_manager.errors import OasisException

__all__ = [
    'OasisException',
    'OasisExceptionNoKeys',
]


class OasisStreamException(OasisException):
    def __init__(self, msg, original_exception=None):
        super().__init__(msg, original_exception)


class OasisExceptionNoKeys(OasisException):
    def __init__(self, msg, original_exception=None):
        super().__init__(msg, original_exception)
