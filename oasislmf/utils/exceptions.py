__all__ = [
    'OasisException'
]


class OasisException(Exception):
    """
    Oasis base exception class

    Example
    -------
    In [call]: raise OasisException('Error Message 1', OSError('Root of error'))

    OasisException: Error Message 1, from OSError: Root of error
    """
    def __init__(self, msg, original_exception=None):
        self.original_exception = original_exception
        if original_exception:
            # This is wrapped execption
            super(OasisException, self).__init__(msg + (", {}: {}".format(
                original_exception.__class__.__name__,
                original_exception)
            ))
        else:
            # Message only exception
            super(OasisException, self).__init__(msg)
