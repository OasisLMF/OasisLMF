import difflib

import io

from .exceptions import OasisException


def unified_diff(file1, file2, as_string=False):
    """
    Generates a unified diff of two files: ``file1`` and ``file2``. The files must
    be passed in as absolute paths.
    """
    try:
        with io.open(file1, 'r') as f1:
            with io.open(file2, 'r') as f2:
                diff = difflib.unified_diff(
                    f1.readlines(),
                    f2.readlines(),
                    fromfile=f1.name,
                    tofile=f2.name,
                )
    except (OSError, IOError) as e:
        raise OasisException(str(e))

    if as_string:
        return ''.join(diff)
    return diff
