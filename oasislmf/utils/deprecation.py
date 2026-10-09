"""Deprecation notices that actually reach the user."""
import warnings


def warn_deprecated(message, stacklevel=2):
    """Raise a DeprecationWarning that Python's default filters cannot hide.

    Python's default filter set shows a ``DeprecationWarning`` only when the warning is
    *attributed to* ``__main__``. Attribution follows ``stacklevel``, not where ``warn()`` is
    written, so a notice raised anywhere below the entry point -- the CLI, the computation
    layer, a lookup -- is silently dropped. ``simplefilter`` inside a ``catch_warnings`` block
    overrides that for this one call without disturbing the process-wide filters.

    Args:
        message (str): the notice.
        stacklevel (int): frames to skip, counted as if ``warnings.warn`` were called directly
            by the caller. The default of 2 blames the caller's caller, which is normally the
            code still using the deprecated thing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always", DeprecationWarning)
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)
