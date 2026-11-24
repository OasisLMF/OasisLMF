"""Deprecation notices that actually reach the user."""
import warnings


def warn_deprecated(message, stacklevel=2):
    """Warn that something the user called is deprecated.

    The category is ``FutureWarning``, not ``DeprecationWarning``. Python's default filters show
    a ``DeprecationWarning`` only when it is attributed to ``__main__``, so a notice raised below
    the entry point -- the CLI, the computation layer, a lookup -- would be silently dropped.
    ``FutureWarning`` is the category the Python docs give to deprecations aimed at end users: it
    is shown by default, and ``-W error`` or ``-W ignore`` still apply to it.

    Args:
        message (str): the notice.
        stacklevel (int): frames to skip, counted as if ``warnings.warn`` were called directly
            by the caller. The default of 2 blames the caller's caller, which is normally the
            code still using the deprecated thing.
    """
    warnings.warn(message, FutureWarning, stacklevel=stacklevel + 1)
