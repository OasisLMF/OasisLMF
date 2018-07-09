from datetime import datetime

import sys

import pytz
import six

NULL_VALUES = [None, '', 'n/a', 'N/A', 'null', 'Null', 'NULL']


def get_timestamp(thedate=None, fmt='%Y%m%d%H%M%S'):
    """ Get a timestamp """
    d = thedate if thedate else datetime.now()
    return d.strftime(fmt)


def get_utctimestamp(thedate=None, fmt='%Y-%b-%d %H:%M:%S'):
    """
    Returns a UTC timestamp for a given ``datetime.datetime`` in the
    specified string format - the default format is::

        YYYY-MMM-DD HH:MM:SS
    """
    d = thedate.astimezone(pytz.utc) if thedate else datetime.utcnow()
    return d.strftime(fmt)


def is_string(s):
        return type(s) in (
            six.types.StringTypes if sys.version_info.major < 3
            else six.string_types
        )


def to_string(val):
    """
    Converts value to string, with possible additional formatting.
    """
    return '' if val is None else str(val)


def to_int(val):
    """
    Parse a string to int
    """
    return None if val in NULL_VALUES else int(val)


def to_float(val):
    """
    Parse a string to float
    """
    return None if val in NULL_VALUES else float(val)
