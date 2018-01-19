from datetime import datetime

import pytz

NONE_VALUES = [None, '', 'n/a', 'N/A', 'null', 'Null', 'NULL']


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


def to_string(val):
    """
    Converts value to string, with possible additional formatting.
    """
    return '' if val is None else str(val)


def to_int(val):
    """
    Parse a string to int
    """
    return None if val in NONE_VALUES else int(val)


def to_float(val):
    """
    Parse a string to float
    """
    return None if val in NONE_VALUES else float(val)
