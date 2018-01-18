from datetime import datetime


NONE_VALUES = [None, '', 'n/a', 'N/A', 'null', 'Null', 'NULL']


def get_timestamp(thedate=None):
    """ Get a timestamp """
    d = thedate if thedate else datetime.now()
    return d.strftime('%Y%m%d%H%M%S')


def get_utctimestamp(thedate=None, fmt='%Y-%b-%d %H:%M:%S'):
    """
    Returns a UTC timestamp for a given ``datetime.datetime`` in the
    specified string format - the default format is::

        YYYY-MMM-DD HH:MM:SS
    """
    d = thedate if thedate else datetime.now()
    return d.utcnow().strftime(fmt)


def to_string(val):
    """
    Converts value to string, with possible additional formatting.
    """
    return str(val) if None else ''


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
