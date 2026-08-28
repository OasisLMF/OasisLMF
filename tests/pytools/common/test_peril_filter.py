"""Pin the peril filter shared by the gulpy and gulmc structure builders.

``filter_area_peril_id`` narrows a run to the area perils covered by the requested perils. It is
reached only through ``build_structures``, which needs a whole run directory, so neither of its
two call sites is exercised by the gulpy or gulmc suites -- they never pass a ``peril_filter``.
"""
import numpy as np
import pytest

from oasislmf.pytools.common.data import areaperil_int
from oasislmf.pytools.common.input_files import KEYS_DTYPE, filter_area_peril_id

KEYS_ROWS = [
    # (LocID, PerilID, CoverageTypeID, AreaPerilID, VulnerabilityID)
    (1, 'WTC', 1, 10, 1),
    (1, 'WSS', 1, 11, 1),
    (2, 'WTC', 1, 10, 2),      # the same area peril as row 0, so the result must de-duplicate
    (2, 'ORF', 1, 12, 2),
    (3, 'WSS', 3, 13, 3),
    (3, 'WTC', 3, 14, 3),
]


def make_keys(rows=KEYS_ROWS):
    return np.array(rows, dtype=KEYS_DTYPE)


def reference_filter_area_peril_id(keys_tb, peril_filter):
    """The python set membership loop this replaced, kept as the reference implementation."""
    peril_set = set(peril_filter)
    mask = np.array([p in peril_set for p in keys_tb['PerilID']])
    return np.unique(keys_tb['AreaPerilID'][mask])


@pytest.mark.parametrize('peril_filter', [
    ['WSS'],
    ['WSS', 'WTC'],
    ['WTC', 'WSS', 'ORF'],
    ['ORF'],
    {'WSS'},                 # a set, which is what the call sites used to build internally
    ('WSS', 'ORF'),          # a tuple
    ['wss'],                 # case mismatch matches nothing; peril ids are case sensitive
    ['WSSX'],                # longer than the U3 field
    ['WS'],                  # a prefix is not a match
    ['QQ1'],                 # simply absent from the table
])
def test_matches_the_reference_implementation(peril_filter):
    keys_tb = make_keys()

    np.testing.assert_array_equal(
        filter_area_peril_id(keys_tb, peril_filter),
        reference_filter_area_peril_id(keys_tb, peril_filter),
    )


def test_selects_the_distinct_area_perils_of_the_matching_rows():
    keys_tb = make_keys()

    np.testing.assert_array_equal(filter_area_peril_id(keys_tb, ['WTC']), [10, 14])
    np.testing.assert_array_equal(filter_area_peril_id(keys_tb, ['WSS']), [11, 13])
    np.testing.assert_array_equal(filter_area_peril_id(keys_tb, ['WTC', 'WSS']), [10, 11, 13, 14])
    assert filter_area_peril_id(keys_tb, ['QQ1']).size == 0


def test_area_peril_id_dtype_is_preserved():
    """The result indexes items['areaperil_id'], so it has to stay an integer array."""
    result = filter_area_peril_id(make_keys(), ['WTC'])

    assert result.dtype == np.dtype(areaperil_int)


def test_empty_keys_table_gives_an_empty_integer_array():
    """An empty keys table is not an error, and the result is still usable as an index.

    The list comprehension this replaced built its mask with ``np.array([...])``, which on an
    empty table produced a float64 array rather than a bool one -- so indexing with it raised
    ``IndexError: arrays used as indices must be of integer (or boolean) type``.
    """
    empty = make_keys([])

    result = filter_area_peril_id(empty, ['WTC'])

    assert result.size == 0
    assert result.dtype == np.dtype(areaperil_int)
    # the same call against the implementation this replaced
    with pytest.raises(IndexError):
        reference_filter_area_peril_id(empty, ['WTC'])


def test_a_bare_string_filter_matches_nothing():
    """A string is iterated character by character, so it cannot match a three character peril id.

    Pre-existing and unchanged: ``set('WSS')`` and ``list('WSS')`` are both ``['W', 'S', 'S']``.
    ``oasislmf/execution/bash.py`` always passes a list, so this is not reachable from the CLI,
    but it fails silently rather than loudly and is worth stating.
    """
    keys_tb = make_keys()

    assert filter_area_peril_id(keys_tb, 'WSS').size == 0
    np.testing.assert_array_equal(
        filter_area_peril_id(keys_tb, 'WSS'), reference_filter_area_peril_id(keys_tb, 'WSS'))
