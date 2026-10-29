"""The building-packed loss stream's wire format.

The buildings of one item are multiplexed into its sample dimension, and the producers
(gulmc, gulpy) and consumers (fmpy, summarypy, plapy, and anything outside this repository)
agree on it through these values alone:

    sample   : sidx = (building - 1) * S + local_sidx                 local_sidx in [1, S]
    special  : sidx = local_sidx - (building - 1) * NUM_SPECIAL_SIDX  local_sidx in [-5, -1]

and a header that says the stream is packed. Which runs write a packed stream is tested end to
end, in tests/pytools/gulmc/test_building_packing.py.
"""
from oasislmf.pytools.common.event_stream import (ITEM_PACKED_STREAM, ITEM_STREAM, LOSS_STREAM_ID, MAX_LOSS_IDX,
                                                  MEAN_IDX, decode_building, decode_local_sidx, encode_sidx,
                                                  stream_info_to_bytes)

SPECIALS = range(-5, 0)


def test_the_encoding_of_a_packed_sidx():
    S = 10
    # building 2's samples occupy 11..20, building 3's from 21
    assert (encode_sidx(2, 1, S), encode_sidx(2, S, S), encode_sidx(3, 1, S)) == (11, 20, 21)
    # building 1's specials are -1..-5, building 2's -6..-10
    assert (encode_sidx(1, MEAN_IDX, S), encode_sidx(1, MAX_LOSS_IDX, S)) == (-1, -5)
    assert (encode_sidx(2, MEAN_IDX, S), encode_sidx(2, MAX_LOSS_IDX, S)) == (-6, -10)


def test_a_packed_sidx_decodes_to_its_building_and_local_sidx():
    """Building 1 is the identity, so an unpacked stream reads as building 1 throughout."""
    for sample_size in (1, 10, 1000):
        for building in range(1, 6):
            for local_sidx in [*SPECIALS, 1, sample_size]:
                sidx = encode_sidx(building, local_sidx, sample_size)
                assert (decode_building(sidx, sample_size), decode_local_sidx(sidx, sample_size)) == (building, local_sidx)
                if building == 1:
                    assert sidx == local_sidx


def test_the_delimiter_is_no_building():
    assert (decode_building(0, 10), decode_local_sidx(0, 10)) == (0, 0)


def test_the_header_of_a_packed_stream():
    """An unpacked header is byte for byte what it was before packing existed."""
    assert stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM).hex() == '01000002'
    assert stream_info_to_bytes(LOSS_STREAM_ID, ITEM_PACKED_STREAM).hex() == '03000002'
