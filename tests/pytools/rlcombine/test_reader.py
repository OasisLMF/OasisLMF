"""Tests for SummaryCombineReader.read_events, which reads the summary streams of several inuring
priorities and yields each input's events as soon as that input has closed them with E:S0."""
from contextlib import ExitStack

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import FM_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY, stream_info_to_bytes
from oasislmf.pytools.rlcombine.manager import init_summary_streams_in
from oasislmf.utils.exceptions import OasisException, OasisStreamException

from .common import (IP1, IP2, TERMINATOR, make_summary_stream, read_all, read_events, record_header_dtype,
                     sidx_loss_dtype, write_stream_files)


def test_read_events_event_buffer_layout(tmp_path):
    """The event buffer holds the event's records in CSR form and no sentinel record."""
    event_id, records = IP1[0]
    files = write_stream_files(tmp_path, [make_summary_stream([(event_id, records)])])
    (file_idx, read_event_id, event_buffer), = read_events(files)

    pairs = [pair for _, _, record_pairs in records for pair in record_pairs]
    assert (file_idx, read_event_id) == (0, event_id)
    assert set(event_buffer) == {'summary_ids', 'expval', 'offsets', 'sidx', 'loss'}
    np.testing.assert_array_equal(event_buffer['summary_ids'], [summary_id for summary_id, _, _ in records])
    np.testing.assert_array_equal(event_buffer['expval'], [expval for _, expval, _ in records])
    np.testing.assert_array_equal(event_buffer['offsets'], np.cumsum([0] + [len(record_pairs) for _, _, record_pairs in records]))
    np.testing.assert_array_equal(event_buffer['sidx'], [sidx for sidx, _ in pairs])
    np.testing.assert_array_equal(event_buffer['loss'], [loss for _, loss in pairs])


def test_read_events_sentinel_only_event(tmp_path):
    """An event made only of its E:S0 sentinel is yielded with an empty buffer."""
    files = write_stream_files(tmp_path, [make_summary_stream([(7, [])])])
    (file_idx, event_id, event_buffer), = read_events(files)

    assert (file_idx, event_id) == (0, 7)
    assert event_buffer['summary_ids'].shape == (0,)
    np.testing.assert_array_equal(event_buffer['offsets'], [0])
    assert event_buffer['sidx'].shape == event_buffer['loss'].shape == (0,)


def test_read_events_multiple_inputs(tmp_path):
    """Each input's events are yielded in that input's order and tagged with its file_idx."""
    files = write_stream_files(tmp_path, [make_summary_stream(IP1), make_summary_stream(IP2)])
    assert read_all(files) == {0: IP1, 1: IP2}


def test_read_events_larger_than_pipe_capacity(tmp_path):
    """An event larger than PIPE_CAPACITY, holding a single record larger than PIPE_CAPACITY, is read in
    full across several 64 KiB reads, and the event after it is read too."""
    len_sample = 10000
    big_record = (4, 1.0, [(-5, 2.0), (-1, 1.0)] + [(sidx, 1.0) for sidx in range(1, len_sample + 1)])
    events = [(5, [(1, 3.0, [(-5, 1.0), (1, 1.0)]), big_record, (6, 2.0, [(-1, 2.0)])]),
              (6, [(1, 2.0, [(-5, 1.0)])])]
    assert len(big_record[2]) * sidx_loss_dtype.itemsize > PIPE_CAPACITY

    files = write_stream_files(tmp_path, [make_summary_stream(events, len_sample=len_sample)])
    assert read_all(files) == {0: events}


@pytest.mark.parametrize('events, sentinel, match', [
    pytest.param([(1, [(1, 1, [(1, 1.0)])]), (2, [(1, 1, [(1, 1.0)])])], False, 'event not closed by an S0',
                 id='no S0 before the next event'),
    pytest.param([(1, [(11, 1, [])])], True, 'summary_id 11 outside', id='summary_id above max_summary_id'),
    pytest.param([(1, [(1, 1, [(1, 1.0)] * 6)])], True, 'more .sidx, loss. pairs in a record',
                 id='more pairs than samples and special sidx'),
    pytest.param([(1, [(1, 1, [(-4, 1.0)])])], True, 'unexpected sidx -4', id='sidx -4 not written for rl'),
    pytest.param([(1, [(1, 1, [(-2, 1.0)])])], True, 'unexpected sidx -2', id='sidx -2 never written'),
    pytest.param([(1, [(1, 1, [(4, 1.0)])])], True, 'sample sidx 4 above len_sample', id='sample sidx above len_sample'),
    pytest.param([(1, [(1, 1, [(-5, 1.0)])] * 11)], True, 'more records in an event than max_summary_id',
                 id='more records than max_summary_id'),
])
def test_read_events_invalid_records(tmp_path, events, sentinel, match):
    """Records that break the rl summary stream format are reported rather than read
    (len_sample is 3 and max_summary_id 10)."""
    blob = make_summary_stream(events, len_sample=3, sentinel=sentinel)
    with pytest.raises(OasisStreamException, match=match):
        read_all(write_stream_files(tmp_path, [blob]), max_summary_id=10)


def test_read_events_sentinel_with_pairs(tmp_path):
    """An E:S0 sentinel may only hold the (0, 0) terminator."""
    sentinel_header = np.array([(1, 0, 0.0)], dtype=record_header_dtype)  # event 1, summary_id 0, expval 0
    sentinel_pairs = np.array([(1, 1.0), TERMINATOR], dtype=sidx_loss_dtype)  # a sample pair before the terminator
    blob = make_summary_stream([(1, [(1, 1, [(1, 1.0)])])], sentinel=False) + sentinel_header.tobytes() + sentinel_pairs.tobytes()

    with pytest.raises(OasisStreamException, match='S0 sentinel record must only hold'):
        read_all(write_stream_files(tmp_path, [blob]))


@pytest.mark.parametrize('cut', [
    pytest.param(record_header_dtype.itemsize + sidx_loss_dtype.itemsize, id='last sentinel missing'),
    pytest.param(record_header_dtype.itemsize + 2 * sidx_loss_dtype.itemsize - 3, id='cut inside the last record'),
])
def test_read_events_input_ends_mid_event(tmp_path, cut):
    """An input that ends before its last event is closed by E:S0 raises an error, whether the stream
    stops after a full record or part way through one."""
    blob = make_summary_stream(IP1)[:-cut]
    with pytest.raises(OasisException, match='input 0 ended in the middle of event 12'):
        read_all(write_stream_files(tmp_path, [blob]))


@pytest.mark.parametrize('blobs, error, match', [
    pytest.param([make_summary_stream(IP1), make_summary_stream(IP2, summary_set_id=2)], OasisException,
                 'different summary_set_id', id='different summary_set_id'),
    pytest.param([make_summary_stream(IP1), make_summary_stream(IP2, len_sample=4)], OSError,
                 'multiple stream type', id='different len_sample'),
])
def test_init_summary_streams_in_header_mismatch(tmp_path, blobs, error, match):
    """Inputs whose headers disagree are rejected before any event is read."""
    with ExitStack() as stack, pytest.raises(error, match=match):
        init_summary_streams_in(write_stream_files(tmp_path, blobs), stack)


def test_init_summary_streams_in_not_a_summary_stream(tmp_path):
    """A stream that is not a summary stream is rejected."""
    path = tmp_path / 'fm_stream'
    path.write_bytes(stream_info_to_bytes(FM_STREAM_ID, ITEM_STREAM) + np.int32(3).tobytes() + np.int32(1).tobytes())
    with ExitStack() as stack, pytest.raises(OasisException, match='summary stream'):
        init_summary_streams_in([str(path)], stack)


def test_init_summary_streams_in_returns_header(tmp_path):
    """The streams come back positioned after the header, with len_sample and summary_set_id."""
    files = write_stream_files(tmp_path, [make_summary_stream(IP1), make_summary_stream(IP2)])
    with ExitStack() as stack:
        streams_in, len_sample, summary_set_id = init_summary_streams_in(files, stack)
        assert (len(streams_in), len_sample, summary_set_id) == (2, 3, 1)
        assert all(stream.tell() == 12 for stream in streams_in)
