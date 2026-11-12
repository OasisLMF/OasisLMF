"""Tests for combining the inuring priorities' events in rlcombine: accumulate_input_events, combine_events,
EventAccumulator, write_combined_event, and run() end to end."""
import numpy as np
import pytest

from oasislmf.pytools.common.data import summary_stream_index_dtype
from oasislmf.pytools.common.event_stream import PIPE_CAPACITY
from oasislmf.pytools.rlcombine.manager import (EventAccumulator, accumulate_input_events, combine_events, run,
                                                write_combined_event)
from oasislmf.pytools.rlcombine.reader import loss_dtype, sidx_dtype, summary_id_dtype
from oasislmf.utils.exceptions import OasisException

from .common import (IP1, IP1_IP2_COMBINED, IP2, event_bytes, make_summary_stream, write_rl_static,
                     write_stream_files)

MAX_SUMMARY_ID = 5
LEN_SAMPLE = 3


def records_to_buffer(records):
    """Turn ``[(summary_id, expval, [(sidx, loss), ...]), ...]`` into an event buffer like the ones
    SummaryCombineReader.read_events yields, with the reader's dtypes."""
    pairs = [pair for _, _, record_pairs in records for pair in record_pairs]
    return {
        'summary_ids': np.array([summary_id for summary_id, _, _ in records], dtype=summary_id_dtype),
        'expval': np.array([expval for _, expval, _ in records], dtype=loss_dtype),
        'offsets': np.cumsum([0] + [len(record_pairs) for _, _, record_pairs in records]).astype(np.int64),
        'sidx': np.array([sidx for sidx, _ in pairs], dtype=sidx_dtype),
        'loss': np.array([loss for _, loss in pairs], dtype=loss_dtype),
    }


def accumulate(acc, records):
    """Call accumulate_input_events on one input's records."""
    buffer = records_to_buffer(records)
    accumulate_input_events(buffer['summary_ids'], buffer['expval'], buffer['offsets'], buffer['sidx'], buffer['loss'],
                            acc.present, acc.expval, acc.max_loss, acc.mean, acc.losses)


def test_accumulate_sums_all_fields():
    """Two inputs' records for event 10 are summed field by field into the rows of their summary_ids;
    rows of summary_ids neither input has stay 0."""
    acc = EventAccumulator(MAX_SUMMARY_ID, LEN_SAMPLE)
    accumulate(acc, dict(IP1)[10])
    accumulate(acc, dict(IP2)[10])

    expected = dict(IP1_IP2_COMBINED)[10]
    for summary_id, expval, pairs in expected:
        row = summary_id - 1
        pairs = dict(pairs)
        assert acc.present[row]
        assert acc.expval[row] == expval
        assert acc.max_loss[row] == pairs[-5]
        assert acc.mean[row] == pairs[-1]
        np.testing.assert_array_equal(acc.losses[row], [pairs.get(sidx, 0) for sidx in range(1, LEN_SAMPLE + 1)])

    untouched = [summary_id - 1 for summary_id in range(1, MAX_SUMMARY_ID + 1)
                 if summary_id not in {summary_id for summary_id, _, _ in expected}]
    assert not acc.present[untouched].any()
    for values in (acc.expval, acc.max_loss, acc.mean, acc.losses):
        assert not values[untouched].any()


@pytest.mark.parametrize('summary_ids', [pytest.param([2, 1], id='decreasing'), pytest.param([1, 1], id='repeated')])
def test_accumulate_rejects_unsorted_or_repeated_ids(summary_ids):
    """The summary_ids of one input's event must be strictly increasing."""
    acc = EventAccumulator(MAX_SUMMARY_ID, LEN_SAMPLE)
    with pytest.raises(ValueError, match='summary_ids are not strictly increasing'):
        accumulate(acc, [(summary_id, 1.0, [(1, 1.0)]) for summary_id in summary_ids])


def test_accumulate_rejects_unknown_special_sidx():
    """Only the max loss (-5) and mean (-1) special sidx can be combined."""
    acc = EventAccumulator(MAX_SUMMARY_ID, LEN_SAMPLE)
    with pytest.raises(ValueError, match='unknown special sidx'):
        accumulate(acc, [(1, 1.0, [(-3, 1.0)])])


def test_combine_events_returns_sorted_union():
    """combine_events returns every summary_id that any input has for the event, sorted."""
    acc = EventAccumulator(MAX_SUMMARY_ID, LEN_SAMPLE)
    present_ids = combine_events([records_to_buffer([(3, 1.0, []), (5, 1.0, [])]),
                                  records_to_buffer([(1, 1.0, []), (5, 1.0, [])])], acc)
    np.testing.assert_array_equal(present_ids, [1, 3, 5])


def test_reset_clears_used_rows():
    """After reset, the next event starts from zero: nothing is left over from the previous one."""
    acc = EventAccumulator(MAX_SUMMARY_ID, LEN_SAMPLE)
    present_ids = combine_events([records_to_buffer(dict(ip)[10]) for ip in (IP1, IP2)], acc)
    acc.reset(present_ids)

    for values in (acc.present, acc.expval, acc.max_loss, acc.mean, acc.losses):
        assert not values.any()


def write_event(tmp_path, event_id, records_per_input, len_sample, output_zeros):
    """Combine one event from several inputs and write it with write_combined_event; return the bytes written."""
    max_summary_id = max(summary_id for records in records_per_input for summary_id, _, _ in records)
    acc = EventAccumulator(max_summary_id, len_sample)
    present_ids = combine_events([records_to_buffer(records) for records in records_per_input], acc)
    byte_mv = np.frombuffer(buffer=memoryview(bytearray(PIPE_CAPACITY)), dtype='b')
    out_path = tmp_path / 'combined.bin'
    with open(out_path, 'wb') as pipe:
        write_combined_event(pipe, None, byte_mv, event_id, len_sample, present_ids, acc, output_zeros,
                             np.zeros(1, dtype=np.int64), 0, np.empty(max_summary_id, dtype=summary_stream_index_dtype))
    return out_path.read_bytes()


@pytest.mark.parametrize('output_zeros, samples', [
    pytest.param(False, [(1, 2.0)], id='non-zero-samples'),
    pytest.param(True, [(1, 2.0), (2, 0.0), (3, 0.0)], id='all-samples'),
])
def test_write_combined_event_output_zeros(tmp_path, output_zeros, samples):
    """Sample 2 sums to 0 and sample 3 is in neither input: without output_zeros only the non-zero
    sample is written, with it every sample is. The max loss and mean are always written."""
    records_per_input = [[(1, 10.0, [(-5, 4.0), (-1, 1.0), (1, 2.0), (2, 0.0)])],
                         [(1, 5.0, [(-5, 1.0), (-1, 0.5), (2, 0.0)])]]
    expected = [(1, 15.0, [(-5, 5.0), (-1, 1.5)] + samples)]

    written = write_event(tmp_path, 7, records_per_input, LEN_SAMPLE, output_zeros)
    assert written == event_bytes(7, expected, sentinel=False)


def test_write_combined_event_larger_than_pipe_capacity(tmp_path):
    """An event larger than PIPE_CAPACITY is written across several flushes, byte for byte the same as
    its records written one after the other."""
    len_sample = 100
    summary_ids = range(1, 201)
    records = [(summary_id, 10.0, [(-5, 2.0), (-1, 1.0)] + [(sidx, 1.0) for sidx in range(1, len_sample + 1)])
               for summary_id in summary_ids]
    expected = event_bytes(1, [(summary_id, 20.0, [(-5, 4.0), (-1, 2.0)] + [(sidx, 2.0) for sidx in range(1, len_sample + 1)])
                               for summary_id in summary_ids], sentinel=False)
    assert len(expected) > 2 * PIPE_CAPACITY

    assert write_event(tmp_path, 1, [records, records], len_sample, output_zeros=False) == expected


def run_combine(tmp_path, ips, output_zeros=False):
    """Run rlcombine on the inputs' summary streams; return the bytes of the combined output."""
    files = write_stream_files(tmp_path, [make_summary_stream(ip) for ip in ips])
    static_paths = [write_rl_static(tmp_path / f'RI_{i + 1}', MAX_SUMMARY_ID) for i in range(len(ips))]
    out_path = tmp_path / 'rl_S1_summary_P1'
    run(files, str(out_path), static_paths, False, None, output_zeros)
    return out_path.read_bytes()


def test_run_output_is_sum_of_inputs(tmp_path):
    """The combined stream has the inputs' header and, for every event, the inputs' records summed
    per summary_id, in summary_id order and with no E:S0 sentinel."""
    expected = make_summary_stream(IP1_IP2_COMBINED, len_sample=LEN_SAMPLE, sentinel=False)
    assert run_combine(tmp_path, [IP1, IP2]) == expected


def test_run_event_sequence_mismatch(tmp_path):
    """Inputs must send their events in the same order."""
    with pytest.raises(OasisException, match='event sequence mismatch'):
        run_combine(tmp_path, [IP1, IP2[::-1]])


def test_run_input_ends_early(tmp_path):
    """An input with fewer events than the others leaves events that cannot be combined."""
    with pytest.raises(OasisException, match='only partially combined'):
        run_combine(tmp_path, [IP1, IP2[:-1]])
