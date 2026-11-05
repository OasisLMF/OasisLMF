"""Tests of the event load balancer that fans loss streams out to several downstream processes."""
import struct
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oasislmf.execution.load_balancer import balance

HEADER = struct.pack('<ii', 0x01000002, 3)  # gul item stream, max sample index 3
RECORD_TERMINATOR = struct.pack('<if', 0, 0.)


def item_record(event_id, item_id, losses):
    """Encode one item record: event_id, item_id, then (sidx, loss) pairs closed by a (0, 0) pair."""
    body = b''.join(struct.pack('<if', sidx, loss) for sidx, loss in losses)
    return struct.pack('<ii', event_id, item_id) + body + RECORD_TERMINATOR


def parse_records(data):
    """Split a stream body (after the header) into (event_id, record bytes) pairs."""
    records = []
    cursor = 0
    while cursor < len(data):
        start = cursor
        event_id, _ = struct.unpack_from('<ii', data, cursor)
        cursor += 8
        while True:
            sidx, _ = struct.unpack_from('<if', data, cursor)
            cursor += 8
            if sidx == 0:
                break
        records.append((event_id, data[start:cursor]))
    return records


def run_balance(input_streams, n_outputs, read_size, write_size=64, queue_size=4):
    """Write `input_streams` (lists of record bytes) to files, balance them and return each output's bytes."""
    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        pipe_in = []
        for i, records in enumerate(input_streams):
            path = tmp / f'in_{i}.bin'
            path.write_bytes(HEADER + b''.join(records))
            pipe_in.append(str(path))
        pipe_out = [str(tmp / f'out_{i}.bin') for i in range(n_outputs)]
        balance(pipe_in, pipe_out, read_size, write_size, queue_size)
        return [Path(p).read_bytes() for p in pipe_out]


losses = st.lists(st.tuples(st.sampled_from([-3, -2, -1, 1, 2, 3]), st.floats(0, 1e6, width=32)), max_size=4)


@st.composite
def input_streams(draw):
    """Draw 1-3 input streams; each holds whole events with ids no other stream uses."""
    n_inputs = draw(st.integers(1, 3))
    streams = []
    next_event_id = 1
    for _ in range(n_inputs):
        records = []
        for _ in range(draw(st.integers(0, 6))):
            for item_id in range(1, draw(st.integers(1, 4)) + 1):
                records.append(item_record(next_event_id, item_id, draw(losses)))
            next_event_id += 1
        streams.append(records)
    return streams


@settings(max_examples=100, deadline=None)
@given(streams=input_streams(), n_outputs=st.integers(1, 3), read_size=st.sampled_from([32, 36, 44, 104, 152, 256, 1_048_576]))
def test_every_record_comes_out_once_and_events_are_not_split(streams, n_outputs, read_size):
    """The largest record drawn is 48 bytes, so small read sizes carry partial records over between reads and the
    smallest ones must grow to fit a whole record."""
    outputs = run_balance(streams, n_outputs, read_size)

    assert all(out[:8] == HEADER for out in outputs)
    parsed = [parse_records(out[8:]) for out in outputs]
    assert Counter(rec for out in parsed for _, rec in out) == Counter(rec for stream in streams for rec in stream)

    event_outputs = {}
    for out_i, out in enumerate(parsed):
        for event_id, _ in out:
            event_outputs.setdefault(event_id, set()).add(out_i)
    assert all(len(outs) == 1 for outs in event_outputs.values()), "an event was split across outputs"


def test_items_of_an_event_keep_their_order():
    records = [item_record(1, item_id, [(1, float(item_id))]) for item_id in range(1, 6)]

    [output] = run_balance([records], n_outputs=1, read_size=48)

    assert [rec for _, rec in parse_records(output[8:])] == records


def test_an_empty_input_stream_produces_only_the_header():
    outputs = run_balance([[]], n_outputs=2, read_size=64)

    assert outputs == [HEADER, HEADER]


@pytest.mark.parametrize('n_samples', [2, 19], ids=['record_fills_the_read_buffer', 'record_larger_than_the_read_buffer'])
def test_a_record_that_does_not_fit_the_read_buffer_does_not_truncate_the_stream(n_samples):
    """A record whose terminator lands in the last 8 bytes of the read buffer is carried over whole; when
    it fills the buffer, the next read gets no space, returns 0 and is taken for the end of the stream.
    See https://github.com/OasisLMF/OasisLMF/issues/2254.
    """
    records = [item_record(event_id, 1, [(sidx, float(sidx)) for sidx in range(1, n_samples + 1)]) for event_id in range(1, 4)]

    [output] = run_balance([records], n_outputs=1, read_size=32)

    assert output == HEADER + b''.join(records)


def test_inputs_with_different_headers_are_rejected():
    with TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        (tmp / 'a.bin').write_bytes(HEADER)
        (tmp / 'b.bin').write_bytes(struct.pack('<ii', 0x01000002, 10))
        try:
            balance([str(tmp / 'a.bin'), str(tmp / 'b.bin')], [str(tmp / 'out.bin')], 64, 64, 4)
        except Exception as err:
            assert 'different header' in str(err)
        else:
            raise AssertionError('mismatched headers were accepted')
