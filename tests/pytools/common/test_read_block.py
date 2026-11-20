"""EventReader.read_block must load and yield events exactly as read_streams does for the same bytes."""
import logging

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import (EventReader, LOSS_STREAM_ID, ITEM_STREAM, PIPE_CAPACITY,
                                                  stream_info_to_bytes)

PAIR_DTYPE = np.dtype([('sidx', '<i4'), ('loss', '<f4')])
HEADER_DTYPE = np.dtype([('event_id', '<i4'), ('item_id', '<i4')])


class RecordingReader(EventReader):
    """Minimal reader: collects each event's (item_id, sidx, loss) records and the hook calls."""

    def __init__(self):
        self.current = []
        self.events = []
        self.calls = []
        self.logger = logging.getLogger(__name__)

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        last_event_id = event_id
        while True:
            if item_id:
                if valid_buff - cursor < PAIR_DTYPE.itemsize:
                    break
                pair = byte_mv[cursor:cursor + PAIR_DTYPE.itemsize].view(PAIR_DTYPE)[0]
                cursor += PAIR_DTYPE.itemsize
                if pair['sidx']:
                    self.current.append((int(item_id), int(pair['sidx']), float(pair['loss'])))
                else:
                    item_id = 0
            else:
                if valid_buff - cursor < HEADER_DTYPE.itemsize:
                    break
                header = byte_mv[cursor:cursor + HEADER_DTYPE.itemsize].view(HEADER_DTYPE)[0]
                event_id = int(header['event_id'])
                if event_id != last_event_id:
                    if last_event_id:
                        return cursor, last_event_id, 0, 1
                    last_event_id = event_id
                item_id = int(header['item_id'])
                cursor += HEADER_DTYPE.itemsize
        return cursor, event_id, item_id, 0

    def item_exit(self):
        self.calls.append('item_exit')

    def event_read_log(self, event_id):
        self.calls.append(('log', event_id))

    def finalize_event(self):
        self.calls.append('finalize')
        self.events.append(self.current)
        self.current = []


def make_events(n_events, n_items, n_samples, first_event_id=1):
    """Return the bytes of a loss stream body (no header) with every item closed by a (0, 0) delimiter."""
    rng = np.random.default_rng(n_events * 1000 + n_items * 10 + n_samples)
    parts = []
    for event_id in range(first_event_id, first_event_id + n_events):
        for item_id in range(1, n_items + 1):
            parts.append(np.array([(event_id, item_id)], dtype=HEADER_DTYPE).tobytes())
            pairs = np.zeros(n_samples + 1, dtype=PAIR_DTYPE)
            pairs['sidx'][:n_samples] = np.arange(1, n_samples + 1)
            pairs['loss'][:n_samples] = rng.random(n_samples, dtype=np.float32)
            parts.append(pairs.tobytes())
    return b''.join(parts)


def read_with_streams(tmp_path, body):
    path = tmp_path / 'stream.bin'
    path.write_bytes(stream_info_to_bytes(LOSS_STREAM_ID, ITEM_STREAM) + np.int32(10).tobytes() + body)
    reader = RecordingReader()
    with open(path, 'rb') as stream_in:
        stream_in.read(8)
        event_ids = list(reader.read_streams([stream_in]))
    return reader, event_ids


def read_with_block(block, n=None):
    reader = RecordingReader()
    event_ids = list(reader.read_block(block, n))
    return reader, event_ids


def logged_events(reader):
    return [call[1] for call in reader.calls if isinstance(call, tuple)]


@pytest.mark.parametrize('n_events, n_items, n_samples', [
    (1, 1, 3),
    (1, 5, 10),
    (4, 3, 2),
    (60, 20, 10),  # larger than PIPE_CAPACITY, so read_streams goes through read_event too
])
def test_read_block_matches_read_streams(tmp_path, n_events, n_items, n_samples):
    body = make_events(n_events, n_items, n_samples)
    stream_reader, stream_ids = read_with_streams(tmp_path, body)
    block_reader, block_ids = read_with_block(np.frombuffer(bytearray(body), dtype=np.int8))

    assert block_ids == stream_ids == list(range(1, n_events + 1))
    assert block_reader.events == stream_reader.events
    assert logged_events(block_reader) == logged_events(stream_reader) == block_ids
    assert block_reader.calls.count('finalize') == n_events
    if n_events == 60:
        assert len(body) > PIPE_CAPACITY


def test_read_block_calls_hooks_in_order_for_each_event():
    block = np.frombuffer(make_events(3, 2, 2), dtype=np.int8)
    reader, event_ids = read_with_block(block)

    assert event_ids == [1, 2, 3]
    assert reader.calls == [call for event_id in event_ids for call in ('item_exit', ('log', event_id), 'finalize')]


def test_read_block_yields_each_event_before_reading_the_next():
    block = np.frombuffer(make_events(3, 2, 2), dtype=np.int8)
    reader = RecordingReader()
    seen = []
    for event_id in reader.read_block(block):
        seen.append((event_id, len(reader.events)))
    assert seen == [(1, 1), (2, 2), (3, 3)]


def test_read_block_one_event_per_block_matches_whole_block():
    body = make_events(5, 3, 4)
    whole_reader, whole_ids = read_with_block(np.frombuffer(body, dtype=np.int8))

    event_size = len(body) // 5
    split_reader = RecordingReader()
    split_ids = []
    for i in range(5):
        split_ids.extend(split_reader.read_block(np.frombuffer(body[i * event_size:(i + 1) * event_size], dtype=np.int8)))

    assert split_ids == whole_ids == [1, 2, 3, 4, 5]
    assert split_reader.events == whole_reader.events


def test_read_block_last_event_ends_at_block_end():
    body = make_events(2, 2, 3)
    reader, event_ids = read_with_block(np.frombuffer(body, dtype=np.int8))
    assert event_ids == [1, 2]
    assert len(reader.events[1]) == 2 * 3


def test_read_block_empty_block_yields_nothing():
    reader, event_ids = read_with_block(np.zeros(0, dtype=np.int8))
    assert event_ids == []
    assert reader.calls == []


def test_read_block_n_limits_the_valid_bytes():
    body = make_events(3, 2, 2)
    buffer = np.full(len(body) + 1000, -1, dtype=np.int8)
    buffer[:len(body)] = np.frombuffer(body, dtype=np.int8)

    reader, event_ids = read_with_block(buffer, len(body))
    expected_reader, expected_ids = read_with_block(np.frombuffer(body, dtype=np.int8))
    assert event_ids == expected_ids == [1, 2, 3]
    assert reader.events == expected_reader.events

    assert read_with_block(buffer, 0)[1] == []


def test_read_block_accepts_read_only_block():
    body = make_events(3, 2, 2)
    block = np.frombuffer(body, dtype=np.int8)
    assert not block.flags.writeable
    assert read_with_block(block)[1] == [1, 2, 3]
