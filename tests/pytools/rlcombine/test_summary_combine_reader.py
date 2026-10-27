import os
import threading
import time
from contextlib import ExitStack

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import stream_info_to_bytes, SUMMARY_STREAM_ID, ITEM_STREAM
from oasislmf.pytools.rlcombine.wip_reader import CombineStreamError, SummaryCombineReader, init_summary_streams_in
from oasislmf.utils.exceptions import OasisStreamException


def summary_stream(events, len_sample=3, summary_set_id=1, sentinel=True):
    """Build a summary stream with an E:S0 sentinel after every event.

    events: [(event_id, [(summary_id, expval, [(sidx, loss), ...]), ...]), ...]
    """
    out = bytearray(stream_info_to_bytes(SUMMARY_STREAM_ID, ITEM_STREAM))
    out += np.array([len_sample, summary_set_id], '<i4').tobytes()
    for event_id, records in events:
        for summary_id, expval, pairs in records:
            out += np.array([event_id, summary_id], '<i4').tobytes() + np.float32(expval).tobytes()
            for sidx, loss in pairs + [(0, 0.0)]:
                out += np.int32(sidx).tobytes() + np.float32(loss).tobytes()
        if sentinel:
            out += np.array([event_id, 0], '<i4').tobytes() + np.float32(0).tobytes()
            out += np.int32(0).tobytes() + np.float32(0).tobytes()
    return bytes(out)


def write_files(tmp_path, blobs):
    paths = []
    for i, blob in enumerate(blobs):
        path = tmp_path / f'rl_IP{i + 1}_S1_combine_P1'
        path.write_bytes(blob)
        paths.append(str(path))
    return paths


def combine(files_in, max_summary_id=10, n_slots=64):
    with ExitStack() as stack:
        streams_in, len_sample, summary_set_id = init_summary_streams_in(files_in, stack)
        reader = SummaryCombineReader(len(streams_in), len_sample, max_summary_id, n_slots=n_slots)
        return [(event_id, summary_ids.tolist(), expval.tolist(), losses.tolist())
                for event_id, summary_ids, expval, losses in reader.read_combined_events(streams_in)]


IP1 = [(10, [(1, 100, [(-5, 50), (-1, 5), (1, 4)]), (3, 100, [(-5, 9), (-1, 1)])]),
       (7, []),
       (12, [(2, 30, [(-5, 30), (-1, 3), (3, 2.5)])])]
IP2 = [(10, [(3, 10, [(-5, 1), (-1, 1), (2, 7)])]),
       (7, [(1, 5, [(-5, 2), (-1, 2)])]),
       (12, [])]


def test_sums_per_event_and_summary_id(tmp_path):
    result = combine(write_files(tmp_path, [summary_stream(IP1), summary_stream(IP2)]))
    # losses columns: max loss, mean, sample 1, sample 2, sample 3
    assert result == [
        (10, [1, 3], [100.0, 110.0], [[50, 5, 4, 0, 0], [10, 2, 0, 7, 0]]),
        (7, [1], [5.0], [[2, 2, 0, 0, 0]]),
        (12, [2], [30.0], [[30, 3, 0, 0, 2.5]]),
    ]


def test_record_larger_than_read_buffer(tmp_path):
    events = [(5, [(4, 1, [(i, 1.0) for i in range(1, 10001)])])]  # ~80 KB record
    result = combine(write_files(tmp_path, [summary_stream(events, len_sample=10000)] * 2))
    assert result[0][0] == 5
    assert result[0][3][0][2:] == [2.0] * 10000


def test_pipes_with_lagging_input():
    """One input floods well past pipe capacity while the other trickles; no writer may block."""
    n_events = 60
    flood = [(e, [(1, 1, [(i, 1.0) for i in range(1, 3001)])]) for e in range(1, n_events)]
    trickle = [(e, []) for e in range(1, n_events)]
    blobs = [summary_stream(flood, len_sample=3000), summary_stream(trickle, len_sample=3000)]
    pipes = [os.pipe() for _ in blobs]

    def writer(fd, blob, delay):
        with os.fdopen(fd, 'wb', buffering=0) as f:
            for i in range(0, len(blob), 4096):
                f.write(blob[i:i + 4096])
                time.sleep(delay)

    threads = [threading.Thread(target=writer, args=(w, blob, delay))
               for (_, w), blob, delay in zip(pipes, blobs, (0, 0.01))]
    for thread in threads:
        thread.start()
    result = combine([f'/dev/fd/{r}' for r, _ in pipes])
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert [event_id for event_id, *_ in result] == list(range(1, n_events))


@pytest.mark.parametrize('blobs, error', [
    ([summary_stream(IP1), summary_stream([IP2[0], IP2[2]])], CombineStreamError),  # event 7 missing on IP2
    ([summary_stream(IP1)[:-5], summary_stream(IP2)], CombineStreamError),  # truncated input
    ([summary_stream(IP1, sentinel=False), summary_stream(IP2)], OasisStreamException),  # no S0 sentinels
    ([summary_stream(IP1), summary_stream(IP2, len_sample=4)], OSError),  # header mismatch (init_streams_in)
    ([summary_stream(IP1), summary_stream(IP2, summary_set_id=2)], CombineStreamError),
])
def test_invalid_inputs(tmp_path, blobs, error):
    with pytest.raises(error):
        combine(write_files(tmp_path, blobs))
