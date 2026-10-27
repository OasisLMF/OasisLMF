import os
import threading
import time
from contextlib import ExitStack

import numpy as np
import pytest

from oasislmf.pytools.common.event_stream import stream_info_to_bytes, SUMMARY_STREAM_ID, ITEM_STREAM
from oasislmf.pytools.rlcombine.wip_reader import SummaryCombineReader, init_summary_streams_in
from oasislmf.utils.exceptions import OasisException, OasisStreamException


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


def read_all(files_in, max_summary_id=10):
    """Run the reader and return {file_idx: [(event_id, [(summary_id, expval, [(sidx, loss), ...]), ...]), ...]}."""
    with ExitStack() as stack:
        streams_in, len_sample, summary_set_id = init_summary_streams_in(files_in, stack)
        reader = SummaryCombineReader(len(streams_in), max_summary_id, len_sample)
        events = {i: [] for i in range(len(streams_in))}
        for file_idx, event_id, summary_ids, expval, offsets, sidx, loss in reader.read_events(streams_in):
            records = [(int(summary_ids[i]), float(expval[i]),
                        list(zip(sidx[offsets[i]:offsets[i + 1]].tolist(), loss[offsets[i]:offsets[i + 1]].tolist())))
                       for i in range(len(summary_ids))]
            events[file_idx].append((int(event_id), records))
    return events


IP1 = [(10, [(1, 100, [(-5, 50), (-1, 5), (1, 4)]), (3, 100, [(-5, 9), (-1, 1)])]),
       (7, []),
       (12, [(2, 30, [(-5, 30), (-1, 3), (3, 2.5)])])]
IP2 = [(10, [(3, 10, [(-5, 1), (-1, 1), (2, 7)])]),
       (7, [(1, 5, [(-5, 2), (-1, 2)])]),
       (12, [])]


def test_yields_every_event_of_every_input(tmp_path):
    events = read_all(write_files(tmp_path, [summary_stream(IP1), summary_stream(IP2)]))
    assert events == {0: IP1, 1: IP2}


def test_record_larger_than_read_buffer(tmp_path):
    big = [(5, [(4, 1, [(i, 1.0) for i in range(1, 10001)])]), (6, [(1, 2, [(-5, 1)])])]  # first record ~80 KB
    events = read_all(write_files(tmp_path, [summary_stream(big, len_sample=10000)]))
    assert events == {0: big}


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
    events = read_all([f'/dev/fd/{r}' for r, _ in pipes])
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert events == {0: flood, 1: trickle}


@pytest.mark.parametrize('blobs, error', [
    ([summary_stream(IP1)[:-5], summary_stream(IP2)], OasisException),  # truncated input
    ([summary_stream(IP1, sentinel=False), summary_stream(IP2)], OasisStreamException),  # no S0 sentinels
    ([summary_stream(IP1), summary_stream(IP2, len_sample=4)], OSError),  # header mismatch (init_streams_in)
    ([summary_stream(IP1), summary_stream(IP2, summary_set_id=2)], OasisException),
    ([summary_stream([(1, [(11, 1, [])])])], OasisStreamException),  # summary_id above max_summary_id
    ([summary_stream([(1, [(1, 1, [(1, 1.0)] * 6)])])], OasisStreamException),  # more pairs than samples + specials
    ([summary_stream([(1, [(1, 1, [(-4, 1.0)])])])], OasisStreamException),  # -4 is not written for rl
    ([summary_stream([(1, [(1, 1, [(-3, 1.0)])])])], OasisStreamException),  # TIV belongs in the header expval
    ([summary_stream([(1, [(1, 1, [(-2, 1.0)])])])], OasisStreamException),  # std dev is never written
    ([summary_stream([(1, [(1, 1, [(4, 1.0)])])])], OasisStreamException),  # sample sidx above len_sample (3)
    ([summary_stream([(1, [(s, 1, [(-5, 1.0)]) for s in range(1, 12)])])], OasisStreamException),  # more records than max_summary_id
])
def test_invalid_inputs(tmp_path, blobs, error):
    with pytest.raises(error):
        read_all(write_files(tmp_path, blobs))
