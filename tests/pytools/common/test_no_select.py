import logging
import os
import select
import subprocess
import sys
import threading

import numpy as np
import pytest

from oasislmf.pytools.common import event_stream
from oasislmf.pytools.common.event_stream import EventReader, SequentialSelector, write_mv_to_stream


class _IdReader(EventReader):
    """Each event is one int32 event id; enough to drive ``read_streams``."""

    logger = logging.getLogger(__name__)

    def read_buffer(self, byte_mv, cursor, valid_buff, event_id, item_id, **kwargs):
        if valid_buff - cursor < 4:
            return cursor, event_id, item_id, 0
        return cursor + 4, int(byte_mv[cursor:cursor + 4].view(np.int32)[0]), 0, 1


def _flag_in_child(value):
    env = {k: v for k, v in os.environ.items() if k != 'OASIS_PYTOOLS_NO_SELECT'}
    if value is not None:
        env['OASIS_PYTOOLS_NO_SELECT'] = value
    script = ('from oasislmf.pytools.common import event_stream as e; import select\n'
              'print(e.NO_SELECT, e.select_streams is select.select)')
    return subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, check=True).stdout.split()


@pytest.mark.skipif(os.name == 'nt', reason='always on on Windows')
@pytest.mark.parametrize('value, expected', [(None, ['False', 'True']), ('', ['False', 'True']), ('0', ['False', 'True']),
                                             ('1', ['True', 'False'])])
def test_no_select_flag(value, expected):
    assert _flag_in_child(value) == expected


def test_sequential_selector():
    selector = SequentialSelector()
    assert selector.select() == []
    a, b = object(), object()
    selector.register(a, 1, 'da')
    selector.register(b, 1, 'db')
    assert [(key.fileobj, key.data, events) for key, events in selector.select()] == [(a, 'da', 1)]
    assert selector.unregister(a).data == 'da'
    assert [key.fileobj for key, _ in selector.select()] == [b]
    assert list(selector.get_map()) == [b]
    selector.close()
    assert not selector.get_map()


@pytest.mark.parametrize('no_select', [False, True])
def test_read_streams_all_events(monkeypatch, no_select):
    """Every event of every pipe is read, multiplexed or (no select) one stream after the other."""
    monkeypatch.setattr(event_stream, 'NO_SELECT', no_select)
    ids = {0: list(range(1, 20001)), 1: list(range(100001, 120001))}
    pipes = [os.pipe() for _ in ids]
    writers = [threading.Thread(target=lambda fd, values: (os.write(fd, np.array(values, dtype=np.int32).tobytes()), os.close(fd)),
                                args=(w, ids[i])) for i, (_r, w) in enumerate(pipes)]
    for t in writers:
        t.start()
    streams = [os.fdopen(r, 'rb') for r, _w in pipes]
    try:
        got = list(_IdReader().read_streams(streams))
    finally:
        for t in writers:
            t.join()
        for s in streams:
            s.close()
    assert sorted(got) == ids[0] + ids[1]


@pytest.mark.parametrize('no_select', [False, True])
def test_write_mv_to_stream(monkeypatch, no_select):
    monkeypatch.setattr(event_stream, 'select_streams', event_stream._select_no_wait if no_select else select.select)
    r, w = os.pipe()
    payload = np.arange(200000, dtype=np.int32).view('b')
    received = bytearray()

    def drain():
        for chunk in iter(lambda: os.read(r, 65536), b''):
            received.extend(chunk)

    reader = threading.Thread(target=drain)
    reader.start()
    with os.fdopen(w, 'wb', buffering=0) as stream:
        write_mv_to_stream(stream, payload, len(payload))
    reader.join()
    os.close(r)
    assert bytes(received) == payload.tobytes()
