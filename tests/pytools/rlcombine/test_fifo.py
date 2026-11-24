"""rlcombine's inputs are fed by one fmpy chain, so an input can stop sending until rlcombine has read
another one. The reader must read whatever any input has sent, without waiting on a single input."""
import os
import signal
from contextlib import contextmanager

from oasislmf.pytools.rlcombine.reader import SummaryCombineReader

from .common import IP1, IP2, buffer_to_records, event_bytes, make_summary_stream

HEADER_SIZE = len(make_summary_stream([]))


@contextmanager
def fails_if_blocked(seconds=10):
    """Raise TimeoutError instead of hanging if the reader blocks."""
    def blocked(signum, frame):
        raise TimeoutError('the reader is blocked waiting on a single input')

    previous_handler = signal.signal(signal.SIGALRM, blocked)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def test_reader_reads_ip2_while_ip1_is_waiting():
    """IP1 stops part way through its header, then part way through its first event. IP2 is read meanwhile."""
    ip1, ip2 = make_summary_stream(IP1), make_summary_stream(IP2)
    ip1_mid_event = HEADER_SIZE + len(event_bytes(*IP1[0])) // 2
    ip2_first_event = HEADER_SIZE + len(event_bytes(*IP2[0]))
    (ip1_read, ip1_write), (ip2_read, ip2_write) = os.pipe(), os.pipe()

    with open(ip1_read, 'rb', buffering=0) as ip1_in, open(ip2_read, 'rb', buffering=0) as ip2_in, fails_if_blocked():
        reader = SummaryCombineReader([ip1_in, ip2_in])

        # IP1 has only sent half its header: IP2's header and first event are read anyway
        os.write(ip1_write, ip1[:HEADER_SIZE // 2])
        os.write(ip2_write, ip2[:ip2_first_event])
        reader.read_available()
        assert reader.pending[1] == ip2[:ip2_first_event]

        # IP1 is in the middle of its first event: IP2's events, sent now, are read and yielded anyway
        os.write(ip1_write, ip1[HEADER_SIZE // 2:ip1_mid_event])
        assert reader.read_headers() == (3, 1)
        events = reader.read_events(max_summary_id=10)
        os.write(ip2_write, ip2[ip2_first_event:])
        os.close(ip2_write)
        assert [(file_idx, event_id, buffer_to_records(buffer)) for file_idx, event_id, buffer in
                (next(events) for _ in IP2)] == [(1, *event) for event in IP2]

        # IP1 sends the rest
        os.write(ip1_write, ip1[ip1_mid_event:])
        os.close(ip1_write)
        assert [(file_idx, event_id, buffer_to_records(buffer)) for file_idx, event_id, buffer in events] == \
            [(0, *event) for event in IP1]
