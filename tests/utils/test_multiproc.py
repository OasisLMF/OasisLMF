from queue import Empty, Full

import pytest

from oasislmf.utils.multiproc import (
    chunk_producer,
    multiproc_worker,
    result_producer,
    run_multiproc,
    with_error_queue,
)


class FakeQueue:
    """A queue.Queue-like stub with no real blocking, so unit tests run synchronously."""

    def __init__(self, items=None, full_times=0, empty_times=0):
        self.items = list(items) if items is not None else []
        self.put_items = []
        self.full_times = full_times
        self.empty_times = empty_times

    def get(self, timeout=None):
        if self.empty_times > 0:
            self.empty_times -= 1
            raise Empty
        if not self.items:
            raise Empty
        return self.items.pop(0)

    def put(self, item, timeout=None):
        if self.full_times > 0:
            self.full_times -= 1
            raise Full
        self.put_items.append(item)

    def empty(self):
        return not self.items


class FakeErrorQueue:
    def __init__(self, has_error=False):
        self.put_items = []
        self._has_error = has_error

    def empty(self):
        return not self._has_error

    def put(self, item):
        self.put_items.append(item)
        self._has_error = True

    def get(self, timeout=None):
        return self.put_items.pop(0)


class TogglingErrorQueue:
    """Reports empty() as True for the first `true_calls` calls, then False."""

    def __init__(self, true_calls):
        self.calls = 0
        self.true_calls = true_calls

    def empty(self):
        self.calls += 1
        return self.calls <= self.true_calls


def test_with_error_queue_passes_through_return_value():
    @with_error_queue
    def fct(error_queue, x, y):
        return x + y

    error_queue = FakeErrorQueue()
    assert fct(error_queue, 2, 3) == 5
    assert error_queue.empty()


def test_with_error_queue_captures_exception():
    @with_error_queue
    def fct(error_queue):
        raise ValueError('boom')

    error_queue = FakeErrorQueue()
    result = fct(error_queue)

    assert result is None
    assert not error_queue.empty()
    exc_type, exc_value, _ = error_queue.get()
    assert exc_type is ValueError
    assert str(exc_value) == 'boom'


def test_chunk_producer_puts_items_then_sentinel():
    error_queue = FakeErrorQueue()
    chunk_queue = FakeQueue()

    chunk_producer(error_queue, [1, 2, 3], chunk_queue)

    assert error_queue.empty()
    assert chunk_queue.put_items == [1, 2, 3, None]


def test_chunk_producer_returns_immediately_if_error_already_present():
    error_queue = FakeErrorQueue(has_error=True)
    chunk_queue = FakeQueue()

    chunk_producer(error_queue, [1, 2, 3], chunk_queue)

    assert chunk_queue.put_items == []


def test_chunk_producer_retries_when_queue_full():
    error_queue = FakeErrorQueue()
    chunk_queue = FakeQueue(full_times=2)

    chunk_producer(error_queue, [1], chunk_queue)

    assert chunk_queue.put_items == [1, None]


def test_multiproc_worker_processes_chunks_and_reports_result():
    error_queue = FakeErrorQueue()
    chunk_queue = FakeQueue(items=[3, None])
    result_queue = FakeQueue()

    multiproc_worker(error_queue, lambda worker_id: (lambda chunk: chunk * 2), 0, chunk_queue, result_queue)

    assert error_queue.empty()
    assert chunk_queue.put_items == [None]
    assert result_queue.put_items == [6, None]


def test_multiproc_worker_passes_its_own_worker_id_to_make_process_chunk():
    error_queue = FakeErrorQueue()
    chunk_queue = FakeQueue(items=['x', None])
    result_queue = FakeQueue()

    multiproc_worker(error_queue, lambda worker_id: (lambda chunk: f'{chunk}-{worker_id}'), 7, chunk_queue, result_queue)

    assert result_queue.put_items == ['x-7', None]


def test_multiproc_worker_returns_immediately_if_error_already_present():
    error_queue = FakeErrorQueue(has_error=True)
    chunk_queue = FakeQueue()
    result_queue = FakeQueue()

    multiproc_worker(error_queue, lambda worker_id: (lambda chunk: chunk), 0, chunk_queue, result_queue)

    assert chunk_queue.put_items == []
    assert result_queue.put_items == []


def test_multiproc_worker_retries_get_on_empty_queue():
    error_queue = FakeErrorQueue()
    chunk_queue = FakeQueue(items=[1, None], empty_times=1)
    result_queue = FakeQueue()

    multiproc_worker(error_queue, lambda worker_id: (lambda chunk: chunk), 0, chunk_queue, result_queue)

    assert result_queue.put_items == [1, None]


def test_multiproc_worker_retries_put_when_result_queue_full():
    error_queue = FakeErrorQueue()
    chunk_queue = FakeQueue(items=[1, None])
    result_queue = FakeQueue(full_times=2)

    multiproc_worker(error_queue, lambda worker_id: (lambda chunk: chunk), 0, chunk_queue, result_queue)

    assert result_queue.put_items == [1, None]


def test_result_producer_yields_until_all_workers_finish():
    result_queue = FakeQueue(items=['r1', None, 'r2', None])
    error_queue = FakeErrorQueue()

    results = list(result_producer(result_queue, error_queue, worker_count=2))

    assert results == ['r1', 'r2']


def test_result_producer_stops_early_if_error_present():
    result_queue = FakeQueue(items=['r1'])
    error_queue = FakeErrorQueue(has_error=True)

    results = list(result_producer(result_queue, error_queue, worker_count=2))

    assert results == []


def test_result_producer_retries_get_on_empty_queue():
    result_queue = FakeQueue(items=[None], empty_times=1)
    error_queue = FakeErrorQueue()

    results = list(result_producer(result_queue, error_queue, worker_count=1))

    assert results == []


def test_result_producer_stops_if_error_appears_mid_loop():
    error_queue = TogglingErrorQueue(true_calls=1)
    result_queue = FakeQueue()

    results = list(result_producer(result_queue, error_queue, worker_count=5))

    assert results == []


@pytest.mark.parametrize('pool_count', [0, 1])
def test_run_multiproc_rejects_pool_count_of_one_or_fewer(pool_count):
    """Guards against a caller forgetting to check pool_count > 1 before calling in - it should
    fail loudly rather than silently spin up a full process pool for a single chunk."""
    with pytest.raises(ValueError, match='pool_count'):
        run_multiproc([1, 2, 3], lambda worker_id: (lambda chunk: chunk), pool_count, list)


def test_run_multiproc_end_to_end():
    """A real (small) multiprocess run: doubles each chunk across 2 worker processes and
    checks every input is accounted for exactly once in the merged results."""
    results = run_multiproc(
        range(1, 7),
        lambda worker_id: (lambda chunk: chunk * 2),
        pool_count=2,
        on_results=lambda results: sorted(results),
    )

    assert results == [2, 4, 6, 8, 10, 12]


def test_run_multiproc_propagates_worker_exception():
    """If a worker raises, the original exception must propagate out of run_multiproc rather
    than being swallowed."""
    def process_chunk(chunk):
        raise ValueError('boom-for-test')

    with pytest.raises(ValueError, match='boom-for-test'):
        run_multiproc([1, 2, 3], lambda worker_id: process_chunk, pool_count=2, on_results=list)
