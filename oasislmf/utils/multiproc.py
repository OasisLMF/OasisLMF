"""Generic fork-based multiprocessing engine: split an iterable of chunks across a pool of
worker processes, apply a per-chunk function, and stream the results back to the caller.

Shared by oasislmf.lookup.factory (keys/lookup generation) and
oasislmf.computation.hooks.pre_analysis_multiproc (pre-analysis hook parallelisation), which
otherwise each maintained their own copy of this producer/worker/result-relay machinery.
"""
__all__ = [
    'with_error_queue',
    'chunk_producer',
    'multiproc_worker',
    'result_producer',
    'reorder_results',
    'run_multiproc',
]

import pickle
import sys

try:
    import billiard as multiprocessing
except ImportError:
    import multiprocessing

from queue import Empty, Full

# add pickling support for traceback object
import tblib.pickling_support

tblib.pickling_support.install()


def with_error_queue(fct):
    """Decorator for a function run in a worker/producer process: any exception it raises is
    put on error_queue (as sys.exc_info()) rather than propagating, so the parent process can
    re-raise it with its original traceback once all child processes have been torn down.
    """
    def wrapped_fct(error_queue, *args, **kwargs):
        try:
            return fct(error_queue, *args, **kwargs)
        except Exception:
            error_queue.put(sys.exc_info())

    return wrapped_fct


@with_error_queue
def chunk_producer(error_queue, chunks, chunk_queue):
    """Put each item from the `chunks` iterable onto chunk_queue, followed by a None sentinel
    to signal completion to the first worker to see it (which relays it to the next, and so
    on - see multiproc_worker).
    """
    chunks_iter = iter(chunks)
    chunk = True
    while chunk is not None:
        chunk = next(chunks_iter, None)
        while error_queue.empty():
            try:
                chunk_queue.put(chunk, timeout=5)
                break
            except Full:
                pass
        else:
            return


@with_error_queue
def multiproc_worker(error_queue, make_process_chunk, worker_id, chunk_queue, result_queue):
    """Build this worker's process_chunk(chunk) callable once via make_process_chunk(worker_id)
    - called here, inside the forked worker process, so any setup it does (e.g. constructing a
    stateful per-worker object) is properly worker-scoped - then pull chunks from chunk_queue
    and call process_chunk(chunk) on each, putting the return value on result_queue, until the
    None sentinel is seen - which is relayed back onto chunk_queue for the next worker, and
    reported on result_queue as this worker finishing.

    Each result is pickled here, before it's put on result_queue: Queue.put() otherwise pickles
    in a background feeder thread, where a failure (e.g. a hook returning an object holding a
    lock or connection) never reaches this process's error handling - the result is dropped, or
    the feeder thread dies and the run hangs waiting for it. Pickling here raises in the worker
    instead, so the error is reported through error_queue like any other.
    """
    process_chunk = make_process_chunk(worker_id)
    while True:
        while error_queue.empty():
            try:
                chunk = chunk_queue.get(timeout=5)
                break
            except Empty:
                pass
        else:
            return

        if chunk is None:
            chunk_queue.put(None)
            result_queue.put(None)
            break

        result = pickle.dumps(process_chunk(chunk), protocol=pickle.HIGHEST_PROTOCOL)

        while error_queue.empty():
            try:
                result_queue.put(result, timeout=5)
                break
            except Full:
                pass
        else:
            return


def result_producer(result_queue, error_queue, worker_count):
    """Yield each non-None result taken off result_queue (unpickled - see multiproc_worker),
    until worker_count workers have reported finished (a None result each), or error_queue
    receives an error.
    """
    finished_workers = 0
    while finished_workers < worker_count and error_queue.empty():
        while error_queue.empty():
            try:
                res = result_queue.get(timeout=5)
                break
            except Empty:
                pass
        else:
            break

        if res is None:
            finished_workers += 1
        else:
            yield pickle.loads(res)


def reorder_results(indexed_results):
    """Yield the result from each (index, result) pair in index order (0, 1, 2, ...),
    buffering any result that arrives before its predecessors.
    """
    pending = {}
    next_index = 0
    for index, result in indexed_results:
        pending[index] = result
        while next_index in pending:
            yield pending.pop(next_index)
            next_index += 1


def _indexed(make_process_chunk):
    """Wrap make_process_chunk so its process_chunk takes and returns (index, ...) pairs."""
    def make_indexed_process_chunk(worker_id):
        process_chunk = make_process_chunk(worker_id)
        return lambda indexed_chunk: (indexed_chunk[0], process_chunk(indexed_chunk[1]))

    return make_indexed_process_chunk


def run_multiproc(chunks, make_process_chunk, pool_count, on_results, ordered=False):
    """Run make_process_chunk(worker_id)(chunk) for each chunk in `chunks`, across pool_count
    worker processes, and call on_results(...) in the main process with a generator yielding
    each result as it arrives.

    Args:
        chunks: an iterable of chunk objects, iterated in a separate forked process (so the
                iterable itself only needs to be safe to pass through a fork). Each chunk is
                sent to a worker through a queue, so must be picklable.
        make_process_chunk (worker_id) -> ((chunk) -> result): called once per worker process,
                inside that (already forked) process, to build its process_chunk callable - so
                any setup it does is properly worker-scoped. worker_id is that worker's index
                in [0, pool_count). A callable ignoring worker_id and always returning the same
                process_chunk is fine when no per-worker setup is needed.
        pool_count (int): number of worker processes; must be > 1 (the caller should process
                          chunks directly, without this function, for pool_count <= 1).
        on_results (callable): called once in the main process, with a generator yielding each
                result as it arrives - (Iterator[result]) -> T. Must fully consume the
                generator it's given. Its return value is returned from run_multiproc.
        ordered (bool): if True, on_results receives results in the same order as `chunks`
                rather than in whichever order workers finish them, so the combined output
                doesn't depend on process scheduling. Results that arrive early are held in
                memory until their predecessors arrive.

    Returns:
        T: on_results(...)'s return value.
    """
    if pool_count <= 1:
        raise ValueError(
            f'run_multiproc requires pool_count > 1 (got {pool_count}) - '
            'the caller should process chunks directly instead of spinning up a pool of one.'
        )

    if ordered:
        chunks = enumerate(chunks)
        make_process_chunk = _indexed(make_process_chunk)

    ct = multiprocessing.get_context("fork")
    chunk_queue = ct.Queue(maxsize=pool_count)
    result_queue = ct.Queue(maxsize=pool_count)
    error_queue = ct.Queue()

    producer = ct.Process(target=chunk_producer, args=(error_queue, chunks, chunk_queue))
    workers = [
        ct.Process(target=multiproc_worker, args=(error_queue, make_process_chunk, worker_id, chunk_queue, result_queue))
        for worker_id in range(pool_count)
    ]

    producer.start()
    [worker.start() for worker in workers]

    try:
        results = result_producer(result_queue, error_queue, worker_count=pool_count)
        return on_results(reorder_results(results) if ordered else results)
    except Exception:
        error_queue.put(sys.exc_info())
    finally:
        for process in [producer] + workers:
            if process.is_alive():
                process.terminate()
                process.join()
        chunk_queue.close()
        result_queue.close()
        if not error_queue.empty():
            exc_info = error_queue.get()
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
