"""Tests of the JIT warmup's process pool that do not need Numba to compile anything."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from numba.core.caching import IndexDataCacheFile

from oasislmf import warmup

N_WORKERS = 6
N_SAVES = 30


def _save_entries(cache_dir, worker):
    cache_file = IndexDataCacheFile(cache_dir, 'func', 'stamp')
    for i in range(N_SAVES):
        cache_file.save((worker, i), (worker, i))


def test_concurrent_cache_saves_in_warmup_workers_keep_each_variant_with_its_own_data(tmp_path):
    """Workers adding variants of one function at once must neither drop one nor map it to another's data.

    Without the warmup's lock, Numba's save gives two processes the same data file name, so most entries
    are lost and some signatures load code compiled for another one.
    """
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=warmup._init_worker,
                             initargs=(multiprocessing.Lock(),)) as pool:
        for future in [pool.submit(_save_entries, str(tmp_path), worker) for worker in range(N_WORKERS)]:
            future.result()

    cache_file = IndexDataCacheFile(str(tmp_path), 'func', 'stamp')
    index = cache_file._load_index()
    assert len(index) == N_WORKERS * N_SAVES
    assert all(cache_file.load(key) == key for key in index)


def _save_entries_file_locked(cache_dir, worker):
    warmup.lock_cache_saves()
    _save_entries(cache_dir, worker)


def test_concurrent_cache_saves_with_file_locks_keep_each_variant_with_its_own_data(tmp_path):
    """Processes sharing no lock object (as the orchestrator's pool workers) keep every variant with lock_cache_saves."""
    context = multiprocessing.get_context('spawn')
    processes = [context.Process(target=_save_entries_file_locked, args=(str(tmp_path), worker)) for worker in range(N_WORKERS)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0

    cache_file = IndexDataCacheFile(str(tmp_path), 'func', 'stamp')
    index = cache_file._load_index()
    assert len(index) == N_WORKERS * N_SAVES
    assert all(cache_file.load(key) == key for key in index)
