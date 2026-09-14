__all__ = [
    'run_pre_analysis_multiproc',
]

import copy
import sys

import numpy as np
import pandas as pd

try:
    import billiard as multiprocessing
except ImportError:
    import multiprocessing

from queue import Empty, Full

# add pickling support for traceback object
import tblib.pickling_support

tblib.pickling_support.install()


def with_error_queue(fct):
    def wrapped_fct(error_queue, *args, **kwargs):
        try:
            return fct(error_queue, *args, **kwargs)
        except Exception:
            error_queue.put(sys.exc_info())

    return wrapped_fct


@with_error_queue
def exposure_producer(error_queue, loc_df, acc_df, part_count, group_cols, exposure_queue):
    """Split loc_df/acc_df into part_count chunks and put them on exposure_queue.

    If group_cols is given, chunks are formed from unique combinations of those columns
    (e.g. ['PortNumber', 'AccNumber']) so that every location/account row belonging to the
    same group ends up in the same chunk. Otherwise (no account grouping possible), chunks
    are formed from unique 'loc_id' values, matching oasislmf.lookup.factory's location split.
    """
    if group_cols:
        group_keys = loc_df[group_cols].drop_duplicates().reset_index(drop=True)
        key_parts = np.array_split(group_keys, part_count)
        loc_parts = (loc_df.merge(key_part, on=group_cols) for key_part in key_parts)
        acc_parts = (
            (acc_df.merge(key_part, on=group_cols) if acc_df is not None else None)
            for key_part in key_parts
        )
    else:
        loc_id_parts = np.array_split(np.unique(loc_df['loc_id']), part_count)
        loc_parts = (loc_df[loc_df['loc_id'].isin(loc_id_parts[i])] for i in range(part_count))
        acc_parts = (None for _ in range(part_count))

    parts_iter = zip(loc_parts, acc_parts)
    part = True
    while part is not None:
        part = next(parts_iter, None)
        loc_part, acc_part = part if part is not None else (None, None)
        while error_queue.empty():
            try:
                exposure_queue.put((loc_part, acc_part), timeout=5)
                break
            except Full:
                pass
        else:
            return


@with_error_queue
def pre_analysis_multiproc_worker(error_queue, exposure_data, hook_cls, hook_kwargs, exposure_queue, result_queue):
    has_account = exposure_data.account is not None
    while True:
        while error_queue.empty():
            try:
                loc_part, acc_part = exposure_queue.get(timeout=5)
                break
            except Empty:
                pass
        else:
            return

        if loc_part is None:
            exposure_queue.put((None, None))
            result_queue.put(None)
            break

        chunk_exposure_data = copy.copy(exposure_data)
        chunk_exposure_data.location = copy.copy(exposure_data.location)
        chunk_exposure_data.location.dataframe = loc_part
        if has_account:
            chunk_exposure_data.account = copy.copy(exposure_data.account)
            chunk_exposure_data.account.dataframe = acc_part

        chunk_kwargs = dict(hook_kwargs)
        chunk_kwargs['exposure_data'] = chunk_exposure_data
        class_return = hook_cls(**chunk_kwargs).run()

        result = (
            chunk_exposure_data.location.dataframe,
            chunk_exposure_data.account.dataframe if has_account else None,
            class_return,
        )
        while error_queue.empty():
            try:
                result_queue.put(result, timeout=5)
                break
            except Full:
                pass
        else:
            return


def result_producer(result_queue, error_queue, worker_count):
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
            yield res


def run_pre_analysis_multiproc(exposure_data, hook_cls, hook_kwargs, pool_count, part_count, group_cols):
    """Run hook_cls(**hook_kwargs, exposure_data=<chunk>).run() across pool_count worker
    processes, splitting exposure_data's location/account dataframes into part_count chunks.

    Returns (location_df, account_df, [chunk_return, ...]) with the per-chunk results merged
    back together; account_df is None if exposure_data has no account data.
    """
    loc_df = exposure_data.location.dataframe
    acc_df = exposure_data.account.dataframe if exposure_data.account is not None else None

    ct = multiprocessing.get_context("fork")
    exposure_queue = ct.Queue(maxsize=pool_count)
    result_queue = ct.Queue(maxsize=pool_count)
    error_queue = ct.Queue()

    producer = ct.Process(
        target=exposure_producer,
        args=(error_queue, loc_df, acc_df, part_count, group_cols, exposure_queue),
    )
    workers = [
        ct.Process(
            target=pre_analysis_multiproc_worker,
            args=(error_queue, exposure_data, hook_cls, hook_kwargs, exposure_queue, result_queue),
        )
        for _ in range(pool_count)
    ]

    producer.start()
    [worker.start() for worker in workers]

    try:
        loc_results, acc_results, class_returns = [], [], []
        for loc_part, acc_part, class_return in result_producer(result_queue, error_queue, worker_count=pool_count):
            loc_results.append(loc_part)
            if acc_part is not None:
                acc_results.append(acc_part)
            class_returns.append(class_return)

        return (
            pd.concat(loc_results, ignore_index=True),
            pd.concat(acc_results, ignore_index=True) if acc_results else None,
            class_returns,
        )
    except Exception:
        error_queue.put(sys.exc_info())
    finally:
        for process in [producer] + workers:
            if process.is_alive():
                process.terminate()
                process.join()
        exposure_queue.close()
        result_queue.close()
        if not error_queue.empty():
            exc_info = error_queue.get()
            raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
