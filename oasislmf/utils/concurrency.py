# -*- coding: utf-8 -*-

import multiprocessing

try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty

import sys
import types

from multiprocessing import Pool

from signal import (
    signal,
    SIGINT,
)
from threading import (
    Event,
    Thread,
)

from .exceptions import OasisException

__all__ = [
    'multiprocess',
    'multithread',
    'SignalHandler',
    'Task'
]


class SignalHandler(object):
    def __init__(self, stopper, threads):
        self.stopper = stopper
        self.threads = threads

    def __call__(self, signum, frame):
        self.stopper.set()

        for task in self.threads:
            task.join()

        sys.exit(0)


class Task(object):
    def __init__(self, func, args=(), key=None):
        self._func = func
        self._args = args
        self._key = key if key is not None else func.__name__
        self._result = None
        self._is_done = False

    @property
    def func(self):
        """
        Task function/method property - getter only.

            :getter: Gets the task function/method object
        """
        return self._func

    @property
    def args(self):
        """
        Task function/method arguments property - getter only.

            :getter: Gets the task function/method arguments
        """
        return self._args

    @property
    def key(self):
        """
        Task function/method key - getter only.

            :getter: Gets the task function/method key
        """
        return self._key

    @property
    def result(self):
        """
        Task function/method result property.

            :getter: Gets the task function/method result (produced by calling
                     the function on the defined arguments)
            :setter: Sets the task function/method result
        """
        return self._result

    @result.setter
    def result(self, r):
        self._result = r
        self._is_done = True

    @property
    def is_done(self):
        """
        Task function/method status property - getter only.

            :getter: Gets the task function/method status
        """
        return self._is_done


def multithread(tasks, pool_size=10):
    """
    Executes several tasks concurrently via ``threading`` threads, puts the
    results into a queue, and generates these back to the caller.
    """
    
    task_q = Queue()

    num_tasks = 0

    for task in tasks:
        task_q.put(task)
        num_tasks += 1

    def run(i, task_q, result_q, stopper):
        while not stopper.is_set():
            try:
                task = task_q.get_nowait()
            except Empty:
                break
            else:
                task.result = task.func(*task.args) if task.args else task.func()
                if type(task.result) in (types.GeneratorType, list, tuple, set):
                    for r in task.result:
                        result_q.put((task.key, r,))
                else:
                    result_q.put((task.key, task.result,))
                task_q.task_done()

    result_q = Queue()

    stopper = Event()

    threads = tuple(Thread(target=run, args=(i, task_q, result_q, stopper,)) for i in range(pool_size))

    handler = SignalHandler(stopper, threads)
    signal(SIGINT, handler)

    for thread in threads:
        thread.start()

    task_q.join()

    while not result_q.empty():
        key, result = result_q.get_nowait()
        yield key, result


def multiprocess(tasks, pool_size=10):
    """
    Executes several tasks concurrently via Python ``multiprocessing``
    processes, puts the results into a queue, and generates these back to the
    caller.
    """

    pool = Pool(pool_size)

    result_q = Queue()

    def build_results(result):
        if type(result) in (types.GeneratorType, list, tuple, set):
            for r in result:
                result_q.put(r)
        else:
            result_q.put(result)

    for task in tasks:
        pool.apply_async(task.func, args=task.args, callback=build_results)

    pool.close()
    pool.join()

    while not result_q.empty():
        result = result_q.get_nowait()
        yield result
 