# -*- coding: utf-8 -*-

import queue
import signal
import sys
import threading

from .exceptions import OasisException

__all__ = [
    'aggregate',
    'SignalHandler',
    'Task',
    'slice'
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


def aggregate(tasks, pool_size=None):
    """
    Executes several tasks concurrently, puts the results into a queue,
    and generates these back to the caller.
    """
    
    task_q = queue.Queue()

    num_tasks = 0

    for task in tasks:
        task_q.put(task)
        num_tasks += 1

    def run(i, task_q, result_q, stopper):
        while not stopper.is_set():
            try:
                task = task_q.get_nowait()
            except queue.Empty:
                break
            else:
                task.result = task.func(*task.args) if task.args else task.func()
                result_q.put((task.key, task.result,))
                task_q.task_done()

    result_q = queue.Queue()

    stopper = threading.Event()

    pool_size = num_tasks if pool_size is None else pool_size

    threads = tuple(threading.Thread(target=run, args=(i, task_q, result_q, stopper,)) for i in range(pool_size))

    handler = SignalHandler(stopper, threads)
    signal.signal(signal.SIGINT, handler)

    for thread in threads:
        thread.start()

    task_q.join()

    while not result_q.empty():
        key, result = result_q.get_nowait()
        yield key, result


def slice(task, slices=2):
    pass
