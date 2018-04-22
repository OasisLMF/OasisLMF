# -*- coding: utf-8 -*-

import queue
import signal
import sys
import threading

__all__ = [
    'aggregate',
    'SignalHandler',
    'Task',
    'ThreadedTask',
    'slice'
]


class SignalHandler(object):
    def __init__(self, stopper, threaded_tasks):
        self.stopper = stopper
        self.threaded_tasks = threaded_tasks

    def __call__(self, signum, frame):
        self.stopper.set()

        for task in self.threaded_tasks:
            task.join()

        sys.exit(0)


class Task(object):
    def __init__(self, func, args=(), key=None):
        self._func = func
        self._args = args
        self._key = key if key else func.__name__
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


class ThreadedTask(Task):

    def __init__(self, func, thread=None, args=(), key=None):
        super(self.__class__, self).__init__(func, args=args, key=key)
        self._thread = thread

    @property
    def thread(self):
        """
        Task function/method thread property.

            :getter: Gets the thread object
            :setter: Sets the thread object
        """
        return self._thread

    @thread.setter
    def thread(self, t):
        self._thread = t

    def start(self):
        self._thread.start()

    def join(self):
        self._thread.join()

    @property
    def is_done(self):
        return self._thread.isAlive()


def aggregate(threaded_tasks):
    """
    Given 
    """
    
    task_q = queue.Queue()

    for task in threaded_tasks:
        task_q.put(task)

    def run(task_q, result_q, stopper):
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

    for task in threaded_tasks:
        task.thread = threading.Thread(target=run, args=(task_q, result_q, stopper,))

    handler = SignalHandler(stopper, threaded_tasks)
    signal.signal(signal.SIGINT, handler)

    for task in threaded_tasks:
        task.start()

    task_q.join()

    while not result_q.empty():
        key, result = result_q.get_nowait()
        yield key, result


def slice(task, slices=2):
    pass
