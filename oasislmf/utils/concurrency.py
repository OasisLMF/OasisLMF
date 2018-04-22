# -*- coding: utf-8 -*-

import queue
import signal
import sys
import threading

__all__ = [
    'aggregate_tasks',
    'SignalHandler',
    'slice_task'
]


class SignalHandler(object):
    def __init__(self, stopper, workers):
        self.stopper = stopper
        self.workers = workers

    def __call__(self, signum, frame):
        self.stopper.set()

        for worker in self.workers:
            worker.join()

        sys.exit(0)


def aggregate_tasks(tasks):
    
    task_q = queue.Queue()

    for task in tasks:
        task_q.put(task)

    def run(task_q, result_q, stopper):
        while not stopper.is_set():
            try:
                result_label, func, args = task_q.get_nowait()
            except queue.Empty:
                break
            else:
                result = func(*args)
                result_q.put((result_label, result,))
                task_q.task_done()

    result_q = queue.Queue()

    stopper = threading.Event()

    workers = [threading.Thread(target=run, args=(task_q, result_q, stopper,)) for t in tasks]

    handler = SignalHandler(stopper, workers)
    signal.signal(signal.SIGINT, handler)

    for worker in workers:
        worker.start()

    task_q.join()

    while not result_q.empty():
        result_label, result = result_q.get_nowait()
        yield result_label, result


def slice_task():
    pass
