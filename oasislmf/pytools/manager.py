import math
import asyncio
import os

import ray.util.queue as queue
import ray
import numpy as np


from getmodel.manager_ray import ModelPy
from gul.manager_ray import GulPy


ray.init()



@ray.remote
class Eve:
    def __init__(self, input_path, eve_queue, chunk_size = 100):
        self.events = np.memmap(os.path.join(input_path, "events.bin"), dtype=np.int32, mode='r')
        self.nb_events = self.events.shape[0]
        self.bloc_i = 0
        self.chunk_size = chunk_size
        self.nb_chunk = math.ceil(self.nb_events / self.chunk_size)
        self.eve_queue = eve_queue

    def get_nb_events(self):
        return self.nb_events

    def get_event(self):
        while self.bloc_i < self.nb_chunk:
            self.eve_queue.put(self.events[self.bloc_i * self.chunk_size: (self.bloc_i+1) * self.chunk_size])
            self.bloc_i += 1
        self.eve_queue.put(None)


@ray.remote
def consume_cdf(_queue):
    while True:
        try:
            event = _queue.get(timeout=30)
        except ray.util.queue.Empty:
            pass
        else:
            if event is None:
                break
    print('all consumed')



sample_size = 1000
loss_threshold = 1
alloc_rule = 1
debug = False
random_generator = 1
ignore_files = []


nb_model = 5
nb_gulpy = 8

run_dir = '.'
static_path = os.path.join(run_dir, 'static')
input_path = os.path.join(run_dir, 'input')

eve_queue = queue.Queue(maxsize=10)
cdf_queue = queue.Queue(maxsize=10)
gul_queue = queue.Queue(maxsize=10)

eve = Eve.remote(input_path, eve_queue)
modelpys = [ModelPy.remote(run_dir, eve_queue, cdf_queue, ignore_files, False) for _ in range(nb_model)]
gulpys = [GulPy.remote(cdf_queue, gul_queue, run_dir, ignore_files, sample_size, loss_threshold, alloc_rule, debug, random_generator) for _ in range(nb_gulpy)]
consumer = consume_cdf.remote(gul_queue)

eve_end =eve.get_event.remote()
modelpy_status = [modelpy.run.remote() for modelpy in modelpys]
gulpy_status = [gulpy.run.remote() for gulpy in gulpys]

ray.get(eve_end)
eve_queue.put(None)
ray.get(modelpy_status)
cdf_queue.put(None)
ray.get(gulpy_status)
gul_queue.put(None)
ray.get(consumer)
