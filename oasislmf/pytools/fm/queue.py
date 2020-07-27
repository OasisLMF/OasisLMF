import queue


class QueueTerminated(Exception):
    pass


class TerminableQueue(queue.Queue):
    check_terminated_time_out = 1

    def __init__(self, maxsize, sentinel):
        self.sentinel = sentinel
        self.terminated = False
        super().__init__(maxsize)

    def get(self, block=True, timeout=None):
        if timeout:
            if self.terminated:
                return self.sentinel
            else:
                return super().get(block, timeout)
        else:
            while not self.terminated:
                try:
                    return super().get(block, self.check_terminated_time_out)
                except queue.Empty:
                    pass
            else:
                return self.sentinel

    def put(self, obj, block=True, timeout=None):
        if timeout:
            if self.terminated:
                raise QueueTerminated('Queue has been terminated')
            else:
                return super().put(obj, block, timeout)

        else:
            while not self.terminated:
                try:
                    return super().put(obj, block,  self.check_terminated_time_out)
                except queue.Full:
                    pass
            else:
                raise QueueTerminated('Queue has been terminated')

try:
    import ray.experimental.queue
except ImportError:
    pass
else:
    class RayTerminableQueue(ray.experimental.queue.Queue):
        check_terminated_time_out = 1

        def __init__(self, maxsize, sentinel):
            self.sentinel = sentinel
            self.terminated = False
            super().__init__(maxsize)

        def get(self, block=True, timeout=None):
            if timeout:
                if self.terminated:
                    return self.sentinel
                else:
                    return super().get(block, timeout)
            else:
                while not self.terminated:
                    try:
                        return super().get(block, self.check_terminated_time_out)
                    except ray.experimental.queue.Empty:
                        pass
                else:
                    return self.sentinel

        def put(self, obj, block=True, timeout=None):
            if timeout:
                if self.terminated:
                    raise QueueTerminated('Queue has been terminated')
                else:
                    return super().put(obj, block, timeout)

            else:
                while not self.terminated:
                    try:
                        return super().put(obj, block, self.check_terminated_time_out)
                    except ray.experimental.queue.Full:
                        pass
                else:
                    raise QueueTerminated('Queue has been terminated')
