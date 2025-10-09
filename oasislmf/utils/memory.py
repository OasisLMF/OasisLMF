import psutil
import os


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return psutil._common.bytes2human(mem_info.rss)
