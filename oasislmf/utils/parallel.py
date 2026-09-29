__all__ = [
    'resolve_partition_count',
]

import math
import os

try:
    import billiard as multiprocessing
except ImportError:
    import multiprocessing

MIN_BLOC_SIZE = int(os.environ.get('OASIS_MIN_BLOC_SIZE', 1000))
MAX_BLOC_SIZE = int(os.environ.get('OASIS_MAX_BLOC_SIZE', 10000))


def resolve_partition_count(row_count, num_cores, num_partitions, min_bloc_size=MIN_BLOC_SIZE, max_bloc_size=MAX_BLOC_SIZE):
    """Resolve the (pool_count, part_count) to use for a multiprocessing split of `row_count` items.

    If num_partitions is explicitly set (> 0) it is used directly, otherwise the partition
    count is derived from row_count so that each partition holds between min_bloc_size and
    max_bloc_size items (bounded by the number of available cores). Either way, part_count is
    then capped at row_count - an explicit num_partitions larger than row_count would otherwise
    produce empty partitions with nothing to process.
    """
    pool_count = num_cores if num_cores > 0 else multiprocessing.cpu_count()
    if num_partitions > 0:
        part_count = num_partitions
    else:
        bloc_size = min(max(math.ceil(row_count / pool_count), min_bloc_size), max_bloc_size)
        part_count = math.ceil(row_count / bloc_size)
    if row_count > 0:
        part_count = min(part_count, row_count)
    pool_count = min(pool_count, part_count)
    return pool_count, part_count
