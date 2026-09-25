import pytest

from oasislmf.utils.parallel import resolve_partition_count


def test_resolve_partition_count_auto_sizes_from_row_count():
    pool_count, part_count = resolve_partition_count(
        row_count=50000, num_cores=5, num_partitions=-1, min_bloc_size=1000, max_bloc_size=10000)

    assert part_count == 5
    assert pool_count == 5


def test_resolve_partition_count_uses_explicit_num_partitions():
    pool_count, part_count = resolve_partition_count(
        row_count=50000, num_cores=4, num_partitions=8, min_bloc_size=1000, max_bloc_size=10000)

    assert part_count == 8
    assert pool_count == 4


@pytest.mark.parametrize('num_partitions', [10, 1000])
def test_resolve_partition_count_clamps_explicit_num_partitions_to_row_count(num_partitions):
    """An explicit num_partitions larger than row_count would otherwise produce empty
    partitions with nothing to process - part_count should never exceed row_count."""
    pool_count, part_count = resolve_partition_count(
        row_count=3, num_cores=4, num_partitions=num_partitions, min_bloc_size=1000, max_bloc_size=10000)

    assert part_count == 3
    assert pool_count <= part_count


def test_resolve_partition_count_zero_rows_does_not_raise():
    pool_count, part_count = resolve_partition_count(
        row_count=0, num_cores=-1, num_partitions=-1, min_bloc_size=1000, max_bloc_size=10000)

    assert part_count == 0
    assert pool_count == 0
