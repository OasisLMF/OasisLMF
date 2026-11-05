"""Contract tests for the rare out-of-order path of fmpy's stream reader.

read_buffer appends in-order (sidx, loss) pairs itself; a pair whose sidx is below the node's
last stored sidx goes through insert_loss_out_of_order, which must keep each node's sidx sorted
and reject a duplicate sidx as stream corruption.
"""
import numpy as np
import pytest

from oasislmf.pytools.fm.stream_sparse import insert_loss_out_of_order


def _node_with(sidx_values, loss_values, capacity=8):
    """Arrays for one populated node (compute_i=1) holding the given sorted pairs."""
    sidx_indptr = np.array([0, len(sidx_values)], dtype=np.int64)
    sidx_val = np.zeros(capacity, dtype=np.int32)
    loss_val = np.zeros(capacity, dtype=np.float64)
    sidx_val[:len(sidx_values)] = sidx_values
    loss_val[:len(loss_values)] = loss_values
    return sidx_indptr, sidx_val, loss_val


@pytest.mark.parametrize("sidx, expected_sidx, expected_loss", [
    (1, [1, 2, 5, 9], [10., 20., 50., 90.]),  # before the first
    (4, [2, 4, 5, 9], [20., 40., 50., 90.]),  # in the middle
    (7, [2, 5, 7, 9], [20., 50., 70., 90.]),  # just before the last
])
def test_out_of_order_sidx_is_inserted_in_sorted_position(sidx, expected_sidx, expected_loss):
    sidx_indptr, sidx_val, loss_val = _node_with([2, 5, 9], [20., 50., 90.])

    insert_loss_out_of_order(sidx, sidx * 10., 1, sidx_indptr, sidx_val, loss_val)

    assert sidx_indptr[1] == 4
    np.testing.assert_array_equal(sidx_val[:4], expected_sidx)
    np.testing.assert_array_equal(loss_val[:4], expected_loss)


def test_insert_only_shifts_the_current_node():
    # node 1 holds sidx [3, 8] (indices 0-1), node 2 holds [1, 6] (indices 2-3)
    sidx_indptr = np.array([0, 2, 4], dtype=np.int64)
    sidx_val = np.array([3, 8, 1, 6, 0, 0], dtype=np.int32)
    loss_val = np.array([30., 80., 10., 60., 0., 0.])

    insert_loss_out_of_order(4, 40., 2, sidx_indptr, sidx_val, loss_val)

    np.testing.assert_array_equal(sidx_indptr, [0, 2, 5])
    np.testing.assert_array_equal(sidx_val[:5], [3, 8, 1, 4, 6])
    np.testing.assert_array_equal(loss_val[:5], [30., 80., 10., 40., 60.])


def test_duplicate_sidx_is_rejected():
    sidx_indptr, sidx_val, loss_val = _node_with([2, 5, 9], [20., 50., 90.])

    with pytest.raises(ValueError, match="duplicated sidx"):
        insert_loss_out_of_order(5, 1., 1, sidx_indptr, sidx_val, loss_val)
