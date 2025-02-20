import numba as nb
import numpy as np


"""
Custom numba heap implementation for summaries index (based on Python heapq)
A custom heap is required for summaries index as we need to store and sort
based on a tuple of 3 np.int32s. This complex datatype is currently not supported
by numba for standard python heapq.

Stores 5 ints, and sorts lexographically on the first 3 ints, then next 2 ints
First 3 ints are summary_index, period_no, file_idx
Last 2 ints are partial_file_idx, row_number in file[file_idx]
"""


@nb.njit(cache=True, error_model="numpy")
def _resize_heap(heap, current_capacity):
    """Doubles the heap capacity"""
    new_capacity = current_capacity * 2
    new_heap = np.zeros((new_capacity, 5), dtype=heap.dtype)
    for i in range(current_capacity):
        new_heap[i] = heap[i]
    return new_heap


@nb.njit(cache=True, error_model="numpy")
def _lex_compare(a, b):
    """Performs lexicographical comparison for all elements in arrays a and b.
    We compare all elements, and not just the first 3, as we also need to order
    by file_offset when summary_id, period_no, file_idx, which the partial_file_idx
    and row_num values do"""
    for i in range(5):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    return False


@nb.njit(cache=True, error_model="numpy")
def _swap_rows(heap, i, j):
    """Swap rows i and j in heap"""
    for k in range(5):
        temp = heap[i, k]
        heap[i, k] = heap[j, k]
        heap[j, k] = temp


@nb.njit(cache=True, error_model="numpy")
def _sift_down(heap, startpos, pos):
    """Heapq _siftdown"""
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        if _lex_compare(heap[pos], heap[parentpos]):
            _swap_rows(heap, pos, parentpos)
            pos = parentpos
            continue
        break


@nb.njit(cache=True, error_model="numpy")
def _sift_up(heap, pos, endpos):
    """Heapq _siftup"""
    startpos = pos
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not _lex_compare(heap[childpos], heap[rightpos]):
            childpos = rightpos
        # Move the smaller child up.
        if pos != childpos:
            _swap_rows(heap, pos, childpos)
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    _sift_down(heap, startpos, pos)


@nb.njit(cache=True, error_model="numpy")
def heap_push(heap, size, element):
    """Heapq heappush"""
    if size >= len(heap):
        heap = _resize_heap(heap, len(heap))
    heap[size] = element
    _sift_down(heap, 0, size)
    return heap, size + 1


@nb.njit(cache=True, error_model="numpy")
def heap_pop(heap, size):
    """Heapq heappop"""
    if size <= 0:
        raise ValueError("Heap underflow: Cannot pop from an empty heap.")
    lastelt = heap[size - 1].copy()
    if size - 1 > 0:
        returnitem = heap[0].copy()
        heap[0] = lastelt
        _sift_up(heap, 0, size - 1)
        return returnitem, heap, size - 1
    return lastelt, heap, size - 1


@nb.njit(cache=True, error_model="numpy")
def init_heap(num_rows=4):
    """Initialise heap"""
    return np.zeros((num_rows, 5), dtype=np.int32)


@nb.njit(cache=True, error_model="numpy")
def exact_binary_search(arr, value):
    """
    if no match is found return len(arr) instead of the insert index like in normal binary search
    >>> exact_binary_search(np.array([1,3,5,7,9]), 3)
    1

    >>> exact_binary_search(np.array([1,3,5,7,9]), 0)
    5

    >>> exact_binary_search(np.array([1,3,5,7,9]), 4)
    5

    >>> exact_binary_search(np.array([1,3,5,7,9]), 9)
    4
    """
    l = 0
    r = len(arr)
    while l < r:
        mid = (l + r) // 2
        if arr[mid] < value:
            l = mid + 1
        else:
            r = mid
    if l < len(arr) and value != arr[l]:
        return len(arr)
    else:
        return l
