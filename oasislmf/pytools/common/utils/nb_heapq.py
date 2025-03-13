import numba as nb
import numpy as np


"""
Custom numba heap implementation for for dtype np.int32

Stores num_compare + 2 ints, and sorts lexographically on all ints.
Last 2 ints are file_idx and row_number in file[file_idx]
"""


@nb.njit(cache=True, error_model="numpy")
def _resize_heap(heap, current_capacity):
    """Doubles the heap capacity"""
    new_capacity = current_capacity * 2
    new_heap = np.zeros((new_capacity, heap.shape[1]), dtype=heap.dtype)
    for i in range(current_capacity):
        new_heap[i] = heap[i]
    return new_heap


@nb.njit(cache=True, error_model="numpy")
def _lex_compare(a, b, row_size):
    """Lexicographically compare two rows from heap"""
    for i in range(row_size):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    return False


@nb.njit(cache=True, error_model="numpy")
def _swap_rows(heap, i, j):
    """Swap rows i and j in heap"""
    for k in range(heap.shape[1]):
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
        if _lex_compare(heap[pos], heap[parentpos], heap.shape[1]):
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
        if rightpos < endpos and not _lex_compare(heap[childpos], heap[rightpos], heap.shape[1]):
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
def init_heap(num_rows=4, num_compare=1):
    """Initialise heap with (num_compare + 2) * num_rows elements, where
    num_compare is the number of elements to order by in lexicographical
    order, and the remaining two elements are the file and row idxs

    Currently limited to dtype np.int32 as it is tricky to use custom dtypes
    """
    return np.zeros((num_rows, num_compare + 2), dtype=np.int32)
