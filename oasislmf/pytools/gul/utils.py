"""This file contains general-purpose utilities used in gulpy."""
from numba import njit


@njit(cache=True, fastmath=True)
def binary_search(value, array, n):
    """Assuming `array` is a sorted array (increasing order), find the bin
    where where `array` gets larger than `value`.

    Args:
        value (float,int): the value to be searched
        array (float, int): the array where `value` is to be searched.
        n (int): number of elements of `array` where `value` is to be searched, starting
          from the beginning of `array`.

    Returns:
        int: the first index of `array` where `array` is larger than `value`.
          Returns -1 if this condition never occurs.
    """
    lo = 0
    hi = n
    while lo < hi:
        mid = (lo + hi) >> 1  # divide by two

        if array[mid] < value:
            lo = mid + 1
        else:
            hi = mid

    return lo
