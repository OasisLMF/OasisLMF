import numba as nb


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
