"""
This file contains general-purpose utilities used in gulpy.

"""
from numba import njit
from numba.typed import List


@njit(cache=True, fastmath=True)
def binary_search(value, array, n):
    """
    Assuming `array` is a sorted array (increasing order), find the bin
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


@njit(cache=True, fastmath=True)
def append_to_dict_value(d, key, value, value_type):
    """Append a value to the list populating a dictionary value.
    If the key is not present in the dictionary, populate the entry with a list with
    just the passed value.
    The dictionary `d` is modified *in-place*, thus it is not returned by the function.
    If `d` is a dictionary and `d[key]` is a list, this function appends
    `value` to the list. Example: if d = {0: [1, 2], 1: [3]}, then:

       append_to_dict_entry(d, 0, 3, int)

    will modify `d` to:

       d = {0: [1, 2, 3], 1: [3]}

    Designed to be used with numba.typed.Dict and numba.typed.List.

    Args:
        d (numba.typed.Dict[*,numba.typedList[value_type]]): dictionary to be modified,
          by appending `value` to the list in d[key].
        key (same as d.key_type): key of the element to modify.
        value (value_type): value to be appended to the list in d[key].
        value_type (built-in Python or numba type): value data type.
    """
    def_lst = List.empty_list(value_type)
    d.setdefault(key, def_lst)
    lst = d[key]
    lst.append(value)
    d[key] = lst
