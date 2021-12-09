import numba as nb
import numpy as np
import math

hash_entry_dtype =  nb.from_dtype(np.dtype([('distance', np.int64),
                                            ('key', np.uint64),
                                            ('value', np.int64),
                                            ]))

load_factor = 0.8
size_up_attempt = 2

mask = np.int64(2 ** np.arange(64) - 1)

p = np.uint64(0x5555555555555555)  # pattern of alternating 0 and 1
c = np.uint64(17316035218449499591)  # random uneven integer constant;
shift = np.uint64(32)


@nb.njit([nb.uint64(nb.uint64, nb.uint64),
          ], cache=True)
def xorshift(n, i):
    return n^(n>>i)


@nb.njit([nb.int64(nb.uint64),
          ], cache=True)
def custom_hash(n):
    return np.int64(c*xorshift(p*xorshift(n,shift),shift))


@nb.njit(cache=True)
def index_hash_table(items, key_name, value_name):
    p2_size = math.ceil(np.log2(items.shape[0] / load_factor))
    size_up = 0

    while size_up < size_up_attempt:
        hash_table = np.empty(2 ** p2_size + p2_size, dtype=hash_entry_dtype)
        hash_table['distance'].fill(-1)
        for i in range(items.shape[0]):
            key = items[i][key_name]
            value = items[i][value_name]
            distance = 0
            index = custom_hash(key) & mask[p2_size]
            while distance < p2_size:
                it = hash_table[index]
                if it['distance'] == -1: # empty slot
                    it['distance'] = distance
                    it['key'] = key
                    it['value'] = value
                    break
                elif it['distance'] < distance: # we take from the rich
                    it['key'], key = key, it['key']
                    it['value'], value = value, it['value']
                    it['distance'], distance = distance, it['distance'] + 1
                    index += 1

                elif it['key'] == key:
                    break
                else:
                    distance += 1
                    index += 1

            else: # distance is more than p2_size
                size_up += 1
                p2_size += 1
                break
        else:
            break
    else:
        raise Exception("too many collisions found need a better hash function")

    return hash_table, p2_size


@nb.njit(cache=True)
def hash_table_get(hash_table, size, key):
    index = custom_hash(key) & mask[size]
    for distance in range(size):
        it = hash_table[index + distance]
        if it['distance'] < distance:
            return 0, np.int64(0)
        elif it['key'] == key:
            return 1, it['value']
    else:
        raise Exception('distance exceeded max, something is very wrong')



if __name__ == "__main__":
    nb_key_value = 100000000
    array = np.arange(nb_key_value, dtype = np.uint64).reshape(nb_key_value // 2, 2)
    _table, _size = index_hash_table(array, 0, 1)
    print(f"{_table.shape} {_size} {mask[_size]} {mask[_size]:b}")
    print(np.max(_table['distance']), _table.nbytes / 1024 / 1024)
    print(hash_table_get(_table, _size, 10))
    print(hash_table_get(_table, _size, 11))

