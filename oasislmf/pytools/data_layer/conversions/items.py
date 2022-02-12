import hashlib

import numpy as np

from oasislmf.pytools.getmodel.common import Item


def process_hashed_item(item_data: "np.array[Item]", total_len: int) -> "np.array[Item]":
    """
    Hash the group IDs of the items data.

    Args:
        item_data: (np.array[Item]) data to be processed from
        total_len: (int) the length of the item data being processed

    Returns: (np.array[Item]) Item data that has been hashed
    """
    buffer = np.zeros(total_len, dtype=Item)
    for x in range(total_len):
        group_id = item_data[x]["group_id"]
        np.random.seed(group_id)
        buffer[x]["id"] = item_data[x]["id"]
        buffer[x]["coverage_id"] = item_data[x]["coverage_id"]
        buffer[x]["areaperil_id"] = item_data[x]["areaperil_id"]
        buffer[x]["vulnerability_id"] = item_data[x]["vulnerability_id"]
        buffer[x]["group_id"] = int(hashlib.sha1(str(np.random.random()).encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    return buffer
