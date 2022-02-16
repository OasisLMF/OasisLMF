import hashlib
import json
from typing import Dict

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
        buffer[x]["id"] = item_data[x]["id"]
        buffer[x]["coverage_id"] = item_data[x]["coverage_id"]
        buffer[x]["areaperil_id"] = item_data[x]["areaperil_id"]
        buffer[x]["vulnerability_id"] = item_data[x]["vulnerability_id"]
        buffer[x]["group_id"] = generate_group_id_hash(group_id=item_data[x]["group_id"])
    return buffer


def convert_item_file_ids_to_hash(static_path: str) -> None:
    """
    Hashes the group ID in the items.bin file writing the result to that file.

    Args:
        static_path: (str) the path to the static directory

    Returns: None
    """
    item_data = np.memmap(f"{static_path}/items.bin", dtype=Item, mode='readwrite')
    group_id_map: Dict[int, int] = dict()

    for x in range(item_data.shape[0]):
        group_id = item_data[x]["group_id"]
        hashed_id = generate_group_id_hash(group_id=item_data[x]["group_id"])
        group_id_map[group_id] = hashed_id
        item_data[x]["group_id"] = hashed_id

    with open(f"{static_path}/group_id_map.json", "w") as file:
        json_data = json.dumps(group_id_map)
        file.write(json_data)


def generate_group_id_hash(group_id: int) -> int:
    """
    Generates a hashed version of the group_id

    Args:
        group_id: (int) the group ID to be hashed

    Returns: (int) the hashed group ID
    """
    np.random.seed(group_id)
    return int(hashlib.sha1(str(np.random.random()).encode("utf-8")).hexdigest(), 16) % (10 ** 8)
