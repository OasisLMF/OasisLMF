import hashlib
import json

import numpy as np
import pandas as pd

from oasislmf.pytools.getmodel.common import Item


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class IdMap(dict, metaclass=Singleton):

    def __init__(self) -> None:
        super().__init__({})


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


def convert_item_file_ids_to_hash(input_directory: str) -> None:
    """
    Hashes the group ID in the items.bin file writing the result to that file.

    Args:
        input_directory: (str) the path to the input directory

    Returns: None
    """
    item_data = np.memmap(f"{input_directory}/items.bin", dtype=Item, mode='readwrite')

    for x in range(item_data.shape[0]):
        hashed_id = generate_group_id_hash(group_id=item_data[x]["group_id"])
        item_data[x]["group_id"] = hashed_id

    _write_hashed_id_map(input_directory=input_directory)


def convert_item_csv_to_hash(input_directory: str) -> None:
    """
    Rewrites the item CSV file to the same CSV file with hashed group IDs.

    Args:
        input_directory: (str) the directory of where the file is to be altered

    Returns: None
    """
    file_path: str = f"{input_directory}/items.csv"
    df = pd.read_csv(file_path)
    df['group_id'] = df['group_id'].apply(generate_group_id_hash)

    _write_hashed_id_map(input_directory=input_directory)
    df.to_csv(file_path, index=False)


def generate_group_id_hash(group_id: int) -> int:
    """
    Generates a hashed version of the group_id.

    Args:
        group_id: (int) the group ID to be hashed

    Returns: (int) the hashed group ID
    """
    id_map = IdMap()
    np.random.seed(group_id)
    hashed_id: int = id_map.get(
        group_id, int(hashlib.sha1(str(np.random.random()).encode("utf-8")).hexdigest(), 16) % (10 ** 8)
    )
    id_map[group_id] = hashed_id
    return hashed_id


def _write_hashed_id_map(input_directory: str) -> None:
    """
    Writes the hashed ID maps to the JSON file.

    Args:
        input_directory: (str) the directory of where the map is going to be written

    Returns: None
    """
    with open(f"{input_directory}/group_id_map.json", "w") as file:
        id_map = IdMap()
        json_data = json.dumps(id_map)
        file.write(json_data)
