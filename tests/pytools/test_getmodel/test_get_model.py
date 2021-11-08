from unittest import main, TestCase
from mock import patch

from oasislmf.pytools.getmodel.get_model import get_items
from oasislmf.pytools.getmodel.vulnerability import vulnerability_to_parquet, Vulnerability

import numpy as np
import numba as nb
import pyarrow as pa
import pyarrow.parquet as pq


def open_parquet(path: str, input_struct: Vulnerability = Vulnerability):
    pass


class GetModelTests(TestCase):

    def test_get_items(self):

        outcome = get_items(input_path="./", file_type="bin")
        print(len(outcome))
        print(list(outcome[0].keys()))

    def test_filtering(self):

        # use legacy dataset set to True enables us to pass filters to all columns
        table2 = pq.read_table("./vulnerability_dataset/part_0.parquet", memory_map=True, use_legacy_dataset=True,
                               )
        meta = table2.schema.metadata
        number_of_vulnerability_ids = int(meta[b"num_vulnerability_id"].decode("utf-8"))
        number_of_intensity_bins = int(meta[b"num_intensity_bins"].decode("utf-8"))
        number_of_damage_bins = int(meta[b"num_damage_bins"].decode("utf-8"))
        print(f"here is the length of the data {len(table2)}")
        print(table2)


if __name__ == "__main__":
    main()
