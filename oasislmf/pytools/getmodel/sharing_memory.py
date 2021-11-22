import os
from enum import Enum
from multiprocessing.shared_memory import SharedMemory
from typing import Optional
from typing import Tuple

import numpy as np
import pyarrow.parquet as pq


class DataEnum(Enum):
    VULNERABILITY = "vulnerability"


class SharedMemoryManager:

    def __init__(self, name: DataEnum, static_path: Optional[str] = None, vuln_dict: Optional[dict] = None) -> None:
        self.name: DataEnum = name
        self.static_path: Optional[str] = static_path
        self.vuln_dict: Optional[dict] = vuln_dict
        self.loaded_from_file = False

    def _read(self) -> Tuple[np.array, np.array]:
        parquet_handle = pq.ParquetDataset(os.path.join(self.static_path, "vulnerability_dataset"), use_legacy_dataset=False,
                                           filters=[("vulnerability_id", "in", list(self.vuln_dict))])
        vuln_table = parquet_handle.read()
        vuln_meta = vuln_table.schema.metadata
        num_damage_bins = int(vuln_meta[b"num_damage_bins"].decode("utf-8"))
        number_of_intensity_bins = int(vuln_meta[b"num_intensity_bins"].decode("utf-8"))
        vuln_array = np.vstack(vuln_table['vuln_array'].to_numpy()).reshape(vuln_table['vuln_array'].length(),
                                                                            num_damage_bins,
                                                                            number_of_intensity_bins)
        vulns_id = vuln_table['vulnerability_id'].to_numpy()
        return vuln_array, vulns_id

    def get_pointer_name(self, array_type: str) -> str:
        return self.name.value + "_" + array_type

    @property
    def pointers(self) -> Tuple[SharedMemory, SharedMemory]:
        try:
            vuln_array_pointer = SharedMemory(self.get_pointer_name(array_type="vuln_array"), create=False)
            vulns_id_pointer = SharedMemory(self.get_pointer_name(array_type="vulns_id"), create=False)
            return vuln_array_pointer, vulns_id_pointer
        except FileNotFoundError:
            vuln_array, vulns_id = self._read()

            vuln_array_pointer = SharedMemory(self.get_pointer_name(array_type="vuln_array"), create=True,
                                              size=vuln_array.size * vuln_array.itemsize)
            vulns_id_pointer = SharedMemory(self.get_pointer_name(array_type="vulns_id"), create=True,
                                            size=vulns_id.size * vulns_id.itemsize)

            vuln_array_shared = np.ndarray(vuln_array.shape, dtype=vuln_array.dtype, buffer=vuln_array_pointer.buf)
            vulns_id_shared = np.ndarray(vulns_id.shape, dtype=vulns_id.dtype, buffer=vulns_id_pointer.buf)

            vuln_array_shared[:] = vuln_array[:]
            vulns_id_shared[:] = vulns_id[:]
            self.loaded_from_file = True

            return vuln_array_pointer, vulns_id_pointer
