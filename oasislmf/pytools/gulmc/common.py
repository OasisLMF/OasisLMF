import numba as nb
import numpy as np

AggregateVulnerability = nb.from_dtype(np.dtype([('aggregate_vulnerability_id', np.int32),
                                                 ('vulnerability_id', np.int32),]))
