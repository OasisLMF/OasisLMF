import numba as nb
import numpy as np

AggregateVulnerability = nb.from_dtype(np.dtype([('aggregate_vulnerability_id', np.int32),
                                                 ('vulnerability_id', np.int32),]))

VulnerabilityWeight = nb.from_dtype(np.dtype([('areaperil_id', np.int32),
                                              ('vulnerability_id', np.int32),
                                              ('weight', np.int32)]))
