import numpy as np

BUFFER_SIZE = 65536
DATA_SIZE = 8   # 4 bytes * 2 values
N_PAIRS = BUFFER_SIZE // DATA_SIZE

# Numpy data types for reading binary files
event_count_dtype = np.dtype([('event_id', 'i4'), ('count', 'i4')])
amp_factor_dtype = np.dtype([('amplification_id', 'i4'), ('factor', 'f4')])

# File names for creating dictionary of factors
PLAFACTORS_FILE = 'lossfactors.bin'
