import numpy as np

BUFFER_SIZE = 65536
DATA_SIZE = 8   # 4 bytes * 2 values
N_PAIRS = BUFFER_SIZE // DATA_SIZE
FILE_HEADER_SIZE = 4   # 4-byte integer

# Numpy data types for reading binary files
event_item_dtype = np.dtype([('event_id', 'i4'), ('item_id', 'i4')])
sidx_loss_dtype = np.dtype([('sidx', 'i4'), ('loss', 'f4')])
event_count_dtype = np.dtype([('event_id', 'i4'), ('count', 'i4')])
amp_factor_dtype = np.dtype([('amplification_id', 'i4'), ('factor', 'f4')])

# File names for creating dictionary of factors
AMPLIFICATIONS_FILE_NAME = 'amplifications.bin'
LOSS_FACTORS_FILE_NAME = 'lossfactors.bin'
