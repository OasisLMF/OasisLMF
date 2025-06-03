import numba as nb
import numpy as np

from oasislmf.pytools.common.data import oasis_float, oasis_int


VALID_EXT = ["csv", "bin", "parquet"]

# Output flags
AGG_FULL_UNCERTAINTY = 0
AGG_WHEATSHEAF = 1
AGG_SAMPLE_MEAN = 2
AGG_WHEATSHEAF_MEAN = 3
OCC_FULL_UNCERTAINTY = 4
OCC_WHEATSHEAF = 5
OCC_SAMPLE_MEAN = 6
OCC_WHEATSHEAF_MEAN = 7

# EPCalcs
MEANDR = 1
FULL = 2
PERSAMPLEMEAN = 3
MEANSAMPLE = 4

# EPTypes
OEP = 1
OEPTVAR = 2
AEP = 3
AEPTVAR = 4


# Outloss mean and sample dtype, summary_id, period_no (and sidx) obtained from index
OUTLOSS_DTYPE = np.dtype([
    ("row_used", np.bool_),
    ("agg_out_loss", oasis_float),
    ("max_out_loss", oasis_float),
])

EPT_output = [
    ('SummaryId', oasis_int, '%d'),
    ('EPCalc', oasis_int, '%d'),
    ('EPType', oasis_int, '%d'),
    ('ReturnPeriod', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

PSEPT_output = [
    ('SummaryId', oasis_int, '%d'),
    ('SampleId', oasis_int, '%d'),
    ('EPType', oasis_int, '%d'),
    ('ReturnPeriod', oasis_float, '%.6f'),
    ('Loss', oasis_float, '%.6f'),
]

EPT_headers = [c[0] for c in EPT_output]
PSEPT_headers = [c[0] for c in PSEPT_output]
EPT_dtype = np.dtype([(c[0], c[1]) for c in EPT_output])
PSEPT_dtype = np.dtype([(c[0], c[1]) for c in PSEPT_output])
EPT_fmt = ','.join([c[2] for c in EPT_output])
PSEPT_fmt = ','.join([c[2] for c in PSEPT_output])

LOSSVEC2MAP_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("period_no", np.int32),
    ("period_weighting", np.float64),
    ("value", oasis_float),
])

WHEATKEYITEMS_dtype = np.dtype([
    ("summary_id", oasis_int),
    ("sidx", oasis_int),
    ("period_no", np.int32),
    ("period_weighting", np.float64),
    ("value", oasis_float),
])

MEANMAP_dtype = np.dtype([
    ("retperiod", np.float64),
    ("mean", np.float64),
    ("count", np.int32),
])

# For Dict of summary_id (oasis_int) to nb_Tail_valtype
TAIL_valtype = np.dtype([
    ("retperiod", np.float64),
    ("tvar", oasis_float),
])
NB_TAIL_valtype = nb.types.Array(nb.from_dtype(TAIL_valtype), 1, "C")
