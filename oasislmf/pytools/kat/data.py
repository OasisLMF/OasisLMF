from oasislmf.pytools.elt.data import MELT_dtype, MELT_fmt, MELT_headers, QELT_dtype, QELT_fmt, QELT_headers, SELT_dtype, SELT_fmt, SELT_headers
from oasislmf.pytools.plt.data import MPLT_dtype, MPLT_fmt, MPLT_headers, QPLT_dtype, QPLT_fmt, QPLT_headers, SPLT_dtype, SPLT_fmt, SPLT_headers


VALID_EXT = ["csv", "parquet", "bin", None]

KAT_SELT = 0
KAT_MELT = 1
KAT_QELT = 2
KAT_SPLT = 3
KAT_MPLT = 4
KAT_QPLT = 5

KAT_MAP = {
    KAT_SELT: {
        "name": "SELT",
        "headers": SELT_headers,
        "dtype": SELT_dtype,
        "fmt": SELT_fmt,
    },
    KAT_MELT: {
        "name": "MELT",
        "headers": MELT_headers,
        "dtype": MELT_dtype,
        "fmt": MELT_fmt,
    },
    KAT_QELT: {
        "name": "QELT",
        "headers": QELT_headers,
        "dtype": QELT_dtype,
        "fmt": QELT_fmt,
    },
    KAT_SPLT: {
        "name": "SPLT",
        "headers": SPLT_headers,
        "dtype": SPLT_dtype,
        "fmt": SPLT_fmt,
    },
    KAT_MPLT: {
        "name": "MPLT",
        "headers": MPLT_headers,
        "dtype": MPLT_dtype,
        "fmt": MPLT_fmt,
    },
    KAT_QPLT: {
        "name": "QPLT",
        "headers": QPLT_headers,
        "dtype": QPLT_dtype,
        "fmt": QPLT_fmt,
    },
}
