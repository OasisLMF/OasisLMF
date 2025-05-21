from oasislmf.pytools.common.data import aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt
from oasislmf.pytools.common.data import amplifications_headers, amplifications_dtype, amplifications_fmt
from oasislmf.pytools.common.data import coverages_headers, coverages_dtype, coverages_fmt
from oasislmf.pytools.common.data import damagebin_headers, damagebin_dtype, damagebin_fmt
from oasislmf.pytools.common.data import eve_headers, eve_dtype, eve_fmt
from oasislmf.pytools.common.data import fmpolicytc_headers, fmpolicytc_dtype, fmpolicytc_fmt


SUPPORTED_TYPES = {
    "aggregatevulnerability": {
        "headers": aggregatevulnerability_headers,
        "dtype": aggregatevulnerability_dtype,
        "fmt": aggregatevulnerability_fmt,
    },
    "amplifications": {
        "headers": amplifications_headers,
        "dtype": amplifications_dtype,
        "fmt": amplifications_fmt,
    },
    "coverages": {
        "headers": coverages_headers,
        "dtype": coverages_dtype,
        "fmt": coverages_fmt,
    },
    "damagebin": {
        "headers": damagebin_headers,
        "dtype": damagebin_dtype,
        "fmt": damagebin_fmt,
    },
    "eve": {
        "headers": eve_headers,
        "dtype": eve_dtype,
        "fmt": eve_fmt,
    },
    "fmpolicytc": {
        "headers": fmpolicytc_headers,
        "dtype": fmpolicytc_dtype,
        "fmt": fmpolicytc_fmt,
    },
}
