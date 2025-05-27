from oasislmf.pytools.common.data import aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt
from oasislmf.pytools.common.data import amplifications_headers, amplifications_dtype, amplifications_fmt
from oasislmf.pytools.common.data import coverages_headers, coverages_dtype, coverages_fmt
from oasislmf.pytools.common.data import damagebin_headers, damagebin_dtype, damagebin_fmt
from oasislmf.pytools.common.data import eve_headers, eve_dtype, eve_fmt
from oasislmf.pytools.common.data import fm_policytc_headers, fm_policytc_dtype, fm_policytc_fmt


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
    "fm_policytc": {
        "headers": fm_policytc_headers,
        "dtype": fm_policytc_dtype,
        "fmt": fm_policytc_fmt,
    },
}
