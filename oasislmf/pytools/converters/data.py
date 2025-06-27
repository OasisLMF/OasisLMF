from oasislmf.pytools.common.data import aggregatevulnerability_headers, aggregatevulnerability_dtype, aggregatevulnerability_fmt
from oasislmf.pytools.common.data import amplifications_headers, amplifications_dtype, amplifications_fmt
from oasislmf.pytools.common.data import complex_items_meta_headers, complex_items_meta_dtype, complex_items_meta_fmt
from oasislmf.pytools.common.data import coverages_headers, coverages_dtype, coverages_fmt
from oasislmf.pytools.common.data import damagebin_headers, damagebin_dtype, damagebin_fmt
from oasislmf.pytools.common.data import eve_headers, eve_dtype, eve_fmt
from oasislmf.pytools.common.data import fm_policytc_headers, fm_policytc_dtype, fm_policytc_fmt
from oasislmf.pytools.common.data import fm_profile_headers, fm_profile_dtype, fm_profile_fmt
from oasislmf.pytools.common.data import fm_profile_step_headers, fm_profile_step_dtype, fm_profile_step_fmt
from oasislmf.pytools.common.data import fm_programme_headers, fm_programme_dtype, fm_programme_fmt
from oasislmf.pytools.common.data import fm_summary_xref_headers, fm_summary_xref_dtype, fm_summary_xref_fmt
from oasislmf.pytools.common.data import fm_xref_headers, fm_xref_dtype, fm_xref_fmt
from oasislmf.pytools.common.data import gul_summary_xref_headers, gul_summary_xref_dtype, gul_summary_xref_fmt
from oasislmf.pytools.common.data import items_headers, items_dtype, items_fmt


SUPPORTED_CSVTOBIN = [
    "aggregatevulnerability",
    "amplifications",
    "complex_items",
    "coverages",
    "damagebin",
    "eve",
    "fm_policytc",
    "fm_profile",
    "fm_profile_step",
    "fm_programme",
    "fm_summary_xref",
    "fm_xref",
    "gul_summary_xref",
    "items",
]


SUPPORTED_BINTOCSV = [
    "aggregatevulnerability",
    "amplifications",
    "complex_items",
    "coverages",
    "damagebin",
    "eve",
    "fm_policytc",
    "fm_profile",
    "fm_profile_step",
    "fm_programme",
    "fm_summary_xref",
    "fm_xref",
    "gul_summary_xref",
    "items",
]


TYPE_MAP = {
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
    "complex_items": {
        "headers": complex_items_meta_headers,
        "dtype": complex_items_meta_dtype,
        "fmt": complex_items_meta_fmt,
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
    "fm_profile": {
        "headers": fm_profile_headers,
        "dtype": fm_profile_dtype,
        "fmt": fm_profile_fmt,
    },
    "fm_profile_step": {
        "headers": fm_profile_step_headers,
        "dtype": fm_profile_step_dtype,
        "fmt": fm_profile_step_fmt,
    },
    "fm_programme": {
        "headers": fm_programme_headers,
        "dtype": fm_programme_dtype,
        "fmt": fm_programme_fmt,
    },
    "fm_summary_xref": {
        "headers": fm_summary_xref_headers,
        "dtype": fm_summary_xref_dtype,
        "fmt": fm_summary_xref_fmt,
    },
    "fm_xref": {
        "headers": fm_xref_headers,
        "dtype": fm_xref_dtype,
        "fmt": fm_xref_fmt,
    },
    "gul_summary_xref": {
        "headers": gul_summary_xref_headers,
        "dtype": gul_summary_xref_dtype,
        "fmt": gul_summary_xref_fmt,
    },
    "items": {
        "headers": items_headers,
        "dtype": items_dtype,
        "fmt": items_fmt,
    },
}
