
from tests.pytools.converters.helpers import case_runner


def test_fm_policytc():
    case_runner("bintocsv", "fm_policytc", "input")
    case_runner("csvtobin", "fm_policytc", "input")


def test_fm_profile():
    case_runner("bintocsv", "fm_profile", "input")
    case_runner("csvtobin", "fm_profile", "input")


def test_fm_profile_step():
    case_runner("bintocsv", "fm_profile_step", "input")
    case_runner("csvtobin", "fm_profile_step", "input")


def test_fm_programme():
    case_runner("bintocsv", "fm_programme", "input")
    case_runner("csvtobin", "fm_programme", "input")


def test_fm_summary_xref():
    case_runner("bintocsv", "fm_summary_xref", "input")
    case_runner("csvtobin", "fm_summary_xref", "input")


def test_gul_summary_xref():
    case_runner("bintocsv", "gul_summary_xref", "input")
    case_runner("csvtobin", "gul_summary_xref", "input")


def test_fm_xref():
    case_runner("bintocsv", "fm_xref", "input")
    case_runner("csvtobin", "fm_xref", "input")
