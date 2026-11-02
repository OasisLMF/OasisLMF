"""Numba keeps the arguments of the call that compiles a function which formats a number as a string
(str() or an f-string) alive until the next garbage collection, even if that code never runs (#2238):
with a cold numba cache, the failed typing attempts it keeps hold the compiling frame in a reference
cycle (as in numba/numba#5419). A memory-mapped array passed to that call stays mapped meanwhile, and
Windows can't delete the file. Error messages in jitted code therefore pass values as extra
exception arguments instead of formatting them into the message.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from oasislmf.pytools.aal.manager import _AAL_REC_DTYPE, _AAL_REC_PERIOD_DTYPE, get_aal_data, get_aal_data_meanonly
from oasislmf.pytools.common.data import fm_profile_step_dtype, vulnerability_dtype
from oasislmf.pytools.common.event_stream import reservation_overflows
from oasislmf.pytools.fm import policy, policy_extras
from oasislmf.pytools.getmodel.manager import load_vuln_probability
from oasislmf.pytools.lec.utils import _inverse_remap_sidx, _remap_sidx

PYTOOLS_DIR = Path(__file__).parents[2] / "oasislmf" / "pytools"


def _formats_string(node):
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        return (isinstance(func, ast.Name) and func.id in ("str", "repr", "format")) or \
            (isinstance(func, ast.Attribute) and func.attr == "format")
    return False


def test_jitted_functions_do_not_format_strings():
    found = []
    for path in sorted(PYTOOLS_DIR.rglob("*.py")):
        for func in ast.walk(ast.parse(path.read_text())):
            if isinstance(func, ast.FunctionDef) and any("jit" in ast.unparse(d) for d in func.decorator_list):
                found += [f"{path.relative_to(PYTOOLS_DIR)}:{node.lineno} in {func.name}"
                          for node in ast.walk(func) if _formats_string(node)]
    assert not found, "string formatting in jitted code, pass values as exception arguments instead:\n" + "\n".join(found)


@pytest.mark.skipif(os.environ.get("NUMBA_DISABLE_JIT") == "1", reason="needs numba compilation")
def test_load_vulns_bin_releases_its_array_on_a_cold_cache(tmp_path):
    """The Windows failure in #2238: with an empty numba cache, get_vulns kept vulnerability.bin
    mapped because load_vulns_bin's first call held on to the array (in a reference cycle, so gc is
    disabled to keep it from being collected before the check). Runs in a subprocess with its own
    cache dir so the functions are compiled by the call being checked."""
    script = dedent("""
        import gc
        gc.disable()
        import sys
        import numpy as np
        from oasislmf.pytools.common.data import items_dtype, vulnerability_dtype
        from oasislmf.pytools.getmodel.manager import load_items, load_vulns_bin

        items = np.zeros(1, dtype=items_dtype)
        items["item_id"] = 1
        items["areaperil_id"] = 1
        items["vulnerability_id"] = 1
        vuln_map, keys, *_ = load_items(items)
        vulns = np.zeros(2, dtype=vulnerability_dtype)
        vulns["vulnerability_id"] = 1
        vulns["intensity_bin_id"] = [1, 2]
        vulns["damage_bin_id"] = 1
        vulns["probability"] = 1

        before = sys.getrefcount(vulns)
        load_vulns_bin(vulns, vuln_map, keys, 1, 2)
        print(sys.getrefcount(vulns) - before)
    """)
    result = subprocess.run([sys.executable, "-c", script], env={**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)},
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split()[-1] == "0"


def test_lec_sidx_errors_report_the_value():
    with pytest.raises(ValueError) as e:
        _remap_sidx(-5)
    assert e.value.args == ("Invalid sidx value:", -5)

    with pytest.raises(ValueError) as e:
        _inverse_remap_sidx(-1)
    assert e.value.args == ("Invalid remapped_sidx value:", -1)


@pytest.mark.parametrize("stepped", [True, None])
def test_fm_unknown_calcrule_reports_the_id(stepped):
    profile = np.zeros(1, dtype=fm_profile_step_dtype)[0]
    profile["calcrule_id"] = 999
    loss = np.ones(2)

    with pytest.raises(ValueError) as e:
        policy.calc(profile, np.zeros(2), loss, stepped)
    assert e.value.args == ("UnknownCalcrule", 999)

    with pytest.raises(ValueError) as e:
        policy_extras.calc(profile, np.zeros(2), loss, np.zeros(2), np.zeros(2), np.zeros(2), stepped)
    assert e.value.args == ("UnknownCalcrule", 999)


@pytest.mark.parametrize("damage_bin_id, intensity_bin_id, problem", [
    (5, 1, "has damage_bin_id bigger that expected maximum"),
    (0, 1, "has a damage_bin_id or intensity_bin_id lower than 1"),
    (1, 0, "has a damage_bin_id or intensity_bin_id lower than 1"),
])
def test_load_vuln_probability_errors_report_the_vulnerability_id(damage_bin_id, intensity_bin_id, problem):
    vuln = np.zeros(1, dtype=vulnerability_dtype)[0]
    vuln["damage_bin_id"] = damage_bin_id
    vuln["intensity_bin_id"] = intensity_bin_id

    with pytest.raises(Exception) as e:
        load_vuln_probability(np.zeros((2, 2)), vuln, 7)
    assert e.value.args == ("vulnerability_id", 7, problem)


def test_reservation_overflow_error_reports_the_sizes():
    with pytest.raises(ValueError) as e:
        reservation_overflows(0, 10, 5, "SELT")
    assert e.value.args[0] == "SELT"
    assert e.value.args[2] == 10 and e.value.args[4] == 5
    assert "OASIS_DEFAULT_BUFFER_SIZE" in e.value.args[-1]


@pytest.mark.parametrize("get_data", [get_aal_data, get_aal_data_meanonly])
def test_aal_length_mismatch_reports_both_lengths(get_data):
    with pytest.raises(AssertionError) as e:
        get_data(np.zeros(2, dtype=_AAL_REC_DTYPE), np.zeros(3, dtype=_AAL_REC_PERIOD_DTYPE),
                 np.zeros(2, dtype=np.bool_), 1, 1)
    assert e.value.args == (("Lengths of analytical and sample aal data differ:", 2, 3),)
