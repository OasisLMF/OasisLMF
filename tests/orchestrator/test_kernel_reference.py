"""Tests of the full-kernel reference harness: tools/kernel_reference.py and tools/compare_outputs.py."""
import json
import os
import shutil
import subprocess

import pandas as pd
import pytest

from oasislmf.execution.bash import (ORD_ALT_MEANONLY_OUTPUT_SWITCHES, ORD_ALT_OUTPUT_SWITCHES, ORD_LECCALC,
                                     OUTPUT_SWITCHES)
from tests.orchestrator.conftest import REPO_ROOT
from tests.orchestrator.tools import compare_outputs
from tests.orchestrator.tools.compare_outputs import compare_outputs_rows, compare_trees_bytes, first_difference, list_files
from tests.orchestrator.tools.kernel_reference import (ALL_ORD_OUTPUTS, BASH_PARAMS_FILE, CONSOLE_SCRIPTS, MONITOR_SCRIPT,
                                                       REQUIRE_REF_ENV, SCRIPT_NAME, full_ord_settings, make_kernel_reference,
                                                       resolve_source, write_tool_wrappers)

K = 2
SUMMARY_SETS = {'gul': (1, 2), 'il': (1, 2, 3), 'ri': (1,)}
WORK_STEMS = {'gul': 'gul_', 'il': 'il_', 'ri': 'ri_IP4_'}
"""Run type -> prefix of its work files: genbash names the final RI level's (the fixture has 4) ``ri_IP4_``, and renames
its output files ``ri_`` at the end."""
ORD_TABLES = {'elt_sample': 'selt', 'elt_quantile': 'qelt', 'elt_moment': 'melt',
              'plt_sample': 'splt', 'plt_quantile': 'qplt', 'plt_moment': 'mplt'}
WORK_SUMMARY_DIRS = ('summary_palt', 'summary_altmeanonly', 'summaryleccalc')
AGG_OUTPUTS = ('palt', 'alct', 'altmeanonly', 'ept', 'psept')


def write_tree(root, files):
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return root


def flip_byte(path, offset):
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))


def copy_run(src, dst):
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns('log'))
    return dst


def expected_files(n_chunks):
    """Relative paths every full-kernel reference of the test fixture must hold."""
    files = set()
    for run_type, sets in SUMMARY_SETS.items():
        for n in sets:
            ext = 'parquet' if n == 3 else 'csv'
            for k in range(1, n_chunks + 1):
                files.update(f'work/kat/{WORK_STEMS[run_type]}S{n}_{table}_P{k}' for table in ORD_TABLES)
                files.update(f'work/{WORK_STEMS[run_type]}S{n}_{sub}/P{k}.bin' for sub in WORK_SUMMARY_DIRS)
            files.update(f'output/{run_type}_S{n}_{name}.{ext}' for name in list(ORD_TABLES.values()) + list(AGG_OUTPUTS))
            files.add(f'output/{run_type}_S{n}_summary-info.{ext}')
    return files


def test_full_ord_settings_switch_on_every_output_genbash_knows():
    settings = full_ord_settings(10)
    every_switch = set(ORD_LECCALC) | set(ORD_ALT_OUTPUT_SWITCHES) | set(ORD_ALT_MEANONLY_OUTPUT_SWITCHES)
    for tables in OUTPUT_SWITCHES.values():
        every_switch |= set(tables)
    assert every_switch <= set(ALL_ORD_OUTPUTS)
    for run_type, sets in SUMMARY_SETS.items():
        summaries = settings[f'{run_type}_summaries']
        assert settings[f'{run_type}_output'] is True
        assert [s['id'] for s in summaries] == list(sets)
        for summary in summaries:
            assert all(summary['ord_output'][name] for name in every_switch)
        assert any(s['ord_output'].get('parquet_format') for s in summaries) == (3 in sets)


@pytest.mark.requires_bash
def test_write_tool_wrappers(tmp_path, monkeypatch):
    monkeypatch.delenv(REQUIRE_REF_ENV, raising=False)
    bin_dir = write_tool_wrappers(tmp_path / 'bin', source=REPO_ROOT)
    for name, module in CONSOLE_SCRIPTS.items():
        text = (bin_dir / name).read_text()
        assert f'PYTHONPATH={REPO_ROOT} ' in text and f'from {module} import main' in text
        assert os.access(bin_dir / name, os.X_OK)
    assert (bin_dir / MONITOR_SCRIPT).exists()
    proc = subprocess.run([str(bin_dir / 'katpy'), '--help'], capture_output=True, text=True)
    assert proc.returncode == 0 and 'Concatenate' in proc.stdout


def test_write_tool_wrappers_rejects_a_directory_that_is_not_a_checkout(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_tool_wrappers(tmp_path / 'bin', source=tmp_path)


def test_first_difference(tmp_path):
    a = tmp_path / 'a'
    b = tmp_path / 'b'
    a.write_bytes(b'abcdef')
    b.write_bytes(b'abcdef')
    assert first_difference(a, b) is None
    b.write_bytes(b'abcXef')
    assert first_difference(a, b) == 3
    b.write_bytes(b'abc')
    assert first_difference(a, b) == 3


def test_compare_trees_bytes_on_synthetic_runs(tmp_path):
    files = {'work/kat/il_S1_elt_sample_P1': b'\x01\x02\x03', 'work/il_S1_summary_palt/P1.bin': b'xyz',
             'output/il_S1_selt.csv': 'EventId,Loss\n1,2.0\n', 'work/il_S1_summaryleccalc/lec_files/scratch.bdat': b'1'}
    a = write_tree(tmp_path / 'a', files)
    b = write_tree(tmp_path / 'b', {**files, 'work/il_S1_summaryleccalc/lec_files/scratch.bdat': b'2'})
    (a / 'input').symlink_to(tmp_path, target_is_directory=True)
    if hasattr(os, 'mkfifo'):
        os.mkfifo(a / 'work' / 'kat' / 'fifo_P1')
    report = compare_trees_bytes(a, b)
    assert report.ok and report.compared == 3, report.summary()

    flip_byte(b / 'work/kat/il_S1_elt_sample_P1', 1)
    (b / 'output/il_S1_selt.csv').unlink()
    (b / 'output/extra.csv').write_text('x')
    report = compare_trees_bytes(a, b)
    assert not report.ok
    assert report.only_in_a == ['output/il_S1_selt.csv']
    assert report.only_in_b == ['output/extra.csv']
    assert report.different == ['work/kat/il_S1_elt_sample_P1: first difference at byte 1 (sizes 3 and 3)']
    assert compare_trees_bytes(a, b, subdirs=('output',)).different == []


def test_compare_outputs_rows_on_synthetic_runs(tmp_path):
    a = write_tree(tmp_path / 'a', {'output/x_selt.csv': 'EventId,SummaryId,Loss\n1,1,10.0\n2,1,20.0\n3,2,30.0\n',
                                    'output/settings.json': '{}'})
    pd.DataFrame({'Period': [1, 2], 'Loss': [1.5, 2.5]}).to_parquet(a / 'output' / 'x_palt.parquet')
    b = copy_run(a, tmp_path / 'b')
    (b / 'output/x_selt.csv').write_text('EventId,SummaryId,Loss\n3,2,30.0\n1,1,10.000001\n2,1,20.0\n')
    pd.DataFrame({'Period': [2, 1], 'Loss': [2.5, 1.5]}).to_parquet(b / 'output' / 'x_palt.parquet')
    report = compare_outputs_rows(a, b)
    assert report.ok and report.compared == 3, report.summary()
    assert not compare_trees_bytes(a, b).ok

    (b / 'output/x_selt.csv').write_text('EventId,SummaryId,Loss\n3,2,30.0\n1,1,10.5\n2,1,20.0\n')
    assert compare_outputs_rows(a, b).different == [
        "output/x_selt.csv: 1 sorted rows differ in column 'Loss', first at row 0: np.float64(10.0) != np.float64(10.5)"]
    assert compare_outputs_rows(a, b, atol=1.0).ok
    (b / 'output/x_selt.csv').write_text('EventId,SummaryId,Loss\n3,2,30.0\n1,1,10.0\n')
    assert compare_outputs_rows(a, b).different == ['output/x_selt.csv: 3 rows != 2 rows']
    (b / 'output/x_selt.csv').write_text('EventId,SummaryId,Other\n3,2,30.0\n1,1,10.0\n2,1,20.0\n')
    assert 'columns' in compare_outputs_rows(a, b).different[0]
    (b / 'output/x_selt.csv').write_text('EventId,SummaryId,Loss\n3,2,30.0\n1,1,10.0\n4,1,20.0\n')
    assert "column 'EventId'" in compare_outputs_rows(a, b).different[0]
    (b / 'output/settings.json').write_text('{ }')
    assert 'output/settings.json: first difference at byte 1' in compare_outputs_rows(a, b).different


def test_compare_outputs_cli(tmp_path, capsys):
    a = write_tree(tmp_path / 'a', {'output/x.csv': 'A,B\n1,2.0\n', 'work/kat/x_P1': b'\x00'})
    b = copy_run(a, tmp_path / 'b')
    assert compare_outputs.main(['bytes', str(a), str(b)]) == 0
    assert 'EQUAL' in capsys.readouterr().out
    flip_byte(b / 'work/kat/x_P1', 0)
    assert compare_outputs.main(['bytes', str(a), str(b)]) == 1
    assert 'work/kat/x_P1' in capsys.readouterr().out
    assert compare_outputs.main(['bytes', str(a), str(b), '--subdirs', 'output']) == 0
    assert compare_outputs.main(['rows', str(a), str(b)]) == 0
    capsys.readouterr()


def test_full_kernel_reference_contents(full_kernel_reference):
    ref = full_kernel_reference(K)
    files = list_files(ref)
    missing = expected_files(K) - set(files)
    assert not missing, sorted(missing)
    assert all(files[rel].stat().st_size > 0 for rel in expected_files(K))
    assert not (ref / 'fifo').exists()
    assert not [rel for rel in files if rel.startswith('work/kat/') and rel not in expected_files(K)]

    params = json.loads((ref / BASH_PARAMS_FILE).read_text())
    assert params['max_process_id'] == K
    assert params['remove_working_files'] is False
    assert params['fifo_queue_dir'] == 'fifo/'
    assert params['work_dir'] == 'work/' and params['output_dir'] == 'output/'
    assert params['num_reinsurance_iterations'] == 4
    assert params['gul_output'] and params['il_output'] and params['ri_output'] and not params['rl_output']
    assert params['analysis_settings'] == json.loads((ref / 'analysis_settings.json').read_text())
    assert params['process_counter'] == {} and params['_get_getmodel_cmd'] is None

    script = (ref / SCRIPT_NAME).read_text()
    assert script.count('rm -R -f work/*') == 1
    assert script.index('rm -R -f work/*') < script.index(f'evepy 1 {K}')
    assert 'katpy' in script and 'lecpy' in script and 'aalpy' in script


def test_full_kernel_reference_is_deterministic(full_kernel_run, full_kernel_reference, tmp_path):
    ref = full_kernel_reference(K)
    again = make_kernel_reference(full_kernel_run, K, tmp_path / 'again', source=resolve_source())
    report = compare_trees_bytes(ref, again)
    assert report.ok, report.summary()
    assert report.compared == len(list_files(ref))
    assert (ref / SCRIPT_NAME).read_bytes() == (again / SCRIPT_NAME).read_bytes()
    assert (ref / BASH_PARAMS_FILE).read_bytes() == (again / BASH_PARAMS_FILE).read_bytes()


def test_full_kernel_reference_k1_and_k2_agree_as_rows(full_kernel_reference):
    one, two = full_kernel_reference(1), full_kernel_reference(K)
    report = compare_outputs_rows(one, two)
    assert report.ok, report.summary()
    assert report.compared == len(list_files(two, ('output',)))
    assert not compare_trees_bytes(one, two, subdirs=('work',)).ok


def test_compare_tools_detect_a_changed_byte_and_a_missing_file_in_a_reference(full_kernel_reference, tmp_path):
    ref = full_kernel_reference(K)
    copy = copy_run(ref, tmp_path / 'copy')
    assert compare_trees_bytes(ref, copy).ok
    flip_byte(copy / 'work' / 'kat' / 'il_S2_elt_sample_P2', 20)
    (copy / 'work' / 'ri_IP4_S1_summaryleccalc' / 'P2.bin').unlink()
    (copy / 'output' / 'gul_S1_ept.csv').unlink()
    report = compare_trees_bytes(ref, copy)
    assert report.different == [f'work/kat/il_S2_elt_sample_P2: first difference at byte 20 '
                                f'(sizes {(copy / "work/kat/il_S2_elt_sample_P2").stat().st_size}'
                                f' and {(copy / "work/kat/il_S2_elt_sample_P2").stat().st_size})']
    assert report.only_in_a == ['output/gul_S1_ept.csv', 'work/ri_IP4_S1_summaryleccalc/P2.bin']
    rows = compare_outputs_rows(ref, copy)
    assert rows.only_in_a == ['output/gul_S1_ept.csv'] and not rows.different


def test_full_kernel_reference_with_summarypy_low_memory_keeps_idx_files(full_kernel_reference):
    """With summarypy_low_memory, bash runs ``summarypy -m`` and tees a ``.idx`` beside each palt and leccalc work file.

    The ``.bin`` work files are unchanged, but lecpy takes its per-summary-id path when every
    ``.bin`` has an ``.idx``, which sums losses in another order: EPT/PSEPT of multi-summary sets
    differ in the last float digits, so a low-memory run must be checked against a low-memory
    reference byte for byte, and only agrees with a plain one as rows.
    """
    plain, low_memory = full_kernel_reference(K), full_kernel_reference(K, summarypy_low_memory=True)
    assert 'summarypy -m ' in (low_memory / SCRIPT_NAME).read_text()
    assert json.loads((low_memory / BASH_PARAMS_FILE).read_text())['summarypy_low_memory'] is True
    files = list_files(low_memory)
    idx = {rel for rel in files if rel.endswith('.idx')}
    expected_idx = {f'work/{WORK_STEMS[run_type]}S{n}_{sub}/P{k}.idx' for run_type, sets in SUMMARY_SETS.items() for n in sets
                    for sub in ('summary_palt', 'summaryleccalc') for k in range(1, K + 1)}
    assert idx == expected_idx
    assert all(files[rel].stat().st_size > 0 for rel in idx)
    report = compare_trees_bytes(plain, low_memory, subdirs=('work',))
    assert sorted(report.only_in_b) == sorted(idx) and not report.only_in_a and not report.different, report.summary()
    rows = compare_outputs_rows(plain, low_memory)
    assert rows.ok, rows.summary()
