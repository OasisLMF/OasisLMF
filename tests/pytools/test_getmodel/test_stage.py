"""ModelpyStage gives exactly the bytes of the ``modelpy`` command, whatever order the events come in.

The reference is ``modelpy`` run by the pristine checkout ``tests/orchestrator/tools/kernel_reference.py`` resolves
(``OASIS_ORCH_REF_SOURCE``, else ``~/GitHub/OasisLMF-ref`` when it exists, else the installed oasislmf). The
``modelpy`` command of the code under test, built on the stage, is checked against it too.
"""
import io
import subprocess
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.getmodel.manager import run as run_modelpy
from oasislmf.pytools.getmodel.stage import ModelpyStage
from tests.orchestrator.tools.kernel_reference import resolve_source, source_env, tool_command
from tests.pytools.utils import copy_test_model

TESTS_ASSETS_DIR = Path(__file__).parent.parent.parent.joinpath("assets")
KEYS_CSV = 'LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n1,WTC,1,154,8\n2,WSS,1,54,2\n'

CASES = {
    'test_model_1': ('test_model_1', None),
    'test_model_1_peril_filter': ('test_model_1', ['WTC']),
    'test_model_3': ('test_model_3', None),
    'test_model_7': ('test_model_7', None),
}
"""case -> (model, peril filter). The other test models need what only gulmc has (aggregate or conditional vulnerabilities)."""


@pytest.fixture(scope='module')
def ref_source():
    return resolve_source()


def model_copy(root, model):
    run_dir = root / model
    if not run_dir.exists():
        copy_test_model(run_dir, TESTS_ASSETS_DIR / model)
        if model == 'test_model_1':
            (run_dir / 'input' / 'keys.csv').write_text(KEYS_CSV)
    return run_dir


def stage_bytes(stage, event_ids):
    """Run ``event_ids`` through the stage in the order given; return {event_id: bytes} and the event ids it counted empty."""
    per_event, empty = {}, []
    for event_id in event_ids:
        out = io.BytesIO()
        if not stage.process_event(event_id, out):
            empty.append(int(event_id))
        per_event[int(event_id)] = out.getvalue()
    return per_event, empty


@pytest.mark.parametrize('case', list(CASES))
def test_stage_and_command_match_the_reference_modelpy(ref_source, case, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model, peril_filter = CASES[case]
    run_dir = model_copy(tmp_path, model)
    flags = ['--peril-filter', *peril_filter] if peril_filter else []
    ref = tmp_path / 'ref.bin'
    proc = subprocess.run(tool_command('modelpy') + ['-i', 'input/events.bin', '-o', str(ref), *flags], cwd=run_dir,
                          env=source_env(ref_source), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-3000:]
    expected = ref.read_bytes()
    assert len(expected) > 4
    event_ids = np.fromfile(run_dir / 'input' / 'events.bin', dtype='i4')

    with ModelpyStage(str(run_dir), peril_filter=peril_filter) as stage:
        in_order, empty = stage_bytes(stage, event_ids)
        assert stage.header + b''.join(in_order[int(e)] for e in event_ids) == expected
        assert stage_bytes(stage, event_ids[::-1]) == (in_order, empty[::-1])
        assert empty == [int(e) for e in event_ids if not in_order[int(e)]]

    out = tmp_path / 'cdf.bin'
    run_modelpy(run_dir=str(run_dir), file_in=str(run_dir / 'input' / 'events.bin'), file_out=str(out), ignore_file_type=set(),
                data_server=False, peril_filter=peril_filter)
    assert out.read_bytes() == expected


def test_stage_counts_an_event_without_footprint_as_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = model_copy(tmp_path, 'test_model_1')
    event_ids = np.fromfile(run_dir / 'input' / 'events.bin', dtype='i4')
    with ModelpyStage(str(run_dir)) as stage:
        out = io.BytesIO()
        assert stage.process_event(int(event_ids.max()) + 1000, out) is False
        assert out.getvalue() == b''
