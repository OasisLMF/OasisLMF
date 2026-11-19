"""GulpyStage and ModelpyGulpyStage give exactly the bytes of ``modelpy | gulpy``, whatever order the events come in.

The reference is ``modelpy | gulpy`` run by the pristine checkout ``tests/orchestrator/tools/kernel_reference.py``
resolves (``OASIS_ORCH_REF_SOURCE``, else ``~/GitHub/OasisLMF-ref`` when it exists, else the installed oasislmf).
The ``modelpy`` and ``gulpy`` commands of the code under test, built on the stages, are checked against it too.
"""
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from oasislmf.pytools.common.data import areaperil_int
from oasislmf.pytools.getmodel.manager import run as run_modelpy
from oasislmf.pytools.gul.common import ProbMean_size
from oasislmf.pytools.gul.io import read_getmodel_block
from oasislmf.pytools.gul.manager import run as run_gulpy
from oasislmf.pytools.gul.stage import GulpyStage, ModelpyGulpyStage
from tests.orchestrator.tools.kernel_reference import resolve_source, source_env, tool_command
from tests.pytools.utils import copy_test_model, set_coverage_buildings

KEYS_CSV = 'LocID,PerilID,CoverageTypeID,AreaPerilID,VulnerabilityID\n1,WTC,1,154,8\n2,WSS,1,54,2\n'

MODELS = {'plain': None, 'packed': -2, 'summed': 3}
"""Copies of test_model_1: as it is (with a keys.csv giving its two area perils two perils), every coverage with two
buildings kept separate, and with three buildings summed at source."""

CASES = {
    'S10_a0_rng2': ('plain', {}, dict(sample_size=10, alloc_rule=0, random_generator=2, loss_threshold=0.)),
    'S10_a1_rng1': ('plain', {}, dict(sample_size=10, alloc_rule=1, random_generator=1, loss_threshold=0.)),
    'S10_a2_rng0': ('plain', {}, dict(sample_size=10, alloc_rule=2, random_generator=0, loss_threshold=0.)),
    'S10_a3_rng2_threshold': ('plain', {}, dict(sample_size=10, alloc_rule=3, random_generator=2, loss_threshold=5000.)),
    'S10_ignore_correlation': ('plain', {}, dict(sample_size=10, alloc_rule=1, random_generator=2, loss_threshold=0.,
                                                 ignore_correlation=True)),
    'S0_ignore_correlation': ('plain', {}, dict(sample_size=0, alloc_rule=1, loss_threshold=0., ignore_correlation=True)),
    'S10_debug_rng1': ('plain', {}, dict(sample_size=10, random_generator=1, loss_threshold=0., debug=True)),
    'S2000_a1_rng2': ('plain', {}, dict(sample_size=2000, alloc_rule=1, random_generator=2, loss_threshold=0.)),
    'S10_gulpy_defaults': ('plain', {}, dict(sample_size=10)),
    'peril_filter': ('plain', {'peril_filter': ['WTC']}, dict(sample_size=10, alloc_rule=1, random_generator=2, loss_threshold=0.)),
    'packed_a1': ('packed', {}, dict(sample_size=10, alloc_rule=1, random_generator=2, loss_threshold=0.)),
    'summed_a2': ('summed', {}, dict(sample_size=10, alloc_rule=2, random_generator=2, loss_threshold=0.)),
}


def gulpy_flags(kwargs):
    """The gulpy command-line options of ``GulpyStage`` keyword arguments."""
    flags = []
    for key, flag in (('sample_size', '-S'), ('alloc_rule', '-a'), ('loss_threshold', '-L')):
        if key in kwargs:
            flags.append(f'{flag}{kwargs[key]}')
    if 'random_generator' in kwargs:
        flags.append(f"--random-generator={kwargs['random_generator']}")
    if kwargs.get('debug'):
        flags.append('-d')
    if kwargs.get('ignore_correlation'):
        flags.append('--ignore-correlation')
    return flags


def modelpy_flags(kwargs):
    """The modelpy command-line options of ``ModelpyStage`` keyword arguments."""
    return ['--peril-filter', *kwargs['peril_filter']] if kwargs.get('peril_filter') else []


def run_reference(source, name, args, cwd):
    """Run a pytools command of the reference checkout."""
    proc = subprocess.run(tool_command(name) + [str(a) for a in args], cwd=cwd, env=source_env(source), capture_output=True,
                          text=True)
    assert proc.returncode == 0, f'reference {name} {args} failed:\n{proc.stderr[-3000:]}'


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    """The model directories of ``MODELS``, by name."""
    root = tmp_path_factory.mktemp('gulpy_stage_models')
    dirs = {}
    for name, n_building in MODELS.items():
        run_dir = copy_test_model(root / name)
        (run_dir / 'input' / 'keys.csv').write_text(KEYS_CSV)
        if n_building is not None:
            set_coverage_buildings(run_dir / 'input', n_building)
        dirs[name] = run_dir
    return dirs


@pytest.fixture(scope='module')
def reference(models, tmp_path_factory):
    """``reference(model, modelpy_kwargs, gulpy_kwargs)``: the paths of the CDF and loss streams the reference writes."""
    source = resolve_source()
    out_dir = tmp_path_factory.mktemp('gulpy_stage_reference')
    cache = {}

    def make(model, modelpy_kwargs, gulpy_kwargs):
        run_dir = models[model]
        cdf_key = (model, tuple(modelpy_flags(modelpy_kwargs)))
        if cdf_key not in cache:
            cdf = out_dir / f'cdf_{len(cache)}.bin'
            run_reference(source, 'modelpy', ['-i', 'input/events.bin', '-o', cdf, *cdf_key[1]], run_dir)
            cache[cdf_key] = cdf
        cdf = cache[cdf_key]
        key = (cdf_key, tuple(gulpy_flags(gulpy_kwargs)))
        if key not in cache:
            gul = out_dir / f'gul_{len(cache)}.bin'
            run_reference(source, 'gulpy', ['-i', cdf, '-o', gul, *key[1]], run_dir)
            cache[key] = gul
        return cdf, cache[key]

    return make


def stage_bytes(stage, event_ids):
    """Run ``event_ids`` through a stage in the order given; return {event_id: bytes}."""
    per_event = {}
    for event_id in event_ids:
        out = io.BytesIO()
        stage.process_event(event_id, out)
        per_event[int(event_id)] = out.getvalue()
    return per_event


def read_event_ids(run_dir):
    return np.fromfile(Path(run_dir) / 'input' / 'events.bin', dtype='i4')


@pytest.mark.parametrize('case', list(CASES))
def test_stage_and_commands_match_the_reference_modelpy_gulpy(models, reference, case, tmp_path, monkeypatch):
    """Header + per-event output of the stage is the reference ``modelpy | gulpy``, in any event order; so are the commands."""
    monkeypatch.chdir(tmp_path)
    model, modelpy_kwargs, gulpy_kwargs = CASES[case]
    run_dir = models[model]
    ref_cdf, ref_gul = reference(model, modelpy_kwargs, gulpy_kwargs)
    expected = ref_gul.read_bytes()
    assert len(expected) > 8
    event_ids = read_event_ids(run_dir)

    with ModelpyGulpyStage(str(run_dir), modelpy=modelpy_kwargs, gulpy=gulpy_kwargs) as stage:
        in_order = stage_bytes(stage, event_ids)
        assert stage.header + b''.join(in_order[int(e)] for e in event_ids) == expected
        assert stage_bytes(stage, event_ids[::-1]) == in_order
    for event_id in event_ids[:2]:
        with ModelpyGulpyStage(str(run_dir), modelpy=modelpy_kwargs, gulpy=gulpy_kwargs) as stage:
            assert stage_bytes(stage, [event_id]) == {int(event_id): in_order[int(event_id)]}

    cdf = tmp_path / 'cdf.bin'
    run_modelpy(run_dir=str(run_dir), file_in=str(run_dir / 'input' / 'events.bin'), file_out=str(cdf), ignore_file_type=set(),
                data_server=False, peril_filter=modelpy_kwargs.get('peril_filter'))
    assert cdf.read_bytes() == ref_cdf.read_bytes()
    gul = tmp_path / 'gul.bin'
    gulpy_args = {'sample_size': 0, 'loss_threshold': 1e-6, 'alloc_rule': 0, 'debug': False, 'random_generator': 2, **gulpy_kwargs}
    run_gulpy(run_dir=str(run_dir), ignore_file_type=set(), file_in=str(cdf), file_out=str(gul), **gulpy_args)
    assert gul.read_bytes() == expected


def test_peril_filter_and_packing_change_the_losses(models, reference):
    """The cases on the modified models are not the plain run under another name."""
    plain = reference('plain', {}, CASES['peril_filter'][2])[1].read_bytes()
    filtered = reference('plain', {'peril_filter': ['WTC']}, CASES['peril_filter'][2])[1].read_bytes()
    packed = reference('packed', {}, CASES['packed_a1'][2])[1].read_bytes()
    assert 8 < len(filtered) < len(plain)
    assert packed != reference('plain', {}, CASES['packed_a1'][2])[1].read_bytes()


def test_gulpy_stage_reads_a_block_as_gulpy_reads_its_stream(models, reference, tmp_path, monkeypatch):
    """GulpyStage on the reference CDF stream split into events gives the reference loss stream."""
    monkeypatch.chdir(tmp_path)
    gulpy_kwargs = CASES['S10_a1_rng1'][2]
    ref_cdf, ref_gul = reference('plain', {}, gulpy_kwargs)
    records = cdf_records(ref_cdf.read_bytes()[4:])
    stage = GulpyStage(str(models['plain']), **gulpy_kwargs)
    out = io.BytesIO()
    for event_bytes in records.values():
        stage.process_cdf_event(stage.read_block(np.frombuffer(event_bytes, dtype=np.int8)), out)
    assert stage.header + out.getvalue() == ref_gul.read_bytes()
    assert len(records) > 1


def cdf_records(data):
    """Split a CDF stream (without header) into its events: {event_id: bytes}, in stream order."""
    record_header = 4 + areaperil_int.itemsize + 4 + 4
    events, pos = {}, 0
    while pos < len(data):
        event_id = int.from_bytes(data[pos:pos + 4], 'little', signed=True)
        n_bins = int.from_bytes(data[pos + record_header - 4:pos + record_header], 'little', signed=True)
        end = pos + record_header + n_bins * ProbMean_size
        events[event_id] = events.get(event_id, b'') + data[pos:end]
        pos = end
    return events


def test_read_getmodel_block_rejects_what_is_not_one_whole_event(models, reference, tmp_path, monkeypatch):
    """An empty block is no event; a truncated block or one holding two events is an error."""
    monkeypatch.chdir(tmp_path)
    ref_cdf, _ = reference('plain', {}, CASES['S10_a1_rng1'][2])
    records = list(cdf_records(ref_cdf.read_bytes()[4:]).values())
    stage = GulpyStage(str(models['plain']), sample_size=10)
    assert stage.read_block(np.empty(0, dtype=np.int8)) is None
    with pytest.raises(ValueError, match='ended prematurely'):
        stage.read_block(np.frombuffer(records[0][:-4], dtype=np.int8))
    with pytest.raises(ValueError, match='more than one event'):
        stage.read_block(np.frombuffer(records[0] + records[1], dtype=np.int8))
    assert read_getmodel_block(np.frombuffer(records[0], dtype=np.int8), stage.items, stage.item_map_hm, stage.item_map_hm_keys,
                               stage.item_map_ja_offsets, stage.coverages, stage.compute, stage.seeds)[0] == \
        int(np.frombuffer(records[0][:4], dtype='<i4')[0])


def test_modelpy_gulpy_stage_writes_nothing_for_an_event_without_footprint(models, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_dir = models['plain']
    with ModelpyGulpyStage(str(run_dir), gulpy={'sample_size': 10}) as stage:
        out = io.BytesIO()
        assert stage.process_event(int(read_event_ids(run_dir).max()) + 1000, out) is False
        assert out.getvalue() == b''


def test_modelpy_gulpy_stage_never_uses_the_data_server(models):
    with pytest.raises(ValueError, match='data server'):
        ModelpyGulpyStage(str(models['plain']), modelpy={'data_server': True})


def test_gulpy_stage_rejects_an_unknown_alloc_rule(models):
    with pytest.raises(ValueError, match='alloc_rule'):
        GulpyStage(str(models['plain']), sample_size=10, alloc_rule=4)


def test_a_structure_cache_gives_the_same_stage_output(models, tmp_path, monkeypatch):
    """With the getmodel and gulpy structures saved, as GenerateLosses saves them, the stage reads them and agrees."""
    monkeypatch.chdir(tmp_path)
    run_dir = copy_test_model(tmp_path / 'cached', models['plain'])
    shutil.rmtree(run_dir / 'input' / 'getmodel_structure', ignore_errors=True)
    event_ids = read_event_ids(run_dir)
    gulpy_kwargs = CASES['S10_a0_rng2'][2]
    with ModelpyGulpyStage(str(run_dir), gulpy=gulpy_kwargs) as stage:
        without = stage_bytes(stage, event_ids)
    for command in ('modelpy', 'gulpy'):
        subprocess.run(tool_command(command) + ['--create-structures', '--run-dir', str(run_dir)], check=True, capture_output=True)
    assert (run_dir / 'input' / 'getmodel_structure').is_dir() and (run_dir / 'input' / 'gulpy_structure').is_dir()
    with ModelpyGulpyStage(str(run_dir), gulpy=gulpy_kwargs) as stage:
        assert stage_bytes(stage, event_ids) == without
