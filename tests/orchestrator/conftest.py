"""Fixtures for the orchestrator tests: a small prepared kernel run directory and its reference streams.

The run directory is built from what is in this repository, in a few seconds:

* the model (``static/`` footprint, vulnerability and damage bins, ``input/events.bin``) comes
  from ``GenerateDummyModelFiles`` with a fixed seed;
* the exposure is the OED of ``validation/reinsurance1`` (290 locations, accounts and a
  reinsurance programme with four inuring priorities, so ``input/RI_1`` ... ``input/RI_4``);
* keys come from ``GenerateKeysDeterministic`` with the area peril and vulnerability ids spread
  over the dummy model's, and ``GenerateFiles`` writes the Oasis input files;
* the fmpy financial structure files are written for the IL step and every RI step.

Full-kernel references (gul, il and ri with summaries and every ORD output, work files kept) are
produced with ``tools/kernel_reference.py`` from a pristine checkout named by the
``OASIS_ORCH_REF_SOURCE`` environment variable, by default ``~/GitHub/OasisLMF-ref`` when that
exists, else (or with ``OASIS_ORCH_REF_SOURCE=installed``) by the installed ``oasislmf``.

Reference streams are produced with ``tools/reference_streams.py``: by the tools of the running
interpreter, or by those in the directory named by the ``OASIS_ORCH_REF_TOOLS`` environment
variable (for example wrappers around a pristine checkout of main, written by
``tools/kernel_reference.py wrappers``).

With ``OASIS_ORCH_REQUIRE_REF`` set (CI sets it) both must come from a pristine checkout: a fixture
that would make them with the code under test fails instead. Which code makes them is written to
the terminal when the fixture is first set up.
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ods_tools.oed import OedExposure, OedSchema

from oasislmf.computation.generate.files import GenerateDummyModelFiles, GenerateFiles
from oasislmf.computation.generate.keys import GenerateKeysDeterministic
from oasislmf.utils.data import DEFAULT_ADDITIONAL_FIELDS, prepare_oed_exposure
from tests.orchestrator.tools.kernel_reference import (REQUIRE_REF_ENV, NoPristineReference, describe_source, full_ord_settings,
                                                       make_kernel_reference, prepare_run_dir, reference_required, resolve_source)
from tests.orchestrator.tools.reference_streams import (DEFAULT_GUL_ARGS, DEFAULT_IL_ALLOC_RULE, DEFAULT_RI_ALLOC_RULE,
                                                        DEFAULT_RI_LAYERS, DEFAULT_SAMPLES, make_reference_streams,
                                                        prepare_financial_structures, resolve_tools)

REPO_ROOT = Path(__file__).resolve().parents[2]
OED_CASE_DIR = REPO_ROOT / 'validation' / 'reinsurance1'
REF_TOOLS_ENV = 'OASIS_ORCH_REF_TOOLS'

N_EVENTS = 30
N_AREAPERILS = 20
AREAPERILS_PER_EVENT = 12
N_VULNERABILITIES = 10
N_INTENSITY_BINS = 10
N_DAMAGE_BINS = 10
MODEL_SEED = 1234
PERIL = 'WTC'
RI_LAYERS_AVAILABLE = 4
OED_FILES = {'location': 'location.csv', 'account': 'account.csv', 'ri_info': 'ri_info.csv', 'ri_scope': 'ri_scope.csv'}


@dataclass
class KernelRun:
    """A prepared kernel run directory.

    Attributes:
        run_dir (Path): the directory; the tools and stages run with it as working directory.
        event_ids (np.ndarray): the events of ``input/events.bin``, in file order.
        ri_layers (int): RI inuring priorities available (``input/RI_1`` ... ``input/RI_<n>``).
        samples (int): sample count the references use by default.
        gul_args (tuple): gulmc arguments besides -S the references use.
        il_alloc_rule (int): IL allocation rule the references use.
        ri_alloc_rule (int): RI allocation rule the references use.
        ref_tools (str | None): directory of the tools the references are made with, None for this interpreter's.
    """
    run_dir: Path
    event_ids: np.ndarray
    ri_layers: int = RI_LAYERS_AVAILABLE
    samples: int = DEFAULT_SAMPLES
    gul_args: tuple = DEFAULT_GUL_ARGS
    il_alloc_rule: int = DEFAULT_IL_ALLOC_RULE
    ri_alloc_rule: int = DEFAULT_RI_ALLOC_RULE
    ref_tools: str = field(default_factory=lambda: os.environ.get(REF_TOOLS_ENV) or None)


def oed_case_exposure(*oed_types):
    """The OED of ``OED_CASE_DIR`` as the computation steps would load it, read from Path objects.

    A computation step turns its OED file parameters into strings, which ods_tools reads through
    ``LocalStorage.get_storage_url`` as ``file://<path>``: on Windows that is ``file://C:\\...``,
    whose path part is empty. A Path is read by pandas directly, on any OS.

    Args:
        *oed_types (str): keys of ``OED_FILES`` to load.

    Returns:
        OedExposure: the exposure with the internal columns added, for a step's ``exposure_data`` argument.
    """
    exposure = OedExposure(**{oed_type: OED_CASE_DIR / OED_FILES[oed_type] for oed_type in oed_types},
                           check_oed=True, use_field=True, backend_dtype='pd_dtype', additional_fields=DEFAULT_ADDITIONAL_FIELDS)
    prepare_oed_exposure(exposure)
    return exposure


def portable_exposure_config(oasis_files_dir):
    """Record the OED schema of ``exposure_info.json`` as its version, not as the bundled schema's path.

    ``OedExposure.save_config`` stores the schema's full path, which ``OedSchema.from_oed_schema_info``
    first formats into its default schema path as if it were a version. On POSIX that file does not
    exist and it falls back to the path; on Windows the ``C:`` in the middle of the file name raises
    ``OSError`` (EINVAL) rather than ``FileNotFoundError``, so the fallback is never reached and no
    step can reload the config.

    Args:
        oasis_files_dir (Path): directory holding the ``exposure_info.json`` ``GenerateFiles`` wrote.
    """
    config_path = Path(oasis_files_dir, OedExposure.DEFAULT_EXPOSURE_CONFIG_NAME)
    config = json.loads(config_path.read_text())
    schema_file = re.escape(OedSchema.DEFAULT_ODS_SCHEMA_FILE).replace(re.escape('{}'), '(.+)')
    match = re.fullmatch(schema_file, os.path.basename(str(config.get('oed_schema_info') or '')))
    if match:
        config['oed_schema_info'] = match.group(1)
        config_path.write_text(json.dumps(config, indent=2))


def build_kernel_run_dir(run_dir, tools=None):
    """Write a small GUL + IL + RI run directory.

    Args:
        run_dir (Path): directory to create.
        tools (Tools | str | None): tools used to write the fmpy financial structure files.

    Returns:
        Path: run_dir.
    """
    run_dir = Path(run_dir)
    input_dir = run_dir / 'input'
    GenerateDummyModelFiles(
        target_dir=str(run_dir), num_vulnerabilities=N_VULNERABILITIES, num_intensity_bins=N_INTENSITY_BINS,
        num_damage_bins=N_DAMAGE_BINS, num_events=N_EVENTS, num_areaperils=N_AREAPERILS,
        areaperils_per_event=AREAPERILS_PER_EVENT, num_periods=10, random_seed=MODEL_SEED,
    ).run()

    keys_path = input_dir / 'keys.csv'
    GenerateKeysDeterministic(
        oasis_files_dir=str(input_dir), oed_location_csv=str(OED_CASE_DIR / 'location.csv'), keys_data_path=str(keys_path),
        keys_format='oasis', model_perils_covered=[PERIL], exposure_data=oed_case_exposure('location'),
    ).run()
    keys = pd.read_csv(keys_path)
    keys['AreaPerilID'] = np.arange(len(keys)) % N_AREAPERILS + 1
    keys['VulnerabilityID'] = np.arange(len(keys)) % N_VULNERABILITIES + 1
    keys.to_csv(keys_path, index=False)

    GenerateFiles(
        oasis_files_dir=str(input_dir), keys_data_path=str(keys_path),
        oed_location_csv=str(OED_CASE_DIR / 'location.csv'), oed_accounts_csv=str(OED_CASE_DIR / 'account.csv'),
        oed_info_csv=str(OED_CASE_DIR / 'ri_info.csv'), oed_scope_csv=str(OED_CASE_DIR / 'ri_scope.csv'),
        exposure_data=oed_case_exposure(*OED_FILES),
    ).run()
    portable_exposure_config(input_dir)
    prepare_financial_structures(run_dir, tools, DEFAULT_IL_ALLOC_RULE, DEFAULT_RI_ALLOC_RULE, RI_LAYERS_AVAILABLE)
    return run_dir


def report_reference(request, line):
    """Write which code makes the references to the terminal, whatever the capture and verbosity settings."""
    reporter = request.config.pluginmanager.get_plugin('terminalreporter')
    if reporter is not None:
        reporter.write_line(f'orchestrator tests: {line}')


@pytest.fixture(scope='session')
def kernel_run(tmp_path_factory, request):
    """A prepared GUL + IL + RI kernel run directory (30 events, 290 locations, 4 RI inuring priorities).

    Returns:
        KernelRun: the run directory and the parameters its references are made with.

    Raises:
        NoPristineReference: ``OASIS_ORCH_REQUIRE_REF`` is set and ``OASIS_ORCH_REF_TOOLS`` is not.
    """
    ref_tools = os.environ.get(REF_TOOLS_ENV) or None
    if ref_tools is None and reference_required():
        raise NoPristineReference(f'{REQUIRE_REF_ENV} is set but {REF_TOOLS_ENV} is not: the reference streams would be made by '
                                  f'the code under test')
    report_reference(request, f'reference streams made by {ref_tools or describe_source(None)}')
    run_dir = build_kernel_run_dir(tmp_path_factory.mktemp('kernel_run'), resolve_tools(ref_tools))
    event_ids = np.fromfile(run_dir / 'input' / 'events.bin', dtype='<i4')
    return KernelRun(run_dir=run_dir, event_ids=event_ids, ref_tools=ref_tools)


@pytest.fixture(scope='session')
def kernel_run_dir(kernel_run):
    """Path of the prepared kernel run directory, see kernel_run.

    Returns:
        Path: the run directory.
    """
    return kernel_run.run_dir


@pytest.fixture(scope='session')
def kernel_reference(kernel_run, tmp_path_factory):
    """Factory for reference streams of the kernel run, made once per set of arguments in a session.

    Call it as ``kernel_reference(n_chunks, samples=..., ri_layers=..., ...)``; keyword arguments
    default to the KernelRun's and are passed to ``make_reference_streams``. It returns the
    directory holding ``<tap>_P<k>.bin`` for k = 1..n_chunks: gul, il and ri (the net of RI step
    ``ri_layers``, by default 1, as in the bash pipeline the design compares with).

    Returns:
        callable: ``make(n_chunks, **options) -> Path``.
    """
    cache = {}

    def make(n_chunks, **options):
        options = {'samples': kernel_run.samples, 'gul_args': kernel_run.gul_args, 'il_alloc_rule': kernel_run.il_alloc_rule,
                   'ri_alloc_rule': kernel_run.ri_alloc_rule, 'ri_layers': DEFAULT_RI_LAYERS, **options}
        options['gul_args'] = tuple(options['gul_args'])
        key = (n_chunks, tuple(sorted(options.items())))
        if key not in cache:
            out_dir = tmp_path_factory.mktemp(f'kernel_ref_K{n_chunks}')
            make_reference_streams(kernel_run.run_dir, n_chunks, out_dir, tools=kernel_run.ref_tools, **options)
            cache[key] = out_dir
        return cache[key]

    return make


FULL_KERNEL_SAMPLES = 10
FULL_KERNEL_QUANTILES = [0.0, 0.2, 0.5, 0.8, 1.0]
FULL_KERNEL_RETURN_PERIODS = [1, 2, 5, 10]


@pytest.fixture(scope='session')
def full_kernel_run(kernel_run, tmp_path_factory, request):
    """The kernel run prepared for a full-kernel bash run: summary xrefs, every ORD output switched on.

    The pristine ``GenerateLossesDir`` step (from the checkout ``resolve_source`` picks) makes the
    run directory from the kernel run's Oasis files and model data, with
    ``full_ord_settings``: summary set 1 for gul, il and ri, set 2 (by AccNumber) for gul and il,
    the parquet set 3 for il; quantiles and return periods given in the settings. The tool
    wrappers the references run with are in ``bin/``.

    Returns:
        Path: the prepared run directory.
    """
    settings = full_ord_settings(FULL_KERNEL_SAMPLES, quantiles=FULL_KERNEL_QUANTILES, return_periods=FULL_KERNEL_RETURN_PERIODS)
    source = resolve_source()
    report_reference(request, f'full-kernel references made by {describe_source(source)}')
    return prepare_run_dir(kernel_run.run_dir / 'input', kernel_run.run_dir / 'static', settings,
                           tmp_path_factory.mktemp('full_kernel_run'), source=source)


@pytest.fixture(scope='session')
def full_kernel_reference(full_kernel_run, tmp_path_factory):
    """Factory for full-kernel bash references of the kernel run, made once per set of arguments in a session.

    Call it as ``full_kernel_reference(n_chunks, samples=None, **genbash_options)``. It returns a
    reference directory (see ``tools/kernel_reference.py``) holding ``work/`` (every
    ``work/kat/*_P<k>`` and ``work/*_summary_*/P<k>.bin``), ``output/``, ``run_kernel.sh`` and
    ``bash_params.json`` of a K-process run of the pristine kernel script.

    Returns:
        callable: ``make(n_chunks, samples=None, **options) -> Path``.
    """
    cache = {}
    source = resolve_source()

    def make(n_chunks, samples=None, **options):
        key = (n_chunks, samples, tuple(sorted(options.items())))
        if key not in cache:
            out_dir = tmp_path_factory.mktemp(f'full_kernel_ref_K{n_chunks}')
            cache[key] = make_kernel_reference(full_kernel_run, n_chunks, out_dir, samples=samples, source=source, **options)
        return cache[key]

    return make
