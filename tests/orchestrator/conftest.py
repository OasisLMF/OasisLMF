"""Fixtures for the orchestrator tests: a small prepared kernel run directory and its reference streams.

The run directory is built from what is in this repository, in a few seconds:

* the model (``static/`` footprint, vulnerability and damage bins, ``input/events.bin``) comes
  from ``GenerateDummyModelFiles`` with a fixed seed;
* the exposure is the OED of ``validation/reinsurance1`` (290 locations, accounts and a
  reinsurance programme with four inuring priorities, so ``input/RI_1`` ... ``input/RI_4``);
* keys come from ``GenerateKeysDeterministic`` with the area peril and vulnerability ids spread
  over the dummy model's, and ``GenerateFiles`` writes the Oasis input files;
* the fmpy financial structure files are written for the IL step and every RI step.

Reference streams are produced with ``tools/reference_streams.py``: by the tools of the running
interpreter, or by those in the directory named by the ``OASIS_ORCH_REF_TOOLS`` environment
variable (for example wrappers around a pristine checkout of main).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.computation.generate.files import GenerateDummyModelFiles, GenerateFiles
from oasislmf.computation.generate.keys import GenerateKeysDeterministic
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
        keys_format='oasis', model_perils_covered=[PERIL],
    ).run()
    keys = pd.read_csv(keys_path)
    keys['AreaPerilID'] = np.arange(len(keys)) % N_AREAPERILS + 1
    keys['VulnerabilityID'] = np.arange(len(keys)) % N_VULNERABILITIES + 1
    keys.to_csv(keys_path, index=False)

    GenerateFiles(
        oasis_files_dir=str(input_dir), keys_data_path=str(keys_path),
        oed_location_csv=str(OED_CASE_DIR / 'location.csv'), oed_accounts_csv=str(OED_CASE_DIR / 'account.csv'),
        oed_info_csv=str(OED_CASE_DIR / 'ri_info.csv'), oed_scope_csv=str(OED_CASE_DIR / 'ri_scope.csv'),
    ).run()
    prepare_financial_structures(run_dir, tools, DEFAULT_IL_ALLOC_RULE, DEFAULT_RI_ALLOC_RULE, RI_LAYERS_AVAILABLE)
    return run_dir


@pytest.fixture(scope='session')
def kernel_run(tmp_path_factory):
    """A prepared GUL + IL + RI kernel run directory (30 events, 290 locations, 4 RI inuring priorities).

    Returns:
        KernelRun: the run directory and the parameters its references are made with.
    """
    ref_tools = os.environ.get(REF_TOOLS_ENV) or None
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
