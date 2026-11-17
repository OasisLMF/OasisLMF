"""Full-kernel bash reference runs: gul, il and ri with summaries and every ORD output, work files kept.

A reference is what ``oasislmf model generate-losses`` writes for a K-process run, made by the
tools of a pristine checkout of main, with the per-partition work files kept so that an
orchestrated run can be checked byte for byte:

* ``work/kat/<rt>_S<n>_<table>_P<k>`` (eltpy/pltpy binary output, katpy input);
* ``work/<rt>_S<n>_summary_palt/P<k>.bin``, ``..._summary_altmeanonly/P<k>.bin`` and
  ``..._summaryleccalc/P<k>.bin`` (the summary stream as ``tee`` writes it, aalpy/lecpy input);
* ``output/*`` (katpy, aalpy and lecpy output, plus the summary-info files of the run directory).

How a reference is made:

1. **Prepare** (once per model run directory, :func:`prepare_run_dir`): the pristine
   ``GenerateLossesDir`` step turns an Oasis files directory and model data into a run directory
   (``input/`` with the summary xref files, ``static/``, ``output/*summary-info*``), then
   ``fmpy --create-financial-structure-files`` writes the fm structure files for IL and each RI
   step, as the kernel script expects them to exist.
2. **Script** (:func:`write_kernel_script`): the pristine ``bash_params`` builds the parameter
   dict exactly as ``genbash`` does (same arguments as ``runner.run`` gets from
   ``GenerateLosses``), except ``remove_working_files=False`` and ``fifo_tmp_dir=False``. The dict
   is saved as ``bash_params.json`` and ``create_bash_analysis`` + ``create_bash_outputs`` write
   ``run_kernel.sh`` inside ``bash_wrapper``, which is the body of ``genbash``.

   ``genbash`` does not forward ``remove_working_files``, so the script it writes always ends
   with ``rm -R -f work/*``; ``bash_params(remove_working_files=False)`` is the supported switch
   that leaves that line out (``GenerateLossesOutput`` has the same option, misspelt
   ``remove_working_file``, which ``bash_params`` swallows in ``**kwargs``). No script patching is
   needed. ``fifo_tmp_dir=False`` puts the FIFOs in ``fifo/`` of the reference directory, so the
   script text does not depend on a random /tmp name; the FIFOs are removed after the run.
3. **Run** (:func:`run_kernel_script`): ``bash run_kernel.sh`` with a ``PATH`` that starts with a
   directory of wrappers (:func:`write_tool_wrappers`) running every pytools console script, and
   ``oasis_exec_monitor.sh``, from the pristine checkout (``PYTHONPATH=<checkout>``). The editable
   install of the working tree is therefore never used by the reference.

A reference directory holds ``input`` and ``static`` symlinks to the prepared run directory,
``analysis_settings.json``, ``bash_params.json``, ``run_kernel.sh``, ``work/``, ``output/`` and
``log/``; it is itself a runnable run directory.

The pristine checkout is given by ``--source`` or the ``OASIS_ORCH_REF_SOURCE`` environment
variable, else ``~/GitHub/OasisLMF-ref`` when it exists; otherwise (or with the value
``installed``) the tools of the running interpreter's ``oasislmf`` are used, with a warning, as the
references are then made by the code under test. With ``OASIS_ORCH_REQUIRE_REF`` set (CI sets it)
that is an error instead, as is a checkout that is the running ``oasislmf`` itself.

Command line::

    python3 tests/orchestrator/tools/kernel_reference.py prepare OASIS_FILES_DIR MODEL_DATA_DIR SETTINGS_JSON RUN_DIR [--source DIR]
    python3 tests/orchestrator/tools/kernel_reference.py reference RUN_DIR K OUT_DIR [--samples S] [--source DIR] [--bin-dir DIR]
    python3 tests/orchestrator/tools/kernel_reference.py wrappers BIN_DIR [--source DIR]

``_prepare`` and ``_script`` are internal commands run under the pristine ``PYTHONPATH``.
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import oasislmf
from oasislmf.computation.generate.losses import GenerateLossesDir
from oasislmf.execution.bash import bash_params, bash_wrapper, create_bash_analysis, create_bash_outputs

REF_SOURCE_ENV = 'OASIS_ORCH_REF_SOURCE'
REQUIRE_REF_ENV = 'OASIS_ORCH_REQUIRE_REF'
RUNNING_CHECKOUT = Path(oasislmf.__file__).resolve().parents[1]
DEFAULT_REF_SOURCE = Path('~/GitHub/OasisLMF-ref')
INSTALLED = 'installed'
SCRIPT_NAME = 'run_kernel.sh'
BASH_PARAMS_FILE = 'bash_params.json'
SETTINGS_FILE = 'analysis_settings.json'
LINKED_DIRS = ('input', 'static')
IL_ALLOC_RULE = 2
RI_ALLOC_RULE = 3
DF_ENGINE = 'oasis_data_manager.df_reader.reader.OasisPandasReader'
MONITOR_SCRIPT = 'oasis_exec_monitor.sh'

CONSOLE_SCRIPTS = {
    'evepy': 'oasislmf.pytools.eve.cli',
    'gulmc': 'oasislmf.pytools.gulmc.cli',
    'gulpy': 'oasislmf.pytools.gulpy',
    'modelpy': 'oasislmf.pytools.modelpy',
    'fmpy': 'oasislmf.pytools.fm.cli',
    'plapy': 'oasislmf.pytools.pla.cli',
    'summarypy': 'oasislmf.pytools.summary.cli',
    'eltpy': 'oasislmf.pytools.elt.cli',
    'pltpy': 'oasislmf.pytools.plt.cli',
    'katpy': 'oasislmf.pytools.kat.cli',
    'aalpy': 'oasislmf.pytools.aal.cli',
    'lecpy': 'oasislmf.pytools.lec.cli',
    'join-summary-info': 'oasislmf.pytools.join_summary_info.cli',
    'servedata': 'oasislmf.pytools.data_layer.footprint_layer',
}

EPT_OUTPUTS = ('ept_full_uncertainty_aep', 'ept_full_uncertainty_oep', 'ept_mean_sample_aep', 'ept_mean_sample_oep',
               'ept_per_sample_mean_aep', 'ept_per_sample_mean_oep', 'psept_aep', 'psept_oep')
ELT_PLT_OUTPUTS = ('elt_sample', 'elt_quantile', 'elt_moment', 'plt_sample', 'plt_quantile', 'plt_moment')
ALL_ORD_OUTPUTS = {**{name: True for name in ELT_PLT_OUTPUTS + EPT_OUTPUTS},
                   'alt_period': True, 'alt_meanonly': True, 'alct_convergence': True, 'alct_confidence': 0.95,
                   'return_period_file': True}

GENBASH_DEFAULTS = dict(
    gul_alloc_rule=None, il_alloc_rule=None, ri_alloc_rule=None, num_gul_per_lb=None, num_fm_per_lb=None,
    stderr_guard=True, bash_trace=False, gulmc=True, gul_random_generator=2, gulmc_effective_damageability=False,
    gulmc_vuln_cache_size=200, fmpy_low_memory=False, fmpy_sort_output=False, summarypy_low_memory=False,
    event_shuffle=None, model_py_server=False, peril_filter=[], join_summary_info=False, model_df_engine=DF_ENGINE,
    dynamic_footprint=False, log_level=None,
)


class KernelRunError(RuntimeError):
    """The kernel script, or a pristine helper process, failed."""


class NoPristineReference(RuntimeError):
    """A pristine reference is required (``OASIS_ORCH_REQUIRE_REF``) and none is given."""


def reference_required():
    """Whether ``OASIS_ORCH_REQUIRE_REF`` is set to anything but empty, ``0``, ``false`` or ``no``."""
    return os.environ.get(REQUIRE_REF_ENV, '').strip().lower() not in ('', '0', 'false', 'no')


def _not_pristine(reason):
    message = f'{reason}: the references would be made by the code under test, so it would be compared with itself'
    if reference_required():
        raise NoPristineReference(f'{message} ({REQUIRE_REF_ENV} is set; give a checkout of main in {REF_SOURCE_ENV})')
    warnings.warn(message, stacklevel=3)


def resolve_source(source=None):
    """The pristine checkout to run.

    In order: ``source``; the ``OASIS_ORCH_REF_SOURCE`` environment variable (the value
    ``installed`` means the running interpreter's ``oasislmf``); ``DEFAULT_REF_SOURCE`` when it is
    an OasisLMF checkout; otherwise the running interpreter's ``oasislmf``, with a warning.

    Args:
        source (str | Path | None): checkout directory, or None to look further.

    Returns:
        Path | None: the resolved checkout, None for the running interpreter's ``oasislmf``.

    Raises:
        FileNotFoundError: if an explicitly given directory is not an OasisLMF checkout.
        NoPristineReference: with ``OASIS_ORCH_REQUIRE_REF`` set, if the result would be the running ``oasislmf``.
    """
    source = source or os.environ.get(REF_SOURCE_ENV) or None
    if source == INSTALLED:
        if reference_required():
            _not_pristine(f'{REF_SOURCE_ENV}={INSTALLED}')
        return None
    if source is None:
        default = DEFAULT_REF_SOURCE.expanduser()
        if (default / 'oasislmf' / '__init__.py').exists():
            return default.resolve()
        _not_pristine(f'no pristine checkout ({REF_SOURCE_ENV} is not set and {default} is not an OasisLMF checkout)')
        return None
    source = Path(source).expanduser().resolve()
    if not (source / 'oasislmf' / '__init__.py').exists():
        raise FileNotFoundError(f'{source} is not an OasisLMF checkout')
    if source == RUNNING_CHECKOUT:
        _not_pristine(f'the reference checkout {source} is the one the running oasislmf is imported from')
    return source


def describe_source(source):
    """Which code a reference is made by, for logs and test reports.

    Args:
        source (Path | None): a ``resolve_source`` result.

    Returns:
        str: the checkout and its commit, or the running ``oasislmf``'s location.
    """
    if source is None:
        return (f'installed oasislmf {oasislmf.__version__} at {RUNNING_CHECKOUT} (the code under test: same-checkout '
                f'comparison, orchestrator against bash of this checkout, not of the base branch)')
    proc = subprocess.run(['git', '-C', str(source), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
    commit = proc.stdout.strip() if proc.returncode == 0 else 'unknown commit'
    return f'pristine checkout {source} ({commit})'


def source_env(source=None, env=None):
    """Environment that makes ``python3`` import ``oasislmf`` from the checkout.

    Args:
        source (Path | None): resolved checkout, None to leave ``PYTHONPATH`` as it is.
        env (dict | None): base environment, the current one if None.

    Returns:
        dict: the environment.
    """
    env = dict(os.environ if env is None else env)
    if source is not None:
        env['PYTHONPATH'] = str(source)
    return env


def tool_command(name, python=sys.executable):
    """The command running a pytools console script with an interpreter, as its wrapper does, without bash.

    Args:
        name (str): console script name, a key of ``CONSOLE_SCRIPTS``.
        python (str): interpreter to run it with.

    Returns:
        list[str]: the command; run it with ``source_env(source)`` to use a checkout.
    """
    return [python, '-c', f"import sys; from {CONSOLE_SCRIPTS[name]} import main; sys.argv[0] = '{name}'; sys.exit(main())"]


def link_dir(link, target):
    """Make ``link`` a symbolic link to the directory ``target``, or a copy of it where links cannot be made.

    Windows allows symbolic links only with a privilege or developer mode.
    """
    try:
        Path(link).symlink_to(os.path.realpath(target), target_is_directory=True)
    except OSError:
        shutil.copytree(target, link)


def write_tool_wrappers(bin_dir, source=None, python=sys.executable):
    """Write executables running every pytools console script (and the exec monitor) from a checkout.

    Each wrapper is ``PYTHONPATH=<source> exec <python> -c "from <module> import main; ..."``,
    the same as the console script entry point, so a ``PATH`` starting with ``bin_dir`` makes a
    kernel script run the checkout's code whatever is installed.

    Args:
        bin_dir (str | Path): directory to write, created if needed.
        source (str | Path | None): pristine checkout, see resolve_source.
        python (str): interpreter the wrappers run.

    Returns:
        Path: bin_dir.
    """
    source = resolve_source(source)
    bin_dir = Path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'PYTHONPATH={source} ' if source is not None else ''
    for name, module in CONSOLE_SCRIPTS.items():
        code = f"import sys; from {module} import main; sys.argv[0] = '{name}'; sys.exit(main())"
        path = bin_dir / name
        path.write_text(f'#!/bin/bash\n# Runs {name} from {source or "the installed oasislmf"}.\n{prefix}exec {python} -c "{code}" "$@"\n')
        path.chmod(0o755)
    if source is not None:
        monitor = source / 'bin' / MONITOR_SCRIPT
    else:
        monitor = shutil.which(MONITOR_SCRIPT) or Path(oasislmf.__file__).resolve().parents[1] / 'bin' / MONITOR_SCRIPT
    if not Path(monitor).exists():
        raise FileNotFoundError(f'{MONITOR_SCRIPT} not found')
    shutil.copy2(monitor, bin_dir / MONITOR_SCRIPT)
    return bin_dir


def tools_env(bin_dir, env=None):
    """Environment for a kernel script run with the wrappers of ``bin_dir`` first on ``PATH``.

    Args:
        bin_dir (str | Path): wrapper directory.
        env (dict | None): base environment, the current one if None.

    Returns:
        dict: the environment; ``PYTHONPATH`` is removed so only the wrappers choose the code.
    """
    env = dict(os.environ if env is None else env)
    env.pop('PYTHONPATH', None)
    env['PATH'] = os.pathsep.join([str(Path(bin_dir).resolve()), env.get('PATH', '')])
    return env


def full_ord_summary(summary_id, oed_fields=None, outputs=None, **extra):
    """A summary set with every ORD output (or the given ones) switched on.

    Args:
        summary_id (int): summary set id.
        oed_fields (list[str] | None): OED fields to group by; the whole portfolio if None.
        outputs (iterable[str] | None): ORD switches to set, all of ALL_ORD_OUTPUTS if None.
        **extra: further ``ord_output`` entries (for example ``parquet_format=True``).

    Returns:
        dict: the summaries entry.
    """
    ord_output = dict(ALL_ORD_OUTPUTS) if outputs is None else {name: ALL_ORD_OUTPUTS[name] for name in outputs}
    ord_output.update(extra)
    summary = {'id': summary_id, 'ord_output': ord_output}
    if oed_fields:
        summary['oed_fields'] = list(oed_fields)
    return summary


def full_ord_settings(samples, second_set_fields=('AccNumber',), second_set_run_types=('gul', 'il'), second_set_outputs=None,
                      parquet_set_run_types=('il',), model_settings=None, **extra):
    """Analysis settings for gul, il and ri with every ORD output the pytools bash path supports.

    Summary set 1 of every run type is the whole portfolio with every output; run types in
    ``second_set_run_types`` get a set 2 grouped by ``second_set_fields``; run types in
    ``parquet_set_run_types`` get a set 3, the whole portfolio with every output written as
    parquet (``parquet_format``) instead of CSV.

    Args:
        samples (int): number_of_samples.
        second_set_fields (iterable[str]): OED fields of summary set 2.
        second_set_run_types (iterable[str]): run types that get summary set 2.
        second_set_outputs (iterable[str] | None): ORD switches of set 2, all if None.
        parquet_set_run_types (iterable[str]): run types that get the parquet summary set 3.
        model_settings (dict | None): model_settings section.
        **extra: further top level settings (``quantiles``, ``return_periods`` ...).

    Returns:
        dict: the analysis settings.
    """
    settings = {'version': '3', 'analysis_tag': 'orchestrator_reference', 'source_tag': 'orchestrator', 'model_name_id': 'reference',
                'model_supplier_id': 'OasisLMF', 'number_of_samples': samples, 'gul_threshold': 0,
                'model_settings': dict(model_settings or {})}
    for run_type in ('gul', 'il', 'ri'):
        summaries = [full_ord_summary(1)]
        if run_type in second_set_run_types:
            summaries.append(full_ord_summary(2, second_set_fields, second_set_outputs))
        if run_type in parquet_set_run_types:
            summaries.append(full_ord_summary(3, parquet_format=True))
        settings[f'{run_type}_output'] = True
        settings[f'{run_type}_summaries'] = summaries
    settings.update(extra)
    return settings


def num_ri_layers(run_dir, analysis_settings):
    """RI iterations, as ``GenerateLosses._get_num_ri_layers`` counts them.

    Args:
        run_dir (str | Path): prepared run directory.
        analysis_settings (dict): the settings.

    Returns:
        int: number of entries of ``input/ri_layers.json`` when ri or rl output is on, else 0.
    """
    if not (analysis_settings.get('ri_output') or analysis_settings.get('rl_output')):
        return 0
    with open(Path(run_dir) / 'input' / 'ri_layers.json') as f:
        return len(json.load(f))


def _run_helper(command, args, source, cwd=None):
    """Run this module's ``command`` with ``args`` (a JSON-able dict) under the checkout's ``PYTHONPATH``."""
    proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), command, json.dumps(args)], cwd=cwd,
                          env=source_env(source), capture_output=True, text=True)
    if proc.returncode != 0:
        raise KernelRunError(f'{command} failed ({proc.returncode}):\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}')
    return proc


def _prepare_in_process(oasis_files_dir, model_data_dir, settings_json, run_dir, il_alloc_rule, ri_alloc_rule):
    """Body of the ``_prepare`` helper: the pristine ``GenerateLossesDir`` step."""
    GenerateLossesDir(oasis_files_dir=oasis_files_dir, model_data_dir=model_data_dir, analysis_settings_json=settings_json,
                      model_run_dir=run_dir, copy_model_data=True, kernel_alloc_rule_il=il_alloc_rule,
                      kernel_alloc_rule_ri=ri_alloc_rule, check_oed=False).run()


def prepare_run_dir(oasis_files_dir, model_data_dir, analysis_settings, run_dir, source=None, bin_dir=None,
                    il_alloc_rule=IL_ALLOC_RULE, ri_alloc_rule=RI_ALLOC_RULE):
    """Make a run directory with the pristine ``GenerateLossesDir`` step and write the fm structure files.

    Args:
        oasis_files_dir (str | Path): Oasis files (``GenerateFiles`` output).
        model_data_dir (str | Path): model data (footprint, vulnerability, damage bins, event sets).
        analysis_settings (dict): settings the run directory is made for (summary sets decide
            the summary xref files; ORD outputs decide occurrence, quantile and return period files).
        run_dir (str | Path): directory to create.
        source (str | Path | None): pristine checkout, see resolve_source.
        bin_dir (str | Path | None): wrappers from write_tool_wrappers, written to ``run_dir/bin`` if None, but not
            on Windows, which cannot run them (``make_kernel_reference`` writes them when they are missing).
        il_alloc_rule (int): IL allocation rule of the fm structure files.
        ri_alloc_rule (int): RI allocation rule of the fm structure files.

    Returns:
        Path: run_dir.
    """
    source = resolve_source(source)
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    settings_json = run_dir / SETTINGS_FILE
    settings_json.write_text(json.dumps(analysis_settings, indent=4))
    _run_helper('_prepare', dict(oasis_files_dir=str(Path(oasis_files_dir).resolve()), model_data_dir=str(Path(model_data_dir).resolve()),
                                 settings_json=str(settings_json), run_dir=str(run_dir), il_alloc_rule=il_alloc_rule,
                                 ri_alloc_rule=ri_alloc_rule), source)
    if bin_dir is None and sys.platform != 'win32':
        bin_dir = write_tool_wrappers(run_dir / 'bin', source)
    steps = [('input', il_alloc_rule)] + [(f'input/RI_{i}', ri_alloc_rule)
                                          for i in range(1, num_ri_layers(run_dir, analysis_settings) + 1)]
    for static_path, rule in steps:
        proc = subprocess.run(tool_command('fmpy') + [f'-a{rule}', '--create-financial-structure-files', '-p', static_path],
                              cwd=run_dir, env=source_env(source), capture_output=True, text=True)
        if proc.returncode != 0:
            raise KernelRunError(f'fmpy structure files for {static_path} failed:\n{proc.stderr[-4000:]}')
    return run_dir


def _json_default(value):
    if isinstance(value, Counter):
        return dict(value)
    if callable(value):
        return None
    return str(value)


def _script_in_process(ref_dir, analysis_settings, n_chunks, num_reinsurance_iterations, options):
    """Body of the ``_script`` helper: ``genbash`` with ``remove_working_files=False``; writes the script and bash_params.json."""
    os.chdir(ref_dir)
    kwargs = {**GENBASH_DEFAULTS, **options}
    params = bash_params(
        analysis_settings=analysis_settings, max_process_id=n_chunks, num_reinsurance_iterations=num_reinsurance_iterations,
        fifo_tmp_dir=False, filename=SCRIPT_NAME, remove_working_files=False, **kwargs)
    params['analysis_pk'] = None
    params['socket_server_size'] = None
    params['socket_server_port'] = None
    snapshot = copy.deepcopy(params)
    if os.path.exists(SCRIPT_NAME):
        os.remove(SCRIPT_NAME)
    with bash_wrapper(SCRIPT_NAME, params['bash_trace'], params['stderr_guard']):
        create_bash_analysis(**params)
        create_bash_outputs(**params)
    with open(BASH_PARAMS_FILE, 'w') as f:
        json.dump(snapshot, f, indent=2, sort_keys=True, default=_json_default)
        f.write('\n')


def write_kernel_script(ref_dir, analysis_settings, n_chunks, num_reinsurance_iterations, source=None, **options):
    """Write ``run_kernel.sh`` and ``bash_params.json`` into ``ref_dir`` with the pristine ``bash_params``.

    Args:
        ref_dir (str | Path): the reference directory (cwd of the script).
        analysis_settings (dict): the settings.
        n_chunks (int): number of processes K (``max_process_id``).
        num_reinsurance_iterations (int): RI iterations.
        source (str | Path | None): pristine checkout, see resolve_source.
        **options: genbash keyword arguments overriding GENBASH_DEFAULTS.

    Returns:
        dict: the bash_params, as saved in ``bash_params.json``.
    """
    ref_dir = Path(ref_dir).resolve()
    _run_helper('_script', dict(ref_dir=str(ref_dir), analysis_settings=analysis_settings, n_chunks=n_chunks,
                                num_reinsurance_iterations=num_reinsurance_iterations, options=options), resolve_source(source))
    with open(ref_dir / BASH_PARAMS_FILE) as f:
        return json.load(f)


def run_kernel_script(ref_dir, bin_dir, timeout=None):
    """Run ``run_kernel.sh`` in ``ref_dir`` with the wrappers of ``bin_dir`` and remove the FIFOs afterwards.

    Args:
        ref_dir (str | Path): reference directory holding the script.
        bin_dir (str | Path): wrappers from write_tool_wrappers.
        timeout (float | None): seconds before the run is abandoned.

    Raises:
        KernelRunError: if the script exits non-zero, with its output and the end of ``log/stderror.err``.
    """
    ref_dir = Path(ref_dir)
    proc = subprocess.run(['bash', SCRIPT_NAME], cwd=ref_dir, env=tools_env(bin_dir), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        err = ref_dir / 'log' / 'stderror.err'
        tail = err.read_text(errors='replace')[-4000:] if err.exists() else ''
        raise KernelRunError(f'{ref_dir / SCRIPT_NAME} exited {proc.returncode}:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}\n{tail}')
    shutil.rmtree(ref_dir / 'fifo', ignore_errors=True)


def make_kernel_reference(run_dir, n_chunks, out_dir, samples=None, source=None, bin_dir=None, analysis_settings=None,
                          timeout=None, **options):
    """Run the whole kernel on a prepared run directory with K processes, keeping work files.

    Args:
        run_dir (str | Path): run directory from prepare_run_dir.
        n_chunks (int): number of processes K.
        out_dir (str | Path): reference directory to create (must not hold an earlier reference).
        samples (int | None): number_of_samples, the run directory's settings' if None.
        source (str | Path | None): pristine checkout, see resolve_source.
        bin_dir (str | Path | None): wrappers, ``run_dir/bin`` if None.
        analysis_settings (dict | None): settings, ``run_dir/analysis_settings.json`` if None.
        timeout (float | None): seconds before the kernel run is abandoned.
        **options: genbash keyword arguments overriding GENBASH_DEFAULTS.

    Returns:
        Path: out_dir, holding ``work/``, ``output/``, ``log/``, ``run_kernel.sh``, ``bash_params.json``,
            ``analysis_settings.json`` and the ``input``/``static`` links.
    """
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    if analysis_settings is None:
        analysis_settings = json.loads((run_dir / SETTINGS_FILE).read_text())
    analysis_settings = copy.deepcopy(analysis_settings)
    if samples is not None:
        analysis_settings['number_of_samples'] = samples
    bin_dir = Path(bin_dir) if bin_dir is not None else run_dir / 'bin'
    if not (bin_dir / 'fmpy').exists():
        write_tool_wrappers(bin_dir, source)
    if (out_dir / 'work').exists() or (out_dir / 'output').exists():
        raise FileExistsError(f'{out_dir} already holds a reference')
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in LINKED_DIRS:
        link = out_dir / name
        if not link.exists():
            link.symlink_to(run_dir / name, target_is_directory=True)
    (out_dir / 'output').mkdir()
    (out_dir / 'fifo').mkdir(exist_ok=True)
    for path in sorted((run_dir / 'output').glob('*summary-info*')):
        shutil.copy2(path, out_dir / 'output' / path.name)
    (out_dir / SETTINGS_FILE).write_text(json.dumps(analysis_settings, indent=4))
    write_kernel_script(out_dir, analysis_settings, n_chunks, num_ri_layers(run_dir, analysis_settings), source, **options)
    run_kernel_script(out_dir, bin_dir, timeout=timeout)
    return out_dir


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ('_prepare', '_script'):
        args = json.loads(argv[1])
        (_prepare_in_process if argv[0] == '_prepare' else _script_in_process)(**args)
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='make a run directory from Oasis files and model data')
    prep.add_argument('oasis_files_dir')
    prep.add_argument('model_data_dir')
    prep.add_argument('settings_json', help='analysis settings (summary sets and ORD outputs)')
    prep.add_argument('run_dir')
    ref = sub.add_parser('reference', help='run the kernel with K processes into OUT_DIR')
    ref.add_argument('run_dir')
    ref.add_argument('n_chunks', type=int)
    ref.add_argument('out_dir')
    ref.add_argument('--samples', '-S', type=int, default=None)
    ref.add_argument('--bin-dir', default=None, help='tool wrappers (default RUN_DIR/bin, written if missing)')
    wrap = sub.add_parser('wrappers', help='write the pristine tool wrappers')
    wrap.add_argument('bin_dir')
    for p in (prep, ref, wrap):
        p.add_argument('--source', default=None, help=f'pristine checkout (default ${REF_SOURCE_ENV}, else the installed oasislmf)')
    args = parser.parse_args(argv)
    start = time.perf_counter()
    print(f'reference tools: {describe_source(resolve_source(args.source))}')
    if args.command == 'prepare':
        settings = json.loads(Path(args.settings_json).read_text())
        prepare_run_dir(args.oasis_files_dir, args.model_data_dir, settings, args.run_dir, source=args.source)
        print(f'prepared {args.run_dir} in {time.perf_counter() - start:.1f}s')
    elif args.command == 'reference':
        out = make_kernel_reference(args.run_dir, args.n_chunks, args.out_dir, samples=args.samples, source=args.source,
                                    bin_dir=args.bin_dir)
        n_files = sum(1 for sub_dir in ('work', 'output') for p in (out / sub_dir).rglob('*') if p.is_file())
        print(f'wrote {out} ({n_files} work/output files) in {time.perf_counter() - start:.1f}s')
    else:
        write_tool_wrappers(args.bin_dir, args.source)
    return 0


if __name__ == '__main__':
    sys.exit(main())
