"""The planner: the stage tree of a kernel run, built from the ``bash_params`` dict that ``genbash`` consumes.

``plan_from_bash_params`` reproduces what the generated bash script runs, partition by partition, as a ``ChainSpec``
whose stages are the in-process equivalents of the commands (``gulmc``, or ``modelpy | gulpy`` as one stage, ``fmpy``
per IL/RI step, one ``summarypy`` per run type and inuring priority, ``pltpy``/``eltpy`` per summary set, and the
``tee`` work files), plus the ``katpy``/``aalpy``/``lecpy`` commands bash runs once every partition is done, as ``OutputJob``s.

Every per-chunk path is a template holding ``{k}`` and is exactly the name bash uses for partition k, so the work
directory of an orchestrated run is interchangeable with a bash one. ``render_chunk_commands`` and
``OutputJob.command`` turn a plan back into the bash commands it stands for; the tests compare them with the
script ``genbash`` writes, which is what ties the planner to ``genbash``.

In a distributed run (``process_number`` k of ``max_process_id``) bash runs partition k alone, writing its work files in
``k.work/`` (the ``work_dir`` of bash_params); the plan's templates name the same files and ``Plan.chunk_numbers`` is
``[k]``. Such a run does not run the output jobs: the outputs step plans from its own bash_params, without
``process_number``, so its jobs read ``work/``, where the platform gathers the work files of every chunk.

Settings the stages do not cover raise ``Unsupported`` with the reason, and the run falls back to bash.
"""
import csv
import glob
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field

from oasislmf.execution.bash import (ORD_ALT_MEANONLY_OUTPUT_SWITCHES, ORD_ALT_OUTPUT_SWITCHES,
                                     ORD_EPT_OUTPUT_SWITCHES, ORD_LECCALC, ORD_PSEPT_OUTPUT_SWITCHES, OUTPUT_SWITCHES,
                                     INTERMEDIATE_INURING_PRIORITY_PREFIX, get_reinsurance_inuring_priorities, ord_enabled)
from oasislmf.pytools.getmodel.common import SECTION_MERGE_DEFAULT
from oasislmf.utils.defaults import (EVE_DEFAULT_SHUFFLE, EVE_FISHER_YATES, EVE_NO_SHUFFLE, EVE_ROUND_ROBIN,
                                     EVE_STD_SHUFFLE)

from .chain import CHUNK_KEY, ChainSpec, chunk_path_templates, fill_chunk
from .chunks import make_chunks

__all__ = ['Unsupported', 'OutputJob', 'Plan', 'plan_from_bash_params', 'render_chunk_commands', 'fill_chunk']

logger = logging.getLogger(__name__)

DEFAULT_VULN_CACHE_MB = 200
EVE_FLAGS = {EVE_NO_SHUFFLE: '-n ', EVE_ROUND_ROBIN: '', EVE_FISHER_YATES: '-r ', EVE_STD_SHUFFLE: '-R '}

ORD_TABLES = {
    ord_type: [(table['table_name'], table['csv_flag'], ord_table) for ord_table, table in switches.items()]
    for ord_type, switches in OUTPUT_SWITCHES.items()
}
"""ord_type -> [(table name, eltpy/pltpy flag, ord_output key)], in the order bash writes the flags."""

ORD_KIND = {ord_type: switches[next(iter(switches))]['executable'][:3] for ord_type, switches in OUTPUT_SWITCHES.items()}
"""ord_type -> stage kind, 'plt' or 'elt'."""

KAT_FLAGS = {table['table_name']: table['kat_flag'] for switches in OUTPUT_SWITCHES.values() for table in switches.values()}

LEC_KWARGS = {
    'ept_full_uncertainty_aep': 'agg_full_uncertainty',
    'ept_full_uncertainty_oep': 'occ_full_uncertainty',
    'ept_mean_sample_aep': 'agg_sample_mean',
    'ept_mean_sample_oep': 'occ_sample_mean',
    'ept_per_sample_mean_aep': 'agg_wheatsheaf_mean',
    'ept_per_sample_mean_oep': 'occ_wheatsheaf_mean',
    'psept_aep': 'agg_wheatsheaf',
    'psept_oep': 'occ_wheatsheaf',
}
"""analysis settings ord_output option -> lecpy ``main`` keyword argument."""

LEGACY_SUMMARY_OUTPUTS = ('eltcalc', 'pltcalc', 'summarycalc', 'aalcalc', 'aalcalcmeanonly', 'lec_output', 'leccalc')

OUTPUT_TOOLS = {
    'katpy': 'oasislmf.pytools.kat.manager:main',
    'aalpy': 'oasislmf.pytools.aal.manager:main',
    'lecpy': 'oasislmf.pytools.lec.manager:main',
}
"""Output tool -> the ``'module:function'`` an ``OutputJob``'s kwargs are passed to."""

OUTPUT_KWARGS = {'katpy': ('out',), 'aalpy': ('aal', 'alct'), 'lecpy': ('ept', 'psept')}
"""tool -> the keyword arguments that name the files it writes."""


class Unsupported(Exception):
    """The run needs something the orchestrator does not cover; it must run with bash.

    Attributes:
        reason (str): why.
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass
class OutputRename:
    """Output files bash renames, or copies, once every output job is done (``do_ri_final_output``).

    Attributes:
        src (str): the prefix of the files to rename, relative to the run directory.
        dst (str): the prefix that replaces it.
        copy (bool): copy the files, keeping the originals, instead of renaming them.
    """
    src: str
    dst: str
    copy: bool = False

    def apply(self, run_dir):
        """Rename (or copy) every file of ``run_dir`` whose path starts with ``src`` to the same path starting with ``dst``."""
        src = os.path.join(run_dir, self.src)
        dst = os.path.join(run_dir, self.dst)
        for path in sorted(glob.glob(glob.escape(src) + '*')):
            target = dst + path[len(src):]
            if self.copy:
                shutil.copyfile(path, target)
            else:
                os.replace(path, target)


@dataclass
class OutputJob:
    """One command bash runs after every partition is done.

    Attributes:
        tool (str): ``katpy``, ``aalpy`` or ``lecpy``.
        kwargs (dict): keyword arguments of the tool's ``main`` (``OUTPUT_TOOLS``), i.e. its command-line options
            by ``dest``. Paths are as bash writes them, relative to the run directory unless bash made them absolute.
        summary_info (str | None): with join_summary_info, the summary-info file ``join-summary-info`` joins onto
            every file the job writes once the tool is done; None otherwise.
    """
    tool: str
    kwargs: dict = field(default_factory=dict)
    summary_info: str = None

    @property
    def entry_point(self):
        """str: ``'module:function'`` to call with ``kwargs``."""
        return OUTPUT_TOOLS[self.tool]

    def command(self):
        """The command bash runs for this job, as a list of arguments in bash's order."""
        kw = self.kwargs
        if self.tool == 'katpy':
            argv = ['katpy'] + ([] if not kw.get('unsorted') else ['-u'])
            argv += [KAT_FLAGS[t] for t in KAT_FLAGS if kw.get(t)]
            return argv + ['-f', kw['file_type'], '-i', *kw['files_in'], '-o', kw['out']]
        if self.tool == 'aalpy':
            argv = ['aalpy', f"-K{kw['subfolder']}"] + (['-M'] if kw.get('meanonly') else [])
            if kw.get('alct'):
                argv += ['-c', kw['alct']]
                if 'confidence' in kw:
                    argv += ['-l', str(kw['confidence'])]
            if kw.get('ext', 'csv') != 'csv':
                argv += ['-E', kw['ext']]
            return argv + ['-a', kw['aal']]
        if self.tool == 'lecpy':
            argv = ['lecpy'] + (['-r'] if kw.get('use_return_period') else []) + [f"-K{kw['subfolder']}"]
            switches = {**ORD_EPT_OUTPUT_SWITCHES, **ORD_PSEPT_OUTPUT_SWITCHES}
            argv += [switches[option] for option in sorted(LEC_KWARGS) if kw.get(LEC_KWARGS[option])]
            if kw.get('ext', 'csv') != 'csv':
                argv += ['-E', kw['ext']]
            if kw.get('ept'):
                argv += ['-O', kw['ept']]
            if kw.get('psept'):
                argv += ['-o', kw['psept']]
            return argv
        raise ValueError(f'unknown output tool {self.tool!r}')

    def outputs(self):
        """The files the job writes, as the plan names them, in the order bash post-processes them."""
        return [self.kwargs[key] for key in OUTPUT_KWARGS.get(self.tool, ()) if self.kwargs.get(key)]

    def join_commands(self):
        """The ``join-summary-info`` commands bash runs on this job's files after every job is done."""
        if not self.summary_info:
            return []
        return [['join-summary-info', '-s', self.summary_info, '-d', path, '-o', path] for path in self.outputs()]

    def to_dict(self):
        """Return the job as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build a job from the output of ``to_dict``."""
        return cls(tool=data['tool'], kwargs=dict(data['kwargs']), summary_info=data.get('summary_info'))


@dataclass
class Plan:
    """What a kernel run does, as the orchestrator runs it.

    Attributes:
        spec (ChainSpec): the stage tree, with its per-chunk path templates.
        output_jobs (list[OutputJob]): the katpy/aalpy/lecpy runs after the last chunk, in any order.
        n_chunks (int): the partition count bash would use (``max_process_id``).
        event_shuffle (int): evepy partitioning rule (``EVE_*`` of ``oasislmf.utils.defaults``).
        process_number (int | None): in a distributed run, the one partition this run computes (bash's
            ``process_number``); its work files are in ``<process_number>.work/`` as bash names them. None for all.
        output_renames (list[OutputRename]): the output files renamed or copied once every output job is done.
    """
    spec: ChainSpec
    output_jobs: list = field(default_factory=list)
    n_chunks: int = 1
    event_shuffle: int = EVE_DEFAULT_SHUFFLE
    process_number: int = None
    output_renames: list = field(default_factory=list)

    @property
    def chunk_numbers(self):
        """list[int]: the 1-based numbers of the chunks this run computes: ``process_number`` alone, or 1 to ``n_chunks``."""
        if self.process_number is not None:
            return [self.process_number]
        return list(range(1, self.n_chunks + 1))

    def make_chunks(self, events, n_chunks=None):
        """Split events into chunks as ``evepy`` partitions them with this plan's shuffle rule.

        Args:
            events (array-like): event ids, in the order they are held in ``events.bin``.
            n_chunks (int, optional): number of chunks; ``n_chunks`` of the plan by default.

        Returns:
            list[np.ndarray]: element ``k - 1`` holds the events ``evepy k n_chunks`` emits, in the same order.
        """
        return make_chunks(events, self.n_chunks if n_chunks is None else n_chunks, self.event_shuffle)

    def chunk_paths(self, k):
        """Every file chunk k writes (its work and kat files), relative to the run directory as bash names them."""
        return [fill_chunk(template, k) for template in chunk_path_templates(self.spec.summaries)]

    def work_dirs(self):
        """The directories the chunks write into, sorted."""
        return sorted({os.path.dirname(p) for p in self.chunk_paths(1)})

    def to_dict(self):
        """Return the plan as a plain JSON-serialisable dict."""
        return {'spec': self.spec.to_dict(), 'output_jobs': [job.to_dict() for job in self.output_jobs],
                'n_chunks': self.n_chunks, 'event_shuffle': self.event_shuffle, 'process_number': self.process_number,
                'output_renames': [asdict(rename) for rename in self.output_renames]}

    @classmethod
    def from_dict(cls, data):
        """Build a plan from the output of ``to_dict``."""
        return cls(spec=ChainSpec.from_dict(data['spec']), output_jobs=[OutputJob.from_dict(j) for j in data['output_jobs']],
                   n_chunks=data['n_chunks'], event_shuffle=data['event_shuffle'], process_number=data.get('process_number'),
                   output_renames=[OutputRename(**rename) for rename in data.get('output_renames', [])])

    def to_json(self):
        """Return the plan as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Build a plan from the output of ``to_json``."""
        return cls.from_dict(json.loads(text))


def _check_supported(bash_params, analysis_settings, run_dir):
    """Raise ``Unsupported`` for every setting that changes the pipeline in a way the stages do not reproduce.

    Settings that only change how bash spreads the work, not the losses, are ignored with a log message: the
    load balancer between gul and fm (the pool already hands chunks to whichever worker is free) and the model
    data server (each worker reads the footprint itself). Legacy (non-ORD) output flags are ignored as genbash
    ignores them: their summary sets are computed and discarded.
    """
    if bash_params.get('_get_getmodel_cmd'):
        raise Unsupported('a custom gulcalc (model supplier binary or custom getmodel command) produces the ground up loss')
    if bash_params.get('full_correlation'):
        raise Unsupported('full correlation output (genbash reads FIFOs nothing writes, issue #2311)')
    if bash_params.get('custom_args'):
        raise Unsupported(f"custom_args {sorted(bash_params['custom_args'])} are passed to the getmodel command")
    if bash_params.get('rl_output'):
        raise Unsupported("rl output (the reinsurer's loss is combined across inuring priorities by rlcombine, "
                          "which the orchestrator does not run)")
    event_shuffle = bash_params.get('event_shuffle', EVE_DEFAULT_SHUFFLE)
    if event_shuffle not in EVE_FLAGS:
        raise Unsupported(f'unknown event shuffle rule {event_shuffle!r}')
    if bash_params.get('model_py_server'):
        logger.info('The orchestrator ignores model_py_server: each worker reads the footprint without the data server')
    il_or_ri = bash_params.get('il_output') or bash_params.get('ri_output')
    distributed = bash_params.get('process_number') is not None
    if bash_params.get('num_gul_per_lb') and bash_params.get('num_fm_per_lb') and il_or_ri and not distributed:
        logger.info('The orchestrator ignores the load balancer (num_gul_per_lb / num_fm_per_lb): '
                    'its workers take the next chunk when they are free')
    for runtype in ('gul', 'il', 'ri'):
        if not bash_params.get(f'{runtype}_output'):
            continue
        for summary in analysis_settings.get(f'{runtype}_summaries', []):
            legacy = [key for key in LEGACY_SUMMARY_OUTPUTS if summary.get(key)]
            if legacy:
                logger.info(f"Legacy (non-ORD) outputs {legacy} in {runtype}_summaries id {summary.get('id')} are not produced, "
                            "as with the bash script")


def _gulmc_kwargs(bash_params):
    """``GulmcStage`` keyword arguments of the ``gulmc`` command ``get_getmodel_cmd`` writes."""
    return {
        'sample_size': int(bash_params['number_of_samples']),
        'loss_threshold': float(bash_params['gul_threshold']),
        'alloc_rule': int(bash_params['gul_alloc_rule']),
        'debug': 0,
        'random_generator': int(bash_params.get('gul_random_generator', 2)),
        'peril_filter': list(bash_params.get('peril_filter') or []),
        'effective_damageability': bool(bash_params.get('gulmc_effective_damageability')),
        'max_cached_vuln_cdf_size_MB': int(bash_params.get('gulmc_vuln_cache_size') or DEFAULT_VULN_CACHE_MB),
        'model_df_engine': bash_params.get('model_df_engine') or 'oasis_data_manager.df_reader.reader.OasisPandasReader',
        'dynamic_footprint': bool(bash_params.get('dynamic_footprint')),
        'section_merge': bash_params.get('section_merge') or SECTION_MERGE_DEFAULT,
    }


def _modelpy_gulpy_kwargs(bash_params):
    """``ModelpyGulpyStage`` keyword arguments of the ``modelpy | gulpy`` commands ``get_getmodel_cmd`` writes.

    Only the options bash passes are set, so the stages get the commands' defaults for the others: modelpy gets
    ``--peril-filter`` when there is one (and ``--data-server`` with model_py_server, which the orchestrator
    ignores), gulpy ``--random-generator``, ``-S``, ``-L`` and ``-a``.
    """
    modelpy = {}
    if bash_params.get('peril_filter'):
        modelpy['peril_filter'] = list(bash_params['peril_filter'])
    gulpy = {
        'random_generator': int(bash_params.get('gul_random_generator', 2)),
        'sample_size': int(bash_params['number_of_samples']),
        'loss_threshold': float(bash_params['gul_threshold']),
        'alloc_rule': int(bash_params['gul_alloc_rule']),
    }
    return {'modelpy': modelpy, 'gulpy': gulpy}


def _pla_kwargs(analysis_settings):
    """``PlaStage`` keyword arguments of the ``plapy`` command ``get_pla_cmd`` writes, or None without pla.

    Only the option bash passes is set, so the stage gets plapy's defaults for the other one.
    """
    if not analysis_settings.get('pla', False):
        return None
    uniform_factor = analysis_settings.get('pla_uniform_factor', 0)
    secondary_factor = analysis_settings.get('pla_secondary_factor', 1)
    if uniform_factor > 0:
        return {'uniform_factor': uniform_factor}
    if secondary_factor != 1:
        return {'secondary_factor': secondary_factor}
    return {}


def _step_policies(run_dir):
    """True when ``input/fm_profile.csv`` has a ``step_id`` column, i.e. when genbash adds ``-S`` to the IL fmpy."""
    try:
        with open(os.path.join(run_dir, 'input', 'fm_profile.csv'), newline='') as f:
            header = next(csv.reader(f), [])
    except OSError:
        return False
    return 'step_id' in header


def _summary_sets(summaries, runtype, ip, work_dir, kat_dir, low_memory=False):
    """The summary sets of one summarypy, each with the consumers bash's ``tee`` feeds, in bash's order.

    With ``low_memory`` (summarypy ``-m``) a set with ALT period or leccalc output also gets ``idx_paths``: the work
    files the second ``tee`` copies summarypy's ``.idx`` records into, next to the ``.bin`` ones.
    """
    sets = []
    for summary in summaries:
        if 'id' not in summary:
            continue
        n = summary['id']
        ord_output = summary.get('ord_output', {})
        stem = f'{runtype}_{ip}S{n}'
        outputs = []
        for ord_type, tables in ORD_TABLES.items():
            active = [(name, ord_table) for name, _flag, ord_table in tables if ord_output.get(ord_table)]
            if not active:
                continue
            outputs.append({
                'kind': ORD_KIND[ord_type], 'ord_type': ord_type,
                'kwargs': {**{name: True for name, _ in active}, 'output_format': 'bin', 'noheader': False},
                'paths': {name: f'{kat_dir}{stem}_{ord_table}_P{CHUNK_KEY}' for name, ord_table in active},
            })
        if ord_output.get('alt_period'):
            outputs.append({'kind': 'file', 'path': f'{work_dir}{stem}_summary_palt/P{CHUNK_KEY}.bin'})
        if ord_output.get('alt_meanonly'):
            outputs.append({'kind': 'file', 'path': f'{work_dir}{stem}_summary_altmeanonly/P{CHUNK_KEY}.bin'})
        idx_paths = []
        if ord_enabled(summary, ORD_LECCALC):
            outputs.append({'kind': 'file', 'path': f'{work_dir}{stem}_summaryleccalc/P{CHUNK_KEY}.bin'})
        if low_memory and (ord_enabled(summary, ORD_LECCALC) or ord_enabled(summary, ORD_ALT_OUTPUT_SWITCHES)):
            if ord_output.get('alt_period'):
                idx_paths.append(f'{work_dir}{stem}_summary_palt/P{CHUNK_KEY}.idx')
            if ord_enabled(summary, ORD_LECCALC):
                idx_paths.append(f'{work_dir}{stem}_summaryleccalc/P{CHUNK_KEY}.idx')
        sets.append({'id': n, 'outputs': outputs, 'idx_paths': idx_paths})
    return sets


def _summary_node(runtype, ip, step, output, static_path, summaries, work_dir, kat_dir, low_memory=False):
    """One summarypy of the tree: where it reads, its stage kwargs and its summary sets."""
    sets = _summary_sets(summaries, runtype, ip, work_dir, kat_dir, low_memory)
    stage = {'run_type': runtype, 'summary_set_ids': [s['id'] for s in sets],
             'static_path': 'input' if static_path is None else static_path, 'low_memory': low_memory, 'output_zeros': False}
    return {'runtype': runtype, 'inuring_priority': ip, 'source': {'step': step, 'output': output},
            'static_path_flag': static_path is not None, 'stage': stage, 'sets': sets}


def _output_jobs(bash_params, analysis_settings, nodes, n_chunks):
    """The katpy, aalpy and lecpy runs of ``create_bash_outputs``, one per output file."""
    output_dir = bash_params.get('output_dir', 'output/')
    kat_dir = bash_params.get('work_kat_dir', 'work/kat/')
    unsorted = not bash_params.get('kat_sort_by_event', True)
    join = bool(bash_params.get('join_summary_info') or analysis_settings.get('join_summary_info'))
    jobs = []
    for node in nodes:
        runtype, ip = node['runtype'], node['inuring_priority']
        summaries = {s['id']: s for s in analysis_settings.get(f'{runtype}_summaries', []) if 'id' in s}
        for summary_set in node['sets']:
            n = summary_set['id']
            ord_output = summaries[n].get('ord_output', {})
            ext = 'parquet' if ord_output.get('parquet_format') else 'csv'
            stem = f'{runtype}_{ip}S{n}'
            summary_info = f'{output_dir}{runtype}_S{n}_summary-info.{ext}' if join else None
            for ord_type, tables in ORD_TABLES.items():
                for name, _flag, ord_table in tables:
                    if not ord_output.get(ord_table):
                        continue
                    jobs.append(OutputJob('katpy', {
                        'out': f'{output_dir}{stem}_{name}.{ext}', 'file_type': 'bin',
                        'files_in': [f'{kat_dir}{stem}_{ord_table}_P{k}' for k in range(1, n_chunks + 1)],
                        name: True, 'unsorted': unsorted,
                    }, summary_info))
            if ord_enabled(summaries[n], ORD_ALT_OUTPUT_SWITCHES):
                kwargs = {'subfolder': f'{stem}_summary_palt', 'aal': f'{output_dir}{stem}_palt.{ext}', 'ext': ext}
                if ord_output.get('alct_convergence'):
                    kwargs['alct'] = f'{output_dir}{stem}_alct.{ext}'
                    if ord_output.get('alct_confidence'):
                        kwargs['confidence'] = ord_output['alct_confidence']
                jobs.append(OutputJob('aalpy', kwargs, summary_info))
            if ord_enabled(summaries[n], ORD_ALT_MEANONLY_OUTPUT_SWITCHES):
                jobs.append(OutputJob('aalpy', {'subfolder': f'{stem}_summary_altmeanonly', 'meanonly': True,
                                                'aal': f'{output_dir}{stem}_altmeanonly.{ext}', 'ext': ext}, summary_info))
            if ord_enabled(summaries[n], ORD_LECCALC):
                kwargs = {'subfolder': f'{stem}_summaryleccalc', 'use_return_period': bool(ord_output.get('return_period_file')),
                          'ext': ext}
                for option, kwarg in LEC_KWARGS.items():
                    if ord_output.get(option):
                        kwargs[kwarg] = True
                if any(ord_output.get(o) for o in ORD_EPT_OUTPUT_SWITCHES):
                    kwargs['ept'] = f'{output_dir}{stem}_ept.{ext}'
                if any(ord_output.get(o) for o in ORD_PSEPT_OUTPUT_SWITCHES):
                    kwargs['psept'] = f'{output_dir}{stem}_psept.{ext}'
                jobs.append(OutputJob('lecpy', kwargs, summary_info))
    return jobs


def plan_from_bash_params(bash_params, run_dir=None):
    """Build the plan of the run ``genbash`` would script from ``bash_params``.

    Args:
        bash_params (dict): the dict ``oasislmf.execution.bash.bash_params`` returns, or its JSON form.
        run_dir (str, optional): the run directory; ``model_run_dir`` of bash_params, else the current directory.
            genbash looks at ``input/fm_profile.csv`` in it (step policies add ``-S`` to the IL fmpy).

    Returns:
        Plan: the stage tree and the output jobs.

    Raises:
        Unsupported: the run needs something the orchestrator does not cover; the reason says what.
    """
    analysis_settings = bash_params['analysis_settings']
    run_dir = os.path.abspath(run_dir or bash_params.get('model_run_dir') or '.')
    _check_supported(bash_params, analysis_settings, run_dir)

    gul_output = bool(bash_params.get('gul_output'))
    il_output = bool(bash_params.get('il_output'))
    ri_output = bool(bash_params.get('ri_output'))
    n_ri = int(bash_params.get('num_reinsurance_iterations') or 0) if ri_output else 0
    work_dir = bash_params.get('work_dir', 'work/')
    low_memory = bool(bash_params.get('summarypy_low_memory'))
    kat_dir = bash_params.get('work_kat_dir', 'work/kat/')

    gul_engine = 'gulmc' if bash_params.get('gulmc', True) else 'gulpy'
    gul = _gulmc_kwargs(bash_params) if gul_engine == 'gulmc' else _modelpy_gulpy_kwargs(bash_params)
    fm_common = {'low_memory': bool(bash_params.get('fmpy_low_memory')), 'sort_output': bool(bash_params.get('fmpy_sort_output'))}

    fm = []
    if il_output or ri_output:
        il_step = {'name': 'il', 'allocation_rule': int(bash_params['il_alloc_rule']), 'static_path': 'input',
                   'gross': True, 'net': False, **fm_common}
        if _step_policies(run_dir):
            il_step['step_policies'] = True
        fm.append(il_step)
    ri_levels = {ip['level']: ip['text'] for ip in get_reinsurance_inuring_priorities(analysis_settings, n_ri)} if ri_output else {}
    for i in range(1, n_ri + 1):
        fm.append({'name': 'ri', 'allocation_rule': int(bash_params['ri_alloc_rule']), 'static_path': os.path.join('input', f'RI_{i}'),
                   'gross': False, 'net': i < n_ri or i in ri_levels, **fm_common})
    for i, step in enumerate(fm):
        step['feeds_next'] = None if i + 1 == len(fm) else ('gross' if step['name'] == 'il' else 'net')

    nodes = []
    if gul_output:
        nodes.append(_summary_node('gul', '', None, None, None, analysis_settings['gul_summaries'], work_dir, kat_dir, low_memory))
    if il_output:
        nodes.append(_summary_node('il', '', 0, 'gross', None, analysis_settings['il_summaries'], work_dir, kat_dir, low_memory))
    for level, text in ri_levels.items():
        nodes.append(_summary_node('ri', text, level, 'net', os.path.join('input', f'RI_{level}'),
                                   analysis_settings['ri_summaries'], work_dir, kat_dir, low_memory))

    spec = ChainSpec(run_dir=run_dir, gul=gul, fm=fm, summaries=nodes, pla=_pla_kwargs(analysis_settings), gul_engine=gul_engine)
    n_chunks = int(bash_params.get('max_process_id') or 1)
    process_number = bash_params.get('process_number')
    if process_number is not None:
        process_number = int(process_number)
        if not 1 <= process_number <= n_chunks:
            raise ValueError(f'process_number {process_number} is not a partition of 1..{n_chunks}')
    return Plan(spec=spec, output_jobs=_output_jobs(bash_params, analysis_settings, nodes, n_chunks), n_chunks=n_chunks,
                event_shuffle=bash_params.get('event_shuffle', EVE_DEFAULT_SHUFFLE), process_number=process_number,
                output_renames=_ri_final_output(bash_params, analysis_settings, n_ri) if ri_output else [])


def _ri_final_output(bash_params, analysis_settings, n_ri):
    """The renaming ``do_ri_final_output`` scripts once the output files are written.

    The final inuring priority's files are written as ``ri_IP<n>_...``, then renamed ``ri_...``, or copied there when
    ``ri_inuring_priorities`` also asks for that priority on its own.
    """
    output_dir = bash_params.get('output_dir', 'output/')
    return [OutputRename(src=f'{output_dir}ri_{INTERMEDIATE_INURING_PRIORITY_PREFIX}{n_ri}_', dst=f'{output_dir}ri_',
                         copy=n_ri in analysis_settings.get('ri_inuring_priorities', []))]


def _fifo(fifo_dir, runtype, k, consumer=''):
    """Return the fifo path bash's ``get_fifo_name`` builds."""
    return f'{fifo_dir}{runtype}_{consumer}_P{k}' if consumer else f'{fifo_dir}{runtype}_P{k}'


def _gul_command(spec, k, n_chunks, eve_flag):
    """``evepy ... | gulmc ...`` or ``evepy ... | modelpy ... | gulpy ...`` as ``get_getmodel_cmd`` writes it."""
    gul = spec.gul
    argv = ['evepy', *eve_flag.split(), str(k), str(n_chunks), '|']
    if spec.gul_engine == 'gulpy':
        modelpy, gulpy = gul.get('modelpy', {}), gul['gulpy']
        argv.append('modelpy')
        if modelpy.get('data_server'):
            argv.append('--data-server')
        if modelpy.get('peril_filter'):
            argv += ['--peril-filter', *modelpy['peril_filter']]
        argv += ['|', 'gulpy', f"--random-generator={gulpy['random_generator']}"]
        return argv + [f"-S{gulpy['sample_size']}", f"-L{gulpy['loss_threshold']}", f"-a{gulpy['alloc_rule']}"]
    argv += ['gulmc', f"--random-generator={gul['random_generator']}"]
    if gul.get('data_server'):
        argv.append('--data-server')
    argv.append(f"--model-df-engine={gul['model_df_engine']}")
    if gul.get('peril_filter'):
        argv += ['--peril-filter', *gul['peril_filter']]
    if gul.get('effective_damageability'):
        argv.append('--effective-damageability')
    if gul.get('max_cached_vuln_cdf_size_MB'):
        argv += ['--vuln-cache-size', str(gul['max_cached_vuln_cdf_size_MB'])]
    if gul.get('dynamic_footprint'):
        argv += ['--dynamic-footprint', 'True']
    if gul.get('section_merge') and gul['section_merge'] != SECTION_MERGE_DEFAULT:
        argv += ['--section-merge', gul['section_merge']]
    return argv + [f"-S{gul['sample_size']}", f"-L{gul['loss_threshold']}", f"-a{gul['alloc_rule']}"]


def _fm_command(step):
    argv = ['fmpy'] + (['-l'] if step.get('low_memory') else []) + (['--sort-output'] if step.get('sort_output') else [])
    argv.append(f"-a{step['allocation_rule']}")
    if step.get('step_policies'):
        argv.append('-S')
    if step['name'] != 'il':
        argv += ['-p', step['static_path']]
    return argv


def render_chunk_commands(spec, k, n_chunks, event_shuffle=EVE_DEFAULT_SHUFFLE, fifo_dir='fifo/'):
    """The commands bash runs for partition k that a chain built from ``spec`` replaces, as argument lists.

    The result is derived from the spec alone, in genbash's spelling, without the stderr guards and pid bookkeeping
    of the script: the main pipeline (``evepy | gulmc | tee | fmpy ...``, or ``evepy | modelpy | gulpy | ...``), then for
    each summary node its ``summarypy``, and per summary set its ``tee`` and ``pltpy``/``eltpy`` commands.

    Args:
        spec (ChainSpec): a spec from ``plan_from_bash_params``.
        k (int): 1-based partition number.
        n_chunks (int): partition count.
        event_shuffle (int): evepy shuffle rule.
        fifo_dir (str): the fifo directory prefix bash uses.

    Returns:
        list[list[str]]: the commands.
    """
    nodes = spec.summaries
    by_source = {}
    for node in nodes:
        by_source.setdefault((node['source']['step'], node['source']['output']), []).append(node)

    def node_fifo(node):
        return _fifo(fifo_dir, node['runtype'], k, node['inuring_priority'].rstrip('_'))

    main = _gul_command(spec, k, n_chunks, EVE_FLAGS[event_shuffle])
    if spec.pla is not None:
        main += ['|', 'plapy']
        if 'uniform_factor' in spec.pla:
            main += ['-F', str(spec.pla['uniform_factor'])]
        elif 'secondary_factor' in spec.pla:
            main += ['-f', str(spec.pla['secondary_factor'])]
    gul_nodes = by_source.get((None, None), [])
    if spec.fm:
        if gul_nodes:
            main += ['|', 'tee', node_fifo(gul_nodes[0])]
    elif gul_nodes:
        main += ['>', node_fifo(gul_nodes[0])]
    last = len(spec.fm) - 1
    for i, step in enumerate(spec.fm):
        main += ['|', *_fm_command(step)]
        gross_nodes = by_source.get((i, 'gross'), [])
        net_nodes = by_source.get((i, 'net'), [])
        if step['name'] == 'il':
            if not gross_nodes:
                continue
            main += ['>', node_fifo(gross_nodes[0])] if i == last else ['|', 'tee', node_fifo(gross_nodes[0])]
            continue
        if gross_nodes:
            main += ['-o', node_fifo(gross_nodes[0])]
        if i < last:
            main += ['-n', '-']
        if net_nodes:
            main += ['-n', '-', '>', node_fifo(net_nodes[0])] if i == last else ['|', 'tee', node_fifo(net_nodes[0])]
    commands = [main]

    for node in nodes:
        runtype, ip = node['runtype'], node['inuring_priority']
        stage = node['stage']
        argv = ['summarypy'] + (['-m'] if stage.get('low_memory') else []) + ['-t', stage['run_type']]
        if node['static_path_flag']:
            argv += ['-p', stage['static_path']]
        for summary_set in node['sets']:
            argv += [f"-{summary_set['id']}", _fifo(fifo_dir, runtype, k, f"{ip}S{summary_set['id']}_summary")]
        commands.append(argv + ['<', node_fifo(node)])
        for summary_set in node['sets']:
            n = summary_set['id']
            tee = ['tee', '<', _fifo(fifo_dir, runtype, k, f'{ip}S{n}_summary')]
            for consumer in summary_set['outputs']:
                if consumer['kind'] == 'file':
                    tee.append(fill_chunk(consumer['path'], k))
                    continue
                ord_fifo = _fifo(fifo_dir, runtype, k, f"{ip}S{n}_{consumer['ord_type']}")
                tee.append(ord_fifo)
                tool = ['pltpy' if consumer['kind'] == 'plt' else 'eltpy', '-E', consumer['kwargs']['output_format']]
                if k != 1:
                    tool.append('-H')
                for name, flag, _ord_table in ORD_TABLES[consumer['ord_type']]:
                    if name in consumer['paths']:
                        tool += [flag, fill_chunk(consumer['paths'][name], k)]
                commands.append(tool + ['<', ord_fifo])
            commands.append(tee)
            if summary_set.get('idx_paths'):
                idx_fifo = _fifo(fifo_dir, runtype, k, f'{ip}S{n}_summary') + '.idx'
                commands.append(['tee', '<', idx_fifo, *(fill_chunk(p, k) for p in summary_set['idx_paths'])])
    return commands
