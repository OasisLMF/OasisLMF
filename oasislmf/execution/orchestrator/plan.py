"""The planner: the stage tree of a kernel run, built from the ``bash_params`` dict that ``genbash`` consumes.

``plan_from_bash_params`` reproduces what the generated bash script runs, partition by partition, as a ``ChainSpec``
whose stages are the in-process equivalents of the commands (``gulmc``, ``fmpy`` per IL/RI step, one ``summarypy``
per run type and inuring priority, ``pltpy``/``eltpy`` per summary set, and the ``tee`` work files), plus the
``katpy``/``aalpy``/``lecpy`` commands bash runs once every partition is done, as ``OutputJob``s.

Every per-chunk path is a template holding ``{k}`` and is exactly the name bash uses for partition k, so the work
directory of an orchestrated run is interchangeable with a bash one. ``render_chunk_commands`` and
``OutputJob.command`` turn a plan back into the bash commands it stands for; the tests compare them with the
script ``genbash`` writes, which is what ties the planner to ``genbash``.

Settings the stages do not cover raise ``Unsupported`` with the reason, and the run falls back to bash.
"""
import csv
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from oasislmf.execution.bash import (ORD_ALT_MEANONLY_OUTPUT_SWITCHES, ORD_ALT_OUTPUT_SWITCHES,
                                     ORD_EPT_OUTPUT_SWITCHES, ORD_LECCALC, ORD_PSEPT_OUTPUT_SWITCHES, OUTPUT_SWITCHES,
                                     get_ri_inuring_priorities, get_rl_inuring_priorities, ord_enabled)
from oasislmf.pytools.eve.manager import (event_id_dtype, partition_events__no_shuffle, partition_events__random,
                                          partition_events__random_builtin, partition_events__round_robin)
from oasislmf.utils.defaults import (EVE_DEFAULT_SHUFFLE, EVE_FISHER_YATES, EVE_NO_SHUFFLE, EVE_ROUND_ROBIN,
                                     EVE_STD_SHUFFLE)

from .chain import CHUNK_KEY, ChainSpec, fill_chunk

__all__ = ['Unsupported', 'OutputJob', 'Plan', 'plan_from_bash_params', 'render_chunk_commands', 'fill_chunk']

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


class Unsupported(Exception):
    """The run needs something the orchestrator does not cover; it must run with bash.

    Attributes:
        reason (str): why.
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass
class OutputJob:
    """One command bash runs after every partition is done.

    Attributes:
        tool (str): ``katpy``, ``aalpy`` or ``lecpy``.
        kwargs (dict): keyword arguments of the tool's ``main`` (``OUTPUT_TOOLS``), i.e. its command-line options
            by ``dest``. Paths are as bash writes them, relative to the run directory unless bash made them absolute.
    """
    tool: str
    kwargs: dict = field(default_factory=dict)

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
            argv = ['aalpy', f"-K{kw['subfolder']}"]
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

    def to_dict(self):
        """Return the job as a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build a job from the output of ``to_dict``."""
        return cls(tool=data['tool'], kwargs=dict(data['kwargs']))


@dataclass
class Plan:
    """What a kernel run does, as the orchestrator runs it.

    Attributes:
        spec (ChainSpec): the stage tree, with its per-chunk path templates.
        output_jobs (list[OutputJob]): the katpy/aalpy/lecpy runs after the last chunk, in any order.
        n_chunks (int): the partition count bash would use (``max_process_id``).
        event_shuffle (int): evepy partitioning rule (``EVE_*`` of ``oasislmf.utils.defaults``).
    """
    spec: ChainSpec
    output_jobs: list = field(default_factory=list)
    n_chunks: int = 1
    event_shuffle: int = EVE_DEFAULT_SHUFFLE

    def make_chunks(self, events, n_chunks=None):
        """Split events into chunks as ``evepy`` partitions them with this plan's shuffle rule.

        Args:
            events (array-like): event ids, in the order they are held in ``events.bin``.
            n_chunks (int, optional): number of chunks; ``n_chunks`` of the plan by default.

        Returns:
            list[np.ndarray]: element ``k - 1`` holds the events ``evepy k n_chunks`` emits, in the same order.
        """
        n_chunks = self.n_chunks if n_chunks is None else n_chunks
        if n_chunks <= 0:
            raise ValueError(f"n_chunks must be positive, got {n_chunks}")
        events = np.asarray(events, dtype=event_id_dtype)
        return [partition_events(events, k, n_chunks, self.event_shuffle) for k in range(1, n_chunks + 1)]

    def chunk_paths(self, k):
        """Every file chunk k writes (its work and kat files), relative to the run directory as bash names them."""
        paths = []
        for node in self.spec.summaries:
            for summary_set in node['sets']:
                for consumer in summary_set['outputs']:
                    if consumer['kind'] == 'file':
                        paths.append(fill_chunk(consumer['path'], k))
                    else:
                        paths.extend(fill_chunk(p, k) for p in consumer['paths'].values())
                paths.extend(fill_chunk(p, k) for p in summary_set.get('idx_paths', []))
        return paths

    def work_dirs(self):
        """The directories the chunks write into, sorted."""
        return sorted({os.path.dirname(p) for p in self.chunk_paths(1)})

    def to_dict(self):
        """Return the plan as a plain JSON-serialisable dict."""
        return {'spec': self.spec.to_dict(), 'output_jobs': [job.to_dict() for job in self.output_jobs],
                'n_chunks': self.n_chunks, 'event_shuffle': self.event_shuffle}

    @classmethod
    def from_dict(cls, data):
        """Build a plan from the output of ``to_dict``."""
        return cls(spec=ChainSpec.from_dict(data['spec']), output_jobs=[OutputJob.from_dict(j) for j in data['output_jobs']],
                   n_chunks=data['n_chunks'], event_shuffle=data['event_shuffle'])

    def to_json(self):
        """Return the plan as a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text):
        """Build a plan from the output of ``to_json``."""
        return cls.from_dict(json.loads(text))


def partition_events(events, k, n_chunks, event_shuffle=EVE_DEFAULT_SHUFFLE):
    """The events ``evepy`` emits for partition k of n_chunks with a shuffle rule, as an array."""
    if event_shuffle == EVE_ROUND_ROBIN:
        return partition_events__round_robin(events, k, n_chunks)
    if event_shuffle == EVE_NO_SHUFFLE:
        return partition_events__no_shuffle(events, k, n_chunks)
    if event_shuffle == EVE_FISHER_YATES:
        return np.fromiter(partition_events__random(events.copy(), k, n_chunks), dtype=event_id_dtype)
    if event_shuffle == EVE_STD_SHUFFLE:
        return partition_events__random_builtin(events.copy(), k, n_chunks)
    raise ValueError(f'unknown event shuffle rule {event_shuffle!r}')


def _check_supported(bash_params, analysis_settings, run_dir):
    """Raise ``Unsupported`` for every setting that changes the pipeline in a way the stages do not reproduce."""
    if bash_params.get('_get_getmodel_cmd'):
        raise Unsupported('a custom gulcalc (model supplier binary or custom getmodel command) produces the ground up loss')
    if not bash_params.get('gulmc', True):
        raise Unsupported('the ground up loss is computed by modelpy | gulpy, only gulmc has a stage')
    if bash_params.get('full_correlation'):
        raise Unsupported('full correlation output (only produced with gulpy)')
    if bash_params.get('custom_args'):
        raise Unsupported(f"custom_args {sorted(bash_params['custom_args'])} are passed to the getmodel command")
    if bash_params.get('process_number') is not None:
        raise Unsupported('distributed chunk mode (process_number): the script runs a single partition in its own work dir')
    if bash_params.get('model_py_server'):
        raise Unsupported('the model data server (servedata, gulmc --data-server)')
    if bash_params.get('join_summary_info') or analysis_settings.get('join_summary_info'):
        raise Unsupported('join-summary-info post-processing of the output files')
    if analysis_settings.get('pla'):
        raise Unsupported('post loss amplification (plapy)')
    il_or_ri = bash_params.get('il_output') or bash_params.get('ri_output')
    if bash_params.get('num_gul_per_lb') and bash_params.get('num_fm_per_lb') and il_or_ri:
        raise Unsupported('load balancer (num_gul_per_lb / num_fm_per_lb) between gul and fm')
    event_shuffle = bash_params.get('event_shuffle', EVE_DEFAULT_SHUFFLE)
    if event_shuffle not in EVE_FLAGS:
        raise Unsupported(f'unknown event shuffle rule {event_shuffle!r}')
    for runtype in ('gul', 'il', 'ri', 'rl'):
        if not bash_params.get(f'{runtype}_output'):
            continue
        for summary in analysis_settings.get(f'{runtype}_summaries', []):
            legacy = [key for key in LEGACY_SUMMARY_OUTPUTS if summary.get(key)]
            if legacy:
                raise Unsupported(f"legacy (non-ORD) outputs {legacy} in {runtype}_summaries id {summary.get('id')}")


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
    stage = {'run_type': 'ri' if runtype == 'rl' else runtype, 'summary_set_ids': [s['id'] for s in sets],
             'static_path': 'input' if static_path is None else static_path, 'low_memory': low_memory, 'output_zeros': False}
    return {'runtype': runtype, 'inuring_priority': ip, 'source': {'step': step, 'output': output},
            'static_path_flag': static_path is not None, 'stage': stage, 'sets': sets}


def _output_jobs(bash_params, analysis_settings, nodes, n_chunks):
    """The katpy, aalpy and lecpy runs of ``create_bash_outputs``, one per output file."""
    output_dir = bash_params.get('output_dir', 'output/')
    kat_dir = bash_params.get('work_kat_dir', 'work/kat/')
    unsorted = not bash_params.get('kat_sort_by_event', True)
    jobs = []
    for node in nodes:
        runtype, ip = node['runtype'], node['inuring_priority']
        summaries = {s['id']: s for s in analysis_settings.get(f'{runtype}_summaries', []) if 'id' in s}
        for summary_set in node['sets']:
            n = summary_set['id']
            ord_output = summaries[n].get('ord_output', {})
            ext = 'parquet' if ord_output.get('parquet_format') else 'csv'
            stem = f'{runtype}_{ip}S{n}'
            for ord_type, tables in ORD_TABLES.items():
                for name, _flag, ord_table in tables:
                    if not ord_output.get(ord_table):
                        continue
                    jobs.append(OutputJob('katpy', {
                        'out': f'{output_dir}{stem}_{name}.{ext}', 'file_type': 'bin',
                        'files_in': [f'{kat_dir}{stem}_{ord_table}_P{k}' for k in range(1, n_chunks + 1)],
                        name: True, 'unsorted': unsorted,
                    }))
            if ord_enabled(summaries[n], ORD_ALT_OUTPUT_SWITCHES):
                kwargs = {'subfolder': f'{stem}_summary_palt', 'aal': f'{output_dir}{stem}_palt.{ext}', 'ext': ext}
                if ord_output.get('alct_convergence'):
                    kwargs['alct'] = f'{output_dir}{stem}_alct.{ext}'
                    if ord_output.get('alct_confidence'):
                        kwargs['confidence'] = ord_output['alct_confidence']
                jobs.append(OutputJob('aalpy', kwargs))
            if ord_enabled(summaries[n], ORD_ALT_MEANONLY_OUTPUT_SWITCHES):
                jobs.append(OutputJob('aalpy', {'subfolder': f'{stem}_summary_altmeanonly',
                                                'aal': f'{output_dir}{stem}_altmeanonly.{ext}', 'ext': ext}))
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
                jobs.append(OutputJob('lecpy', kwargs))
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
    rl_output = bool(bash_params.get('rl_output'))
    n_ri = int(bash_params.get('num_reinsurance_iterations') or 0) if (ri_output or rl_output) else 0
    work_dir = bash_params.get('work_dir', 'work/')
    low_memory = bool(bash_params.get('summarypy_low_memory'))
    kat_dir = bash_params.get('work_kat_dir', 'work/kat/')

    gul = {
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
    }
    fm_common = {'low_memory': bool(bash_params.get('fmpy_low_memory')), 'sort_output': bool(bash_params.get('fmpy_sort_output'))}

    fm = []
    if il_output or ri_output or rl_output:
        il_step = {'name': 'il', 'allocation_rule': int(bash_params['il_alloc_rule']), 'static_path': 'input',
                   'gross': True, 'net': False, **fm_common}
        if _step_policies(run_dir):
            il_step['step_policies'] = True
        fm.append(il_step)
    ri_levels = {ip['level']: ip['text'] for ip in get_ri_inuring_priorities(analysis_settings, n_ri)} if ri_output else {}
    rl_levels = {ip['level']: ip['text'] for ip in get_rl_inuring_priorities(n_ri)} if rl_output else {}
    for i in range(1, n_ri + 1):
        fm.append({'name': 'ri', 'allocation_rule': int(bash_params['ri_alloc_rule']), 'static_path': os.path.join('input', f'RI_{i}'),
                   'gross': i in rl_levels, 'net': i < n_ri or i in ri_levels, **fm_common})
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
    for level, text in rl_levels.items():
        nodes.append(_summary_node('rl', text, level, 'gross', os.path.join('input', f'RI_{level}'),
                                   analysis_settings['rl_summaries'], work_dir, kat_dir, low_memory))

    spec = ChainSpec(run_dir=run_dir, gul=gul, fm=fm, summaries=nodes)
    n_chunks = int(bash_params.get('max_process_id') or 1)
    return Plan(spec=spec, output_jobs=_output_jobs(bash_params, analysis_settings, nodes, n_chunks), n_chunks=n_chunks,
                event_shuffle=bash_params.get('event_shuffle', EVE_DEFAULT_SHUFFLE))


def _fifo(fifo_dir, runtype, k, consumer=''):
    """Return the fifo path bash's ``get_fifo_name`` builds."""
    return f'{fifo_dir}{runtype}_{consumer}_P{k}' if consumer else f'{fifo_dir}{runtype}_P{k}'


def _gul_command(gul, k, n_chunks, eve_flag):
    """``evepy ... | gulmc ...`` as ``get_getmodel_cmd`` writes it."""
    argv = ['evepy', *eve_flag.split(), str(k), str(n_chunks), '|', 'gulmc', f"--random-generator={gul['random_generator']}"]
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
    of the script: the main pipeline (``evepy | gulmc | tee | fmpy ...``), then for each summary node its ``summarypy``,
    and per summary set its ``tee`` and ``pltpy``/``eltpy`` commands.

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

    main = _gul_command(spec.gul, k, n_chunks, EVE_FLAGS[event_shuffle])
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
