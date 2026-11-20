"""Command line entry point of the event-worker orchestrator.

Runs ``evepy k K | gulmc ... | tee gul | fmpy -a<il> | tee il | fmpy -a<ri> -p input/RI_1 -n - > ri`` for every
k in one go, with the executor chosen by ``--executor``, and writes ``<out-dir>/<tap>_P<k>.bin``::

    python -m oasislmf.execution.orchestrator.cli --executor inline --run-dir RUN --chunks 12 --out-dir OUT --taps gul,il,ri
        -S10 -L0 -a1 --random-generator 2 --vuln-cache-size 200 --il-alloc-rule 2 --ri-layers 1 --ri-alloc-rule 3

With ``--plan bash_params.json`` (the dict ``genbash`` consumes) or ``--analysis-settings`` the chain is built by
the planner instead: the gul and fm options are ignored, and every chunk writes the work files bash's partition k
writes (``work/kat/*_P<k>``, ``work/*_summary*/P<k>.bin``), in the run directory; once every chunk is done the
katpy/aalpy/lecpy output jobs write ``output/``, on the same executor (``--no-output-jobs`` skips them). A plan the
orchestrator does not support exits with status 3 and the reason::

    python -m oasislmf.execution.orchestrator.cli --executor pool --run-dir RUN --plan RUN/bash_params.json --workers 8
"""
import argparse
import importlib
import json
import logging
import os
import shutil
import sys

import numpy as np

from oasislmf.execution.bash import bash_params as make_bash_params
from oasislmf.pytools.getmodel.common import SECTION_MERGE_DEFAULT, SECTION_MERGE_OPTIONS
from oasislmf.utils.data import get_json

from .chain import TAPS, build_chain_spec
from .chunks import make_chunks, read_events
from .outputs import jobs_for_chunks
from .plan import Unsupported, plan_from_bash_params
from .resources import MB, ResourceManager

EXECUTORS_PACKAGE = 'oasislmf.execution.orchestrator'
EXECUTORS = {'inline': '.inline', 'pool': '.pool'}
EVENTS_FILE = os.path.join('input', 'events.bin')
RI_LAYERS_FILE = os.path.join('input', 'ri_layers.json')
EXIT_UNSUPPORTED = 3

logger = logging.getLogger(__name__)


def get_executor(name):
    """Import and return the executor module registered under ``name``.

    Args:
        name (str): ``inline`` or ``pool``.

    Returns:
        module: a module exposing ``run_chunks``.
    """
    if name not in EXECUTORS:
        raise ValueError(f"unknown executor {name!r}, expected one of {sorted(EXECUTORS)}")
    return importlib.import_module(EXECUTORS[name], package=EXECUTORS_PACKAGE)


def parse_taps(text):
    """Parse a comma-separated list of taps, e.g. ``'gul,il,ri'``."""
    taps = [t.strip() for t in text.split(',') if t.strip()]
    unknown = [t for t in taps if t not in TAPS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown taps {unknown}, expected some of {list(TAPS)}")
    if not taps:
        raise argparse.ArgumentTypeError("no tap given")
    return taps


def parse_chunk_order(text):
    """Parse ``--chunk-order``: ``reverse``, ``shuffle:<seed>`` or a comma-separated list of 1-based chunk numbers.

    Args:
        text (str): the option value.

    Returns:
        str | list[int]: ``'reverse'`` or ``'shuffle:<seed>'`` as given, otherwise the list of chunk numbers.
    """
    if text == 'reverse':
        return text
    if text.startswith('shuffle:'):
        try:
            int(text.partition(':')[2])
        except ValueError:
            raise argparse.ArgumentTypeError(f"bad shuffle seed in {text!r}")
        return text
    try:
        return [int(k) for k in text.split(',') if k.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected reverse, shuffle:<seed> or chunk numbers, got {text!r}")


def resolve_chunk_order(order, n_chunks):
    """Turn a parsed ``--chunk-order`` into the list of 1-based chunk numbers to run, in order.

    Args:
        order (str | list[int] | None): output of ``parse_chunk_order``, or None for chunk 1 to K.
        n_chunks (int): number of chunks K.

    Returns:
        list[int] | None: the chunk numbers, or None for the natural order.
    """
    if order is None:
        return None
    if order == 'reverse':
        return list(range(n_chunks, 0, -1))
    if isinstance(order, str):
        rng = np.random.default_rng(int(order.partition(':')[2]))
        return [int(k) for k in rng.permutation(np.arange(1, n_chunks + 1))]
    return list(order)


def build_parser():
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog='python -m oasislmf.execution.orchestrator.cli',
        description='Run gulmc and fmpy over the events of a run directory, chunk by chunk, without pipes.',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    run = parser.add_argument_group('run')
    run.add_argument('--executor', choices=sorted(EXECUTORS), default='inline', help='executor. Default: inline')
    run.add_argument('--run-dir', default=None,
                     help='run directory (with input/events.bin). Default: "." (with --plan: model_run_dir of the plan, else ".")')
    run.add_argument('--chunks', type=int, default=None,
                     help='number of chunks K; chunk k holds the events of `evepy k K`. Default: 1 (with --plan: max_process_id)')
    run.add_argument('--workers', type=int, default=1, help='number of workers (with --resource-manager on, the maximum). Default: 1')
    run.add_argument('--resource-manager', choices=('on', 'off'), default='off',
                     help='pool executor: size the pool from measured memory and CPU, starting --initial-workers and\n'
                          'never above --workers (on), or start --workers workers (off). Default: off')
    run.add_argument('--initial-workers', type=int, default=None,
                     help='with --resource-manager on: workers started before anything is measured. Default: 2')
    run.add_argument('--memory-margin-mb', type=int, default=None,
                     help='with --resource-manager on: memory kept free, in MB. Default: max(1 GB, 10%% of memory)')
    run.add_argument('--out-dir', default=None, help='output directory for <tap>_P<k>.bin, relative to the current directory;\n'
                                                     'required when taps are written')
    run.add_argument('--taps', type=parse_taps, default=None,
                     help='comma-separated taps among gul,il,ri. Default: gul,il,ri (with --plan: none)')
    run.add_argument('--plan', default=None, metavar='BASH_PARAMS_JSON',
                     help='build the chain with the planner from this bash_params JSON (as genbash consumes it)')
    run.add_argument('--analysis-settings', default=None, metavar='JSON',
                     help='build the chain with the planner from bash_params made from these analysis settings with the\n'
                          'genbash defaults (RI layers from input/ri_layers.json, K = --chunks)')
    run.add_argument('--keep-work', action='store_true',
                     help='with --plan/--analysis-settings: do not empty the work directories the chunks write into first')
    run.add_argument('--no-output-jobs', action='store_true',
                     help='with --plan/--analysis-settings: run the chunks only, not the katpy/aalpy/lecpy output jobs')
    run.add_argument('--chunk-order', type=parse_chunk_order, default=None,
                     help='inline executor only, for testing: order the chunks are run in, as reverse, shuffle:<seed> or\n'
                          'a comma-separated permutation of 1..K. Output file names do not change. Default: 1..K')
    run.add_argument('--stats-json', default=None, help='write the executor statistics to this JSON file')
    run.add_argument('--logging-level', type=int, default=30,
                     help='logging level (debug:10, info:20, warning:30, error:40, critical:50). Default: 30')

    gul = parser.add_argument_group('gul (as gulmc)')
    gul.add_argument('-a', '--alloc-rule', type=int, default=0, dest='alloc_rule', help='gul back-allocation rule. Default: 0')
    gul.add_argument('-d', '--debug', type=int, default=0, dest='debug',
                     help='output the ground up loss (0), the hazard (1) or damage (2) random numbers. Default: 0')
    gul.add_argument('-L', '--loss-threshold', type=float, default=1e-6, dest='loss_threshold', help='loss threshold. Default: 1e-6')
    gul.add_argument('-S', '--sample-size', type=int, default=0, dest='sample_size', help='sample size. Default: 0')
    gul.add_argument('--effective-damageability', action='store_true', default=False,
                     help='draw loss samples from the effective damageability instead of full MC')
    gul.add_argument('--ignore-correlation', action='store_true', default=False, help='ignore peril correlation groups')
    gul.add_argument('--ignore-haz-correlation', action='store_true', default=False, help='ignore hazard correlation groups')
    gul.add_argument('--ignore-file-type', nargs='*', default=[], help='file types not to load')
    gul.add_argument('--data-server', action='store_true', default=False, help='use tcp/sockets for IPC data sharing')
    gul.add_argument('--vuln-cache-size', type=int, default=200, dest='max_cached_vuln_cdf_size_MB',
                     help='size in MB of the vulnerability cdf cache. Default: 200')
    gul.add_argument('--peril-filter', nargs='+', default=[], help='ids of the perils to keep; all if empty')
    gul.add_argument('--random-generator', type=int, default=2,
                     help='0: numpy default (MT19937), 1: Latin Hypercube, 2: Latin Hypercube on Philox4x32-7. Default: 2')
    gul.add_argument('--model-df-engine', default='oasis_data_manager.df_reader.reader.OasisPandasReader',
                     help='engine used to load model dataframes')
    gul.add_argument('--dynamic-footprint', action='store_true', default=False, help='apply dynamic footprint logic')
    gul.add_argument('--section-merge', default=SECTION_MERGE_DEFAULT, choices=SECTION_MERGE_OPTIONS,
                     help='how a dynamic footprint merges the intensities of an areaperil lying in several sections of one '
                          f'event. Default: {SECTION_MERGE_DEFAULT}')

    fm = parser.add_argument_group('fm (as fmpy)')
    fm.add_argument('--il-alloc-rule', type=int, default=2, help='IL back-allocation rule (fmpy -a). Default: 2')
    fm.add_argument('--ri-layers', type=int, default=0,
                    help='number of RI layers; layer i reads static files from input/RI_<i> and passes its net loss on. Default: 0')
    fm.add_argument('--ri-alloc-rule', type=int, default=3, help='RI back-allocation rule (fmpy -a). Default: 3')
    fm.add_argument('-l', '--low-memory', action='store_true', help='fm loss arrays in memory map')
    fm.add_argument('--sort-output', action='store_true', help='sort the fm output streams by item_id')
    fm.add_argument('--storage-method', default='sparse', help='fm storage method. Default: sparse')
    return parser


def spec_from_args(args):
    """Build the ``ChainSpec`` described by parsed arguments.

    Args:
        args (argparse.Namespace): output of ``build_parser().parse_args()``.

    Returns:
        ChainSpec: gul, IL step (``fmpy -a<il>``) and ``--ri-layers`` RI steps (``fmpy -a<ri> -p input/RI_<i> -n -``).
    """
    gul = {
        'sample_size': args.sample_size,
        'loss_threshold': args.loss_threshold,
        'alloc_rule': args.alloc_rule,
        'debug': args.debug,
        'random_generator': args.random_generator,
        'ignore_file_type': list(args.ignore_file_type or []),
        'peril_filter': list(args.peril_filter or []),
        'data_server': args.data_server,
        'ignore_correlation': args.ignore_correlation,
        'ignore_haz_correlation': args.ignore_haz_correlation,
        'effective_damageability': args.effective_damageability,
        'max_cached_vuln_cdf_size_MB': args.max_cached_vuln_cdf_size_MB,
        'model_df_engine': args.model_df_engine,
        'dynamic_footprint': args.dynamic_footprint,
        'section_merge': args.section_merge,
    }
    fm_common = {'low_memory': args.low_memory, 'sort_output': args.sort_output, 'storage_method': args.storage_method}
    il = {'allocation_rule': args.il_alloc_rule, 'static_path': 'input', **fm_common}
    ri = [{'allocation_rule': args.ri_alloc_rule, 'static_path': os.path.join('input', f'RI_{i}'), **fm_common}
          for i in range(1, args.ri_layers + 1)]
    return build_chain_spec(os.path.abspath(args.run_dir or '.'), gul, il=il, ri=ri)


def is_planned(args):
    """True when the chain comes from the planner (``--plan`` or ``--analysis-settings``)."""
    return bool(args.plan or args.analysis_settings)


def resolve_taps(args):
    """The taps to write.

    ``--taps`` if given, else none when the chain comes from the planner, else gul and il, plus ri when
    ``--ri-layers`` is at least 1.
    """
    if args.taps is not None:
        return list(args.taps)
    if is_planned(args):
        return []
    return [tap for tap in TAPS if tap != 'ri' or args.ri_layers >= 1]


def load_plan(args):
    """Build the plan of ``--plan`` or ``--analysis-settings``.

    Args:
        args (argparse.Namespace): parsed arguments.

    Returns:
        Plan: the plan.

    Raises:
        Unsupported: the planner does not cover the run.
    """
    if args.plan:
        with open(args.plan) as f:
            params = json.load(f)
        return plan_from_bash_params(params, run_dir=args.run_dir)
    run_dir = os.path.abspath(args.run_dir or '.')
    analysis_settings = get_json(args.analysis_settings)
    n_ri = 0
    if analysis_settings.get('ri_output') or analysis_settings.get('rl_output'):
        n_ri = len(get_json(os.path.join(run_dir, RI_LAYERS_FILE)))
    params = make_bash_params(analysis_settings, max_process_id=args.chunks or -1, num_reinsurance_iterations=n_ri,
                              fifo_tmp_dir=False, model_run_dir='',
                              peril_filter=[p.upper() for p in analysis_settings.get('peril_filter') or []])
    return plan_from_bash_params(params, run_dir=run_dir)


def clear_work_dirs(plan):
    """Empty the directories the chunks of a plan write into, as bash empties ``work/`` before a run.

    Args:
        plan (Plan): the plan; relative directories are taken in its run directory.
    """
    run_dir = os.path.abspath(plan.spec.run_dir)
    for directory in plan.work_dirs():
        path = os.path.abspath(os.path.join(run_dir, directory))
        if path != run_dir and os.path.isdir(path):
            shutil.rmtree(path)


def main(argv=None):
    """Parse the arguments, run the executor and write the statistics.

    Args:
        argv (list[str], optional): arguments; ``sys.argv[1:]`` if None.

    Returns:
        dict: the executor statistics.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    planned = is_planned(args)
    if args.plan and args.analysis_settings:
        parser.error('--plan and --analysis-settings are exclusive')
    if args.chunks is not None and args.chunks <= 0:
        parser.error('--chunks must be positive')
    if args.workers <= 0:
        parser.error('--workers must be positive')
    if args.ri_layers < 0:
        parser.error('--ri-layers must not be negative')
    managed = args.resource_manager == 'on'
    if managed and args.executor != 'pool':
        parser.error('--resource-manager on needs --executor pool')
    if not managed and (args.initial_workers is not None or args.memory_margin_mb is not None):
        parser.error('--initial-workers and --memory-margin-mb need --resource-manager on')
    if args.initial_workers is not None and args.initial_workers <= 0:
        parser.error('--initial-workers must be positive')
    taps = resolve_taps(args)
    if taps and not args.out_dir:
        parser.error('--out-dir is required when taps are written')
    if not planned and 'ri' in taps and args.ri_layers == 0:
        parser.error("tap 'ri' needs --ri-layers >= 1")
    logging.basicConfig(level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if planned:
        try:
            plan = load_plan(args)
        except Unsupported as exc:
            parser.exit(EXIT_UNSUPPORTED, f'{parser.prog}: unsupported by the orchestrator, run with bash: {exc.reason}\n')
        spec = plan.spec
        n_chunks = args.chunks or plan.n_chunks
    else:
        spec = spec_from_args(args)
        n_chunks = args.chunks or 1
    chunk_order = resolve_chunk_order(args.chunk_order, n_chunks)
    if chunk_order is not None:
        if args.executor != 'inline':
            parser.error('--chunk-order is only supported by the inline executor')
        if sorted(chunk_order) != list(range(1, n_chunks + 1)):
            parser.error(f'--chunk-order must be a permutation of 1..{n_chunks}')

    events = read_events(os.path.join(spec.run_dir, EVENTS_FILE))
    if planned:
        chunks = plan.make_chunks(events, n_chunks)
        if not args.keep_work:
            clear_work_dirs(plan)
    else:
        chunks = make_chunks(events, n_chunks)
    out_dir = args.out_dir or os.path.join(spec.run_dir, 'work')
    executor = get_executor(args.executor)
    options = {'chunk_order': chunk_order} if chunk_order is not None else {}
    if planned and not args.no_output_jobs:
        options['output_jobs'] = jobs_for_chunks(plan.output_jobs, n_chunks)
    if managed:
        manager_options = {}
        if args.initial_workers is not None:
            manager_options['initial_workers'] = args.initial_workers
        if args.memory_margin_mb is not None:
            manager_options['margin'] = args.memory_margin_mb * MB
        options['resource_manager'] = ResourceManager(args.workers, **manager_options)
    stats = executor.run_chunks(spec, chunks, out_dir, taps, args.workers, **options)
    logger.info("executor %s: %d events in %d chunks, %d output jobs, %.2f s (output jobs %.2f s)", args.executor, len(events),
                n_chunks, len(stats.get('jobs', [])), stats.get('wall_seconds', 0.0), stats.get('output_seconds', 0.0))
    if args.stats_json:
        with open(args.stats_json, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    return stats


if __name__ == '__main__':
    main(sys.argv[1:])
