"""Command line entry point of the event-worker orchestrator.

Runs ``evepy k K | gulmc ... | tee gul | fmpy -a<il> | tee il | fmpy -a<ri> -p input/RI_1 -n - > ri`` for every
k in one go, with the executor chosen by ``--executor``, and writes ``<out-dir>/<tap>_P<k>.bin``::

    python -m oasislmf.execution.orchestrator.cli --executor inline --run-dir RUN --chunks 12 --out-dir OUT --taps gul,il,ri
        -S10 -L0 -a1 --random-generator 2 --vuln-cache-size 200 --il-alloc-rule 2 --ri-layers 1 --ri-alloc-rule 3
"""
import argparse
import importlib
import json
import logging
import os
import sys

import numpy as np

from .chain import TAPS, build_chain_spec
from .chunks import make_chunks, read_events

EXECUTORS_PACKAGE = 'oasislmf.execution.orchestrator'
EXECUTORS = {'inline': '.inline', 'pool': '.pool'}
EVENTS_FILE = os.path.join('input', 'events.bin')

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
    run.add_argument('--run-dir', default='.', help='run directory (with input/events.bin). Default: "."')
    run.add_argument('--chunks', type=int, default=1, help='number of chunks K; chunk k holds the events of `evepy k K`. Default: 1')
    run.add_argument('--workers', type=int, default=1, help='number of workers. Default: 1')
    run.add_argument('--out-dir', required=True, help='output directory for <tap>_P<k>.bin, relative to the current directory')
    run.add_argument('--taps', type=parse_taps, default=['gul', 'il', 'ri'], help='comma-separated taps among gul,il,ri. Default: gul,il,ri')
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
    }
    fm_common = {'low_memory': args.low_memory, 'sort_output': args.sort_output, 'storage_method': args.storage_method}
    il = {'allocation_rule': args.il_alloc_rule, 'static_path': 'input', **fm_common}
    ri = [{'allocation_rule': args.ri_alloc_rule, 'static_path': os.path.join('input', f'RI_{i}'), **fm_common}
          for i in range(1, args.ri_layers + 1)]
    return build_chain_spec(os.path.abspath(args.run_dir), gul, il=il, ri=ri)


def main(argv=None):
    """Parse the arguments, run the executor and write the statistics.

    Args:
        argv (list[str], optional): arguments; ``sys.argv[1:]`` if None.

    Returns:
        dict: the executor statistics.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunks <= 0:
        parser.error('--chunks must be positive')
    if args.workers <= 0:
        parser.error('--workers must be positive')
    if args.ri_layers < 0:
        parser.error('--ri-layers must not be negative')
    if 'ri' in args.taps and args.ri_layers == 0:
        parser.error("tap 'ri' needs --ri-layers >= 1")
    chunk_order = resolve_chunk_order(args.chunk_order, args.chunks)
    if chunk_order is not None:
        if args.executor != 'inline':
            parser.error('--chunk-order is only supported by the inline executor')
        if sorted(chunk_order) != list(range(1, args.chunks + 1)):
            parser.error(f'--chunk-order must be a permutation of 1..{args.chunks}')
    logging.basicConfig(level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    spec = spec_from_args(args)
    events = read_events(os.path.join(spec.run_dir, EVENTS_FILE))
    chunks = make_chunks(events, args.chunks)
    executor = get_executor(args.executor)
    options = {'chunk_order': chunk_order} if chunk_order is not None else {}
    stats = executor.run_chunks(spec, chunks, args.out_dir, args.taps, args.workers, **options)
    logger.info("executor %s: %d events in %d chunks, %.2f s", args.executor, len(events), args.chunks, stats.get('wall_seconds', 0.0))
    if args.stats_json:
        with open(args.stats_json, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    return stats


if __name__ == '__main__':
    main(sys.argv[1:])
