"""Produce reference per-chunk loss streams with the command-line kernel tools.

For each chunk k of K this runs, without a shell, the pipeline the bash kernel script runs::

    evepy k K | gulmc -S<samples> <gul args> | tee gul_P<k>.bin
              | fmpy -a<il rule> | tee il_P<k>.bin
              | fmpy -a<ri rule> -p input/RI_1 -n - | ... | fmpy -a<ri rule> -p input/RI_<n> -n - > ri_P<k>.bin

The processes are connected with ``subprocess`` pipes, and a ``tee`` is a thread copying a
process's output to both a file and the next process, so the same code runs on any OS.

The tools are either those of the running interpreter (``sys.executable -c "from
oasislmf.pytools.<tool>.cli import main; ..."``) or executables in a directory, such as
wrappers around a pristine checkout.

Command line::

    python tests/orchestrator/tools/reference_streams.py RUN_DIR K OUT_DIR [--samples 10] [--jobs 6]
        [--tools-dir DIR] [--il-alloc-rule 2] [--ri-alloc-rule 3] [--ri-layers 1] [--gul-args "..."]
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

MAX_JOBS = 6
COPY_BUFFER = 1 << 20

DEFAULT_SAMPLES = 10
DEFAULT_GUL_ARGS = ('--random-generator=2', '--model-df-engine=oasis_data_manager.df_reader.reader.OasisPandasReader',
                    '--vuln-cache-size', '200', '-L0', '-a1')
DEFAULT_IL_ALLOC_RULE = 2
DEFAULT_RI_ALLOC_RULE = 3
DEFAULT_RI_LAYERS = 1

TOOL_MODULES = {
    'evepy': 'oasislmf.pytools.eve.cli',
    'gulmc': 'oasislmf.pytools.gulmc.cli',
    'fmpy': 'oasislmf.pytools.fm.cli',
}


@dataclass
class Tools:
    """Command prefixes of the kernel tools: each is a list of arguments the tool's own arguments are appended to."""
    evepy: list
    gulmc: list
    fmpy: list


def tools_from_python(python=sys.executable):
    """The tools of an interpreter's installed ``oasislmf``.

    Args:
        python (str): interpreter to run them with.

    Returns:
        Tools: commands of the form ``python -c "from oasislmf.pytools.<tool>.cli import main; ..."``.
    """
    def command(name):
        code = f"import sys; from {TOOL_MODULES[name]} import main; sys.argv[0] = {name!r}; sys.exit(main())"
        return [python, '-c', code]
    return Tools(command('evepy'), command('gulmc'), command('fmpy'))


def tools_from_dir(directory):
    """Executables ``evepy``, ``gulmc`` and ``fmpy`` in a directory.

    Args:
        directory (str | Path): directory holding the executables (or ``.exe``/``.cmd`` files on Windows).

    Returns:
        Tools: their absolute paths.

    Raises:
        FileNotFoundError: if one of them is missing.
    """
    directory = Path(directory).resolve()
    found = {}
    for name in TOOL_MODULES:
        path = shutil.which(name, path=str(directory))
        if path is None:
            raise FileNotFoundError(f'{name} not found in {directory}')
        found[name] = [path]
    return Tools(**found)


def resolve_tools(tools=None):
    """Tools given as a Tools, a directory, or None for the running interpreter's.

    Args:
        tools (Tools | str | Path | None): what to resolve.

    Returns:
        Tools: the resolved tools.
    """
    if tools is None:
        return tools_from_python()
    if isinstance(tools, Tools):
        return tools
    return tools_from_dir(tools)


def fm_static_paths(ri_layers):
    """Static directories of the fm steps, relative to the run directory.

    Args:
        ri_layers (int): number of reinsurance inuring priorities.

    Returns:
        list[str]: ``input`` then ``input/RI_1`` ... ``input/RI_<ri_layers>``.
    """
    return ['input'] + [os.path.join('input', f'RI_{i}') for i in range(1, ri_layers + 1)]


def prepare_financial_structures(run_dir, tools=None, il_alloc_rule=DEFAULT_IL_ALLOC_RULE, ri_alloc_rule=DEFAULT_RI_ALLOC_RULE,
                                 ri_layers=DEFAULT_RI_LAYERS, force=False):
    """Write the fmpy financial structure files (``compute_info_<rule>.npy`` ...) that the fm steps load.

    The bash kernel script does this with ``fmpy --create-financial-structure-files`` before the
    pipelines start. fmpy computes rule 3 with the rule 2 structure.

    Args:
        run_dir (str | Path): prepared run directory.
        tools (Tools | str | Path | None): tools to use, see resolve_tools.
        il_alloc_rule (int | None): IL allocation rule, None for no IL step.
        ri_alloc_rule (int): RI allocation rule.
        ri_layers (int): number of RI steps.
        force (bool): rewrite the files even when they exist.
    """
    if il_alloc_rule is None:
        return
    tools = resolve_tools(tools)
    rules = [il_alloc_rule] + [ri_alloc_rule] * ri_layers
    for static_path, rule in zip(fm_static_paths(ri_layers), rules):
        stored_rule = 2 if rule == 3 else rule
        if not force and os.path.exists(os.path.join(run_dir, static_path, f'compute_info_{stored_rule}.npy')):
            continue
        subprocess.run(tools.fmpy + [f'-a{rule}', '--create-financial-structure-files', '-p', static_path],
                       cwd=run_dir, check=True, capture_output=True)


def pipeline_commands(k, n_chunks, tools, samples=DEFAULT_SAMPLES, gul_args=DEFAULT_GUL_ARGS, il_alloc_rule=DEFAULT_IL_ALLOC_RULE,
                      ri_alloc_rule=DEFAULT_RI_ALLOC_RULE, ri_layers=DEFAULT_RI_LAYERS, fm_args=()):
    """The commands of one chunk's pipeline and the tap each command's output is.

    Args:
        k (int): chunk number, 1-based.
        n_chunks (int): number of chunks K.
        tools (Tools): the tools.
        samples (int): number of samples.
        gul_args (iterable[str]): gulmc arguments besides -S.
        il_alloc_rule (int | None): IL allocation rule, None to stop after gul.
        ri_alloc_rule (int): RI allocation rule.
        ri_layers (int): number of RI steps; the ri tap is the net output of the last.
        fm_args (iterable[str]): extra arguments for every fmpy step.

    Returns:
        list[tuple[str, list[str]]]: (tap or None, command) in pipeline order. A tap name means
            that command's output is written to ``<tap>_P<k>.bin``.
    """
    commands = [(None, tools.evepy + [str(k), str(n_chunks)]),
                ('gul', tools.gulmc + [f'-S{samples}'] + list(gul_args))]
    if il_alloc_rule is None:
        return commands
    commands.append(('il', tools.fmpy + [f'-a{il_alloc_rule}'] + list(fm_args)))
    for i, static_path in enumerate(fm_static_paths(ri_layers)[1:], start=1):
        tap = 'ri' if i == ri_layers else None
        commands.append((tap, tools.fmpy + [f'-a{ri_alloc_rule}', '-p', static_path, '-n', '-'] + list(fm_args)))
    return commands


def _write_all(sink, data):
    view = memoryview(data)
    while view:
        view = view[sink.write(view):]


def _tee(source, file_path, sink, errors):
    """Copy ``source`` to a file and to ``sink`` until end of file, then close both ends."""
    try:
        with open(file_path, 'wb') as out:
            while True:
                data = source.read(COPY_BUFFER)
                if not data:
                    break
                out.write(data)
                if sink is not None:
                    try:
                        _write_all(sink, data)
                    except BrokenPipeError:
                        errors.append(f'the reader of {file_path.name} closed its input early')
                        sink = None
    except Exception as exc:  # reported with the process failures
        errors.append(f'copy to {file_path}: {exc!r}')
    finally:
        source.close()
        if sink is not None:
            try:
                sink.close()
            except BrokenPipeError:
                pass


class PipelineError(RuntimeError):
    """A chunk pipeline failed: a tool exited non-zero or a copy failed."""


def run_pipeline(run_dir, k, n_chunks, out_dir, tools=None, env=None, **options):
    """Run one chunk's pipeline and write its taps to ``out_dir``.

    Args:
        run_dir (str | Path): prepared run directory, the tools' working directory.
        k (int): chunk number, 1-based.
        n_chunks (int): number of chunks K.
        out_dir (str | Path): where ``<tap>_P<k>.bin`` are written.
        tools (Tools | str | Path | None): tools to use, see resolve_tools.
        env (dict | None): environment of the tools, the current one if None.
        **options: pipeline_commands keyword arguments.

    Returns:
        dict: tap -> Path of the file written.

    Raises:
        PipelineError: with each failing tool's command, exit status and the end of its stderr.
    """
    tools = resolve_tools(tools)
    out_dir = Path(out_dir)
    commands = pipeline_commands(k, n_chunks, tools, **options)
    procs, threads, errors, written, stderr_files = [], [], [], {}, []
    upstream, tee_from = None, None
    try:
        for i, (tap, command) in enumerate(commands):
            last = i == len(commands) - 1
            path = out_dir / f'{tap}_P{k}.bin' if tap else None
            if path:
                written[tap] = path
            if tee_from is not None:
                stdin = subprocess.PIPE
            else:
                stdin = upstream if upstream is not None else subprocess.DEVNULL
            stdout = open(path, 'wb') if last else subprocess.PIPE
            stderr = tempfile.TemporaryFile()
            stderr_files.append(stderr)
            try:
                proc = subprocess.Popen(command, cwd=run_dir, env=env, stdin=stdin, stdout=stdout, stderr=stderr, bufsize=0)
            finally:
                if last:
                    stdout.close()
            procs.append(proc)
            if tee_from is not None:
                thread = threading.Thread(target=_tee, args=(*tee_from, proc.stdin, errors), daemon=True)
                thread.start()
                threads.append(thread)
            elif upstream is not None:
                upstream.close()
            upstream, tee_from = (None, (proc.stdout, path)) if path and not last else (proc.stdout, None)
    except BaseException:
        for proc in procs:
            proc.kill()
        raise
    finally:
        for thread in threads:
            thread.join()
        for proc in procs:
            proc.wait()
    failures = []
    for (tap, command), proc, stderr in zip(commands, procs, stderr_files):
        if proc.returncode != 0:
            stderr.seek(0)
            tail = stderr.read().decode(errors='replace')[-4000:]
            failures.append(f'{shlex.join(command)} exited {proc.returncode}:\n{tail}')
        stderr.close()
    if failures or errors:
        raise PipelineError(f'chunk {k}/{n_chunks} in {run_dir} failed:\n' + '\n'.join(errors + failures))
    return written


def make_reference_streams(run_dir, n_chunks, out_dir, samples=DEFAULT_SAMPLES, gul_args=DEFAULT_GUL_ARGS,
                           il_alloc_rule=DEFAULT_IL_ALLOC_RULE, ri_alloc_rule=DEFAULT_RI_ALLOC_RULE, ri_layers=DEFAULT_RI_LAYERS,
                           fm_args=(), tools=None, jobs=None, env=None, prepare=True):
    """Write the reference streams ``<tap>_P<k>.bin`` of every chunk of a K-chunk run.

    Args:
        run_dir (str | Path): prepared run directory (``input/`` and ``static/``).
        n_chunks (int): number of chunks K, as in ``evepy k K``.
        out_dir (str | Path): output directory, created if needed.
        samples (int): number of samples (gulmc -S).
        gul_args (iterable[str]): gulmc arguments besides -S.
        il_alloc_rule (int | None): IL allocation rule, None for the gul tap only.
        ri_alloc_rule (int): RI allocation rule.
        ri_layers (int): RI steps (``input/RI_1`` ...); 0 for no ri tap.
        fm_args (iterable[str]): extra arguments for every fmpy step.
        tools (Tools | str | Path | None): the running interpreter's tools if None, else a Tools or a directory of executables.
        jobs (int | None): chunk pipelines run at once, default min(MAX_JOBS, CPUs, K).
        env (dict | None): environment of the tools, the current one if None.
        prepare (bool): create missing fmpy financial structure files first.

    Returns:
        dict: tap -> list of Paths, in chunk order.

    Raises:
        PipelineError: if any chunk's pipeline fails (all chunks are still waited for).
    """
    run_dir = Path(run_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tools = resolve_tools(tools)
    if prepare:
        prepare_financial_structures(run_dir, tools, il_alloc_rule, ri_alloc_rule, ri_layers)
    options = dict(samples=samples, gul_args=tuple(gul_args), il_alloc_rule=il_alloc_rule, ri_alloc_rule=ri_alloc_rule,
                   ri_layers=ri_layers, fm_args=tuple(fm_args))
    jobs = max(1, min(jobs or MAX_JOBS, os.cpu_count() or 1, n_chunks))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_pipeline, run_dir, k, n_chunks, out_dir, tools, env, **options) for k in range(1, n_chunks + 1)]
    results, failures = {}, []
    for future in futures:
        try:
            for tap, path in future.result().items():
                results.setdefault(tap, []).append(path)
        except PipelineError as exc:
            failures.append(str(exc))
    if failures:
        raise PipelineError('\n\n'.join(failures))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('run_dir', help='prepared run directory')
    parser.add_argument('n_chunks', type=int, help='number of chunks K')
    parser.add_argument('out_dir', help='output directory for <tap>_P<k>.bin')
    parser.add_argument('--samples', '-S', type=int, default=DEFAULT_SAMPLES)
    parser.add_argument('--gul-args', default=shlex.join(DEFAULT_GUL_ARGS), help='gulmc arguments besides -S, as one string')
    parser.add_argument('--il-alloc-rule', type=int, default=DEFAULT_IL_ALLOC_RULE, help='-1 for gul only')
    parser.add_argument('--ri-alloc-rule', type=int, default=DEFAULT_RI_ALLOC_RULE)
    parser.add_argument('--ri-layers', type=int, default=DEFAULT_RI_LAYERS, help='0 for no ri tap')
    parser.add_argument('--fm-args', default='', help='extra arguments for every fmpy step, as one string')
    parser.add_argument('--tools-dir', default=None, help='directory with evepy, gulmc, fmpy (default: this interpreter\'s oasislmf)')
    parser.add_argument('--jobs', '-j', type=int, default=None, help=f'pipelines at once (default min({MAX_JOBS}, CPUs, K))')
    args = parser.parse_args(argv)
    start = time.perf_counter()
    results = make_reference_streams(
        args.run_dir, args.n_chunks, args.out_dir, samples=args.samples, gul_args=shlex.split(args.gul_args),
        il_alloc_rule=None if args.il_alloc_rule < 0 else args.il_alloc_rule, ri_alloc_rule=args.ri_alloc_rule,
        ri_layers=args.ri_layers, fm_args=shlex.split(args.fm_args), tools=args.tools_dir, jobs=args.jobs)
    total = sum(path.stat().st_size for paths in results.values() for path in paths)
    print(f'wrote {sum(len(p) for p in results.values())} files ({total} bytes) for taps {", ".join(results)} '
          f'to {args.out_dir} in {time.perf_counter() - start:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
