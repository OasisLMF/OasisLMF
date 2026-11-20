"""Run a command and report the CPU time and memory of its processes, grouped by tool.

The command's process tree is sampled with ``psutil`` every ``interval`` seconds. For each
process the last CPU times seen and the highest RSS seen are kept, so a process that lives
shorter than one interval may be missed and the CPU of the last interval of each process is
not counted: this is a profile, not an accounting. Shells (bash, sh, ...) are left out of the
table and the totals.

A process is named after its executable, or for a Python interpreter after the script it runs,
the module of ``-m``, or the tool named in the code of ``-c`` (as ``reference_streams`` and the
``ref-bin`` wrappers start the kernel tools).

Command line::

    python tests/orchestrator/tools/profile_kernel.py [--cwd DIR] [--json OUT] [--interval S] -- COMMAND [ARGS...]

Prints the summary and the per-tool table, optionally writes them as JSON, and exits with the
command's exit status.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time

import psutil

SHELLS = {'bash', 'sh', 'dash', 'zsh', 'ksh', 'fish', 'cmd', 'cmd.exe', 'powershell', 'powershell.exe', 'pwsh', 'time', 'env'}
PSUTIL_ERRORS = (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess)
CODE_TOOL_PATTERNS = (
    re.compile(r"sys\.argv\[0\]\s*=\s*['\"]([\w.-]+)['\"]"),
    re.compile(r'oasislmf\.pytools\.([\w.]+?)(?:\.cli|\.manager)?\s+import'),
)


def tool_name(cmdline):
    """Name of the tool a command line runs.

    Args:
        cmdline (list[str]): the process's command line.

    Returns:
        str | None: the tool name, None for an empty command line.
    """
    if not cmdline:
        return None
    exe = os.path.basename(cmdline[0])
    if not re.match(r'^python[\d.]*(\.exe)?$', exe, re.IGNORECASE):
        return exe
    args = cmdline[1:]
    while args and args[0].startswith('-') and args[0] not in ('-c', '-m'):
        args = args[1:]
    if not args:
        return exe
    if args[0] == '-m' and len(args) > 1:
        return args[1]
    if args[0] == '-c' and len(args) > 1:
        for pattern in CODE_TOOL_PATTERNS:
            match = pattern.search(args[1])
            if match:
                return match.group(1)
        return f'{exe} -c'
    return os.path.basename(args[0])


def _sample(processes, seen, now):
    for proc in processes:
        try:
            name = tool_name(proc.cmdline())
            if name is None or name in SHELLS:
                continue
            times = proc.cpu_times()
            rss = proc.memory_info().rss
        except PSUTIL_ERRORS:
            continue
        rec = seen.setdefault(proc.pid, {'name': name, 'first': now, 'rss': 0, 'user': 0.0, 'sys': 0.0})
        rec['name'] = name
        rec['user'], rec['sys'], rec['last'] = times.user, times.system, now
        rec['rss'] = max(rec['rss'], rss)


def profile_command(command, cwd=None, env=None, interval=0.05, shell=False):
    """Run a command to completion while sampling its process tree.

    Args:
        command (list[str] | str): the command; a string only with shell=True.
        cwd (str | None): working directory.
        env (dict | None): environment, the current one if None.
        interval (float): seconds between samples.
        shell (bool): run the command through the system shell.

    Returns:
        dict: ``command``, ``cwd``, ``returncode``, ``wall`` (s), ``cpu``, ``cpu_user`` and ``cpu_sys``
            (CPU-seconds summed over the sampled processes), ``peak_rss_sum`` (bytes: the sum of every
            process's own peak RSS, an upper bound on the memory in use at once), ``n_processes``,
            ``cores`` and ``tools``, a dict tool -> ``n``, ``user``, ``sys``, ``cpu_share`` (fraction of
            ``cpu``), ``avg_life``, ``begin`` and ``end`` (s from the start), ``peak_rss_sum`` and
            ``peak_rss_max`` (bytes).
    """
    start = time.time()
    root = subprocess.Popen(command, cwd=cwd, env=env, shell=shell)
    seen = {}
    try:
        parent = psutil.Process(root.pid)
        while root.poll() is None:
            now = time.time()
            try:
                processes = [parent] + parent.children(recursive=True)
            except PSUTIL_ERRORS:
                break
            _sample(processes, seen, now)
            time.sleep(interval)
    except PSUTIL_ERRORS:
        pass
    finally:
        root.wait()
    wall = time.time() - start

    tools = collections.defaultdict(lambda: {'n': 0, 'user': 0.0, 'sys': 0.0, 'life': 0.0, 'peak_rss_sum': 0, 'peak_rss_max': 0,
                                             'begin': float('inf'), 'end': 0.0})
    for rec in seen.values():
        tool = tools[rec['name']]
        tool['n'] += 1
        tool['user'] += rec['user']
        tool['sys'] += rec['sys']
        tool['life'] += rec['last'] - rec['first']
        tool['peak_rss_sum'] += rec['rss']
        tool['peak_rss_max'] = max(tool['peak_rss_max'], rec['rss'])
        tool['begin'] = min(tool['begin'], rec['first'] - start)
        tool['end'] = max(tool['end'], rec['last'] - start)
    cpu_user = sum(t['user'] for t in tools.values())
    cpu_sys = sum(t['sys'] for t in tools.values())
    cpu = cpu_user + cpu_sys
    table = {}
    for name, tool in sorted(tools.items(), key=lambda kv: -(kv[1]['user'] + kv[1]['sys'])):
        table[name] = {
            'n': tool['n'], 'user': tool['user'], 'sys': tool['sys'],
            'cpu_share': (tool['user'] + tool['sys']) / cpu if cpu else 0.0,
            'avg_life': tool['life'] / tool['n'], 'begin': tool['begin'], 'end': tool['end'],
            'peak_rss_sum': tool['peak_rss_sum'], 'peak_rss_max': tool['peak_rss_max'],
        }
    return {
        'command': command, 'cwd': cwd, 'returncode': root.returncode, 'wall': wall,
        'cpu': cpu, 'cpu_user': cpu_user, 'cpu_sys': cpu_sys,
        'peak_rss_sum': sum(t['peak_rss_sum'] for t in table.values()),
        'n_processes': len(seen), 'cores': os.cpu_count(), 'interval': interval, 'tools': table,
    }


def format_report(report):
    """Text report: one summary line, then one row per tool.

    Args:
        report (dict): as returned by profile_command.

    Returns:
        str: the report.
    """
    lines = [f"wall {report['wall']:.1f}s  exit {report['returncode']}  cpu {report['cpu']:.1f}s "
             f"(user {report['cpu_user']:.1f}s, sys {report['cpu_sys']:.1f}s)  summed peak rss {report['peak_rss_sum'] / 1e6:.0f} MB  "
             f"processes {report['n_processes']}  cores {report['cores']}",
             f"{'tool':<16}{'n':>5}{'user':>9}{'sys':>8}{'cpu%':>7}{'avg_life':>10}{'begin':>8}{'end':>8}{'rss_sum_MB':>12}{'rss_max_MB':>12}"]
    for name, t in report['tools'].items():
        lines.append(f"{name[:16]:<16}{t['n']:>5}{t['user']:>9.1f}{t['sys']:>8.1f}{100 * t['cpu_share']:>6.1f}%{t['avg_life']:>10.1f}"
                     f"{t['begin']:>8.1f}{t['end']:>8.1f}{t['peak_rss_sum'] / 1e6:>12.0f}{t['peak_rss_max'] / 1e6:>12.0f}")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cwd', default=None, help='working directory of the command')
    parser.add_argument('--json', dest='json_path', default=None, help='also write the report as JSON to this path')
    parser.add_argument('--interval', type=float, default=0.05, help='seconds between samples (default 0.05)')
    parser.add_argument('--shell', action='store_true', help='run the command through the system shell (pass it as one string)')
    parser.add_argument('command', nargs=argparse.REMAINDER, help='the command, after --')
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('no command given')
    if args.shell:
        command = ' '.join(command)
    report = profile_command(command, cwd=args.cwd, interval=args.interval, shell=args.shell)
    print(format_report(report))
    if args.json_path:
        with open(args.json_path, 'w') as out:
            json.dump(report, out, indent=2)
    return report['returncode']


if __name__ == '__main__':
    sys.exit(main())
