import json
import os
import subprocess
import sys
import threading
import time
import types

import psutil
import pytest

from oasislmf.execution.orchestrator import executor as executor_module
from oasislmf.execution.orchestrator.build import IGNORABLE_STDERR_LINE, ChunkCheck, Run
from oasislmf.execution.orchestrator.executor import connected_components, execute
from oasislmf.execution.orchestrator.model import Process, Stage, Stream, StreamPath

pytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='the Windows transport is not implemented')

PY = sys.executable
TEEPY = [PY, '-m', 'oasislmf.execution.orchestrator.tee']
TIMEOUT = 60
LOG = 'log'

WRITE_STDOUT = 'import sys; sys.stdout.buffer.write(sys.argv[1].encode() * int(sys.argv[2]))'
WRITE_PATH = 'import sys; f = open(sys.argv[1], "wb"); f.write(sys.argv[2].encode()); f.close()'
COPY = 'import sys, shutil; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)'
UPPER = 'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().upper())'
SLEEP_FOREVER = 'import time; time.sleep(600)'
WRITE_PID_AND_SLEEP = 'import os, sys, time; open(sys.argv[1], "w").write(str(os.getpid())); time.sleep(600)'
EXIT_AFTER = 'import sys, time; time.sleep(float(sys.argv[1])); sys.exit(int(sys.argv[2]))'
WAIT_FOR_FILE_THEN_EXIT = ('import os, sys, time\n'
                           'while not os.path.exists(sys.argv[1]): time.sleep(0.05)\n'
                           'sys.exit(int(sys.argv[2]))')
SPAWN_GRANDCHILD = ('import subprocess, sys, time\n'
                    'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])\n'
                    'open(sys.argv[1] + ".tmp", "w").write(str(p.pid))\n'
                    'import os; os.replace(sys.argv[1] + ".tmp", sys.argv[1])\n'
                    'time.sleep(600)')
WRITE_STDERR = 'import sys; sys.stderr.write(sys.argv[1]); sys.stderr.flush()'
WRITE_STDERR_AND_SLEEP = 'import sys, time; sys.stderr.write(sys.argv[1]); sys.stderr.flush(); time.sleep(600)'
WRITE_ENV = 'import os, sys; open(sys.argv[1], "w").write(os.environ.get(sys.argv[2], ""))'

PYTHON_WARNING = '/some/module.py:12: UserWarning: something is deprecated\n  warnings.warn("something is deprecated")\n\n'


def call(*args, **kwargs):
    """Run ``execute`` in a thread so a hang fails the test instead of blocking it."""
    result = {}

    def target():
        try:
            result['value'] = execute(*args, **kwargs)
        except BaseException as e:
            result['error'] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(TIMEOUT)
    if thread.is_alive():
        for child in psutil.Process().children(recursive=True):
            child.kill()
        thread.join(10)
        pytest.fail('execute did not return in time')
    if 'error' in result:
        raise result['error']
    return result['value']


def assert_no_children():
    children = psutil.Process().children(recursive=True)
    alive = [c for c in children if c.is_running() and c.status() != psutil.STATUS_ZOMBIE]
    assert alive == []


def assert_dead(pid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.05)
    pytest.fail(f'process {pid} survived the run')


def wait_for_file(path):
    deadline = time.monotonic() + TIMEOUT
    while not os.path.exists(path):
        assert time.monotonic() < deadline, f'{path} never written'
        time.sleep(0.05)


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    for child in psutil.Process().children(recursive=True):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def test_multi_stage_run_with_chain_and_tee_fan_out(tmp_path):
    raw, upper, fan, copy_a = Stream('raw'), Stream('upper'), Stream('fan'), Stream('copy_a')
    first = Stage('first', processes=[
        Process('sink_a', [PY, '-c', COPY], stdin=copy_a, stdout='out/a.txt'),
        Process('sink_b', [PY, '-c', COPY], stdin=fan, stdout='out/b.txt'),
        Process('tee', [*TEEPY, StreamPath(copy_a), '-'], stdin=upper, stdout=fan),
        Process('upper', [PY, '-c', UPPER], stdin=raw, stdout=upper),
        Process('src', [PY, '-c', WRITE_STDOUT, 'abc', '50000'], stdout=raw),
    ], make_dirs=['out'])
    second = Stage('second', processes=[
        Process('concat', [PY, '-c', COPY], stdin='out/a.txt', stdout='out/c.txt'),
        Process('env', [PY, '-c', WRITE_ENV, 'out/env.txt', 'OASIS_PYTOOLS_LOG_DIR']),
        Process('run_env', [PY, '-c', WRITE_ENV, 'out/run_env.txt', 'MY_RUN_VAR']),
    ])
    text = call(Run(stages=[first, second], env={'MY_RUN_VAR': 'hello'}), LOG)

    expected = b'ABC' * 50000
    assert (tmp_path / 'out/a.txt').read_bytes() == expected
    assert (tmp_path / 'out/b.txt').read_bytes() == expected
    assert (tmp_path / 'out/c.txt').read_bytes() == expected
    assert (tmp_path / 'out/env.txt').read_text() == LOG
    assert (tmp_path / 'out/run_env.txt').read_text() == 'hello'
    assert (tmp_path / LOG / 'stderror.err').read_bytes() == b''
    assert "stage 'first': starting 5 processes in 1 groups" in text
    assert "stage 'second': starting 3 processes in 3 groups" in text
    assert "stage 'second': finished" in text
    assert_no_children()


def test_log_dir_is_emptied_and_stderror_created(tmp_path):
    (tmp_path / LOG / 'sub').mkdir(parents=True)
    (tmp_path / LOG / 'old.log').write_text('old')
    (tmp_path / LOG / 'sub' / 'old.log').write_text('old')
    call(Run(stages=[Stage('empty')]), LOG)
    assert sorted(os.listdir(tmp_path / LOG)) == ['stderror.err']


def test_failure_kills_others_and_names_first_failure(tmp_path):
    stage = Stage('analysis', processes=[
        Process('sleeper_1', [PY, '-c', SLEEP_FOREVER]),
        Process('boom', [PY, '-c', EXIT_AFTER, '0.5', '3']),
        Process('sleeper_2', [PY, '-c', SLEEP_FOREVER]),
    ])
    start = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(Run(stages=[stage]), LOG, kill_grace=1)
    assert time.monotonic() - start < 20
    assert error.value.returncode == 3
    assert error.value.cmd[-1] == '3'
    output = error.value.output.decode()
    assert "stage 'analysis': process 'boom'" in output
    assert 'exited with code 3' in output
    assert 'sleeper' not in output.split('Kernel execution error:')[1]
    assert_no_children()


def test_reader_failure_is_reported_not_the_killed_writer(tmp_path):
    s = Stream('s')
    write_slowly = 'import sys, time\nwhile True:\n    sys.stdout.buffer.write(b"x"); sys.stdout.flush(); time.sleep(1)'
    stage = Stage('analysis', processes=[
        Process('reader', [PY, '-c', 'import sys; sys.exit(5)'], stdin=s),
        Process('writer', [PY, '-c', write_slowly], stdout=s),
    ])
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(Run(stages=[stage]), LOG, kill_grace=1)
    assert error.value.returncode == 5
    reason = error.value.output.decode().split('Kernel execution error:')[1]
    assert "'reader'" in reason
    assert 'writer' not in reason
    assert_no_children()


def test_grandchildren_are_killed(tmp_path):
    pid_file = str(tmp_path / 'grandchild.pid')
    stage = Stage('analysis', processes=[
        Process('parent', [PY, '-c', SPAWN_GRANDCHILD, pid_file]),
        Process('boom', [PY, '-c', WAIT_FOR_FILE_THEN_EXIT, pid_file, '2']),
    ])
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(Run(stages=[stage]), LOG, kill_grace=1)
    assert error.value.returncode == 2
    assert_dead(int(open(pid_file).read()))
    assert_no_children()


def test_stderr_error_line_fails_the_run(tmp_path):
    stage = Stage('analysis', processes=[
        Process('complainer', [PY, '-c', WRITE_STDERR_AND_SLEEP, 'Something went badly wrong\n']),
        Process('sleeper', [PY, '-c', SLEEP_FOREVER]),
    ])
    start = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(Run(stages=[stage]), LOG, poll_interval=0.1, kill_grace=1)
    assert time.monotonic() - start < 20
    assert error.value.returncode == 1
    output = error.value.output.decode()
    assert 'Something went badly wrong' in output
    assert "stage 'analysis'" in output
    assert_no_children()


def test_stderr_error_line_written_at_exit_is_caught_by_final_check(tmp_path):
    stage = Stage('analysis', processes=[Process('complainer', [PY, '-c', WRITE_STDERR, 'late error without newline'])])
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(Run(stages=[stage]), LOG, poll_interval=3600)
    assert 'late error without newline' in error.value.output.decode()


def test_python_warning_on_stderr_does_not_fail(tmp_path):
    stage = Stage('analysis', processes=[Process('warner', [PY, '-c', WRITE_STDERR, PYTHON_WARNING])])
    call(Run(stages=[stage]), LOG, poll_interval=0.1)
    assert (tmp_path / LOG / 'stderror.err').read_text() == PYTHON_WARNING


def test_gul_stderror_is_not_watched(tmp_path):
    stage = Stage('analysis', processes=[
        Process('gul', [PY, '-c', WRITE_STDERR, 'custom gulcalc chatter\n'], stderr_log='gul_stderror.err')])
    call(Run(stages=[stage]), LOG, poll_interval=0.1)
    assert (tmp_path / LOG / 'gul_stderror.err').read_text() == 'custom gulcalc chatter\n'


def test_stderr_guard_off_ignores_stderr_content(tmp_path):
    stage = Stage('analysis', processes=[Process('complainer', [PY, '-c', WRITE_STDERR, 'an error line\n'])])
    call(Run(stages=[stage]), LOG, stderr_guard=False, poll_interval=0.1)
    assert (tmp_path / LOG / 'stderror.err').read_text() == 'an error line\n'


def test_unguarded_process_inherits_stderr(tmp_path, capfd):
    stage = Stage('analysis', processes=[Process('loud', [PY, '-c', WRITE_STDERR, 'to parent stderr\n'], stderr_log=None)])
    call(Run(stages=[stage]), LOG)
    assert 'to parent stderr' in capfd.readouterr().err
    assert (tmp_path / LOG / 'stderror.err').read_text() == ''


def test_background_killed_on_success(tmp_path):
    pid_file = str(tmp_path / 'bg.pid')
    run = Run(stages=[Stage('analysis', processes=[Process('waiter', [PY, '-c', WAIT_FOR_FILE_THEN_EXIT, pid_file, '0'])])],
              background=[Process('server', [PY, '-c', WRITE_PID_AND_SLEEP, pid_file], stderr_log=None)])
    text = call(run, LOG, kill_grace=1)
    assert_dead(int(open(pid_file).read()))
    assert 'background processes stopped (1)' in text
    assert_no_children()


def test_background_killed_on_failure(tmp_path):
    pid_file = str(tmp_path / 'bg.pid')
    run = Run(stages=[Stage('analysis', processes=[Process('boom', [PY, '-c', WAIT_FOR_FILE_THEN_EXIT, pid_file, '4'])])],
              background=[Process('server', [PY, '-c', WRITE_PID_AND_SLEEP, pid_file], stderr_log=None)])
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(run, LOG, kill_grace=1)
    assert error.value.returncode == 4
    assert_dead(int(open(pid_file).read()))
    assert_no_children()


def test_missing_executable(tmp_path):
    stage = Stage('analysis', processes=[
        Process('sleeper', [PY, '-c', SLEEP_FOREVER]),
        Process('ghost', ['oasis-no-such-executable-xyz', '-a', '1']),
    ])
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(Run(stages=[stage]), LOG, kill_grace=1)
    assert error.value.returncode == 127
    output = error.value.output.decode()
    assert "executable not found for process 'ghost'" in output
    assert 'oasis-no-such-executable-xyz' in output
    assert_no_children()


def test_keyboard_interrupt_cleans_up(tmp_path, monkeypatch):
    pid_file = str(tmp_path / 'sleeper.pid')
    stage = Stage('analysis', processes=[Process('sleeper', [PY, '-c', WRITE_PID_AND_SLEEP, pid_file])])
    real_sleep = time.sleep

    def interrupted_sleep(seconds):
        if os.path.exists(pid_file):
            raise KeyboardInterrupt
        real_sleep(seconds)

    monkeypatch.setattr(executor_module, 'time', types.SimpleNamespace(monotonic=time.monotonic, sleep=interrupted_sleep))
    with pytest.raises(KeyboardInterrupt):
        call(Run(stages=[stage]), LOG, kill_grace=1)
    assert_dead(int(open(pid_file).read()))
    assert_no_children()


def make_tree(root, files):
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('x')


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def test_filesystem_steps_and_order(tmp_path):
    make_tree(tmp_path, ['work/old/P1.bin', 'work/old.bin', 'output/gul_S1_eltcalc.csv', 'output/gul_S1_summary-info.csv',
                         'output/analysis_settings.json', 'output/sub/il_S1_pltcalc.csv', 'keep.txt'])
    listing = ('import os, sys\npaths = []\nfor d in ("work", "output"):\n    for r, ds, fs in os.walk(d):\n'
               '        paths.append(r)\n        paths += [os.path.join(r, f) for f in fs]\n'
               'open(sys.argv[1], "w").write("\\n".join(sorted(paths)))')
    first = Stage('analysis', processes=[Process('lister', [PY, '-c', listing, 'listing.txt'])],
                  remove_paths=['work/*'], clean_output_dir='output', make_dirs=['work/kat'])
    second = Stage('scratch', processes=[Process('scratch', [PY, '-c', WRITE_PATH, 'work/kat/tmp_scratch', 'x'])],
                   cleanup_paths=['work/kat/tmp_*'])
    call(Run(stages=[first, second]), LOG)
    seen = (tmp_path / 'listing.txt').read_text().split('\n')
    assert seen == sorted(['output', os.path.join('output', 'analysis_settings.json'), os.path.join('output', 'gul_S1_summary-info.csv'),
                           os.path.join('output', 'sub'), 'work', os.path.join('work', 'kat')])
    assert not (tmp_path / 'work/kat/tmp_scratch').exists()
    assert (tmp_path / 'work/kat').is_dir()
    assert (tmp_path / 'keep.txt').exists()


def test_cleanup_paths_not_applied_on_failure(tmp_path):
    make_tree(tmp_path, ['work/keep.bin'])
    stage = Stage('analysis', processes=[Process('boom', [PY, '-c', EXIT_AFTER, '0', '1'])], cleanup_paths=['work/*'])
    with pytest.raises(subprocess.CalledProcessError):
        call(Run(stages=[stage]), LOG)
    assert (tmp_path / 'work/keep.bin').exists()


def test_stage_without_processes_only_does_filesystem_work(tmp_path):
    make_tree(tmp_path, ['work/a.bin', 'work/kat/b.bin'])
    text = call(Run(stages=[Stage('cleanup', remove_paths=['work/*'])]), LOG)
    assert tree(tmp_path / 'work') == []
    assert "stage 'cleanup': finished" in text


def chunk_run():
    return Run(stages=[Stage('analysis')],
               chunk_check=ChunkCheck(stderr_log='stderror.err', ignorable_stderr_line=IGNORABLE_STDERR_LINE,
                                      non_empty_glob='1.work/**/P1.bin'))


def test_chunk_check_passes(tmp_path):
    make_tree(tmp_path, ['1.work/kat/gul_S1_elt_P1', '1.work/gul_S1_summaryleccalc/P1.bin', '1.work/gul_S1_summary_palt/P1.bin'])
    text = call(chunk_run(), LOG)
    assert '=== Checking analysis output chunks ===' in text
    assert f"1 {os.path.join('1.work', 'gul_S1_summaryleccalc', 'P1.bin')}" in text
    assert 'Chunk output check [OK]' in text


def test_chunk_check_fails_on_empty_file(tmp_path):
    make_tree(tmp_path, ['1.work/gul_S1_summaryleccalc/P1.bin'])
    (tmp_path / '1.work/gul_S1_summary_palt').mkdir()
    (tmp_path / '1.work/gul_S1_summary_palt/P1.bin').write_bytes(b'')
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(chunk_run(), LOG)
    assert f"Chunk output error: File '{os.path.join('1.work', 'gul_S1_summary_palt', 'P1.bin')}' is empty" in error.value.output.decode()


def test_chunk_check_fails_on_stderr_error_line(tmp_path):
    run = chunk_run()
    run.stages[0].processes = [Process('complainer', [PY, '-c', WRITE_STDERR, PYTHON_WARNING + 'real error\n'])]
    with pytest.raises(subprocess.CalledProcessError) as error:
        call(run, LOG, stderr_guard=False)
    assert 'real error' in error.value.output.decode()


def test_manifest_content(tmp_path):
    s, t = Stream('gul_P1'), Stream('pipe_1')
    stage = Stage('analysis', processes=[
        Process('summarypy', ['summarypy', '-t', 'gul'], stdin=s),
        Process('teepy', ['teepy', StreamPath(s), '-'], stdin=t),
        Process('gulmc', ['gulmc', '-S10'], stdout=t, stderr_log=None),
    ], make_dirs=['work/kat'], remove_paths=['work/*'], clean_output_dir='output')
    run = Run(stages=[stage, Stage('cleanup', remove_paths=['work/*'])],
              background=[Process('socket-server', ['socket-server', '10'], stderr_log=None)], env={'A': '1'},
              chunk_check=ChunkCheck('stderror.err', IGNORABLE_STDERR_LINE, '1.work/**/P1.bin'))
    executor_module.write_manifest(run, LOG, 'run_kernel.json')
    data = json.loads((tmp_path / 'run_kernel.json').read_text())
    assert data['log_dir'] == LOG
    assert data['env'] == {'A': '1'}
    assert data['background'] == [{'name': 'socket-server', 'argv': ['socket-server', '10'], 'stdin': None, 'stdout': None,
                                   'stderr_log': None}]
    first = data['stages'][0]
    assert first['name'] == 'analysis'
    assert first['make_dirs'] == ['work/kat']
    assert first['remove_paths'] == ['work/*']
    assert first['clean_output_dir'] == 'output'
    assert first['processes'][1] == {'name': 'teepy', 'argv': ['teepy', '<stream:gul_P1>', '-'], 'stdin': '<stream:pipe_1>',
                                     'stdout': None, 'stderr_log': 'stderror.err'}
    assert first['processes'][2]['stdout'] == '<stream:pipe_1>'
    assert data['stages'][1] == {'name': 'cleanup', 'remove_paths': ['work/*'], 'clean_output_dir': None, 'make_dirs': [],
                                 'cleanup_paths': [], 'processes': []}
    assert data['chunk_check']['non_empty_glob'] == '1.work/**/P1.bin'
    assert (tmp_path / 'run_kernel.json').read_text().startswith('{\n  "log_dir"')


def test_manifest_written_by_execute_before_processes_start(tmp_path):
    check = 'import json, sys; json.load(open(sys.argv[1]))["stages"][0]["processes"][0]["name"] == "checker" or sys.exit(9)'
    stage = Stage('analysis', processes=[Process('checker', [PY, '-c', check, 'manifest/run.json'])])
    call(Run(stages=[stage]), LOG, manifest_path='manifest/run.json')


def test_connected_components_keep_listed_order():
    a, b, c = Stream('a'), Stream('b'), Stream('c')
    processes = [
        Process('read_a', ['x'], stdin=a),
        Process('read_c', ['x'], stdin=c),
        Process('lone', ['x']),
        Process('tee_b', ['x', StreamPath(a)], stdin=b),
        Process('write_c', ['x'], stdout=c),
        Process('write_b', ['x'], stdout=b),
    ]
    groups = connected_components(Stage('s', processes=processes))
    assert [[p.name for p in group] for group in groups] == [['read_a', 'tee_b', 'write_b'], ['read_c', 'write_c'], ['lone']]


@pytest.mark.skipif(not hasattr(os, 'pipe') or sys.platform == 'win32', reason='POSIX rlimit')
def test_many_components_do_not_exhaust_file_descriptors(tmp_path):
    resource = pytest.importorskip('resource')
    n_chains = 300
    streams = [Stream(f's{i}') for i in range(n_chains)]
    readers = [Process(f'sink_{i}', ['cat'], stdin=s, stdout=f'out/{i}.txt') for i, s in enumerate(streams)]
    writers = [Process(f'src_{i}', ['echo', str(i)], stdout=s) for i, s in enumerate(streams)]
    stage = Stage('analysis', processes=readers + writers, make_dirs=['out'])

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    n_open = len(os.listdir('/proc/self/fd')) if os.path.isdir('/proc/self/fd') else 64
    limit = max(n_open + 64, 128)
    assert limit < n_open + n_chains
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))
    try:
        pipes = []
        with pytest.raises(OSError):
            for _ in range(n_chains):
                pipes.append(os.pipe())
        for r, w in pipes:
            os.close(r)
            os.close(w)
        text = call(Run(stages=[stage]), LOG)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert f'starting {2 * n_chains} processes in {n_chains} groups' in text
    assert all((tmp_path / f'out/{i}.txt').read_text() == f'{i}\n' for i in range(n_chains))
