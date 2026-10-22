import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

import pytest

from oasislmf.model_execution.bash import (add_server_call, bash_params, bash_wrapper, print_command, progress_server_message,
                                           create_bash_analysis,
                                           create_bash_outputs, genbash)
from oasislmf.utils import diff
from oasislmf.utils.socket_server import GulProgressServer

TEST_DIRECTORY = os.path.dirname(__file__)
UPDATE_BASH_TESTS = os.environ.get('UPDATE_BASH_TESTS', '').lower() in ('1', 'true', 'yes')


class GenbashBase(TestCase):
    """Base class for genbash tests with dynamic test generation"""
    KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "input_json_settings")
    KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_base")
    KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_base")

    @classmethod
    def setUpClass(cls):

        # Default parameters
        cls.ri_iterations = 0
        cls.gul_alloc_rule = 0
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 2
        cls.num_gul_per_lb = 0
        cls.num_fm_per_lb = 0
        cls.event_shuffle = 1
        cls.gulmc = True
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.summarypy_low_memory = True

    @classmethod
    def tearDownClass(cls):
        if not UPDATE_BASH_TESTS:
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER, ignore_errors=True)

    def setUp(self):
        self.temp_reference_file = None

    def tearDown(self):
        if self.temp_reference_file:
            self.temp_reference_file.close()
            if os.path.exists(self.temp_reference_file.name):
                os.remove(self.temp_reference_file.name)

    def _load_analysis_settings(self, name):
        """Load analysis settings from JSON file"""
        input_filename = os.path.join(self.KPARSE_INPUT_FOLDER, f"{name}.json")
        with open(input_filename, encoding='utf-8') as file:
            return json.load(file)['analysis_settings']

    def _get_output_filename(self, name, num_partitions, num_reinsurance_iterations=None):
        """Generate output filename based on parameters"""
        if not num_reinsurance_iterations:
            return os.path.join(self.KPARSE_OUTPUT_FOLDER, f"{name}_{num_partitions}_partition.sh")
        return os.path.join(
            self.KPARSE_OUTPUT_FOLDER,
            f"{name}_{num_reinsurance_iterations}_reins_layer_{num_partitions}_partition.sh")

    def genbash(self, name, num_partitions, num_reinsurance_iterations=None,
                fifo_tmp_dir=None, stderr_guard=None, gulmc=True, gul_alloc_rule=None,
                il_alloc_rule=None, ri_alloc_rule=None, bash_trace=None,
                _get_getmodel_cmd=None):
        """Generate bash script from analysis settings"""

        analysis_settings = self._load_analysis_settings(name)
        output_filename = self._get_output_filename(name, num_partitions, num_reinsurance_iterations)

        genbash(
            num_partitions,
            analysis_settings,
            filename=output_filename,
            num_reinsurance_iterations=num_reinsurance_iterations or self.ri_iterations,
            fifo_tmp_dir=fifo_tmp_dir or self.fifo_tmp_dir,
            stderr_guard=stderr_guard or self.stderr_guard,
            gulmc=gulmc or self.gulmc,
            gul_alloc_rule=gul_alloc_rule or self.gul_alloc_rule,
            il_alloc_rule=il_alloc_rule or self.il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule or self.ri_alloc_rule,
            num_gul_per_lb=self.num_gul_per_lb,
            num_fm_per_lb=self.num_fm_per_lb,
            event_shuffle=self.event_shuffle,
            bash_trace=bash_trace or self.bash_trace,
            _get_getmodel_cmd=_get_getmodel_cmd,
            summarypy_low_memory=self.summarypy_low_memory,
        )

    def gen_chunked_bash(self, name, num_partitions, num_reinsurance_iterations=None,
                         fifo_tmp_dir=None, stderr_guard=None, gulmc=True, gul_alloc_rule=None,
                         il_alloc_rule=None, ri_alloc_rule=None, bash_trace=None,
                         _get_getmodel_cmd=None):
        """Generate chunked bash scripts (one per partition plus output script)"""

        analysis_settings = self._load_analysis_settings(name)
        output_base = self._get_output_filename(name, num_partitions, num_reinsurance_iterations).replace('.sh', '')

        params = bash_params(
            max_process_id=num_partitions,
            analysis_settings=analysis_settings,
            num_reinsurance_iterations=num_reinsurance_iterations or self.ri_iterations,
            fifo_tmp_dir=fifo_tmp_dir or self.fifo_tmp_dir,
            stderr_guard=stderr_guard or self.stderr_guard,
            gulmc=gulmc or self.gulmc,
            gul_alloc_rule=gul_alloc_rule or self.gul_alloc_rule,
            il_alloc_rule=il_alloc_rule or self.il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule or self.ri_alloc_rule,
            num_gul_per_lb=self.num_gul_per_lb,
            num_fm_per_lb=self.num_fm_per_lb,
            event_shuffle=self.event_shuffle,
            bash_trace=bash_trace or self.bash_trace,
            _get_getmodel_cmd=_get_getmodel_cmd,
            summarypy_low_memory=self.summarypy_low_memory,
        )

        # Generate partition scripts
        fifo_tmp_dir = params['fifo_tmp_dir']
        for process_id in range(num_partitions):
            params['filename'] = f'{output_base}.{process_id}.sh'
            if os.path.exists(params['filename']):
                os.remove(params['filename'])

            with bash_wrapper(
                params['filename'],
                bash_trace or self.bash_trace,
                stderr_guard or self.stderr_guard,
                custom_gulcalc_log_start=params['custom_gulcalc_log_start'],
                custom_gulcalc_log_finish=params['custom_gulcalc_log_finish'],
            ):
                create_bash_analysis(
                    **{**params, 'process_number': process_id + 1, 'fifo_tmp_dir': fifo_tmp_dir}
                )
                fifo_tmp_dir = False

        # Generate output script
        params['filename'] = f'{output_base}.output.sh'
        if os.path.exists(params['filename']):
            os.remove(params['filename'])

        with bash_wrapper(
            params['filename'],
            bash_trace or self.bash_trace,
            stderr_guard or self.stderr_guard,
            custom_gulcalc_log_start=params['custom_gulcalc_log_start'],
            custom_gulcalc_log_finish=params['custom_gulcalc_log_finish'],
        ):
            create_bash_outputs(**params)

    def check_chunks(self, name, num_partitions):
        for i in range(num_partitions):
            self.check(f'{name}.{i}')
        self.check(f'{name}.output')

    def check(self, name, reference_filename=None):
        """Compare generated output with reference file"""
        output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, f"{name}.sh")
        if not reference_filename:
            reference_filename = os.path.join(self.KPARSE_REFERENCE_FOLDER, f"{name}.sh")

        # Handle dynamic FIFO directory replacement
        if self.fifo_tmp_dir:
            reference_filename = self._prepare_dynamic_reference(output_filename, reference_filename)

        d = diff.unified_diff(reference_filename, output_filename, as_string=True)
        if d:
            self.fail(d)

    def _prepare_dynamic_reference(self, output_filename, ref_template):
        """Create temp reference file with dynamic FIFO directory replaced"""
        # Clean up any previous temp reference file (check_chunks calls check() multiple times
        # per test, and we must not leak earlier temp files when overwriting self.temp_reference_file)
        if self.temp_reference_file:
            if os.path.exists(self.temp_reference_file.name):
                os.remove(self.temp_reference_file.name)
            self.temp_reference_file = None

        # Extract the full fifo base path from the generated script.
        # Three patterns are tried in order:
        #   1. Analysis partition-0: "mkdir -p .../fifo/"
        #   2. Output script:        "rm -R -f .../" (absolute path, no wildcard)
        #   3. Partition scripts 1+: any line with an absolute ".../fifo/" path
        #      (e.g. "find .../fifo/ ..." or "mkfifo .../fifo/name")
        fifo_base = None
        with open(output_filename, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith('mkdir -p') and stripped.endswith('/fifo/'):
                    path = stripped.split()[-1]
                    fifo_base = path[:path.rindex('/fifo/')]
                    break
                if stripped.startswith('rm -R -f') and not stripped.endswith('*'):
                    path = stripped.split()[-1].rstrip('/')
                    if os.path.isabs(path):
                        fifo_base = path
                        break
                match = re.search(r'(\S+)/fifo/', line)
                if match and os.path.isabs(match.group(1)):
                    fifo_base = match.group(1)
                    break

        # Create temp reference with /tmp/%FIFO_DIR% replaced by the full base path
        ref_tmp_file = NamedTemporaryFile("w+", delete=False, prefix='bash')
        with open(ref_template, 'r') as f:
            kernel_script = f.read()
        kernel_script = kernel_script.replace('/tmp/%FIFO_DIR%', fifo_base)
        ref_tmp_file.write(kernel_script)
        ref_tmp_file.close()

        self.temp_reference_file = ref_tmp_file
        return ref_tmp_file.name


def discover_and_add_tests(cls):
    """Dynamically discover JSON files and add test methods to the class"""

    input_folder = cls.KPARSE_INPUT_FOLDER
    if not os.path.exists(input_folder):
        return cls

    # Discover all JSON files
    json_files = [f.stem for f in Path(input_folder).glob('*.json')]

    # Common partition counts to test
    partition_counts = [1, 8]

    for json_name in json_files:
        for num_partitions in partition_counts:
            expected_name = f"{json_name}_{num_partitions}_partition"
            ref_file = os.path.join(cls.KPARSE_REFERENCE_FOLDER, f"{expected_name}.sh")

            # Only add tests if reference file exists
            # if not os.path.exists(ref_file):
            #    continue

            # Add standard test
            def make_test(name, partitions, expected):
                def test(self):
                    self.genbash(name, partitions)
                    self.check(expected)
                return test

            test_name = f"test_{expected_name}"
            setattr(cls, test_name, make_test(json_name, num_partitions, expected_name))

            # Add chunked test
            def make_chunk_test(name, partitions, expected):
                def test(self):
                    self.gen_chunked_bash(name, partitions)
                    self.check_chunks(expected, partitions)
                return test

            chunk_test_name = f"test_{expected_name}_chunk"
            setattr(cls, chunk_test_name, make_chunk_test(json_name, num_partitions, expected_name))

    return cls


# Apply dynamic test generation
@discover_and_add_tests
class Genbash_base(GenbashBase):
    """Standard genbash tests with auto-discovered test cases"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if UPDATE_BASH_TESTS:
            cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_base")
            os.makedirs(cls.KPARSE_OUTPUT_FOLDER, exist_ok=True)
        else:
            cls.KPARSE_OUTPUT_FOLDER = tempfile.mkdtemp(prefix='output_bash_base_')
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_base")


@discover_and_add_tests
class Genbash_ErrorGuard_and_TempDir(GenbashBase):
    """Tests with error guard and temp directory enabled"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if UPDATE_BASH_TESTS:
            cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_err")
            os.makedirs(cls.KPARSE_OUTPUT_FOLDER, exist_ok=True)
        else:
            cls.KPARSE_OUTPUT_FOLDER = tempfile.mkdtemp(prefix='output_bash_err_')
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_err")

        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = True
        cls.stderr_guard = True


@discover_and_add_tests
class Genbash_LoadBalancer_and_gulpy(GenbashBase):
    """Tests with load balancer configuration"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if UPDATE_BASH_TESTS:
            cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_lb")
            os.makedirs(cls.KPARSE_OUTPUT_FOLDER, exist_ok=True)
        else:
            cls.KPARSE_OUTPUT_FOLDER = tempfile.mkdtemp(prefix='output_bash_lb_')
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_lb")

        cls.num_gul_per_lb = 2
        cls.num_fm_per_lb = 2
        cls.gulmc = False

    def test_gulpy_pipeline_renders(self):
        """gulmc=False must render the modelpy|gulpy pipeline without error.

        Regression test: the gulpy branch of get_getmodel_cmd passed conflicting
        arguments to get_gulcmd, raising a TypeError during bash generation and
        breaking every gulpy run. Note this class sets cls.gulmc=False, but the
        genbash() helper resolves ``gulmc or self.gulmc`` so gulmc=False must be
        passed explicitly to actually exercise the gulpy path (otherwise the
        True default wins and gulmc is rendered instead).
        """
        name = "gul_summarycalc_1_output"
        self.genbash(name, 1, gulmc=False)
        with open(self._get_output_filename(name, 1), encoding='utf-8') as f:
            script = f.read()
        # gulpy command rendered (bare 'gulmc'/'gulpy' also appear in proc_list,
        # so key on the actual invocation with its --random-generator flag).
        self.assertIn("gulpy --random-generator", script)
        self.assertNotIn("gulmc --random-generator", script)


# Special case: Custom gulcalc tests
class Genbash_CustomGulcalc(GenbashBase):
    """Tests for custom gulcalc commands"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if UPDATE_BASH_TESTS:
            cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_csm")
            os.makedirs(cls.KPARSE_OUTPUT_FOLDER, exist_ok=True)
        else:
            cls.KPARSE_OUTPUT_FOLDER = tempfile.mkdtemp(prefix='output_bash_csm_')
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_csm")

    @staticmethod
    def _get_getmodel_cmd(**kwargs):
        error_guard = kwargs.get('stderr_guard')
        cmd = 'custom_gulcalc_command'
        if error_guard:
            cmd = f'({cmd}) 2>> log/gul_stderror.err'
        return cmd

    def test_custom_gul_summarycalc_1_partition(self):
        self.genbash("custom_gul_summarycalc_1_output", 1,
                     _get_getmodel_cmd=self._get_getmodel_cmd)
        self.check("custom_gul_summarycalc_1_output_1_partition")

    def test_custom_gul_summarycalc_1_partition_chunk(self):
        self.gen_chunked_bash("custom_gul_summarycalc_1_output", 1,
                              _get_getmodel_cmd=self._get_getmodel_cmd)
        self.check_chunks("custom_gul_summarycalc_1_output_1_partition", 1)


class Genbash_summarypy_default(GenbashBase):
    """Exercises the summarypy_low_memory=False default (no -m flag emitted).

    The base GenbashBase fixtures were captured when summarypy unconditionally
    received -m; they keep summarypy_low_memory=True for compatibility. This
    class flips the flag back to its real default so a single end-to-end render
    is checked against a fixture without -m.
    """
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if UPDATE_BASH_TESTS:
            cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_summarypy_default")
            os.makedirs(cls.KPARSE_OUTPUT_FOLDER, exist_ok=True)
        else:
            cls.KPARSE_OUTPUT_FOLDER = tempfile.mkdtemp(prefix='output_bash_summarypy_default_')
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_summarypy_default")
        cls.summarypy_low_memory = False

    def test_gul_summarycalc_no_low_memory_flag(self):
        self.genbash("gul_summarycalc_1_output", 1)
        self.check("gul_summarycalc_1_output_1_partition")


def test_add_server_call_flags_modelpy_feeding_gulpy():
    """modelpy reports the empty events it drops, so it needs the same server flags as gulpy."""
    call = '( ( evepy 1 2 | modelpy | gulpy --random-generator=1 -S1 -L0 -a0 | tee fifo/gul_P1 ) ) & pid1=$!'
    flagged = add_server_call(call, socket_server_port=9999)
    assert "modelpy --socket-server='9999' |" in flagged
    assert "gulpy --socket-server='9999' --random-generator=1" in flagged


def test_add_server_call_flags_modelpy_for_platform(monkeypatch):
    monkeypatch.setenv('OASIS_ANALYSIS_STATUS_URL', 'http://server/analysis-status/')
    call = '( ( evepy 1 2 | modelpy | gulpy -S1 -L0 -a0 | tee fifo/gul_P1 ) ) & pid1=$!'
    flagged = add_server_call(call, analysis_pk=7)
    assert "modelpy --socket-server='True' --analysis-pk='7' |" in flagged
    assert "gulpy --socket-server='True' --analysis-pk='7' -S1" in flagged


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _palt_settings():
    with open(os.path.join(TEST_DIRECTORY, 'input_json_settings', 'gul_il_ord_palt_output.json')) as f:
        return json.load(f)['analysis_settings']


REPORTING_OUTPUT = r'^[( ]*(?:katpy|aalpy|lecpy|join-summary-info)\b'


def _all_outputs_settings():
    with open(os.path.join(TEST_DIRECTORY, 'input_json_settings', 'all_outputs_1_csv.json')) as f:
        return json.load(f)['analysis_settings']


@pytest.mark.parametrize("settings_fn", [_palt_settings, _all_outputs_settings], ids=["palt", "all-outputs"])
def test_create_bash_outputs_returns_reporting_processes(tmp_path, settings_fn):
    """The count covers every reporting command, including the joins after the kats (their pid counter is reset)."""
    settings = {**settings_fn(), 'join_summary_info': True}
    params = bash_params(settings, max_process_id=2, gulmc=True, filename=str(tmp_path / 'run.sh'))
    count = create_bash_outputs(**params)
    with open(params['filename']) as f:
        script = f.read()
    assert count > 0
    assert len(re.findall(REPORTING_OUTPUT, script, re.MULTILINE)) == count
    assert re.search(r'^join-summary-info\b', script, re.MULTILINE)
    assert '/dev/tcp' not in script


def test_genbash_reports_gul_and_outputs_to_local_server(tmp_path):
    filename = str(tmp_path / 'run_kernel.sh')
    settings = {**_palt_settings(), 'join_summary_info': True}
    count = genbash(2, settings, filename=filename, gulmc=True, socket_server_size=10, socket_server_port=9999)
    with open(filename) as f:
        script = f.read()

    assert "socket-server 10 9999 --gul-processes 2 > /dev/null & spid=$!" in script
    flagged_outputs = re.findall(r"\b(?:katpy|aalpy|lecpy|join-summary-info) --socket-server='9999'", script)
    assert len(flagged_outputs) == count
    assert len(flagged_outputs) == len(re.findall(REPORTING_OUTPUT, script, re.MULTILINE))

    total_msg = progress_server_message({'outputs_total': count}, 9999)
    done_msg = progress_server_message({'outputs_done': 1}, 9999)
    assert total_msg == ("{ printf '%s\\n' '{\"outputs_total\": " + str(count) + "}' "
                         "> /dev/tcp/${OASIS_SOCKET_SERVER_IP:-127.0.0.1}/9999; } 2>/dev/null || true")
    first_output = re.search(REPORTING_OUTPUT, script, re.MULTILINE).start()
    assert script.rindex('exec_wait $pid1') < script.index(total_msg) < first_output
    assert script.rindex('join-summary-info') < script.index(done_msg) < script.rindex('kill -9 "${spid:-}"')


def test_create_bash_outputs_with_local_server_keeps_script_and_runs_without_spid(tmp_path):
    """An outputs-only script keeps what was already written and does not need the server's pid to be set."""
    filename = str(tmp_path / 'run_outputs.sh')
    print_command(filename, '# existing line')
    settings = {**_palt_settings(), 'join_summary_info': True}
    params = bash_params(settings, max_process_id=2, gulmc=True, filename=filename)
    count = create_bash_outputs(**params, socket_server_size=10, socket_server_port=9999)
    with open(filename) as f:
        script = f.read()

    total_msg = progress_server_message({'outputs_total': count}, 9999)
    assert script.startswith('# existing line\n' + total_msg + '\n')
    assert len(re.findall(REPORTING_OUTPUT, script, re.MULTILINE)) == count

    shutdown = script[script.index(progress_server_message({'outputs_done': 1}, 9999)):]
    result = subprocess.run(['bash', '-c', 'set -euET\n' + shutdown], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_local_server_messages_reach_server(tmp_path):
    """The script's /dev/tcp messages are read by the progress server, and a missing server does not fail the script."""
    port = get_free_port()
    server = GulProgressServer(10, host="127.0.0.1", port=port)
    server.start()
    try:
        script = tmp_path / 'send.sh'
        script.write_text('set -euET -o pipefail\n'
                          + progress_server_message({'outputs_total': 4}, port) + '\n'
                          + progress_server_message({'outputs_done': 1}, get_free_port()) + '\n')
        subprocess.run(['bash', str(script)], check=True, timeout=30)
        time.sleep(0.1)
        assert server.outputs_total == 4
        assert not server.finished
    finally:
        server.stop()


def test_genbash_platform_run_has_no_local_server(tmp_path, monkeypatch):
    monkeypatch.setenv('OASIS_ANALYSIS_STATUS_URL', 'http://server/analysis-status/')
    filename = str(tmp_path / 'run_kernel.sh')
    count = genbash(2, _palt_settings(), filename=filename, gulmc=True, socket_server_size=10, analysis_pk=3)
    with open(filename) as f:
        script = f.read()
    assert 'socket-server ' not in script
    assert '/dev/tcp' not in script
    assert len(re.findall(r"\b(?:katpy|aalpy|lecpy) --socket-server='True' --analysis-pk='3'", script)) == count


def test_output_commands_unflagged_without_progress_server(tmp_path):
    filename = str(tmp_path / 'run_kernel.sh')
    genbash(2, _palt_settings(), filename=filename, gulmc=True)
    with open(filename) as f:
        script = f.read()
    assert 'socket-server' not in script
    assert '--socket-server' not in script


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_exit_handler_cleans_up_when_exec_monitor_already_gone(tmp_path):
    """A failing kernel still kills the progress server and its other processes after the exec monitor has exited.

    The monitor (pid0) is gone when oasis_exec_monitor.sh is not on PATH or has already stopped. The handler
    used to abort under `set -e` when it failed to kill it, skipping the rest of the clean-up.
    """
    filename = str(tmp_path / 'run_kernel.sh')
    with bash_wrapper(filename, False, True, run_check_complete=False):
        print_command(filename, 'kill -9 $pid0 2>/dev/null; wait $pid0 2>/dev/null || true')
        print_command(filename, 'sleep 300 & spid=$!')
        print_command(filename, 'sleep 300 & pid1=$!')
        print_command(filename, 'echo "$spid $pid1" > pids.txt')
        print_command(filename, 'false')

    proc = subprocess.run(['bash', filename], cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          start_new_session=True, timeout=60)
    server_pid, other_pid = (int(pid) for pid in (tmp_path / 'pids.txt').read_text().split())
    try:
        assert proc.returncode != 0
        assert not _pid_alive(server_pid)
        assert not _pid_alive(other_pid)
        assert (tmp_path / 'log' / 'killout.txt').exists()
    finally:
        for pid in (server_pid, other_pid):
            if _pid_alive(pid):
                os.kill(pid, 9)
