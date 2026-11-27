import importlib
import json
import os
import subprocess
import tempfile
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from unittest.mock import patch

from oasislmf.execution import runner
from oasislmf.execution.resource_monitor import ResourceMonitor
from oasislmf.execution.bash import bash_params, genbash, genbash_params, get_complex_model_cmd
from oasislmf.execution.runner import rerun
from oasislmf.utils.exceptions import OasisException

# The deprecated oasislmf.model_execution alias loads a second copy of bash.py and rebinds the
# oasislmf.execution.bash attribute to it, which Python 3.10's patch() resolves by attribute.
bash_module = importlib.import_module('oasislmf.execution.bash')


class TestSnapshotLogDir(TestCase):
    def test_empty_dir_returns_empty_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(runner._snapshot_log_dir(tmp_dir), {})

    def test_captures_size_and_mtime_for_each_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f1 = os.path.join(tmp_dir, 'a.log')
            with open(f1, 'w') as f:
                f.write('hello')
            sub = os.path.join(tmp_dir, 'sub')
            os.mkdir(sub)
            f2 = os.path.join(sub, 'b.log')
            with open(f2, 'w') as f:
                f.write('world!')

            snapshot = runner._snapshot_log_dir(tmp_dir)

            self.assertEqual(set(snapshot), {f1, f2})
            st1 = os.stat(f1)
            self.assertEqual(snapshot[f1], (st1.st_size, st1.st_mtime))

    def test_skips_file_removed_between_listing_and_stat(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            f1 = os.path.join(tmp_dir, 'a.log')
            with open(f1, 'w') as f:
                f.write('hello')

            real_stat = os.stat

            def flaky_stat(path, *args, **kwargs):
                if str(path) == f1:
                    raise OSError('vanished')
                return real_stat(path, *args, **kwargs)

            with mock.patch('oasislmf.execution.runner.os.stat', side_effect=flaky_stat):
                snapshot = runner._snapshot_log_dir(tmp_dir)

            self.assertEqual(snapshot, {})


class TestWaitForLogWriters(TestCase):
    @staticmethod
    def _clock(step=1):
        times = iter(range(0, 10000, step))
        return lambda: next(times)

    def test_returns_once_files_stable_for_stable_seconds(self):
        with tempfile.TemporaryDirectory() as log_dir:
            with mock.patch('oasislmf.execution.runner.time.sleep'), \
                    mock.patch('oasislmf.execution.runner.time.time', side_effect=self._clock()), \
                    mock.patch('oasislmf.execution.runner.logging.warning') as mock_warn:
                runner._wait_for_log_writers(log_dir, timeout=100, poll_interval=1, stable_seconds=5.0)

            mock_warn.assert_not_called()

    def test_short_quiet_gap_does_not_count_as_settled(self):
        with tempfile.TemporaryDirectory() as log_dir:
            # stable for 2 polls, then changes again, then stable for good
            snapshots = iter([
                {'f': (1, 1.0)},
                {'f': (1, 1.0)},
                {'f': (1, 1.0)},
                {'f': (2, 2.0)},
            ])
            calls = []

            def fake_snapshot(_log_dir):
                calls.append(1)
                return next(snapshots, {'f': (2, 2.0)})

            with mock.patch('oasislmf.execution.runner._snapshot_log_dir', side_effect=fake_snapshot), \
                    mock.patch('oasislmf.execution.runner.time.sleep'), \
                    mock.patch('oasislmf.execution.runner.time.time', side_effect=self._clock()), \
                    mock.patch('oasislmf.execution.runner.logging.warning') as mock_warn:
                runner._wait_for_log_writers(log_dir, timeout=100, poll_interval=1, stable_seconds=5.0)

            mock_warn.assert_not_called()
            # didn't return during the 2-poll quiet gap before the change
            self.assertGreater(len(calls), 4)

    def test_times_out_and_logs_warning_when_files_never_settle(self):
        with tempfile.TemporaryDirectory() as log_dir:
            counter = iter(range(10000))

            def fake_snapshot(_log_dir):
                return {'f': (next(counter), 0.0)}

            with mock.patch('oasislmf.execution.runner._snapshot_log_dir', side_effect=fake_snapshot), \
                    mock.patch('oasislmf.execution.runner.time.sleep'), \
                    mock.patch('oasislmf.execution.runner.time.time', side_effect=self._clock()), \
                    mock.patch('oasislmf.execution.runner.logging.warning') as mock_warn:
                runner._wait_for_log_writers(log_dir, timeout=20, poll_interval=1, stable_seconds=5.0)

            self.assertTrue(mock_warn.called)


class TestFindIncompletePytoolLogs(TestCase):
    def test_no_log_files_present_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir), {})

    def test_log_with_finish_marker_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'fmpy_123.log'), 'w') as f:
                f.write('starting process\nfinishing process\n')
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir), {})

    def test_log_missing_finish_marker_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'fmpy_123.log')
            with open(path, 'w') as f:
                f.write('starting process\n')
            lost = runner._find_incomplete_pytool_logs(tmp_dir)
            self.assertEqual(lost, {'fmpy': [path]})

    def test_unreadable_log_file_is_reported_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'fmpy_123.log')
            with open(path, 'w') as f:
                f.write('starting process\nfinishing process\n')

            with mock.patch('builtins.open', side_effect=OSError('boom')):
                lost = runner._find_incomplete_pytool_logs(tmp_dir)

            self.assertEqual(lost, {'fmpy': [path]})

    def test_lost_custom_gulcalc_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, 'gul_stderror.err')
            with open(path, 'w') as f:
                f.write('gulcalc started\ngulcalc started\ngulcalc finished\n')
            lost = runner._find_incomplete_pytool_logs(tmp_dir, 'gulcalc started', 'gulcalc finished')
            self.assertEqual(lost, {'gulcalc': [path]})

    def test_finished_custom_gulcalc_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'gul_stderror.err'), 'w') as f:
                f.write('gulcalc started\ngulcalc finished\n')
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir, 'gulcalc started', 'gulcalc finished'), {})

    def test_custom_gulcalc_ignored_without_markers(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'gul_stderror.err'), 'w') as f:
                f.write('gulcalc started\n')
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir), {})
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir, 'gulcalc started', None), {})

    def test_missing_gul_stderror_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir, 'gulcalc started', 'gulcalc finished'), {})

    def test_non_utf8_gul_stderror_is_still_counted(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'gul_stderror.err'), 'wb') as f:
                f.write(b'gulcalc started\xff\xfe\ngulcalc finished\n\xc3gulcalc started\n')
            lost = runner._find_incomplete_pytool_logs(tmp_dir, 'gulcalc started', 'gulcalc finished')
            self.assertEqual(lost, {'gulcalc': [os.path.join(tmp_dir, 'gul_stderror.err')]})

    def test_markers_match_as_plain_substrings(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'gul_stderror.err'), 'w') as f:
                f.write('*** gulcalc [start] *** 1\n*** gulcalc [start] *** 2\nfinished.* 1\ngulcalc finished 2\n')
            lost = runner._find_incomplete_pytool_logs(tmp_dir, '*** gulcalc [start] ***', 'finished.*')
            self.assertEqual(lost, {'gulcalc': [os.path.join(tmp_dir, 'gul_stderror.err')]})


class TestRunChecksLogsComplete(TestCase):
    def _run(self, tmp_dir, returncode, analysis_settings=None, **kwargs):
        proc = mock.Mock(pid=1, returncode=returncode)
        proc.communicate.return_value = (b'', None)
        with mock.patch('oasislmf.execution.runner.genbash'), \
                mock.patch('oasislmf.execution.runner.ResourceMonitor'), \
                mock.patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc), \
                mock.patch('oasislmf.execution.runner._ensure_pytool_logs_complete') as mock_ensure:
            try:
                runner.run(analysis_settings or {}, filename=os.path.join(tmp_dir, 'run_kernel.sh'), **kwargs)
            except subprocess.CalledProcessError:
                pass
        return mock_ensure

    def test_checks_run_dir_logs_after_successful_script(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_ensure = self._run(tmp_dir, 0, custom_gulcalc_log_start='start', custom_gulcalc_log_finish='finish')
        mock_ensure.assert_called_once_with(os.path.join(tmp_dir, 'log'), 'start', 'finish')

    def test_markers_set_only_in_analysis_settings_are_checked(self):
        settings = {'model_custom_gulcalc_log_start': 'start', 'model_custom_gulcalc_log_finish': 'finish'}
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_ensure = self._run(tmp_dir, 0, analysis_settings=settings)
        mock_ensure.assert_called_once_with(os.path.join(tmp_dir, 'log'), 'start', 'finish')

    def test_failed_script_skips_log_check(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_ensure = self._run(tmp_dir, 1)
        mock_ensure.assert_not_called()

    def test_incomplete_logs_raise_called_process_error(self):
        proc = mock.Mock(pid=1, returncode=0)
        proc.communicate.return_value = (b'script output', None)
        with tempfile.TemporaryDirectory() as tmp_dir, \
                mock.patch('oasislmf.execution.runner.genbash'), \
                mock.patch('oasislmf.execution.runner.ResourceMonitor'), \
                mock.patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc), \
                mock.patch('oasislmf.execution.runner._ensure_pytool_logs_complete',
                           side_effect=OasisException('fmpy (1 lost)')):
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                runner.run({}, filename=os.path.join(tmp_dir, 'run_kernel.sh'))
        self.assertIn(b'script output', ctx.exception.output)
        self.assertIn(b'fmpy (1 lost)', ctx.exception.output)


GUL_OUTPUT = {'gul_output': True, 'gul_summaries': [{'id': 1}]}


class TestBashParamsCustomGulcalcMarkers(TestCase):
    """run_analysis reads the markers from bash_params, so they must be set on every path."""

    def test_markers_set_with_custom_gulcalc_cmd(self):
        settings = {'model_custom_gulcalc_log_start': 'start', 'model_custom_gulcalc_log_finish': 'finish', **GUL_OUTPUT}
        params = bash_params(settings, custom_gulcalc_cmd='ls')
        self.assertEqual(('start', 'finish'), (params['custom_gulcalc_log_start'], params['custom_gulcalc_log_finish']))

    def test_markers_set_without_custom_gulcalc_cmd(self):
        params = bash_params(dict(GUL_OUTPUT), custom_gulcalc_log_start='start', custom_gulcalc_log_finish='finish')
        self.assertEqual(('start', 'finish'), (params['custom_gulcalc_log_start'], params['custom_gulcalc_log_finish']))


class TestCustomGulcalcCommand(TestCase):
    """The loss stream of a custom gulcalc binary goes to stdout: no ``-i`` is passed."""

    ITEM_OUTPUT = ' | tee fifo/gul_P1'

    def _args(self, item_output):
        return dict(number_of_samples=10, gul_threshold=0, use_random_number_file=False, item_output=item_output,
                    process_id=2, max_process_id=8, gul_alloc_rule=0, stderr_guard=False)

    def _assert_command(self, get_cmd):
        binary_cmd = f"my_gulcalc -e 2 8 -a {os.path.abspath('analysis_settings.json')} -p input"
        self.assertEqual(get_cmd(**self._args(self.ITEM_OUTPUT)), binary_cmd + self.ITEM_OUTPUT)
        self.assertEqual(get_cmd(**self._args('')), binary_cmd)

    def test_bash_params_command(self):
        with mock.patch('oasislmf.execution.bash.shutil.which', return_value='/usr/bin/my_gulcalc'):
            self._assert_command(get_complex_model_cmd('my_gulcalc', {}))

    def test_run_command(self):
        proc = mock.Mock(pid=1, returncode=0)
        proc.communicate.return_value = (b'', None)
        with TemporaryDirectory() as tmp_dir, \
                mock.patch('oasislmf.execution.runner.shutil.which', return_value='/usr/bin/my_gulcalc'), \
                mock.patch('oasislmf.execution.runner.check_custom_gulcalc_can_read_inputs'), \
                mock.patch('oasislmf.execution.runner.genbash') as genbash, \
                mock.patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc), \
                mock.patch('oasislmf.execution.runner._ensure_pytool_logs_complete'):
            runner.run({}, filename=os.path.join(tmp_dir, 'run_kernel.sh'), custom_gulcalc_cmd='my_gulcalc',
                       disable_resource_report=True)
        self._assert_command(genbash.call_args.kwargs['_get_getmodel_cmd'])


class TestEnsurePytoolLogsComplete(TestCase):
    def test_returns_immediately_when_all_logs_complete(self):
        with mock.patch('oasislmf.execution.runner._find_incomplete_pytool_logs', return_value={}) as mock_find, \
                mock.patch('oasislmf.execution.runner._wait_for_log_writers') as mock_wait:
            runner._ensure_pytool_logs_complete('/log')

        mock_find.assert_called_once()
        mock_wait.assert_not_called()

    def test_waits_and_succeeds_if_stragglers_finish(self):
        lost_then_done = [{'fmpy': ['f.log']}, {}]
        with mock.patch('oasislmf.execution.runner._find_incomplete_pytool_logs', side_effect=lost_then_done), \
                mock.patch('oasislmf.execution.runner._wait_for_log_writers') as mock_wait:
            runner._ensure_pytool_logs_complete('/log')

        mock_wait.assert_called_once()

    def test_raises_oasis_exception_if_still_incomplete_after_wait(self):
        lost = {'fmpy': ['f.log']}
        with mock.patch('oasislmf.execution.runner._find_incomplete_pytool_logs', side_effect=[lost, lost]), \
                mock.patch('oasislmf.execution.runner._wait_for_log_writers'):
            with self.assertRaises(OasisException):
                runner._ensure_pytool_logs_complete('/log')


class RerunTestCase(TestCase):
    """rerun() replays a single failing event outside numba. It parses the
    gul command straight out of run_kernel.sh, which still carries that
    run's own output redirect (e.g. to a fifo whose reader has already
    exited). That stale redirect must be stripped before rerun() appends
    its own `-o` output flag, otherwise the replayed command blocks
    forever trying to write to an orphaned fifo.
    """

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.cwd = os.getcwd()
        os.chdir(self.tmp_dir.name)
        self.addCleanup(os.chdir, self.cwd)

        with open("event_error.json", "w") as f:
            json.dump({"event_id": "1"}, f)

        with open("run_kernel.sh", "w") as f:
            f.write(
                "( ( eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a1 -i - "
                "> /tmp/xyz123/fifo/gul_P1 ) | fmcalc -a2 -o /tmp/il_P1 ) "
                "2>> log/stderror.err &\n"
            )

    def test_stale_fifo_redirect_is_stripped_from_gul_cmd(self):
        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        gul_pipe = subprocess_run.call_args_list[0].args[0]

        self.assertNotIn("/tmp/xyz123/fifo/gul_P1", gul_pipe)
        self.assertIn("-o 1_gul.bin", gul_pipe)
        self.assertTrue(gul_pipe.startswith("printf"))

    def test_no_event_error_file_returns_without_running(self):
        os.remove("event_error.json")

        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        subprocess_run.assert_not_called()

    def test_non_trailing_redirect_is_not_mangled_in_gul_cmd(self):
        # a non-trailing redirect (e.g. the stderr guard appended around the whole
        # pipeline) must survive untouched: only the genuinely trailing output
        # redirect on the gul segment itself should be stripped.
        with open("run_kernel.sh", "w") as f:
            f.write(
                "( ( eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a1 -i - "
                "2>> log/gul_stderror.err > /tmp/xyz123/fifo/gul_P1 ) | fmcalc -a2 > /tmp/il_P1 ) "
                "2>> log/stderror.err &\n"
            )

        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        gul_pipe = subprocess_run.call_args_list[0].args[0]

        self.assertNotIn("/tmp/xyz123/fifo/gul_P1", gul_pipe)
        self.assertIn("2>> log/gul_stderror.err", gul_pipe)
        self.assertIn("-o 1_gul.bin", gul_pipe)

    def test_stale_redirect_is_stripped_from_fm_cmd(self):
        # the fm segment carries the same kind of stale output redirect (e.g. to a
        # fifo consumed by the original run) and must be stripped before rerun()
        # points it at its own output file, otherwise it hangs the same way the
        # gul segment used to.
        with open("run_kernel.sh", "w") as f:
            f.write(
                "( ( eve 1 2 | getmodel | gulcalc -S100 -L100 -r -a1 -i - "
                "> /tmp/xyz123/fifo/gul_P1 ) | fmcalc -a2 > /tmp/il_P1 ) "
                "2>> log/stderror.err &\n"
            )

        with patch("oasislmf.execution.runner.subprocess.run") as subprocess_run:
            rerun()

        fm_pipe = subprocess_run.call_args_list[1].args[0]

        self.assertNotIn("/tmp/il_P1", fm_pipe)
        self.assertIn("-o 1_fm1.bin", fm_pipe)
        self.assertTrue(fm_pipe.startswith("fmcalc"))


def _run_with_mocked_script(**kwargs):
    with (patch.object(runner, 'genbash', return_value={'gul_processes': 4, 'outputs_total': 7}),
          patch.object(runner, 'ResourceMonitor'),
          patch.object(runner, '_ensure_pytool_logs_complete'),
          patch.object(runner.subprocess, 'Popen') as mock_popen,
          patch.object(runner, 'oasis_ping') as mock_ping):
        mock_popen.return_value.communicate.return_value = (b'', None)
        mock_popen.return_value.returncode = 0
        runner.run({}, **kwargs)
    return mock_ping


def test_run_reports_outputs_total_to_platform(monkeypatch):
    """A single-script platform run reports how many GUL and output processes will report progress."""
    monkeypatch.setenv('OASIS_ANALYSIS_STATUS_URL', 'http://server/analysis-status/')
    mock_ping = _run_with_mocked_script(analysis_pk=5)
    mock_ping.assert_called_once_with({'analysis_pk': 5, 'gul_processes': 4, 'outputs_total': 7})


def test_run_local_does_not_ping_outputs_total(monkeypatch):
    monkeypatch.delenv('OASIS_ANALYSIS_STATUS_URL', raising=False)
    monkeypatch.delenv('OASIS_WEBSOCKET_URL', raising=False)
    mock_ping = _run_with_mocked_script(socket_server_size=10, socket_server_port=9999)
    mock_ping.assert_not_called()


def _run_analysis_with_mocked_script(**params):
    with (patch.object(runner, 'create_bash_analysis', return_value=2),
          patch.object(runner, 'bash_wrapper'),
          patch.object(runner, 'ResourceMonitor'),
          patch.object(runner, '_ensure_pytool_logs_complete'),
          patch.object(runner.subprocess, 'Popen') as mock_popen,
          patch.object(runner, 'oasis_ping') as mock_ping):
        mock_popen.return_value.communicate.return_value = (b'', None)
        mock_popen.return_value.returncode = 0
        runner.run_analysis(filename='run.sh', bash_trace=False, stderr_guard=True, fifo_queue_dir='fifo/', **params)
    return mock_ping


def test_run_analysis_reports_gul_processes_to_platform(monkeypatch):
    """Each analysis chunk on the platform reports how many of its GUL processes will report `gul_ready`."""
    monkeypatch.setenv('OASIS_ANALYSIS_STATUS_URL', 'http://server/analysis-status/')
    mock_ping = _run_analysis_with_mocked_script(analysis_pk=5, process_number=1)
    mock_ping.assert_called_once_with({'analysis_pk': 5, 'gul_processes': 2})


def test_run_analysis_local_does_not_ping_gul_processes(monkeypatch):
    monkeypatch.delenv('OASIS_ANALYSIS_STATUS_URL', raising=False)
    monkeypatch.delenv('OASIS_WEBSOCKET_URL', raising=False)
    mock_ping = _run_analysis_with_mocked_script(process_number=1)
    mock_ping.assert_not_called()


class TestDisableResourceReport(TestCase):
    def _run(self, func, create_bash_name, **extra):
        proc = mock.Mock(pid=123, returncode=0)
        proc.communicate.return_value = (b'', None)
        with TemporaryDirectory() as tmp_dir, \
                patch('oasislmf.execution.runner.bash_wrapper'), \
                patch(f'oasislmf.execution.runner.{create_bash_name}') as create_bash, \
                patch('oasislmf.execution.runner._ensure_pytool_logs_complete'), \
                patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc), \
                patch('oasislmf.execution.runner.ResourceMonitor') as monitor_cls:
            func(filename=os.path.join(tmp_dir, 'run.sh'), bash_trace=False, stderr_guard=True,
                 fifo_queue_dir='fifo', **extra)
        return monitor_cls, create_bash

    def test_run_outputs_monitor_enabled_by_default(self):
        monitor_cls, _ = self._run(runner.run_outputs, 'create_bash_outputs')
        self.assertTrue(monitor_cls.call_args.kwargs['enabled'])

    def test_run_outputs_flag_disables_monitor(self):
        monitor_cls, create_bash = self._run(runner.run_outputs, 'create_bash_outputs', disable_resource_report=True)
        self.assertFalse(monitor_cls.call_args.kwargs['enabled'])
        self.assertNotIn('disable_resource_report', create_bash.call_args.kwargs)

    def test_run_analysis_monitor_enabled_by_default(self):
        monitor_cls, _ = self._run(runner.run_analysis, 'create_bash_analysis', process_number=1)
        self.assertTrue(monitor_cls.call_args.kwargs['enabled'])

    def test_run_analysis_flag_disables_monitor(self):
        monitor_cls, create_bash = self._run(runner.run_analysis, 'create_bash_analysis', process_number=1,
                                             disable_resource_report=True)
        self.assertFalse(monitor_cls.call_args.kwargs['enabled'])
        self.assertNotIn('disable_resource_report', create_bash.call_args.kwargs)

    def _run_single_script(self, **extra):
        proc = mock.Mock(pid=123, returncode=0)
        proc.communicate.return_value = (b'', None)
        with TemporaryDirectory() as tmp_dir, \
                patch('oasislmf.execution.runner.genbash') as genbash, \
                patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc), \
                patch('oasislmf.execution.runner.ResourceMonitor') as monitor_cls:
            runner.run({}, filename=os.path.join(tmp_dir, 'run_kernel.sh'), **extra)
        return monitor_cls, genbash

    def test_run_monitor_enabled_by_default(self):
        monitor_cls, _ = self._run_single_script()
        self.assertTrue(monitor_cls.call_args.kwargs['enabled'])

    def test_run_flag_disables_monitor(self):
        monitor_cls, genbash = self._run_single_script(disable_resource_report=True)
        self.assertFalse(monitor_cls.call_args.kwargs['enabled'])
        self.assertNotIn('disable_resource_report', genbash.call_args.kwargs)


class TestResourceMonitorDisabled(TestCase):
    def test_disabled_monitor_writes_no_csv_or_report(self):
        with TemporaryDirectory() as tmp_dir:
            log_root = os.path.join(tmp_dir, 'log')
            monitor = ResourceMonitor(output_dir=os.path.join(log_root, 'out'), poll_interval=0.1,
                                      log_root=log_root, enabled=False)
            with patch.object(monitor, '_generate_report') as generate_report:
                monitor.start(os.getpid())
                monitor.stop()
            self.assertIsNone(monitor._thread)
            generate_report.assert_not_called()
            self.assertFalse(os.path.exists(log_root))


class TestRunWithOrchestrator(TestCase):
    """runner.run / run_analysis / run_outputs with and without the ``orchestrator`` option, run_plan mocked."""

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.script = os.path.join(self.tmp_dir.name, 'run_kernel.sh')
        for target in ['ResourceMonitor', '_ensure_pytool_logs_complete', 'check_custom_gulcalc_can_read_inputs']:
            patcher = patch(f'oasislmf.execution.runner.{target}')
            setattr(self, target.strip('_'), patcher.start())
            self.addCleanup(patcher.stop)

    def _run(self, settings=None, **kwargs):
        proc = mock.Mock(pid=1, returncode=0)
        proc.communicate.return_value = (b'bash output', None)
        with patch('oasislmf.execution.runner.genbash') as genbash, \
                patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc) as popen, \
                patch('oasislmf.execution.runner.run_plan', return_value='orchestrator output') as run_plan:
            runner.run(dict(settings or GUL_OUTPUT), filename=self.script, **kwargs)
        return genbash, popen, run_plan

    def test_option_off_runs_bash(self):
        genbash, popen, run_plan = self._run(number_of_processes=2)
        genbash.assert_called_once()
        self.assertEqual(['bash', self.script], popen.call_args.args[0])
        run_plan.assert_not_called()

    def test_option_on_runs_the_plan_and_still_writes_the_script(self):
        genbash, popen, run_plan = self._run(orchestrator=True, number_of_processes=2)
        genbash.assert_called_once()
        self.assertNotIn('orchestrator', genbash.call_args.kwargs)
        popen.assert_not_called()
        plan, params, mode, log_dir, script = run_plan.call_args.args
        self.assertEqual(('all', 'log', self.script), (mode, log_dir, script))
        self.assertEqual(2, plan.n_chunks)
        self.assertEqual([1, 2], plan.chunk_numbers)
        self.assertEqual(os.path.realpath(self.tmp_dir.name), os.path.realpath(plan.spec.run_dir))
        self.assertEqual(2, params['max_process_id'])
        self.assertIs(self.ResourceMonitor.return_value, run_plan.call_args.kwargs['monitor'])
        self.ensure_pytool_logs_complete.assert_called_once_with(os.path.join(self.tmp_dir.name, 'log'), None, None)

    def test_unsupported_run_falls_back_to_bash_with_a_warning(self):
        with self.assertLogs('oasislmf.execution.runner', level='WARNING') as logs:
            genbash, popen, run_plan = self._run(orchestrator=True, custom_args={'modelpy_server': False})
        run_plan.assert_not_called()
        self.assertEqual(['bash', self.script], popen.call_args.args[0])
        self.assertIn('custom_args', logs.output[0])
        self.assertIn('running the kernel with the bash script instead', logs.output[0])

    def test_custom_getmodel_command_falls_back_to_bash(self):
        with self.assertLogs('oasislmf.execution.runner', level='WARNING') as logs:
            genbash, popen, run_plan = self._run(orchestrator=True, custom_get_getmodel_cmd=lambda **kw: 'my_gul')
        run_plan.assert_not_called()
        popen.assert_called_once()
        self.assertIn('custom gulcalc', logs.output[0])

    def test_custom_gulcalc_binary_falls_back_to_bash(self):
        with self.assertLogs('oasislmf.execution.runner', level='WARNING') as logs, \
                patch('oasislmf.execution.runner.shutil.which', return_value='/usr/bin/my_gulcalc'):
            genbash, popen, run_plan = self._run(orchestrator=True, custom_gulcalc_cmd='my_gulcalc')
        run_plan.assert_not_called()
        popen.assert_called_once()
        self.assertIn('custom gulcalc', logs.output[0])

    def test_orchestrator_failure_surfaces_as_called_process_error(self):
        error = subprocess.CalledProcessError(1, ['oasislmf.execution.orchestrator.kernel.run_plan', 'all', self.script], output=b'FAILED')
        with patch('oasislmf.execution.runner.genbash'), \
                patch('oasislmf.execution.runner.run_plan', side_effect=error), \
                self.assertRaises(subprocess.CalledProcessError) as ctx:
            runner.run(dict(GUL_OUTPUT), filename=self.script, orchestrator=True)
        self.assertIs(error, ctx.exception)
        self.ensure_pytool_logs_complete.assert_not_called()

    def test_incomplete_logs_raise_called_process_error_naming_the_orchestrator(self):
        self.ensure_pytool_logs_complete.side_effect = OasisException('katpy (1 lost)')
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            self._run(orchestrator=True)
        self.assertEqual(['oasislmf.execution.orchestrator.kernel.run_plan', 'all', self.script], ctx.exception.cmd)
        self.assertIn(b'orchestrator output', ctx.exception.output)
        self.assertIn(b'katpy (1 lost)', ctx.exception.output)

    def _chunk_params(self, **extra):
        params = bash_params(dict(GUL_OUTPUT), max_process_id=4, process_number=3,
                             filename=os.path.join(self.tmp_dir.name, '3.run_analysis.sh'))
        params.update(extra)
        return params

    def test_run_analysis_of_a_chunk_with_the_orchestrator(self):
        params = self._chunk_params(orchestrator=True, resource_monitor_interval=0.5)
        with patch('oasislmf.execution.runner.run_plan', return_value='chunk text') as run_plan, \
                patch('oasislmf.execution.runner.subprocess.Popen') as popen:
            result = runner.run_analysis(**params)
        popen.assert_not_called()
        self.assertEqual((params['fifo_queue_dir'], 'chunk text'), result)
        plan, plan_params, mode, log_dir, script = run_plan.call_args.args
        self.assertEqual(('analysis', os.path.join('log', '3'), params['filename']), (mode, log_dir, script))
        self.assertEqual([3], plan.chunk_numbers)
        self.assertNotIn('orchestrator', plan_params)
        self.assertTrue(os.path.isfile(params['filename']))
        self.ensure_pytool_logs_complete.assert_called_once_with(os.path.join(self.tmp_dir.name, 'log', '3'), None, None)

    def test_run_analysis_option_off_or_unsupported_runs_bash(self):
        proc = mock.Mock(pid=1, returncode=0)
        proc.communicate.return_value = (b'', None)
        for extra in [{}, {'orchestrator': True, 'custom_args': {'modelpy_server': False}}]:
            with self.subTest(**extra), \
                    patch('oasislmf.execution.runner.run_plan') as run_plan, \
                    patch('oasislmf.execution.runner.create_bash_analysis') as create_bash, \
                    patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc):
                runner.run_analysis(**self._chunk_params(**extra))
            run_plan.assert_not_called()
            create_bash.assert_called_once()
            self.assertNotIn('orchestrator', create_bash.call_args.kwargs)

    def test_run_outputs_with_the_orchestrator(self):
        params = bash_params(dict(GUL_OUTPUT), max_process_id=2, filename=os.path.join(self.tmp_dir.name, 'run_outputs.sh'))
        params['orchestrator'] = True
        with patch('oasislmf.execution.runner.run_plan', return_value='outputs text') as run_plan, \
                patch('oasislmf.execution.runner.subprocess.Popen') as popen:
            self.assertEqual('outputs text', runner.run_outputs(**params))
        popen.assert_not_called()
        self.assertEqual(('outputs', os.path.join('log', 'out')), run_plan.call_args.args[2:4])
        self.ensure_pytool_logs_complete.assert_called_once_with(os.path.join(self.tmp_dir.name, 'log', 'out'))

    def test_run_outputs_failure_surfaces_as_called_process_error(self):
        params = bash_params(dict(GUL_OUTPUT), max_process_id=2, filename=os.path.join(self.tmp_dir.name, 'run_outputs.sh'))
        params['orchestrator'] = True
        with patch('oasislmf.execution.runner.run_plan', side_effect=subprocess.CalledProcessError(1, ['x'], output=b'x')), \
                self.assertRaises(subprocess.CalledProcessError):
            runner.run_outputs(**params)


class TestGenbashParams(TestCase):
    def test_genbash_writes_its_script_from_genbash_params(self):
        with TemporaryDirectory() as tmp_dir:
            script = os.path.join(tmp_dir, 'run.sh')
            kwargs = dict(analysis_pk=4, socket_server_size=30, socket_server_port=1234, fifo_tmp_dir=False, filename=script)
            params = genbash_params(2, dict(GUL_OUTPUT), **kwargs)
            self.assertEqual((4, 30, 1234, 2), (params['analysis_pk'], params['socket_server_size'], params['socket_server_port'],
                                                params['max_process_id']))
            with patch.object(bash_module, 'create_bash_analysis') as analysis, \
                    patch.object(bash_module, 'create_bash_outputs'):
                genbash(2, dict(GUL_OUTPUT), **kwargs)
            self.assertEqual(params, analysis.call_args.kwargs)


class TestRunCustomGulcalcCommand(TestCase):
    """run() picks the gul command builder handed to genbash for complex (custom gulcalc) models."""

    def _getmodel_cmd_passed_to_genbash(self, analysis_settings=None, executables=(), **kwargs):
        proc = mock.Mock(pid=1, returncode=0)
        proc.communicate.return_value = (b'', None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name in executables:
                path = os.path.join(tmp_dir, name)
                with open(path, 'w') as f:
                    f.write('#!/bin/sh\n')
                os.chmod(path, 0o755)
            with mock.patch.dict(os.environ, {'PATH': tmp_dir + os.pathsep + os.environ['PATH']}), \
                    mock.patch('oasislmf.execution.runner.genbash') as mock_genbash, \
                    mock.patch('oasislmf.execution.runner.ResourceMonitor'), \
                    mock.patch('oasislmf.execution.runner.subprocess.Popen', return_value=proc), \
                    mock.patch('oasislmf.execution.runner._ensure_pytool_logs_complete'):
                runner.run(analysis_settings or {}, filename=os.path.join(tmp_dir, 'run_kernel.sh'), **kwargs)
        return mock_genbash.call_args.kwargs['_get_getmodel_cmd']

    @staticmethod
    def _build(getmodel_cmd, item_output=' | tee fifo/gul_P2', stderr_guard=False):
        return getmodel_cmd(number_of_samples=10, gul_threshold=0, use_random_number_file=False, item_output=item_output,
                            process_id=2, max_process_id=4, gul_alloc_rule=1, stderr_guard=stderr_guard)

    def test_custom_gulcalc_builds_its_own_command(self):
        getmodel_cmd = self._getmodel_cmd_passed_to_genbash(executables=['fake_gulcalc'], custom_gulcalc_cmd='fake_gulcalc')

        binary_cmd = 'fake_gulcalc -e 2 4 -a {} -p input'.format(os.path.abspath('analysis_settings.json'))
        self.assertEqual(self._build(getmodel_cmd), binary_cmd + ' | tee fifo/gul_P2')
        self.assertEqual(self._build(getmodel_cmd, item_output=''), binary_cmd)
        self.assertTrue(self._build(getmodel_cmd, stderr_guard=True).endswith(') 2>> log/gul_stderror.err'))

    def test_custom_gulcalc_missing_from_path_is_an_error(self):
        with self.assertRaises(OasisException):
            self._getmodel_cmd_passed_to_genbash(custom_gulcalc_cmd='not_a_real_gulcalc_binary')

    def test_supplier_model_gulcalc_on_path_is_inferred(self):
        settings = {'model_supplier_id': 'Acme', 'model_name_id': 'Quake'}
        getmodel_cmd = self._getmodel_cmd_passed_to_genbash(settings, executables=['Acme_Quake_gulcalc'])

        self.assertTrue(self._build(getmodel_cmd).startswith('Acme_Quake_gulcalc -e 2 4 '))

    def test_no_custom_gulcalc_leaves_the_default_command(self):
        self.assertIsNone(self._getmodel_cmd_passed_to_genbash())

    def test_caller_command_builder_is_used_without_a_custom_gulcalc(self):
        """A caller's builder must reach genbash even when no custom gulcalc is set or inferred."""
        def caller_builder(**kwargs):
            return 'caller command'

        self.assertIs(self._getmodel_cmd_passed_to_genbash(custom_get_getmodel_cmd=caller_builder), caller_builder)

    def test_caller_command_builder_is_used_with_a_custom_gulcalc(self):
        """A model runner that passes its own builder for its custom gulcalc must have it used.
        See https://github.com/OasisLMF/OasisLMF/issues/2255.
        """
        def caller_builder(**kwargs):
            return 'caller command'

        getmodel_cmd = self._getmodel_cmd_passed_to_genbash(executables=['fake_gulcalc'], custom_gulcalc_cmd='fake_gulcalc',
                                                            custom_get_getmodel_cmd=caller_builder)

        self.assertIs(getmodel_cmd, caller_builder)
