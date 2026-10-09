import json
import os
import subprocess
import tempfile
from tempfile import TemporaryDirectory
from unittest import TestCase, mock
from unittest.mock import patch

from oasislmf.execution import runner
from oasislmf.execution.bash import bash_params
from oasislmf.execution.runner import rerun
from oasislmf.utils.exceptions import OasisException


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

    def test_markers_match_as_grep_basic_regex(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'gul_stderror.err'), 'w') as f:
                f.write('Start (gulcalc) 1\nStart (gulcalc) 2\nfinished after 12 events\nfinished after 7 events\n')
            self.assertEqual(runner._find_incomplete_pytool_logs(tmp_dir, 'Start (gulcalc)', 'finished.*[0-9]\\+ events'), {})

    def test_invalid_marker_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, 'gul_stderror.err'), 'w') as f:
                f.write('gulcalc started\n')
            with self.assertRaisesRegex(OasisException, 'Invalid custom gulcalc log marker'):
                runner._find_incomplete_pytool_logs(tmp_dir, 'gulcalc [started', 'gulcalc finished')


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
