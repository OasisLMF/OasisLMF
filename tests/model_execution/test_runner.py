import os
import tempfile
from unittest import TestCase, mock

from oasislmf.execution import runner
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
