import os
import tempfile
from collections import namedtuple
from unittest import TestCase, mock

import psutil

from oasislmf.execution import runner
from oasislmf.utils.exceptions import OasisException

_Uids = namedtuple('_Uids', ['real', 'effective', 'saved'])


class FakeProcess:
    def __init__(self, pid, name, uid=None, open_file_paths=None, cmdline=None,
                 raise_on_open_files=None, raise_on_cmdline=None):
        self.pid = pid
        self.info = {'pid': pid, 'name': name, 'uids': _Uids(uid, uid, uid) if uid is not None else None}
        self._open_file_paths = open_file_paths or []
        self._cmdline = cmdline or []
        self._raise_on_open_files = raise_on_open_files
        self._raise_on_cmdline = raise_on_cmdline

    def open_files(self):
        if self._raise_on_open_files:
            raise self._raise_on_open_files
        return [mock.Mock(path=p) for p in self._open_file_paths]

    def cmdline(self):
        if self._raise_on_cmdline:
            raise self._raise_on_cmdline
        return self._cmdline


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


class TestFindOpenWriters(TestCase):
    def test_returns_writer_for_pytool_process_with_open_file_in_log_dir(self):
        own_uid = os.getuid()
        proc = FakeProcess(1, 'fmpy', uid=own_uid, open_file_paths=['/log/fmpy_1.log'])
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, fully_inspected = runner._find_open_writers('/log')

        self.assertEqual(writers, [(1, 'fmpy', '/log/fmpy_1.log')])
        self.assertTrue(fully_inspected)

    def test_ignores_processes_owned_by_a_different_uid(self):
        other_uid = os.getuid() + 1
        proc = FakeProcess(1, 'fmpy', uid=other_uid, open_file_paths=['/log/fmpy_1.log'])
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, fully_inspected = runner._find_open_writers('/log')

        self.assertEqual(writers, [])
        self.assertTrue(fully_inspected)

    def test_detects_pytool_via_cmdline_when_name_is_interpreter(self):
        own_uid = os.getuid()
        proc = FakeProcess(
            1, 'python3', uid=own_uid,
            open_file_paths=['/log/fmpy_1.log'],
            cmdline=['python3', '-m', 'fmpy'],
        )
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, _fully_inspected = runner._find_open_writers('/log')

        self.assertEqual(writers, [(1, 'python3', '/log/fmpy_1.log')])

    def test_cmdline_access_denied_treated_as_not_pytool(self):
        own_uid = os.getuid()
        proc = FakeProcess(
            1, 'python3', uid=own_uid,
            open_file_paths=['/log/fmpy_1.log'],
            raise_on_cmdline=psutil.AccessDenied(1),
        )
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, fully_inspected = runner._find_open_writers('/log')

        # still matched via open file path regardless of is_pytool flag
        self.assertEqual(writers, [(1, 'python3', '/log/fmpy_1.log')])
        self.assertTrue(fully_inspected)

    def test_ignores_open_files_outside_log_dir(self):
        own_uid = os.getuid()
        proc = FakeProcess(1, 'fmpy', uid=own_uid, open_file_paths=['/elsewhere/fmpy_1.log'])
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, fully_inspected = runner._find_open_writers('/log')

        self.assertEqual(writers, [])
        self.assertTrue(fully_inspected)

    def test_access_denied_on_open_files_marks_not_fully_inspected(self):
        own_uid = os.getuid()
        proc = FakeProcess(1, 'fmpy', uid=own_uid, raise_on_open_files=psutil.AccessDenied(1))
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, fully_inspected = runner._find_open_writers('/log')

        self.assertEqual(writers, [])
        self.assertFalse(fully_inspected)

    def test_no_such_process_is_skipped(self):
        own_uid = os.getuid()
        proc = FakeProcess(1, 'fmpy', uid=own_uid, raise_on_open_files=psutil.NoSuchProcess(1))
        with mock.patch('oasislmf.execution.runner.psutil.process_iter', return_value=[proc]):
            writers, fully_inspected = runner._find_open_writers('/log')

        self.assertEqual(writers, [])
        self.assertTrue(fully_inspected)


class TestWaitForLogWriters(TestCase):
    def test_returns_once_files_stable_and_no_writers(self):
        with tempfile.TemporaryDirectory() as log_dir:
            with mock.patch('oasislmf.execution.runner._find_open_writers', return_value=([], True)), \
                    mock.patch('oasislmf.execution.runner.time.sleep'):
                runner._wait_for_log_writers(log_dir, timeout=5, poll_interval=0.01, stable_checks=2)
        # no exception / hang means success

    def test_degraded_mode_widens_stability_window_on_access_denied(self):
        with tempfile.TemporaryDirectory() as log_dir:
            times = iter(range(0, 200))

            def fake_time():
                return next(times, 999)

            with mock.patch('oasislmf.execution.runner._find_open_writers', return_value=([], False)), \
                    mock.patch('oasislmf.execution.runner.time.sleep'), \
                    mock.patch('oasislmf.execution.runner.time.time', side_effect=fake_time), \
                    mock.patch('oasislmf.execution.runner.logging.warning') as mock_warn:
                runner._wait_for_log_writers(
                    log_dir, timeout=100, poll_interval=1, stable_checks=2, degraded_stable_seconds=5.0,
                )
            self.assertTrue(mock_warn.called)

    def test_resets_stability_count_while_files_still_changing(self):
        with tempfile.TemporaryDirectory() as log_dir:
            snapshots = iter([
                {},
                {'f': (1, 1.0)},
                {'f': (2, 2.0)},  # still changing - resets stability streak
                {'f': (2, 2.0)},
                {'f': (2, 2.0)},
            ])

            def fake_snapshot(_log_dir):
                return next(snapshots, {'f': (2, 2.0)})

            with mock.patch('oasislmf.execution.runner._snapshot_log_dir', side_effect=fake_snapshot), \
                    mock.patch('oasislmf.execution.runner._find_open_writers', return_value=([], True)), \
                    mock.patch('oasislmf.execution.runner.time.sleep'):
                runner._wait_for_log_writers(log_dir, timeout=5, poll_interval=0.01, stable_checks=2)
        # no exception / hang means the instability branch was exercised and it still settled

    def test_times_out_and_logs_warning_when_writers_never_settle(self):
        with tempfile.TemporaryDirectory() as log_dir:
            with mock.patch('oasislmf.execution.runner._find_open_writers', return_value=([(1, 'fmpy', 'f')], True)), \
                    mock.patch('oasislmf.execution.runner.time.sleep'), \
                    mock.patch('oasislmf.execution.runner.logging.warning') as mock_warn:
                runner._wait_for_log_writers(log_dir, timeout=0.01, poll_interval=0.01, stable_checks=2)

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
