from unittest import TestCase
from unittest.mock import patch, MagicMock

from oasislmf.cli.root import RootCmd, main
from oasislmf.utils.exceptions import OasisException


class RootCmdRun(TestCase):
    def setUp(self):
        self.cmd = RootCmd(argv=[])
        self.cmd._logger = MagicMock()

    def test_oasis_exception_raised___verbose_off___error_logged_and_exit_code_1(self):
        self.cmd.log_verbose = False

        with patch('argparsetree.BaseCommand.run', side_effect=OasisException('boom')):
            result = self.cmd.run()

        self.assertEqual(result, 1)
        self.cmd.logger.error.assert_called_once_with('boom')
        self.cmd.logger.exception.assert_not_called()

    def test_oasis_exception_raised___verbose_on___exception_logged_with_traceback_and_exit_code_1(self):
        self.cmd.log_verbose = True

        with patch('argparsetree.BaseCommand.run', side_effect=OasisException('boom')):
            result = self.cmd.run()

        self.assertEqual(result, 1)
        self.cmd.logger.exception.assert_called_once_with('boom')
        self.cmd.logger.error.assert_not_called()

    def test_keyboard_interrupt_raised___error_logged_and_exit_code_130(self):
        with patch('argparsetree.BaseCommand.run', side_effect=KeyboardInterrupt()):
            result = self.cmd.run()

        self.assertEqual(result, 130)
        self.cmd.logger.error.assert_called_once_with('Aborted (Ctrl-C)')

    def test_run_succeeds___return_value_passed_through(self):
        with patch('argparsetree.BaseCommand.run', return_value=0):
            result = self.cmd.run()

        self.assertEqual(result, 0)


class RootCmdMain(TestCase):
    def test_main___calls_sys_exit_with_run_result(self):
        with patch('oasislmf.cli.root.RootCmd') as mock_root_cmd_cls, \
                patch('oasislmf.cli.root.sys.exit') as mock_exit:
            mock_root_cmd_cls.return_value.run.return_value = 0

            main()

            mock_root_cmd_cls.return_value.run.assert_called_once_with()
            mock_exit.assert_called_once_with(0)
