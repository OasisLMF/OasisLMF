from unittest.mock import MagicMock, patch

from oasislmf.cli.api import GetApiCmd
from oasislmf.utils.exceptions import OasisNoDownloadSelectedException


def _make_cmd():
    cmd = GetApiCmd.__new__(GetApiCmd)
    cmd._arg_parser = MagicMock()
    return cmd


def test_action_returns_super_result_when_no_exception_raised():
    cmd = _make_cmd()
    args = MagicMock()

    with patch('oasislmf.cli.api.OasisComputationCommand.action', return_value=0) as mock_super_action:
        result = cmd.action(args)

    mock_super_action.assert_called_once_with(args)
    cmd._arg_parser.print_help.assert_not_called()
    assert result == 0


def test_action_prints_help_and_returns_1_on_no_download_selected():
    cmd = _make_cmd()
    args = MagicMock()

    with patch(
        'oasislmf.cli.api.OasisComputationCommand.action',
        side_effect=OasisNoDownloadSelectedException('no download target selected'),
    ):
        result = cmd.action(args)

    cmd._arg_parser.print_help.assert_called_once()
    assert result == 1
