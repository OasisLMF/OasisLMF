"""Tests that katpy, aalpy, lecpy and join-summary-info report completion to the progress server."""
import sys
from unittest.mock import patch

import pytest

from oasislmf.pytools.aal import cli as aal_cli
from oasislmf.pytools.join_summary_info import cli as join_cli
from oasislmf.pytools.kat import cli as kat_cli
from oasislmf.pytools.lec import cli as lec_cli

TOOLS = [
    (kat_cli, ['katpy', '-S', '-i', 'in.bin', '-o', 'out.csv']),
    (aal_cli, ['aalpy', '-a', 'out.csv']),
    (lec_cli, ['lecpy', '-O', 'out.csv']),
    (join_cli, ['join-summary-info', '-s', 'info.csv', '-d', 'out.csv', '-o', 'out.csv']),
]


@pytest.mark.parametrize("cli_module,argv", TOOLS, ids=lambda v: getattr(v, '__name__', None))
def test_output_tool_reports_completion(cli_module, argv, monkeypatch):
    monkeypatch.setattr(sys, 'argv', argv + ["--socket-server=8888", "--analysis-pk=4"])
    with (patch.object(cli_module.manager, 'main') as mock_main,
          patch.object(cli_module, 'oasis_ping_progress') as mock_ping):
        cli_module.main()

    assert {'socket_server', 'analysis_pk'}.isdisjoint(mock_main.call_args.kwargs)
    mock_ping.assert_called_once_with('8888', '4', outputs_complete=1)


@pytest.mark.parametrize("cli_module,argv", TOOLS, ids=lambda v: getattr(v, '__name__', None))
def test_output_tool_does_not_report_failed_run(cli_module, argv, monkeypatch):
    monkeypatch.setattr(sys, 'argv', argv + ["--socket-server=8888"])
    with (patch.object(cli_module.manager, 'main', side_effect=RuntimeError("boom")),
          patch.object(cli_module, 'oasis_ping_progress') as mock_ping):
        with pytest.raises(RuntimeError):
            cli_module.main()
    mock_ping.assert_not_called()
