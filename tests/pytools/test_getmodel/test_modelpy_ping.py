"""Tests for the empty-event progress ping sent by modelpy."""
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from oasislmf.pytools.getmodel.manager import run as modelpy_run

TEST_MODEL_DIR = Path(__file__).parent.parent.parent.joinpath("assets", "test_model_1")


@pytest.mark.parametrize("socket_server,ping_expected,port_expected", [
    ('False', False, None),
    ('True', True, None),
    ('8888', True, 8888),
], ids=lambda v: f"socket_server={v}")
def test_modelpy_empty_event_ping(socket_server, ping_expected, port_expected):
    """modelpy only reports the events it drops when progress updates are enabled.

    A numeric socket_server value is the local socket server port and is passed through
    as a `port_override`, so the count reaches the server gulpy is reporting to.
    """
    with TemporaryDirectory() as tmp_dir_str:
        run_dir = Path(tmp_dir_str)
        for sub_dir in ('input', 'static'):
            shutil.copytree(TEST_MODEL_DIR.joinpath(sub_dir), run_dir.joinpath(sub_dir))
        events_stream = run_dir.joinpath('events_stream.bin')
        with open(events_stream, 'wb') as fout:
            subprocess.run("evepy 1 1", cwd=run_dir, shell=True, check=True, stdout=fout)

        with patch('oasislmf.pytools.getmodel.manager.oasis_ping') as mock_ping:
            modelpy_run(run_dir=str(run_dir), file_in=str(events_stream), file_out=str(run_dir.joinpath('out.bin')),
                        ignore_file_type=set(), data_server=False, peril_filter=None,
                        analysis_pk=None, socket_server=socket_server)

        if ping_expected:
            mock_ping.assert_called_once()
            payload = mock_ping.call_args.args[0]
            assert 'events_complete' in payload
            assert payload.get('port_override') == port_expected
        else:
            mock_ping.assert_not_called()
