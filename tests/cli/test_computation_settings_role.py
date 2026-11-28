"""CLI side of computation_settings_role (#1819): computation_settings from settings files are merged
into the MDK config values, which explicit command line flags still take priority over."""
import json
import logging
from argparse import Namespace

import pytest

from oasislmf.cli.command import OasisComputationCommand
from oasislmf.computation.base import ComputationStep


def _load_json(fp):
    with open(fp) as f:
        return json.load(f)


class DummyStep(ComputationStep):
    step_params = [
        {'name': 'num_processes', 'type': int, 'default': 2, 'help': 'n'},
        {'name': 'mode', 'default': 'fast', 'help': 'm'},
        {'name': 'analysis_settings_json', 'is_path': True, 'help': 'a'},
    ]
    settings_params = [{'name': 'analysis_settings_json', 'loader': _load_json, 'user_role': 'user'}]


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.delenv('OASIS_ENV_OVERRIDE', raising=False)
    analysis = tmp_path / 'analysis_settings.json'
    analysis.write_text(json.dumps({'computation_settings': {'num_processes': 8, 'mode': 'slow'}}))
    config = tmp_path / 'oasislmf.json'
    config.write_text(json.dumps({'num_processes': 4}))
    return str(analysis), str(config)


def get_arguments(analysis, config, **cli):
    args = Namespace(config=config, analysis_settings_json=analysis, num_processes=None, mode=None,
                     computation_settings_role=None, verbose=False, log_level=None, log_format=None)
    for key, value in cli.items():
        setattr(args, key, value)
    return OasisComputationCommand.get_arguments(args, DummyStep)


def test_cli_default__settings_file_role_kept(files, caplog):
    """Default unchanged: analysis settings ('user' role) change defaults but not MDK config values."""
    with caplog.at_level(logging.WARNING, logger='oasislmf'):
        kwargs = get_arguments(*files)
    assert (kwargs['num_processes'], kwargs['mode']) == (4, 'slow')
    assert "'num_processes': 8 ignored" in caplog.text


def test_cli_admin__settings_override_mdk_config(files):
    kwargs = get_arguments(*files, computation_settings_role='admin')
    assert (kwargs['num_processes'], kwargs['mode']) == (8, 'slow')


def test_cli_user__cannot_override_mdk_config_or_flags(files):
    kwargs = get_arguments(*files, computation_settings_role='user')
    assert (kwargs['num_processes'], kwargs['mode']) == (4, 'slow')
    kwargs = get_arguments(*files, computation_settings_role='user', mode='medium')
    assert kwargs['mode'] == 'medium'


def test_cli_role_from_mdk_config(files, tmp_path):
    analysis, _ = files
    config = tmp_path / 'oasislmf_admin.json'
    config.write_text(json.dumps({'num_processes': 4, 'computation_settings_role': 'admin'}))
    kwargs = get_arguments(analysis, str(config))
    assert kwargs['num_processes'] == 8
