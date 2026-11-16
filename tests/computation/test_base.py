import json
import logging
import os

import pytest

from oasislmf.computation.base import ComputationStep
from oasislmf.utils.log_config import OasisLogConfig


class DummyComputationStep(ComputationStep):
    """Minimal concrete ComputationStep used to exercise _apply_log_config()."""

    def run(self):
        pass


@pytest.fixture
def oasis_loggers():
    """Give each test a clean 'oasislmf'/'ods_tools' logger state to assert against,
    and restore whatever was there before once the test is done so other tests /
    modules relying on global logger state aren't affected.
    """
    logger = logging.getLogger('oasislmf')
    ods_logger = logging.getLogger('ods_tools')

    orig_level = logger.level
    orig_ods_level = ods_logger.level
    orig_handlers = list(logger.handlers)

    for h in orig_handlers:
        logger.removeHandler(h)

    handler = logging.StreamHandler()
    handler.name = 'oasislmf'
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    ods_logger.setLevel(logging.WARNING)

    yield logger, ods_logger, handler

    logger.removeHandler(handler)
    for h in orig_handlers:
        logger.addHandler(h)
    logger.setLevel(orig_level)
    ods_logger.setLevel(orig_ods_level)


def test_no_log_kwargs__logger_untouched(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep()

    assert logger.level == logging.WARNING
    assert ods_logger.level == logging.WARNING
    assert handler.formatter._fmt == '%(message)s'


def test_verbose_only__logger_untouched(oasis_loggers):
    """verbose must never influence the logger here - that's setup_logger()'s job,
    and it has context (the nested MDK 'logging' config block) this class doesn't."""
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(verbose=True)

    assert logger.level == logging.WARNING
    assert ods_logger.level == logging.WARNING
    assert handler.formatter._fmt == '%(message)s'


def test_log_level_set__applied_to_logger_and_ods_tools(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_level='DEBUG')

    assert logger.level == logging.DEBUG
    # DEBUG on the main logger also raises ods_tools to DEBUG (see get_ods_tools_level)
    assert ods_logger.level == logging.DEBUG


def test_log_level_set_non_debug__ods_tools_stays_warning(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_level='ERROR')

    assert logger.level == logging.ERROR
    assert ods_logger.level == logging.WARNING


def test_log_format_set__formatter_applied_to_all_handlers(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_format='compact')

    expected_fmt = OasisLogConfig.FORMAT_TEMPLATES['compact']
    assert handler.formatter._fmt == expected_fmt
    # log_format alone must not touch the level
    assert logger.level == logging.WARNING


def test_log_level_and_log_format_set_together(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_level='DEBUG', log_format='simple')

    assert logger.level == logging.DEBUG
    assert ods_logger.level == logging.DEBUG
    assert handler.formatter._fmt == OasisLogConfig.FORMAT_TEMPLATES['simple']


def test_verbose_does_not_override_explicit_log_level(oasis_loggers):
    """Regression test: verbose must not be folded into the level computation
    alongside an explicit log_level - the explicit value always wins outright."""
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(verbose=True, log_level='ERROR')

    assert logger.level == logging.ERROR


def test_global_params_choices_and_no_cli_flag():
    """log_level/log_format/verbose should be collectable step params with the right
    choices, so they show up correctly in the generated computation settings schema,
    but must not be exposed as their own CLI flag (no 'help' key) since those are
    already registered directly by OasisBaseCommand."""
    params = {p['name']: p for p in DummyComputationStep.get_params()}

    assert params['log_level']['choices'] == OasisLogConfig.STANDARD_LEVELS
    assert params['log_format']['choices'] == list(OasisLogConfig.FORMAT_TEMPLATES.keys())
    assert params['verbose']['default'] is False

    for name in ('verbose', 'log_level', 'log_format'):
        assert 'help' not in params[name]


def test_mdk_config_logging_block_honoured_when_log_level_reapplied(oasis_loggers, tmp_path):
    """ods_tools_level from the MDK config file's "logging" block must still be applied
    when the level is re-applied from computation settings."""
    logger, ods_logger, handler = oasis_loggers
    config_fp = tmp_path / 'oasislmf.json'
    config_fp.write_text(json.dumps({'logging': {'ods_tools_level': 'ERROR'}}))

    DummyComputationStep(log_level='DEBUG', config=str(config_fp))

    assert logger.level == logging.DEBUG
    assert ods_logger.level == logging.ERROR


@pytest.mark.parametrize('config_content', [None, '{not valid json'])
def test_unreadable_mdk_config__warns_and_falls_back_to_defaults(oasis_loggers, tmp_path, caplog, config_content):
    logger, ods_logger, handler = oasis_loggers
    config_fp = tmp_path / 'oasislmf.json'
    if config_content is not None:
        config_fp.write_text(config_content)

    with caplog.at_level(logging.WARNING, logger='oasislmf.computation.base'):
        DummyComputationStep(log_level='DEBUG', config=str(config_fp))

    assert f"Could not re-load MDK config file for logging: {config_fp}" in caplog.text
    assert logger.level == logging.DEBUG
    assert ods_logger.level == logging.DEBUG


def test_log_format_set__other_handlers_untouched(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers
    other_handler = logging.StreamHandler()
    other_handler.name = 'other'
    other_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(other_handler)
    try:
        DummyComputationStep(log_format='compact')
    finally:
        logger.removeHandler(other_handler)

    assert handler.formatter._fmt == OasisLogConfig.FORMAT_TEMPLATES['compact']
    assert other_handler.formatter._fmt == '%(message)s'


def test_get_default_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    DummyComputationStep.run_dir_key = 'dummy'
    try:
        run_dir = DummyComputationStep.get_default_run_dir()
    finally:
        del DummyComputationStep.run_dir_key

    assert os.path.dirname(run_dir) == os.path.join(str(tmp_path), 'runs')
    assert os.path.basename(run_dir).startswith('dummy-')


class DummySchemaStep(DummyComputationStep):
    step_params = [
        {'name': 'some_flag', 'default': False, 'help': 'A boolean flag'},
        {'name': 'some_choice', 'choices': ['a', 'b']},
        {'name': 'some_count', 'type': int},
        {'name': 'some_ratio', 'type': float},
        {'name': 'some_settings_json', 'is_path': True},
    ]
    settings_params = [{'name': 'some_settings_json', 'loader': json.load}]


def test_computation_settings_json_schema():
    """Global logging params appear in the schema, but 'config' (exclude_from_schema)
    and settings files (settings_params) must not."""
    properties = DummySchemaStep.get_computation_settings_json_schema()['properties']

    assert 'config' not in properties
    assert 'some_settings_json' not in properties
    assert properties['log_level']['enum'] == OasisLogConfig.STANDARD_LEVELS
    assert properties['log_format']['enum'] == list(OasisLogConfig.FORMAT_TEMPLATES.keys())
    assert properties['verbose'] == {'type': 'boolean'}
    assert properties['some_flag'] == {'type': 'boolean', 'description': 'A boolean flag'}
    assert properties['some_choice'] == {'type': 'string', 'enum': ['a', 'b']}
    assert properties['some_count'] == {'type': 'integer'}
    assert properties['some_ratio'] == {'type': 'number'}
