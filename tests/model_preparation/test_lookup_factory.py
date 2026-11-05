"""Tests of the configuration paths through KeyServerFactory and BasicKeyServer in oasislmf/lookup/factory.py."""
import json

import pandas as pd
import pytest

from oasislmf.lookup.builtin import Lookup as NewLookup
from oasislmf.lookup.builtin import PerilCoveredDeterministicLookup
from oasislmf.lookup.factory import BasicKeyServer, KeyServerFactory
from oasislmf.utils.exceptions import OasisException

MODEL = {'supplier_id': 'Acme', 'model_id': 'Quake', 'model_version': '1'}
BUILTIN_CONFIG = {'model': MODEL, 'builtin_lookup_type': 'peril_covered_deterministic'}

CUSTOM_LOOKUP_MODULE = '''
class QuakeKeysLookup:
    interface_version = '1'

    def __init__(self, config, config_dir=None, user_data_dir=None, output_dir=None):
        self.config = config
'''


def write_module(path, source):
    path.write_text(source)
    return str(path)


def test_lookup_config_json_is_read_like_a_dict():
    model_info, key_server = KeyServerFactory.create(lookup_config_json=json.dumps(BUILTIN_CONFIG))

    assert model_info == MODEL
    assert key_server.config == BUILTIN_CONFIG
    assert key_server.lookup_cls is PerilCoveredDeterministicLookup


def test_run_settings_passed_as_a_dict_go_into_the_config():
    settings = {'some_run_setting': 1}

    _, key_server = KeyServerFactory.create(lookup_config=dict(BUILTIN_CONFIG), complex_lookup_config_fp=settings)

    assert key_server.config['settings'] == settings
    assert not hasattr(key_server, 'complex_lookup_config_fp')


def test_custom_key_server_module_is_loaded(tmp_path):
    module_path = write_module(tmp_path / 'key_server.py', '''
class QuakeKeysServer:
    interface_version = '1'

    def __init__(self, config, config_dir=None, user_data_dir=None, output_dir=None):
        self.config = config
''')

    _, key_server = KeyServerFactory.create(lookup_config={'model': MODEL, 'key_server_module_path': module_path})

    assert type(key_server).__name__ == 'QuakeKeysServer'


def test_unsupported_key_server_interface_version_is_rejected(tmp_path):
    module_path = write_module(tmp_path / 'key_server.py', '''
class QuakeKeysServer:
    interface_version = '2'
''')

    with pytest.raises(OasisException, match='interface version 2'):
        KeyServerFactory.create(lookup_config={'model': MODEL, 'key_server_module_path': module_path})


def test_lookup_class_is_imported_from_its_dotted_path():
    config = {'model': MODEL, 'lookup_class': 'oasislmf.lookup.builtin.PerilCoveredDeterministicLookup'}

    assert BasicKeyServer(config).lookup_cls is PerilCoveredDeterministicLookup


def test_lookup_module_provides_the_model_lookup_class(tmp_path, monkeypatch):
    write_module(tmp_path / 'acme_quake_lookup.py', CUSTOM_LOOKUP_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))

    lookup_cls = BasicKeyServer({'model': MODEL, 'lookup_module': 'acme_quake_lookup'}).lookup_cls

    assert lookup_cls.__name__ == 'QuakeKeysLookup'


def test_relative_lookup_module_path_is_resolved_against_the_config_dir(tmp_path):
    write_module(tmp_path / 'lookup.py', CUSTOM_LOOKUP_MODULE)

    lookup_cls = BasicKeyServer({'model': MODEL, 'lookup_module_path': 'lookup.py'}, config_dir=str(tmp_path)).lookup_cls

    assert lookup_cls.__name__ == 'QuakeKeysLookup'


def test_step_definition_selects_the_new_builtin_lookup():
    assert BasicKeyServer({'model': MODEL, 'step_definition': {}}).lookup_cls is NewLookup


def test_unknown_builtin_lookup_type_is_rejected():
    with pytest.raises(OasisException, match='Unrecognised lookup config'):
        BasicKeyServer({'model': MODEL, 'builtin_lookup_type': 'not_a_lookup'})


class LegacyLookup:
    interface_version = '0'

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize('output_dir, user_data_dir, expected_extra', [
    (None, None, {}),
    ('out', None, {'complex_lookup_config_fp': 'cfg', 'output_directory': 'out'}),
    ('out', 'user', {'complex_lookup_config_fp': 'cfg', 'output_directory': 'out', 'user_data_dir': 'user'}),
])
def test_deprecated_lookup_interface_gets_the_legacy_arguments(output_dir, user_data_dir, expected_extra):
    config = {'model': MODEL, 'keys_data_path': 'keys'}

    with pytest.warns(UserWarning, match='deprecated'):
        lookup = BasicKeyServer.create_lookup(LegacyLookup, config, 'cfg', user_data_dir, output_dir, lookup_id=0)

    assert lookup.kwargs == {'keys_data_directory': 'keys', 'supplier': 'Acme', 'model_name': 'Quake',
                             'model_version': '1', **expected_extra}


def test_unknown_lookup_interface_version_is_rejected():
    class FutureLookup:
        interface_version = '9'

    with pytest.raises(OasisException, match='interface version 9'):
        BasicKeyServer.create_lookup(FutureLookup, {'model': MODEL}, '.', None, None, lookup_id=0)


@pytest.mark.parametrize('columns, keys_success_msg, expected', [
    (['model_data', 'amplification_id'], False, 'model_data_with_amplification_heading_row'),
    (['model_data'], False, 'model_data_heading_row'),
    (['amplification_id', 'intensity_adjustment', 'return_period'], False, 'dynamic_model_with_amplification_data_heading_row'),
    (['intensity_adjustment', 'return_period'], False, 'dynamic_model_data_heading_row'),
    (['amplification_id'], True, 'key_success_with_amplification_and_message_heading_row'),
    ([], True, 'key_success_with_message_heading_row'),
    (['amplification_id'], False, 'key_success_with_amplification_heading_row'),
    ([], False, 'key_success_heading_row'),
])
def test_success_heading_row_matches_the_key_columns(columns, keys_success_msg, expected):
    key_server = BasicKeyServer(BUILTIN_CONFIG)

    assert key_server.get_success_heading_row(columns, keys_success_msg) is getattr(key_server, expected)


def test_json_keys_file_without_an_errors_path_counts_no_errors(tmp_path):
    results = [pd.DataFrame({'loc_id': [1, 2], 'status': ['success', 'fail']}), pd.DataFrame(columns=['loc_id', 'status'])]
    successes_fp = tmp_path / 'keys.json'

    successes, errors = BasicKeyServer(BUILTIN_CONFIG).write_json_keys_file(iter(results), False, successes_fp, None)

    assert (successes, errors) == (1, 0)
    assert [row['loc_id'] for row in json.loads(successes_fp.read_text())] == [1]
