"""Tests for the optional keys of the lookup config."""

import filecmp
import json
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from oasislmf.computation.generate.keys import GenerateKeys
from oasislmf.lookup.builtin import Lookup, PerilCoveredDeterministicLookup
from oasislmf.lookup.factory import KeyServerFactory
from oasislmf.utils.exceptions import OasisException

META_DATA_PATH = pathlib.Path(os.path.realpath(__file__)).parent.joinpath('meta_data')


def strip_placeholders(obj):
    if isinstance(obj, dict):
        return {k: strip_placeholders(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_placeholders(v) for v in obj]
    if isinstance(obj, str):
        return obj.replace('%%KEYS_DATA_PATH%%/', '')
    return obj


def without(config, *keys):
    return {k: v for k, v in config.items() if k not in keys}


def shipped_config():
    with open(META_DATA_PATH.joinpath('lookup_config.json'), encoding='utf-8') as f:
        return json.load(f)


def write_model_dir(target_dir, config, data_subdir=None):
    data_dir = pathlib.Path(target_dir, data_subdir) if data_subdir else pathlib.Path(target_dir)
    shutil.copytree(META_DATA_PATH, data_dir, dirs_exist_ok=True)
    for stale in ('lookup_config.json', 'lookup_config-geo_grid.json'):
        pathlib.Path(data_dir, stale).unlink(missing_ok=True)

    config_fp = pathlib.Path(target_dir, 'lookup_config.json')
    with open(config_fp, 'w', encoding='utf-8') as f:
        json.dump(config, f)
    return config_fp


def run_keys(config, tmp_dir, data_subdir=None):
    model_dir = pathlib.Path(tmp_dir, 'model')
    model_dir.mkdir(parents=True, exist_ok=True)
    config_fp = write_model_dir(model_dir, config, data_subdir=data_subdir)

    out_dir = pathlib.Path(tmp_dir, 'out')
    out_dir.mkdir(parents=True, exist_ok=True)
    keys_fp = pathlib.Path(out_dir, 'keys.csv')

    GenerateKeys(
        oed_location_csv=META_DATA_PATH.joinpath('location.csv'),
        lookup_config_json=config_fp,
        keys_data_path=str(keys_fp),
        keys_format='oasis',
    ).run()
    return keys_fp, pathlib.Path(out_dir, 'keys-errors.csv')


def assert_matches_golden(keys_fp, keys_errors_fp):
    assert filecmp.cmp(keys_fp, META_DATA_PATH.joinpath('keys.csv'), shallow=False)
    assert filecmp.cmp(keys_errors_fp, META_DATA_PATH.joinpath('keys-errors.csv'), shallow=False)


CONFIG_VARIANTS = {
    'as_shipped': lambda c: c,
    'no_model': lambda c: without(c, 'model'),
    'no_builtin_lookup_type': lambda c: without(c, 'builtin_lookup_type'),
    'no_keys_data_path': lambda c: without(c, 'keys_data_path'),
    'no_placeholders': lambda c: strip_placeholders(c),
    'no_placeholders_no_keys_data_path': lambda c: without(strip_placeholders(c), 'keys_data_path'),
    'minimal': lambda c: without(
        strip_placeholders(c), 'model', 'builtin_lookup_type', 'keys_data_path'),
}


@pytest.mark.parametrize('variant', list(CONFIG_VARIANTS))
def test_optional_keys_omitted___keys_are_unchanged(variant):
    config = CONFIG_VARIANTS[variant](shipped_config())
    with TemporaryDirectory() as d:
        assert_matches_golden(*run_keys(config, d))


def test_keys_data_path_omitted___defaults_to_config_dir_not_cwd():
    config = without(strip_placeholders(shipped_config()), 'keys_data_path')
    with TemporaryDirectory() as d:
        model_dir = pathlib.Path(d, 'model')
        model_dir.mkdir()
        config_fp = write_model_dir(model_dir, config)

        _, key_server = KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)
        lookup = key_server.lookup_cls(key_server.config, config_dir=key_server.config_dir, output_dir=d)

        assert lookup.config['keys_data_path'] == str(model_dir.resolve())
        assert lookup.config['keys_data_path'] != os.getcwd()


def test_keys_data_path_is_relative___resolved_against_config_dir():
    config = dict(shipped_config(), keys_data_path='keys_data')
    with TemporaryDirectory() as d:
        keys_fp, keys_errors_fp = run_keys(config, d, data_subdir='keys_data')
        assert_matches_golden(keys_fp, keys_errors_fp)


def test_keys_data_path_does_not_exist___error_is_raised():
    config = dict(shipped_config(), keys_data_path='no_such_dir')
    with TemporaryDirectory() as d:
        with pytest.raises(OasisException, match='keys_data_path'):
            run_keys(config, d)


def test_placeholder_with_keys_data_path_omitted___resolves_to_config_dir():
    config = without(shipped_config(), 'keys_data_path')
    with TemporaryDirectory() as d:
        assert_matches_golden(*run_keys(config, d))


def test_keys_data_storage___is_used_instead_of_the_default_keys_data_path():
    config = without(strip_placeholders(shipped_config()), 'keys_data_path')
    config['keys_data_storage'] = {
        'storage_class': 'oasis_data_manager.filestore.backends.local.LocalStorage',
        'options': {'root_dir': str(META_DATA_PATH)},
    }
    with TemporaryDirectory() as d:
        model_dir = pathlib.Path(d, 'empty_model')
        model_dir.mkdir()
        config_fp = pathlib.Path(model_dir, 'lookup_config.json')
        with open(config_fp, 'w', encoding='utf-8') as f:
            json.dump(config, f)

        _, key_server = KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)
        lookup = key_server.lookup_cls(key_server.config, config_dir=key_server.config_dir, output_dir=d)

        assert lookup.storage.root_dir == str(META_DATA_PATH)
        keys_df = lookup.process_locations(
            pd.read_csv(META_DATA_PATH.joinpath('location.csv')).assign(loc_id=lambda df: range(1, len(df) + 1)))
        assert not keys_df.empty


def test_model_omitted___create_returns_no_model_info():
    config = without(shipped_config(), 'model')
    with TemporaryDirectory() as d:
        config_fp = write_model_dir(pathlib.Path(d), config)
        model_info, key_server = KeyServerFactory.create(
            lookup_config_fp=str(config_fp), output_directory=d)

        assert model_info is None
        assert key_server.lookup_cls is Lookup


def test_model_omitted_with_custom_lookup_module___error_is_raised():
    with TemporaryDirectory() as d:
        module_fp = pathlib.Path(d, 'custom_lookup.py')
        with open(module_fp, 'w', encoding='utf-8') as f:
            f.write('from oasislmf.lookup.builtin import Lookup\n'
                    'class MyModelKeysLookup(Lookup):\n'
                    '    pass\n')

        config = without(shipped_config(), 'model')
        config['lookup_module_path'] = str(module_fp)
        config_fp = write_model_dir(pathlib.Path(d), config)

        with pytest.raises(KeyError):
            KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)


def test_builtin_lookup_type_omitted___step_definition_selects_new_lookup():
    config = without(shipped_config(), 'builtin_lookup_type')
    with TemporaryDirectory() as d:
        config_fp = write_model_dir(pathlib.Path(d), config)
        _, key_server = KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)
        assert key_server.lookup_cls is Lookup


def test_deprecated_pre_1_17_config___error_still_names_the_old_module():
    config = {
        'model': {'supplier_id': 'OasisLMF', 'model_id': 'PiWind', 'model_version': '1'},
        'keys_data_path': './',
        'peril': {'peril_ids': ['WTC']},
        'coverage': {'coverage_types': [1]},
        'vulnerability': {},
    }
    with TemporaryDirectory() as d:
        config_fp = write_model_dir(pathlib.Path(d), config)
        with pytest.raises(OasisException, match='oasislmf<=1.16.0'):
            KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)


def test_builtin_lookup_type_unrecognised___error_is_raised():
    config = dict(shipped_config(), builtin_lookup_type='not_a_lookup_type')
    with TemporaryDirectory() as d:
        config_fp = write_model_dir(pathlib.Path(d), config)
        with pytest.raises(OasisException, match='Unrecognised lookup config'):
            KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)


def test_peril_covered_deterministic___is_still_selected():
    config = {
        'builtin_lookup_type': 'peril_covered_deterministic',
        'supported_oed_coverage_types': [1],
        'model_perils_covered': ['AA1'],
    }
    with TemporaryDirectory() as d:
        _, key_server = KeyServerFactory.create(lookup_config=config, output_directory=d)
        assert key_server.lookup_cls is PerilCoveredDeterministicLookup


@pytest.mark.parametrize('missing_key', ['step_definition', 'strategy'])
def test_step_definition_or_strategy_missing___error_names_the_key(missing_key):
    config = without(shipped_config(), missing_key)
    locations = pd.DataFrame({'loc_id': [1], 'occupancycode': [1050]})
    with TemporaryDirectory() as d:
        config_fp = write_model_dir(pathlib.Path(d), config)
        _, key_server = KeyServerFactory.create(lookup_config_fp=str(config_fp), output_directory=d)
        lookup = key_server.lookup_cls(key_server.config, config_dir=key_server.config_dir, output_dir=d)

        with pytest.raises(OasisException, match=missing_key):
            lookup.process_locations(locations)


def test_bare_lookup_is_constructible_for_builders():
    assert Lookup(config={}).build_split_loc_perils_covered(model_perils_covered=['WTC']) is not None
