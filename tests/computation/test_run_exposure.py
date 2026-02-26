import os
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import pandas as pd

from oasislmf.computation.run.exposure import RunExposure
from oasislmf.utils.defaults import KERNEL_ALLOC_FM_MAX
from oasislmf.utils.exceptions import OasisException

from .test_computation import ComputationChecker

ASSETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'test_exposure_run')
LOCATION = os.path.join(ASSETS_DIR, 'location.csv')
ACCOUNTS = os.path.join(ASSETS_DIR, 'account.csv')
RI_INFO = os.path.join(ASSETS_DIR, 'ri_info.csv')
RI_SCOPE = os.path.join(ASSETS_DIR, 'ri_scope.csv')
LOCATION_INVALID = os.path.join(ASSETS_DIR, 'location_invalid.csv')
CURRENCY_CONFIG = os.path.join(ASSETS_DIR, 'currency_config.json')
EXPECTED_ACC_LOC = os.path.join(ASSETS_DIR, 'expected_output_acc_loc.csv')
EXPECTED_ACC_LOC_USD = os.path.join(ASSETS_DIR, 'expected_output_acc_loc_usd.csv')
EXPECTED_ALL = os.path.join(ASSETS_DIR, 'expected_output_all.csv')
EXPECTED_ALL_USD = os.path.join(ASSETS_DIR, 'expected_output_all_usd.csv')
EXPECTED_LOSS_HALF = os.path.join(ASSETS_DIR, 'expected_loss_factor_half.csv')

BASE_PARAMS = dict(
    model_perils_covered=['WW1'],
    keys_format='oasis',
    print_summary=False,
)


def _make_step(**kwargs):
    """Construct a RunExposure without providing src_dir (None skips pre_exist)."""
    return RunExposure(**kwargs)


def _step_with_oasis_files_dir(tmp_dir_path, **kwargs):
    """Construct a step and set oasis_files_dir as run() would."""
    step = _make_step(**kwargs)
    step.oasis_files_dir = tmp_dir_path
    return step


class TestGetExposureDataConfigResolution(ComputationChecker):
    def setUp(self):
        self.tmp = self.tmp_dir()

    def test_no_overrides_uses_find_exposure_fp_for_all_files(self):
        step = _step_with_oasis_files_dir(self.tmp.name)
        with patch('oasislmf.computation.run.exposure.find_exposure_fp') as mock_fp:
            mock_fp.side_effect = lambda d, k, **kw: f'/fallback/{k}.csv'
            config = step.get_exposure_data_config()

        assert config['location'] == '/fallback/loc.csv'
        assert config['account'] == '/fallback/acc.csv'
        assert config['ri_info'] == '/fallback/info.csv'
        assert config['ri_scope'] == '/fallback/scope.csv'

    def test_no_overrides_passes_oasis_files_dir_to_find_exposure_fp(self):
        step = _step_with_oasis_files_dir(self.tmp.name)
        with patch('oasislmf.computation.run.exposure.find_exposure_fp') as mock_fp:
            mock_fp.return_value = None
            step.get_exposure_data_config()

        dirs_used = {call.args[0] for call in mock_fp.call_args_list}
        self.assertEqual(dirs_used, {self.tmp.name})

    def test_oed_location_csv_bypasses_fallback_for_location_only(self):
        with NamedTemporaryFile(suffix='.csv') as loc_file:
            step = _step_with_oasis_files_dir(self.tmp.name, oed_location_csv=loc_file.name)
            with patch('oasislmf.computation.run.exposure.find_exposure_fp') as mock_fp:
                mock_fp.return_value = '/fallback.csv'
                config = step.get_exposure_data_config()

        self.assertEqual(config['location'], loc_file.name)
        self.assertEqual(config['account'], '/fallback.csv')
        self.assertEqual(config['ri_info'], '/fallback.csv')
        self.assertEqual(config['ri_scope'], '/fallback.csv')

    def test_all_overrides_bypass_find_exposure_fp_entirely(self):
        with (
            NamedTemporaryFile(suffix='.csv') as loc_file,
            NamedTemporaryFile(suffix='.csv') as acc_file,
            NamedTemporaryFile(suffix='.csv') as info_file,
            NamedTemporaryFile(suffix='.csv') as scope_file,
        ):
            step = _step_with_oasis_files_dir(
                self.tmp.name,
                oed_location_csv=loc_file.name,
                oed_accounts_csv=acc_file.name,
                oed_info_csv=info_file.name,
                oed_scope_csv=scope_file.name,
            )
            with patch('oasislmf.computation.run.exposure.find_exposure_fp') as mock_fp:
                config = step.get_exposure_data_config()

        mock_fp.assert_not_called()
        self.assertEqual(config['location'], loc_file.name)
        self.assertEqual(config['account'], acc_file.name)
        self.assertEqual(config['ri_info'], info_file.name)
        self.assertEqual(config['ri_scope'], scope_file.name)

    def test_account_and_oed_info_csvs_only(self):
        with (
            NamedTemporaryFile(suffix='.csv') as acc_file,
            NamedTemporaryFile(suffix='.csv') as info_file,
        ):
            step = _step_with_oasis_files_dir(
                self.tmp.name,
                oed_accounts_csv=acc_file.name,
                oed_info_csv=info_file.name,
            )
            with patch('oasislmf.computation.run.exposure.find_exposure_fp') as mock_fp:
                mock_fp.return_value = '/fallback.csv'
                config = step.get_exposure_data_config()

        self.assertEqual(config['location'], '/fallback.csv')
        self.assertEqual(config['account'], acc_file.name)
        self.assertEqual(config['ri_info'], info_file.name)
        self.assertEqual(config['ri_scope'], '/fallback.csv')


class TestGetExposureDataConfigSettings(ComputationChecker):
    def setUp(self):
        self.tmp = self.tmp_dir()

    def _config(self, **kwargs):
        step = _step_with_oasis_files_dir(self.tmp.name, **kwargs)
        with patch('oasislmf.computation.run.exposure.find_exposure_fp', return_value=None):
            return step.get_exposure_data_config()

    def test_check_oed_true_is_passed_through(self):
        self.assertTrue(self._config(check_oed=True)['check_oed'])

    def test_check_oed_false_is_passed_through(self):
        self.assertFalse(self._config(check_oed=False)['check_oed'])

    def test_reporting_currency_is_passed_through(self):
        config = self._config(reporting_currency='USD')
        self.assertEqual(config['reporting_currency'], 'USD')

    def test_backend_dtype_is_passed_through(self):
        config = self._config(oed_backend_dtype='pa_dtype')
        self.assertEqual(config['backend_dtype'], 'pa_dtype')

    def test_oed_schema_info_is_passed_through_when_set(self):
        config = self._config(oed_schema_info='v3.0.0')
        self.assertEqual(config['oed_schema_info'], 'v3.0.0')

    def test_oed_schema_info_falls_back_to_settings_oed_version(self):
        step = _step_with_oasis_files_dir(self.tmp.name)
        step.settings = {'oed_version': 'v2.0.0'}
        with patch('oasislmf.computation.run.exposure.find_exposure_fp', return_value=None):
            config = step.get_exposure_data_config()
        self.assertEqual(config['oed_schema_info'], 'v2.0.0')

    def test_oed_schema_info_none_when_neither_set(self):
        config = self._config()
        self.assertIsNone(config['oed_schema_info'])

    def test_use_field_always_true(self):
        self.assertTrue(self._config()['use_field'])

    def test_config_contains_all_expected_keys(self):
        expected = {'location', 'account', 'ri_info', 'ri_scope',
                    'oed_schema_info', 'currency_conversion', 'check_oed',
                    'use_field', 'reporting_currency', 'backend_dtype'}
        self.assertEqual(set(self._config().keys()), expected)


class TestOverrideParamPreExist(ComputationChecker):

    def test_oed_location_csv_with_nonexistent_file_raises_at_init(self):
        with self.assertRaises(OasisException):
            RunExposure(oed_location_csv='/nonexistent/path/location.csv')

    def test_oed_accounts_csv_with_nonexistent_file_raises_at_init(self):
        with self.assertRaises(OasisException):
            RunExposure(oed_accounts_csv='/nonexistent/path/account.csv')

    def test_oed_info_csv_with_nonexistent_file_raises_at_init(self):
        with self.assertRaises(OasisException):
            RunExposure(oed_info_csv='/nonexistent/path/ri_info.csv')

    def test_oed_scope_csv_with_nonexistent_file_raises_at_init(self):
        with self.assertRaises(OasisException):
            RunExposure(oed_scope_csv='/nonexistent/path/ri_scope.csv')

    def test_existing_oed_location_csv_does_not_raise(self):
        with NamedTemporaryFile(suffix='.csv') as f:
            step = RunExposure(oed_location_csv=f.name)
            self.assertEqual(step.oed_location_csv, f.name)

    def test_all_overrides_with_existing_files_do_not_raise(self):
        with (
            NamedTemporaryFile(suffix='.csv') as loc,
            NamedTemporaryFile(suffix='.csv') as acc,
            NamedTemporaryFile(suffix='.csv') as info,
            NamedTemporaryFile(suffix='.csv') as scope,
        ):
            step = RunExposure(
                oed_location_csv=loc.name,
                oed_accounts_csv=acc.name,
                oed_info_csv=info.name,
                oed_scope_csv=scope.name,
            )
            self.assertEqual(step.oed_location_csv, loc.name)
            self.assertEqual(step.oed_accounts_csv, acc.name)
            self.assertEqual(step.oed_info_csv, info.name)
            self.assertEqual(step.oed_scope_csv, scope.name)


class TestCheckAllocRules(ComputationChecker):

    def test_default_alloc_rules_do_not_raise(self):
        step = _make_step()
        step._check_alloc_rules()  # should not raise

    def test_il_rule_above_max_raises(self):
        step = _make_step(kernel_alloc_rule_il=KERNEL_ALLOC_FM_MAX + 1)
        with self.assertRaises(OasisException):
            step._check_alloc_rules()

    def test_il_rule_below_zero_raises(self):
        step = _make_step(kernel_alloc_rule_il=-1)
        with self.assertRaises(OasisException):
            step._check_alloc_rules()

    def test_ri_rule_above_max_raises(self):
        step = _make_step(kernel_alloc_rule_ri=KERNEL_ALLOC_FM_MAX + 1)
        with self.assertRaises(OasisException):
            step._check_alloc_rules()

    def test_ri_rule_below_zero_raises(self):
        step = _make_step(kernel_alloc_rule_ri=-1)
        with self.assertRaises(OasisException):
            step._check_alloc_rules()

    def test_il_rule_at_max_boundary_does_not_raise(self):
        step = _make_step(kernel_alloc_rule_il=KERNEL_ALLOC_FM_MAX)
        step._check_alloc_rules()

    def test_il_rule_at_zero_boundary_does_not_raise(self):
        step = _make_step(kernel_alloc_rule_il=0)
        step._check_alloc_rules()

    def test_ri_rule_at_max_boundary_does_not_raise(self):
        step = _make_step(kernel_alloc_rule_ri=KERNEL_ALLOC_FM_MAX)
        step._check_alloc_rules()


def _assert_output_matches(actual_path, expected_path):
    expected = pd.read_csv(expected_path)
    actual = pd.read_csv(actual_path)
    cols = [c for c in expected.columns if c != 'acc_idx']
    pd.testing.assert_frame_equal(
        expected[cols],
        actual[cols],
        check_exact=False,
        rtol=1e-4,
        check_dtype=False,
    )


def _run_exposure(output_file, **kwargs):
    return RunExposure(
        **BASE_PARAMS,
        output_file=output_file,
        **kwargs,
    ).run()


class TestRunExposureIntegration(ComputationChecker):
    def setUp(self):
        self.tmp = self.tmp_dir()

    def _output_file(self):
        return os.path.join(self.tmp.name, 'output.csv')

    def test_acc_loc_run_returns_il_true_ril_false(self):
        il, ril = _run_exposure(
            self._output_file(),
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
        )
        self.assertTrue(il)
        self.assertFalse(ril)

    def test_all_files_run_returns_il_true_ril_true(self):
        il, ril = _run_exposure(
            self._output_file(),
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
            oed_info_csv=RI_INFO,
            oed_scope_csv=RI_SCOPE,
        )
        self.assertTrue(il)
        self.assertTrue(ril)

    def test_invalid_location_file_raises(self):
        with self.assertRaises(Exception):
            _run_exposure(
                self._output_file(),
                oed_location_csv=LOCATION_INVALID,
                oed_accounts_csv=ACCOUNTS,
            )

    def test_acc_loc_output_matches_expected(self):
        out = self._output_file()
        _run_exposure(
            out,
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
        )
        _assert_output_matches(out, EXPECTED_ACC_LOC)

    def test_acc_loc_usd_output_matches_expected(self):
        out = self._output_file()
        _run_exposure(
            out,
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
            currency_conversion_json=CURRENCY_CONFIG,
            reporting_currency='USD',
        )
        _assert_output_matches(out, EXPECTED_ACC_LOC_USD)

    def test_all_files_output_matches_expected(self):
        out = self._output_file()
        _run_exposure(
            out,
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
            oed_info_csv=RI_INFO,
            oed_scope_csv=RI_SCOPE,
        )
        _assert_output_matches(out, EXPECTED_ALL)

    def test_src_dir_discovers_files_and_output_matches_expected(self):
        import shutil
        src_tmp = self.tmp_dir()
        for f in (LOCATION, ACCOUNTS, RI_INFO, RI_SCOPE):
            shutil.copy(f, src_tmp.name)
        out = self._output_file()
        _run_exposure(out, src_dir=src_tmp.name)
        _assert_output_matches(out, EXPECTED_ALL)

    def test_all_files_usd_output_matches_expected(self):
        out = self._output_file()
        _run_exposure(
            out,
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
            oed_info_csv=RI_INFO,
            oed_scope_csv=RI_SCOPE,
            currency_conversion_json=CURRENCY_CONFIG,
            reporting_currency='USD',
        )
        _assert_output_matches(out, EXPECTED_ALL_USD)

    def test_loss_factor_half_output_matches_expected(self):
        out = self._output_file()
        _run_exposure(
            out,
            oed_location_csv=LOCATION,
            oed_accounts_csv=ACCOUNTS,
            oed_info_csv=RI_INFO,
            oed_scope_csv=RI_SCOPE,
            loss_factor=[0.5],
        )
        _assert_output_matches(out, EXPECTED_LOSS_HALF)
