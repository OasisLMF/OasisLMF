"""
Tests for the inuring-priority-to-output-level mapping introduced in #1845.

Covers:
- get_ri_inuring_priority_output_levels() helper
- Mapping file written during input generation (logic extracted from files.py)
- generate_summaryxref_files() RI section: OED→RI-layer conversion, validation,
  output-directory file
- bash_params() conversion of analysis_settings['ri_inuring_priorities']
"""
import io
import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

from oasislmf.preparation.summaries import get_ri_inuring_priority_output_levels
from oasislmf.utils.exceptions import OasisException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ri_layers(spec):
    """Build an ri_layers dict (as returned by write_files_for_reinsurance) from a
    list of (inuring_priority, risk_level) pairs, assigning sequential layer indices."""
    return {
        i + 1: {'inuring_priority': ip, 'risk_level': rl, 'directory': f'/run/input/RI_{i + 1}'}
        for i, (ip, rl) in enumerate(spec)
    }


def _write_json(path, data):
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def _read_json(path):
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tests: get_ri_inuring_priority_output_levels
# ---------------------------------------------------------------------------

class TestGetRiInuringPriorityOutputLevels(TestCase):

    def test_loads_mapping_with_int_keys_and_values(self):
        with tempfile.TemporaryDirectory() as d:
            _write_json(os.path.join(d, 'ri_inuring_priority_output_levels.json'),
                        {'1': 2, '2': 3})
            result = get_ri_inuring_priority_output_levels(d)
        self.assertEqual(result, {1: 2, 2: 3})
        for k, v in result.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, int)

    def test_single_priority(self):
        with tempfile.TemporaryDirectory() as d:
            _write_json(os.path.join(d, 'ri_inuring_priority_output_levels.json'), {'1': 1})
            result = get_ri_inuring_priority_output_levels(d)
        self.assertEqual(result, {1: 1})

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                get_ri_inuring_priority_output_levels(d)


# ---------------------------------------------------------------------------
# Tests: mapping computation (logic mirrored from files.py)
# ---------------------------------------------------------------------------

class TestInuringPriorityMappingComputation(TestCase):
    """Verify the mapping formula: for each OED InuringPriority, the output level
    is the highest RI layer index that belongs to that priority."""

    def _compute_mapping(self, ri_layers):
        """Replicate the mapping logic from files.py."""
        mapping = {}
        for layer_idx, layer_info in ri_layers.items():
            ip = layer_info['inuring_priority']
            idx = int(layer_idx)
            if ip not in mapping or idx > mapping[ip]:
                mapping[ip] = idx
        return mapping

    def test_single_priority_single_risk_level(self):
        ri_layers = _make_ri_layers([(1, 'LOC')])
        self.assertEqual(self._compute_mapping(ri_layers), {1: 1})

    def test_single_priority_multiple_risk_levels(self):
        # InuringPriority 1 spans LOC (RI_1) and ACC (RI_2); output level should be 2
        ri_layers = _make_ri_layers([(1, 'LOC'), (1, 'ACC')])
        self.assertEqual(self._compute_mapping(ri_layers), {1: 2})

    def test_two_priorities_each_one_risk_level(self):
        ri_layers = _make_ri_layers([(1, 'LOC'), (2, 'SEL')])
        self.assertEqual(self._compute_mapping(ri_layers), {1: 1, 2: 2})

    def test_two_priorities_first_spans_multiple_risk_levels(self):
        # Priority 1: RI_1 (LOC), RI_2 (ACC)  →  output level 2
        # Priority 2: RI_3 (SEL)               →  output level 3
        ri_layers = _make_ri_layers([(1, 'LOC'), (1, 'ACC'), (2, 'SEL')])
        self.assertEqual(self._compute_mapping(ri_layers), {1: 2, 2: 3})

    def test_three_priorities(self):
        ri_layers = _make_ri_layers([
            (1, 'LOC'), (1, 'ACC'),   # priority 1 → output level 2
            (2, 'LOC'),                # priority 2 → output level 3
            (3, 'SEL'),                # priority 3 → output level 4
        ])
        self.assertEqual(self._compute_mapping(ri_layers), {1: 2, 2: 3, 3: 4})

    def test_mapping_file_written_to_input_dir(self):
        """files.py should write the mapping JSON to target_dir."""
        ri_layers = _make_ri_layers([(1, 'LOC'), (1, 'ACC'), (2, 'SEL')])
        expected = {1: 2, 2: 3}

        with tempfile.TemporaryDirectory() as d:
            mapping_fp = os.path.join(d, 'ri_inuring_priority_output_levels.json')
            with io.open(mapping_fp, 'w', encoding='utf-8') as f:
                f.write(json.dumps(self._compute_mapping(ri_layers), ensure_ascii=False, indent=4))

            loaded = _read_json(mapping_fp)
            # JSON round-trip: keys become strings
            self.assertEqual({int(k): v for k, v in loaded.items()}, expected)


# ---------------------------------------------------------------------------
# Tests: generate_summaryxref_files() RI section
# ---------------------------------------------------------------------------

class TestGenerateSummaryxrefRI(TestCase):
    """Unit-test the RI branch of generate_summaryxref_files().

    We mock the heavy I/O (df_to_ndarray, get_summary_xref_df, write_df_to_csv_file,
    write_df_to_parquet_file) and focus on the orchestration logic: which RI
    directories receive the summary xref file, error handling, and the mapping
    file written to the output directory.
    """

    def _setup_run_dir(self, tmpdir, ri_layers_spec, mapping=None):
        """Create a minimal run directory with the JSON files that the RI section reads."""
        input_dir = os.path.join(tmpdir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        # Build ri_layers.json
        ri_layers = {}
        for i, (ip, rl) in enumerate(ri_layers_spec):
            ri_dir = os.path.join(input_dir, f'RI_{i + 1}')
            os.makedirs(ri_dir, exist_ok=True)
            ri_layers[str(i + 1)] = {
                'inuring_priority': ip,
                'risk_level': rl,
                'directory': ri_dir,
            }
        _write_json(os.path.join(input_dir, 'ri_layers.json'), ri_layers)

        # Build ri_inuring_priority_output_levels.json (derived if not supplied)
        if mapping is None:
            computed = {}
            for idx_str, info in ri_layers.items():
                ip = info['inuring_priority']
                idx = int(idx_str)
                if ip not in computed or idx > computed[ip]:
                    computed[ip] = idx
            mapping = computed
        _write_json(os.path.join(input_dir, 'ri_inuring_priority_output_levels.json'), mapping)

        return ri_layers, mapping

    def _analysis_settings(self, ri_inuring_priorities=None):
        settings = {
            'ri_output': True,
            'ri_summaries': [{'id': 1}],
        }
        if ri_inuring_priorities is not None:
            settings['ri_inuring_priorities'] = ri_inuring_priorities
        return settings

    def _call_summaryxref(self, tmpdir, analysis_settings, il=True, ri=True, rl=False):
        """Call generate_summaryxref_files() with heavy I/O mocked out."""
        from oasislmf.preparation.summaries import generate_summaryxref_files

        # Minimal DataFrames so the function gets past the IL map loading
        import pandas as pd
        dummy_df = pd.DataFrame()

        with patch('oasislmf.preparation.summaries.get_summary_xref_df') as mock_xref, \
                patch('oasislmf.preparation.summaries.df_to_ndarray') as mock_ndarray, \
                patch('oasislmf.preparation.summaries.write_df_to_csv_file') as mock_csv, \
                patch('oasislmf.preparation.summaries.write_df_to_parquet_file') as mock_parquet, \
                patch('oasislmf.preparation.summaries.get_dataframe') as mock_get_df, \
                patch('oasislmf.preparation.summaries.os.path.exists', return_value=True):

            # get_summary_xref_df returns (xref_df, summary_desc)
            xref_array = MagicMock()
            mock_xref.return_value = (MagicMock(), {})
            mock_ndarray.return_value = xref_array
            mock_get_df.return_value = MagicMock(
                __getitem__=lambda s, x: MagicMock(),
                columns=MagicMock(return_value=[]),
            )

            generate_summaryxref_files(
                location_df=dummy_df,
                account_df=dummy_df,
                model_run_fp=tmpdir,
                analysis_settings=analysis_settings,
                il=il,
                ri=ri,
                rl=rl,
                intermediary_csv=False,
            )

            return mock_ndarray, xref_array

    def test_final_priority_always_written(self):
        """Without ri_inuring_priorities, only the final OED priority's output level is written."""
        with tempfile.TemporaryDirectory() as d:
            # Two priorities: priority 1 → RI_1, priority 2 → RI_2
            ri_layers, mapping = self._setup_run_dir(d, [(1, 'LOC'), (2, 'SEL')])
            settings = self._analysis_settings()  # no ri_inuring_priorities

            mock_ndarray, xref_array = self._call_summaryxref(d, settings)

            # The xref should be written to RI_2 (final priority output level)
            written_paths = [str(c.args[0]) for c in xref_array.tofile.call_args_list]
            self.assertTrue(any('RI_2' in p for p in written_paths))
            # RI_1 should NOT be written
            self.assertFalse(any('RI_1' in p for p in written_paths))

    def test_intermediate_oed_priority_written_to_correct_layer(self):
        """When ri_inuring_priorities=[1] and priority 1 spans two risk levels (RI_1, RI_2),
        the xref should be written to RI_2 (last layer of priority 1), not RI_1."""
        with tempfile.TemporaryDirectory() as d:
            # Priority 1: RI_1 (LOC) + RI_2 (ACC) → output level 2
            # Priority 2: RI_3 (SEL) → output level 3
            ri_layers, mapping = self._setup_run_dir(
                d, [(1, 'LOC'), (1, 'ACC'), (2, 'SEL')]
            )
            settings = self._analysis_settings(ri_inuring_priorities=[1])

            mock_ndarray, xref_array = self._call_summaryxref(d, settings)

            written_paths = [str(c.args[0]) for c in xref_array.tofile.call_args_list]
            # RI_2 (output level for OED priority 1) and RI_3 (final) should be written
            self.assertTrue(any('RI_2' in p for p in written_paths), written_paths)
            self.assertTrue(any('RI_3' in p for p in written_paths), written_paths)
            # RI_1 should never receive the xref file
            self.assertFalse(any('RI_1' in p for p in written_paths), written_paths)

    def test_out_of_scope_oed_priority_raises(self):
        """Requesting an OED priority that does not exist in the mapping raises OasisException."""
        with tempfile.TemporaryDirectory() as d:
            self._setup_run_dir(d, [(1, 'LOC'), (2, 'SEL')])
            settings = self._analysis_settings(ri_inuring_priorities=[99])

            with self.assertRaises(OasisException):
                self._call_summaryxref(d, settings)

    def test_mapping_file_written_to_output_dir(self):
        """generate_summaryxref_files() should copy the mapping JSON to output/."""
        with tempfile.TemporaryDirectory() as d:
            self._setup_run_dir(d, [(1, 'LOC'), (2, 'SEL')])
            settings = self._analysis_settings()

            self._call_summaryxref(d, settings)

            output_fp = os.path.join(d, 'output', 'ri_inuring_priority_output_levels.json')
            self.assertTrue(os.path.exists(output_fp), f"Mapping file not found at {output_fp}")
            loaded = _read_json(output_fp)
            self.assertEqual({int(k): v for k, v in loaded.items()}, {1: 1, 2: 2})

    def test_rl_mode_writes_all_ri_layers(self):
        """When rl_summaries is requested, the xref is written to every RI layer directory."""
        with tempfile.TemporaryDirectory() as d:
            ri_layers, mapping = self._setup_run_dir(
                d, [(1, 'LOC'), (1, 'ACC'), (2, 'SEL')]
            )
            settings = {
                'ri_output': True,
                'ri_summaries': [{'id': 1}],
                'rl_output': True,
                'rl_summaries': [{'id': 1}],
            }

            mock_ndarray, xref_array = self._call_summaryxref(d, settings, rl=True)

            written_paths = [str(c.args[0]) for c in xref_array.tofile.call_args_list]
            # All three RI directories must receive a file
            for ri_dir in ('RI_1', 'RI_2', 'RI_3'):
                self.assertTrue(any(ri_dir in p for p in written_paths), written_paths)


# ---------------------------------------------------------------------------
# Tests: bash_params() OED priority conversion
# ---------------------------------------------------------------------------

class TestBashParamsInuringPriorityConversion(TestCase):
    """Verify bash_params() converts ri_inuring_priorities from OED values to RI layer indices
    when the mapping file exists in model_run_dir/input/.

    Conversion must be reflected in bash_params['analysis_settings'] but must NOT mutate
    the caller's original dict — this makes repeated calls with the same settings object safe.
    """

    def _make_analysis_settings(self, ri_inuring_priorities):
        return {
            'ri_output': True,
            'ri_summaries': [{'id': 1}],
            'ri_inuring_priorities': list(ri_inuring_priorities),
        }

    def _call_bash_params(self, model_run_dir, analysis_settings):
        from oasislmf.execution.bash import bash_params
        with tempfile.NamedTemporaryFile(suffix='.sh', delete=False) as f:
            script_fp = f.name
        try:
            return bash_params(
                analysis_settings=analysis_settings,
                filename=script_fp,
                num_reinsurance_iterations=3,
                model_run_dir=model_run_dir,
                fifo_tmp_dir=False,
            )
        finally:
            if os.path.exists(script_fp):
                os.remove(script_fp)

    def test_oed_priorities_converted_in_bash_params(self):
        """OED priority 1 → layer 2 in bash_params['analysis_settings']."""
        mapping = {1: 2, 2: 3}
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'input'))
            _write_json(os.path.join(d, 'input', 'ri_inuring_priority_output_levels.json'), mapping)

            settings = self._make_analysis_settings([1])
            result = self._call_bash_params(d, settings)
            self.assertEqual(result['analysis_settings']['ri_inuring_priorities'], [2])

    def test_caller_dict_not_mutated(self):
        """The original analysis_settings dict passed by the caller is never modified."""
        mapping = {1: 2, 2: 3}
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'input'))
            _write_json(os.path.join(d, 'input', 'ri_inuring_priority_output_levels.json'), mapping)

            settings = self._make_analysis_settings([1])
            original_priorities = list(settings['ri_inuring_priorities'])
            self._call_bash_params(d, settings)
            # caller's dict must be unchanged
            self.assertEqual(settings['ri_inuring_priorities'], original_priorities)

    def test_repeated_calls_are_idempotent(self):
        """Calling bash_params() twice with the same settings dict produces the same result both times."""
        mapping = {1: 2, 2: 3}
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'input'))
            _write_json(os.path.join(d, 'input', 'ri_inuring_priority_output_levels.json'), mapping)

            settings = self._make_analysis_settings([1])
            result1 = self._call_bash_params(d, settings)
            result2 = self._call_bash_params(d, settings)
            self.assertEqual(
                result1['analysis_settings']['ri_inuring_priorities'],
                result2['analysis_settings']['ri_inuring_priorities'],
            )
            self.assertEqual(result1['analysis_settings']['ri_inuring_priorities'], [2])

    def test_no_conversion_without_model_run_dir(self):
        """When model_run_dir is empty, ri_inuring_priorities passes through unchanged."""
        settings = self._make_analysis_settings([1])
        result = self._call_bash_params('', settings)
        self.assertEqual(result['analysis_settings']['ri_inuring_priorities'], [1])

    def test_no_conversion_without_mapping_file(self):
        """When the mapping file does not exist, ri_inuring_priorities passes through unchanged."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'input'))
            settings = self._make_analysis_settings([1])
            result = self._call_bash_params(d, settings)
            self.assertEqual(result['analysis_settings']['ri_inuring_priorities'], [1])

    def test_multiple_oed_priorities_converted(self):
        """Multiple OED priorities are each converted to their respective RI layer indices."""
        mapping = {1: 2, 2: 3, 3: 5}
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'input'))
            _write_json(os.path.join(d, 'input', 'ri_inuring_priority_output_levels.json'), mapping)

            settings = self._make_analysis_settings([1, 2])
            result = self._call_bash_params(d, settings)
            self.assertEqual(sorted(result['analysis_settings']['ri_inuring_priorities']), [2, 3])

    def test_unknown_oed_priority_silently_dropped(self):
        """An OED priority value absent from the mapping is silently dropped."""
        mapping = {1: 2, 2: 3}
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'input'))
            _write_json(os.path.join(d, 'input', 'ri_inuring_priority_output_levels.json'), mapping)

            settings = self._make_analysis_settings([1, 99])
            result = self._call_bash_params(d, settings)
            # 99 is not in the mapping so it is dropped; 1 → 2
            self.assertEqual(result['analysis_settings']['ri_inuring_priorities'], [2])
