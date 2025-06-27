import glob
import io
import os
import shutil
import tarfile

from itertools import chain, islice
from tempfile import TemporaryDirectory
from copy import copy, deepcopy
from tempfile import NamedTemporaryFile
from unittest import TestCase

from hypothesis import (
    given,
    HealthCheck,
    settings
)
from hypothesis.strategies import sampled_from, lists
from mock import patch, Mock
from pathlib import Path

from oasislmf.execution.files import GUL_INPUT_FILES, IL_INPUT_FILES, TAR_FILE, INPUT_FILES
from oasislmf.execution.bin import (
    check_binary_tar_file,
    check_conversion_tools,
    check_inputs_directory,
    cleanup_bin_directory,
    create_binary_tar_file,
    csv_to_bin,
    prepare_run_directory,
    prepare_run_inputs,
    set_footprint_set,
    set_vulnerability_set
)

from oasis_data_manager.filestore.backends.local import LocalStorage
from oasislmf.utils.exceptions import OasisException
from oasislmf.pytools.getmodel.vulnerability import vulnerability_dataset, parquetvulnerability_meta_filename

from tests.data import (
    standard_input_files,
    il_input_files,
    tar_file_targets,
    ECHO_CONVERSION_INPUT_FILES
)


class CsvToBin(TestCase):
    def test_directory_only_contains_excluded_files___tar_is_empty(self):
        with TemporaryDirectory() as csv_dir, TemporaryDirectory() as bin_dir:
            with io.open(os.path.join(csv_dir, 'another_file'), 'w', encoding='utf-8') as f:
                f.write('file data')

            csv_to_bin(csv_dir, bin_dir)

            self.assertEqual(0, len(glob.glob(os.path.join(csv_dir, '*.bin'))))

    @given(standard_input_files(min_size=1), il_input_files(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_contains_il_and_standard_files_but_il_is_false___il_files_are_excluded(self, standard, il):
        with patch('oasislmf.model_execution.bin.INPUT_FILES', ECHO_CONVERSION_INPUT_FILES), TemporaryDirectory() as csv_dir, TemporaryDirectory() as bin_dir:
            for target in chain(standard, il):
                with io.open(os.path.join(csv_dir, target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")

            csv_to_bin(csv_dir, bin_dir, il=False)

            self.assertEqual(len(standard), len(glob.glob(os.path.join(bin_dir, '*.bin'))))
            for filename in (f + '.bin' for f in standard):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, filename)))

    @given(standard_input_files(min_size=1), il_input_files(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_contains_il_and_standard_files_but_il_is_true___all_files_are_included(self, standard, il):
        with patch('oasislmf.model_execution.bin.INPUT_FILES', ECHO_CONVERSION_INPUT_FILES), TemporaryDirectory() as csv_dir, TemporaryDirectory() as bin_dir:
            for target in chain(standard, il):
                with io.open(os.path.join(csv_dir, target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")

            csv_to_bin(csv_dir, bin_dir, il=True)

            self.assertEqual(len(standard) + len(il), len(glob.glob(os.path.join(bin_dir, '*.bin'))))
            for filename in (f + '.bin' for f in chain(standard, il)):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, filename)))

    def test_subprocess_raises___oasis_exception_is_raised(self):
        with TemporaryDirectory() as csv_dir, TemporaryDirectory() as bin_dir:
            Path(os.path.join(csv_dir, 'events.csv')).touch()

            with patch('oasislmf.model_execution.bin.csvtobin', Mock(side_effect=Exception(1, ''))):
                with self.assertRaises(OasisException):
                    csv_to_bin(csv_dir, bin_dir, il=True)

    @given(standard_input_files(min_size=1), il_input_files(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_single_ri_folder(self, standard, il):
        with patch('oasislmf.model_execution.bin.INPUT_FILES', ECHO_CONVERSION_INPUT_FILES), TemporaryDirectory() as csv_dir, TemporaryDirectory() as bin_dir:
            files = standard + il

            for target in files:
                with io.open(os.path.join(csv_dir, target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")
            os.mkdir(os.path.join(csv_dir, "RI_1"))
            for target in files:
                with io.open(os.path.join(csv_dir, "RI_1", target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")

            csv_to_bin(csv_dir, bin_dir, il=True, ri=True)

            self.assertEqual(len(files), len(glob.glob(os.path.join(bin_dir, '*.bin'))))
            for filename in (f + '.bin' for f in files):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, filename)))

            self.assertEqual(len(files), len(glob.glob(os.path.join(bin_dir, 'RI_1{}*.bin'.format(os.sep)))))
            for filename in (f + '.bin' for f in files):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, 'RI_1', filename)))
            print("ok")

    @given(standard_input_files(min_size=1), il_input_files(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_multiple_ri_folders(self, standard, il):
        with patch('oasislmf.model_execution.bin.INPUT_FILES', ECHO_CONVERSION_INPUT_FILES), TemporaryDirectory() as csv_dir, TemporaryDirectory() as bin_dir:
            files = standard + il

            for target in files:
                with io.open(os.path.join(csv_dir, target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")
            os.mkdir(os.path.join(csv_dir, "RI_1"))
            for target in files:
                with io.open(os.path.join(csv_dir, "RI_1", target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")
            os.mkdir(os.path.join(csv_dir, "RI_2"))
            for target in files:
                with io.open(os.path.join(csv_dir, "RI_2", target + '.csv'), 'w', encoding='utf-8') as f:
                    f.write("")

            csv_to_bin(csv_dir, bin_dir, il=True, ri=True)

            self.assertEqual(len(files), len(glob.glob(os.path.join(bin_dir, '*.bin'))))
            for filename in (f + '.bin' for f in files):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, filename)))

            self.assertEqual(len(files), len(glob.glob(os.path.join(bin_dir, 'RI_1{}*.bin'.format(os.sep)))))
            for filename in (f + '.bin' for f in files):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, 'RI_1', filename)))

            self.assertEqual(len(files), len(glob.glob(os.path.join(bin_dir, 'RI_2{}*.bin'.format(os.sep)))))
            for filename in (f + '.bin' for f in files):
                self.assertTrue(os.path.exists(os.path.join(bin_dir, 'RI_2', filename)))


class CreateBinaryTarFile(TestCase):
    def test_directory_only_contains_excluded_files___tar_is_empty(self):
        with TemporaryDirectory() as d:
            with io.open(os.path.join(d, 'another_file'), 'w', encoding='utf-8') as f:
                f.write('file data')

            create_binary_tar_file(d)

            with tarfile.open(os.path.join(d, TAR_FILE), 'r:gz', encoding='utf-8') as tar:
                self.assertEqual(0, len(tar.getnames()))

    @given(tar_file_targets(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_directory_contains_some_target_files___target_files_are_included(self, targets):
        with TemporaryDirectory() as d:
            for target in targets:
                with io.open(os.path.join(d, target), 'w', encoding='utf-8') as f:
                    f.write(target)

            create_binary_tar_file(d)

            with tarfile.open(os.path.join(d, TAR_FILE), 'r:gz', encoding='utf-8') as tar:
                self.assertEqual(len(targets), len(tar.getnames()))
                self.assertEqual(set(targets), set(tar.getnames()))

    @given(tar_file_targets(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_with_single_reinsurance_subfolder(self, targets):
        with TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, 'RI_1'))
            for target in targets:
                with io.open(os.path.join(d, target), 'w', encoding='utf-8') as f:
                    f.write(target)
                with io.open(os.path.join(d, 'RI_1', target), 'w', encoding='utf-8') as f:
                    f.write(target)

            create_binary_tar_file(d)

            all_targets = copy(targets)
            for t in targets:
                # tarfile converts os-specific separators to forward slashes
                all_targets.append("RI_1/{}".format(t))

            with tarfile.open(os.path.join(d, TAR_FILE), 'r:gz', encoding='utf-8') as tar:
                self.assertEqual(len(all_targets), len(tar.getnames()))
                self.assertEqual(set(all_targets), set(tar.getnames()))

    @given(tar_file_targets(min_size=1))
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_with_multiple_reinsurance_subfolders(self, targets):
        with TemporaryDirectory() as d:
            os.mkdir(os.path.join(d, 'RI_1'))
            os.mkdir(os.path.join(d, 'RI_2'))

            for target in targets:
                with io.open(os.path.join(d, target), 'w', encoding='utf-8') as f:
                    f.write(target)
                with io.open(os.path.join(d, 'RI_1', target), 'w', encoding='utf-8') as f:
                    f.write(target)
                with io.open(os.path.join(d, 'RI_2', target), 'w', encoding='utf-8') as f:
                    f.write(target)

            create_binary_tar_file(d)

            all_targets = copy(targets)
            for t in targets:
                # tarfile converts os-specific separators to forward slashes
                all_targets.append("RI_1/{}".format(t))
                all_targets.append("RI_2/{}".format(t))

            with tarfile.open(os.path.join(d, TAR_FILE), 'r:gz', encoding='utf-8') as tar:
                self.assertEqual(len(all_targets), len(tar.getnames()))
                self.assertEqual(set(all_targets), set(tar.getnames()))


class CheckConversionTools(TestCase):
    def test_il_is_false_il_tools_are_missing___result_is_true(self):
        existing_conversions = deepcopy(INPUT_FILES)
        for value in existing_conversions.values():
            if value['type'] == 'il':
                value['conversion_tool'] = 'missing_executable'
            else:
                value['conversion_tool'] = 'python'

        with patch('oasislmf.model_execution.bin.INPUT_FILES', existing_conversions):
            self.assertTrue(check_conversion_tools())

    def test_il_is_false_il_tools_are_present_but_non_il_are_missing___errors_is_raised(self):
        existing_conversions = deepcopy(INPUT_FILES)
        for value in existing_conversions.values():
            if value['type'] == 'il':
                value['conversion_tool'] = 'pytohn'
            else:
                value['conversion_tool'] = 'missing_executable'

        with patch('oasislmf.model_execution.bin.INPUT_FILES', existing_conversions):
            with self.assertRaises(OasisException):
                check_conversion_tools()

    def test_il_is_true_il_tools_are_missing___error_is_raised(self):
        existing_conversions = deepcopy(INPUT_FILES)
        for value in existing_conversions.values():
            if value['type'] == 'il':
                value['conversion_tool'] = 'missing_executable'
            else:
                value['conversion_tool'] = 'python'

        with patch('oasislmf.model_execution.bin.INPUT_FILES', existing_conversions):
            with self.assertRaises(OasisException):
                check_conversion_tools(il=True)

    def test_il_is_true_non_il_are_missing___errror_is_raised(self):
        existing_conversions = deepcopy(INPUT_FILES)
        for value in existing_conversions.values():
            if value['type'] == 'il':
                value['conversion_tool'] = 'pytohn'
            else:
                value['conversion_tool'] = 'missing_executable'

        with patch('oasislmf.model_execution.bin.INPUT_FILES', existing_conversions):
            with self.assertRaises(OasisException):
                check_conversion_tools(il=True)

    def test_il_is_true_conversion_tools_all_exist___result_is_true(self):
        existing_conversions = deepcopy(INPUT_FILES)
        for value in existing_conversions.values():
            value['conversion_tool'] = 'python'

        with patch('oasislmf.model_execution.bin.INPUT_FILES', existing_conversions):
            self.assertTrue(check_conversion_tools(il=True))


class CheckInputsDirectory(TestCase):
    def test_tar_file_already_exists___exception_is_raised(self):
        with TemporaryDirectory() as d:
            Path(os.path.join(d, TAR_FILE)).touch()
            with self.assertRaises(OasisException):
                check_inputs_directory(d, False)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(il_input_files())
    def test_il_is_false_non_il_input_files_are_missing__exception_is_raised(self, il_files):
        with TemporaryDirectory() as d:
            for p in il_files:
                Path(os.path.join(d, p + '.csv')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, False)

    def test_do_is_is_false_non_il_input_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            for input_file in GUL_INPUT_FILES.values():
                Path(os.path.join(d, input_file['name'] + '.csv')).touch()

            try:
                check_inputs_directory(d, False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_il_is_true_all_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(OasisException):
                check_inputs_directory(d, True)

    def test_il_is_true_gul_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in IL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, True)

    def test_il_is_true_il_input_files_are_missing__exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in GUL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, True)

    def test_il_is_true_all_input_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values()):
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            try:
                check_inputs_directory(d, True)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_il_is_false_il_bin_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values()):
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            for p in IL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.bin')).touch()

            try:
                check_inputs_directory(d, False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_il_is_false_gul_bin_files_are_present___exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values()):
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            for p in GUL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.bin')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, False)

    def test_il_is_true_gul_bin_files_are_present___exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values()):
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            for p in GUL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.bin')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, True)

    def test_il_is_true_il_bin_files_are_present___exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values()):
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            for p in IL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.bin')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, True)

    def test_il_is_true_no_bin_files_are_present___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values()):
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            for p in IL_INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.bin')).touch()

            try:
                check_inputs_directory(d, False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_il_is_true_bin_files_are_present_but_check_bin_files_are_true___no_exception_is_raised(self):
        with TemporaryDirectory() as d:
            for p in INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            for p in INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.bin')).touch()

            try:
                check_inputs_directory(d, il=True, check_binaries=False)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_check_gul_and_il_and_single_ri_directory_structure(self):
        with TemporaryDirectory() as d:
            for p in INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()
            os.mkdir(os.path.join(d, "RI_1"))
            for p in INPUT_FILES.values():
                f = os.path.join(d, "RI_1", p['name'] + '.csv')
                Path(f).touch()
            try:
                check_inputs_directory(d, il=True, ri=True, check_binaries=True)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_check_gul_and_il_and_single_ri_directory_structure_binaries_fail(self):
        with TemporaryDirectory() as d:
            for p in INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()
                Path(os.path.join(d, p['name'] + '.bin')).touch()
            os.mkdir(os.path.join(d, "RI_1"))
            for p in INPUT_FILES.values():
                Path(os.path.join(d, "RI_1", p['name'] + '.csv')).touch()
                Path(os.path.join(d, "RI_1", p['name'] + '.bin')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, il=True, ri=True, check_binaries=True)

    def test_check_gul_and_il_and_multiple_ri_directories(self):
        with TemporaryDirectory() as d:
            for p in INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()

            os.mkdir(os.path.join(d, "RI_1"))
            for p in INPUT_FILES.values():
                Path(os.path.join(d, "RI_1", p['name'] + '.csv')).touch()

            os.mkdir(os.path.join(d, "RI_2"))
            for p in INPUT_FILES.values():
                Path(os.path.join(d, "RI_2", p['name'] + '.csv')).touch()

            try:
                check_inputs_directory(d, il=True, ri=True, check_binaries=True)
            except Exception as e:
                self.fail('Exception was raised {}: {}'.format(type(e), e))

    def test_check_gul_and_il_and_multiple_ri_directories_binaries_fail(self):
        with TemporaryDirectory() as d:
            for p in INPUT_FILES.values():
                Path(os.path.join(d, p['name'] + '.csv')).touch()
                Path(os.path.join(d, p['name'] + '.bin')).touch()
            os.mkdir(os.path.join(d, "RI_1"))
            for p in INPUT_FILES.values():
                Path(f=os.path.join(d, "RI_1", p['name'] + '.csv')).touch()
                Path(f=os.path.join(d, "RI_1", p['name'] + '.bin')).touch()
            os.mkdir(os.path.join(d, "RI_2"))
            for p in INPUT_FILES.values():
                Path(os.path.join(d, "RI_2", p['name'] + '.bin')).touch()
                Path(os.path.join(d, "RI_2", p['name'] + '.bin')).touch()

            with self.assertRaises(OasisException):
                check_inputs_directory(d, il=True, ri=True, check_binaries=True)


class PrepareRunDirectory(TestCase):
    def test_directory_is_empty___child_directories_are_created(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp)

            self.assertTrue(os.path.exists(os.path.join(run_dir, 'input')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'output')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'static')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'work')))

    def test_directory_has_some_existing_directories___other_child_directories_are_created(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            os.mkdir(os.path.join(run_dir, 'fifo'))
            os.mkdir(os.path.join(run_dir, 'input'))

            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp)

            self.assertTrue(os.path.exists(os.path.join(run_dir, 'input')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'output')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'static')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'work')))

    def test_input_directory_is_supplied___input_files_are_copied_to_input_csv(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            Path(os.path.join(oasis_src_fp, 'a_file.csv')).touch()

            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp)

            self.assertTrue(os.path.exists(os.path.join(run_dir, 'input', 'a_file.csv')))

    def test_analysis_settings_file_is_supplied___file_is_copied_into_run_dir(self):
        analysis_settings_fp = NamedTemporaryFile('w', delete=False, prefix='bin')
        try:
            with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
                analysis_settings_fp.write('{"analysis_settings": "analysis_settings"}')
                analysis_settings_fp.close()

                prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp.name)

                with io.open(os.path.join(run_dir, 'analysis_settings.json'), encoding='utf-8') as expected_analysis_settings:
                    self.assertEqual('{"analysis_settings": "analysis_settings"}', expected_analysis_settings.read())
        finally:
            os.remove(analysis_settings_fp.name)

    def test_model_data_src_is_supplied___symlink_to_output_dir_static_is_created(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            Path(os.path.join(model_data_fp, 'linked_file')).touch()

            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp)

            self.assertTrue(os.path.exists(os.path.join(run_dir, 'static', 'linked_file')))

    def test_model_data_src_is_supplied_sym_link_raises_exception(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            Path(os.path.join(model_data_fp, 'linked_file')).touch()

            with self.assertRaises(OasisException):
                with patch('os.symlink', Mock(side_effect=OSError())):
                    prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp)

    def test_inputs_archive_is_supplied_no_ri___archive_is_extracted_into_inputs(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            tar_path = os.path.join(oasis_src_fp, 'archive.tar')

            with tarfile.open(tar_path, 'w', encoding='utf-8') as tar:
                archived_file_path = Path(oasis_src_fp, 'archived_file')
                archived_file_path.touch()
                tar.add(str(archived_file_path), arcname='archived_file')

            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp, inputs_archive=tar_path)

            self.assertTrue(Path(run_dir, 'input', 'archived_file').exists())

    def test_inputs_archive_is_supplied_with_ri___archive_is_extracted_into_inputs(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            tar_path = os.path.join(oasis_src_fp, 'archive.tar')

            with tarfile.open(tar_path, 'w', encoding='utf-8') as tar:
                archived_file_path = Path(oasis_src_fp, 'archived_file')
                archived_file_path.touch()
                tar.add(str(archived_file_path), arcname='archived_file')
            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp, inputs_archive=tar_path, ri=True)

            self.assertTrue(Path(run_dir, 'input', 'archived_file').exists())

    def test_inputs_archive_with_subfolder_is_supplied_no_ri___archive_is_extracted_into_inputs(self):
        with TemporaryDirectory() as run_dir, TemporaryDirectory() as oasis_src_fp, TemporaryDirectory() as model_data_fp:
            analysis_settings_fp = os.path.join(oasis_src_fp, "settings.json")
            Path(analysis_settings_fp).touch()
            tar_path = os.path.join(oasis_src_fp, 'archive.tar')

            with tarfile.open(tar_path, 'w') as tar:
                archived_file_path = Path(oasis_src_fp, 'archived_file')
                archived_file_path.touch()
                tar.add(str(archived_file_path), arcname='archived_file')
                os.remove(str(archived_file_path))
                os.mkdir(os.path.join(oasis_src_fp, "sub1"))
                archived_file_path = Path(oasis_src_fp, "sub1", 'archived_file')
                archived_file_path.touch()
                tar.add(str(archived_file_path), arcname='sub1{}archived_file'.format(os.sep))
                shutil.rmtree(os.path.join(oasis_src_fp, "sub1"))

            prepare_run_directory(run_dir, oasis_src_fp, model_data_fp, analysis_settings_fp, inputs_archive=tar_path)

            self.assertTrue(Path(run_dir, 'input', 'archived_file').exists())
            self.assertTrue(Path(run_dir, 'input', 'sub1', 'archived_file').exists())


class PrepareRunInputs(TestCase):
    def make_fake_bins(self, d):
        os.mkdir(os.path.join(d, 'input'))
        os.mkdir(os.path.join(d, 'static'))
        Path(os.path.join(d, 'static', 'events.bin')).touch()
        Path(os.path.join(d, 'static', 'returnperiods.bin')).touch()
        Path(os.path.join(d, 'static', 'occurrence.bin')).touch()
        Path(os.path.join(d, 'static', 'periods.bin')).touch()

    def test_prepare_input_bin_raises___oasis_exception_is_raised(self):
        with patch('oasislmf.model_execution.bin._prepare_input_bin', Mock(side_effect=OSError('os error'))):
            with self.assertRaises(OasisException):
                prepare_run_inputs({}, 'some_dir', Mock())

    def test_events_bin_already_exists___existing_bin_is_uncahnged(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'input', 'events.bin'), 'w', encoding='utf-8') as events_file:
                events_file.write('events bin')
                events_file.flush()

                prepare_run_inputs({}, d, model_storage)

            with io.open(os.path.join(d, 'input', 'events.bin'), 'r', encoding='utf-8') as new_events_file:
                self.assertEqual('events bin', new_events_file.read())

    def test_events_bin_doesnt_not_exist_event_set_isnt_specified___bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'events.bin'), 'w', encoding='utf-8') as events_file:
                events_file.write('events bin')
                events_file.flush()

                prepare_run_inputs({}, d, model_storage)

            with io.open(os.path.join(d, 'input', 'events.bin'), 'r', encoding='utf-8') as new_events_file:
                self.assertEqual('events bin', new_events_file.read())

    def test_events_bin_doesnt_not_exist_event_set_is_specified___event_set_specific_bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'events_from_set.bin'), 'w', encoding='utf-8') as events_file:
                events_file.write('events from set bin')
                events_file.flush()

                prepare_run_inputs({'model_settings': {'event_set': 'from set'}}, d, model_storage)

            with io.open(os.path.join(d, 'input', 'events.bin'), 'r', encoding='utf-8') as new_events_file:
                self.assertEqual('events from set bin', new_events_file.read())

    def test_no_events_bin_exists___oasis_exception_is_raised(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)
            os.remove(os.path.join(d, 'static', 'events.bin'))

            with self.assertRaises(OasisException):
                prepare_run_inputs({}, d, model_storage)

    def test_returnperiods_bin_already_exists___existing_bin_is_uncahnged(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'input', 'returnperiods.bin'), 'w', encoding='utf-8') as returnperiods_file:
                returnperiods_file.write('returnperiods bin')
                returnperiods_file.flush()

                prepare_run_inputs({}, d, model_storage)

            with io.open(os.path.join(d, 'input', 'returnperiods.bin'), 'r', encoding='utf-8') as new_returnperiods_file:
                self.assertEqual('returnperiods bin', new_returnperiods_file.read())

    def test_returnperiods_bin_doesnt_not_exist_event_set_isnt_specified___bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'returnperiods.bin'), 'w', encoding='utf-8') as returnperiods_file:
                returnperiods_file.write('returnperiods bin')
                returnperiods_file.flush()

                settings = {"gul_summaries": [{
                    "lec_output": True,
                    "leccalc": {"full_uncertainty_aep": True},
                }]}
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'returnperiods.bin'), 'r', encoding='utf-8') as new_returnperiods_file:
                self.assertEqual('returnperiods bin', new_returnperiods_file.read())

    def test_ord_returnperiods_bin_doesnt_not_exist_event_set_isnt_specified___bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'returnperiods.bin'), 'w', encoding='utf-8') as returnperiods_file:
                returnperiods_file.write('returnperiods bin')
                returnperiods_file.flush()

                settings = {"gul_summaries": [{
                    "ord_output": {"psept_oep": True},
                }]}
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'returnperiods.bin'), 'r', encoding='utf-8') as new_returnperiods_file:
                self.assertEqual('returnperiods bin', new_returnperiods_file.read())

    def test_no_returnperiods_bin_exists___oasis_exception_is_raised(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)
            os.remove(os.path.join(d, 'static', 'returnperiods.bin'))

            with self.assertRaises(OasisException):
                settings = {"gul_summaries": [{
                    "lec_output": True,
                    "leccalc": {"full_uncertainty_aep": True},
                }]}
                prepare_run_inputs(settings, d, model_storage)

    def test_ord_no_returnperiods_bin_exists___oasis_exception_is_raised(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)
            os.remove(os.path.join(d, 'static', 'returnperiods.bin'))

            with self.assertRaises(OasisException):
                settings = {"gul_summaries": [{
                    "ord_output": {"ept_full_uncertainty_aep": True},
                }]}
                prepare_run_inputs(settings, d, model_storage)

    def test_occurrence_bin_already_exists___existing_bin_is_uncahnged(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'input', 'occurrence.bin'), 'w', encoding='utf-8') as occurrence_file:
                occurrence_file.write('occurrence bin')
                occurrence_file.flush()

                prepare_run_inputs({}, d, model_storage)

            with io.open(os.path.join(d, 'input', 'occurrence.bin'), 'r', encoding='utf-8') as new_occurrence_file:
                self.assertEqual('occurrence bin', new_occurrence_file.read())

    def test_occurrence_bin_doesnt_not_exist_event_set_isnt_specified___bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'occurrence.bin'), 'w', encoding='utf-8') as occurrence_file:
                occurrence_file.write('occurrence bin')
                occurrence_file.flush()

                settings = {"gul_summaries": [{
                    "lec_output": True,
                    "leccalc": {"full_uncertainty_aep": True},
                }]}
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'occurrence.bin'), 'r', encoding='utf-8') as new_occurrence_file:
                self.assertEqual('occurrence bin', new_occurrence_file.read())

    def test_ord_occurrence_bin_doesnt_not_exist_event_set_isnt_specified___bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'occurrence.bin'), 'w', encoding='utf-8') as occurrence_file:
                occurrence_file.write('occurrence bin')
                occurrence_file.flush()

                settings = {"gul_summaries": [{
                    "ord_output": {"ept_per_sample_mean_aep": True},
                }]}
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'occurrence.bin'), 'r', encoding='utf-8') as new_occurrence_file:
                self.assertEqual('occurrence bin', new_occurrence_file.read())

    def test_occurrence_bin_doesnt_not_exist_event_set_is_specified___event_occurrence_id_specific_bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'occurrence_occurrence_id.bin'), 'w', encoding='utf-8') as occurrence_file:
                occurrence_file.write('occurrence occurrence id bin')
                occurrence_file.flush()

                settings = {
                    "gul_summaries": [{
                        "lec_output": True,
                        "leccalc": {"full_uncertainty_aep": True},
                    }],
                    'model_settings': {'event_occurrence_id': 'occurrence id'}
                }
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'occurrence.bin'), 'r', encoding='utf-8') as new_occurrence_file:
                self.assertEqual('occurrence occurrence id bin', new_occurrence_file.read())

    def test_ord_occurrence_bin_doesnt_not_exist_event_set_is_specified___event_occurrence_id_specific_bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'occurrence_occurrence_id.bin'), 'w', encoding='utf-8') as occurrence_file:
                occurrence_file.write('occurrence occurrence id bin')
                occurrence_file.flush()

                settings = {
                    "gul_summaries": [{
                        "ord_output": {"psept_oep": True},
                    }],
                    'model_settings': {'event_occurrence_id': 'occurrence id'}
                }
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'occurrence.bin'), 'r', encoding='utf-8') as new_occurrence_file:
                self.assertEqual('occurrence occurrence id bin', new_occurrence_file.read())

    def test_no_occurrence_bin_exists___oasis_exception_is_raised(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)
            os.remove(os.path.join(d, 'static', 'occurrence.bin'))

            with self.assertRaises(OasisException):
                settings = {"gul_summaries": [{
                    "eltcalc": True,
                    "aalcalc": True,
                    "pltcalc": True,
                    "lec_output": True,
                }]}
                prepare_run_inputs(settings, d, model_storage)

    def test_periods_bin_already_exists___existing_bin_is_unchanged(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'input', 'periods.bin'), 'w', encoding='utf-8') as periods_file:
                periods_file.write('periods bin')
                periods_file.flush()

                settings = {"gul_summaries": [{
                    "eltcalc": True,
                    "aalcalc": True,
                    "pltcalc": True,
                    "lec_output": True,
                }]}
                prepare_run_inputs(settings, d, model_storage)

            with io.open(os.path.join(d, 'input', 'periods.bin'), 'r', encoding='utf-8') as new_periods_file:
                self.assertEqual('periods bin', new_periods_file.read())

    def test_periods_bin_doesnt_not_exist_event_set_isnt_specified___bin_is_copied_from_static(self):
        with TemporaryDirectory() as d:
            self.make_fake_bins(d)
            model_storage = LocalStorage(root_dir=os.path.join(d, "static"), cache_dir=None)

            with io.open(os.path.join(d, 'static', 'periods.bin'), 'w', encoding='utf-8') as periods_file:
                periods_file.write('periods bin')
                periods_file.flush()

                prepare_run_inputs({}, d, model_storage)

            with io.open(os.path.join(d, 'input', 'periods.bin'), 'r', encoding='utf-8') as new_periods_file:
                self.assertEqual('periods bin', new_periods_file.read())


class SetFootprintSet(TestCase):

    def setUp(self):
        """
        Declare identifier for footprint file set, footprint file names and
        symbolic link names for tests.
        """
        self.setting_val = 'f'
        self.parquet_format, self.zip_bin_format, self.bin_format, self.csv_format = range(4)

        # Ordered by priority, i.e. parquet files take priority over all others
        self.fp_filenames = {
            'parquet': ['footprint.parquet', 'footprint_parquet_meta.json'],
            'binz': ['footprint.bin.z', 'footprint.idx.z'],
            'bin': ['footprint.bin', 'footprint.idx'],
            'csv': ['footprint.csv']
        }
        self.fp_set_filenames = {}
        for fp_format, fp_filenames in self.fp_filenames.items():
            self.fp_set_filenames[fp_format] = []
            for filename in fp_filenames:
                stem, extension = filename.split('.', 1)
                self.fp_set_filenames[fp_format].append(f'{stem}_{self.setting_val}.{extension}')

    def make_fake_footprint_files(self, d, priority_level):
        """
        Write footprint files to directory.

        Args:
            d (str): directory name
            priority_level (int): file format being tested - formats with higher
                priority (closer to 1) include those of lower priority
        """
        os.makedirs(os.path.join(d, 'static'), exist_ok=True)
        for fp_format, fp_filenames in islice(self.fp_filenames.items(), priority_level, None):
            if fp_format == 'parquet':
                # Handle parquet as a directory
                for filename in fp_filenames:
                    stem, extension = filename.split('.', 1)
                    if 'meta' in filename:
                        # For the meta file, create a file instead of a directory
                        meta_fp = os.path.join(d, 'static', f'{stem}_{self.setting_val}.{extension}')
                        Path(meta_fp).touch()
                    else:
                        # Create a directory for the parquet format
                        parquet_dir_path = os.path.join(d, 'static', f'{stem}_{self.setting_val}.{extension}')
                        os.makedirs(parquet_dir_path, exist_ok=True)
            else:
                # For other formats, just create files
                for filename in fp_filenames:
                    stem, extension = filename.split('.', 1)
                    file_path = os.path.join(d, 'static', f'{stem}_{self.setting_val}.{extension}')
                    Path(file_path).touch()

    def test_symbolic_links_to_parquet_files(self):
        """
        Test symbolic links pointing to footprint files in parquet format are
        created. Also test links to zipped binary, binary and csv formatted
        footprint files are not created.
        """
        with TemporaryDirectory() as d:
            self.make_fake_footprint_files(d, self.parquet_format)

            set_footprint_set(self.setting_val, d)

            for fp_filename, fp_set_filename in zip(
                self.fp_filenames['parquet'], self.fp_set_filenames['parquet']
            ):
                self.assertEqual(
                    os.readlink(os.path.join(d, 'static', fp_filename)),
                    os.path.join(d, 'static', fp_set_filename)
                )

            for fp_format, fp_filename in self.fp_filenames.items():
                if fp_format == 'parquet':
                    continue
                for filename in fp_filename:
                    assert os.path.exists(os.path.join(d, 'static', filename)) is False

    def test_symbolic_links_to_binz_files(self):
        """
        Test symbolic links pointing to footprint files in zipped binary format
        are created. Also test links to parquet, binary and csv formatted
        footprint files are not created.
        """
        with TemporaryDirectory() as d:
            self.make_fake_footprint_files(d, self.zip_bin_format)

            set_footprint_set(self.setting_val, d)

            for fp_filename, fp_set_filename in zip(
                self.fp_filenames['binz'], self.fp_set_filenames['binz']
            ):
                self.assertEqual(
                    os.readlink(os.path.join(d, 'static', fp_filename)),
                    os.path.join(d, 'static', fp_set_filename)
                )

            for fp_format, fp_filename in self.fp_filenames.items():
                if fp_format == 'binz':
                    continue
                for filename in fp_filename:
                    assert os.path.exists(os.path.join(d, 'static', filename)) is False

    def test_symbolic_links_to_bin_files(self):
        """
        Test symbolic links pointing to footprint files in binary format are
        created. Also test links to parquet, zipped binary and csv formatted
        footprint files are not created.
        """
        with TemporaryDirectory() as d:
            self.make_fake_footprint_files(d, self.bin_format)

            set_footprint_set(self.setting_val, d)

            for fp_filename, fp_set_filename in zip(
                self.fp_filenames['bin'], self.fp_set_filenames['bin']
            ):
                self.assertEqual(
                    os.readlink(os.path.join(d, 'static', fp_filename)),
                    os.path.join(d, 'static', fp_set_filename)
                )

            for fp_format, fp_filename in self.fp_filenames.items():
                if fp_format == 'bin':
                    continue
                for filename in fp_filename:
                    assert os.path.exists(os.path.join(d, 'static', filename)) is False

    def test_symbolic_link_to_csv_file(self):
        """
        Test symbolic link pointing to footprint file in csv format is created.
        Also test links to parquet, zipped binary and binary formatted footprint
        files are not created.
        """
        with TemporaryDirectory() as d:
            self.make_fake_footprint_files(d, self.csv_format)

            set_footprint_set(self.setting_val, d)

            self.assertEqual(
                os.readlink(os.path.join(d, 'static', self.fp_filenames['csv'][0])),
                os.path.join(d, 'static', self.fp_set_filenames['csv'][0])
            )

            for fp_format, fp_filename in self.fp_filenames.items():
                if fp_format == 'csv':
                    continue
                for filename in fp_filename:
                    assert os.path.exists(os.path.join(d, 'static', filename)) is False

    def test_no_valid_symbolic_links_raises_exception(self):
        """
        Test exception is raised should there be no valid footprint file format
        available.
        """
        with TemporaryDirectory() as d:
            with self.assertRaises(OasisException):
                set_footprint_set(self.setting_val, d)


class SetVulnerabilitySet(TestCase):

    def setUp(self):
        """ Declare identifier for vulnerability file set for tests. """
        self.setting_val = 'test'
        self.vulnerability_dataset = vulnerability_dataset
        self.parquetvulnerability_meta_filename = parquetvulnerability_meta_filename

    def make_mock_vulnerability_files(self, directory, file_format):
        """ Write a mock vulnerability file in the specified format to the directory. """
        os.makedirs(os.path.join(directory, 'static'), exist_ok=True)
        if file_format == 'parquet':
            # Create a directory for parquet format
            base_name, extension = os.path.splitext(self.vulnerability_dataset)
            os.makedirs(os.path.join(directory, 'static', f'{base_name}_{self.setting_val}{extension}'))
            base_meta_name, extension = os.path.splitext(self.parquetvulnerability_meta_filename)
            Path(os.path.join(directory, 'static', f'{base_meta_name}_{self.setting_val}{extension}')).touch()
        else:
            # Create a file for other formats
            filename = f'vulnerability_{self.setting_val}.{file_format}'
            Path(os.path.join(directory, 'static', filename)).touch()

    def test_symbolic_link_creation(self):
        """ Test that symbolic links or directories are correctly created. """
        vulnerability_formats = ['bin', 'parquet', 'csv']

        for file_format in vulnerability_formats:
            with TemporaryDirectory() as d:
                self.make_mock_vulnerability_files(d, file_format)
                set_vulnerability_set(self.setting_val, d)

                if file_format == 'parquet':
                    # Check for symbolic link to directory
                    base_name, extension = os.path.splitext(self.vulnerability_dataset)
                    src_dir = os.path.join(d, 'static', f'{base_name}_{self.setting_val}{extension}')
                    target_dir = os.path.join(d, 'static', self.vulnerability_dataset)
                    self.assertTrue(os.path.islink(target_dir))
                    self.assertEqual(os.readlink(target_dir), src_dir)
                    # Check for symbolic link to meta file
                    base_meta_name, extension = os.path.splitext(self.parquetvulnerability_meta_filename)
                    src_meta_fp = os.path.join(d, 'static', f'{base_meta_name}_{self.setting_val}{extension}')
                    target_meta_fp = os.path.join(d, 'static', self.parquetvulnerability_meta_filename)
                    self.assertTrue(os.path.exists(target_meta_fp))
                    self.assertEqual(src_meta_fp, os.readlink(target_meta_fp))
                else:
                    # Check for symbolic link to file
                    vulnerability_fp = os.path.join(d, 'static', f'vulnerability_{self.setting_val}.{file_format}')
                    vulnerability_target_fp = os.path.join(d, 'static', f'vulnerability.{file_format}')
                    self.assertTrue(os.path.islink(vulnerability_target_fp))
                    self.assertEqual(os.readlink(vulnerability_target_fp), vulnerability_fp)

    def test_no_valid_files_raises_exception(self):
        """ Test that an exception is raised if no valid vulnerability files are found. """
        with TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'static'), exist_ok=True)

            with self.assertRaises(OasisException):
                set_vulnerability_set(self.setting_val, d)


class CleanBinDirectory(TestCase):
    def test_output_and_bin_input_files_are_removed(self):
        with TemporaryDirectory() as d:
            Path(os.path.join(d, TAR_FILE)).touch()

            for f in INPUT_FILES.values():
                Path(os.path.join(d, f['name'] + '.bin')).touch()

            cleanup_bin_directory(d)

            self.assertFalse(os.path.exists(os.path.join(d, TAR_FILE)))
            for f in INPUT_FILES:
                self.assertFalse(os.path.exists(os.path.join(d, f + '.bin')))


class CheckBinTarFile(TestCase):
    def test_all_il_files_are_missing_check_il_is_false___result_is_true(self):
        with TemporaryDirectory() as d:
            tar_file_name = os.path.join(d, 'exposures.tar')

            with tarfile.open(tar_file_name, 'w', encoding='utf-8') as tar:
                for f in GUL_INPUT_FILES.values():
                    Path(os.path.join(d, '{}.bin'.format(f['name']))).touch()

                tar.add(d, arcname='/')

            self.assertTrue(check_binary_tar_file(tar_file_name))

    def test_all_files_are_present_check_il_is_true___result_is_true(self):
        with TemporaryDirectory() as d:
            tar_file_name = os.path.join(d, 'exposures.tar')

            with tarfile.open(tar_file_name, 'w', encoding='utf-8') as tar:
                for f in INPUT_FILES.values():
                    Path(os.path.join(d, '{}.bin'.format(f['name']))).touch()

                tar.add(d, arcname='/')

            self.assertTrue(check_binary_tar_file(tar_file_name, check_il=True))

    @given(
        lists(sampled_from([f['name'] for f in chain(GUL_INPUT_FILES.values(), IL_INPUT_FILES.values())]), min_size=1, unique=True)
    )
    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_some_files_are_missing_check_il_is_true___error_is_raised(self, missing):
        with TemporaryDirectory() as d:
            tar_file_name = os.path.join(d, 'exposures.tar')

            with tarfile.open(tar_file_name, 'w', encoding='utf-8') as tar:
                for f in INPUT_FILES.values():
                    if f['name'] in missing:
                        continue

                    Path(os.path.join(d, '{}.bin'.format(f['name']))).touch()

                tar.add(d, arcname='/')

            with self.assertRaises(OasisException):
                check_binary_tar_file(tar_file_name, check_il=True)
