import json
import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import TestCase

from oasislmf.model_execution.bash import (bash_params, bash_wrapper,
                                           create_bash_analysis,
                                           create_bash_outputs, genbash)
from oasislmf.utils import diff

TEST_DIRECTORY = os.path.dirname(__file__)


class GenbashBase(TestCase):
    """Base class for genbash tests with dynamic test generation"""
    KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "input_json_settings")
    KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_base")
    KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_base")

    @classmethod
    def setUpClass(cls):

        # Default parameters
        cls.ri_iterations = 0
        cls.gul_alloc_rule = 0
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 2
        cls.num_gul_per_lb = 0
        cls.num_fm_per_lb = 0
        cls.event_shuffle = 1
        cls.gulmc = True
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False

        # Recreate output folder
        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)

    def setUp(self):
        self.temp_reference_file = None

    def tearDown(self):
        if self.temp_reference_file:
            self.temp_reference_file.close()
            if os.path.exists(self.temp_reference_file.name):
                os.remove(self.temp_reference_file.name)

    def _load_analysis_settings(self, name):
        """Load analysis settings from JSON file"""
        input_filename = os.path.join(self.KPARSE_INPUT_FOLDER, f"{name}.json")
        with open(input_filename, encoding='utf-8') as file:
            return json.load(file)['analysis_settings']

    def _get_output_filename(self, name, num_partitions, num_reinsurance_iterations=None):
        """Generate output filename based on parameters"""
        if not num_reinsurance_iterations:
            return os.path.join(self.KPARSE_OUTPUT_FOLDER, f"{name}_{num_partitions}_partition.sh")
        return os.path.join(
            self.KPARSE_OUTPUT_FOLDER,
            f"{name}_{num_reinsurance_iterations}_reins_layer_{num_partitions}_partition.sh")

    def genbash(self, name, num_partitions, num_reinsurance_iterations=None,
                fifo_tmp_dir=None, stderr_guard=None, gulmc=True, gul_alloc_rule=None,
                il_alloc_rule=None, ri_alloc_rule=None, bash_trace=None,
                _get_getmodel_cmd=None):
        """Generate bash script from analysis settings"""

        analysis_settings = self._load_analysis_settings(name)
        output_filename = self._get_output_filename(name, num_partitions, num_reinsurance_iterations)

        genbash(
            num_partitions,
            analysis_settings,
            filename=output_filename,
            num_reinsurance_iterations=num_reinsurance_iterations or self.ri_iterations,
            fifo_tmp_dir=fifo_tmp_dir or self.fifo_tmp_dir,
            stderr_guard=stderr_guard or self.stderr_guard,
            gulmc=gulmc or self.gulmc,
            gul_alloc_rule=gul_alloc_rule or self.gul_alloc_rule,
            il_alloc_rule=il_alloc_rule or self.il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule or self.ri_alloc_rule,
            num_gul_per_lb=self.num_gul_per_lb,
            num_fm_per_lb=self.num_fm_per_lb,
            event_shuffle=self.event_shuffle,
            bash_trace=bash_trace or self.bash_trace,
            _get_getmodel_cmd=_get_getmodel_cmd,
        )

    def gen_chunked_bash(self, name, num_partitions, num_reinsurance_iterations=None,
                         fifo_tmp_dir=None, stderr_guard=None, gulmc=True, gul_alloc_rule=None,
                         il_alloc_rule=None, ri_alloc_rule=None, bash_trace=None,
                         _get_getmodel_cmd=None):
        """Generate chunked bash scripts (one per partition plus output script)"""

        analysis_settings = self._load_analysis_settings(name)
        output_base = self._get_output_filename(name, num_partitions, num_reinsurance_iterations).replace('.sh', '')

        params = bash_params(
            max_process_id=num_partitions,
            analysis_settings=analysis_settings,
            num_reinsurance_iterations=num_reinsurance_iterations or self.ri_iterations,
            fifo_tmp_dir=fifo_tmp_dir or self.fifo_tmp_dir,
            stderr_guard=stderr_guard or self.stderr_guard,
            gulmc=gulmc or self.gulmc,
            gul_alloc_rule=gul_alloc_rule or self.gul_alloc_rule,
            il_alloc_rule=il_alloc_rule or self.il_alloc_rule,
            ri_alloc_rule=ri_alloc_rule or self.ri_alloc_rule,
            num_gul_per_lb=self.num_gul_per_lb,
            num_fm_per_lb=self.num_fm_per_lb,
            event_shuffle=self.event_shuffle,
            bash_trace=bash_trace or self.bash_trace,
            _get_getmodel_cmd=_get_getmodel_cmd,
        )

        # Generate partition scripts
        fifo_tmp_dir = params['fifo_tmp_dir']
        for process_id in range(num_partitions):
            params['filename'] = f'{output_base}.{process_id}.sh'
            if os.path.exists(params['filename']):
                os.remove(params['filename'])

            with bash_wrapper(
                params['filename'],
                bash_trace or self.bash_trace,
                stderr_guard or self.stderr_guard,
                custom_gulcalc_log_start=params['custom_gulcalc_log_start'],
                custom_gulcalc_log_finish=params['custom_gulcalc_log_finish'],
            ):
                create_bash_analysis(
                    **{**params, 'process_number': process_id + 1, 'fifo_tmp_dir': fifo_tmp_dir}
                )
                fifo_tmp_dir = False

        # Generate output script
        params['filename'] = f'{output_base}.output.sh'
        if os.path.exists(params['filename']):
            os.remove(params['filename'])

        with bash_wrapper(
            params['filename'],
            bash_trace or self.bash_trace,
            stderr_guard or self.stderr_guard,
            custom_gulcalc_log_start=params['custom_gulcalc_log_start'],
            custom_gulcalc_log_finish=params['custom_gulcalc_log_finish'],
        ):
            create_bash_outputs(**params)

    def check_chunks(self, name, num_partitions):
        for i in range(num_partitions):
            self.check(f'{name}.{i}')
        self.check(f'{name}.output')

    def check(self, name, reference_filename=None):
        """Compare generated output with reference file"""
        output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, f"{name}.sh")
        if not reference_filename:
            reference_filename = os.path.join(self.KPARSE_REFERENCE_FOLDER, f"{name}.sh")

        # Handle dynamic FIFO directory replacement
        if self.fifo_tmp_dir:
            reference_filename = self._prepare_dynamic_reference(output_filename, reference_filename)

        d = diff.unified_diff(reference_filename, output_filename, as_string=True)
        if d:
            self.fail(d)

    def _prepare_dynamic_reference(self, output_filename, ref_template):
        """Create temp reference file with dynamic FIFO directory replaced"""
        # Extract FIFO directory from output
        tmp_fifo_dir = None
        with open(output_filename, 'r') as f:
            for line in f:
                if '/tmp/' in line:
                    tmp_fifo_dir = line.split('/')[2]
                    break

        # Create temp reference with placeholder replaced
        ref_tmp_file = NamedTemporaryFile("w+", delete=False, prefix='bash')
        with open(ref_template, 'r') as f:
            kernel_script = f.read()
        kernel_script = kernel_script.replace('%FIFO_DIR%', tmp_fifo_dir)
        ref_tmp_file.write(kernel_script)
        ref_tmp_file.close()

        self.temp_reference_file = ref_tmp_file
        return ref_tmp_file.name


def discover_and_add_tests(cls):
    """Dynamically discover JSON files and add test methods to the class"""

    input_folder = cls.KPARSE_INPUT_FOLDER
    if not os.path.exists(input_folder):
        return cls

    # Discover all JSON files
    json_files = [f.stem for f in Path(input_folder).glob('*.json')]

    # Common partition counts to test
    partition_counts = [1, 8]

    for json_name in json_files:
        for num_partitions in partition_counts:
            expected_name = f"{json_name}_{num_partitions}_partition"
            ref_file = os.path.join(cls.KPARSE_REFERENCE_FOLDER, f"{expected_name}.sh")

            # Only add tests if reference file exists
            # if not os.path.exists(ref_file):
            #    continue

            # Add standard test
            def make_test(name, partitions, expected):
                def test(self):
                    self.genbash(name, partitions)
                    self.check(expected)
                return test

            test_name = f"test_{expected_name}"
            setattr(cls, test_name, make_test(json_name, num_partitions, expected_name))

            # Add chunked test
            def make_chunk_test(name, partitions, expected):
                def test(self):
                    self.gen_chunked_bash(name, partitions)
                    self.check_chunks(expected, partitions)
                return test

            chunk_test_name = f"test_{expected_name}_chunk"
            setattr(cls, chunk_test_name, make_chunk_test(json_name, num_partitions, expected_name))

    return cls


# Apply dynamic test generation
@discover_and_add_tests
class Genbash_base(GenbashBase):
    """Standard genbash tests with auto-discovered test cases"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_base")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_base")


@discover_and_add_tests
class Genbash_ErrorGuard_and_TempDir(GenbashBase):
    """Tests with error guard and temp directory enabled"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_err")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_err")

        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = True
        cls.stderr_guard = True

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


@discover_and_add_tests
class Genbash_LoadBalancer_and_gulpy(GenbashBase):
    """Tests with load balancer configuration"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_lb")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_lb")

        cls.num_gul_per_lb = 2
        cls.num_fm_per_lb = 2
        cls.gulmc = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


# Special case: Custom gulcalc tests
class Genbash_CustomGulcalc(GenbashBase):
    """Tests for custom gulcalc commands"""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "output_bash_csm")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "reference_bash_csm")

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)

    @staticmethod
    def _get_getmodel_cmd(**kwargs):
        error_guard = kwargs.get('stderr_guard')
        cmd = 'custom_gulcalc_command'
        if error_guard:
            cmd = f'({cmd}) 2>> log/gul_stderror.err'
        return cmd

    def test_custom_gul_summarycalc_1_partition(self):
        self.genbash("custom_gul_summarycalc_1_output", 1,
                     _get_getmodel_cmd=self._get_getmodel_cmd)
        self.check("custom_gul_summarycalc_1_output_1_partition")

    def test_custom_gul_summarycalc_1_partition_chunk(self):
        self.gen_chunked_bash("custom_gul_summarycalc_1_output", 1,
                              _get_getmodel_cmd=self._get_getmodel_cmd)
        self.check_chunks("custom_gul_summarycalc_1_output_1_partition", 1)
