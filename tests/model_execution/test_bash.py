
import hashlib
import io
import json
import os
import shutil
from tempfile import NamedTemporaryFile
from unittest import TestCase

from oasislmf.model_execution.bash import (bash_params, bash_wrapper,
                                           create_bash_analysis,
                                           create_bash_outputs, genbash)
from oasislmf.utils import diff

TEST_DIRECTORY = os.path.dirname(__file__)


class Genbash(TestCase):
    @classmethod
    def setUpClass(cls):
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "cov_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "cov_kparse_reference")

        # defaults
        cls.ri_iterations = 0
        cls.gul_alloc_rule = 0
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 2
        cls.num_gul_per_lb = 0
        cls.num_fm_per_lb = 0
        cls.event_shuffle = 1
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = True
        cls.fmpy = False
        cls.get_getmodel_cmd = None

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)

    def setUp(self):
        self.temp_reference_file = None

    def tearDown(self):
        if self.temp_reference_file is not None:
            # If already closed, no exception is raised
            self.temp_reference_file.close()
            os.remove(self.temp_reference_file.name)

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with io.open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def genbash(self, name, num_partitions,
                num_reinsurance_iterations=None,
                fifo_tmp_dir=None,
                stderr_guard=None,
                gul_alloc_rule=None,
                il_alloc_rule=None,
                ri_alloc_rule=None,
                bash_trace=None,
                gul_legacy_stream=None,
                fmpy=None,
                _get_getmodel_cmd=None,
                ):

        input_filename = os.path.join(self.KPARSE_INPUT_FOLDER, "{}.json".format(name))
        if not num_reinsurance_iterations:
            output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, "{}_{}_partition.sh".format(name, num_partitions))
        else:
            output_filename = os.path.join(
                self.KPARSE_OUTPUT_FOLDER,
                "{}_{}_reins_layer_{}_partition.sh".format(name, num_reinsurance_iterations, num_partitions))

        with io.open(input_filename, encoding='utf-8') as file:
            analysis_settings = json.load(file)['analysis_settings']

        genbash(
            num_partitions,
            analysis_settings,
            filename=output_filename,
            num_reinsurance_iterations=(num_reinsurance_iterations or self.ri_iterations),
            fifo_tmp_dir=(fifo_tmp_dir or self.fifo_tmp_dir),
            stderr_guard=(stderr_guard or self.stderr_guard),
            gul_alloc_rule=(gul_alloc_rule or self.gul_alloc_rule),
            il_alloc_rule=(il_alloc_rule or self.il_alloc_rule),
            ri_alloc_rule=(ri_alloc_rule or self.ri_alloc_rule),
            num_gul_per_lb=self.num_gul_per_lb,
            num_fm_per_lb=self.num_fm_per_lb,
            event_shuffle=self.event_shuffle,
            bash_trace=(bash_trace or self.bash_trace),
            gul_legacy_stream=(gul_legacy_stream or self.gul_legacy_stream),
            fmpy=(fmpy or self.fmpy),
            _get_getmodel_cmd=_get_getmodel_cmd,
        )

    def gen_chunked_bash(self, name,
                         num_partitions,
                         num_reinsurance_iterations=None,
                         fifo_tmp_dir=None,
                         stderr_guard=None,
                         gul_alloc_rule=None,
                         il_alloc_rule=None,
                         ri_alloc_rule=None,
                         bash_trace=None,
                         gul_legacy_stream=None,
                         fmpy=None,
                         _get_getmodel_cmd=None,
                         ):

        input_filename = os.path.join(self.KPARSE_INPUT_FOLDER, "{}.json".format(name))
        if not num_reinsurance_iterations:
            output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, "{}_{}_partition".format(name, num_partitions))
        else:
            output_filename = os.path.join(
                self.KPARSE_OUTPUT_FOLDER,
                "{}_{}_reins_layer_{}_partition".format(name, num_reinsurance_iterations, num_partitions))

        with io.open(input_filename, encoding='utf-8') as file:
            analysis_settings = json.load(file)['analysis_settings']
        params = bash_params(
            max_process_id=num_partitions,
            analysis_settings=analysis_settings,
            num_reinsurance_iterations=(num_reinsurance_iterations or self.ri_iterations),
            fifo_tmp_dir=(fifo_tmp_dir or self.fifo_tmp_dir),
            stderr_guard=(stderr_guard or self.stderr_guard),
            gul_alloc_rule=(gul_alloc_rule or self.gul_alloc_rule),
            il_alloc_rule=(il_alloc_rule or self.il_alloc_rule),
            ri_alloc_rule=(ri_alloc_rule or self.ri_alloc_rule),
            num_gul_per_lb=self.num_gul_per_lb,
            num_fm_per_lb=self.num_fm_per_lb,
            event_shuffle=self.event_shuffle,
            bash_trace=(bash_trace or self.bash_trace),
            gul_legacy_stream=(gul_legacy_stream or self.gul_legacy_stream),
            fmpy=(fmpy or self.fmpy),
            _get_getmodel_cmd=_get_getmodel_cmd,
        )

        fifo_tmp_dir = params['fifo_tmp_dir']
        for process_id in range(num_partitions):
            params['filename'] = f'{output_filename}.{process_id}.sh'
            # remove the file if it already exists
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
                    **{
                        **params,
                        'process_number': process_id + 1,
                        'fifo_tmp_dir': fifo_tmp_dir,
                    }
                )
                fifo_tmp_dir = False

        # remove the file if it already exists
        params['filename'] = f'{output_filename}.output.sh'
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
        pass
        output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, "{}.sh".format(name))
        if not reference_filename:
            reference_filename = os.path.join(self.KPARSE_REFERENCE_FOLDER, "{}.sh".format(name))
        if self.fifo_tmp_dir:
            # Create temp Ref file
            ref_template = reference_filename
            ref_tmp_file = NamedTemporaryFile("w+", delete=False)
            with io.open(output_filename, 'r') as f:
                for line in f:
                    if '/tmp/' in line:
                        tmp_fifo_dir = line.split('/')[2]
                        break

            # Replace placeholder '%FIFO_DIR%' with '<RandomDirName>'
            with io.open(ref_template, 'r') as f:
                ktools_script = f.read()
            ktools_script = ktools_script.replace('%FIFO_DIR%', tmp_fifo_dir)
            ref_tmp_file.write(ktools_script)
            ref_tmp_file.close()
            reference_filename = ref_tmp_file.name

        d = diff.unified_diff(reference_filename, output_filename, as_string=True)
        if d:
            self.fail(d)

    def test_gul_summarycalc_1_partition(self):
        self.genbash("gul_summarycalc_1_output", 1)
        self.check("gul_summarycalc_1_output_1_partition")

    def test_gul_summarycalc_20_partition(self):
        self.genbash("gul_summarycalc_1_output", 20)
        self.check("gul_summarycalc_1_output_20_partition")

    def test_gul_eltcalc_1_partition(self):
        self.genbash("gul_eltcalc_1_output", 1)
        self.check("gul_eltcalc_1_output_1_partition")

    def test_gul_eltcalc_20_partition(self):
        self.genbash("gul_eltcalc_1_output", 20)
        self.check("gul_eltcalc_1_output_20_partition")

    def test_gul_aalcalc_1_partition(self):
        self.genbash("gul_aalcalc_1_output", 1)
        self.check("gul_aalcalc_1_output_1_partition")

    def test_gul_aalcalc_20_partition(self):
        self.genbash("gul_aalcalc_1_output", 20)
        self.check("gul_aalcalc_1_output_20_partition")

    def test_gul_pltcalc_1_partition(self):
        self.genbash("gul_pltcalc_1_output", 1)
        self.check("gul_pltcalc_1_output_1_partition")

    def test_gul_pltcalc_20_partition(self):
        self.genbash("gul_pltcalc_1_output", 20)
        self.check("gul_pltcalc_1_output_20_partition")

    def test_gul_agg_fu_lec_1_partition(self):
        self.genbash("gul_agg_fu_lec_1_output", 1)
        self.check("gul_agg_fu_lec_1_output_1_partition")

    def test_gul_agg_fu_lec_20_partition(self):
        self.genbash("gul_agg_fu_lec_1_output", 20)
        self.check("gul_agg_fu_lec_1_output_20_partition")

    def test_gul_occ_fu_lec_1_output_1_partition(self):
        self.genbash("gul_occ_fu_lec_1_output", 1)
        self.check("gul_occ_fu_lec_1_output_1_partition")

    def test_gul_occ_fu_lec_1_output_20_partition(self):
        self.genbash("gul_occ_fu_lec_1_output", 20)
        self.check("gul_occ_fu_lec_1_output_20_partition")

    def test_gul_agg_ws_lec_1_partition(self):
        self.genbash("gul_agg_ws_lec_1_output", 1)
        self.check("gul_agg_ws_lec_1_output_1_partition")

    def test_gul_agg_ws_lec_20_partition(self):
        self.genbash("gul_agg_ws_lec_1_output", 20)
        self.check("gul_agg_ws_lec_1_output_20_partition")

    def test_gul_occ_ws_lec_1_partition(self):
        self.genbash("gul_occ_ws_lec_1_output", 1)
        self.check("gul_occ_ws_lec_1_output_1_partition")

    def test_gul_occ_ws_lec_20_partition(self):
        self.genbash("gul_occ_ws_lec_1_output", 20)
        self.check("gul_occ_ws_lec_1_output_20_partition")

    def test_gul_agg_ws_mean_lec_1_partition(self):
        self.genbash("gul_agg_ws_mean_lec_1_output", 1)
        self.check("gul_agg_ws_mean_lec_1_output_1_partition")

    def test_gul_agg_ws_mean_lec_20_partition(self):
        self.genbash("gul_agg_ws_mean_lec_1_output", 20)
        self.check("gul_agg_ws_mean_lec_1_output_20_partition")

    def test_gul_occ_ws_mean_lec_1_partition(self):
        self.genbash("gul_occ_ws_mean_lec_1_output", 1)
        self.check("gul_occ_ws_mean_lec_1_output_1_partition")

    def test_gul_occ_ws_mean_lec_20_partition(self):
        self.genbash("gul_occ_ws_mean_lec_1_output", 20)
        self.check("gul_occ_ws_mean_lec_1_output_20_partition")

    def test_il_agg_sample_mean_lec_1_partition(self):
        self.genbash("il_agg_sample_mean_lec_1_output", 1)
        self.check("il_agg_sample_mean_lec_1_output_1_partition")

    def test_il_agg_sample_mean_lec_20_partition(self):
        self.genbash("il_agg_sample_mean_lec_1_output", 20)
        self.check("il_agg_sample_mean_lec_1_output_20_partition")

    def test_il_occ_sample_mean_lec_1_partition(self):
        self.genbash("il_occ_sample_mean_lec_1_output", 1)
        self.check("il_occ_sample_mean_lec_1_output_1_partition")

    def test_il_occ_sample_mean_lec_20_partition(self):
        self.genbash("il_occ_sample_mean_lec_1_output", 20)
        self.check("il_occ_sample_mean_lec_1_output_20_partition")

    def test_il_summarycalc_1_partition(self):
        self.genbash("il_summarycalc_1_output", 1)
        self.check("il_summarycalc_1_output_1_partition")

    def test_il_summarycalc_20_partition(self):
        self.genbash("il_summarycalc_1_output", 20)
        self.check("il_summarycalc_1_output_20_partition")

    def test_il_eltcalc_1_partition(self):
        self.genbash("il_eltcalc_1_output", 1)
        self.check("il_eltcalc_1_output_1_partition")

    def test_il_eltcalc_20_partition(self):
        self.genbash("il_eltcalc_1_output", 20)
        self.check("il_eltcalc_1_output_20_partition")

    def test_il_aalcalc_1_partition(self):
        self.genbash("il_aalcalc_1_output", 1)
        self.check("il_aalcalc_1_output_1_partition")

    def test_il_aalcalc_20_partition(self):
        self.genbash("il_aalcalc_1_output", 20)
        self.check("il_aalcalc_1_output_20_partition")

    def test_il_pltcalc_1_partition(self):
        self.genbash("il_pltcalc_1_output", 1)
        self.check("il_pltcalc_1_output_1_partition")

    def test_il_pltcalc_20_partition(self):
        self.genbash("il_pltcalc_1_output", 20)
        self.check("il_pltcalc_1_output_20_partition")

    def test_il_agg_fu_lec_1_partition(self):
        self.genbash("il_agg_fu_lec_1_output", 1)
        self.check("il_agg_fu_lec_1_output_1_partition")

    def test_il_agg_fu_lec_20_partition(self):
        self.genbash("il_agg_fu_lec_1_output", 20)
        self.check("il_agg_fu_lec_1_output_20_partition")

    def test_il_occ_fu_lec_1_output_1_partition(self):
        self.genbash("il_occ_fu_lec_1_output", 1)
        self.check("il_occ_fu_lec_1_output_1_partition")

    def test_il_occ_fu_lec_1_output_20_partition(self):
        self.genbash("il_occ_fu_lec_1_output", 20)
        self.check("il_occ_fu_lec_1_output_20_partition")

    def test_il_agg_ws_lec_1_partition(self):
        self.genbash("il_agg_ws_lec_1_output", 1)
        self.check("il_agg_ws_lec_1_output_1_partition")

    def test_il_agg_ws_lec_20_partition(self):
        self.genbash("il_agg_ws_lec_1_output", 20)
        self.check("il_agg_ws_lec_1_output_20_partition")

    def test_il_occ_ws_lec_1_partition(self):
        self.genbash("il_occ_ws_lec_1_output", 1)
        self.check("il_occ_ws_lec_1_output_1_partition")

    def test_il_occ_ws_lec_20_partition(self):
        self.genbash("il_occ_ws_lec_1_output", 20)
        self.check("il_occ_ws_lec_1_output_20_partition")

    def test_il_agg_ws_mean_lec_1_partition(self):
        self.genbash("il_agg_ws_mean_lec_1_output", 1)
        self.check("il_agg_ws_mean_lec_1_output_1_partition")

    def test_il_agg_ws_mean_lec_20_partition(self):
        self.genbash("il_agg_ws_mean_lec_1_output", 20)
        self.check("il_agg_ws_mean_lec_1_output_20_partition")

    def test_il_occ_ws_mean_lec_1_partition(self):
        self.genbash("il_occ_ws_mean_lec_1_output", 1)
        self.check("il_occ_ws_mean_lec_1_output_1_partition")

    def test_il_occ_ws_mean_lec_20_partition(self):
        self.genbash("il_occ_ws_mean_lec_1_output", 20)
        self.check("il_occ_ws_mean_lec_1_output_20_partition")

    def test_il_agg_sample_mean_lec_1_output_1_partition(self):
        self.genbash("il_agg_sample_mean_lec_1_output", 1)
        self.check("il_agg_sample_mean_lec_1_output_1_partition")

    def test_il_agg_sample_mean_lec_1_output_20_partition(self):
        self.genbash("il_agg_sample_mean_lec_1_output", 20)
        self.check("il_agg_sample_mean_lec_1_output_20_partition")

    def test_il_occ_sample_mean_lec_1_output_1_partition(self):
        self.genbash("il_occ_sample_mean_lec_1_output", 1)
        self.check("il_occ_sample_mean_lec_1_output_1_partition")

    def test_il_occ_sample_mean_lec_1_output_20_partition(self):
        self.genbash("il_occ_sample_mean_lec_1_output", 20)
        self.check("il_occ_sample_mean_lec_1_output_20_partition")

    def test_all_calcs_1_partition(self):
        self.genbash("all_calcs_1_output", 1)
        self.check("all_calcs_1_output_1_partition")

    def test_all_calcs_20_partition(self):
        self.genbash("all_calcs_1_output", 20)
        self.check("all_calcs_1_output_20_partition")

    def test_all_calcs_40_partition(self):
        self.genbash("all_calcs_1_output", 40)
        self.check("all_calcs_1_output_40_partition")

    def test_gul_no_lec_1_output_1_partition(self):
        self.genbash("gul_no_lec_1_output", 1)
        self.check("gul_no_lec_1_output_1_partition")

    def test_gul_no_lec_1_output_2_partition(self):
        self.genbash("gul_no_lec_1_output", 2)
        self.check("gul_no_lec_1_output_2_partition")

    def test_gul_no_lec_2_output_1_partition(self):
        self.genbash("gul_no_lec_2_output", 1)
        self.check("gul_no_lec_2_output_1_partition")

    def test_gul_no_lec_2_output_2_partitions(self):
        self.genbash("gul_no_lec_2_output", 2)
        self.check("gul_no_lec_2_output_2_partition")

    def test_gul_lec_1_output_1_partition(self):
        self.genbash("gul_lec_1_output", 1)
        self.check("gul_lec_1_output_1_partition")

    def test_gul_lec_1_output_2_partitions(self):
        self.genbash("gul_lec_1_output", 2)
        self.check("gul_lec_1_output_2_partition")

    def test_gul_lec_2_output_1_partition(self):
        self.genbash("gul_lec_2_output", 1)
        self.check("gul_lec_2_output_1_partition")

    def test_gul_lec_2_output_2_partitions(self):
        self.genbash("gul_lec_2_output", 2)
        self.check("gul_lec_2_output_2_partition")

    def test_il_no_lec_1_output_1_partition(self):
        self.genbash("il_no_lec_1_output", 1)
        self.check("il_no_lec_1_output_1_partition")

    def test_il_no_lec_1_output_2_partition(self):
        self.genbash("il_no_lec_1_output", 2)
        self.check("il_no_lec_1_output_2_partition")

    def test_il_no_lec_2_output_1_partition(self):
        self.genbash("il_no_lec_2_output", 1)
        self.check("il_no_lec_2_output_1_partition")

    def test_il_no_lec_2_output_2_partitions(self):
        self.genbash("il_no_lec_2_output", 2)
        self.check("il_no_lec_2_output_2_partition")

    def test_il_lec_1_output_1_partition(self):
        self.genbash("il_lec_1_output", 1)
        self.check("il_lec_1_output_1_partition")

    def test_il_lec_1_output_2_partitions(self):
        self.genbash("il_lec_1_output", 2)
        self.check("il_lec_1_output_2_partition")

    def test_il_lec_2_output_1_partition(self):
        self.genbash("il_lec_2_output", 1)
        self.check("il_lec_2_output_1_partition")

    def test_il_lec_2_output_2_partitions(self):
        self.genbash("il_lec_2_output", 2)
        self.check("il_lec_2_output_2_partition")

    def test_gul_il_no_lec_1_output_1_partition(self):
        self.genbash("gul_il_no_lec_1_output", 1)
        self.check("gul_il_no_lec_1_output_1_partition")

    def test_gul_il_no_lec_1_output_2_partition(self):
        self.genbash("gul_il_no_lec_1_output", 2)
        self.check("gul_il_no_lec_1_output_2_partition")

    def test_gul_il_no_lec_2_output_1_partition(self):
        self.genbash("gul_il_no_lec_2_output", 1)
        self.check("gul_il_no_lec_2_output_1_partition")

    def test_gul_il_no_lec_2_output_2_partitions(self):
        self.genbash("gul_il_no_lec_2_output", 2)
        self.check("gul_il_no_lec_2_output_2_partition")

    def test_gul_il_lec_1_output_1_partition(self):
        self.genbash("gul_il_lec_1_output", 1)
        self.check("gul_il_lec_1_output_1_partition")

    def test_gul_il_lec_1_output_2_partitions(self):
        self.genbash("gul_il_lec_1_output", 2)
        self.check("gul_il_lec_1_output_2_partition")

    def test_gul_il_lec_2_output_1_partition(self):
        self.genbash("gul_il_lec_2_output", 1)
        self.check("gul_il_lec_2_output_1_partition")

    def test_gul_il_lec_2_output_2_partitions(self):
        self.genbash("gul_il_lec_2_output", 2)
        self.check("gul_il_lec_2_output_2_partition")

    def test_gul_il_lec_2_output_10_partitions(self):
        self.genbash("gul_il_lec_2_output", 10)
        self.check("gul_il_lec_2_output_10_partition")

    # RI checks
    def test_analysis_settings_1(self):
        self.genbash("analysis_settings_1", 1)
        self.check("analysis_settings_1_1_partition")

    def test_analysis_settings_2(self):
        self.genbash("analysis_settings_2", 1)
        self.check("analysis_settings_2_1_partition")

    def test_analysis_settings_3_0_reins_iters(self):
        self.genbash("analysis_settings_3", 1, 1)
        self.check("analysis_settings_3_1_reins_layer_1_partition")

    def test_analysis_settings_4_0_reins_iters(self):
        self.genbash("analysis_settings_4", 1, 1)
        self.check("analysis_settings_4_1_reins_layer_1_partition")

    def test_analysis_settings_5_0_reins_iters(self):
        self.genbash("analysis_settings_5", 1, 1)
        self.check("analysis_settings_5_1_reins_layer_1_partition")

    # ORD checks
    def test_gul_ord_ept_1_output_1_partitions(self):
        self.genbash("gul_ord_ept_1_output", 1)
        self.check("gul_ord_ept_1_output_1_partition")

    def test_gul_ord_ept_1_output_20_partitions(self):
        self.genbash("gul_ord_ept_1_output", 20)
        self.check("gul_ord_ept_1_output_20_partition")

    def test_gul_ord_psept_2_output_10_partitions(self):
        self.genbash("gul_ord_psept_2_output", 10)
        self.check("gul_ord_psept_2_output_10_partition")

    def test_gul_ord_ept_psept_lec_2_output_10_partitions(self):
        self.genbash("gul_ord_ept_psept_lec_2_output", 10)
        self.check("gul_ord_ept_psept_lec_2_output_10_partition")

    def test_gul_il_ord_ept_psept_2_output_10_partitions(self):
        self.genbash("gul_il_ord_ept_psept_2_output", 10)
        self.check("gul_il_ord_ept_psept_2_output_10_partition")

    def test_gul_il_ord_psept_lec_1_output_10_partitions(self):
        self.genbash("gul_il_ord_psept_lec_1_output", 10)
        self.check("gul_il_ord_psept_lec_1_output_10_partition")

    def test_gul_ord_palt_output_10_partitions(self):
        self.genbash("gul_ord_palt_output", 10)
        self.check("gul_ord_palt_output_10_partition")

    def test_gul_il_ord_palt_output_10_partitions(self):
        self.genbash("gul_il_ord_palt_output", 10)
        self.check("gul_il_ord_palt_output_10_partition")

# =============================================================================
# chunked analysis checks
# =============================================================================

    def test_gul_summarycalc_1_partition_chunk(self):
        self.gen_chunked_bash("gul_summarycalc_1_output", 1)
        self.check_chunks("gul_summarycalc_1_output_1_partition", 1)

    def test_gul_summarycalc_20_partition_chunk(self):
        self.gen_chunked_bash("gul_summarycalc_1_output", 20)
        self.check_chunks("gul_summarycalc_1_output_20_partition", 20)

    def test_gul_eltcalc_1_partition_chunk(self):
        self.gen_chunked_bash("gul_eltcalc_1_output", 1)
        self.check_chunks("gul_eltcalc_1_output_1_partition", 1)

    def test_gul_eltcalc_20_partition_chunk(self):
        self.gen_chunked_bash("gul_eltcalc_1_output", 20)
        self.check_chunks("gul_eltcalc_1_output_20_partition", 20)

    def test_gul_aalcalc_1_partition_chunk(self):
        self.gen_chunked_bash("gul_aalcalc_1_output", 1)
        self.check_chunks("gul_aalcalc_1_output_1_partition", 1)

    def test_gul_aalcalc_20_partition_chunk(self):
        self.gen_chunked_bash("gul_aalcalc_1_output", 20)
        self.check_chunks("gul_aalcalc_1_output_20_partition", 20)

    def test_gul_pltcalc_1_partition_chunk(self):
        self.gen_chunked_bash("gul_pltcalc_1_output", 1)
        self.check_chunks("gul_pltcalc_1_output_1_partition", 1)

    def test_gul_pltcalc_20_partition_chunk(self):
        self.gen_chunked_bash("gul_pltcalc_1_output", 20)
        self.check_chunks("gul_pltcalc_1_output_20_partition", 20)

    def test_gul_agg_fu_lec_1_partition_chunk(self):
        self.gen_chunked_bash("gul_agg_fu_lec_1_output", 1)
        self.check_chunks("gul_agg_fu_lec_1_output_1_partition", 1)

    def test_gul_agg_fu_lec_20_partition_chunk(self):
        self.gen_chunked_bash("gul_agg_fu_lec_1_output", 20)
        self.check_chunks("gul_agg_fu_lec_1_output_20_partition", 20)

    def test_gul_occ_fu_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_occ_fu_lec_1_output", 1)
        self.check_chunks("gul_occ_fu_lec_1_output_1_partition", 1)

    def test_gul_occ_fu_lec_1_output_20_partition_chunk(self):
        self.gen_chunked_bash("gul_occ_fu_lec_1_output", 20)
        self.check_chunks("gul_occ_fu_lec_1_output_20_partition", 20)

    def test_gul_agg_ws_lec_1_partition_chunk(self):
        self.gen_chunked_bash("gul_agg_ws_lec_1_output", 1)
        self.check_chunks("gul_agg_ws_lec_1_output_1_partition", 1)

    def test_gul_agg_ws_lec_20_partition_chunk(self):
        self.gen_chunked_bash("gul_agg_ws_lec_1_output", 20)
        self.check_chunks("gul_agg_ws_lec_1_output_20_partition", 20)

    def test_gul_occ_ws_lec_1_partition_chunk(self):
        self.gen_chunked_bash("gul_occ_ws_lec_1_output", 1)
        self.check_chunks("gul_occ_ws_lec_1_output_1_partition", 1)

    def test_gul_occ_ws_lec_20_partition_chunk(self):
        self.gen_chunked_bash("gul_occ_ws_lec_1_output", 20)
        self.check_chunks("gul_occ_ws_lec_1_output_20_partition", 20)

    def test_gul_agg_ws_mean_lec_1_partition_chunk(self):
        self.gen_chunked_bash("gul_agg_ws_mean_lec_1_output", 1)
        self.check_chunks("gul_agg_ws_mean_lec_1_output_1_partition", 1)

    def test_gul_agg_ws_mean_lec_20_partition_chunk(self):
        self.gen_chunked_bash("gul_agg_ws_mean_lec_1_output", 20)
        self.check_chunks("gul_agg_ws_mean_lec_1_output_20_partition", 20)

    def test_gul_occ_ws_mean_lec_1_partition_chunk(self):
        self.gen_chunked_bash("gul_occ_ws_mean_lec_1_output", 1)
        self.check_chunks("gul_occ_ws_mean_lec_1_output_1_partition", 1)

    def test_gul_occ_ws_mean_lec_20_partition_chunk(self):
        self.gen_chunked_bash("gul_occ_ws_mean_lec_1_output", 20)
        self.check_chunks("gul_occ_ws_mean_lec_1_output_20_partition", 20)

    def test_il_agg_sample_mean_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_agg_sample_mean_lec_1_output", 1)
        self.check_chunks("il_agg_sample_mean_lec_1_output_1_partition", 1)

    def test_il_agg_sample_mean_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_agg_sample_mean_lec_1_output", 20)
        self.check_chunks("il_agg_sample_mean_lec_1_output_20_partition", 20)

    def test_il_occ_sample_mean_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_occ_sample_mean_lec_1_output", 1)
        self.check_chunks("il_occ_sample_mean_lec_1_output_1_partition", 1)

    def test_il_occ_sample_mean_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_occ_sample_mean_lec_1_output", 20)
        self.check_chunks("il_occ_sample_mean_lec_1_output_20_partition", 20)

    def test_il_summarycalc_1_partition_chunk(self):
        self.gen_chunked_bash("il_summarycalc_1_output", 1)
        self.check_chunks("il_summarycalc_1_output_1_partition", 1)

    def test_il_summarycalc_20_partition_chunk(self):
        self.gen_chunked_bash("il_summarycalc_1_output", 20)
        self.check_chunks("il_summarycalc_1_output_20_partition", 20)

    def test_il_eltcalc_1_partition_chunk(self):
        self.gen_chunked_bash("il_eltcalc_1_output", 1)
        self.check_chunks("il_eltcalc_1_output_1_partition", 1)

    def test_il_eltcalc_20_partition_chunk(self):
        self.gen_chunked_bash("il_eltcalc_1_output", 20)
        self.check_chunks("il_eltcalc_1_output_20_partition", 20)

    def test_il_aalcalc_1_partition_chunk(self):
        self.gen_chunked_bash("il_aalcalc_1_output", 1)
        self.check_chunks("il_aalcalc_1_output_1_partition", 1)

    def test_il_aalcalc_20_partition_chunk(self):
        self.gen_chunked_bash("il_aalcalc_1_output", 20)
        self.check_chunks("il_aalcalc_1_output_20_partition", 20)

    def test_il_pltcalc_1_partition_chunk(self):
        self.gen_chunked_bash("il_pltcalc_1_output", 1)
        self.check_chunks("il_pltcalc_1_output_1_partition", 1)

    def test_il_pltcalc_20_partition_chunk(self):
        self.gen_chunked_bash("il_pltcalc_1_output", 20)
        self.check_chunks("il_pltcalc_1_output_20_partition", 20)

    def test_il_agg_fu_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_agg_fu_lec_1_output", 1)
        self.check_chunks("il_agg_fu_lec_1_output_1_partition", 1)

    def test_il_agg_fu_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_agg_fu_lec_1_output", 20)
        self.check_chunks("il_agg_fu_lec_1_output_20_partition", 20)

    def test_il_occ_fu_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_occ_fu_lec_1_output", 1)
        self.check_chunks("il_occ_fu_lec_1_output_1_partition", 1)

    def test_il_occ_fu_lec_1_output_20_partition_chunk(self):
        self.gen_chunked_bash("il_occ_fu_lec_1_output", 20)
        self.check_chunks("il_occ_fu_lec_1_output_20_partition", 20)

    def test_il_agg_ws_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_agg_ws_lec_1_output", 1)
        self.check_chunks("il_agg_ws_lec_1_output_1_partition", 1)

    def test_il_agg_ws_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_agg_ws_lec_1_output", 20)
        self.check_chunks("il_agg_ws_lec_1_output_20_partition", 20)

    def test_il_occ_ws_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_occ_ws_lec_1_output", 1)
        self.check_chunks("il_occ_ws_lec_1_output_1_partition", 1)

    def test_il_occ_ws_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_occ_ws_lec_1_output", 20)
        self.check_chunks("il_occ_ws_lec_1_output_20_partition", 20)

    def test_il_agg_ws_mean_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_agg_ws_mean_lec_1_output", 1)
        self.check_chunks("il_agg_ws_mean_lec_1_output_1_partition", 1)

    def test_il_agg_ws_mean_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_agg_ws_mean_lec_1_output", 20)
        self.check_chunks("il_agg_ws_mean_lec_1_output_20_partition", 20)

    def test_il_occ_ws_mean_lec_1_partition_chunk(self):
        self.gen_chunked_bash("il_occ_ws_mean_lec_1_output", 1)
        self.check_chunks("il_occ_ws_mean_lec_1_output_1_partition", 1)

    def test_il_occ_ws_mean_lec_20_partition_chunk(self):
        self.gen_chunked_bash("il_occ_ws_mean_lec_1_output", 20)
        self.check_chunks("il_occ_ws_mean_lec_1_output_20_partition", 20)

    def test_il_agg_sample_mean_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_agg_sample_mean_lec_1_output", 1)
        self.check_chunks("il_agg_sample_mean_lec_1_output_1_partition", 1)

    def test_il_agg_sample_mean_lec_1_output_20_partition_chunk(self):
        self.gen_chunked_bash("il_agg_sample_mean_lec_1_output", 20)
        self.check_chunks("il_agg_sample_mean_lec_1_output_20_partition", 20)

    def test_il_occ_sample_mean_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_occ_sample_mean_lec_1_output", 1)
        self.check_chunks("il_occ_sample_mean_lec_1_output_1_partition", 1)

    def test_il_occ_sample_mean_lec_1_output_20_partition_chunk(self):
        self.gen_chunked_bash("il_occ_sample_mean_lec_1_output", 20)
        self.check_chunks("il_occ_sample_mean_lec_1_output_20_partition", 20)

    def test_all_calcs_1_partition_chunk(self):
        self.gen_chunked_bash("all_calcs_1_output", 1)
        self.check_chunks("all_calcs_1_output_1_partition", 1)

    def test_all_calcs_20_partition_chunk(self):
        self.gen_chunked_bash("all_calcs_1_output", 20)
        self.check_chunks("all_calcs_1_output_20_partition", 20)

    def test_all_calcs_40_partition_chunk(self):
        self.gen_chunked_bash("all_calcs_1_output", 40)
        self.check_chunks("all_calcs_1_output_40_partition", 40)

    def test_gul_no_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_no_lec_1_output", 1)
        self.check_chunks("gul_no_lec_1_output_1_partition", 1)

    def test_gul_no_lec_1_output_2_partition_chunk(self):
        self.gen_chunked_bash("gul_no_lec_1_output", 2)
        self.check_chunks("gul_no_lec_1_output_2_partition", 2)

    def test_gul_no_lec_2_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_no_lec_2_output", 1)
        self.check_chunks("gul_no_lec_2_output_1_partition", 1)

    def test_gul_no_lec_2_output_2_partitions_chunk(self):
        self.gen_chunked_bash("gul_no_lec_2_output", 2)
        self.check_chunks("gul_no_lec_2_output_2_partition", 2)

    def test_gul_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_lec_1_output", 1)
        self.check_chunks("gul_lec_1_output_1_partition", 1)

    def test_gul_lec_1_output_2_partitions_chunk(self):
        self.gen_chunked_bash("gul_lec_1_output", 2)
        self.check_chunks("gul_lec_1_output_2_partition", 2)

    def test_gul_lec_2_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_lec_2_output", 1)
        self.check_chunks("gul_lec_2_output_1_partition", 1)

    def test_gul_lec_2_output_2_partitions_chunk(self):
        self.gen_chunked_bash("gul_lec_2_output", 2)
        self.check_chunks("gul_lec_2_output_2_partition", 2)

    def test_il_no_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_no_lec_1_output", 1)
        self.check_chunks("il_no_lec_1_output_1_partition", 1)

    def test_il_no_lec_1_output_2_partition_chunk(self):
        self.gen_chunked_bash("il_no_lec_1_output", 2)
        self.check_chunks("il_no_lec_1_output_2_partition", 2)

    def test_il_no_lec_2_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_no_lec_2_output", 1)
        self.check_chunks("il_no_lec_2_output_1_partition", 1)

    def test_il_no_lec_2_output_2_partitions_chunk(self):
        self.gen_chunked_bash("il_no_lec_2_output", 2)
        self.check_chunks("il_no_lec_2_output_2_partition", 2)

    def test_il_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_lec_1_output", 1)
        self.check_chunks("il_lec_1_output_1_partition", 1)

    def test_il_lec_1_output_2_partitions_chunk(self):
        self.gen_chunked_bash("il_lec_1_output", 2)
        self.check_chunks("il_lec_1_output_2_partition", 2)

    def test_il_lec_2_output_1_partition_chunk(self):
        self.gen_chunked_bash("il_lec_2_output", 1)
        self.check_chunks("il_lec_2_output_1_partition", 1)

    def test_il_lec_2_output_2_partitions_chunk(self):
        self.gen_chunked_bash("il_lec_2_output", 2)
        self.check_chunks("il_lec_2_output_2_partition", 2)

    def test_gul_il_no_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_il_no_lec_1_output", 1)
        self.check_chunks("gul_il_no_lec_1_output_1_partition", 1)

    def test_gul_il_no_lec_1_output_2_partition_chunk(self):
        self.gen_chunked_bash("gul_il_no_lec_1_output", 2)
        self.check_chunks("gul_il_no_lec_1_output_2_partition", 2)

    def test_gul_il_no_lec_2_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_il_no_lec_2_output", 1)
        self.check_chunks("gul_il_no_lec_2_output_1_partition", 1)

    def test_gul_il_no_lec_2_output_2_partitions_chunk(self):
        self.gen_chunked_bash("gul_il_no_lec_2_output", 2)
        self.check_chunks("gul_il_no_lec_2_output_2_partition", 2)

    def test_gul_il_lec_1_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_il_lec_1_output", 1)
        self.check_chunks("gul_il_lec_1_output_1_partition", 1)

    def test_gul_il_lec_1_output_2_partitions_chunk(self):
        self.gen_chunked_bash("gul_il_lec_1_output", 2)
        self.check_chunks("gul_il_lec_1_output_2_partition", 2)

    def test_gul_il_lec_2_output_1_partition_chunk(self):
        self.gen_chunked_bash("gul_il_lec_2_output", 1)
        self.check_chunks("gul_il_lec_2_output_1_partition", 1)

    def test_gul_il_lec_2_output_2_partitions_chunk(self):
        self.gen_chunked_bash("gul_il_lec_2_output", 2)
        self.check_chunks("gul_il_lec_2_output_2_partition", 2)

    def test_gul_il_lec_2_output_10_partitions_chunk(self):
        self.gen_chunked_bash("gul_il_lec_2_output", 10)
        self.check_chunks("gul_il_lec_2_output_10_partition", 10)

    def test_analysis_settings_1_chunk(self):
        self.gen_chunked_bash("analysis_settings_1", 1)
        self.check_chunks("analysis_settings_1_1_partition", 1)

    def test_analysis_settings_2_chunk(self):
        self.gen_chunked_bash("analysis_settings_2", 1)
        self.check_chunks("analysis_settings_2_1_partition", 1)

    def test_analysis_settings_3_0_reins_iters_chunk(self):
        self.gen_chunked_bash("analysis_settings_3", 1, 1)
        self.check_chunks("analysis_settings_3_1_reins_layer_1_partition", 1)

    def test_analysis_settings_4_0_reins_iters_chunk(self):
        self.gen_chunked_bash("analysis_settings_4", 1, 1)
        self.check_chunks("analysis_settings_4_1_reins_layer_1_partition", 1)

    # =============================================================================
    # Custom GulCalc checks
    # =============================================================================
    @staticmethod
    def _get_getmodel_cmd(**kwargs):
        error_guard = kwargs.get('stderr_guard')
        cmd = 'custom_gulcalc_command'

        if error_guard:
            cmd = '({}) 2>> log/gul_stderror.err'.format(cmd)
        return cmd

    def test_custom_gul_summarycalc_1_partition(self):
        self.genbash("custom_gul_summarycalc_1_output", 1, _get_getmodel_cmd=self._get_getmodel_cmd)
        self.check("custom_gul_summarycalc_1_output_1_partition")

    def test_custom_gul_summarycalc_1_partition_chunk(self):
        self.gen_chunked_bash("custom_gul_summarycalc_1_output", 1, _get_getmodel_cmd=self._get_getmodel_cmd)
        self.check_chunks("custom_gul_summarycalc_1_output_1_partition", 1)


class Genbash_GulItemStream(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "itm_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "itm_kparse_reference")

        cls.ri_iterations = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_ErrorGuard(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "err_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "err_kparse_reference")

        cls.ri_iterations = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = True
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_TempDir(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "tmp_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "tmp_kparse_reference")

        cls.ri_iterations = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = True
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_FullCorrItemStream(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "fc_kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "itm_fc_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "itm_fc_kparse_reference")

        cls.ri_iterations = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_FullCorrErrorGuard(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "fc_kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "err_fc_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "err_fc_kparse_reference")

        cls.ri_iterations = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = True
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_FullCorrTempDir(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "fc_kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "tmp_fc_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "tmp_fc_kparse_reference")

        cls.ri_iterations = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 3
        cls.fifo_tmp_dir = True
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_LoadBalancerFmpy(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "lb_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "lb_kparse_reference")

        # defaults
        cls.ri_iterations = 0
        cls.gul_alloc_rule = 0
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 2
        cls.num_gul_per_lb = 2
        cls.num_fm_per_lb = 2
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = False
        cls.fmpy = True

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


class Genbash_EventShuffle(Genbash):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # test dirs
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "eve_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "eve_kparse_reference")

        # defaults
        cls.ri_iterations = 0
        cls.gul_alloc_rule = 0
        cls.il_alloc_rule = 2
        cls.ri_alloc_rule = 2
        cls.num_gul_per_lb = 2
        cls.num_fm_per_lb = 2
        cls.event_shuffle = 3
        cls.fifo_tmp_dir = False
        cls.bash_trace = False
        cls.stderr_guard = False
        cls.gul_legacy_stream = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)
