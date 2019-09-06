
import hashlib
import io
import json
import os
import shutil
from tempfile import NamedTemporaryFile
from unittest import TestCase

from oasislmf.model_execution.bash import genbash
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
        cls.ri_iterations  = 0
        cls.gul_alloc_rule = 0
        cls.il_alloc_rule  = 2
        cls.fifo_tmp_dir   = False
        cls.bash_trace     = False
        cls.mem_limit      = False

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
                mem_limit=None,
                gul_alloc_rule=None,
                il_alloc_rule=None,
                bash_trace=None):

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
            mem_limit=(mem_limit or self.mem_limit),
            gul_alloc_rule=(gul_alloc_rule or self.gul_alloc_rule),
            il_alloc_rule=(il_alloc_rule or self.il_alloc_rule),
            bash_trace=(bash_trace or self.bash_trace),
        )

    def check(self, name, reference_filename=None):
        output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, "{}.sh".format(name))
        if not reference_filename:
            reference_filename = os.path.join(self.KPARSE_REFERENCE_FOLDER, "{}.sh".format(name))

        d = diff.unified_diff(reference_filename, output_filename, as_string=True)
        if d:
            self.fail(d)

    def update_fifo_tmpfile(self, name):
        self.temp_reference_file = NamedTemporaryFile("w+", delete=False)
        # Read random fifo dir name from generated file and replace in reference
        output_filename = os.path.join(self.KPARSE_OUTPUT_FOLDER, "{}.sh".format(name))
        ref_template = os.path.join(self.KPARSE_REFERENCE_FOLDER, "{}.template".format(name))
        with io.open(output_filename, 'r') as f:
            for line in f:
                if '/tmp/' in line:
                    tmp_fifo_dir = line.split('/')[-2]
                    break

        # Replace placeholder '%FIFO_DIR%' with '<RandomDirName>'
        with io.open(ref_template, 'r') as f:
            ktools_script = f.read()
        ktools_script = ktools_script.replace('%FIFO_DIR%', tmp_fifo_dir)
        self.temp_reference_file.write(ktools_script)
        self.temp_reference_file.close()

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

    def test_gul_il_lec_2_output_10_partitions_tmpfifo(self):
        self.genbash("gul_il_lec_2_tmpfifo_output", 10, 0, True)
        self.update_fifo_tmpfile("gul_il_lec_2_tmpfifo_output_10_partition")
        self.check("gul_il_lec_2_tmpfifo_output_10_partition",
                   self.temp_reference_file.name)

    def test_gul_agg_ws_mean_lec_20_partition_tmpfifo_memlim(self):
        self.genbash("gul_agg_ws_mean_lec_1_tmpfifo_memlim_output", 20, 0, True, True)
        self.update_fifo_tmpfile("gul_agg_ws_mean_lec_1_tmpfifo_memlim_output_20_partition")
        self.check("gul_agg_ws_mean_lec_1_tmpfifo_memlim_output_20_partition",
                   self.temp_reference_file.name)

    def test_analysis_settings_3_0_reins_iters_tmpfifo(self):
        self.genbash("analysis_settings_tmpfifo_3", 1, 1, True)
        self.update_fifo_tmpfile("analysis_settings_tmpfifo_3_1_reins_layer_1_partition")
        self.check("analysis_settings_tmpfifo_3_1_reins_layer_1_partition",
                   self.temp_reference_file.name)

    def test_analysis_settings_4_0_reins_iters_tmpfifo_memlim(self):
        self.genbash("analysis_settings_tmpfifo_memlim_4", 1, 1, True, True)
        self.update_fifo_tmpfile("analysis_settings_tmpfifo_memlim_4_1_reins_layer_1_partition")
        self.check("analysis_settings_tmpfifo_memlim_4_1_reins_layer_1_partition",
                   self.temp_reference_file.name)


class Genbash_GulItemStream(Genbash):
    @classmethod
    def setUpClass(cls):
        # test dirs 
        cls.KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
        cls.KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "itm_kparse_output")
        cls.KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "itm_kparse_reference")

        # defaults 
        cls.ri_iterations  = 0
        cls.gul_alloc_rule = 1
        cls.il_alloc_rule  = 2
        cls.fifo_tmp_dir   = False
        cls.bash_trace     = False
        cls.mem_limit      = False

        if os.path.exists(cls.KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(cls.KPARSE_OUTPUT_FOLDER)
        os.makedirs(cls.KPARSE_OUTPUT_FOLDER)


