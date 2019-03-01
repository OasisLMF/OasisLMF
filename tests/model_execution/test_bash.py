# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import hashlib
import json
import os
import io
import shutil
from unittest import TestCase

from oasislmf.model_execution.bash import genbash
from oasislmf.utils import diff

TEST_DIRECTORY = os.path.dirname(__file__)

KPARSE_INPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_input")
KPARSE_OUTPUT_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_output")
KPARSE_REFERENCE_FOLDER = os.path.join(TEST_DIRECTORY, "kparse_reference")


class Genbash(TestCase):
    @classmethod
    def setUpClass(cls):
        if os.path.exists(KPARSE_OUTPUT_FOLDER):
            shutil.rmtree(KPARSE_OUTPUT_FOLDER)
        os.makedirs(KPARSE_OUTPUT_FOLDER)

    def md5(self, fname):
        hash_md5 = hashlib.md5()
        with io.open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def genbash(self, name, num_partitions, num_reinsurance_iterations=0, fifo_tmp_dir=False, mem_limit=False):
        input_filename = os.path.join(KPARSE_INPUT_FOLDER, "{}.json".format(name))
        if num_reinsurance_iterations <= 0:
            output_filename = os.path.join(KPARSE_OUTPUT_FOLDER, "{}_{}_partition.sh".format(name, num_partitions))
        else:
            output_filename = os.path.join(
                KPARSE_OUTPUT_FOLDER, 
                "{}_{}_reins_layer_{}_partition.sh".format(name, num_reinsurance_iterations, num_partitions))
            
        with io.open(input_filename, encoding='utf-8') as file:
            analysis_settings = json.load(file)['analysis_settings']

        genbash(
            num_partitions,
            analysis_settings,
            filename=output_filename,
            num_reinsurance_iterations=num_reinsurance_iterations,
            fifo_tmp_dir=fifo_tmp_dir,
            mem_limit=mem_limit
        )

    def check(self, name):
        output_filename = os.path.join(KPARSE_OUTPUT_FOLDER, "{}.sh".format(name))
        reference_filename = os.path.join(KPARSE_REFERENCE_FOLDER, "{}.sh".format(name))

        d = diff.unified_diff(reference_filename, output_filename, as_string=True)
        if d:
            self.fail(d)

    def update_fifo_tmpfile(self, name):
        ## Read random fifo dir name from generated file and replace in reference 
        output_filename = os.path.join(KPARSE_OUTPUT_FOLDER, "{}.sh".format(name))
        ref_template = os.path.join(KPARSE_REFERENCE_FOLDER, "{}.template".format(name))
        ref_script = os.path.join(KPARSE_REFERENCE_FOLDER, "{}.sh".format(name))
        with io.open(output_filename, 'r') as f:
           for line in f:  
               if '/tmp/' in line:
                   tmp_fifo_dir = line.split('/')[-2]
                   print(tmp_fifo_dir)
                   break
        
        # Replace placeholder '%FIFO_DIR%' with '<RandomDirName>'
        with io.open(ref_template, 'r') as f:
          ktools_script = f.read()
        ktools_script = ktools_script.replace('%FIFO_DIR%', tmp_fifo_dir)
        with io.open(ref_script, 'w') as f:
          f.write(ktools_script)

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

# -------------------------------------------------------------- #

    def test_gul_il_lec_2_output_10_partitions_tmpfifo(self):
        self.genbash("gul_il_lec_2_tmpfifo_output", 10, 0, True)
        self.update_fifo_tmpfile("gul_il_lec_2_tmpfifo_output_10_partition")
        self.check("gul_il_lec_2_tmpfifo_output_10_partition")

    def test_gul_agg_ws_mean_lec_20_partition_tmpfifo_memlim(self):
        self.genbash("gul_agg_ws_mean_lec_1_tmpfifo_memlim_output", 20, 0, True, True)
        self.update_fifo_tmpfile("gul_agg_ws_mean_lec_1_tmpfifo_memlim_output_20_partition")
        self.check("gul_agg_ws_mean_lec_1_tmpfifo_memlim_output_20_partition")

    def test_analysis_settings_3_0_reins_iters_tmpfifo(self):
        self.genbash("analysis_settings_tmpfifo_3", 1, 1, True)
        self.update_fifo_tmpfile("analysis_settings_tmpfifo_3_1_reins_layer_1_partition")
        self.check("analysis_settings_tmpfifo_3_1_reins_layer_1_partition")

    def test_analysis_settings_4_0_reins_iters_tmpfifo_memlim(self):
        self.genbash("analysis_settings_tmpfifo_memlim_4", 1, 1, True, True)
        self.update_fifo_tmpfile("analysis_settings_tmpfifo_memlim_4_1_reins_layer_1_partition")
        self.check("analysis_settings_tmpfifo_memlim_4_1_reins_layer_1_partition")
