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

    def genbash(self, name, num_partitions):
        input_filename = os.path.join(KPARSE_INPUT_FOLDER, "{}.json".format(name))
        output_filename = os.path.join(KPARSE_OUTPUT_FOLDER, "{}_{}_partition.sh".format(name, num_partitions))

        with io.open(input_filename, encoding='utf-8') as file:
            analysis_settings = json.load(file)['analysis_settings']

        genbash(
            num_partitions,
            analysis_settings,
            output_filename
        )

    def check(self, name):
        output_filename = os.path.join(KPARSE_OUTPUT_FOLDER, "{}.sh".format(name))
        reference_filename = os.path.join(KPARSE_REFERENCE_FOLDER, "{}.sh".format(name))

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

    def test_analysis_settings_1(self):
        self.genbash("analysis_settings_1", 1)
        self.check("analysis_settings_1_1_partition")

    def test_analysis_settings_2(self):
        self.genbash("analysis_settings_2", 1)
        self.check("analysis_settings_2_1_partition")
