#!/bin/bash

if [ ! -d "../tmp_kparse_output" ]; then
    echo 'no output in "tmp_kparse_output" to update - exit'
    exit 1
else 
    cp ../tmp_kparse_output/* .
fi

all_files=(
    all_calcs_1_output_1_partition.sh
    all_calcs_1_output_20_partition.sh
    all_calcs_1_output_40_partition.sh
    analysis_settings_1_1_partition.sh
    analysis_settings_2_1_partition.sh
    analysis_settings_3_1_reins_layer_1_partition.sh
    analysis_settings_4_1_reins_layer_1_partition.sh
    gul_aalcalc_1_output_1_partition.sh
    gul_aalcalc_1_output_20_partition.sh
    gul_agg_fu_lec_1_output_1_partition.sh
    gul_agg_fu_lec_1_output_20_partition.sh
    gul_agg_ws_lec_1_output_1_partition.sh
    gul_agg_ws_lec_1_output_20_partition.sh
    gul_agg_ws_mean_lec_1_output_1_partition.sh
    gul_agg_ws_mean_lec_1_output_20_partition.sh
    gul_eltcalc_1_output_1_partition.sh
    gul_eltcalc_1_output_20_partition.sh
    gul_il_lec_1_output_1_partition.sh
    gul_il_lec_1_output_2_partition.sh
    gul_il_lec_2_output_10_partition.sh
    gul_il_lec_2_output_1_partition.sh
    gul_il_lec_2_output_2_partition.sh
    gul_il_no_lec_1_output_1_partition.sh
    gul_il_no_lec_1_output_2_partition.sh
    gul_il_no_lec_2_output_1_partition.sh
    gul_il_no_lec_2_output_2_partition.sh
    gul_lec_1_output_1_partition.sh
    gul_lec_1_output_2_partition.sh
    gul_lec_2_output_1_partition.sh
    gul_lec_2_output_2_partition.sh
    gul_no_lec_1_output_1_partition.sh
    gul_no_lec_1_output_2_partition.sh
    gul_no_lec_2_output_1_partition.sh
    gul_no_lec_2_output_2_partition.sh
    gul_occ_fu_lec_1_output_1_partition.sh
    gul_occ_fu_lec_1_output_20_partition.sh
    gul_occ_ws_lec_1_output_1_partition.sh
    gul_occ_ws_lec_1_output_20_partition.sh
    gul_occ_ws_mean_lec_1_output_1_partition.sh
    gul_occ_ws_mean_lec_1_output_20_partition.sh
    gul_pltcalc_1_output_1_partition.sh
    gul_pltcalc_1_output_20_partition.sh
    gul_summarycalc_1_output_1_partition.sh
    gul_summarycalc_1_output_20_partition.sh
    il_aalcalc_1_output_1_partition.sh
    il_aalcalc_1_output_20_partition.sh
    il_agg_fu_lec_1_output_1_partition.sh
    il_agg_fu_lec_1_output_20_partition.sh
    il_agg_sample_mean_lec_1_output_1_partition.sh
    il_agg_sample_mean_lec_1_output_20_partition.sh
    il_agg_ws_lec_1_output_1_partition.sh
    il_agg_ws_lec_1_output_20_partition.sh
    il_agg_ws_mean_lec_1_output_1_partition.sh
    il_agg_ws_mean_lec_1_output_20_partition.sh
    il_eltcalc_1_output_1_partition.sh
    il_eltcalc_1_output_20_partition.sh
    il_lec_1_output_1_partition.sh
    il_lec_1_output_2_partition.sh
    il_lec_2_output_1_partition.sh
    il_lec_2_output_2_partition.sh
    il_no_lec_1_output_1_partition.sh
    il_no_lec_1_output_2_partition.sh
    il_no_lec_2_output_1_partition.sh
    il_no_lec_2_output_2_partition.sh
    il_occ_fu_lec_1_output_1_partition.sh
    il_occ_fu_lec_1_output_20_partition.sh
    il_occ_sample_mean_lec_1_output_1_partition.sh
    il_occ_sample_mean_lec_1_output_20_partition.sh
    il_occ_ws_lec_1_output_1_partition.sh
    il_occ_ws_lec_1_output_20_partition.sh
    il_occ_ws_mean_lec_1_output_1_partition.sh
    il_occ_ws_mean_lec_1_output_20_partition.sh
    il_pltcalc_1_output_1_partition.sh
    il_pltcalc_1_output_20_partition.sh
    il_summarycalc_1_output_1_partition.sh
    il_summarycalc_1_output_20_partition.sh
    analysis_settings_5_1_reins_layer_1_partition.sh
	gul_il_ord_ept_psept_2_output_10_partition.sh
	gul_il_ord_psept_lec_1_output_10_partition.sh
	gul_ord_ept_1_output_1_partition.sh
	gul_ord_ept_1_output_20_partition.sh
	gul_ord_ept_psept_lec_2_output_10_partition.sh
	gul_ord_psept_2_output_10_partition.sh
)

for f in "${all_files[@]}"; do 
    echo $f
    sed -i 's|/tmp/[a-zA-Z0-9]*/|/tmp/%FIFO_DIR%/|g' $f
done 
