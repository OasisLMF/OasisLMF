#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


aalcalc -Kri_S1_summaryaalcalc > output/ri_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kri_S1_summaryleccalc -F output/ri_S1_leccalc_full_uncertainty_aep.csv -f output/ri_S1_leccalc_full_uncertainty_oep.csv -S output/ri_S1_leccalc_sample_mean_aep.csv -s output/ri_S1_leccalc_sample_mean_oep.csv -W output/ri_S1_leccalc_wheatsheaf_aep.csv -M output/ri_S1_leccalc_wheatsheaf_mean_aep.csv -m output/ri_S1_leccalc_wheatsheaf_mean_oep.csv -w output/ri_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid3=$!
aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid4=$!
aalcalc -Kfull_correlation/ri_S1_summaryaalcalc > output/full_correlation/ri_S1_aalcalc.csv & lpid5=$!
leccalc -r -Kfull_correlation/ri_S1_summaryleccalc -F output/full_correlation/ri_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/ri_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/ri_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/ri_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/ri_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/ri_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/ri_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/ri_S1_leccalc_wheatsheaf_oep.csv & lpid6=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid7=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid8=$!
wait $lpid1 $lpid2 $lpid3 $lpid4 $lpid5 $lpid6 $lpid7 $lpid8

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
