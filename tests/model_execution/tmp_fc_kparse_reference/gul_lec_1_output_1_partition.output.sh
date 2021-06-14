#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
leccalc -r -Kgul_S1_summaryleccalc -F output/gul_S1_leccalc_full_uncertainty_aep.csv -f output/gul_S1_leccalc_full_uncertainty_oep.csv -S output/gul_S1_leccalc_sample_mean_aep.csv -s output/gul_S1_leccalc_sample_mean_oep.csv -W output/gul_S1_leccalc_wheatsheaf_aep.csv -M output/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/gul_S1_leccalc_wheatsheaf_oep.csv & lpid2=$!
aalcalc -Kfull_correlation/gul_S1_summaryaalcalc > output/full_correlation/gul_S1_aalcalc.csv & lpid3=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -F output/full_correlation/gul_S1_leccalc_full_uncertainty_aep.csv -f output/full_correlation/gul_S1_leccalc_full_uncertainty_oep.csv -S output/full_correlation/gul_S1_leccalc_sample_mean_aep.csv -s output/full_correlation/gul_S1_leccalc_sample_mean_oep.csv -W output/full_correlation/gul_S1_leccalc_wheatsheaf_aep.csv -M output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_aep.csv -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv -w output/full_correlation/gul_S1_leccalc_wheatsheaf_oep.csv & lpid4=$!
wait $lpid1 $lpid2 $lpid3 $lpid4

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
