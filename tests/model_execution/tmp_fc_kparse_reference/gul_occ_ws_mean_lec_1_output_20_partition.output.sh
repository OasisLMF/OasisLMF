#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---


leccalc -r -Kgul_S1_summaryleccalc -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv & lpid1=$!
leccalc -r -Kfull_correlation/gul_S1_summaryleccalc -m output/full_correlation/gul_S1_leccalc_wheatsheaf_mean_oep.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
