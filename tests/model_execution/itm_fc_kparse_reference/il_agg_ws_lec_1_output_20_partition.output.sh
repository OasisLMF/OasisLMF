#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---


leccalc -r -Kil_S1_summaryleccalc -W output/il_S1_leccalc_wheatsheaf_aep.csv & lpid1=$!
leccalc -r -Kfull_correlation/il_S1_summaryleccalc -W output/full_correlation/il_S1_leccalc_wheatsheaf_aep.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f fifo/*
