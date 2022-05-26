#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


leccalc -r -Kil_S1_summaryleccalc -w output/il_S1_leccalc_wheatsheaf_oep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
