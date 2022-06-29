#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do ground up loss kats ---

kat -u work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f fifo/*
