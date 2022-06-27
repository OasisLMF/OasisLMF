#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do reinsurance loss kats ---

kat work/kat/ri_S1_eltcalc_P1 > output/ri_S1_eltcalc.csv & kpid1=$!
kat work/kat/ri_S1_pltcalc_P1 > output/ri_S1_pltcalc.csv & kpid2=$!
kat work/kat/ri_S1_summarycalc_P1 > output/ri_S1_summarycalc.csv & kpid3=$!

# --- Do reinsurance loss kats for fully correlated output ---

