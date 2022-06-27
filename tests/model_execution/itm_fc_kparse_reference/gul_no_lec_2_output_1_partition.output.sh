#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P1 > output/gul_S1_eltcalc.csv & kpid1=$!
kat work/kat/gul_S1_pltcalc_P1 > output/gul_S1_pltcalc.csv & kpid2=$!
kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid3=$!
kat work/kat/gul_S2_eltcalc_P1 > output/gul_S2_eltcalc.csv & kpid4=$!
kat work/kat/gul_S2_pltcalc_P1 > output/gul_S2_pltcalc.csv & kpid5=$!
kat work/kat/gul_S2_summarycalc_P1 > output/gul_S2_summarycalc.csv & kpid6=$!

# --- Do ground up loss kats for fully correlated output ---

