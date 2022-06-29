#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

find fifo/ \( -name '*P6[^0-9]*' -o -name '*P6' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/gul_P6

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_pltcalc_P6



# --- Do ground up loss computes ---
pltcalc -H < fifo/gul_S1_pltcalc_P6 > work/kat/gul_S1_pltcalc_P6 & pid1=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_pltcalc_P6 > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 &

eve 6 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P6  &

wait $pid1 $pid2

