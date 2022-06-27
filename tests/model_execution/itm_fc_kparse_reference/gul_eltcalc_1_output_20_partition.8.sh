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
mkdir -p output/full_correlation/

find fifo/ \( -name '*P9[^0-9]*' -o -name '*P9' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/


mkfifo fifo/gul_P9

mkfifo fifo/gul_S1_summary_P9
mkfifo fifo/gul_S1_eltcalc_P9

mkfifo fifo/full_correlation/gul_P9

mkfifo fifo/full_correlation/gul_S1_summary_P9
mkfifo fifo/full_correlation/gul_S1_eltcalc_P9



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P9 > work/kat/gul_S1_eltcalc_P9 & pid1=$!
tee < fifo/gul_S1_summary_P9 fifo/gul_S1_eltcalc_P9 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P9 < fifo/gul_P9 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P9 > work/full_correlation/kat/gul_S1_eltcalc_P9 & pid3=$!
tee < fifo/full_correlation/gul_S1_summary_P9 fifo/full_correlation/gul_S1_eltcalc_P9 > /dev/null & pid4=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P9 < fifo/full_correlation/gul_P9 &

eve 9 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P9 -a1 -i - > fifo/gul_P9  &

wait $pid1 $pid2 $pid3 $pid4

