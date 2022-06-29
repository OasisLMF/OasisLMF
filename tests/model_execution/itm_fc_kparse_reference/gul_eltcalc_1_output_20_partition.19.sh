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

find fifo/ \( -name '*P20[^0-9]*' -o -name '*P20' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/


mkfifo fifo/gul_P20

mkfifo fifo/gul_S1_summary_P20
mkfifo fifo/gul_S1_eltcalc_P20

mkfifo fifo/full_correlation/gul_P20

mkfifo fifo/full_correlation/gul_S1_summary_P20
mkfifo fifo/full_correlation/gul_S1_eltcalc_P20



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P20 > work/kat/gul_S1_eltcalc_P20 & pid1=$!
tee < fifo/gul_S1_summary_P20 fifo/gul_S1_eltcalc_P20 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 &

# --- Do ground up loss computes ---
eltcalc -s < fifo/full_correlation/gul_S1_eltcalc_P20 > work/full_correlation/kat/gul_S1_eltcalc_P20 & pid3=$!
tee < fifo/full_correlation/gul_S1_summary_P20 fifo/full_correlation/gul_S1_eltcalc_P20 > /dev/null & pid4=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P20 < fifo/full_correlation/gul_P20 &

eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P20 -a1 -i - > fifo/gul_P20  &

wait $pid1 $pid2 $pid3 $pid4

