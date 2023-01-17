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

find fifo/ \( -name '*P11[^0-9]*' -o -name '*P11' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/


mkfifo fifo/full_correlation/gul_fc_P11

mkfifo fifo/il_P11

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_summarycalc_P11

mkfifo fifo/full_correlation/il_P11

mkfifo fifo/full_correlation/il_S1_summary_P11
mkfifo fifo/full_correlation/il_S1_summarycalc_P11



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarycalc_P11 > work/kat/il_S1_summarycalc_P11 & pid1=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_summarycalc_P11 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 &

# --- Do insured loss computes ---
summarycalctocsv -s < fifo/full_correlation/il_S1_summarycalc_P11 > work/full_correlation/kat/il_S1_summarycalc_P11 & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P11 fifo/full_correlation/il_S1_summarycalc_P11 > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P11 < fifo/full_correlation/il_P11 &

( fmcalc -a2 < fifo/full_correlation/gul_fc_P11 > fifo/full_correlation/il_P11 ) & pid5=$!
( eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P11 -a1 -i - | fmcalc -a2 > fifo/il_P11  ) & pid6=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6

