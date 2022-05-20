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

find fifo/ \( -name '*P1[^0-9]*' -o -name '*P1' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summarycalc_P1



# --- Do insured loss computes ---

summarycalctocsv < fifo/il_S1_summarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid1=$!


tee < fifo/il_S1_summary_P1 fifo/il_S1_summarycalc_P1 > /dev/null & pid2=$!

summarycalc -m -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve -R 1 1 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmcalc -a2 > fifo/il_P1  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat -u work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid1=$!
wait $kpid1

