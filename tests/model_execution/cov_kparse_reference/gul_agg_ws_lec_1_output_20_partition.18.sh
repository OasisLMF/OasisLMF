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

find fifo/ \( -name '*P19[^0-9]*' -o -name '*P19' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/gul_S1_summaryleccalc

mkfifo fifo/gul_P19

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_summary_P19.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P19 work/gul_S1_summaryleccalc/P19.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P19.idx work/gul_S1_summaryleccalc/P19.idx > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 &

eve 19 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P19  &

wait $pid1 $pid2

