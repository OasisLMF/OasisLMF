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

find fifo/ \( -name '*P7[^0-9]*' -o -name '*P7' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

mkdir -p work/il_S1_summaryleccalc

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_summary_P7.idx



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &

eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P7  &

wait $pid1 $pid2

