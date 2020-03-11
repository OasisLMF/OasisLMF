#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/

mkdir work/il_S1_summaryleccalc

mkfifo fifo/il_P19

mkfifo fifo/il_S1_summary_P19



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P19 work/il_S1_summaryleccalc/P19.bin > /dev/null & pid1=$!
summarycalc -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 &

eve 19 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P19  &

wait $pid1


# --- Do insured loss kats ---

