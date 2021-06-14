#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/il_S1_summaryleccalc

mkfifo fifo/il_P11

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_summary_P11.idx



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P11 work/il_S1_summaryleccalc/P11.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P11.idx work/il_S1_summaryleccalc/P11.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 &

eve -R 11 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmcalc -a2 > fifo/il_P11  &

wait $pid1 $pid2


# --- Do insured loss kats ---

