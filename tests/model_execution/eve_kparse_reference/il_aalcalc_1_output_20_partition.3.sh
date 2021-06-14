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

mkdir work/il_S1_summaryaalcalc

mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P4



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P4 work/il_S1_summaryaalcalc/P4.bin > /dev/null & pid1=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &

eve -R 4 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmcalc -a2 > fifo/il_P4  &

wait $pid1


# --- Do insured loss kats ---

