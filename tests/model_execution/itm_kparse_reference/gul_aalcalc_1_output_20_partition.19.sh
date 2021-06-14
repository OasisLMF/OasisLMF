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

mkdir work/gul_S1_summaryaalcalc

mkfifo fifo/gul_P20

mkfifo fifo/gul_S1_summary_P20



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P20 work/gul_S1_summaryaalcalc/P20.bin > /dev/null & pid1=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P20 < fifo/gul_P20 &

eve 20 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P20  &

wait $pid1


# --- Do ground up loss kats ---

