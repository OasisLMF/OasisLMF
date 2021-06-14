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

mkdir work/gul_S1_summaryleccalc

mkfifo fifo/gul_P10

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_summary_P10.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P10 work/gul_S1_summaryleccalc/P10.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P10.idx work/gul_S1_summaryleccalc/P10.idx > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 &

eve 10 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_P10  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

