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


mkfifo fifo/gul_P10

mkfifo fifo/gul_S1_summary_P10
mkfifo fifo/gul_S1_eltcalc_P10



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_eltcalc_P10 > work/kat/gul_S1_eltcalc_P10 & pid1=$!
tee < fifo/gul_S1_summary_P10 fifo/gul_S1_eltcalc_P10 > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P10 < fifo/gul_P10 &

eve 10 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P10  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P10 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1

