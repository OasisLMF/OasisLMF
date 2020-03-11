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


mkfifo fifo/gul_P14

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_summaryeltcalc_P14
mkfifo fifo/gul_S1_eltcalc_P14



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P14 > work/kat/gul_S1_eltcalc_P14 & pid1=$!
tee < fifo/gul_S1_summary_P14 fifo/gul_S1_summaryeltcalc_P14 > /dev/null & pid2=$!
summarycalc -g  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 &

eve 14 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P14  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P14 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1

