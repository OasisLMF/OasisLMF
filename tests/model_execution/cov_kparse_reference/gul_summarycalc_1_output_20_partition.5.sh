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


mkfifo fifo/gul_P6

mkfifo fifo/gul_S1_summary_P6
mkfifo fifo/gul_S1_summarysummarycalc_P6
mkfifo fifo/gul_S1_summarycalc_P6



# --- Do ground up loss computes ---
summarycalctocsv -s < fifo/gul_S1_summarysummarycalc_P6 > work/kat/gul_S1_summarycalc_P6 & pid1=$!
tee < fifo/gul_S1_summary_P6 fifo/gul_S1_summarysummarycalc_P6 > /dev/null & pid2=$!
summarycalc -g  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 &

eve 6 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P6  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P6 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1

