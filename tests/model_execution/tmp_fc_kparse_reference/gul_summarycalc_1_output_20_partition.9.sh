#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo /tmp/%FIFO_DIR%/fifo/gul_P10

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P10
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P10

mkfifo gul_S1_summary_P10
mkfifo gul_S1_summarysummarycalc_P10
mkfifo gul_S1_summarycalc_P10



# --- Do ground up loss computes ---
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P10 > work/kat/gul_S1_summarycalc_P10 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 /tmp/%FIFO_DIR%/fifo/gul_S1_summarysummarycalc_P10 > /dev/null & pid2=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/gul_P10 &

eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P10 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P10  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---
summarycalctocsv -s < gul_S1_summarysummarycalc_P10 > work/full_correlation/kat/gul_S1_summarycalc_P10 & pid1=$!
tee < gul_S1_summary_P10 gul_S1_summarysummarycalc_P10 > /dev/null & pid2=$!
summarycalc -i  -1 gul_S1_summary_P10 < gul_P10 &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P10 > output/gul_S1_summarycalc.csv & kpid1=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_summarycalc_P10 > output/full_correlation/gul_S1_summarycalc.csv & kpid2=$!
wait $kpid1 $kpid2

