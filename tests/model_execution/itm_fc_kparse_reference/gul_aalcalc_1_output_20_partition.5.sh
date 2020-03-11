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

mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc

mkfifo fifo/gul_P6

mkfifo fifo/gul_S1_summary_P6

mkfifo gul_S1_summary_P6



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P6 work/gul_S1_summaryaalcalc/P6.bin > /dev/null & pid1=$!
summarycalc -i  -1 fifo/gul_S1_summary_P6 < fifo/gul_P6 &

eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P6 -a1 -i - > fifo/gul_P6  &

wait $pid1

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---
tee < gul_S1_summary_P6 work/full_correlation/gul_S1_summaryaalcalc/P6.bin > /dev/null & pid1=$!
summarycalc -i  -1 gul_S1_summary_P6 < gul_P6 &

wait $pid1


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

