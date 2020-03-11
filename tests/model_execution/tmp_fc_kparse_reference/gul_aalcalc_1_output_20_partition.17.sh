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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18

mkfifo gul_S1_summary_P18



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 work/gul_S1_summaryaalcalc/P18.bin > /dev/null & pid1=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/gul_P18 &

eve 18 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P18 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P18  &

wait $pid1

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---
tee < gul_S1_summary_P18 work/full_correlation/gul_S1_summaryaalcalc/P18.bin > /dev/null & pid1=$!
summarycalc -i  -1 gul_S1_summary_P18 < gul_P18 &

wait $pid1


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

