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

mkdir work/gul_S1_summaryleccalc
mkdir work/full_correlation/gul_S1_summaryleccalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P11

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11

mkfifo gul_S1_summary_P11



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 work/gul_S1_summaryleccalc/P11.bin > /dev/null & pid1=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P11 < /tmp/%FIFO_DIR%/fifo/gul_P11 &

eve 11 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P11 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P11  &

wait $pid1

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---
tee < gul_S1_summary_P11 work/full_correlation/gul_S1_summaryleccalc/P11.bin > /dev/null & pid1=$!
summarycalc -i  -1 gul_S1_summary_P11 < gul_P11 &

wait $pid1


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

