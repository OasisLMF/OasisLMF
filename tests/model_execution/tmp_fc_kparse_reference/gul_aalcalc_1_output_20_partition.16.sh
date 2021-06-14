#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f /tmp/%FIFO_DIR%/fifo/*
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/gul_S1_summaryaalcalc
mkdir work/full_correlation/gul_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P17

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 work/gul_S1_summaryaalcalc/P17.bin > /dev/null & pid1=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/gul_P17 &

# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17 work/full_correlation/gul_S1_summaryaalcalc/P17.bin > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17 &

eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P17 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P17  &

wait $pid1 $pid2


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

