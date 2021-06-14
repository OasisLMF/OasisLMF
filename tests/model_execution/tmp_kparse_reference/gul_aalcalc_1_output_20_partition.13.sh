#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f /tmp/%FIFO_DIR%/fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P14

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 work/gul_S1_summaryaalcalc/P14.bin > /dev/null & pid1=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P14 < /tmp/%FIFO_DIR%/fifo/gul_P14 &

eve 14 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P14  &

wait $pid1


# --- Do ground up loss kats ---

