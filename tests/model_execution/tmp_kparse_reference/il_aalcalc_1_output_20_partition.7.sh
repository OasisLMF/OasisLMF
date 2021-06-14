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

mkdir work/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/il_P8

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 work/il_S1_summaryaalcalc/P8.bin > /dev/null & pid1=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P8 < /tmp/%FIFO_DIR%/fifo/il_P8 &

eve 8 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P8  &

wait $pid1


# --- Do insured loss kats ---

