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

mkdir work/gul_S1_summaryleccalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P20

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20.idx



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 work/gul_S1_summaryleccalc/P20.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20.idx work/gul_S1_summaryleccalc/P20.idx > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/gul_P20 &

eve 20 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P20  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

