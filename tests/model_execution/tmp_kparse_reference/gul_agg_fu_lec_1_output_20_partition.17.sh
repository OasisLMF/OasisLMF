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

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P18

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18.idx



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 work/gul_S1_summaryleccalc/P18.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18.idx work/gul_S1_summaryleccalc/P18.idx > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P18 < /tmp/%FIFO_DIR%/fifo/gul_P18 &

eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P18  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

