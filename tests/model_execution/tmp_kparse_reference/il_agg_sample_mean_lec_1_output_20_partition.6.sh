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

mkdir work/il_S1_summaryleccalc

mkfifo /tmp/%FIFO_DIR%/fifo/il_P7

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 work/il_S1_summaryleccalc/P7.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7.idx work/il_S1_summaryleccalc/P7.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P7 < /tmp/%FIFO_DIR%/fifo/il_P7 &

eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P7  &

wait $pid1 $pid2


# --- Do insured loss kats ---

