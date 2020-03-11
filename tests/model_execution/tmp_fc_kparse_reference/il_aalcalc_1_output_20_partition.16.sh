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

mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/il_P17

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17

mkfifo il_S1_summary_P17



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 work/il_S1_summaryaalcalc/P17.bin > /dev/null & pid1=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/il_P17 &

eve 17 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P17 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P17  &

wait $pid1

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P17 > il_P17 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
tee < il_S1_summary_P17 work/full_correlation/il_S1_summaryaalcalc/P17.bin > /dev/null & pid1=$!
summarycalc -f  -1 il_S1_summary_P17 < il_P17 &

wait $pid1


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

