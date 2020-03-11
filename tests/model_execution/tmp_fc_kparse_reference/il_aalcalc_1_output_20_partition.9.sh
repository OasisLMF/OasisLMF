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

mkfifo /tmp/%FIFO_DIR%/fifo/il_P10

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10

mkfifo il_S1_summary_P10



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 work/il_S1_summaryaalcalc/P10.bin > /dev/null & pid1=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P10 < /tmp/%FIFO_DIR%/fifo/il_P10 &

eve 10 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P10 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P10  &

wait $pid1

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P10 > il_P10 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
tee < il_S1_summary_P10 work/full_correlation/il_S1_summaryaalcalc/P10.bin > /dev/null & pid1=$!
summarycalc -f  -1 il_S1_summary_P10 < il_P10 &

wait $pid1


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

