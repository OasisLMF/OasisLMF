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

mkfifo /tmp/%FIFO_DIR%/fifo/il_P20

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20

mkfifo il_S1_summary_P20



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 work/il_S1_summaryaalcalc/P20.bin > /dev/null & pid1=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P20 < /tmp/%FIFO_DIR%/fifo/il_P20 &

eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P20 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P20  &

wait $pid1

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P20 > il_P20 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
tee < il_S1_summary_P20 work/full_correlation/il_S1_summaryaalcalc/P20.bin > /dev/null & pid1=$!
summarycalc -f  -1 il_S1_summary_P20 < il_P20 &

wait $pid1


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

