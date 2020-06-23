#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1



# --- Do insured loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!

summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  &

wait $pid1

# --- Do computes for fully correlated output ---

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P1 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---


tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 work/full_correlation/il_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!

summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P1 &

wait $pid1


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
aalcalc -Kfull_correlation/il_S1_summaryaalcalc > output/full_correlation/il_S1_aalcalc.csv & lpid2=$!
wait $lpid1 $lpid2

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
