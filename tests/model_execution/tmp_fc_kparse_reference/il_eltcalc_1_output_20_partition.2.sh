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


mkfifo /tmp/%FIFO_DIR%/fifo/il_P3

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P3
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P3

mkfifo il_S1_summary_P3
mkfifo il_S1_summaryeltcalc_P3
mkfifo il_S1_eltcalc_P3



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P3 > work/kat/il_S1_eltcalc_P3 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 /tmp/%FIFO_DIR%/fifo/il_S1_summaryeltcalc_P3 > /dev/null & pid2=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P3 < /tmp/%FIFO_DIR%/fifo/il_P3 &

eve 3 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P3 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P3  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P3 > il_P3 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
eltcalc -s < il_S1_summaryeltcalc_P3 > work/full_correlation/kat/il_S1_eltcalc_P3 & pid1=$!
tee < il_S1_summary_P3 il_S1_summaryeltcalc_P3 > /dev/null & pid2=$!
summarycalc -f  -1 il_S1_summary_P3 < il_P3 &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P3 > output/il_S1_eltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P3 > output/full_correlation/il_S1_eltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

