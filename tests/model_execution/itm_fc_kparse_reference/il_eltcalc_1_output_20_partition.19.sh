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


mkfifo fifo/il_P20

mkfifo fifo/il_S1_summary_P20
mkfifo fifo/il_S1_summaryeltcalc_P20
mkfifo fifo/il_S1_eltcalc_P20

mkfifo il_S1_summary_P20
mkfifo il_S1_summaryeltcalc_P20
mkfifo il_S1_eltcalc_P20



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_summaryeltcalc_P20 > work/kat/il_S1_eltcalc_P20 & pid1=$!
tee < fifo/il_S1_summary_P20 fifo/il_S1_summaryeltcalc_P20 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P20 < fifo/il_P20 &

eve 20 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P20 -a1 -i - | fmcalc -a2 > fifo/il_P20  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P20 > il_P20 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
eltcalc -s < il_S1_summaryeltcalc_P20 > work/full_correlation/kat/il_S1_eltcalc_P20 & pid1=$!
tee < il_S1_summary_P20 il_S1_summaryeltcalc_P20 > /dev/null & pid2=$!
summarycalc -f  -1 il_S1_summary_P20 < il_P20 &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P20 > output/il_S1_eltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_eltcalc_P20 > output/full_correlation/il_S1_eltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

