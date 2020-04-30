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

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1

mkfifo fifo/full_correlation/il_S1_summary_P1
mkfifo fifo/full_correlation/il_S1_summarysummarycalc_P1
mkfifo fifo/full_correlation/il_S1_summarycalc_P1



# --- Do insured loss computes ---

summarycalctocsv < fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summarysummarycalc_P1 > /dev/null & pid2=$!

summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P1 -a1 -i - | fmcalc -a2 > fifo/il_P1  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

fmcalc -a2 < fifo/full_correlation/gul_P1 > fifo/full_correlation/il_P1 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---

summarycalctocsv < fifo/full_correlation/il_S1_summarysummarycalc_P1 > work/full_correlation/kat/il_S1_summarycalc_P1 & pid1=$!

tee < fifo/full_correlation/il_S1_summary_P1 fifo/full_correlation/il_S1_summarysummarycalc_P1 > /dev/null & pid2=$!

summarycalc -f  -1 fifo/full_correlation/il_S1_summary_P1 < fifo/full_correlation/il_P1 &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_summarycalc_P1 > output/full_correlation/il_S1_summarycalc.csv & kpid2=$!
wait $kpid1 $kpid2


rm -R -f work/*
rm -R -f fifo/*
