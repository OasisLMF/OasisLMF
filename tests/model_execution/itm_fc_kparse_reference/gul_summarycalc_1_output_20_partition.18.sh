#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f fifo/*
mkdir fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo fifo/gul_P19

mkfifo fifo/gul_S1_summary_P19
mkfifo fifo/gul_S1_summarycalc_P19

mkfifo fifo/full_correlation/gul_P19

mkfifo fifo/full_correlation/gul_S1_summary_P19
mkfifo fifo/full_correlation/gul_S1_summarycalc_P19



# --- Do ground up loss computes ---
summarycalctocsv -s < fifo/gul_S1_summarycalc_P19 > work/kat/gul_S1_summarycalc_P19 & pid1=$!
tee < fifo/gul_S1_summary_P19 fifo/gul_S1_summarycalc_P19 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P19 < fifo/gul_P19 &

# --- Do ground up loss computes ---
summarycalctocsv -s < fifo/full_correlation/gul_S1_summarycalc_P19 > work/full_correlation/kat/gul_S1_summarycalc_P19 & pid3=$!
tee < fifo/full_correlation/gul_S1_summary_P19 fifo/full_correlation/gul_S1_summarycalc_P19 > /dev/null & pid4=$!
summarycalc -m -i  -1 fifo/full_correlation/gul_S1_summary_P19 < fifo/full_correlation/gul_P19 &

eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_P19 -a1 -i - > fifo/gul_P19  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P19 > output/gul_S1_summarycalc.csv & kpid1=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_summarycalc_P19 > output/full_correlation/gul_S1_summarycalc.csv & kpid2=$!
wait $kpid1 $kpid2

