#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f work/*
mkdir work/kat/


mkfifo fifo/il_P5

mkfifo fifo/il_S1_summary_P5
mkfifo fifo/il_S1_summarysummarycalc_P5
mkfifo fifo/il_S1_summarycalc_P5



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarysummarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid1=$!
tee < fifo/il_S1_summary_P5 fifo/il_S1_summarysummarycalc_P5 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P5 < fifo/il_P5 &

eve 5 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P5  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_summarycalc_P5 > output/il_S1_summarycalc.csv & kpid1=$!
wait $kpid1

