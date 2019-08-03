#!/bin/bash

set -e
set -o pipefail

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryeltcalc_P1
mkfifo fifo/il_S1_eltcalc_P1


# --- Do insured loss computes ---

eltcalc < fifo/il_S1_summaryeltcalc_P1 > work/kat/il_S1_eltcalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryeltcalc_P1 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P1  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_eltcalc_P1 > output/il_S1_eltcalc.csv & kpid1=$!
wait $kpid1



set +e


rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summaryeltcalc_P1
rm fifo/il_S1_eltcalc_P1

rm -rf work/kat
