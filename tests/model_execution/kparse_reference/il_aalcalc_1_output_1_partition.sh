#!/bin/bash

rm -R -f output/*
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat

mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summaryaalcalc_P1

mkdir work/il_S1_aalcalc

# --- Do insured loss computes ---

aalcalc < fifo/il_S1_summaryaalcalc_P1 > work/il_S1_aalcalc/P1.bin & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summaryaalcalc_P1 > /dev/null & pid2=$!
summarycalc -f -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P1  &

wait $pid1 $pid2


# --- Do insured loss kats ---


aalsummary -Kil_S1_aalcalc > output/il_S1_aalcalc.csv & apid1=$!
wait $apid1


rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summaryaalcalc_P1

rm -rf work/kat
rm work/il_S1_aalcalc/*
rmdir work/il_S1_aalcalc
