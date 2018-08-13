#!/bin/bash

rm -R -f output/*
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1
mkfifo fifo/il_S1_summarysummarycalc_P1
mkfifo fifo/il_S1_summarycalc_P1


# --- Do insured loss computes ---

summarycalctocsv < fifo/il_S1_summarysummarycalc_P1 > work/kat/il_S1_summarycalc_P1 & pid1=$!

tee < fifo/il_S1_summary_P1 fifo/il_S1_summarysummarycalc_P1 > /dev/null & pid2=$!
summarycalc -f -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc > fifo/il_P1  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_summarycalc_P1 > output/il_S1_summarycalc.csv & kpid1=$!
wait $kpid1



rm fifo/il_P1

rm fifo/il_S1_summary_P1
rm fifo/il_S1_summarysummarycalc_P1
rm fifo/il_S1_summarycalc_P1

rm -rf work/kat
