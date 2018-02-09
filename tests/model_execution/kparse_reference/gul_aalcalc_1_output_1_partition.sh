#!/bin/bash

rm -R -f output/*
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1
mkfifo fifo/gul_S1_summaryaalcalc_P1

mkdir work/gul_S1_aalcalc


# --- Do insured loss computes ---


# --- Do ground up loss  computes ---

aalcalc < fifo/gul_S1_summaryaalcalc_P1 > work/gul_S1_aalcalc/P1.bin & pid1=$!

tee < fifo/gul_S1_summary_P1 fifo/gul_S1_summaryaalcalc_P1 > /dev/null & pid2=$!
summarycalc -g -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P1  &

wait $pid1 $pid2


# --- Do insured loss kats ---


# --- Do ground up loss kats ---


aalsummary -Kgul_S1_aalcalc > output/gul_S1_aalcalc.csv & apid1=$!
wait $apid1

rm fifo/gul_P1

rm fifo/gul_S1_summary_P1
rm fifo/gul_S1_summaryaalcalc_P1

rm -rf work/kat
rm work/gul_S1_aalcalc/*
rmdir work/gul_S1_aalcalc

