#!/bin/bash

rm -R -f output/*
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1

mkdir work/gul_S1_summaryleccalc


# --- Do insured loss computes ---


# --- Do ground up loss  computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
summarycalc -g -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P1  &

wait $pid1


# --- Do insured loss kats ---


# --- Do ground up loss kats ---


leccalc -r -Kgul_S1_summaryleccalc -m output/gul_S1_leccalc_wheatsheaf_mean_oep.csv & lpid1=$!
wait $lpid1

rm fifo/gul_P1

rm fifo/gul_S1_summary_P1

rm -rf work/kat
rm work/gul_S1_summaryleccalc/*
rmdir work/gul_S1_summaryleccalc

