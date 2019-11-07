#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat
mkfifo fifo/gul_P1
mkfifo fifo/gul_S1_summary_P1

mkdir work/gul_S1_summaryaalcalc

# --- Do ground up loss computes ---


tee < fifo/gul_S1_summary_P1 work/gul_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!

summarycalc -i  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P1  &

wait $pid1


# --- Do ground up loss kats ---


aalcalc -Kgul_S1_summaryaalcalc > output/gul_S1_aalcalc.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
