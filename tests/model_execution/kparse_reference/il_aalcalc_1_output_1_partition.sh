#!/bin/bash

set -e
set -o pipefail

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
rm -R -f fifo/*
rm -R -f work/*

mkdir work/kat
mkfifo fifo/il_P1

mkfifo fifo/il_S1_summary_P1

mkdir work/il_S1_summaryaalcalc

# --- Do insured loss computes ---


tee < fifo/il_S1_summary_P1 work/il_S1_summaryaalcalc/P1.bin > /dev/null & pid1=$!
summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a 2 > fifo/il_P1  &

wait $pid1


# --- Do insured loss kats ---


aalcalc -Kil_S1_summaryaalcalc > output/il_S1_aalcalc.csv & lpid1=$!
wait $lpid1


set +e


rm fifo/il_P1

rm fifo/il_S1_summary_P1

rm -rf work/kat
rm -rf work/il_S1_summaryaalcalc/*
rmdir work/il_S1_summaryaalcalc
