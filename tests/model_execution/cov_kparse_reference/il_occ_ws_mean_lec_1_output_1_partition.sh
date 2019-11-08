#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat
mkfifo fifo/il_P1
mkfifo fifo/il_S1_summary_P1

mkdir work/il_S1_summaryleccalc

# --- Do insured loss computes ---


tee < fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!

summarycalc -f  -1 fifo/il_S1_summary_P1 < fifo/il_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a2 > fifo/il_P1  &

wait $pid1


# --- Do insured loss kats ---


leccalc -r -Kil_S1_summaryleccalc -m output/il_S1_leccalc_wheatsheaf_mean_oep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f fifo/*
