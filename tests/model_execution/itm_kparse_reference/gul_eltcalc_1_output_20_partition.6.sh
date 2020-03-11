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


mkfifo fifo/gul_P7

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summaryeltcalc_P7
mkfifo fifo/gul_S1_eltcalc_P7



# --- Do ground up loss computes ---
eltcalc -s < fifo/gul_S1_summaryeltcalc_P7 > work/kat/gul_S1_eltcalc_P7 & pid1=$!
tee < fifo/gul_S1_summary_P7 fifo/gul_S1_summaryeltcalc_P7 > /dev/null & pid2=$!
summarycalc -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 &

eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P7  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P7 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1

