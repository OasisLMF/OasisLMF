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


mkfifo fifo/gul_P5

mkfifo fifo/gul_S1_summary_P5
mkfifo fifo/gul_S1_summarypltcalc_P5
mkfifo fifo/gul_S1_pltcalc_P5



# --- Do ground up loss computes ---
pltcalc -s < fifo/gul_S1_summarypltcalc_P5 > work/kat/gul_S1_pltcalc_P5 & pid1=$!
tee < fifo/gul_S1_summary_P5 fifo/gul_S1_summarypltcalc_P5 > /dev/null & pid2=$!
summarycalc -g  -1 fifo/gul_S1_summary_P5 < fifo/gul_P5 &

eve 5 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P5  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_pltcalc_P5 > output/gul_S1_pltcalc.csv & kpid1=$!
wait $kpid1

