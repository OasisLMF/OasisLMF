#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output/* ! -name '*summary-info*' -type f -exec rm -f {} +
mkdir output/full_correlation/

rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo fifo/gul_P13

mkfifo fifo/gul_S1_summary_P13
mkfifo fifo/gul_S1_summarypltcalc_P13
mkfifo fifo/gul_S1_pltcalc_P13

mkfifo gul_S1_summary_P13
mkfifo gul_S1_summarypltcalc_P13
mkfifo gul_S1_pltcalc_P13



# --- Do ground up loss computes ---
pltcalc -s < fifo/gul_S1_summarypltcalc_P13 > work/kat/gul_S1_pltcalc_P13 & pid1=$!
tee < fifo/gul_S1_summary_P13 fifo/gul_S1_summarypltcalc_P13 > /dev/null & pid2=$!
summarycalc -i  -1 fifo/gul_S1_summary_P13 < fifo/gul_P13 &

eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P13 -a1 -i - > fifo/gul_P13  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---
pltcalc -s < gul_S1_summarypltcalc_P13 > work/full_correlation/kat/gul_S1_pltcalc_P13 & pid1=$!
tee < gul_S1_summary_P13 gul_S1_summarypltcalc_P13 > /dev/null & pid2=$!
summarycalc -i  -1 gul_S1_summary_P13 < gul_P13 &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_pltcalc_P13 > output/gul_S1_pltcalc.csv & kpid1=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_pltcalc_P13 > output/full_correlation/gul_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

