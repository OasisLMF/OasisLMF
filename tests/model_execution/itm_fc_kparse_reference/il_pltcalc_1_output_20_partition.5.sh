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


mkfifo fifo/il_P6

mkfifo fifo/il_S1_summary_P6
mkfifo fifo/il_S1_summarypltcalc_P6
mkfifo fifo/il_S1_pltcalc_P6

mkfifo il_S1_summary_P6
mkfifo il_S1_summarypltcalc_P6
mkfifo il_S1_pltcalc_P6



# --- Do insured loss computes ---
pltcalc -s < fifo/il_S1_summarypltcalc_P6 > work/kat/il_S1_pltcalc_P6 & pid1=$!
tee < fifo/il_S1_summary_P6 fifo/il_S1_summarypltcalc_P6 > /dev/null & pid2=$!
summarycalc -f  -1 fifo/il_S1_summary_P6 < fifo/il_P6 &

eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P6 -a1 -i - | fmcalc -a2 > fifo/il_P6  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P6 > il_P6 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
pltcalc -s < il_S1_summarypltcalc_P6 > work/full_correlation/kat/il_S1_pltcalc_P6 & pid1=$!
tee < il_S1_summary_P6 il_S1_summarypltcalc_P6 > /dev/null & pid2=$!
summarycalc -f  -1 il_S1_summary_P6 < il_P6 &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P6 > output/il_S1_pltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_pltcalc_P6 > output/full_correlation/il_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

