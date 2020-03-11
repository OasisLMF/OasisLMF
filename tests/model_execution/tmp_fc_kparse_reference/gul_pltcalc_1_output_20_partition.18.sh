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


mkfifo /tmp/%FIFO_DIR%/fifo/gul_P19

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P19
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_pltcalc_P19

mkfifo gul_S1_summary_P19
mkfifo gul_S1_summarypltcalc_P19
mkfifo gul_S1_pltcalc_P19



# --- Do ground up loss computes ---
pltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P19 > work/kat/gul_S1_pltcalc_P19 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 /tmp/%FIFO_DIR%/fifo/gul_S1_summarypltcalc_P19 > /dev/null & pid2=$!
summarycalc -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P19 < /tmp/%FIFO_DIR%/fifo/gul_P19 &

eve 19 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P19 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P19  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---



# --- Do ground up loss computes ---
pltcalc -s < gul_S1_summarypltcalc_P19 > work/full_correlation/kat/gul_S1_pltcalc_P19 & pid1=$!
tee < gul_S1_summary_P19 gul_S1_summarypltcalc_P19 > /dev/null & pid2=$!
summarycalc -i  -1 gul_S1_summary_P19 < gul_P19 &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_pltcalc_P19 > output/gul_S1_pltcalc.csv & kpid1=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_pltcalc_P19 > output/full_correlation/gul_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

