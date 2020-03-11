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


mkfifo /tmp/%FIFO_DIR%/fifo/il_P15

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P15

mkfifo il_S1_summary_P15
mkfifo il_S1_summarypltcalc_P15
mkfifo il_S1_pltcalc_P15



# --- Do insured loss computes ---
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15 > work/kat/il_S1_pltcalc_P15 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 /tmp/%FIFO_DIR%/fifo/il_S1_summarypltcalc_P15 > /dev/null & pid2=$!
summarycalc -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P15 < /tmp/%FIFO_DIR%/fifo/il_P15 &

eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j gul_P15 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P15  &

wait $pid1 $pid2

# --- Do computes for fully correlated output ---

fmcalc-a2 < gul_P15 > il_P15 & fcpid1=$!

wait $fcpid1


# --- Do insured loss computes ---
pltcalc -s < il_S1_summarypltcalc_P15 > work/full_correlation/kat/il_S1_pltcalc_P15 & pid1=$!
tee < il_S1_summary_P15 il_S1_summarypltcalc_P15 > /dev/null & pid2=$!
summarycalc -f  -1 il_S1_summary_P15 < il_P15 &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P15 > output/il_S1_pltcalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_pltcalc_P15 > output/full_correlation/il_S1_pltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

