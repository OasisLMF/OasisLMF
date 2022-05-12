#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f /tmp/%FIFO_DIR%/fifo/*
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/


mkfifo /tmp/%FIFO_DIR%/fifo/gul_P4

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4



# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4 > work/kat/gul_S1_eltcalc_P4 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P4 > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/gul_P4 &

# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4 > work/full_correlation/kat/gul_S1_eltcalc_P4 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_eltcalc_P4 > /dev/null & pid4=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P4 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 &

eve 4 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P4 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P4  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do ground up loss kats ---

kat work/kat/gul_S1_eltcalc_P4 > output/gul_S1_eltcalc.csv & kpid1=$!

# --- Do ground up loss kats for fully correlated output ---

kat work/full_correlation/kat/gul_S1_eltcalc_P4 > output/full_correlation/gul_S1_eltcalc.csv & kpid2=$!
wait $kpid1 $kpid2

