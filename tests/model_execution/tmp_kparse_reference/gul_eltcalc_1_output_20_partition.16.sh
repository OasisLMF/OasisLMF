#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f /tmp/%FIFO_DIR%/fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo /tmp/%FIFO_DIR%/fifo/gul_P17

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17



# --- Do ground up loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17 > work/kat/gul_S1_eltcalc_P17 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 /tmp/%FIFO_DIR%/fifo/gul_S1_eltcalc_P17 > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P17 < /tmp/%FIFO_DIR%/fifo/gul_P17 &

eve 17 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P17  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat -s work/kat/gul_S1_eltcalc_P17 > output/gul_S1_eltcalc.csv & kpid1=$!
wait $kpid1

