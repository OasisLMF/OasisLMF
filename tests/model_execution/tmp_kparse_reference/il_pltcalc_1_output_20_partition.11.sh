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


mkfifo /tmp/%FIFO_DIR%/fifo/il_P12

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12



# --- Do insured loss computes ---
pltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12 > work/kat/il_S1_pltcalc_P12 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 /tmp/%FIFO_DIR%/fifo/il_S1_pltcalc_P12 > /dev/null & pid2=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/il_P12 &

eve 12 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P12  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P12 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1

