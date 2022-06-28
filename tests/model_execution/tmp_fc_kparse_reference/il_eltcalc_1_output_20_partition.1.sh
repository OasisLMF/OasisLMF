#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

LOG_DIR=log
mkdir -p $LOG_DIR
rm -R -f $LOG_DIR/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir -p output/full_correlation/

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P2[^0-9]*' -o -name '*P2' \) -exec rm -R -f {} +
mkdir -p /tmp/%FIFO_DIR%/fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/


mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/il_P2

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2



# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 > work/kat/il_S1_eltcalc_P2 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/il_S1_eltcalc_P2 > /dev/null & pid2=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/il_P2 &

# --- Do insured loss computes ---
eltcalc -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2 > work/full_correlation/kat/il_S1_eltcalc_P2 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_eltcalc_P2 > /dev/null & pid4=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P2 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P2 &
eve 2 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P2 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P2  &

wait $pid1 $pid2 $pid3 $pid4

