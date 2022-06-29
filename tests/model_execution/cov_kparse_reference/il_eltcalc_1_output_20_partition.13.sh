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

find fifo/ \( -name '*P14[^0-9]*' -o -name '*P14' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/il_P14

mkfifo fifo/il_S1_summary_P14
mkfifo fifo/il_S1_eltcalc_P14



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P14 > work/kat/il_S1_eltcalc_P14 & pid1=$!
tee < fifo/il_S1_summary_P14 fifo/il_S1_eltcalc_P14 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P14 < fifo/il_P14 &

eve 14 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a2 > fifo/il_P14  &

wait $pid1 $pid2

