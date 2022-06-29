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

find fifo/ \( -name '*P16[^0-9]*' -o -name '*P16' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/il_P16

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_eltcalc_P16



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P16 > work/kat/il_S1_eltcalc_P16 & pid1=$!
tee < fifo/il_S1_summary_P16 fifo/il_S1_eltcalc_P16 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 &

eve 16 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a2 > fifo/il_P16  &

wait $pid1 $pid2

