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

find fifo/ \( -name '*P19[^0-9]*' -o -name '*P19' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/il_P19

mkfifo fifo/il_S1_summary_P19
mkfifo fifo/il_S1_pltcalc_P19



# --- Do insured loss computes ---
pltcalc -H < fifo/il_S1_pltcalc_P19 > work/kat/il_S1_pltcalc_P19 & pid1=$!
tee < fifo/il_S1_summary_P19 fifo/il_S1_pltcalc_P19 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P19 < fifo/il_P19 &

eve 19 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a2 > fifo/il_P19  &

wait $pid1 $pid2

