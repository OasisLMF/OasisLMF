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

find fifo/ \( -name '*P2[^0-9]*' -o -name '*P2' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/il_P2

mkfifo fifo/il_S1_summary_P2
mkfifo fifo/il_S1_pltcalc_P2



# --- Do insured loss computes ---
pltcalc -H < fifo/il_S1_pltcalc_P2 > work/kat/il_S1_pltcalc_P2 & pid1=$!
tee < fifo/il_S1_summary_P2 fifo/il_S1_pltcalc_P2 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P2 < fifo/il_P2 &

eve 2 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > fifo/il_P2  &

wait $pid1 $pid2

