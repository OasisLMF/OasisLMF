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

find fifo/ \( -name '*P18[^0-9]*' -o -name '*P18' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P18

mkfifo fifo/il_S1_summary_P18
mkfifo fifo/il_S1_summarycalc_P18



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarycalc_P18 > work/kat/il_S1_summarycalc_P18 & pid1=$!
tee < fifo/il_S1_summary_P18 fifo/il_S1_summarycalc_P18 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P18 < fifo/il_P18 &

eve 18 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmpy -a2 > fifo/il_P18  &

wait $pid1 $pid2

