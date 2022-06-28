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

find fifo/ \( -name '*P4[^0-9]*' -o -name '*P4' \) -exec rm -R -f {} +
rm -R -f work/*
mkdir -p work/kat/

#fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P4

mkfifo fifo/il_S1_summary_P4
mkfifo fifo/il_S1_summarycalc_P4



# --- Do insured loss computes ---
summarycalctocsv -s < fifo/il_S1_summarycalc_P4 > work/kat/il_S1_summarycalc_P4 & pid1=$!
tee < fifo/il_S1_summary_P4 fifo/il_S1_summarycalc_P4 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P4 < fifo/il_P4 &

eve 4 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmpy -a2 > fifo/il_P4  &

wait $pid1 $pid2

