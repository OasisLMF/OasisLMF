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

find fifo/ \( -name '*P16[^0-9]*' -o -name '*P16' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P16

mkfifo fifo/il_P16

mkfifo fifo/il_S1_summary_P16
mkfifo fifo/il_S1_summary_P16.idx

mkfifo fifo/full_correlation/il_P16

mkfifo fifo/full_correlation/il_S1_summary_P16
mkfifo fifo/full_correlation/il_S1_summary_P16.idx



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P16 work/il_S1_summaryaalcalc/P16.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P16.idx work/il_S1_summaryaalcalc/P16.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P16 < fifo/il_P16 &

# --- Do insured loss computes ---
tee < fifo/full_correlation/il_S1_summary_P16 work/full_correlation/il_S1_summaryaalcalc/P16.bin > /dev/null & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P16.idx work/full_correlation/il_S1_summaryaalcalc/P16.idx > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P16 < fifo/full_correlation/il_P16 &

( fmcalc -a2 < fifo/full_correlation/gul_fc_P16 > fifo/full_correlation/il_P16 ) & pid5=$!
( eve 16 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P16 -a1 -i - | fmcalc -a2 > fifo/il_P16  ) & pid6=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6

