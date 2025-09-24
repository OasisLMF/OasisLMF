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

find fifo/ \( -name '*P15[^0-9]*' -o -name '*P15' \) -exec rm -R -f {} +
mkdir -p fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc

mkfifo fifo/full_correlation/gul_fc_P15

mkfifo fifo/il_P15

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_summary_P15.idx

mkfifo fifo/full_correlation/il_P15

mkfifo fifo/full_correlation/il_S1_summary_P15
mkfifo fifo/full_correlation/il_S1_summary_P15.idx



# --- Do insured loss computes ---
tee < fifo/il_S1_summary_P15 work/il_S1_summaryaalcalc/P15.bin > /dev/null & pid1=$!
tee < fifo/il_S1_summary_P15.idx work/il_S1_summaryaalcalc/P15.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 &

# --- Do insured loss computes ---
tee < fifo/full_correlation/il_S1_summary_P15 work/full_correlation/il_S1_summaryaalcalc/P15.bin > /dev/null & pid3=$!
tee < fifo/full_correlation/il_S1_summary_P15.idx work/full_correlation/il_S1_summaryaalcalc/P15.idx > /dev/null & pid4=$!
summarycalc -m -f  -1 fifo/full_correlation/il_S1_summary_P15 < fifo/full_correlation/il_P15 &

( fmcalc -a2 < fifo/full_correlation/gul_fc_P15 > fifo/full_correlation/il_P15 ) & pid5=$!
( eve 15 20 | getmodel | gulcalc -S100 -L100 -r -j fifo/full_correlation/gul_fc_P15 -a1 -i - | fmcalc -a2 > fifo/il_P15  ) & pid6=$!

wait $pid1 $pid2 $pid3 $pid4 $pid5 $pid6

