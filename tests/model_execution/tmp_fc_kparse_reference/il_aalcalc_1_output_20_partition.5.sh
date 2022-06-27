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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P6[^0-9]*' -o -name '*P6' \) -exec rm -R -f {} +
mkdir -p /tmp/%FIFO_DIR%/fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/il_S1_summaryaalcalc
mkdir -p work/full_correlation/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6

mkfifo /tmp/%FIFO_DIR%/fifo/il_P6

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6.idx



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 work/il_S1_summaryaalcalc/P6.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6.idx work/il_S1_summaryaalcalc/P6.idx > /dev/null & pid2=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/il_P6 &

# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 work/full_correlation/il_S1_summaryaalcalc/P6.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6.idx work/full_correlation/il_S1_summaryaalcalc/P6.idx > /dev/null & pid4=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P6 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P6 &
eve 6 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P6 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P6  &

wait $pid1 $pid2 $pid3 $pid4

