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

find /tmp/%FIFO_DIR%/fifo/ \( -name '*P13[^0-9]*' -o -name '*P13' \) -exec rm -R -f {} +
mkdir -p /tmp/%FIFO_DIR%/fifo/full_correlation/
rm -R -f work/*
mkdir -p work/kat/
mkdir -p work/full_correlation/
mkdir -p work/full_correlation/kat/

mkdir -p work/gul_S1_summaryaalcalc
mkdir -p work/full_correlation/gul_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P13

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13.idx

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13.idx



# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 work/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13.idx work/gul_S1_summaryaalcalc/P13.idx > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/gul_P13 &

# --- Do ground up loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13 work/full_correlation/gul_S1_summaryaalcalc/P13.bin > /dev/null & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13.idx work/full_correlation/gul_S1_summaryaalcalc/P13.idx > /dev/null & pid4=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/gul_S1_summary_P13 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13 &

eve 13 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_P13 -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P13  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do ground up loss kats ---


# --- Do ground up loss kats for fully correlated output ---

