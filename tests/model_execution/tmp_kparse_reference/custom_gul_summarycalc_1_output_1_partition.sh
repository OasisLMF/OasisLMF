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

rm -R -f work/*
mkdir -p work/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1



# --- Do ground up loss computes ---

summarycalctocsv < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > work/kat/gul_S1_summarycalc_P1 & pid1=$!


tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P1 > /dev/null & pid2=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &

( custom_gulcalc_command > /tmp/%FIFO_DIR%/fifo/gul_P1  ) &  pid3=$!

wait $pid1 $pid2 $pid3


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P1 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1


rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
