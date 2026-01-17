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

rm -R -f fifo/*
rm -R -f work/*
mkdir -p work/kat/


mkfifo fifo/gul_P1

mkfifo fifo/gul_S1_summary_P1



# --- Do ground up loss computes ---



tee < fifo/gul_S1_summary_P1 > /dev/null & pid1=$!

summarypy -m -t gul  -1 fifo/gul_S1_summary_P1 < fifo/gul_P1 &

( custom_gulcalc_command > fifo/gul_P1  ) &  pid2=$!

wait $pid1 $pid2


# --- Do ground up loss kats ---


rm -R -f work/*
rm -R -f fifo/*
