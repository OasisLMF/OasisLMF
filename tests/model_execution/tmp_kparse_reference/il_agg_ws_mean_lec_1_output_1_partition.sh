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
mkdir -p work/il_S1_summaryleccalc

mkfifo /tmp/%FIFO_DIR%/fifo/il_P1

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx



# --- Do insured loss computes ---



tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 work/il_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1.idx work/il_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!

summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/il_P1 &

( eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P1  ) & pid3=$!

wait $pid1 $pid2 $pid3


# --- Do insured loss kats ---


leccalc -r -Kil_S1_summaryleccalc -M output/il_S1_leccalc_wheatsheaf_mean_aep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
