#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +
mkdir output/full_correlation/

rm -R -f /tmp/%FIFO_DIR%/fifo/*
mkdir /tmp/%FIFO_DIR%/fifo/full_correlation/
rm -R -f work/*
mkdir work/kat/
mkdir work/full_correlation/
mkdir work/full_correlation/kat/

mkdir work/il_S1_summaryaalcalc
mkdir work/full_correlation/il_S1_summaryaalcalc

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12

mkfifo /tmp/%FIFO_DIR%/fifo/il_P12

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12



# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 work/il_S1_summaryaalcalc/P12.bin > /dev/null & pid1=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/il_P12 &

# --- Do insured loss computes ---
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 work/full_correlation/il_S1_summaryaalcalc/P12.bin > /dev/null & pid2=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P12 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 &

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P12 &
eve 12 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P12 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P12  &

wait $pid1 $pid2


# --- Do insured loss kats ---


# --- Do insured loss kats for fully correlated output ---

