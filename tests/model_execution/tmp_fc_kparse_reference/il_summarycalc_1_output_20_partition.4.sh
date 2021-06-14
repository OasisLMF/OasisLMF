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


mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/il_P5

mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5

mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5
mkfifo /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5



# --- Do insured loss computes ---
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5 > work/kat/il_S1_summarycalc_P5 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/il_S1_summarycalc_P5 > /dev/null & pid2=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/il_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/il_P5 &

# --- Do insured loss computes ---
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5 > work/full_correlation/kat/il_S1_summarycalc_P5 & pid3=$!
tee < /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summarycalc_P5 > /dev/null & pid4=$!
summarycalc -m -f  -1 /tmp/%FIFO_DIR%/fifo/full_correlation/il_S1_summary_P5 < /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &

fmcalc -a2 < /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 > /tmp/%FIFO_DIR%/fifo/full_correlation/il_P5 &
eve 5 20 | getmodel | gulcalc -S100 -L100 -r -j /tmp/%FIFO_DIR%/fifo/full_correlation/gul_fc_P5 -a1 -i - | fmcalc -a2 > /tmp/%FIFO_DIR%/fifo/il_P5  &

wait $pid1 $pid2 $pid3 $pid4


# --- Do insured loss kats ---

kat work/kat/il_S1_summarycalc_P5 > output/il_S1_summarycalc.csv & kpid1=$!

# --- Do insured loss kats for fully correlated output ---

kat work/full_correlation/kat/il_S1_summarycalc_P5 > output/full_correlation/il_S1_summarycalc.csv & kpid2=$!
wait $kpid1 $kpid2

