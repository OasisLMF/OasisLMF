#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f /tmp/%FIFO_DIR%/fifo/*
rm -R -f work/*
mkdir work/kat/


mkfifo /tmp/%FIFO_DIR%/fifo/gul_P9

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9



# --- Do ground up loss computes ---
summarycalctocsv -s < /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 > work/kat/gul_S1_summarycalc_P9 & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 /tmp/%FIFO_DIR%/fifo/gul_S1_summarycalc_P9 > /dev/null & pid2=$!
summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P9 < /tmp/%FIFO_DIR%/fifo/gul_P9 &

eve 9 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P9  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_summarycalc_P9 > output/gul_S1_summarycalc.csv & kpid1=$!
wait $kpid1

