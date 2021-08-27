#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f work/*
mkdir work/kat/

rm -R -f /tmp/%FIFO_DIR%/
mkdir -p /tmp/%FIFO_DIR%/fifo/
mkdir work/gul_S1_summaryleccalc

mkfifo /tmp/%FIFO_DIR%/fifo/gul_P1

mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1
mkfifo /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx



# --- Do ground up loss computes ---



tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 work/gul_S1_summaryleccalc/P1.bin > /dev/null & pid1=$!
tee < /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1.idx work/gul_S1_summaryleccalc/P1.idx > /dev/null & pid2=$!

summarycalc -m -i  -1 /tmp/%FIFO_DIR%/fifo/gul_S1_summary_P1 < /tmp/%FIFO_DIR%/fifo/gul_P1 &

eve 1 1 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > /tmp/%FIFO_DIR%/fifo/gul_P1  &

wait $pid1 $pid2


# --- Do ground up loss kats ---


leccalc -r -Kgul_S1_summaryleccalc -W output/gul_S1_leccalc_wheatsheaf_aep.csv & lpid1=$!
wait $lpid1

rm -R -f work/*
rm -R -f /tmp/%FIFO_DIR%/
