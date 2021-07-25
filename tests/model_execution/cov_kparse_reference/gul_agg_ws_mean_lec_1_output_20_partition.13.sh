#!/usr/bin/env -S bash -euET -o pipefail -O inherit_errexit
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc

mkfifo fifo/gul_P14

mkfifo fifo/gul_S1_summary_P14
mkfifo fifo/gul_S1_summary_P14.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P14 work/gul_S1_summaryleccalc/P14.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P14.idx work/gul_S1_summaryleccalc/P14.idx > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P14 < fifo/gul_P14 &

eve 14 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P14  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

