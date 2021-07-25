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

mkdir work/gul_S1_summaryaalcalc

mkfifo fifo/gul_P15

mkfifo fifo/gul_S1_summary_P15



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P15 work/gul_S1_summaryaalcalc/P15.bin > /dev/null & pid1=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P15 < fifo/gul_P15 &

eve 15 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P15  &

wait $pid1


# --- Do ground up loss kats ---

