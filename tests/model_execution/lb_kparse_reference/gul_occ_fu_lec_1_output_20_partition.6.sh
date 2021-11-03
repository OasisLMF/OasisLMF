#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---
set -euET -o pipefail
shopt -s inherit_errexit 2>/dev/null || echo "WARNING: Unable to set inherit_errexit. Possibly unsupported by this shell, Subprocess failures may not be detected."

mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

mkdir work/gul_S1_summaryleccalc

mkfifo fifo/gul_P7

mkfifo fifo/gul_S1_summary_P7
mkfifo fifo/gul_S1_summary_P7.idx



# --- Do ground up loss computes ---
tee < fifo/gul_S1_summary_P7 work/gul_S1_summaryleccalc/P7.bin > /dev/null & pid1=$!
tee < fifo/gul_S1_summary_P7.idx work/gul_S1_summaryleccalc/P7.idx > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P7 < fifo/gul_P7 &

eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - > fifo/gul_P7  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

