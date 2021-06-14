#!/bin/bash
SCRIPT=$(readlink -f "$0") && cd $(dirname "$SCRIPT")

# --- Script Init ---

set -e
set -o pipefail
mkdir -p log
rm -R -f log/*

# --- Setup run dirs ---

find output -type f -not -name '*summary-info*' -not -name '*.json' -exec rm -R -f {} +

rm -R -f fifo/*
rm -R -f work/*
mkdir work/kat/

fmpy -a2 --create-financial-structure-files

mkfifo fifo/il_P7

mkfifo fifo/il_S1_summary_P7
mkfifo fifo/il_S1_eltcalc_P7



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P7 > work/kat/il_S1_eltcalc_P7 & pid1=$!
tee < fifo/il_S1_summary_P7 fifo/il_S1_eltcalc_P7 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P7 < fifo/il_P7 &

eve 7 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmpy -a2 > fifo/il_P7  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P7 > output/il_S1_eltcalc.csv & kpid1=$!
wait $kpid1

