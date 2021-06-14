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

mkfifo fifo/il_P15

mkfifo fifo/il_S1_summary_P15
mkfifo fifo/il_S1_eltcalc_P15



# --- Do insured loss computes ---
eltcalc -s < fifo/il_S1_eltcalc_P15 > work/kat/il_S1_eltcalc_P15 & pid1=$!
tee < fifo/il_S1_summary_P15 fifo/il_S1_eltcalc_P15 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P15 < fifo/il_P15 &

eve 15 20 | getmodel | gulcalc -S100 -L100 -r -a0 -i - | fmpy -a2 > fifo/il_P15  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat -s work/kat/il_S1_eltcalc_P15 > output/il_S1_eltcalc.csv & kpid1=$!
wait $kpid1

