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


mkfifo fifo/il_P11

mkfifo fifo/il_S1_summary_P11
mkfifo fifo/il_S1_pltcalc_P11



# --- Do insured loss computes ---
pltcalc -s < fifo/il_S1_pltcalc_P11 > work/kat/il_S1_pltcalc_P11 & pid1=$!
tee < fifo/il_S1_summary_P11 fifo/il_S1_pltcalc_P11 > /dev/null & pid2=$!
summarycalc -m -f  -1 fifo/il_S1_summary_P11 < fifo/il_P11 &

eve 11 20 | getmodel | gulcalc -S100 -L100 -r -i - | fmcalc -a2 > fifo/il_P11  &

wait $pid1 $pid2


# --- Do insured loss kats ---

kat work/kat/il_S1_pltcalc_P11 > output/il_S1_pltcalc.csv & kpid1=$!
wait $kpid1

