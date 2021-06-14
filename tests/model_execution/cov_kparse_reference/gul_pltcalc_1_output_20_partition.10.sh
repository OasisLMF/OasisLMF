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


mkfifo fifo/gul_P11

mkfifo fifo/gul_S1_summary_P11
mkfifo fifo/gul_S1_pltcalc_P11



# --- Do ground up loss computes ---
pltcalc -s < fifo/gul_S1_pltcalc_P11 > work/kat/gul_S1_pltcalc_P11 & pid1=$!
tee < fifo/gul_S1_summary_P11 fifo/gul_S1_pltcalc_P11 > /dev/null & pid2=$!
summarycalc -m -g  -1 fifo/gul_S1_summary_P11 < fifo/gul_P11 &

eve 11 20 | getmodel | gulcalc -S100 -L100 -r -c - > fifo/gul_P11  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_pltcalc_P11 > output/gul_S1_pltcalc.csv & kpid1=$!
wait $kpid1

