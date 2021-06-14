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


mkfifo fifo/gul_P17

mkfifo fifo/gul_S1_summary_P17
mkfifo fifo/gul_S1_pltcalc_P17



# --- Do ground up loss computes ---
pltcalc -s < fifo/gul_S1_pltcalc_P17 > work/kat/gul_S1_pltcalc_P17 & pid1=$!
tee < fifo/gul_S1_summary_P17 fifo/gul_S1_pltcalc_P17 > /dev/null & pid2=$!
summarycalc -m -i  -1 fifo/gul_S1_summary_P17 < fifo/gul_P17 &

eve 17 20 | getmodel | gulcalc -S100 -L100 -r -a1 -i - > fifo/gul_P17  &

wait $pid1 $pid2


# --- Do ground up loss kats ---

kat work/kat/gul_S1_pltcalc_P17 > output/gul_S1_pltcalc.csv & kpid1=$!
wait $kpid1

